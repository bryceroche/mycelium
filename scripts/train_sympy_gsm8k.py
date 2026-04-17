#!/usr/bin/env python3
"""
SymPy Teacher-Forced GSM8K Training (v24.7)

Training script that uses SymPy annotations for teacher forcing. Each thinking
pass receives ground-truth intermediate values computed by SymPy, encoded into
the page via SymPyResultEncoder. This provides the model with perfect step-by-step
supervision during early training.

Teacher forcing schedule:
- Epochs 1-3: 100% teacher forcing (always use teacher SymPy)
- Epochs 4-6: 50% teacher forcing (randomly use teacher or skip)
- Epochs 7+: 0% teacher forcing (model must work independently)

Key features:
1. Load annotated data from data/gsm8k_sympy_annotations.jsonl
2. Separate LRs: perceiver 1e-4, atoms 1e-3, hypernetwork 1e-3, sympy_encoder 1e-4
3. Support --checkpoint for warm starting
4. Log per-epoch: loss, answer accuracy, effective per-step accuracy
5. Save best checkpoint by answer accuracy
6. Support --max_samples for debugging

Usage:
  python scripts/train_sympy_gsm8k.py --checkpoint checkpoints/atom_lora_L4_best.pt
  python scripts/train_sympy_gsm8k.py --epochs 10 --batch_size 4 --max_passes 5
  python scripts/train_sympy_gsm8k.py --max_samples 100  # debugging

Monitor with:
  tail -f logs/sympy_gsm8k_*.log
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Ensure project paths are available
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, '/home/ubuntu/mycelium')  # AWS compat

from scripts.atom_lora import (
    AtomLoRAModel,
    AtomAdditiveLoRAManager,
    AnswerHead,
    answer_head_loss,
    warm_start_atom_from_checkpoint,
)
from scripts.train_stepping_stones import extract_answer
from src.contrastive_page_loss import per_page_contrastive_loss
from src.sympy_evaluator import SymPyEvaluator
from src.sympy_result_encoder import SymPyResultEncoder, format_sympy_context
from src.pattern_classifier import classify_pattern


# ---------------------------------------------------------------------------
# Dataset: GSM8K with SymPy annotations
# ---------------------------------------------------------------------------

class GSM8KSymPyDataset(Dataset):
    """GSM8K dataset with SymPy step annotations for teacher forcing."""

    def __init__(self, jsonl_path: str = None, max_samples: Optional[int] = None):
        """
        Load annotated GSM8K data.

        Args:
            jsonl_path: Path to gsm8k_sympy_annotations.jsonl
            max_samples: Optional limit for debugging
        """
        if jsonl_path is None:
            jsonl_path = os.path.join(PROJECT_DIR, 'data', 'gsm8k_sympy_annotations.jsonl')

        self.samples = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                sample = json.loads(line)
                self.samples.append({
                    'problem': sample['problem'],
                    'gold_answer': sample['gold_answer'],
                    'sympy_steps': sample['sympy_steps'],
                })
                if max_samples and len(self.samples) >= max_samples:
                    break

        print(f"Loaded {len(self.samples)} GSM8K problems with SymPy annotations")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """Collate batch of samples."""
    return {
        'problem': [s['problem'] for s in batch],
        'gold_answer': [s['gold_answer'] for s in batch],
        'sympy_steps': [s['sympy_steps'] for s in batch],
    }


# ---------------------------------------------------------------------------
# Answer extraction helpers
# ---------------------------------------------------------------------------

def extract_answer_from_text(text: str):
    """Extract answer: 'The answer is X' first, then last number fallback."""
    # Stop at next "Problem:" (few-shot leakage)
    if "\nProblem:" in text:
        text = text.split("\nProblem:")[0]
    # Try "The answer is X"
    m = re.search(r'[Tt]he answer is\s*\$?(-?[\d,]+\.?\d*)', text)
    if m:
        try:
            v = float(m.group(1).replace(',', ''))
            return int(v) if v == int(v) else v
        except ValueError:
            pass
    # Fallback: last number
    nums = re.findall(r'-?[\d,]+\.?\d*', text)
    if nums:
        try:
            v = float(nums[-1].replace(',', ''))
            return int(v) if v == int(v) else v
        except ValueError:
            pass
    return None


# ---------------------------------------------------------------------------
# Gradient scaling helper (from train_atom_lora.py)
# ---------------------------------------------------------------------------

def scale_gradient(tensor, scale):
    """Scale gradient during backward without affecting forward."""
    return tensor + (scale - 1.0) * tensor.detach()


# ---------------------------------------------------------------------------
# Teacher forcing forward pass
# ---------------------------------------------------------------------------

def forward_train_with_sympy(
    model: AtomLoRAModel,
    answer_head: AnswerHead,
    sympy_encoder: SymPyResultEncoder,
    problems: List[str],
    gold_answers: List[int],
    sympy_steps_batch: List[List[str]],
    teacher_forcing_prob: float = 1.0,
    num_passes: int = 5,
    max_length: int = 192,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float, List[torch.Tensor]]:
    """
    Forward pass with SymPy teacher forcing.

    Each thinking pass can receive ground-truth SymPy results encoded into the page,
    controlled by teacher_forcing_prob.

    Args:
        model: AtomLoRAModel
        answer_head: AnswerHead for digit prediction
        sympy_encoder: SymPyResultEncoder for encoding SymPy results
        problems: List of problem texts
        gold_answers: List of gold answer integers
        sympy_steps_batch: List of lists of SymPy code strings per problem
        teacher_forcing_prob: Probability of using teacher forcing this batch
        num_passes: Number of thinking passes
        max_length: Max input sequence length

    Returns:
        (answer_loss, contrastive_loss, answer_head_loss, sympy_signal_loss,
         page_cos_mean, active_atoms_mean, state_pages)
    """
    device = model.transformer.device
    batch_size = len(problems)

    # Tokenize problems
    inputs = model.tokenizer(
        problems, return_tensors='pt', padding=True,
        truncation=True, max_length=max_length,
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    embed_layer = model.transformer.model.embed_tokens
    problem_embeds = embed_layer(input_ids)

    state_pages = []
    atom_scales_history = []
    mid_states_history = []

    # Track accumulated SymPy results per problem
    sympy_results_batch = [{} for _ in range(batch_size)]

    # Decide if this batch uses teacher forcing
    use_teacher = random.random() < teacher_forcing_prob

    for pass_num in range(num_passes):
        # Execute teacher SymPy code for this pass if available
        # IMPORTANT: Execute ALL steps up to current pass as one block,
        # so variables from earlier steps are in namespace (e.g., v2 = 48 + v1)
        if use_teacher:
            for b in range(batch_size):
                steps_to_run = sympy_steps_batch[b][:pass_num + 1]
                if steps_to_run:
                    cumulative_code = '\n'.join(steps_to_run)
                    step_results = SymPyEvaluator.safe_eval(cumulative_code)
                    sympy_results_batch[b] = step_results  # Replace, not update

        # First pass: no LoRA (no pages yet)
        if pass_num == 0:
            outputs = model.transformer(
                inputs_embeds=problem_embeds, attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = list(outputs.hidden_states[1:])
            atom_scales = torch.zeros(
                batch_size, model.num_atoms, device=device,
            )
        else:
            # Generate atom scales from pages + pass number
            atom_scales = model.hypernet(state_pages, pass_num=pass_num)

            # Apply atom LoRA via monkey-patching
            manager = AtomAdditiveLoRAManager(model.transformer)
            manager.apply(model.atoms, atom_scales)
            try:
                outputs = model.transformer(
                    inputs_embeds=problem_embeds, attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden_states = list(outputs.hidden_states[1:])
            finally:
                manager.remove()

        # Compress all 16 layers -> page delta
        page_delta, _strategy, current_mid_states = model.compressor(
            hidden_states, pass_num,
            prev_mid_states=mid_states_history if mid_states_history else None,
            state_pages=state_pages,
        )
        mid_states_history.append(current_mid_states)

        # Normalize on hypersphere
        page = F.normalize(page_delta, dim=-1) * model.page_radius

        # Add Fourier structural identity (after normalization)
        page = model.fourier_page.apply(page, pass_num)

        # Apply residual page gate if we have previous pages
        if len(state_pages) > 0:
            page = model.residual_gate(page, state_pages[-1])

        # Inject SymPy signal: add encoded results to page (teacher forcing)
        if use_teacher:
            sympy_vec = sympy_encoder.forward_batch(sympy_results_batch, device=device)
            page = page + sympy_vec.to(dtype=page.dtype)

        # Page noise during training
        if model.training:
            page = page + torch.randn_like(page) * 0.05

        # Gradient scaling per cycle (capped at 4x)
        grad_scale = min(float(num_passes - pass_num), 4.0)
        page = scale_gradient(page, grad_scale)

        state_pages.append(page)
        atom_scales_history.append(atom_scales)

    # ------- Answer loss (teacher-forced generation with atom LoRA) -------
    # Build CoT answer targets from SymPy results
    answer_texts = []
    for b in range(batch_size):
        gold = gold_answers[b]
        # Format accumulated results as CoT
        context = format_sympy_context(sympy_results_batch[b])
        answer_text = f" {context}The answer is {gold}."
        answer_texts.append(answer_text)

    answer_inputs = model.tokenizer(
        answer_texts, return_tensors='pt', padding=True,
        truncation=True, max_length=256, add_special_tokens=False,
    )
    answer_ids = answer_inputs['input_ids'].to(device)

    final_atom_scales = model.hypernet(state_pages, pass_num=num_passes)
    manager = AtomAdditiveLoRAManager(model.transformer)
    manager.apply(model.atoms, final_atom_scales)
    try:
        answer_embeds = embed_layer(answer_ids)
        full_embeds = torch.cat([problem_embeds, answer_embeds], dim=1)
        outputs = model.transformer(inputs_embeds=full_embeds, use_cache=False)
    finally:
        manager.remove()

    prompt_len = input_ids.size(1)
    logits = outputs.logits[:, prompt_len - 1:-1, :]
    answer_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        answer_ids.reshape(-1),
        ignore_index=model.tokenizer.pad_token_id,
    )

    # ------- Contrastive loss (mild weight, pages should encode different info) -------
    finals_t = torch.tensor(gold_answers, dtype=torch.long, device=device)
    contrastive_loss = per_page_contrastive_loss(
        state_pages, finals_t, cross_target=0.7, within_target=0.3,
    )

    # ------- Answer head loss (digit prediction from last page) -------
    last_page = state_pages[-1].float()
    ah_loss = answer_head_loss(answer_head, last_page, finals_t)

    # ------- SymPy signal loss (probe head on last page predicts log-scaled answer) -------
    # gold_log range is [0, ~12] for GSM8K answers (up to ~100K)
    # Clamp probe output to prevent runaway MSE when weights diverge
    raw_probe = model.probe_head(last_page).squeeze(-1)  # (B,)
    probe_pred = torch.clamp(raw_probe, -1.0, 15.0)
    gold_log = torch.log(torch.abs(finals_t.float()) + 1.0)  # log scale
    sympy_signal_loss = F.mse_loss(probe_pred, gold_log)

    # ------- Diagnostics -------
    with torch.no_grad():
        # Log probe head stats on first call (helps debug sympy_signal_loss)
        if not hasattr(forward_train_with_sympy, '_logged_probe') or not forward_train_with_sympy._logged_probe:
            forward_train_with_sympy._logged_probe = True
            print(f"  [probe diag] raw_probe: {raw_probe.detach().tolist()}")
            print(f"  [probe diag] clamped:   {probe_pred.detach().tolist()}")
            print(f"  [probe diag] gold_log:  {gold_log.detach().tolist()}")
            print(f"  [probe diag] gold_ans:  {gold_answers[:4]}")
            print(f"  [probe diag] mse:       {sympy_signal_loss.item():.4f}")

        # Page cosine similarity (off-diagonal mean)
        normed = F.normalize(last_page, dim=-1)
        sim = normed @ normed.T
        B = sim.size(0)
        off_diag = sim - torch.eye(B, device=sim.device)
        page_cos_mean = off_diag.sum().item() / max(B * (B - 1), 1)

        # Active atoms: mean count of |scale| > 0.1 across all passes
        all_scales = torch.stack(atom_scales_history, dim=0)  # (P, B, A)
        active_per_pass = (all_scales.abs() > 0.1).float().sum(dim=-1)  # (P, B)
        active_atoms_mean = active_per_pass.mean().item()

    return (
        answer_loss, contrastive_loss, ah_loss, sympy_signal_loss,
        page_cos_mean, active_atoms_mean, state_pages
    )


# ---------------------------------------------------------------------------
# Evaluation (generation-based + answer head)
# ---------------------------------------------------------------------------

def evaluate(
    model: AtomLoRAModel,
    answer_head: AnswerHead,
    eval_dataset: Dataset,
    device: torch.device,
    num_passes: int = 5,
) -> Tuple[float, float]:
    """
    Evaluate via generation AND answer head digit prediction.

    Args:
        model: AtomLoRAModel
        answer_head: AnswerHead
        eval_dataset: Dataset to evaluate on
        device: torch device
        num_passes: Number of thinking passes

    Returns:
        (generation_accuracy, head_accuracy) as percentages
    """
    model.eval()
    answer_head.eval()
    gen_correct = 0
    head_correct = 0
    total = 0
    eval_batch = 4
    max_length = 192
    max_new_tokens = 80

    samples_printed = 0
    max_diag_prints = 10  # Print first 10 eval samples for debugging
    page_norms_collected = []  # Track page differentiation

    with torch.no_grad():
        for i in range(0, len(eval_dataset), eval_batch):
            batch_samples = [
                eval_dataset[j]
                for j in range(i, min(i + eval_batch, len(eval_dataset)))
            ]
            problems = [s['problem'] for s in batch_samples]
            gold_answers = [s['gold_answer'] for s in batch_samples]

            inputs = model.tokenizer(
                problems, return_tensors='pt', padding=True,
                truncation=True, max_length=max_length,
            )
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            batch_size = input_ids.size(0)

            state_pages = []
            mid_states_history = []

            for pass_num in range(num_passes):
                page, _scales, current_mid_states = model.thinking_pass(
                    input_ids, attention_mask, state_pages, pass_num,
                    prev_mid_states=mid_states_history if mid_states_history else None,
                )
                state_pages.append(page)
                mid_states_history.append(current_mid_states)

            # --- Generation-based eval ---
            # Append "The answer is " prefix so the model just outputs the number.
            # The thinking already happened in the passes — generation just extracts.
            prefix_text = "The answer is "
            prefix_ids = model.tokenizer.encode(
                prefix_text, add_special_tokens=False, return_tensors='pt',
            ).to(device)  # (1, P)
            prefix_ids = prefix_ids.expand(batch_size, -1)  # (B, P)
            gen_input_ids = torch.cat([input_ids, prefix_ids], dim=1)
            gen_attn_mask = torch.cat([
                attention_mask,
                torch.ones(batch_size, prefix_ids.size(1), device=device, dtype=attention_mask.dtype),
            ], dim=1)

            final_atom_scales = model.hypernet(
                state_pages, pass_num=len(state_pages),
            )
            manager = AtomAdditiveLoRAManager(model.transformer)
            manager.apply(model.atoms, final_atom_scales)
            try:
                generated = model.transformer.generate(
                    input_ids=gen_input_ids, attention_mask=gen_attn_mask,
                    max_new_tokens=20, do_sample=False,
                    pad_token_id=model.tokenizer.pad_token_id,
                )
            finally:
                manager.remove()

            # --- Answer head eval ---
            last_page = state_pages[-1].float()
            head_preds = answer_head.decode(last_page)  # (B,) tensor of ints

            # Track page differentiation (are pages different across problems?)
            page_norms_collected.append(last_page.detach().cpu())

            for j in range(batch_size):
                gold = gold_answers[j]

                # Generation accuracy — extract just the completion after prefix
                full_prompt_len = gen_input_ids[j].size(0)
                gen_ids = generated[j][full_prompt_len:]
                gen_text = model.tokenizer.decode(
                    gen_ids, skip_special_tokens=True,
                ).strip()
                # With "The answer is " prefix, model should output just a number
                pred = extract_answer_from_text(gen_text)

                # Diagnostic: print first N eval generations + answer head internals
                if samples_printed < max_diag_prints:
                    samples_printed += 1
                    print(f"  [eval {samples_printed}] gold={gold} pred={pred} "
                          f"head={head_preds[j].item() if gold is not None else '?'}")
                    print(f"    gen: {gen_text[:200]}")
                    # Answer head internals for first 3 samples
                    if samples_printed <= 3:
                        s_log, l_log, d_logs = answer_head(last_page[j:j+1])
                        print(f"    head sign:   {F.softmax(s_log, dim=-1)[0].tolist()}")
                        print(f"    head length: {F.softmax(l_log, dim=-1)[0].tolist()}")
                        digits = [d.argmax(dim=-1).item() for d in d_logs]
                        print(f"    head digits: {digits}")

                if pred is not None and gold is not None:
                    try:
                        pred_f = float(pred)
                        gold_f = float(gold)
                        if gold_f == int(gold_f):
                            if pred_f == gold_f:
                                gen_correct += 1
                        elif abs(pred_f - gold_f) < 0.01:
                            gen_correct += 1
                    except (ValueError, TypeError, OverflowError):
                        pass

                # Answer head accuracy
                if gold is not None:
                    try:
                        head_pred = head_preds[j].item()
                        gold_int = int(gold)
                        if head_pred == gold_int:
                            head_correct += 1
                    except (ValueError, TypeError):
                        pass

                total += 1

    # Page differentiation diagnostic
    if page_norms_collected:
        all_pages = torch.cat(page_norms_collected, dim=0)  # (N, 64)
        normed_pages = F.normalize(all_pages, dim=-1)
        cos_sim = normed_pages @ normed_pages.T
        N = cos_sim.size(0)
        off_diag = (cos_sim.sum() - N) / max(N * (N - 1), 1)
        page_std = all_pages.std(dim=0).mean()
        print(f"  [page diag] N={N} avg_cos_sim={off_diag:.4f} "
              f"per_dim_std={page_std:.4f} page_norm={all_pages.norm(dim=-1).mean():.2f}")

    gen_acc = 100.0 * gen_correct / total if total > 0 else 0.0
    head_acc = 100.0 * head_correct / total if total > 0 else 0.0
    return gen_acc, head_acc


# ---------------------------------------------------------------------------
# Teacher forcing probability schedule
# ---------------------------------------------------------------------------

def get_teacher_forcing_prob(epoch: int) -> float:
    """
    Get teacher forcing probability based on epoch.

    Schedule:
    - Epochs 1-3: 100% teacher forcing
    - Epochs 4-6: 50% teacher forcing
    - Epochs 7+: 0% teacher forcing
    """
    if epoch < 3:
        return 1.0
    elif epoch < 6:
        return 0.5
    else:
        return 0.0


# ---------------------------------------------------------------------------
# Warm-start logic
# ---------------------------------------------------------------------------

def try_warm_start(model: AtomLoRAModel, answer_head: AnswerHead, checkpoint_path: str):
    """
    Warm-start model from checkpoint.

    Tries to load atom checkpoint first; falls back to perceiver-only warm start.
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

    if 'atoms' in ckpt:
        # Atom checkpoint -- load directly
        print(f"  Loading atom checkpoint from {checkpoint_path}")

        # Compressor
        own = model.compressor.state_dict()
        loaded = 0
        for k, v in ckpt['compressor'].items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
                loaded += 1
        model.compressor.load_state_dict(own, strict=False)
        print(f"  compressor: loaded {loaded}/{len(own)}")

        # Atoms
        if 'atoms' in ckpt:
            own_a = model.atoms.state_dict()
            loaded_a = 0
            for k, v in ckpt['atoms'].items():
                if k in own_a and own_a[k].shape == v.shape:
                    own_a[k] = v
                    loaded_a += 1
            model.atoms.load_state_dict(own_a, strict=False)
            print(f"  atoms: loaded {loaded_a}/{len(own_a)}")

        # Hypernet
        if 'hypernet' in ckpt:
            own_h = model.hypernet.state_dict()
            loaded_h = 0
            for k, v in ckpt['hypernet'].items():
                if k in own_h and own_h[k].shape == v.shape:
                    own_h[k] = v
                    loaded_h += 1
            model.hypernet.load_state_dict(own_h, strict=False)
            print(f"  hypernet: loaded {loaded_h}/{len(own_h)}")

        # Confidence head
        if 'confidence_head' in ckpt:
            own_c = model.confidence_head.state_dict()
            loaded_c = 0
            for k, v in ckpt['confidence_head'].items():
                if k in own_c and own_c[k].shape == v.shape:
                    own_c[k] = v
                    loaded_c += 1
            model.confidence_head.load_state_dict(own_c, strict=False)
            print(f"  confidence_head: loaded {loaded_c}/{len(own_c)}")

        # Answer head
        if 'answer_head' in ckpt:
            own_ah = answer_head.state_dict()
            loaded_ah = 0
            for k, v in ckpt['answer_head'].items():
                if k in own_ah and own_ah[k].shape == v.shape:
                    own_ah[k] = v
                    loaded_ah += 1
            answer_head.load_state_dict(own_ah, strict=False)
            print(f"  answer_head: loaded {loaded_ah}/{len(own_ah)}")

        # Residual gate
        if 'residual_gate' in ckpt:
            own_rg = model.residual_gate.state_dict()
            loaded_rg = 0
            for k, v in ckpt['residual_gate'].items():
                if k in own_rg and own_rg[k].shape == v.shape:
                    own_rg[k] = v
                    loaded_rg += 1
            model.residual_gate.load_state_dict(own_rg, strict=False)
            print(f"  residual_gate: loaded {loaded_rg}/{len(own_rg)}")

    else:
        # Non-atom checkpoint -- load perceiver only
        print(f"  Warm-starting perceiver only from {checkpoint_path}")
        warm_start_atom_from_checkpoint(model, ckpt)
        print(f"  atoms: fresh init (no atom equivalent in checkpoint)")
        print(f"  hypernet: fresh init (new architecture)")
        print(f"  answer_head: fresh init")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    print("=" * 70)
    print("SymPy Teacher-Forced GSM8K Training (v24.7)")
    print("=" * 70)
    print(f"  epochs         = {args.epochs}")
    print(f"  batch_size     = {args.batch_size}")
    print(f"  max_passes     = {args.max_passes}")
    print(f"  lr             = {args.lr}")
    print(f"  max_samples    = {args.max_samples or 'all'}")
    print(f"  checkpoint     = {args.checkpoint or 'none'}")
    print(f"  eval_size      = {args.eval_size}")
    print(f"  lam_contrastive = {args.lam_contrastive}")
    print(f"  lam_answer_head = {args.lam_answer_head}")
    print(f"  lam_sympy       = {args.lam_sympy}")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Initialize model
    model = AtomLoRAModel(
        num_atoms=args.num_atoms,
        atom_rank=args.atom_rank,
        use_pattern_memory=args.use_pattern_memory,
        pattern_memory_db_path=args.pattern_memory_db,
    )
    if args.use_pattern_memory:
        print(f"Pattern Memory ENABLED (db: {args.pattern_memory_db})")
    model.compressor = model.compressor.to(device=device, dtype=torch.bfloat16)
    model.hypernet = model.hypernet.to(device=device, dtype=torch.bfloat16)
    model.atoms = model.atoms.to(device=device, dtype=torch.bfloat16)
    model.confidence_head = model.confidence_head.to(device)  # fp32 (small)
    model.residual_gate = model.residual_gate.to(device=device, dtype=torch.bfloat16)
    model.probe_head = model.probe_head.to(device)  # fp32

    # Answer head -- small, kept in float32 for digit classification stability
    answer_head = AnswerHead(page_size=model.page_size).to(device)

    # SymPy result encoder -- small, kept in float32
    sympy_encoder = SymPyResultEncoder(page_size=model.page_size, max_variables=8).to(device)

    # Warm start if checkpoint provided
    if args.checkpoint:
        print(f"\nWarm-starting from {args.checkpoint}")
        try_warm_start(model, answer_head, args.checkpoint)

    # Load datasets
    train_dataset = GSM8KSymPyDataset(max_samples=args.max_samples)
    eval_dataset = GSM8KSymPyDataset(max_samples=args.eval_size)

    print(f"\nDatasets: {len(train_dataset)} train, {len(eval_dataset)} eval")

    # Sample problem preview
    print("\nSample problems:")
    for i in range(min(2, len(train_dataset))):
        s = train_dataset[i]
        print(f"  Problem: {s['problem'][:80]}...")
        print(f"  Answer: {s['gold_answer']}")
        print(f"  Steps: {s['sympy_steps']}")
        print()

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn,
    )

    # ------- Optimizer: separate LRs as specified -------
    optimizer = torch.optim.AdamW([
        # Perceiver: 1e-4
        {'params': list(model.compressor.parameters()), 'lr': args.lr, 'weight_decay': 0.01},
        # Atoms: 1e-3 (10x)
        {'params': list(model.atoms.parameters()), 'lr': args.lr * 10, 'weight_decay': 0.05},
        # Hypernetwork: 1e-3 (10x)
        {'params': list(model.hypernet.parameters()), 'lr': args.lr * 10, 'weight_decay': 0.1},
        # SymPy encoder: 1e-4
        {'params': list(sympy_encoder.parameters()), 'lr': args.lr, 'weight_decay': 0.01},
        # Answer head: 1e-3
        {'params': list(answer_head.parameters()), 'lr': args.lr * 10, 'weight_decay': 0.01},
        # Confidence head: 1e-3
        {'params': list(model.confidence_head.parameters()), 'lr': args.lr * 10, 'weight_decay': 0.01},
        # Probe head: 1e-4
        {'params': list(model.probe_head.parameters()), 'lr': args.lr, 'weight_decay': 0.01},
        # Residual gate: 1e-4
        {'params': list(model.residual_gate.parameters()), 'lr': args.lr, 'weight_decay': 0.01},
    ])

    trainable = (
        list(model.compressor.parameters())
        + list(model.atoms.parameters())
        + list(model.hypernet.parameters())
        + list(sympy_encoder.parameters())
        + list(answer_head.parameters())
        + list(model.confidence_head.parameters())
        + list(model.probe_head.parameters())
        + list(model.residual_gate.parameters())
    )
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")
    print(f"  compressor: {sum(p.numel() for p in model.compressor.parameters()):,}")
    print(f"  atoms: {sum(p.numel() for p in model.atoms.parameters()):,}")
    print(f"  hypernet: {sum(p.numel() for p in model.hypernet.parameters()):,}")
    print(f"  sympy_encoder: {sum(p.numel() for p in sympy_encoder.parameters()):,}")
    print(f"  answer_head: {sum(p.numel() for p in answer_head.parameters()):,}")
    print(f"GPU mem after init: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Baseline evaluation
    base_gen, base_head = evaluate(
        model, answer_head, eval_dataset, device, num_passes=args.max_passes,
    )
    print(f"\nBaseline: gen={base_gen:.1f}% head={base_head:.1f}%\n")

    # Training state
    ckpt_name = 'checkpoints/sympy_gsm8k_best.pt'
    os.makedirs('checkpoints', exist_ok=True)
    best_gen = 0.0
    best_head = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        answer_head.train()
        sympy_encoder.train()
        t0 = time.time()

        # Get teacher forcing probability for this epoch
        tf_prob = get_teacher_forcing_prob(epoch)
        print(f"\n[Epoch {epoch+1}/{args.epochs}] Teacher forcing: {tf_prob:.0%}")
        forward_train_with_sympy._logged_probe = False  # Reset probe diag for this epoch

        ep_ans = ep_ctr = ep_ah = ep_sympy = 0.0
        ep_cos = ep_active = 0.0
        ep_patterns_stored = 0
        nb = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            problems = batch['problem']
            gold_answers = batch['gold_answer']
            sympy_steps = batch['sympy_steps']

            # Convert gold_answers to ints
            gold_ints = []
            for g in gold_answers:
                try:
                    gold_ints.append(int(g))
                except (ValueError, TypeError):
                    gold_ints.append(0)

            optimizer.zero_grad()

            (ans_loss, ctr_loss, ah_loss, sympy_loss,
             page_cos, active_atoms, state_pages) = forward_train_with_sympy(
                model, answer_head, sympy_encoder,
                problems, gold_ints, sympy_steps,
                teacher_forcing_prob=tf_prob,
                num_passes=args.max_passes,
            )

            total_loss = (
                ans_loss
                + args.lam_contrastive * ctr_loss
                + args.lam_answer_head * ah_loss
                + args.lam_sympy * sympy_loss
            )

            # NaN check - catch issues early
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"\n[WARNING] NaN/Inf detected in batch!")
                print(f"  ans_loss: {ans_loss.item()}")
                print(f"  ctr_loss: {ctr_loss.item()}")
                print(f"  ah_loss: {ah_loss.item()}")
                print(f"  sympy_loss: {sympy_loss.item()}")
                print(f"  Skipping this batch...")
                optimizer.zero_grad()
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            # --- Pattern Memory: store successful patterns ---
            if args.use_pattern_memory and model.pattern_memory is not None and len(state_pages) > 0:
                with torch.no_grad():
                    # Decode predictions from answer head
                    last_page = state_pages[-1].float()
                    predictions = answer_head.decode(last_page)  # (B,)

                    # Compare to gold and store patterns for correct answers
                    for b in range(len(problems)):
                        pred = predictions[b].item()
                        gold = gold_ints[b]
                        was_correct = (pred == gold)

                        # Store successful patterns with SymPy steps
                        if was_correct and sympy_steps[b]:
                            model.after_solve(
                                problem_text=problems[b],
                                state_pages=[sp[b:b+1] for sp in state_pages],  # Single sample pages
                                sympy_steps=sympy_steps[b],
                                was_correct=True,
                                used_pattern_id=None,  # No pattern queried during training (yet)
                                epoch=epoch,
                            )
                            ep_patterns_stored += 1

            ep_ans += ans_loss.item()
            ep_ctr += ctr_loss.item()
            ep_ah += ah_loss.item()
            ep_sympy += sympy_loss.item()
            ep_cos += page_cos
            ep_active += active_atoms
            nb += 1

        elapsed = time.time() - t0

        # Evaluation
        gen_acc, head_acc = evaluate(
            model, answer_head, eval_dataset, device, num_passes=args.max_passes,
        )

        # Check for improvement
        improved = False
        if gen_acc > best_gen:
            best_gen = gen_acc
            improved = True
        if head_acc > best_head:
            best_head = head_acc
            improved = True

        if improved:
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'compressor': model.compressor.state_dict(),
                'atoms': model.atoms.state_dict(),
                'hypernet': model.hypernet.state_dict(),
                'sympy_encoder': sympy_encoder.state_dict(),
                'answer_head': answer_head.state_dict(),
                'confidence_head': model.confidence_head.state_dict(),
                'probe_head': model.probe_head.state_dict(),
                'residual_gate': model.residual_gate.state_dict(),
                'gen_accuracy': gen_acc,
                'head_accuracy': head_acc,
                'num_atoms': args.num_atoms,
                'atom_rank': args.atom_rank,
            }, ckpt_name)
            print(f"  -> saved checkpoint {ckpt_name} (gen={gen_acc:.1f}% head={head_acc:.1f}%)")
        else:
            patience_counter += 1

        # Guard against all-NaN epochs
        if nb == 0:
            print(f"Epoch {epoch+1}: ALL BATCHES NaN - training diverged! Stopping.")
            break

        print(
            f"Epoch {epoch+1}: "
            f"ans={ep_ans/nb:.4f} ctr={ep_ctr/nb:.3f} ah={ep_ah/nb:.3f} sympy={ep_sympy/nb:.3f} "
            f"page_cos={ep_cos/nb:.3f} active={ep_active/nb:.1f} | "
            f"Acc: gen={gen_acc:.1f}% head={head_acc:.1f}% "
            f"best_gen={best_gen:.1f}% best_head={best_head:.1f}% "
            f"[{elapsed:.0f}s]"
        )

        # Pattern memory maintenance (if enabled)
        if args.use_pattern_memory and model.pattern_memory is not None:
            # Get stats before pruning
            stats = model.pattern_memory.stats()
            # Prune low-value patterns (after 2nd epoch)
            pruned = 0
            if epoch >= 1:
                pruned = model.pattern_memory.prune(
                    min_success_rate=0.3, min_uses=3, max_patterns=10000,
                )
            print(
                f"  Pattern Memory: +{ep_patterns_stored} stored this epoch, "
                f"{stats.get('total_patterns', 0)} total patterns, "
                f"{stats.get('avg_success_rate', 0):.1%} avg success"
                + (f", pruned {pruned}" if pruned > 0 else "")
            )
            # Print pattern type distribution
            type_dist = stats.get('type_distribution', {})
            if type_dist:
                top_types = sorted(type_dist.items(), key=lambda x: -x[1])[:5]
                types_str = ", ".join(f"{t}:{c}" for t, c in top_types)
                print(f"  Pattern Types: {types_str}")

        if patience_counter >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} epochs")
            break

    print("\n" + "=" * 70)
    print(f"Training complete!")
    print(f"  Best generation accuracy: {best_gen:.1f}%")
    print(f"  Best answer head accuracy: {best_head:.1f}%")
    print(f"  Baseline: gen={base_gen:.1f}% head={base_head:.1f}%")
    print(f"  Checkpoint: {ckpt_name}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SymPy Teacher-Forced GSM8K Training',
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Checkpoint path to warm start from',
    )
    parser.add_argument(
        '--epochs', type=int, default=10,
        help='Number of training epochs (default: 10)',
    )
    parser.add_argument(
        '--batch_size', type=int, default=4,
        help='Batch size (default: 4)',
    )
    parser.add_argument(
        '--max_passes', type=int, default=5,
        help='Number of thinking passes (default: 5)',
    )
    parser.add_argument(
        '--lr', type=float, default=1e-4,
        help='Base learning rate for perceiver (default: 1e-4)',
    )
    parser.add_argument(
        '--max_samples', type=int, default=None,
        help='Max training samples (default: all, use for debugging)',
    )
    parser.add_argument(
        '--eval_size', type=int, default=200,
        help='Number of eval samples (default: 200)',
    )
    parser.add_argument(
        '--patience', type=int, default=4,
        help='Early stopping patience (default: 4)',
    )
    parser.add_argument(
        '--lam_contrastive', type=float, default=0.01,
        help='Contrastive loss weight (default: 0.01, mild)',
    )
    parser.add_argument(
        '--lam_answer_head', type=float, default=0.3,
        help='Answer head loss weight (default: 0.3)',
    )
    parser.add_argument(
        '--lam_sympy', type=float, default=0.1,
        help='SymPy signal loss weight (default: 0.1)',
    )
    parser.add_argument(
        '--num_atoms', type=int, default=64,
        help='Number of LoRA atoms (default: 64)',
    )
    parser.add_argument(
        '--atom_rank', type=int, default=6,
        help='Rank of each LoRA atom (default: 6)',
    )
    parser.add_argument(
        '--use_pattern_memory', action='store_true',
        help='Enable Pattern Memory (SQLite long-term memory for reasoning patterns)',
    )
    parser.add_argument(
        '--pattern_memory_db', type=str, default='pattern_memory.db',
        help='Path to Pattern Memory SQLite database (default: pattern_memory.db)',
    )

    args = parser.parse_args()
    train(args)
