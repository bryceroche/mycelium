#!/usr/bin/env python3
"""
SymPy Decoder Training (v24.9)

Separated comprehension and computation:
- LLM comprehends (reads problem with atom-modified attention)
- Perceiver compresses (hidden states → 64-float page)
- SymPy decoder formulates (page → SymPy expression, ~50 token vocab)
- SymPy computes (exact arithmetic)
- Result feeds back into page

NO LLM generation loss. The generation path killed page diversity because it
doesn't need per-problem pages. The SymPy decoder INHERENTLY needs per-problem
pages (different problems need different formulas), so page diversity falls out
naturally.

Loss = sympy_decoder_loss + answer_head_loss + contrastive_loss + scale_reg

All five loop-alive fixes are mandatory:
  ✓ skip_pass_embed=True
  ✓ scale_reg=0.1
  ✓ lam_answer_head=1.0
  ✓ direct_path (skip connections)
  ✓ residual_gate

Usage:
  python scripts/train_sympy_decoder.py --checkpoint checkpoints/atom_lora_L4_best.pt --fresh_compressor
  python scripts/train_sympy_decoder.py --epochs 10 --max_samples 200
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

from scripts.atom_lora import (
    AtomLoRAModel,
    AtomAdditiveLoRAManager,
    AnswerHead,
    answer_head_loss,
)
from src.sympy_decoder import SymPyDecoder, SymPyVocab
from src.sympy_evaluator import SymPyEvaluator
from src.sympy_result_encoder import SymPyResultEncoder
from src.fourier_init import apply_fourier_inits


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GSM8KSymPyDataset(Dataset):
    def __init__(self, jsonl_path=None, max_samples=None):
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


def collate_fn(batch):
    return {
        'problem': [s['problem'] for s in batch],
        'gold_answer': [s['gold_answer'] for s in batch],
        'sympy_steps': [s['sympy_steps'] for s in batch],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def scale_gradient(tensor, scale):
    """Scale gradient during backward without affecting forward."""
    return tensor + (scale - 1.0) * tensor.detach()


def extract_answer_from_text(text: str):
    """Extract numeric answer from text."""
    if "\nProblem:" in text:
        text = text.split("\nProblem:")[0]
    m = re.search(r'[Tt]he answer is\s*\$?(-?[\d,]+\.?\d*)', text)
    if m:
        try:
            v = float(m.group(1).replace(',', ''))
            return int(v) if v == int(v) else v
        except ValueError:
            pass
    nums = re.findall(r'-?[\d,]+\.?\d*', text)
    if nums:
        try:
            v = float(nums[-1].replace(',', ''))
            return int(v) if v == int(v) else v
        except ValueError:
            pass
    return None


def format_sympy_context(results: Dict) -> str:
    """Format SymPy results as context string for LLM input."""
    if not results:
        return ""
    parts = [f"{k} = {v}" for k, v in results.items()]
    return "Known values: " + ", ".join(parts) + ". "


def get_teacher_forcing_prob(epoch: int) -> float:
    """Progressive teacher forcing schedule."""
    if epoch < 5:
        return 1.0
    elif epoch < 10:
        return 0.5
    else:
        return 0.0


# ---------------------------------------------------------------------------
# Forward pass with SymPy decoder
# ---------------------------------------------------------------------------

def forward_train(
    model: AtomLoRAModel,
    answer_head: AnswerHead,
    sympy_decoder: SymPyDecoder,
    sympy_encoder: SymPyResultEncoder,
    problems: List[str],
    gold_answers: List[int],
    sympy_steps_batch: List[List[str]],
    teacher_forcing_prob: float = 1.0,
    num_passes: int = 5,
    max_length: int = 192,
):
    """
    Forward pass with separated comprehension and computation.

    Each pass:
      1. Hypernetwork reads pages → atom scales
      2. Apply atom LoRA, run Llama (comprehension)
      3. Perceiver compresses to page (collapse)
      4. SymPy decoder reads page → expression (formulation, teacher forced)
      5. SymPy evaluates expression (computation)
      6. Result encoded back into page (feedback)

    NO LLM generation loss — only sympy_decoder_loss + answer_head_loss.
    """
    device = model.transformer.device
    batch_size = len(problems)
    vocab = sympy_decoder.vocab

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
    pre_tanh_history = []

    # Per-problem accumulated SymPy results
    sympy_results_batch = [{} for _ in range(batch_size)]

    # Accumulate decoder loss across passes
    total_sympy_decoder_loss = torch.tensor(0.0, device=device)
    num_decoder_steps = 0

    use_teacher = random.random() < teacher_forcing_prob

    for pass_num in range(num_passes):
        # Execute teacher SymPy for this pass (accumulate results for feedback)
        if use_teacher:
            for b in range(batch_size):
                steps_to_run = sympy_steps_batch[b][:pass_num + 1]
                if steps_to_run:
                    cumulative_code = '\n'.join(steps_to_run)
                    step_results = SymPyEvaluator.safe_eval(cumulative_code)
                    sympy_results_batch[b] = step_results

        # --- Comprehension: Llama with atom LoRA ---
        if pass_num == 0:
            outputs = model.transformer(
                inputs_embeds=problem_embeds, attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = list(outputs.hidden_states[1:])
            atom_scales = torch.zeros(batch_size, model.num_atoms, device=device)
            pre_tanh = torch.zeros_like(atom_scales)
        else:
            atom_scales, pre_tanh = model.hypernet(
                state_pages, pass_num=pass_num, return_pre_tanh=True,
            )
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

        atom_scales_history.append(atom_scales)
        pre_tanh_history.append(pre_tanh)

        # --- Collapse: Perceiver compresses to page ---
        page_delta, _strategy, current_mid_states = model.compressor(
            hidden_states, pass_num,
            prev_mid_states=mid_states_history if mid_states_history else None,
            state_pages=state_pages,
        )
        mid_states_history.append(current_mid_states)

        # Normalize on hypersphere
        page = F.normalize(page_delta, dim=-1) * model.page_radius

        # Fourier structural identity
        page = model.fourier_page.apply(page, pass_num)

        # Residual page gate
        if len(state_pages) > 0:
            page = model.residual_gate(page, state_pages[-1])

        # --- Formulation: SymPy decoder reads page → expression ---
        if use_teacher and pass_num < max(len(s) for s in sympy_steps_batch):
            # Get this pass's SymPy step for each problem
            step_expressions = []
            has_step = []
            for b in range(batch_size):
                if pass_num < len(sympy_steps_batch[b]):
                    step_expressions.append(sympy_steps_batch[b][pass_num])
                    has_step.append(True)
                else:
                    step_expressions.append("")
                    has_step.append(False)

            # Only compute decoder loss for problems that have a step at this pass
            if any(has_step):
                # Encode target tokens
                target_tokens, target_mask = vocab.batch_encode(
                    step_expressions, device=device,
                )

                # Build result embedding from previous pass results
                result_embedding = None
                if any(sympy_results_batch[b] for b in range(batch_size)):
                    result_embedding = sympy_encoder.forward_batch(
                        sympy_results_batch, device=device,
                    ).to(dtype=page.dtype)

                # Decoder loss (teacher forced)
                step_loss = sympy_decoder.compute_loss(
                    page.float(), target_tokens, result_embedding=result_embedding.float() if result_embedding is not None else None,
                )

                # Mask out problems without a step at this pass
                total_sympy_decoder_loss = total_sympy_decoder_loss + step_loss
                num_decoder_steps += 1

        # --- Feedback: Encode SymPy results into page ---
        if use_teacher:
            sympy_vec = sympy_encoder.forward_batch(sympy_results_batch, device=device)
            page = page + sympy_vec.to(dtype=page.dtype)

        # Page noise during training
        if model.training:
            page = page + torch.randn_like(page) * 0.05

        # Gradient scaling per cycle
        grad_scale = min(float(num_passes - pass_num), 4.0)
        page = scale_gradient(page, grad_scale)

        state_pages.append(page)

    # --- Average decoder loss across passes ---
    if num_decoder_steps > 0:
        sympy_decoder_loss = total_sympy_decoder_loss / num_decoder_steps
    else:
        sympy_decoder_loss = torch.tensor(0.0, device=device)

    # --- Contrastive loss ---
    finals_t = torch.tensor(gold_answers, dtype=torch.long, device=device)
    last_page = state_pages[-1].float()

    normed = F.normalize(last_page, dim=-1)
    sim = normed @ normed.T
    B = sim.size(0)
    off_diag = sim - torch.eye(B, device=sim.device)
    contrastive_loss = F.relu(off_diag - 0.7).pow(2).sum() / max(B * (B - 1), 1)

    # --- Answer head loss ---
    ah_loss = answer_head_loss(answer_head, last_page, finals_t)

    # --- Pre-tanh regularization ---
    all_pre_tanh = torch.cat(pre_tanh_history, dim=0)
    scale_reg_loss = (all_pre_tanh ** 2).mean()

    # --- Diagnostics ---
    with torch.no_grad():
        page_cos_mean = off_diag.sum().item() / max(B * (B - 1), 1)
        all_scales = torch.stack(atom_scales_history, dim=0)
        active_per_pass = (all_scales.abs() > 0.1).float().sum(dim=-1)
        active_atoms_mean = active_per_pass.mean().item()

    return (
        sympy_decoder_loss, contrastive_loss, ah_loss, scale_reg_loss,
        page_cos_mean, active_atoms_mean, state_pages,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: AtomLoRAModel,
    answer_head: AnswerHead,
    sympy_decoder: SymPyDecoder,
    sympy_encoder: SymPyResultEncoder,
    eval_dataset: Dataset,
    device: torch.device,
    num_passes: int = 5,
) -> Tuple[float, float, float]:
    """
    Evaluate via:
    1. SymPy decoder (formulate → compute → check answer)
    2. Answer head (digit prediction from last page)
    3. LLM generation with "The answer is " prefix (fallback)

    Returns: (sympy_accuracy, head_accuracy, gen_accuracy)
    """
    model.eval()
    answer_head.eval()
    sympy_decoder.eval()

    sympy_correct = 0
    head_correct = 0
    gen_correct = 0
    total = 0
    eval_batch = 4
    max_length = 192
    samples_printed = 0

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
            sympy_results_batch = [{} for _ in range(batch_size)]

            for pass_num in range(num_passes):
                page, _scales, current_mid_states = model.thinking_pass(
                    input_ids, attention_mask, state_pages, pass_num,
                    prev_mid_states=mid_states_history if mid_states_history else None,
                )
                state_pages.append(page)
                mid_states_history.append(current_mid_states)

                # SymPy decoder generates expression from page
                result_embedding = None
                if any(sympy_results_batch[b] for b in range(batch_size)):
                    result_embedding = sympy_encoder.forward_batch(
                        sympy_results_batch, device=device,
                    ).to(dtype=page.dtype).float()

                expressions = sympy_decoder._generate(
                    sympy_decoder._build_memory(page.float(), result_embedding),
                    device,
                )

                # Execute each expression
                for b in range(batch_size):
                    expr = expressions[b]
                    if expr.strip():
                        try:
                            results = SymPyEvaluator.safe_eval(expr)
                            sympy_results_batch[b].update(results)
                        except Exception:
                            pass

            # --- Evaluate all three paths ---
            last_page = state_pages[-1].float()
            head_preds = answer_head.decode(last_page)

            # LLM generation with prefix
            prefix_text = "The answer is "
            prefix_ids = model.tokenizer.encode(
                prefix_text, add_special_tokens=False, return_tensors='pt',
            ).to(device)
            prefix_ids = prefix_ids.expand(batch_size, -1)
            gen_input_ids = torch.cat([input_ids, prefix_ids], dim=1)
            gen_attn_mask = torch.cat([
                attention_mask,
                torch.ones(batch_size, prefix_ids.size(1), device=device, dtype=attention_mask.dtype),
            ], dim=1)

            final_atom_scales = model.hypernet(state_pages, pass_num=len(state_pages))
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

            for j in range(batch_size):
                gold = gold_answers[j]
                if gold is None:
                    total += 1
                    continue

                try:
                    gold_int = int(gold)
                except (ValueError, TypeError):
                    total += 1
                    continue

                # 1. SymPy decoder accuracy
                sympy_answer = sympy_results_batch[j].get('answer')
                if sympy_answer is not None:
                    try:
                        if int(float(sympy_answer)) == gold_int:
                            sympy_correct += 1
                    except (ValueError, TypeError, OverflowError):
                        pass

                # 2. Answer head accuracy
                try:
                    if head_preds[j].item() == gold_int:
                        head_correct += 1
                except (ValueError, TypeError):
                    pass

                # 3. Generation accuracy
                full_prompt_len = gen_input_ids[j].size(0)
                gen_ids = generated[j][full_prompt_len:]
                gen_text = model.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                pred = extract_answer_from_text(gen_text)

                # Diagnostic prints
                if samples_printed < 10:
                    samples_printed += 1
                    sympy_ans_str = sympy_results_batch[j].get('answer', '?')
                    print(f"  [eval {samples_printed}] gold={gold} "
                          f"sympy={sympy_ans_str} head={head_preds[j].item()} gen_pred={pred}")
                    # Print what the decoder generated on last pass
                    last_expr = expressions[j] if j < len(expressions) else '?'
                    print(f"    decoder: {last_expr}")
                    print(f"    gen: {gen_text[:120]}")

                if pred is not None:
                    try:
                        if float(pred) == float(gold_int):
                            gen_correct += 1
                    except (ValueError, TypeError, OverflowError):
                        pass

                total += 1

    sympy_acc = 100.0 * sympy_correct / total if total > 0 else 0.0
    head_acc = 100.0 * head_correct / total if total > 0 else 0.0
    gen_acc = 100.0 * gen_correct / total if total > 0 else 0.0
    return sympy_acc, head_acc, gen_acc


# ---------------------------------------------------------------------------
# Checkpoint loading (selective)
# ---------------------------------------------------------------------------

def try_warm_start(model, answer_head, checkpoint_path, fresh_compressor=False):
    """Warm-start from checkpoint. Optionally skip compressor (fresh perceiver)."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

    if 'atoms' not in ckpt:
        print(f"  No atom checkpoint found in {checkpoint_path}, skipping")
        return

    print(f"  Loading from {checkpoint_path}")
    if fresh_compressor:
        print(f"  ** FRESH COMPRESSOR ** — resetting perceiver")

    # Compressor
    if not fresh_compressor:
        own = model.compressor.state_dict()
        loaded = 0
        for k, v in ckpt['compressor'].items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
                loaded += 1
        model.compressor.load_state_dict(own, strict=False)
        print(f"  compressor: loaded {loaded}/{len(own)}")
    else:
        print(f"  compressor: FRESH (0/{len(model.compressor.state_dict())})")

    # Atoms
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
        own_ah = model.answer_head if hasattr(model, 'answer_head') else answer_head
        own_ah_sd = own_ah.state_dict()
        loaded_ah = 0
        for k, v in ckpt['answer_head'].items():
            if k in own_ah_sd and own_ah_sd[k].shape == v.shape:
                own_ah_sd[k] = v
                loaded_ah += 1
        own_ah.load_state_dict(own_ah_sd, strict=False)
        print(f"  answer_head: loaded {loaded_ah}/{len(own_ah_sd)}")

    # Residual gate (skip if fresh compressor)
    if 'residual_gate' in ckpt and not fresh_compressor:
        own_rg = model.residual_gate.state_dict()
        loaded_rg = 0
        for k, v in ckpt['residual_gate'].items():
            if k in own_rg and own_rg[k].shape == v.shape:
                own_rg[k] = v
                loaded_rg += 1
        model.residual_gate.load_state_dict(own_rg, strict=False)
        print(f"  residual_gate: loaded {loaded_rg}/{len(own_rg)}")
    elif fresh_compressor:
        print(f"  residual_gate: FRESH (coupled with compressor)")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    print("=" * 70)
    print("SymPy Decoder Training (v24.9)")
    print("=" * 70)
    print(f"  epochs          = {args.epochs}")
    print(f"  batch_size      = {args.batch_size}")
    print(f"  max_passes      = {args.max_passes}")
    print(f"  lr              = {args.lr}")
    print(f"  lam_answer_head = {args.lam_answer_head}")
    print(f"  lam_contrastive = {args.lam_contrastive}")
    print(f"  lam_scale_reg   = {args.lam_scale_reg}")
    print(f"  lam_sympy_dec   = {args.lam_sympy_dec}")
    print(f"  skip_pass_embed = True")
    print(f"  fresh_compressor= {args.fresh_compressor}")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- Model ---
    model = AtomLoRAModel(
        num_atoms=args.num_atoms,
        atom_rank=args.atom_rank,
        skip_pass_embed=True,  # MANDATORY
    )
    model.compressor = model.compressor.to(device=device, dtype=torch.bfloat16)
    model.hypernet = model.hypernet.to(device=device, dtype=torch.bfloat16)
    model.atoms = model.atoms.to(device=device, dtype=torch.bfloat16)
    model.confidence_head = model.confidence_head.to(device)
    model.residual_gate = model.residual_gate.to(device=device, dtype=torch.bfloat16)

    # --- Answer head ---
    answer_head = AnswerHead(page_size=64, max_digits=6).to(device)

    # --- SymPy decoder (NEW) ---
    sympy_decoder = SymPyDecoder(
        page_size=64, d_model=256, nhead=4, num_layers=2, max_tokens=40,
    ).to(device)
    decoder_params = sum(p.numel() for p in sympy_decoder.parameters())
    print(f"SymPy decoder: {decoder_params:,} params ({decoder_params/1e6:.1f}M)")
    print(f"  vocab size: {len(sympy_decoder.vocab)}")

    # --- SymPy result encoder ---
    sympy_encoder = SymPyResultEncoder(page_size=64).to(device)

    # --- Fourier init (fresh training only, before warm start) ---
    if args.fourier_init:
        print()
        apply_fourier_inits(model)

    # --- Warm start (overwrites Fourier-initialized atoms if checkpoint has them) ---
    if args.checkpoint:
        print(f"\nWarm-starting from {args.checkpoint}")
        try_warm_start(model, answer_head, args.checkpoint,
                       fresh_compressor=args.fresh_compressor)

    # --- Data ---
    train_dataset = GSM8KSymPyDataset(max_samples=args.max_samples)
    eval_dataset = GSM8KSymPyDataset(max_samples=args.eval_size)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=True,
    )
    print(f"\nDatasets: {len(train_dataset)} train, {len(eval_dataset)} eval")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW([
        {'params': list(model.compressor.parameters()), 'lr': args.lr, 'weight_decay': 0.01},
        {'params': list(model.atoms.parameters()), 'lr': args.lr, 'weight_decay': 0.01},
        {'params': list(model.hypernet.parameters()), 'lr': args.lr * 10, 'weight_decay': 0.01},
        {'params': list(answer_head.parameters()), 'lr': args.lr * 10, 'weight_decay': 0.01},
        {'params': list(sympy_decoder.parameters()), 'lr': args.lr * 5, 'weight_decay': 0.01},
        {'params': list(sympy_encoder.parameters()), 'lr': args.lr * 5, 'weight_decay': 0.01},
        {'params': list(model.confidence_head.parameters()), 'lr': args.lr, 'weight_decay': 0.01},
        {'params': list(model.residual_gate.parameters()), 'lr': args.lr, 'weight_decay': 0.01},
    ])

    trainable = (
        list(model.compressor.parameters())
        + list(model.atoms.parameters())
        + list(model.hypernet.parameters())
        + list(answer_head.parameters())
        + list(sympy_decoder.parameters())
        + list(sympy_encoder.parameters())
        + list(model.confidence_head.parameters())
        + list(model.residual_gate.parameters())
    )
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")
    print(f"GPU mem after init: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # --- Baseline eval ---
    sympy_acc, head_acc, gen_acc = evaluate(
        model, answer_head, sympy_decoder, sympy_encoder,
        eval_dataset, device, num_passes=args.max_passes,
    )
    print(f"\nBaseline: sympy={sympy_acc:.1f}% head={head_acc:.1f}% gen={gen_acc:.1f}%\n")

    # --- Training ---
    ckpt_name = 'checkpoints/sympy_decoder_best.pt'
    os.makedirs('checkpoints', exist_ok=True)
    best_sympy = 0.0
    best_head = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        answer_head.train()
        sympy_decoder.train()
        sympy_encoder.train()
        t0 = time.time()

        tf_prob = get_teacher_forcing_prob(epoch)
        print(f"\n[Epoch {epoch+1}/{args.epochs}] Teacher forcing: {tf_prob:.0%}")

        ep_dec = ep_ctr = ep_ah = ep_sreg = 0.0
        ep_cos = ep_active = 0.0
        nb = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            problems = batch['problem']
            gold_answers = batch['gold_answer']
            sympy_steps = batch['sympy_steps']

            gold_ints = []
            for g in gold_answers:
                try:
                    gold_ints.append(int(g))
                except (ValueError, TypeError):
                    gold_ints.append(0)

            optimizer.zero_grad()

            (dec_loss, ctr_loss, ah_loss, scale_reg,
             page_cos, active_atoms, state_pages) = forward_train(
                model, answer_head, sympy_decoder, sympy_encoder,
                problems, gold_ints, sympy_steps,
                teacher_forcing_prob=tf_prob,
                num_passes=args.max_passes,
            )

            # NO generation loss! Only decoder + answer head + regularization
            total_loss = (
                args.lam_sympy_dec * dec_loss
                + args.lam_answer_head * ah_loss
                + args.lam_contrastive * ctr_loss
                + args.lam_scale_reg * scale_reg
            )

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"\n[WARNING] NaN/Inf! dec={dec_loss.item():.3f} "
                      f"ah={ah_loss.item():.3f} ctr={ctr_loss.item():.3f} "
                      f"sreg={scale_reg.item():.3f}")
                optimizer.zero_grad()
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            ep_dec += dec_loss.item()
            ep_ctr += ctr_loss.item()
            ep_ah += ah_loss.item()
            ep_sreg += scale_reg.item()
            ep_cos += page_cos
            ep_active += active_atoms
            nb += 1

        elapsed = time.time() - t0

        # Eval
        sympy_acc, head_acc, gen_acc = evaluate(
            model, answer_head, sympy_decoder, sympy_encoder,
            eval_dataset, device, num_passes=args.max_passes,
        )

        # Check improvement
        improved = False
        if sympy_acc > best_sympy:
            best_sympy = sympy_acc
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
                'sympy_decoder': sympy_decoder.state_dict(),
                'sympy_encoder': sympy_encoder.state_dict(),
                'answer_head': answer_head.state_dict(),
                'confidence_head': model.confidence_head.state_dict(),
                'residual_gate': model.residual_gate.state_dict(),
                'sympy_accuracy': sympy_acc,
                'head_accuracy': head_acc,
                'gen_accuracy': gen_acc,
            }, ckpt_name)
            print(f"  -> saved {ckpt_name}")
        else:
            patience_counter += 1

        if nb == 0:
            print(f"Epoch {epoch+1}: ALL BATCHES NaN - stopping.")
            break

        print(
            f"Epoch {epoch+1}: "
            f"dec={ep_dec/nb:.3f} ctr={ep_ctr/nb:.3f} ah={ep_ah/nb:.3f} sreg={ep_sreg/nb:.3f} "
            f"page_cos={ep_cos/nb:.3f} active={ep_active/nb:.1f} | "
            f"Acc: sympy={sympy_acc:.1f}% head={head_acc:.1f}% gen={gen_acc:.1f}% "
            f"best_sympy={best_sympy:.1f}% best_head={best_head:.1f}% "
            f"[{elapsed:.0f}s]"
        )

        if patience_counter >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} epochs")
            break

    print("\n" + "=" * 70)
    print(f"Training complete!")
    print(f"  Best SymPy decoder accuracy: {best_sympy:.1f}%")
    print(f"  Best answer head accuracy: {best_head:.1f}%")
    print(f"  Checkpoint: {ckpt_name}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SymPy Decoder Training (v24.9)')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--fresh_compressor', action='store_true')
    parser.add_argument('--fourier_init', action='store_true',
                        help='Apply Fourier atom init + frequency-aware residual gate (fresh training only)')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_passes', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--eval_size', type=int, default=200)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--lam_sympy_dec', type=float, default=1.0,
                        help='SymPy decoder loss weight (default: 1.0)')
    parser.add_argument('--lam_answer_head', type=float, default=1.0,
                        help='Answer head loss weight (default: 1.0)')
    parser.add_argument('--lam_contrastive', type=float, default=0.05,
                        help='Contrastive loss weight (default: 0.05)')
    parser.add_argument('--lam_scale_reg', type=float, default=0.1,
                        help='Scale regularization weight (default: 0.1)')
    parser.add_argument('--num_atoms', type=int, default=64)
    parser.add_argument('--atom_rank', type=int, default=6)

    args = parser.parse_args()
    train(args)
