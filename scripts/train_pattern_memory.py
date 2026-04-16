"""
Training script with Pattern Memory integration.

This script wraps the AtomLoRA training with SQLite-backed pattern memory,
allowing the model to accumulate verified mathematical reasoning patterns
across training epochs.

Pattern memory is optional (disabled by default) and provides:
- Long-term memory of successful reasoning patterns across training
- Deduplication via canonicalization (e.g., "48/2" and "96/2" become same pattern)
- Success/failure tracking per pattern
- Periodic pruning of failed or stale patterns
- JSON export for analysis

Usage:
    # Without pattern memory (standard training):
    python scripts/train_pattern_memory.py --level L4

    # With pattern memory enabled:
    python scripts/train_pattern_memory.py --level L4 --use_pattern_memory

    # With custom pattern DB path:
    python scripts/train_pattern_memory.py \\
        --level GSM8K \\
        --use_pattern_memory \\
        --pattern_db patterns_gsm8k.db \\
        --checkpoint checkpoints/atom_lora_L4_best.pt \\
        --epochs 10
"""

import argparse
import os
import re
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, List, Dict, Any

sys.path.insert(0, '/Users/bryceroche/Desktop/mycelium')
sys.path.insert(0, '/home/ubuntu/mycelium')

from scripts.atom_lora import (
    AtomLoRAModel,
    AtomAdditiveLoRAManager,
    AnswerHead,
    answer_head_loss,
    warm_start_atom_from_checkpoint,
)
from scripts.train_stepping_stones import extract_answer
from src.contrastive_page_loss import per_page_contrastive_loss
from src.pattern_memory import PatternMemory
from src.pattern_classifier import classify_pattern


# ---------------------------------------------------------------------------
# Pattern Memory Maintenance (from handoff)
# ---------------------------------------------------------------------------

def end_of_epoch_maintenance(pattern_memory: PatternMemory, epoch: int,
                             export_dir: str = "checkpoints"):
    """Run pattern memory maintenance at the end of each training epoch.

    1. Prune bad and stale patterns
    2. Print stats
    3. Export for analysis (every 5 epochs)
    """
    pattern_memory.prune(
        min_uses=10,
        max_failure_rate=0.7,
        stale_epochs=20,
        current_epoch=epoch,
    )

    pattern_memory.stats()

    if epoch % 5 == 0:
        export_path = os.path.join(export_dir, f"pattern_memory_epoch_{epoch}.json")
        pattern_memory.export_json(export_path)


# ---------------------------------------------------------------------------
# Gradient scaling (Fix 1 from v22.3)
# ---------------------------------------------------------------------------

def scale_gradient(tensor, scale):
    """Scale gradient during backward without affecting forward."""
    return tensor + (scale - 1.0) * tensor.detach()


# ---------------------------------------------------------------------------
# Dataset helpers (reused from train_atom_lora.py)
# ---------------------------------------------------------------------------

def make_dataset(level, num_samples, seed):
    """Build the right dataset for *level* with a given seed."""
    if level == 'L4.5':
        from scripts.datasets_L45_L47 import L45TwoStepWordDataset
        return L45TwoStepWordDataset(num_samples=num_samples, seed=seed)
    elif level == 'L4.7':
        from scripts.datasets_L45_L47 import L47ThreeStepWordDataset
        return L47ThreeStepWordDataset(num_samples=num_samples, seed=seed)
    elif level == 'L4.9':
        from scripts.datasets_L49_gsm8k import L49GSM8KEasyDataset
        return L49GSM8KEasyDataset()
    elif level == 'L5' or level == 'GSM8K':
        from scripts.datasets_L49_gsm8k import GSM8KAugmentedDataset
        return GSM8KAugmentedDataset(epoch_seed=seed)
    elif level == 'L4':
        from scripts.train_dual_lora_L4 import L4TwoStepWordDataset
        return L4TwoStepWordDataset(num_samples=num_samples, seed=seed)
    elif level == 'L3':
        from scripts.train_stepping_stones_L3 import L3NamedQtyDataset
        return L3NamedQtyDataset(num_samples=num_samples, seed=seed)
    else:
        raise ValueError(f"Unknown level: {level}")


def make_eval_dataset(level, num_samples=200):
    """Build a fixed eval dataset for *level*."""
    if level == 'L4.5':
        from scripts.datasets_L45_L47 import L45TwoStepWordDataset
        return L45TwoStepWordDataset(num_samples=num_samples, seed=99999)
    elif level == 'L4.7':
        from scripts.datasets_L45_L47 import L47ThreeStepWordDataset
        return L47ThreeStepWordDataset(num_samples=num_samples, seed=99999)
    elif level == 'L4.9':
        from scripts.datasets_L49_gsm8k import L49GSM8KEasyDataset
        return L49GSM8KEasyDataset(split='test', max_samples=num_samples)
    elif level == 'L5' or level == 'GSM8K':
        from scripts.train_dual_lora_gsm8k import GSM8KDataset
        return GSM8KDataset(split='test', max_samples=num_samples)
    elif level == 'L4':
        from scripts.train_dual_lora_L4 import L4TwoStepWordDataset
        return L4TwoStepWordDataset(num_samples=num_samples, seed=99999)
    elif level == 'L3':
        from scripts.train_stepping_stones_L3 import L3NamedQtyDataset
        return L3NamedQtyDataset(num_samples=num_samples, seed=123)
    else:
        raise ValueError(f"Unknown level: {level}")


def is_procedural(level):
    return level in ('L3', 'L4', 'L4.5', 'L4.7')


def is_gsm8k(level):
    return level in ('L4.9', 'L5', 'GSM8K')


# ---------------------------------------------------------------------------
# Default warm-start paths
# ---------------------------------------------------------------------------

DEFAULT_WARM = {
    'L3':   None,
    'L4':   'checkpoints/atom_lora_L3_best.pt',
    'L4.5': 'checkpoints/atom_lora_L4_best.pt',
    'L4.7': 'checkpoints/atom_lora_L45_best.pt',
    'L4.9': 'checkpoints/atom_lora_L47_best.pt',
    'L5':   'checkpoints/atom_lora_L49_best.pt',
    'GSM8K': 'checkpoints/atom_lora_L49_best.pt',
}


# ---------------------------------------------------------------------------
# Forward with AtomLoRA + pattern memory hooks
# ---------------------------------------------------------------------------

def forward_train_with_patterns(
    model: AtomLoRAModel,
    answer_head: AnswerHead,
    problems: List[str],
    answers: List[str],
    finals_t: torch.Tensor,
    pattern_memory: Optional[PatternMemory] = None,
    epoch: int = 0,
    num_passes: int = 5,
    max_length: int = 192,
    max_answer_length: int = 256,
):
    """Forward pass with atom LoRA, with optional pattern memory tracking.

    Returns:
        (ans_loss, c_loss, conf_loss, ah_loss, scale_reg_loss,
         page_cos_mean, active_atoms_mean, scale_std, cross_pass_cos,
         confidence_mean, state_pages, problem_metadata)

        problem_metadata: List of dicts with keys:
            - problem_text: str
            - first_page: Tensor (detached)
            - used_pattern_id: Optional[int]
            - pattern_hint: Optional[str]
    """
    device = model.transformer.device

    # Tokenize problems
    inputs = model.tokenizer(
        problems, return_tensors='pt', padding=True,
        truncation=True, max_length=max_length,
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Tokenize answers
    answer_texts = [f" {a}" for a in answers]
    answer_inputs = model.tokenizer(
        answer_texts, return_tensors='pt', padding=True,
        truncation=True, max_length=max_answer_length,
        add_special_tokens=False,
    )
    answer_ids = answer_inputs['input_ids'].to(device)

    batch_size = input_ids.size(0)
    embed_layer = model.transformer.model.embed_tokens
    problem_embeds = embed_layer(input_ids)

    state_pages = []
    atom_scales_history = []
    mid_states_history = []
    pre_tanh_history = []

    # Pattern memory: track which patterns were used per problem
    problem_metadata = [{
        'problem_text': problems[i],
        'first_page': None,
        'used_pattern_id': None,
        'pattern_hint': None,
    } for i in range(batch_size)]

    for pass_num in range(num_passes):
        if pass_num == 0:
            # First pass: no LoRA (no pages yet)
            outputs = model.transformer(
                inputs_embeds=problem_embeds, attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = list(outputs.hidden_states[1:])
            atom_scales = torch.zeros(
                batch_size, model.num_atoms, device=device,
            )
            pre_tanh = torch.zeros_like(atom_scales)
        else:
            # Atom LoRA from pages (with pre-tanh for regularization)
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

        # Perceiver outputs (page, strategy)
        page_delta, _strategy, current_mid_states = model.compressor(
            hidden_states, pass_num,
            prev_mid_states=mid_states_history if mid_states_history else None,
        )
        mid_states_history.append(current_mid_states)
        page = F.normalize(page_delta, dim=-1) * model.page_radius

        # Add Fourier structural identity (after normalization)
        page = model.fourier_page.apply(page, pass_num)

        # Page noise during training
        if model.training:
            page = page + torch.randn_like(page) * 0.05

        # Gradient scaling per cycle (capped at 4x)
        grad_scale = min(float(num_passes - pass_num), 4.0)
        page = scale_gradient(page, grad_scale)

        state_pages.append(page)

        # === PATTERN MEMORY: Query after pass 1 ===
        if pass_num == 0 and pattern_memory is not None:
            # Store first page embedding for later pattern storage
            for i in range(batch_size):
                problem_metadata[i]['first_page'] = page[i].detach().clone()

                # Query pattern memory for hints
                matches = pattern_memory.query(page[i], top_k=3)
                if matches and matches[0]['score'] > 0.5:
                    best = matches[0]
                    problem_metadata[i]['used_pattern_id'] = best['pattern_id']
                    problem_metadata[i]['pattern_hint'] = (
                        f"Suggested: [{best['type']}] {best['template'][:50]} "
                        f"({best['success_rate']:.0%})"
                    )

        # Atom dropout during training
        if model.training and pass_num > 0:
            atom_mask = (torch.rand_like(atom_scales) > 0.1).float()
            atom_scales = atom_scales * atom_mask

        atom_scales_history.append(atom_scales)
        pre_tanh_history.append(pre_tanh)

    # ------- Answer loss (teacher-forced with atom LoRA) -------
    final_atom_scales, final_pre_tanh = model.hypernet(
        state_pages, pass_num=num_passes, return_pre_tanh=True,
    )
    pre_tanh_history.append(final_pre_tanh)
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
    ans_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        answer_ids.reshape(-1),
        ignore_index=model.tokenizer.pad_token_id,
    )

    # ------- Confidence loss -------
    confidence = model.confidence_head(state_pages)
    conf_target = torch.ones_like(confidence)
    conf_loss = F.binary_cross_entropy(confidence, conf_target)

    # ------- Contrastive loss -------
    c_loss = per_page_contrastive_loss(
        state_pages, finals_t, cross_target=0.7, within_target=0.3,
    )

    # ------- Answer head loss -------
    last_page = state_pages[-1].float()
    ah_loss = answer_head_loss(answer_head, last_page, finals_t)

    # ------- Atom diagnostics -------
    with torch.no_grad():
        normed = F.normalize(last_page, dim=-1)
        sim = normed @ normed.T
        B = sim.size(0)
        off_diag = sim - torch.eye(B, device=sim.device)
        page_cos_mean = off_diag.sum() / max(B * (B - 1), 1)

        all_scales = torch.stack(atom_scales_history, dim=0)
        active_per_pass = (all_scales.abs() > 0.1).float().sum(dim=-1)
        active_atoms_mean = active_per_pass.mean()

        scale_std = all_scales.std(dim=-1).mean()

        if len(atom_scales_history) >= 2:
            s0 = F.normalize(atom_scales_history[0], dim=-1)
            sN = F.normalize(atom_scales_history[-1], dim=-1)
            cross_pass_cos = (s0 * sN).sum(dim=-1).mean()
        else:
            cross_pass_cos = torch.tensor(0.0, device=device)

    # ------- Pre-tanh regularization -------
    all_pre_tanh = torch.cat(pre_tanh_history, dim=0)
    scale_reg_loss = (all_pre_tanh ** 2).mean()

    return (ans_loss, c_loss, conf_loss, ah_loss, scale_reg_loss,
            page_cos_mean, active_atoms_mean, scale_std, cross_pass_cos,
            confidence.mean(), state_pages, problem_metadata)


# ---------------------------------------------------------------------------
# After-solve: Update pattern memory
# ---------------------------------------------------------------------------

def after_solve_update_patterns(
    pattern_memory: PatternMemory,
    problem_metadata: List[Dict[str, Any]],
    predictions: List[Any],
    gold_answers: List[Any],
    sympy_steps: Optional[List[List[str]]] = None,
    epoch: int = 0,
):
    """Update pattern memory after solving a batch of problems.

    - Record outcome for any patterns that were used
    - Store new patterns for successful solves

    Args:
        pattern_memory: PatternMemory instance
        problem_metadata: List of dicts from forward_train_with_patterns
        predictions: Predicted answers (from generation or answer head)
        gold_answers: Ground truth answers
        sympy_steps: Optional list of SymPy step lists per problem
        epoch: Current training epoch
    """
    if pattern_memory is None:
        return

    for i, meta in enumerate(problem_metadata):
        # Determine if this problem was solved correctly
        pred = predictions[i] if i < len(predictions) else None
        gold = gold_answers[i] if i < len(gold_answers) else None

        was_correct = False
        if pred is not None and gold is not None:
            try:
                pred_f = float(pred)
                gold_f = float(gold)
                # Integer comparison for whole numbers
                if gold_f == int(gold_f):
                    was_correct = (pred_f == gold_f)
                else:
                    # Float comparison with tolerance
                    was_correct = abs(pred_f - gold_f) < 0.01 * abs(gold_f)
            except (ValueError, TypeError):
                was_correct = (str(pred) == str(gold))

        # 1. Record outcome for used pattern
        if meta['used_pattern_id'] is not None:
            pattern_memory.record_outcome(
                meta['used_pattern_id'],
                success=was_correct,
                epoch=epoch
            )

        # 2. Store new pattern if correct and we have SymPy steps
        if was_correct and sympy_steps and i < len(sympy_steps) and sympy_steps[i]:
            steps = sympy_steps[i]
            pattern_type = classify_pattern(steps)
            template = "; ".join(steps)

            pattern_memory.store(
                page_embedding=meta['first_page'],
                sympy_template=template,
                pattern_type=pattern_type,
                example_problem=meta['problem_text'][:200],
                epoch=epoch,
            )


# ---------------------------------------------------------------------------
# Evaluate (generation-based)
# ---------------------------------------------------------------------------

def evaluate(model, answer_head, eval_dataset, device, num_passes=5,
             gsm8k_mode=False):
    """Evaluate via generation."""
    model.eval()
    answer_head.eval()
    gen_correct = 0
    head_correct = 0
    total = 0
    eval_batch = 4 if gsm8k_mode else 16
    max_length = 192 if gsm8k_mode else 128
    max_new_tokens = 80 if gsm8k_mode else 50

    with torch.no_grad():
        for i in range(0, len(eval_dataset), eval_batch):
            batch_samples = [
                eval_dataset[j]
                for j in range(i, min(i + eval_batch, len(eval_dataset)))
            ]
            problems = [s['problem'] for s in batch_samples]
            gold_answers = [s['final'] for s in batch_samples]

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

            # Generation-based eval
            final_atom_scales = model.hypernet(
                state_pages, pass_num=len(state_pages),
            )
            manager = AtomAdditiveLoRAManager(model.transformer)
            manager.apply(model.atoms, final_atom_scales)
            try:
                generated = model.transformer.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=model.tokenizer.pad_token_id,
                )
            finally:
                manager.remove()

            # Answer head eval
            last_page = state_pages[-1].float()
            head_preds = answer_head.decode(last_page)

            for j in range(batch_size):
                gold = gold_answers[j]

                # Generation accuracy
                prompt_len = input_ids[j].size(0)
                gen_ids = generated[j][prompt_len:]
                gen_text = model.tokenizer.decode(
                    gen_ids, skip_special_tokens=True,
                ).strip()
                pred = extract_answer(gen_text)

                if pred is not None and gold is not None:
                    if gsm8k_mode:
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
                    else:
                        if pred == gold:
                            gen_correct += 1

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

    gen_acc = 100.0 * gen_correct / total if total > 0 else 0.0
    head_acc = 100.0 * head_correct / total if total > 0 else 0.0
    return gen_acc, head_acc


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(batch):
    return {
        'problem': [s['problem'] for s in batch],
        'answer': [s['answer'] for s in batch],
        'final': [s['final'] for s in batch],
    }


# ---------------------------------------------------------------------------
# Warm-start logic
# ---------------------------------------------------------------------------

def try_warm_start(model, answer_head, warm_path):
    """Try atom checkpoint first; fall back to perceiver-only warm start."""
    ckpt = torch.load(warm_path, map_location='cpu', weights_only=True)

    if 'atoms' in ckpt:
        print(f"  Loading atom checkpoint from {warm_path}")

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

    else:
        print(f"  Warm-starting perceiver only from {warm_path}")
        warm_start_atom_from_checkpoint(model, ckpt)
        print(f"  atoms: fresh init (no atom equivalent in checkpoint)")
        print(f"  hypernet: fresh init (new architecture)")
        print(f"  answer_head: fresh init")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    level = args.level
    warm_path = args.checkpoint or DEFAULT_WARM.get(level)

    print("=" * 60)
    print(f"AtomLoRA Training with Pattern Memory -- Level {level}")
    print("=" * 60)
    print(f"  use_pattern_memory = {args.use_pattern_memory}")
    if args.use_pattern_memory:
        print(f"  pattern_db         = {args.pattern_db}")
    print(f"  num_passes         = {args.num_passes}")
    print(f"  batch_size         = {args.batch_size}")
    print(f"  epochs             = {args.epochs}")
    print(f"  lr                 = {args.lr}")
    print(f"  patience           = {args.patience}")
    print(f"  lam_contrastive    = {args.lam}")
    print(f"  lam_conf           = {args.lam_conf}")
    print(f"  lam_answer         = {args.lam_answer}")
    print(f"  num_train          = {args.num_train}")
    print(f"  checkpoint         = {warm_path}")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = AtomLoRAModel(
        num_atoms=args.num_atoms,
        atom_rank=args.atom_rank,
    )
    model.compressor = model.compressor.to(device=device, dtype=torch.bfloat16)
    model.hypernet = model.hypernet.to(device=device, dtype=torch.bfloat16)
    model.atoms = model.atoms.to(device=device, dtype=torch.bfloat16)
    model.confidence_head = model.confidence_head.to(device)
    model.residual_gate = model.residual_gate.to(device=device, dtype=torch.bfloat16)
    model.probe_head = model.probe_head.to(device)

    # Answer head
    answer_head = AnswerHead(page_size=model.page_size).to(device)

    # Warm start if available
    if warm_path and os.path.exists(warm_path):
        print(f"\nWarm-starting from {warm_path}")
        try_warm_start(model, answer_head, warm_path)

    # Initialize pattern memory (optional)
    pattern_memory = None
    if args.use_pattern_memory:
        pattern_memory = PatternMemory(db_path=args.pattern_db)
        print(f"\nPattern memory initialized: {args.pattern_db}")
        initial_stats = pattern_memory.stats()
        if initial_stats['total'] > 0:
            print(f"  Loaded {initial_stats['total']} existing patterns")

    # Eval dataset (fixed)
    eval_dataset = make_eval_dataset(level, num_samples=args.eval_size)
    print(f"\nEval dataset: {len(eval_dataset)} problems")

    # Sequence length settings
    if is_gsm8k(level):
        max_length = 192
        max_answer_length = 256
    else:
        max_length = 128
        max_answer_length = 128

    # Optimizer
    atom_A_params = list(model.atoms.A.values()) if hasattr(model.atoms.A, 'values') else list(model.atoms.A.parameters())
    atom_B_params = list(model.atoms.B.values()) if hasattr(model.atoms.B, 'values') else list(model.atoms.B.parameters())

    optimizer = torch.optim.AdamW([
        {'params': list(model.compressor.parameters()), 'lr': args.lr, 'weight_decay': 0.01},
        {'params': atom_A_params, 'lr': args.lr, 'weight_decay': 0.05},
        {'params': atom_B_params, 'lr': args.lr, 'weight_decay': 0.05},
        {'params': list(model.hypernet.parameters()), 'lr': args.lr * 10, 'weight_decay': 0.1},
        {'params': list(model.confidence_head.parameters()), 'lr': args.lr * 10, 'weight_decay': 0.01},
        {'params': list(answer_head.parameters()), 'lr': args.lr * 10, 'weight_decay': 0.01},
    ])

    trainable = (
        list(model.compressor.parameters())
        + list(model.atoms.parameters())
        + list(model.hypernet.parameters())
        + list(model.confidence_head.parameters())
        + list(answer_head.parameters())
    )
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    # Baseline evaluation
    base_gen, base_head = evaluate(
        model, answer_head, eval_dataset, device,
        num_passes=args.num_passes, gsm8k_mode=is_gsm8k(level),
    )
    print(f"\nBaseline: gen={base_gen:.1f}% head={base_head:.1f}%\n")

    # Checkpoint naming
    ckpt_name = f"checkpoints/pattern_memory_{level.replace('.', '')}_best.pt"
    best = 0.0
    patience_counter = 0

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        answer_head.train()
        t0 = time.time()

        # Fresh data per epoch for procedural levels
        epoch_seed = epoch * 1000 + 42
        if is_procedural(level):
            train_dataset = make_dataset(level, args.num_train, seed=epoch_seed)
            print(f"  [epoch {epoch+1}] Generated {len(train_dataset)} fresh problems")
        elif level in ('L5', 'GSM8K'):
            train_dataset = make_dataset(level, args.num_train, seed=epoch_seed)
            print(f"  [epoch {epoch+1}] Augmented GSM8K (seed={epoch_seed})")
        else:
            if epoch == 0:
                train_dataset = make_dataset(level, args.num_train, seed=epoch_seed)
                print(f"  [epoch {epoch+1}] Loaded {len(train_dataset)} problems")

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_fn,
        )

        # Epoch metrics
        ep_ans = ep_ctr = ep_conf = ep_head = ep_scale_reg = 0.0
        ep_active = ep_std = ep_xpass = ep_cos = 0.0
        patterns_stored = patterns_used = 0
        nb = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            problems = batch['problem']
            answers = batch['answer']
            finals_raw = batch['final']

            finals_list = []
            for f in finals_raw:
                try:
                    finals_list.append(int(f))
                except (ValueError, TypeError):
                    finals_list.append(0)
            finals_t = torch.tensor(finals_list, dtype=torch.long, device=device)

            optimizer.zero_grad()

            # Forward pass with pattern memory hooks
            (ans_loss, c_loss, conf_loss, ah_loss, scale_reg,
             page_cos, active_atoms, s_std, xpass_cos,
             conf_mean, state_pages, problem_metadata) = forward_train_with_patterns(
                model, answer_head, problems, answers, finals_t,
                pattern_memory=pattern_memory,
                epoch=epoch,
                num_passes=args.num_passes,
                max_length=max_length,
                max_answer_length=max_answer_length,
            )

            total_loss = (ans_loss
                          + args.lam * c_loss
                          + args.lam_conf * conf_loss
                          + args.lam_answer * ah_loss
                          + 0.1 * scale_reg)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            # Update pattern memory after batch
            if pattern_memory is not None:
                # Get predictions from answer head for pattern outcome tracking
                with torch.no_grad():
                    last_page = state_pages[-1].float()
                    pred_answers = answer_head.decode(last_page).tolist()

                after_solve_update_patterns(
                    pattern_memory,
                    problem_metadata,
                    pred_answers,
                    finals_list,
                    sympy_steps=None,  # No SymPy steps in this training loop
                    epoch=epoch,
                )

                # Track pattern usage
                patterns_used += sum(1 for m in problem_metadata if m['used_pattern_id'] is not None)

            # Accumulate metrics
            ep_ans += ans_loss.item()
            ep_ctr += c_loss.item()
            ep_conf += conf_loss.item()
            ep_head += ah_loss.item()
            ep_scale_reg += scale_reg.item()
            ep_cos += page_cos.item()
            ep_active += active_atoms.item()
            ep_std += s_std.item()
            ep_xpass += xpass_cos.item()
            nb += 1

        elapsed = time.time() - t0

        # End-of-epoch pattern memory maintenance
        if pattern_memory is not None:
            print(f"\n  Pattern memory maintenance (epoch {epoch+1}):")
            end_of_epoch_maintenance(pattern_memory, epoch, export_dir="checkpoints")

        # Evaluation
        gen_acc, head_acc = evaluate(
            model, answer_head, eval_dataset, device,
            num_passes=args.num_passes, gsm8k_mode=is_gsm8k(level),
        )

        # Save best checkpoint
        if gen_acc > best:
            best = gen_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'compressor': model.compressor.state_dict(),
                'atoms': model.atoms.state_dict(),
                'hypernet': model.hypernet.state_dict(),
                'confidence_head': model.confidence_head.state_dict(),
                'answer_head': answer_head.state_dict(),
                'residual_gate': model.residual_gate.state_dict(),
                'accuracy': gen_acc,
                'head_accuracy': head_acc,
                'level': level,
                'num_atoms': args.num_atoms,
                'atom_rank': args.atom_rank,
            }, ckpt_name)
            print(f"  -> Saved checkpoint {ckpt_name}")
        else:
            patience_counter += 1

        # Print epoch summary
        print(
            f"Epoch {epoch+1}: "
            f"ans={ep_ans/nb:.4f} contr={ep_ctr/nb:.2f} "
            f"conf={ep_conf/nb:.2f} head={ep_head/nb:.2f} "
            f"page_cos={ep_cos/nb:.2f} "
            f"active={ep_active/nb:.1f}/{args.num_atoms} | "
            f"Acc={gen_acc:.1f}% head={head_acc:.1f}% "
            f"best={best:.1f}% [{elapsed:.0f}s]"
        )
        if pattern_memory is not None:
            print(f"  patterns: {patterns_used} queries, {pattern_memory.count()} total stored")

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} epochs")
            break

    # Final summary
    print(f"\n{'='*60}")
    print(f"Level {level} final: gen={best:.1f}% (baseline gen={base_gen:.1f}%)")
    print(f"Checkpoint: {ckpt_name}")
    if pattern_memory is not None:
        print(f"Pattern memory: {args.pattern_db} ({pattern_memory.count()} patterns)")
        pattern_memory.close()
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='AtomLoRA training with pattern memory integration',
    )

    # Level and checkpoint
    parser.add_argument(
        '--level', type=str, default='L4',
        choices=['L3', 'L4', 'L4.5', 'L4.7', 'L4.9', 'L5', 'GSM8K'],
        help='Curriculum level to train',
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Checkpoint to warm-start from (default per level)',
    )

    # Pattern memory
    parser.add_argument(
        '--use_pattern_memory', action='store_true',
        help='Enable pattern memory (SQLite-backed long-term memory)',
    )
    parser.add_argument(
        '--pattern_db', type=str, default='pattern_memory.db',
        help='Path to SQLite pattern memory database',
    )

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_passes', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--lam', type=float, default=0.05,
                        help='Contrastive loss weight')
    parser.add_argument('--lam_conf', type=float, default=0.1,
                        help='Confidence loss weight')
    parser.add_argument('--lam_answer', type=float, default=0.3,
                        help='Answer head loss weight')
    parser.add_argument('--num_train', type=int, default=20000,
                        help='Number of training problems per epoch')
    parser.add_argument('--eval_size', type=int, default=50,
                        help='Number of eval problems')

    # Model architecture
    parser.add_argument('--num_atoms', type=int, default=64)
    parser.add_argument('--atom_rank', type=int, default=6)

    args = parser.parse_args()

    # Default batch size based on level
    if args.batch_size is None:
        args.batch_size = 32 if is_procedural(args.level) else 16

    train(args)
