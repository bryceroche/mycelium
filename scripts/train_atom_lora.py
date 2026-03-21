"""
AtomLoRA Training (v24).

64 anonymous rank-6 LoRA atoms, independently scaled by a 10M-param hypernetwork.
No named modes, no softmax blend, no entropy regularization. The model discovers
its own cognitive decomposition through training.

Replaces train_quad_lora.py. Key differences:
  - Tanh atom scales (independent, not competing) instead of softmax blend
  - No strategy side channel (perceiver outputs page only, strategy discarded)
  - Atom sparsity logging (active atoms, scale_std, cross_pass_cos)
  - Simpler loss (no entropy regularization term)

Usage:
  python scripts/train_atom_lora.py --level L3
  python scripts/train_atom_lora.py --level L5 --warm checkpoints/atom_lora_L49_best.pt
  python scripts/train_atom_lora.py --level L4 --warm checkpoints/atom_lora_L3_best.pt
"""
import argparse
import re
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

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
from src.page_cache import ReplayBuffer, GraduationTracker


# ---------------------------------------------------------------------------
# Gradient scaling (Fix 1 from v22.3)
# ---------------------------------------------------------------------------

def scale_gradient(tensor, scale):
    """Scale gradient during backward without affecting forward."""
    return tensor + (scale - 1.0) * tensor.detach()


# ---------------------------------------------------------------------------
# Dataset helpers (reused from train_three_fixes.py)
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
    elif level == 'L5':
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
    elif level == 'L5':
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
    return level in ('L4.9', 'L5')


# ---------------------------------------------------------------------------
# Default warm-start paths
# ---------------------------------------------------------------------------

DEFAULT_WARM = {
    'L3':   None,                                      # atoms train from scratch
    'L4':   'checkpoints/atom_lora_L3_best.pt',
    'L4.5': 'checkpoints/atom_lora_L4_best.pt',
    'L4.7': 'checkpoints/atom_lora_L45_best.pt',
    'L4.9': 'checkpoints/atom_lora_L47_best.pt',
    'L5':   'checkpoints/atom_lora_L49_best.pt',
}


# ---------------------------------------------------------------------------
# Forward with AtomLoRA + gradient scaling + answer head
# ---------------------------------------------------------------------------

def forward_train(model, answer_head, problems, answers, finals_t,
                  num_passes=5, max_length=192, max_answer_length=256):
    """Forward pass with atom LoRA, gradient scaling, and answer head.

    Returns:
        (ans_loss, c_loss, conf_loss, ah_loss,
         page_cos_mean, active_atoms_mean, scale_std, cross_pass_cos,
         confidence_mean)
    """
    device = model.transformer.device

    inputs = model.tokenizer(
        problems, return_tensors='pt', padding=True,
        truncation=True, max_length=max_length,
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

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
    atom_scales_history = []  # list of (B, num_atoms) tensors
    mid_states_history = []   # list of (B, num_queries, d_perceiver) tensors

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
        else:
            # Atom LoRA from pages
            atom_scales = model.hypernet(
                state_pages, pass_num=pass_num,
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

        # Perceiver outputs (page, strategy) — we discard strategy
        # Pass mid_states_history for skip connection
        page_delta, _strategy, current_mid_states = model.compressor(
            hidden_states, pass_num,
            prev_mid_states=mid_states_history if mid_states_history else None,
        )
        mid_states_history.append(current_mid_states)
        page = F.normalize(page_delta, dim=-1) * model.page_radius

        # Add Fourier structural identity (after normalization)
        page = model.fourier_page.apply(page, pass_num)

        # Page noise during training: prevents hypernetwork from memorizing exact page values
        if model.training:
            page = page + torch.randn_like(page) * 0.05

        # Gradient scaling per cycle (capped at 4x)
        grad_scale = min(float(num_passes - pass_num), 4.0)
        page = scale_gradient(page, grad_scale)

        state_pages.append(page)

        # Atom dropout during training: randomly zero 10% of atom scales
        if model.training and pass_num > 0:
            atom_mask = (torch.rand_like(atom_scales) > 0.1).float()
            atom_scales = atom_scales * atom_mask

        atom_scales_history.append(atom_scales)

    # ------- Answer loss (teacher-forced with atom LoRA) -------
    final_atom_scales = model.hypernet(
        state_pages, pass_num=num_passes,
    )
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

    # ------- Confidence loss (pages only, no blend history) -------
    confidence = model.confidence_head(state_pages)
    conf_target = torch.ones_like(confidence)
    conf_loss = F.binary_cross_entropy(confidence, conf_target)

    # ------- Contrastive loss -------
    c_loss = per_page_contrastive_loss(
        state_pages, finals_t, cross_target=0.7, within_target=0.3,
    )

    # ------- Answer head loss (digit prediction from last page) -------
    last_page = state_pages[-1].float()
    ah_loss = answer_head_loss(answer_head, last_page, finals_t)

    # ------- Atom diagnostics -------
    with torch.no_grad():
        # Page cosine similarity (off-diagonal mean)
        normed = F.normalize(last_page, dim=-1)
        sim = normed @ normed.T
        B = sim.size(0)
        off_diag = sim - torch.eye(B, device=sim.device)
        page_cos_mean = off_diag.sum() / max(B * (B - 1), 1)

        # Active atoms: mean count of |scale| > 0.1 across all passes
        all_scales = torch.stack(atom_scales_history, dim=0)  # (P, B, A)
        active_per_pass = (all_scales.abs() > 0.1).float().sum(dim=-1)  # (P, B)
        active_atoms_mean = active_per_pass.mean()

        # Scale std: std across atoms (measures differentiation)
        scale_std = all_scales.std(dim=-1).mean()

        # Cross-pass cosine: similarity between pass 1 and last pass scales
        if len(atom_scales_history) >= 2:
            s0 = F.normalize(atom_scales_history[0], dim=-1)
            sN = F.normalize(atom_scales_history[-1], dim=-1)
            cross_pass_cos = (s0 * sN).sum(dim=-1).mean()
        else:
            cross_pass_cos = torch.tensor(0.0, device=device)

    return (ans_loss, c_loss, conf_loss, ah_loss,
            page_cos_mean, active_atoms_mean, scale_std, cross_pass_cos,
            confidence.mean(), state_pages)


# ---------------------------------------------------------------------------
# Forward with cache support (v24.4)
# ---------------------------------------------------------------------------

def forward_train_cached(model, answer_head, problems, answers, finals_t,
                         cached_pages=None, start_pass=0,
                         num_passes=5, max_length=192, max_answer_length=256):
    """Forward pass starting from cached pages.

    If cached_pages is provided and start_pass > 0, skip the first start_pass
    cycles and use the cached pages directly. Only runs passes from start_pass.

    Returns same tuple as forward_train, plus state_pages for caching.
    """
    device = model.transformer.device

    inputs = model.tokenizer(
        problems, return_tensors='pt', padding=True,
        truncation=True, max_length=max_length,
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

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

    # Start from cached pages if provided
    if cached_pages is not None and start_pass > 0:
        state_pages = cached_pages  # already detached, no grad
    else:
        state_pages = []
        start_pass = 0

    atom_scales_history = []
    mid_states_history = []

    for pass_num in range(start_pass, num_passes):
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
        else:
            # Atom LoRA from pages
            atom_scales = model.hypernet(
                state_pages, pass_num=pass_num,
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

        # Perceiver outputs
        page_delta, _strategy, current_mid_states = model.compressor(
            hidden_states, pass_num,
            prev_mid_states=mid_states_history if mid_states_history else None,
        )
        mid_states_history.append(current_mid_states)
        page = F.normalize(page_delta, dim=-1) * model.page_radius

        # Add Fourier structural identity
        page = model.fourier_page.apply(page, pass_num)

        # Page noise during training
        if model.training:
            page = page + torch.randn_like(page) * 0.05

        # Gradient scaling per cycle (capped at 4x)
        grad_scale = min(float(num_passes - pass_num), 4.0)
        page = scale_gradient(page, grad_scale)

        state_pages.append(page)

        # Atom dropout during training
        if model.training and pass_num > 0:
            atom_mask = (torch.rand_like(atom_scales) > 0.1).float()
            atom_scales = atom_scales * atom_mask

        atom_scales_history.append(atom_scales)

    # ------- Answer loss (teacher-forced with atom LoRA) -------
    final_atom_scales = model.hypernet(
        state_pages, pass_num=num_passes,
    )
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

    # ------- Diagnostics -------
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

    return (ans_loss, c_loss, conf_loss, ah_loss,
            page_cos_mean, active_atoms_mean, scale_std, cross_pass_cos,
            confidence.mean(), state_pages)


# ---------------------------------------------------------------------------
# Per-step accuracy measurement (for graduation)
# ---------------------------------------------------------------------------

def measure_per_step_accuracy(model, answer_head, eval_dataset, device,
                              num_passes=5, gsm8k_mode=False):
    """Measure accuracy at each thinking pass.

    Returns: List of per-step accuracies [acc_pass0, ..., acc_passN-1]
    """
    model.eval()
    answer_head.eval()
    per_step_correct = [0] * num_passes
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

                # Evaluate at this pass
                final_atom_scales = model.hypernet(
                    state_pages, pass_num=pass_num + 1,
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

                for j in range(batch_size):
                    gold = gold_answers[j]
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
                                        per_step_correct[pass_num] += 1
                                elif abs(pred_f - gold_f) < 0.01:
                                    per_step_correct[pass_num] += 1
                            except (ValueError, TypeError, OverflowError):
                                pass
                        else:
                            if pred == gold:
                                per_step_correct[pass_num] += 1

            total += batch_size

    per_step_acc = [c / total if total > 0 else 0.0 for c in per_step_correct]
    return per_step_acc


# ---------------------------------------------------------------------------
# Evaluate (generation-based + answer head)
# ---------------------------------------------------------------------------

def evaluate(model, answer_head, eval_dataset, device, num_passes=5,
             gsm8k_mode=False):
    """Evaluate via generation AND answer head digit prediction."""
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

            # --- Generation-based eval ---
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

            # --- Answer head eval ---
            last_page = state_pages[-1].float()
            head_preds = answer_head.decode(last_page)  # (B,) tensor of ints

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
# Atom gradient norm logging
# ---------------------------------------------------------------------------

def log_atom_grad_norms(model):
    """Log gradient norms for atom A params, atom B params, and hypernetwork.

    Returns dict with 3 groups instead of 4 named modes.
    """
    norms = {}

    # Atoms A params
    a_norm = 0.0
    a_count = 0
    for name, param in model.atoms.A.items():
        if param.grad is not None:
            a_norm += param.grad.norm().item()
            a_count += 1
    norms['atoms_A'] = a_norm / max(a_count, 1)

    # Atoms B params
    b_norm = 0.0
    b_count = 0
    for name, param in model.atoms.B.items():
        if param.grad is not None:
            b_norm += param.grad.norm().item()
            b_count += 1
    norms['atoms_B'] = b_norm / max(b_count, 1)

    # Hypernetwork params
    h_norm = 0.0
    h_count = 0
    for p in model.hypernet.parameters():
        if p.grad is not None:
            h_norm += p.grad.norm().item()
            h_count += 1
    norms['hypernet'] = h_norm / max(h_count, 1)

    return norms


# ---------------------------------------------------------------------------
# Collate
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
        # Atom checkpoint — load directly
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
        # Non-atom checkpoint — load perceiver only
        print(f"  Warm-starting perceiver only from {warm_path}")
        warm_start_atom_from_checkpoint(model, ckpt)
        print(f"  atoms: fresh init (no atom equivalent in checkpoint)")
        print(f"  hypernet: fresh init (new architecture)")
        print(f"  answer_head: fresh init")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    level = args.level
    warm_path = args.warm or DEFAULT_WARM.get(level)

    print("=" * 60)
    print(f"AtomLoRA Training -- Level {level}")
    print("=" * 60)
    print(f"  num_passes     = {args.num_passes}")
    print(f"  batch_size     = {args.batch_size}")
    print(f"  epochs         = {args.epochs}")
    print(f"  patience       = {args.patience}")
    print(f"  lam            = {args.lam}")
    print(f"  lam_conf       = {args.lam_conf}")
    print(f"  lam_answer     = {args.lam_answer}")
    print(f"  num_atoms      = {args.num_atoms}")
    print(f"  atom_rank      = {args.atom_rank}")
    print(f"  num_train      = {args.num_train}")
    print(f"  warm           = {warm_path}")
    print(f"  procedural     = {is_procedural(level)}")
    print(f"  gsm8k_mode     = {is_gsm8k(level)}")
    print(f"  use_cache      = {args.use_cache}")
    print("=" * 60)

    device = torch.device('cuda')
    model = AtomLoRAModel(
        num_atoms=args.num_atoms,
        atom_rank=args.atom_rank,
    )
    model.compressor = model.compressor.to(device=device, dtype=torch.bfloat16)
    model.hypernet = model.hypernet.to(device=device, dtype=torch.bfloat16)
    model.atoms = model.atoms.to(device=device, dtype=torch.bfloat16)
    model.confidence_head = model.confidence_head.to(device)  # fp32 (small)

    # Answer head — small, kept in float32 for digit classification stability
    answer_head = AnswerHead(page_size=model.page_size).to(device)

    if warm_path:
        print(f"\nWarm-starting from {warm_path}")
        try_warm_start(model, answer_head, warm_path)

    # Fixed eval dataset
    eval_dataset = make_eval_dataset(level, num_samples=args.eval_size)
    print(f"\nEval dataset: {len(eval_dataset)} problems")

    # Sequence length settings
    if is_gsm8k(level):
        max_length = 192
        max_answer_length = 256
    else:
        max_length = 128
        max_answer_length = 128

    # ------- Optimizer: atoms A/B + hypernet + compressor + heads -------

    # Collect atom A and B params separately for logging clarity
    atom_A_params = list(model.atoms.A.values()) if hasattr(model.atoms.A, 'values') else list(model.atoms.A.parameters())
    atom_B_params = list(model.atoms.B.values()) if hasattr(model.atoms.B, 'values') else list(model.atoms.B.parameters())

    optimizer = torch.optim.AdamW([
        {'params': list(model.compressor.parameters()), 'lr': 1e-4, 'weight_decay': 0.01},
        {'params': atom_A_params, 'lr': 1e-4, 'weight_decay': 0.05},     # slow: atoms are foundation
        {'params': atom_B_params, 'lr': 1e-4, 'weight_decay': 0.05},     # slow: atoms are foundation
        {'params': list(model.hypernet.parameters()), 'lr': 1e-3, 'weight_decay': 0.1},
        {'params': list(model.confidence_head.parameters()), 'lr': 1e-3, 'weight_decay': 0.01},
        {'params': list(answer_head.parameters()), 'lr': 1e-3, 'weight_decay': 0.01},
    ])

    trainable = (
        list(model.compressor.parameters())
        + list(model.atoms.parameters())
        + list(model.hypernet.parameters())
        + list(model.confidence_head.parameters())
        + list(answer_head.parameters())
    )
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")
    print(f"  atoms: {sum(p.numel() for p in model.atoms.parameters()):,} "
          f"({args.num_atoms} atoms, rank {args.atom_rank})")
    print(f"  hypernet: {sum(p.numel() for p in model.hypernet.parameters()):,}")
    print(f"  compressor: {sum(p.numel() for p in model.compressor.parameters()):,}")
    print(f"  answer_head: {sum(p.numel() for p in answer_head.parameters()):,}")
    print(f"  confidence_head: {sum(p.numel() for p in model.confidence_head.parameters()):,}")
    print(f"GPU mem after init: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Baseline
    base_gen, base_head = evaluate(
        model, answer_head, eval_dataset, device,
        num_passes=args.num_passes, gsm8k_mode=is_gsm8k(level),
    )
    print(f"\nBaseline: gen={base_gen:.1f}% head={base_head:.1f}%\n")

    ckpt_name = f"checkpoints/atom_lora_{level.replace('.', '')}_best.pt"
    best = 0.0
    best_head = 0.0
    patience_counter = 0

    # Page cache (v24.4)
    cache = None
    grad_tracker = None
    if args.use_cache:
        cache = ReplayBuffer(max_epochs_stored=5, device=str(device))
        grad_tracker = GraduationTracker(
            max_passes=args.num_passes,
            threshold=0.90,
            stable_epochs=2,
        )
        print("Page cache ENABLED")

    for epoch in range(args.epochs):
        model.train()
        answer_head.train()
        t0 = time.time()

        # Fresh data per epoch (Fix 2)
        epoch_seed = epoch * 1000 + 42
        if is_procedural(level):
            train_dataset = make_dataset(level, args.num_train, seed=epoch_seed)
            print(f"  [epoch {epoch+1}] Generated {len(train_dataset)} fresh problems (seed={epoch_seed})")
        elif level == 'L5':
            train_dataset = make_dataset(level, args.num_train, seed=epoch_seed)
            print(f"  [epoch {epoch+1}] Augmented GSM8K (seed={epoch_seed})")
        else:
            if epoch == 0:
                train_dataset = make_dataset(level, args.num_train, seed=epoch_seed)
                print(f"  [epoch {epoch+1}] Loaded {len(train_dataset)} problems (fixed)")

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_fn,
        )

        ep_ans = ep_ctr = ep_conf = ep_head = ep_cos = 0.0
        ep_active = ep_std = ep_xpass = 0.0
        cache_hits = cache_full = 0
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

            # Cache logic (v24.4)
            cached_pages = None
            start_pass = 0
            if cache is not None and cache.graduated_up_to > 0 and epoch >= 2:
                # Decide cache strategy probabilistically
                import random
                r = random.random()
                graduated = cache.graduated_up_to

                if r < 0.7:
                    # 70%: start from graduation frontier
                    problem_hash = str(hash(problems[0]))  # batch key
                    loaded = cache.load_diverse(problem_hash, graduated, epoch)
                    if loaded is not None:
                        cached_pages = loaded
                        start_pass = graduated
                        cache_hits += 1
                elif r < 0.9:
                    # 20%: start from frontier-1
                    start_from = max(0, graduated - 1)
                    if start_from > 0:
                        problem_hash = str(hash(problems[0]))
                        loaded = cache.load_diverse(problem_hash, start_from, epoch)
                        if loaded is not None:
                            cached_pages = loaded
                            start_pass = start_from
                            cache_hits += 1
                # 10%: full run (implicit else)

            if cached_pages is not None:
                (ans_loss, c_loss, conf_loss, ah_loss,
                 page_cos, active_atoms, s_std, xpass_cos,
                 conf_mean, state_pages) = forward_train_cached(
                    model, answer_head, problems, answers, finals_t,
                    cached_pages=cached_pages, start_pass=start_pass,
                    num_passes=args.num_passes,
                    max_length=max_length,
                    max_answer_length=max_answer_length,
                )
            else:
                (ans_loss, c_loss, conf_loss, ah_loss,
                 page_cos, active_atoms, s_std, xpass_cos,
                 conf_mean, state_pages) = forward_train(
                    model, answer_head, problems, answers, finals_t,
                    num_passes=args.num_passes,
                    max_length=max_length,
                    max_answer_length=max_answer_length,
                )
                cache_full += 1

                # Store pages on full runs
                if cache is not None:
                    problem_hash = str(hash(problems[0]))
                    cache.store(problem_hash, state_pages, epoch)

            total_loss = (ans_loss
                          + args.lam * c_loss
                          + args.lam_conf * conf_loss
                          + args.lam_answer * ah_loss)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            ep_ans += ans_loss.item()
            ep_ctr += c_loss.item()
            ep_conf += conf_loss.item()
            ep_head += ah_loss.item()
            ep_cos += page_cos.item()
            ep_active += active_atoms.item()
            ep_std += s_std.item()
            ep_xpass += xpass_cos.item()
            nb += 1

        elapsed = time.time() - t0

        # Gradient norms (from last batch)
        grad_norms = log_atom_grad_norms(model)

        # Eval
        gen_acc, head_acc = evaluate(
            model, answer_head, eval_dataset, device,
            num_passes=args.num_passes, gsm8k_mode=is_gsm8k(level),
        )

        # Per-step accuracy and graduation update (v24.4)
        if cache is not None and grad_tracker is not None:
            per_step_acc = measure_per_step_accuracy(
                model, answer_head, eval_dataset, device,
                num_passes=args.num_passes, gsm8k_mode=is_gsm8k(level),
            )
            grad_tracker.record(per_step_acc)
            old_graduated = cache.graduated_up_to
            new_graduated = grad_tracker.get_graduation_level()
            if new_graduated > old_graduated:
                cache.graduated_up_to = new_graduated
                print(f"  GRADUATED: cycles 0-{new_graduated-1} now cached!")
            cache_stats = cache.stats()
            print(f"  per_step_acc: {[f'{a:.1%}' for a in per_step_acc]}")
            print(f"  cache: graduated={new_graduated} hits={cache_hits} full={cache_full} "
                  f"stored={cache_stats['total_entries']}")

        # Track both metrics — early stop only when BOTH plateau
        improved = False
        if gen_acc > best:
            best = gen_acc
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
                'confidence_head': model.confidence_head.state_dict(),
                'answer_head': answer_head.state_dict(),
                'accuracy': gen_acc,
                'head_accuracy': head_acc,
                'level': level,
                'num_atoms': args.num_atoms,
                'atom_rank': args.atom_rank,
            }, ckpt_name)
            print(f"  -> saved checkpoint {ckpt_name} (gen={gen_acc:.1f}% head={head_acc:.1f}%)")
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch+1}: "
            f"ans={ep_ans/nb:.4f} contr={ep_ctr/nb:.2f} "
            f"conf={ep_conf/nb:.2f} head={ep_head/nb:.2f} "
            f"page_cos={ep_cos/nb:.2f} "
            f"active={ep_active/nb:.1f}/{args.num_atoms} "
            f"scale_std={ep_std/nb:.3f} "
            f"xpass_cos={ep_xpass/nb:.2f} | "
            f"Acc={gen_acc:.1f}% head={head_acc:.1f}% "
            f"best={best:.1f}% [{elapsed:.0f}s]"
        )
        # Gradient norms
        gn = grad_norms
        print(
            f"  grad norms: "
            f"atoms_A={gn['atoms_A']:.4f} atoms_B={gn['atoms_B']:.4f} "
            f"hypernet={gn['hypernet']:.4f}"
        )

        if patience_counter >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} epochs")
            break

    print(f"\n{'='*60}")
    print(f"Level {level} final: gen={best:.1f}% head={best_head:.1f}% "
          f"(baseline gen={base_gen:.1f}% head={base_head:.1f}%)")
    print(f"Checkpoint: {ckpt_name}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='AtomLoRA (64-atom) training with answer head',
    )
    p.add_argument(
        '--level', type=str, required=True,
        choices=['L3', 'L4', 'L4.5', 'L4.7', 'L4.9', 'L5'],
        help='Curriculum level to train',
    )
    p.add_argument('--warm', type=str, default=None,
                   help='Checkpoint to warm-start from (default per level)')
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch_size', type=int, default=None,
                   help='Batch size (default: 32 procedural, 16 GSM8K)')
    p.add_argument('--num_passes', type=int, default=5)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--lam', type=float, default=0.05,
                   help='Contrastive loss weight')
    p.add_argument('--lam_conf', type=float, default=0.1,
                   help='Confidence loss weight')
    p.add_argument('--lam_answer', type=float, default=0.3,
                   help='Answer head loss weight')
    p.add_argument('--num_atoms', type=int, default=64,
                   help='Number of LoRA atoms')
    p.add_argument('--atom_rank', type=int, default=6,
                   help='Rank of each LoRA atom')
    p.add_argument('--num_train', type=int, default=20000,
                   help='Number of training problems per epoch (procedural)')
    p.add_argument('--eval_size', type=int, default=50,
                   help='Number of eval problems (200 quick, 500 thorough)')
    p.add_argument('--use_cache', action='store_true',
                   help='Enable page cache for faster training (v24.4)')

    args = p.parse_args()

    if args.batch_size is None:
        args.batch_size = 32 if is_procedural(args.level) else 16

    train(args)
