"""
Per-Cycle Answer Head Training (v24.9) -- tinygrad port.

Ported from scripts/train_per_cycle.py. All PyTorch replaced with tinygrad.

Usage:
  python tinygrad_port/train.py --level L3 --data_dir data/per_cycle --epochs 8 --num_passes 3
  python tinygrad_port/train.py --level gsm8k --data_dir data/per_cycle --epochs 50 --warm_from checkpoints/best.safetensor
"""
import argparse
import math
import os
import sys
import time
import numpy as np

from tinygrad import Tensor, dtypes, Device
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tinygrad_port.losses import (
    answer_head_loss,
    generation_loss,
    per_page_contrastive_loss,
    isotropic_regularizer,
    confidence_entropy_loss,
    per_cycle_target_weight,
)
from tinygrad_port.data import PerCycleDataset, SubsetDataset, batch_iterator


# ---------------------------------------------------------------------------
# Gradient scaling (Fix 1 from v22.3)
# ---------------------------------------------------------------------------

def scale_gradient(tensor, scale):
    """Scale gradient during backward without affecting forward.

    In tinygrad we replicate the straight-through trick:
    forward: tensor, backward: tensor * scale.
    Equivalent to: tensor + (scale - 1) * tensor.detach()
    """
    return tensor + (scale - 1.0) * tensor.detach()


# ---------------------------------------------------------------------------
# Normalize helper (L2 normalize along last axis)
# ---------------------------------------------------------------------------

def l2_normalize(x, dim=-1, eps=1e-8):
    """L2-normalize tensor along given dimension."""
    norm = (x * x).sum(axis=dim, keepdim=True).sqrt() + eps
    return x / norm


# ---------------------------------------------------------------------------
# Forward pass with per-cycle answer head loss
# ---------------------------------------------------------------------------

def forward_train_per_cycle(
    model, answer_head, tokenizer, problems, cycle_targets_t, cycle_mask_t,
    finals_t, cycle_gen_targets=None,
    num_passes=5, max_length=192,
    lam_gen=1.0, lam_ah=0.5,
    final_accuracy=0.0,
):
    """Forward pass with hybrid per-cycle loss: generation + answer head.

    Each cycle:
    1. Thinks (transformer + perceiver -> page)
    2. Generates the intermediate result as short text with LoRA ON
    3. Predicts via answer head (shaping signal)

    Args:
        model:            main model with .think(), .embed(), .generate_logits(), etc.
        answer_head:      AnswerHead module
        tokenizer:        tokenizer (for encoding text)
        problems:         list of str
        cycle_targets_t:  Tensor (B, max_steps) int32
        cycle_mask_t:     Tensor (B, max_steps) float32
        finals_t:         list of int -- final gold answers
        cycle_gen_targets: list of max_steps lists of B strings (optional)
        num_passes:       int
        max_length:       int
        lam_gen:          float
        lam_ah:           float
        final_accuracy:   float in [0, 1]

    Returns:
        dict with all loss components and diagnostics
    """
    # Tokenize problem text
    tokens = tokenizer(problems, padding=True, truncation=True, max_length=max_length)
    input_ids = Tensor(tokens['input_ids'], dtype=dtypes.int32)
    attention_mask = Tensor(tokens['attention_mask'], dtype=dtypes.float32)

    batch_size = input_ids.shape[0]
    prompt_len = input_ids.shape[1]

    # Get cycle_targets as numpy for indexing
    ct_np = cycle_targets_t.numpy()
    cm_np = cycle_mask_t.numpy()
    max_steps = ct_np.shape[1]

    state_pages = []
    raw_pages = []
    messages = []
    prev_predictions = []
    # Track consumed targets (per-sample)
    consumed = cm_np.copy()  # 1.0 = available, 0.0 = consumed
    atom_scales_history = []
    pre_tanh_history = []
    per_cycle_preds = []
    per_cycle_ah_loss = Tensor(0.0)
    per_cycle_gen_loss = Tensor(0.0)
    valid_cycles = 0

    for pass_num in range(num_passes):
        # --- Inject previous answers as text context for cycle 2+ ---
        if pass_num > 0 and len(state_pages) > 0:
            with Tensor.no_grad:
                latest_page = state_pages[-1].float()
                latest_preds = answer_head.decode(latest_page, cycle_num=pass_num - 1)
                prev_predictions.append(latest_preds.numpy().tolist())

            context_strs = []
            for b in range(batch_size):
                ctx = ""
                for step_i, preds in enumerate(prev_predictions):
                    ctx += f"Step {step_i + 1} result: {int(preds[b])}\n"
                context_strs.append(ctx)
            augmented = [ctx + prob for ctx, prob in zip(context_strs, problems)]
            aug_tokens = tokenizer(
                augmented, padding=True, truncation=True,
                max_length=max_length + 20 * len(prev_predictions),
            )
            cycle_input_ids = Tensor(aug_tokens['input_ids'], dtype=dtypes.int32)
            cycle_attention_mask = Tensor(aug_tokens['attention_mask'], dtype=dtypes.float32)
            cycle_prompt_len = cycle_input_ids.shape[1]
        else:
            cycle_input_ids = input_ids
            cycle_attention_mask = attention_mask
            cycle_prompt_len = prompt_len

        # --- Embed tokens ---
        cycle_embeds = model.embed(cycle_input_ids)  # (B, S, D)

        # --- Get atom scales from hypernetwork ---
        if pass_num == 0 and len(state_pages) == 0:
            zero_page = Tensor.zeros(batch_size, model.page_size)
            atom_scales, pre_tanh = model.hypernet(
                [zero_page], pass_num=0, return_pre_tanh=True,
                messages=messages if messages else None,
            )
        else:
            atom_scales, pre_tanh = model.hypernet(
                state_pages, pass_num=pass_num, return_pre_tanh=True,
                messages=messages if messages else None,
            )

        # --- Think: run transformer with LoRA ---
        hidden_states = model.forward_with_lora(
            cycle_embeds, cycle_attention_mask, atom_scales,
        )

        # --- Perceiver: compress to page ---
        page_delta, current_mid_states = model.compress(hidden_states, pass_num)
        raw_pages.append(page_delta)

        # Hypersphere normalization
        page = l2_normalize(page_delta, dim=-1) * model.page_radius

        # Page noise during training
        page = page + Tensor.randn(*page.shape) * 0.05

        # Gradient scaling per cycle (capped at 4x)
        grad_scale = min(float(num_passes - pass_num), 4.0)
        page = scale_gradient(page, grad_scale)

        state_pages.append(page)

        # Message: direct signal from last hidden layer
        message = model.message_generator(hidden_states[-1])
        messages.append(message)

        # Atom dropout: randomly zero 10% of atom scales
        if pass_num > 0:
            atom_mask = (Tensor.rand(*atom_scales.shape) > 0.1).float()
            atom_scales = atom_scales * atom_mask

        atom_scales_history.append(atom_scales)
        pre_tanh_history.append(pre_tanh)

        # ------- Per-cycle losses -------
        if pass_num < max_steps:
            cycle_target_col = ct_np[:, pass_num]   # (B,) numpy
            mask_vals = cm_np[:, pass_num]           # (B,) numpy
            mask_sum = mask_vals.sum()

            if mask_sum > 0:
                mask_t = Tensor(mask_vals.tolist())

                # === GENERATION LOSS ===
                if cycle_gen_targets is not None and pass_num < len(cycle_gen_targets):
                    target_strs = cycle_gen_targets[pass_num]
                else:
                    target_strs = [str(int(cycle_target_col[b])) for b in range(batch_size)]
                target_tokens = tokenizer(target_strs, padding=True, add_special_tokens=False)
                target_ids = Tensor(target_tokens['input_ids'], dtype=dtypes.int32)
                pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

                # Target attention mask
                target_mask_np = np.array(target_tokens['attention_mask'], dtype=np.float32)
                target_mask_vals = Tensor(target_mask_np)

                # Teacher-forced: concat prompt embeds + target embeds
                target_embeds = model.embed_ids(target_ids)
                full_embeds = cycle_embeds.cat(target_embeds, dim=1)

                target_attn_np = np.array(target_tokens['attention_mask'], dtype=np.float32)
                full_attn = cycle_attention_mask.cat(Tensor(target_attn_np), dim=1)

                # Forward with LoRA for generation
                gen_hidden = model.forward_with_lora(full_embeds, full_attn, atom_scales)
                gen_logits = model.lm_head(gen_hidden[-1])  # (B, S, V)

                gen_loss_this = generation_loss(gen_logits, target_ids, target_mask_vals, cycle_prompt_len)

                # === ANSWER HEAD LOSS (flexible + consumption) ===
                current_page = page.float()
                all_ah_losses = []
                all_ah_indices = []

                # Check unconsumed intermediate targets
                for t_idx in range(max_steps):
                    if consumed[:, t_idx].sum() > 0:
                        gold_col = ct_np[:, t_idx].tolist()
                        all_ah_losses.append(
                            answer_head_loss(answer_head, current_page, gold_col, cycle_num=pass_num)
                        )
                        all_ah_indices.append(t_idx)

                # Always check final answer (never consumed)
                all_ah_losses.append(
                    answer_head_loss(answer_head, current_page, finals_t, cycle_num=pass_num)
                )
                all_ah_indices.append(-1)

                # Take the best match (lowest loss)
                ah_loss_vals = [l.realize().numpy().item() for l in all_ah_losses]
                best_idx = int(np.argmin(ah_loss_vals))
                ah_loss_this = all_ah_losses[best_idx]

                # Consume the matched target
                matched_target_idx = all_ah_indices[best_idx]
                if matched_target_idx >= 0:
                    consumed[:, matched_target_idx] = 0.0

                mask_frac = float(mask_sum) / batch_size

                # Smooth fading
                total_supervised = max_steps
                fade_w = per_cycle_target_weight(final_accuracy, pass_num, total_supervised)

                # Per-cycle weight flip
                if pass_num == 0:
                    per_cycle_gen_loss = per_cycle_gen_loss + gen_loss_this * 1.0 * fade_w
                    per_cycle_ah_loss = per_cycle_ah_loss + ah_loss_this * mask_frac * 0.5 * fade_w
                else:
                    per_cycle_gen_loss = per_cycle_gen_loss + gen_loss_this * 0.1 * fade_w
                    per_cycle_ah_loss = per_cycle_ah_loss + ah_loss_this * mask_frac * 5.0 * fade_w
                valid_cycles += 1

            # Record predictions for diagnostics
            with Tensor.no_grad:
                preds = answer_head.decode(page.float(), cycle_num=pass_num)
                per_cycle_preds.append(preds.numpy().tolist())

    # Normalize losses
    if valid_cycles > 0:
        per_cycle_gen_loss = per_cycle_gen_loss / float(valid_cycles)
        per_cycle_ah_loss = per_cycle_ah_loss / float(valid_cycles)

    # ------- Confidence + entropy loss -------
    conf_loss = confidence_entropy_loss(
        model.confidence_head, answer_head, state_pages, finals_t,
    )

    # ------- Contrastive loss -------
    c_loss = per_page_contrastive_loss(
        state_pages, finals_t, cross_target=0.7, within_target=0.3,
    )

    # ------- Pre-tanh regularization -------
    if pre_tanh_history:
        all_pre_tanh = Tensor.stack(*pre_tanh_history)  # (P, B, A)
        scale_reg_loss = (all_pre_tanh ** 2).mean()
    else:
        scale_reg_loss = Tensor(0.0)

    # ------- Isotropic regularization -------
    if raw_pages:
        raw_flat = Tensor.stack(*raw_pages).reshape(-1, model.page_size).float()
        iso_loss = isotropic_regularizer(raw_flat, target_var=1.0, corr_weight=0.1)
    else:
        iso_loss = Tensor(0.0)

    # ------- Diagnostics (no grad) -------
    with Tensor.no_grad:
        last_page = state_pages[-1].float()
        normed = l2_normalize(last_page, dim=-1)
        sim = normed.matmul(normed.T)
        B = sim.shape[0]
        eye = Tensor.eye(B)
        off_diag = sim - eye
        page_cos_mean = off_diag.sum().realize().numpy().item() / max(B * (B - 1), 1)

        all_scales = Tensor.stack(*atom_scales_history)  # (P, B, A)
        active_per_pass = (all_scales.abs() > 0.1).float().sum(axis=-1)
        active_atoms_mean = active_per_pass.mean().realize().numpy().item()

        scale_std = all_scales.std(axis=-1).mean().realize().numpy().item()

        if len(atom_scales_history) >= 2:
            s0 = l2_normalize(atom_scales_history[0], dim=-1)
            sN = l2_normalize(atom_scales_history[-1], dim=-1)
            cross_pass_cos = (s0 * sN).sum(axis=-1).mean().realize().numpy().item()
        else:
            cross_pass_cos = 0.0

        # Page variance
        per_dim_var = last_page.var(axis=0)
        page_var = per_dim_var.mean().realize().numpy().item()
        dead_dims = int((per_dim_var.realize().numpy() < 0.01).sum())

    return {
        'gen_loss': per_cycle_gen_loss,
        'ah_loss': per_cycle_ah_loss,
        'c_loss': c_loss,
        'conf_loss': conf_loss,
        'scale_reg': scale_reg_loss,
        'iso_loss': iso_loss,
        # Diagnostics (python floats)
        'page_cos_mean': page_cos_mean,
        'active_atoms_mean': active_atoms_mean,
        'scale_std': scale_std,
        'cross_pass_cos': cross_pass_cos,
        'page_var': page_var,
        'dead_dims': dead_dims,
        'state_pages': state_pages,
        'per_cycle_preds': per_cycle_preds,
    }


# ---------------------------------------------------------------------------
# Evaluate (answer-head only, per-cycle accuracy)
# ---------------------------------------------------------------------------

def evaluate_per_cycle(model, answer_head, eval_dataset, tokenizer,
                       num_passes=5, max_length=192):
    """Evaluate via answer head at each cycle. No text generation.

    Args:
        model:        main model
        answer_head:  AnswerHead module
        eval_dataset: PerCycleDataset or SubsetDataset
        tokenizer:    tokenizer
        num_passes:   int
        max_length:   int

    Returns:
        per_cycle_acc: list of floats (percentage)
        final_acc:     float (percentage)
    """
    eval_batch = 16
    max_steps_seen = 0

    per_cycle_correct = {}
    per_cycle_total = {}
    final_correct = 0
    total = 0

    n_samples = len(eval_dataset)

    with Tensor.no_grad:
        for i in range(0, n_samples, eval_batch):
            end = min(i + eval_batch, n_samples)
            if hasattr(eval_dataset, 'get_sample'):
                batch_samples = [eval_dataset.get_sample(j - i) for j in range(i, end)]
            else:
                batch_samples = [eval_dataset.samples[j] for j in range(i, end)]

            problems = [s['problem'] for s in batch_samples]
            cycle_targets_list = [
                s['cycle_targets'][:num_passes] for s in batch_samples
            ]
            gold_finals = [s['final_answer'] for s in batch_samples]

            tokens = tokenizer(
                problems, padding=True, truncation=True, max_length=max_length,
            )
            input_ids = Tensor(tokens['input_ids'], dtype=dtypes.int32)
            attention_mask = Tensor(tokens['attention_mask'], dtype=dtypes.float32)
            batch_size = input_ids.shape[0]

            state_pages = []
            eval_messages = []
            eval_prev_predictions = []

            for pass_num in range(num_passes):
                # Inject previous answers for cycle 2+
                if pass_num > 0 and len(state_pages) > 0:
                    latest_page = state_pages[-1].float()
                    latest_preds = answer_head.decode(latest_page, cycle_num=pass_num - 1)
                    eval_prev_predictions.append(latest_preds.numpy().tolist())
                    context_strs = []
                    for b in range(batch_size):
                        ctx = ""
                        for step_i, preds in enumerate(eval_prev_predictions):
                            ctx += f"Step {step_i + 1} result: {int(preds[b])}\n"
                        context_strs.append(ctx)
                    augmented = [ctx + prob for ctx, prob in zip(context_strs, problems)]
                    aug_tokens = tokenizer(
                        augmented, padding=True, truncation=True,
                        max_length=max_length + 20 * len(eval_prev_predictions),
                    )
                    eval_ids = Tensor(aug_tokens['input_ids'], dtype=dtypes.int32)
                    eval_mask = Tensor(aug_tokens['attention_mask'], dtype=dtypes.float32)
                else:
                    eval_ids = input_ids
                    eval_mask = attention_mask

                # One thinking pass
                page, scales, message, raw_page = model.thinking_pass(
                    eval_ids, eval_mask, state_pages, pass_num,
                    messages=eval_messages if eval_messages else None,
                )
                state_pages.append(page)
                eval_messages.append(message)

                # Per-cycle answer head eval
                current_page = page.float()
                preds = answer_head.decode(current_page, cycle_num=pass_num)
                preds_np = preds.numpy()

                for j in range(batch_size):
                    ct = cycle_targets_list[j]
                    if pass_num < len(ct):
                        gold_cycle = ct[pass_num]
                        pred_val = preds_np[j]

                        if pass_num not in per_cycle_correct:
                            per_cycle_correct[pass_num] = 0
                            per_cycle_total[pass_num] = 0
                        per_cycle_total[pass_num] += 1

                        try:
                            if int(pred_val) == int(gold_cycle):
                                per_cycle_correct[pass_num] += 1
                        except (ValueError, TypeError):
                            pass

                        max_steps_seen = max(max_steps_seen, pass_num + 1)

            # Final accuracy: last supervised cycle
            for j in range(batch_size):
                gold = gold_finals[j]
                ct = cycle_targets_list[j]
                last_supervised = min(len(ct), len(state_pages)) - 1
                final_page = state_pages[last_supervised].float()
                # Slice single sample
                pred_val = answer_head.decode(
                    final_page[j:j + 1], cycle_num=last_supervised,
                ).numpy()[0]
                try:
                    if int(pred_val) == int(gold):
                        final_correct += 1
                except (ValueError, TypeError):
                    pass
                total += 1

    per_cycle_acc = []
    for p in range(max_steps_seen):
        if p in per_cycle_total and per_cycle_total[p] > 0:
            per_cycle_acc.append(
                100.0 * per_cycle_correct.get(p, 0) / per_cycle_total[p]
            )
        else:
            per_cycle_acc.append(0.0)

    final_acc = 100.0 * final_correct / total if total > 0 else 0.0
    return per_cycle_acc, final_acc


# ---------------------------------------------------------------------------
# Gradient clipping (manual -- tinygrad has no built-in clip_grad_norm_)
# ---------------------------------------------------------------------------

def clip_grad_norm(params, max_norm=1.0):
    """Clip gradient norm across all parameters.

    Computes global norm, then scales each .grad if norm exceeds max_norm.
    """
    total_norm_sq = 0.0
    grads = []
    for p in params:
        if p.grad is not None:
            g = p.grad.realize()
            grads.append((p, g))
            total_norm_sq += (g * g).sum().realize().numpy().item()
    total_norm = math.sqrt(total_norm_sq)
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for p, g in grads:
            p.grad = g * scale


# ---------------------------------------------------------------------------
# Warm-start (safetensor)
# ---------------------------------------------------------------------------

def try_warm_start(model, answer_head, warm_path):
    """Load state dicts from safetensor checkpoint.

    Returns True on success, False on failure.
    """
    if not os.path.exists(warm_path):
        print(f"  WARNING: warm_from path does not exist: {warm_path}")
        return False

    print(f"  Loading checkpoint from {warm_path}")
    state = safe_load(warm_path)

    # Load into model
    model_sd = get_state_dict(model)
    loaded = 0
    for k, v in state.items():
        if k.startswith('answer_head.'):
            continue  # handled separately
        if k in model_sd and model_sd[k].shape == v.shape:
            model_sd[k].assign(v)
            loaded += 1
    print(f"  model: loaded {loaded} tensors")

    # Load answer head
    ah_sd = get_state_dict(answer_head)
    loaded_ah = 0
    for k, v in state.items():
        ah_key = k.replace('answer_head.', '') if k.startswith('answer_head.') else None
        if ah_key and ah_key in ah_sd and ah_sd[ah_key].shape == v.shape:
            ah_sd[ah_key].assign(v)
            loaded_ah += 1
    print(f"  answer_head: loaded {loaded_ah} tensors")
    return True


# ---------------------------------------------------------------------------
# Save checkpoint (safetensor)
# ---------------------------------------------------------------------------

def save_checkpoint(model, answer_head, path, epoch, accuracy, per_cycle_acc, level):
    """Save model + answer_head as a single safetensor file."""
    state = {}
    for k, v in get_state_dict(model).items():
        state[k] = v
    for k, v in get_state_dict(answer_head).items():
        state[f'answer_head.{k}'] = v
    safe_save(state, path)
    print(f"  -> saved checkpoint {path} (epoch={epoch}, final={accuracy:.1f}%)")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    level = args.level
    warm_path = args.warm_from

    print("=" * 60)
    print(f"Per-Cycle Answer Head Training (tinygrad) -- Level {level}")
    print("=" * 60)
    print(f"  num_passes     = {args.num_passes}")
    print(f"  batch_size     = {args.batch_size}")
    print(f"  epochs         = {args.epochs}")
    print(f"  patience       = {args.patience}")
    print(f"  lam_gen        = {args.lam_gen}")
    print(f"  lam_answer     = {args.lam_answer}")
    print(f"  lam_contrastive= {args.lam_contrastive}")
    print(f"  lam_conf       = {args.lam_conf}")
    print(f"  lam_scale_reg  = {args.lam_scale_reg}")
    print(f"  num_atoms      = {args.num_atoms}")
    print(f"  atom_rank      = {args.atom_rank}")
    print(f"  data_dir       = {args.data_dir}")
    print(f"  warm_from      = {warm_path}")
    print(f"  device         = {Device.DEFAULT}")
    print("=" * 60)

    # --- Model ---
    # NOTE: The actual model classes (AtomLoRAModel, AnswerHead, etc.) must be
    # ported to tinygrad separately. This training loop expects them to expose
    # the following interface:
    #
    #   model.page_size           -> int (64)
    #   model.page_radius         -> float (sqrt(64))
    #   model.embed(input_ids)    -> Tensor (B, S, D)
    #   model.embed_ids(ids)      -> Tensor (B, S, D)
    #   model.hypernet(pages, pass_num, return_pre_tanh, messages)
    #   model.forward_with_lora(embeds, attn_mask, atom_scales) -> list of Tensor
    #   model.compress(hidden_states, pass_num) -> (page_delta, mid_states)
    #   model.message_generator(last_hidden) -> Tensor (B, msg_dim)
    #   model.lm_head(hidden) -> Tensor (B, S, vocab)
    #   model.confidence_head(pages_list) -> Tensor (B, 1)
    #   model.thinking_pass(ids, mask, pages, pass_num, messages) -> (page, scales, msg, raw)
    #
    # For now we import them. Replace with tinygrad-native versions when ready.
    try:
        from tinygrad_port.model import AtomLoRAModel, AnswerHead
    except ImportError:
        print("ERROR: tinygrad_port/model.py not found.")
        print("The model classes (AtomLoRAModel, AnswerHead) must be ported to tinygrad.")
        print("This training script provides the loop, losses, and data loading.")
        print("Port the model architecture separately and place it in tinygrad_port/model.py.")
        sys.exit(1)

    model = AtomLoRAModel(
        num_atoms=args.num_atoms,
        atom_rank=args.atom_rank,
    )
    answer_head = AnswerHead(page_size=model.page_size)
    tokenizer = model.tokenizer

    if warm_path:
        print(f"\nWarm-starting from {warm_path}")
        try_warm_start(model, answer_head, warm_path)

    # --- Data ---
    train_path = os.path.join(args.data_dir, f'{level}_train.jsonl')
    eval_path = os.path.join(args.data_dir, f'{level}_eval.jsonl')

    if not os.path.exists(train_path):
        print(f"ERROR: Training data not found at {train_path}")
        print(f"Expected JSONL: {{\"problem\": \"...\", \"cycle_targets\": [...], \"final_answer\": N, \"num_steps\": N}}")
        sys.exit(1)

    train_dataset = PerCycleDataset(train_path, max_passes=args.num_passes)
    print(f"\nTrain dataset: {len(train_dataset)} problems from {train_path}")

    if os.path.exists(eval_path):
        eval_dataset = PerCycleDataset(eval_path, max_passes=args.num_passes)
        print(f"Eval dataset: {len(eval_dataset)} problems from {eval_path}")
    else:
        eval_size = min(200, len(train_dataset))
        start = len(train_dataset) - eval_size
        eval_dataset = SubsetDataset(train_dataset, start, len(train_dataset))
        print(f"  Eval file not found, using last {eval_size} of training data")

    # --- Optimizer ---
    # Collect all trainable parameters with per-group learning rates.
    # tinygrad AdamW does not support param groups natively, so we create
    # separate optimizers per group and step them all.

    s = args.lr_scale
    param_groups = [
        (list(get_state_dict(model.compressor).values()), 1e-4 * s, 0.01),
        (list(get_state_dict(model.atoms).values()), 1e-4 * s, 0.05),
        (list(get_state_dict(model.hypernet).values()), 1e-3 * s, 0.1),
        (list(get_state_dict(model.confidence_head).values()), 1e-3 * s, 0.01),
        (list(get_state_dict(answer_head).values()), 3e-3 * s, 0.01),
        (list(get_state_dict(model.message_generator).values()), 1e-3 * s, 0.01),
    ]

    optimizers = []
    all_params = []
    for params, lr, wd in param_groups:
        if params:
            optimizers.append(AdamW(params, lr=lr, weight_decay=wd))
            all_params.extend(params)

    total_trainable = sum(np.prod(p.shape) for p in all_params)
    print(f"Trainable params: {total_trainable:,}")

    # --- Baseline eval ---
    per_cycle_acc, final_acc = evaluate_per_cycle(
        model, answer_head, eval_dataset, tokenizer,
        num_passes=args.num_passes,
    )
    print(f"\nBaseline: final={final_acc:.1f}%  per_cycle={[f'{a:.1f}%' for a in per_cycle_acc]}\n")

    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_name = os.path.join(ckpt_dir, f"per_cycle_{level.replace('.', '')}_best.safetensor")
    best_final = 0.0
    patience_counter = 0

    # --- Epoch loop ---
    for epoch in range(args.epochs):
        t0 = time.time()

        ep_gen = ep_ah = ep_ctr = ep_conf = ep_scale_reg = ep_iso = 0.0
        ep_cos = ep_active = ep_std = ep_xpass = 0.0
        ep_page_var = 0.0
        ep_dead_dims = 0.0
        nb = 0

        for batch in batch_iterator(train_dataset, args.batch_size, shuffle=True):
            problems = batch['problems']
            cycle_targets = batch['cycle_targets']
            cycle_mask = batch['cycle_mask']
            cycle_gen_targets = batch.get('cycle_gen_targets')

            finals_raw = batch['final_answers']
            finals_list = []
            for f in finals_raw:
                try:
                    finals_list.append(int(f))
                except (ValueError, TypeError):
                    finals_list.append(0)

            # Forward
            result = forward_train_per_cycle(
                model, answer_head, tokenizer,
                problems, cycle_targets, cycle_mask,
                finals_list,
                cycle_gen_targets=cycle_gen_targets,
                num_passes=args.num_passes,
                final_accuracy=best_final / 100.0,
            )

            # Total loss
            total_loss = (
                args.lam_gen * result['gen_loss']
                + args.lam_answer * result['ah_loss']
                + args.lam_contrastive * result['c_loss']
                + args.lam_conf * result['conf_loss']
                + args.lam_scale_reg * result['scale_reg']
                + 0.01 * result['iso_loss']
            )

            # Backward
            total_loss.backward()

            # Clip gradients
            clip_grad_norm(all_params, max_norm=1.0)

            # Step all optimizers
            for opt in optimizers:
                opt.step()

            # Zero grads
            for opt in optimizers:
                opt.zero_grad()

            # Accumulate stats
            ep_gen += result['gen_loss'].realize().numpy().item()
            ep_ah += result['ah_loss'].realize().numpy().item()
            ep_ctr += result['c_loss'].realize().numpy().item()
            ep_conf += result['conf_loss'].realize().numpy().item()
            ep_scale_reg += result['scale_reg'].realize().numpy().item()
            ep_iso += result['iso_loss'].realize().numpy().item()
            ep_cos += result['page_cos_mean']
            ep_active += result['active_atoms_mean']
            ep_std += result['scale_std']
            ep_xpass += result['cross_pass_cos']
            ep_page_var += result['page_var']
            ep_dead_dims += result['dead_dims']
            nb += 1

            if nb % 10 == 0:
                print(f"  batch {nb}: gen={ep_gen/nb:.4f} ah={ep_ah/nb:.4f}")

        elapsed = time.time() - t0

        # Eval
        per_cycle_acc, final_acc = evaluate_per_cycle(
            model, answer_head, eval_dataset, tokenizer,
            num_passes=args.num_passes,
        )

        improved = False
        if final_acc > best_final:
            best_final = final_acc
            improved = True

        if improved:
            patience_counter = 0
            save_checkpoint(
                model, answer_head, ckpt_name,
                epoch=epoch + 1, accuracy=final_acc,
                per_cycle_acc=per_cycle_acc, level=level,
            )
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch+1}: "
            f"gen={ep_gen/nb:.4f} ah={ep_ah/nb:.4f} contr={ep_ctr/nb:.2f} "
            f"conf={ep_conf/nb:.2f} scale_reg={ep_scale_reg/nb:.2f} iso={ep_iso/nb:.4f} "
            f"page_cos={ep_cos/nb:.2f} "
            f"active={ep_active/nb:.1f}/{args.num_atoms} "
            f"scale_std={ep_std/nb:.3f} "
            f"xpass_cos={ep_xpass/nb:.2f} | "
            f"Final={final_acc:.1f}% best={best_final:.1f}% [{elapsed:.0f}s]"
        )
        print(f"  per_cycle_acc: {[f'{a:.1f}%' for a in per_cycle_acc]}")
        print(f"  page_var={ep_page_var/nb:.4f} dead_dims={ep_dead_dims/nb:.1f}/64")

        if patience_counter >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} epochs")
            break

    print(f"\n{'='*60}")
    print(f"Level {level} final: {best_final:.1f}%")
    print(f"Checkpoint: {ckpt_name}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Per-cycle answer head training (tinygrad port)',
    )
    p.add_argument(
        '--level', type=str, required=True,
        choices=['L3', 'L4', 'L4.5', 'L4.7', 'L4.9', 'L5', 'gsm8k'],
        help='Curriculum level to train',
    )
    p.add_argument('--data_dir', type=str, default='data/per_cycle',
                   help='Directory containing JSONL data files')
    p.add_argument('--warm_from', type=str, default=None,
                   help='Safetensor checkpoint to warm-start from')
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch_size', type=int, default=8,
                   help='Batch size (default: 8)')
    p.add_argument('--num_passes', type=int, default=3,
                   help='Number of thinking passes per problem')
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--lam_gen', type=float, default=1.0,
                   help='Per-cycle generation loss weight')
    p.add_argument('--lam_answer', type=float, default=0.5,
                   help='Per-cycle answer head loss weight')
    p.add_argument('--lam_contrastive', type=float, default=0.05,
                   help='Contrastive loss weight')
    p.add_argument('--lam_conf', type=float, default=0.1,
                   help='Confidence loss weight')
    p.add_argument('--lam_scale_reg', type=float, default=0.1,
                   help='Pre-tanh scale regularization weight')
    p.add_argument('--lr_scale', type=float, default=1.0,
                   help='Scale all learning rates')
    p.add_argument('--num_atoms', type=int, default=64,
                   help='Number of LoRA atoms')
    p.add_argument('--atom_rank', type=int, default=6,
                   help='Rank of each LoRA atom')

    args = p.parse_args()
    train(args)
