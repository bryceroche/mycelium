"""
Mycelium v2 Training Loop — Phase 1: Linear Breathing

Cardinal rule: Controller gradient NEVER flows through Llama.
Separate backward passes: gen_loss→atoms2, ST loss→controller.

Usage:
  python scripts/train_v2.py --level L3 --epochs 10 --batch_size 8
  python scripts/train_v2.py --level L4 --warm_from checkpoints/v2_L3_best.pt --epochs 10
"""

import os
import sys
import re
import json
import time
import math
import random
import argparse
from typing import List, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reuse v1 infrastructure
from scripts.train_per_cycle import (
    PerCycleDataset, per_cycle_collate_fn,
    numerical_proximity_reward,
    build_number_token_weights,
)
from scripts.atom_lora import (
    LoRAAtoms, AtomAdditiveLoRAManager,
)
from scripts.controller import BreathingController, TreeNotebook, TreeNode


# ---------------------------------------------------------------------------
# Model wrapper: Llama + baked L1 + L2 atoms + v2 controller
# ---------------------------------------------------------------------------

class MyceliumV2(nn.Module):
    """Llama with baked L1, trainable L2 atoms, and v2 BreathingController."""

    def __init__(self, model_name='unsloth/Llama-3.2-1B', num_atoms=64, atom_rank=6):
        super().__init__()

        # Load Llama
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.transformer = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.transformer.config.pad_token_id = self.tokenizer.eos_token_id

        # Freeze Llama
        for p in self.transformer.parameters():
            p.requires_grad = False

        self.num_layers = len(self.transformer.model.layers)
        self.d_model = self.transformer.config.hidden_size  # 2048

        # L1 atoms (frozen, will be baked)
        self.atoms = LoRAAtoms(
            d_model=self.d_model, d_kv=512,
            rank=atom_rank, num_atoms=num_atoms, num_layers=self.num_layers,
        )
        for p in self.atoms.parameters():
            p.requires_grad = False

        # L2 atoms (trainable)
        self.atoms2 = LoRAAtoms(
            d_model=self.d_model, d_kv=512,
            rank=atom_rank, num_atoms=num_atoms, num_layers=self.num_layers,
        )

        # v2 Controller — OWNS scales 100%
        # atoms2 provides A/B matrices (tools), controller decides scales (tool selection)
        self.controller = BreathingController(
            hidden_dim=self.d_model,
            num_llama_layers=self.num_layers,
        )

        self._l1_baked = False

    def bake_l1_atoms(self, scale=0.46):
        """Permanently embed L1 atoms into Llama's weights."""
        if self._l1_baked:
            return
        with torch.no_grad():
            for layer_idx in range(self.num_layers):
                for proj_name in LoRAAtoms.PROJ_NAMES:
                    proj = getattr(
                        self.transformer.model.layers[layer_idx].self_attn,
                        proj_name,
                    )
                    A = self.atoms.A[proj_name][:, layer_idx]  # (64, d_model, rank)
                    B = self.atoms.B[proj_name][:, layer_idx]  # (64, rank, proj_dim)
                    delta = sum(scale * (A[i] @ B[i]) for i in range(A.shape[0]))
                    proj.weight.data += delta.T.to(dtype=proj.weight.dtype)
        self._l1_baked = True
        print(f"Baked L1 atoms into Llama weights (scale={scale}, frozen)")


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(text):
    """Extract integer answer after #### marker, or fallback to last number."""
    match = re.search(r'####\s*(-?\d+)', text)
    if match:
        val = int(match.group(1))
        return max(min(val, 999999999), -999999999)
    # Fallback: last number in text
    nums = re.findall(r'-?\d+', text)
    if nums:
        val = int(nums[-1])
        return max(min(val, 999999999), -999999999)
    return None


# ---------------------------------------------------------------------------
# Single cycle: controller → Llama generation → loss
# ---------------------------------------------------------------------------

def run_cycle(model, input_ids, attention_mask, problem_embeds,
              hidden_states_all_layers, notebook, cycle_num,
              gen_target_strs, cycle_target_vals, prev_scales,
              available_targets, consumed_targets, device,
              max_inner_passes=1):
    """
    Run one breathing cycle: controller decides scales, Llama generates.

    Returns:
        gen_loss_per_sample: (B,) per-sample generation loss
        reward_weights: (B,) equal-reward weights
        scales: (B, 64) controller output (for ST gradient)
        page: (B, 256) page for notebook
        branch_embed: (B, 64) branch embedding
        hidden_pool: (B, 1024) pooled hidden state
        pred_vals: list of predicted values per sample
    """
    batch_size = input_ids.size(0)
    embed_layer = model.transformer.model.embed_tokens
    prompt_len = input_ids.size(1)

    # Controller produces delta scales (per-problem adjustment)
    ctrl_scales, page, branch_embed, branch_action, energy, confidence, trunk_out = \
        model.controller(
            hidden_states_all_layers, notebook,
            cycle_num=cycle_num, pass_num=0,
            prev_scales=prev_scales,
        )

    hidden_pool = trunk_out[:, :model.controller.latent_dim]  # (B, 1536)

    # Controller OWNS scales — ctrl_scales IS the final scales
    scales = ctrl_scales  # (B, 64), already 0.46*tanh from controller

    # Teacher-forced generation with L2 LoRA
    target_ids = model.tokenizer(
        gen_target_strs, return_tensors='pt', padding=True,
        truncation=True, max_length=60, add_special_tokens=False,
    ).input_ids.to(device)

    # Append EOS
    eos_col = torch.full((batch_size, 1), model.tokenizer.eos_token_id,
                         dtype=torch.long, device=device)
    target_ids = torch.cat([target_ids, eos_col], dim=1)

    target_embeds = embed_layer(target_ids)
    full_embeds = torch.cat([problem_embeds, target_embeds], dim=1)
    target_attn = (target_ids != model.tokenizer.pad_token_id).long()
    full_attn = torch.cat([attention_mask, target_attn], dim=1)

    # Apply L2 atoms with controller's scales
    mgr = AtomAdditiveLoRAManager(model.transformer)
    mgr.apply(model.atoms2, scales)
    try:
        gen_outputs = model.transformer(
            inputs_embeds=full_embeds,
            attention_mask=full_attn,
            use_cache=False,
        )
    finally:
        mgr.remove()

    # Compute per-sample generation loss
    gen_logits = gen_outputs.logits[:, prompt_len - 1:-1, :]
    vocab_size = gen_logits.size(-1)

    target_flat = target_ids.reshape(-1)
    logits_flat = gen_logits.reshape(-1, vocab_size)

    loss_per_token = F.cross_entropy(
        logits_flat, target_flat,
        ignore_index=model.tokenizer.pad_token_id,
        reduction='none',
        label_smoothing=0.05,
    ).view(batch_size, -1)

    # Number token weighting (3x on digits and ####)
    num_weights = build_number_token_weights(target_ids, model.tokenizer, number_weight=3.0)
    target_mask = (target_ids != model.tokenizer.pad_token_id).float()

    gen_loss_per_sample = (loss_per_token * target_mask * num_weights).sum(dim=1)
    gen_loss_per_sample = gen_loss_per_sample / (target_mask * num_weights).sum(dim=1).clamp(min=1)

    # Number-only loss for ST gradient (strips format noise, pure per-problem signal)
    num_only_mask = (num_weights > 1.5).float()  # only number tokens (weight=3.0)
    num_loss_per_sample = (loss_per_token * target_mask * num_only_mask).sum(dim=1)
    num_loss_per_sample = num_loss_per_sample / (target_mask * num_only_mask).sum(dim=1).clamp(min=1)

    # Equal-reward: extract predictions, compute reward weights
    with torch.no_grad():
        pred_tokens = gen_logits.argmax(dim=-1)
        pred_vals = []
        reward_weights = torch.ones(batch_size, device=device)

        for b in range(batch_size):
            text = model.tokenizer.decode(pred_tokens[b], skip_special_tokens=True)
            pred_val = extract_answer(text)
            pred_vals.append(pred_val)

            if pred_val is None:
                reward_weights[b] = 0.3
            elif pred_val in consumed_targets[b]:
                reward_weights[b] = 0.0  # copied consumed target
            elif pred_val in available_targets[b]:
                reward_weights[b] = 1.0  # correct!
                available_targets[b].remove(pred_val)
                consumed_targets[b].append(pred_val)
            else:
                # Proximity reward
                best_prox = 0.3
                for t in available_targets[b]:
                    best_prox = max(best_prox, numerical_proximity_reward(pred_val, t))
                reward_weights[b] = best_prox

    return (gen_loss_per_sample, num_loss_per_sample, reward_weights, scales, page,
            branch_embed, hidden_pool, pred_vals, energy)


# ---------------------------------------------------------------------------
# Full forward pass: comprehend + N cycles
# ---------------------------------------------------------------------------

def forward_pass(model, batch, device, num_passes=3):
    """
    Full breathing loop: comprehend → N cycles.

    Returns:
        atom_loss: scalar, for atoms2 backward
        st_losses: list of (scales, scale_grad_normalized) tuples for controller backward
        metrics: dict of logged values
    """
    problems = batch['problem']
    cycle_targets = batch['cycle_targets'].to(device)  # (B, max_steps)
    cycle_mask = batch['cycle_mask'].to(device)
    gen_targets = batch.get('cycle_gen_targets', None)
    max_steps = batch['max_steps']

    batch_size = cycle_targets.size(0)

    # Tokenize problems
    inputs = model.tokenizer(
        problems, return_tensors='pt', padding=True,
        truncation=True, max_length=192,
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Step 0: Comprehend — vanilla Llama forward (no L2 atoms)
    embed_layer = model.transformer.model.embed_tokens
    problem_embeds = embed_layer(input_ids)

    with torch.no_grad():
        comprehend_out = model.transformer(
            inputs_embeds=problem_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    hidden_states_all = list(comprehend_out.hidden_states)

    # Initialize target tracking
    available_targets = []
    consumed_targets = []
    for b in range(batch_size):
        n_steps = int(cycle_mask[b].sum().item())
        targets_b = [int(cycle_targets[b, i].item()) for i in range(n_steps)]
        available_targets.append(list(targets_b))
        consumed_targets.append([])

    notebook = TreeNotebook()
    prev_scales = None
    total_gen_loss = torch.tensor(0.0, device=device)
    st_loss_pairs = []  # (scales, normalized_grad) pairs
    all_cycle_scales = []  # collect for diversity loss
    valid_cycles = 0
    per_cycle_preds = []

    num_cycles = min(num_passes, max_steps)

    for cycle in range(num_cycles):
        mask_val = cycle_mask[:, cycle]
        if mask_val.sum() == 0:
            continue

        # Get generation targets for this cycle
        if gen_targets is not None:
            gen_strs = gen_targets[cycle]
        else:
            gen_strs = [str(cycle_targets[b, cycle].item()) for b in range(batch_size)]

        # Run cycle
        (gen_loss_ps, num_loss_ps, reward_weights, scales, page, branch_embed,
         hidden_pool, pred_vals, energy) = run_cycle(
            model, input_ids, attention_mask, problem_embeds,
            hidden_states_all, notebook, cycle_num=cycle,
            gen_target_strs=gen_strs,
            cycle_target_vals=[int(cycle_targets[b, cycle].item()) for b in range(batch_size)],
            prev_scales=prev_scales,
            available_targets=available_targets,
            consumed_targets=consumed_targets,
            device=device,
        )

        # Weighted generation loss for atoms2 training
        weighted_gen = (gen_loss_ps * reward_weights * mask_val).sum() / mask_val.sum().clamp(min=1)

        # Number-only loss for ST gradient (strips format noise)
        weighted_num = (num_loss_ps * reward_weights * mask_val).sum() / mask_val.sum().clamp(min=1)

        # Straight-through gradient: use NUMBER-ONLY loss for per-problem signal
        if scales.requires_grad and weighted_num.requires_grad:
            try:
                scale_grad = torch.autograd.grad(
                    weighted_num, scales,
                    retain_graph=True, allow_unused=True,
                )[0]
                if scale_grad is not None:
                    sg_norm = scale_grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    scale_grad_normalized = scale_grad / sg_norm
                    st_loss_pairs.append((scales, scale_grad_normalized.detach()))
            except RuntimeError:
                pass

        total_gen_loss = total_gen_loss + weighted_gen
        valid_cycles += 1
        per_cycle_preds.append(pred_vals)
        all_cycle_scales.append(scales)  # keep in graph for diversity loss

        # Record in notebook
        with torch.no_grad():
            node = TreeNode(
                page=page[0].detach(),
                branch_embed=branch_embed[0].detach(),
                hidden_pool=hidden_pool[0].detach(),
                action='solve',
                parent_idx=len(notebook) - 1 if len(notebook) > 0 else -1,
            )
            notebook.append(node)

        prev_scales = scales.detach()

    # Average gen loss across cycles
    if valid_cycles > 0:
        total_gen_loss = total_gen_loss / valid_cycles

    # Build ST losses
    st_losses = []
    for scales_i, sg_norm_i in st_loss_pairs:
        st_loss = (scales_i * sg_norm_i).sum()
        st_losses.append(st_loss)

    # Scale diversity loss: maximize mean absolute deviation across batch
    # Variance has zero gradient at the fixed point (all identical).
    # MAD = mean(|x - mean(x)|) has gradient sign(x - mean) / N — nonzero
    # everywhere except exactly at the mean, and dropout ensures noise.
    diversity_loss = torch.tensor(0.0, device=device)
    if all_cycle_scales:
        for scales_c in all_cycle_scales:
            bs = scales_c.shape[0]
            if bs > 1:
                batch_mean = scales_c.mean(dim=0, keepdim=True)  # (1, 64)
                mad = (scales_c - batch_mean).abs().mean()  # scalar
                diversity_loss = diversity_loss - mad  # negative = maximize MAD
        diversity_loss = diversity_loss / len(all_cycle_scales)

    # Metrics
    metrics = {
        'gen_loss': total_gen_loss.item(),
        'diversity_loss': diversity_loss.item(),
        'num_cycles': valid_cycles,
        'per_cycle_preds': per_cycle_preds,
    }

    return total_gen_loss, st_losses, diversity_loss, metrics


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, eval_dataset, device, num_passes=3, max_length=192):
    """Evaluate via generation + regex extraction."""
    model.eval()
    eval_batch = 16

    per_cycle_correct = {}
    per_cycle_total = {}
    final_correct = 0
    total = 0
    max_steps_seen = 0

    all_scales_for_diversity = []

    with torch.no_grad():
        for i in range(0, len(eval_dataset), eval_batch):
            batch_samples = [eval_dataset[j] for j in range(i, min(i + eval_batch, len(eval_dataset)))]
            problems = [s['problem'] for s in batch_samples]
            cycle_targets_list = [s['cycle_targets'] for s in batch_samples]
            gold_finals = [s['final_answer'] for s in batch_samples]

            inputs = model.tokenizer(
                problems, return_tensors='pt', padding=True,
                truncation=True, max_length=max_length,
            )
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            bs = input_ids.size(0)

            # Comprehend
            embed_layer = model.transformer.model.embed_tokens
            problem_embeds = embed_layer(input_ids)
            comprehend_out = model.transformer(
                inputs_embeds=problem_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden_states_all = list(comprehend_out.hidden_states)

            notebook = TreeNotebook()
            prev_scales = None
            eval_pred_vals = []

            for cycle in range(num_passes):
                # Controller produces delta
                ctrl_scales, page, branch_embed, action, energy, conf, trunk = \
                    model.controller(
                        hidden_states_all, notebook,
                        cycle_num=cycle, prev_scales=prev_scales,
                    )

                # Controller owns scales directly
                scales = ctrl_scales

                # Collect scales for diversity check (first batch only)
                if i == 0 and cycle == 0:
                    all_scales_for_diversity.append(scales.float().cpu())

                # Generate with L2 atoms
                mgr = AtomAdditiveLoRAManager(model.transformer)
                mgr.apply(model.atoms2, scales)
                try:
                    gen_tokens = model.transformer.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=60,
                        do_sample=False,
                        pad_token_id=model.tokenizer.pad_token_id,
                    )
                finally:
                    mgr.remove()

                # Extract predictions
                cycle_preds = []
                for b in range(bs):
                    gen_text = model.tokenizer.decode(
                        gen_tokens[b, input_ids.size(1):], skip_special_tokens=True,
                    )
                    cycle_preds.append(extract_answer(gen_text))
                eval_pred_vals.append(cycle_preds)

                # Score this cycle
                for j in range(bs):
                    ct = cycle_targets_list[j]
                    if cycle < len(ct):
                        if cycle not in per_cycle_correct:
                            per_cycle_correct[cycle] = 0
                            per_cycle_total[cycle] = 0
                        per_cycle_total[cycle] += 1
                        try:
                            if cycle_preds[j] is not None and int(cycle_preds[j]) == int(ct[cycle]):
                                per_cycle_correct[cycle] += 1
                        except (ValueError, TypeError):
                            pass
                        max_steps_seen = max(max_steps_seen, cycle + 1)

                # Update notebook
                node = TreeNode(
                    page=page[0].detach(),
                    branch_embed=branch_embed[0].detach(),
                    hidden_pool=trunk[0, :model.controller.latent_dim].detach(),
                    action='solve',
                )
                notebook.append(node)
                prev_scales = scales

            # Final accuracy
            for j in range(bs):
                ct = cycle_targets_list[j]
                last_cycle = min(len(ct), len(eval_pred_vals)) - 1
                pred = eval_pred_vals[last_cycle][j] if last_cycle >= 0 else None
                try:
                    if pred is not None and int(pred) == int(gold_finals[j]):
                        final_correct += 1
                except (ValueError, TypeError):
                    pass
                total += 1

    # Per-cycle accuracy
    per_cycle_acc = []
    for p in range(max_steps_seen):
        if p in per_cycle_total and per_cycle_total[p] > 0:
            per_cycle_acc.append(100.0 * per_cycle_correct.get(p, 0) / per_cycle_total[p])
        else:
            per_cycle_acc.append(0.0)

    final_acc = 100.0 * final_correct / total if total > 0 else 0.0

    # Scale diversity (from first eval batch)
    scale_xproblem_cos = 1.0
    scale_mid_frac = 0.0
    if all_scales_for_diversity:
        div_stack = torch.cat(all_scales_for_diversity, dim=0)
        n = div_stack.shape[0]
        if n > 1:
            cos_sims = []
            for di in range(n):
                for dj in range(di + 1, n):
                    cos_sims.append(F.cosine_similarity(
                        div_stack[di:di+1], div_stack[dj:dj+1]).item())
            scale_xproblem_cos = sum(cos_sims) / len(cos_sims)
        scale_mid_frac = (div_stack.abs() < 0.44).float().mean().item()

    return per_cycle_acc, final_acc, scale_xproblem_cos, scale_mid_frac


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    print("=" * 60)
    print(f"Mycelium v2 Training — Level {args.level}")
    print("=" * 60)

    device = torch.device('cuda')

    # Build model
    model = MyceliumV2(num_atoms=args.num_atoms, atom_rank=args.atom_rank)
    model.transformer = model.transformer.to(device)
    model.atoms = model.atoms.to(device=device, dtype=torch.bfloat16)
    model.atoms2 = model.atoms2.to(device=device, dtype=torch.bfloat16)
    model.controller = model.controller.to(device=device, dtype=torch.bfloat16)

    # Warm start
    if args.warm_from:
        print(f"\nWarm-starting from {args.warm_from}")
        ckpt = torch.load(args.warm_from, map_location=device)

        if 'atoms' in ckpt:
            model.atoms.load_state_dict(ckpt['atoms'])
            print(f"  atoms L1: loaded")
        # atoms2 (L2): load A/B matrices (what atoms do)
        if 'atoms2' in ckpt:
            model.atoms2.load_state_dict(ckpt['atoms2'])
            print(f"  atoms L2: loaded (A/B matrices)")
        # Controller: fresh init (owns scales, no legacy weights)
        # Scale head starts with small random weights → near-zero scales
        # Fingerprint provides per-problem diversity from epoch 1
        print(f"  controller: FRESH INIT (346M, owns scales 100%)")

    # Bake L1
    model.bake_l1_atoms()

    # Data
    data_dir = args.data_dir
    level = args.level
    train_path = os.path.join(data_dir, f'{level}_train.jsonl')
    eval_path = os.path.join(data_dir, f'{level}_eval.jsonl')

    if not os.path.exists(train_path):
        print(f"ERROR: {train_path} not found")
        sys.exit(1)

    train_dataset = PerCycleDataset(
        train_path, max_passes=args.num_passes,
        augment=args.augment, dropout_prob=0.15,
    )
    print(f"Train: {len(train_dataset)} problems from {train_path}")

    eval_dataset = None
    if os.path.exists(eval_path):
        eval_dataset = PerCycleDataset(eval_path, max_passes=args.num_passes)
        print(f"Eval: {len(eval_dataset)} problems from {eval_path}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, collate_fn=per_cycle_collate_fn,
    )

    # SEPARATE optimizers: atoms2 vs controller (they never share gradients)
    # atoms2 learns A/B matrices (what tools do) via gen_loss
    # controller learns scales (which tools to use) via ST estimator
    s = args.lr_scale
    atom_params = list(model.atoms2.parameters())
    ctrl_params = list(model.controller.parameters())

    atom_optimizer = torch.optim.AdamW(
        [{'params': atom_params, 'lr': 1e-4 * s, 'weight_decay': 0.05}],
    )
    ctrl_optimizer = torch.optim.AdamW(
        [{'params': ctrl_params, 'lr': 1e-3 * s, 'weight_decay': 0.01}],
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")
    print(f"  atoms L2: {sum(p.numel() for p in atom_params):,}")
    print(f"  controller: {sum(p.numel() for p in ctrl_params):,}")
    print(f"GPU mem after init: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Baseline eval
    best_final = 0.0
    if eval_dataset:
        per_cycle_acc, final_acc, xp_cos, mid_frac = evaluate(
            model, eval_dataset, device, num_passes=args.num_passes,
        )
        print(f"\nBaseline: final={final_acc:.1f}%  per_cycle={[f'{a:.1f}%' for a in per_cycle_acc]}")
        print(f"  scale_xproblem_cos={xp_cos:.4f} scale_mid_frac={mid_frac:.3f}")
        best_final = final_acc

    # Training
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_dataset.set_epoch(epoch)
        t0 = time.time()

        ep_gen = 0.0
        ep_st = 0.0
        ep_div = 0.0
        nb = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            atom_optimizer.zero_grad()
            ctrl_optimizer.zero_grad()

            gen_loss, st_losses, div_loss, metrics = forward_pass(
                model, batch, device, num_passes=args.num_passes,
            )

            # Backward 1: gen_loss trains atoms2 A/B matrices ONLY
            # Controller gets weak gradient through Llama here, but that's fine —
            # the ST gradient (Backward 2) is the primary controller signal
            if gen_loss.requires_grad:
                gen_loss.backward(retain_graph=bool(st_losses))

            # Step atoms2 immediately (its gradient is clean from gen_loss)
            torch.nn.utils.clip_grad_norm_(atom_params, 1.0)
            atom_optimizer.step()

            # Backward 2: ST loss trains controller via direct gradient
            if st_losses:
                # Zero controller grads from the gen_loss backward (weak, noisy)
                ctrl_optimizer.zero_grad()
                st_total = sum(st_losses)
                st_total.backward()
                ep_st += st_total.item()

            # Step controller
            torch.nn.utils.clip_grad_norm_(ctrl_params, 1.0)
            ctrl_optimizer.step()

            ep_gen += metrics['gen_loss']
            ep_div += metrics.get('diversity_loss', 0.0)
            nb += 1

        elapsed = time.time() - t0

        # Gradient norms (from last batch)
        ctrl_grad = 0.0
        ctrl_count = 0
        for p in model.controller.parameters():
            if p.grad is not None:
                ctrl_grad += p.grad.norm().item()
                ctrl_count += 1
        ctrl_grad_avg = ctrl_grad / max(ctrl_count, 1)

        atom_grad = 0.0
        atom_count = 0
        for p in model.atoms2.parameters():
            if p.grad is not None:
                atom_grad += p.grad.norm().item()
                atom_count += 1
        atom_grad_avg = atom_grad / max(atom_count, 1)

        # Eval
        if eval_dataset:
            per_cycle_acc, final_acc, xp_cos, mid_frac = evaluate(
                model, eval_dataset, device, num_passes=args.num_passes,
            )

            improved = final_acc > best_final
            if improved:
                best_final = final_acc
                patience_counter = 0
                ckpt_name = f'checkpoints/v2_{level}_best.pt'
                torch.save({
                    'epoch': epoch + 1,
                    'atoms': model.atoms.state_dict(),
                    'atoms2': model.atoms2.state_dict(),
                    'controller': model.controller.state_dict(),
                    'accuracy': final_acc,
                    'level': level,
                }, ckpt_name)
                print(f"  -> saved {ckpt_name} (final={final_acc:.1f}%)")
            else:
                patience_counter += 1

            print(
                f"Epoch {epoch+1}: gen={ep_gen/nb:.4f} st={ep_st/nb:.4f} div={ep_div/nb:.4f} | "
                f"Final={final_acc:.1f}% best={best_final:.1f}% [{elapsed:.0f}s]"
            )
            print(f"  per_cycle_acc: {[f'{a:.1f}%' for a in per_cycle_acc]}")
            print(f"  grad norms: controller={ctrl_grad_avg:.4f} atoms2={atom_grad_avg:.4f}")
            print(f"  scale_xproblem_cos={xp_cos:.4f} scale_mid_frac={mid_frac:.3f}")

            # Phase 1 health check
            ctrl_healthy = ctrl_grad_avg >= 0.01
            diverse = xp_cos < 0.9
            not_saturated = mid_frac > 0.3
            status = []
            if ctrl_healthy:
                status.append("ctrl_grad OK")
            else:
                status.append("ctrl_grad LOW")
            if diverse:
                status.append("diverse")
            else:
                status.append("CONSTANT")
            if not_saturated:
                status.append("unsaturated")
            else:
                status.append("SATURATED")
            print(f"  controller health: [{', '.join(status)}]")

        if patience_counter >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} epochs")
            break

    print(f"\n{'='*60}")
    print(f"Level {level} final: {best_final:.1f}%")
    print("=" * 60)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--level', type=str, required=True)
    p.add_argument('--data_dir', type=str, default='data/per_cycle')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_passes', type=int, default=3)
    p.add_argument('--lr_scale', type=float, default=1.0)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--warm_from', type=str, default=None)
    p.add_argument('--augment', action='store_true')
    p.add_argument('--num_atoms', type=int, default=64)
    p.add_argument('--atom_rank', type=int, default=6)
    args = p.parse_args()
    train(args)
