"""
Mycelium v3 Training Loop — Baked Llama + Thinking Controller

No atoms. No scales. No ST gradient. No monkey-patching.
Controller thinks in pages, steers Llama via differentiable soft tokens.

Usage:
  python scripts/train_v3.py --level L3 --epochs 20 --batch_size 8
  python scripts/train_v3.py --level gsm8k --warm_from checkpoints/v3_L3_best.pt
"""

import os
import sys
import re
import json
import time
import math
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_per_cycle import (
    PerCycleDataset, per_cycle_collate_fn,
    numerical_proximity_reward,
    build_number_token_weights,
)
from scripts.atom_lora import LoRAAtoms
from scripts.baked_llama import BakedLlama
from scripts.thinking_controller import ThinkingController, TreeNotebook, TreeNode


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(text):
    """Extract integer answer after #### or fallback to last number."""
    match = re.search(r'####\s*(-?\d+)', text)
    if match:
        val = int(match.group(1))
        return max(min(val, 999999999), -999999999)
    nums = re.findall(r'-?\d+', text)
    if nums:
        val = int(nums[-1])
        return max(min(val, 999999999), -999999999)
    return None


# ---------------------------------------------------------------------------
# Single cycle: think → soft tokens → Llama generates
# ---------------------------------------------------------------------------

def run_cycle(llama, controller, hidden_states_all, all_pages, notebook,
              cycle_num, gen_target_strs, available_targets, consumed_targets,
              device, teacher_force=True):
    """
    One breathing cycle:
    1. Controller thinks (inner loop, writes pages)
    2. Controller produces soft tokens
    3. Llama generates with soft tokens (differentiable)

    Returns dict with: gen_loss, reward, scales(soft_tokens), pages, predictions, etc.
    """
    batch_size = hidden_states_all[0].size(0)

    # 1. Controller thinks
    trunk_out, cycle_pages, energies = controller.think(
        hidden_states_all, all_pages, cycle_num=cycle_num,
        max_passes=3, energy_threshold=0.15,
    )

    # 2. Decisions
    action_logits, confidence, branch_embed = controller.decide(trunk_out)

    # 3. Soft tokens — controller's thinking projected into Llama's space
    soft_tokens = controller.make_soft_tokens(trunk_out, cycle_pages)  # trajectory → soft tokens

    # 4. Llama generates with soft tokens
    if teacher_force:
        # Teacher-forced: compute gen_loss (differentiable through soft tokens)
        target_ids = llama.tokenizer(
            gen_target_strs, return_tensors='pt', padding=True,
            truncation=True, max_length=60, add_special_tokens=False,
        ).input_ids.to(device)

        # Append EOS
        eos_col = torch.full((batch_size, 1), llama.tokenizer.eos_token_id,
                             dtype=torch.long, device=device)
        target_ids = torch.cat([target_ids, eos_col], dim=1)

        # Get logits through soft tokens (differentiable!)
        gen_logits = llama.generate_with_soft_tokens(
            soft_tokens, teacher_target_ids=target_ids,
        )  # (batch, target_len, vocab)

        # Per-sample generation loss with number weighting
        vocab_size = gen_logits.size(-1)
        loss_per_token = F.cross_entropy(
            gen_logits.reshape(-1, vocab_size),
            target_ids.reshape(-1),
            ignore_index=llama.tokenizer.pad_token_id,
            reduction='none',
            label_smoothing=0.05,
        ).view(batch_size, -1)

        num_weights = build_number_token_weights(
            target_ids, llama.tokenizer, number_weight=3.0,
        )
        target_mask = (target_ids != llama.tokenizer.pad_token_id).float()

        gen_loss_per_sample = (loss_per_token * target_mask * num_weights).sum(dim=1)
        gen_loss_per_sample = gen_loss_per_sample / (target_mask * num_weights).sum(dim=1).clamp(min=1)
    else:
        gen_logits = None
        gen_loss_per_sample = None

    # 5. Extract predictions and compute rewards
    with torch.no_grad():
        if gen_logits is not None:
            pred_tokens = gen_logits.argmax(dim=-1)
        else:
            # Free generation
            gen_ids = llama.generate_with_soft_tokens(soft_tokens, max_new_tokens=60)
            pred_tokens = gen_ids

        pred_vals = []
        rewards = torch.zeros(batch_size, device=device)

        for b in range(batch_size):
            text = llama.tokenizer.decode(pred_tokens[b], skip_special_tokens=True)
            pred_val = extract_answer(text)
            pred_vals.append(pred_val)

            if pred_val is None:
                rewards[b] = 0.0
            elif pred_val in consumed_targets[b]:
                rewards[b] = 0.0
            elif pred_val in available_targets[b]:
                rewards[b] = 1.0
                available_targets[b].remove(pred_val)
                consumed_targets[b].append(pred_val)
            else:
                best_prox = 0.0
                for t in available_targets[b]:
                    best_prox = max(best_prox, numerical_proximity_reward(
                        pred_val, t, base_correct=1.0, base_wrong=0.0))
                rewards[b] = best_prox

    return {
        'gen_loss_per_sample': gen_loss_per_sample,
        'rewards': rewards,
        'action_logits': action_logits,
        'confidence': confidence,
        'energies': energies,
        'cycle_pages': cycle_pages,
        'branch_embed': branch_embed,
        'pred_vals': pred_vals,
        'soft_tokens': soft_tokens,
    }


# ---------------------------------------------------------------------------
# Full forward: comprehend → N cycles
# ---------------------------------------------------------------------------

def forward_pass(llama, controller, batch, device, num_passes=3):
    """Full breathing loop. Returns losses and metrics."""
    problems = batch['problem']
    cycle_targets = batch['cycle_targets'].to(device)
    cycle_mask = batch['cycle_mask'].to(device)
    gen_targets = batch.get('cycle_gen_targets', None)
    max_steps = batch['max_steps']
    batch_size = cycle_targets.size(0)

    # Tokenize and encode problem ONCE (cached)
    inputs = llama.tokenizer(
        problems, return_tensors='pt', padding=True,
        truncation=True, max_length=192,
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    llama.reset_cache()
    hidden_states_all = llama.encode_problem(input_ids, attention_mask)

    # Target tracking
    available_targets = []
    consumed_targets = []
    for b in range(batch_size):
        n_steps = int(cycle_mask[b].sum().item())
        targets_b = [int(cycle_targets[b, i].item()) for i in range(n_steps)]
        available_targets.append(list(targets_b))
        consumed_targets.append([])

    notebook = TreeNotebook()
    all_pages = []
    total_gen_loss = torch.tensor(0.0, device=device)
    total_reinforce_loss = torch.tensor(0.0, device=device)
    total_energy_loss = torch.tensor(0.0, device=device)
    all_rewards = []
    valid_cycles = 0
    per_cycle_preds = []

    num_cycles = min(num_passes, max_steps)

    for cycle in range(num_cycles):
        mask_val = cycle_mask[:, cycle]
        if mask_val.sum() == 0:
            continue

        # Generation targets
        if gen_targets is not None:
            gen_strs = gen_targets[cycle]
        else:
            gen_strs = [str(cycle_targets[b, cycle].item()) for b in range(batch_size)]

        # Run cycle
        result = run_cycle(
            llama, controller, hidden_states_all, all_pages, notebook,
            cycle_num=cycle, gen_target_strs=gen_strs,
            available_targets=available_targets,
            consumed_targets=consumed_targets,
            device=device, teacher_force=True,
        )

        # Gen loss (differentiable through soft tokens → controller)
        if result['gen_loss_per_sample'] is not None:
            # Weight by reward: correct problems get full gen_loss gradient
            reward_weights = result['rewards'].clamp(min=0.3)  # floor at 0.3
            weighted_gen = (result['gen_loss_per_sample'] * reward_weights * mask_val).sum()
            weighted_gen = weighted_gen / mask_val.sum().clamp(min=1)
            total_gen_loss = total_gen_loss + weighted_gen

        # REINFORCE on action decisions
        with torch.no_grad():
            batch_reward = result['rewards'].mean()
            advantage = result['rewards'] - batch_reward  # (batch,)

        action_log_probs = F.log_softmax(result['action_logits'], dim=-1)
        # For now, all actions are SOLVE (action index 1)
        selected_action = 1
        reinforce_loss = -(advantage * action_log_probs[:, selected_action] * mask_val).mean()
        total_reinforce_loss = total_reinforce_loss + reinforce_loss

        # Energy calibration
        if result['energies']:
            final_energy = result['energies'][-1]  # (batch, 1)
            correct_mask = (result['rewards'] > 0.9).float().unsqueeze(-1)
            wrong_mask = (result['rewards'] < 0.1).float().unsqueeze(-1)
            # Correct → push energy down; Wrong → push energy up
            energy_loss = (correct_mask * final_energy).mean() + \
                          (wrong_mask * F.relu(0.7 - final_energy)).mean()
            total_energy_loss = total_energy_loss + energy_loss

        all_rewards.append(result['rewards'])
        valid_cycles += 1
        per_cycle_preds.append(result['pred_vals'])

        # Record pages and notebook
        for page in result['cycle_pages']:
            all_pages.append(page.detach())

        with torch.no_grad():
            node = TreeNode(
                page=result['cycle_pages'][-1][0].detach(),
                branch_embed=result['branch_embed'][0].detach(),
                action='solve',
                cycle_num=cycle,
            )
            notebook.append(node)

    # Average losses
    if valid_cycles > 0:
        total_gen_loss = total_gen_loss / valid_cycles
        total_reinforce_loss = total_reinforce_loss / valid_cycles
        total_energy_loss = total_energy_loss / valid_cycles

    avg_reward = torch.cat(all_rewards).mean().item() if all_rewards else 0.0

    metrics = {
        'gen_loss': total_gen_loss.item(),
        'reinforce_loss': total_reinforce_loss.item(),
        'energy_loss': total_energy_loss.item(),
        'avg_reward': avg_reward,
        'per_cycle_preds': per_cycle_preds,
    }

    # Combined loss (all differentiable through soft tokens except REINFORCE)
    total_loss = total_gen_loss + 0.1 * total_reinforce_loss + 0.5 * total_energy_loss

    return total_loss, metrics


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(llama, controller, eval_dataset, device, num_passes=3, max_length=192):
    """Evaluate via free generation."""
    controller.eval()
    eval_batch = 16
    per_cycle_correct = {}
    per_cycle_total = {}
    final_correct = 0
    total = 0
    max_steps_seen = 0
    all_soft_tokens = []

    with torch.no_grad():
        for i in range(0, len(eval_dataset), eval_batch):
            batch_samples = [eval_dataset[j] for j in range(i, min(i + eval_batch, len(eval_dataset)))]
            problems = [s['problem'] for s in batch_samples]
            cycle_targets_list = [s['cycle_targets'] for s in batch_samples]
            gold_finals = [s['final_answer'] for s in batch_samples]

            inputs = llama.tokenizer(
                problems, return_tensors='pt', padding=True,
                truncation=True, max_length=max_length,
            )
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            bs = input_ids.size(0)

            llama.reset_cache()
            hidden_states_all = llama.encode_problem(input_ids, attention_mask)

            all_pages = []
            eval_pred_vals = []

            for cycle in range(num_passes):
                trunk_out, cycle_pages, energies = controller.think(
                    hidden_states_all, all_pages, cycle_num=cycle,
                )
                soft_tokens = controller.make_soft_tokens(trunk_out)

                # Collect soft tokens for diversity (first batch, first cycle)
                if i == 0 and cycle == 0:
                    all_soft_tokens.append(soft_tokens.float().cpu())

                # Free generation
                gen_ids = llama.generate_with_soft_tokens(soft_tokens, max_new_tokens=60)

                cycle_preds = []
                for b in range(bs):
                    text = llama.tokenizer.decode(gen_ids[b], skip_special_tokens=True)
                    cycle_preds.append(extract_answer(text))
                eval_pred_vals.append(cycle_preds)

                # Score
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

                for page in cycle_pages:
                    all_pages.append(page)

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

    per_cycle_acc = []
    for p in range(max_steps_seen):
        if p in per_cycle_total and per_cycle_total[p] > 0:
            per_cycle_acc.append(100.0 * per_cycle_correct.get(p, 0) / per_cycle_total[p])
        else:
            per_cycle_acc.append(0.0)

    final_acc = 100.0 * final_correct / total if total > 0 else 0.0

    # Soft token diversity — use float64 for precision
    st_xproblem_cos = 1.0
    if all_soft_tokens:
        st_stack = torch.cat(all_soft_tokens, dim=0).double()  # (N, N_soft, 2048)
        st_flat = st_stack.view(st_stack.size(0), -1)  # (N, N_soft*2048)
        # Normalize for stable cosine
        st_normed = st_flat / st_flat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        n = st_normed.size(0)
        if n > 1:
            cos_matrix = st_normed @ st_normed.T  # (N, N)
            mask = ~torch.eye(n, dtype=torch.bool)
            st_xproblem_cos = cos_matrix[mask].mean().item()

    return per_cycle_acc, final_acc, st_xproblem_cos


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    print("=" * 60)
    print(f"Mycelium v3 Training — Level {args.level}")
    print("=" * 60)

    device = torch.device('cuda')

    # Build baked Llama
    llama = BakedLlama()
    llama = llama.to(device)

    # Bake L1 + L2 atoms
    if args.warm_from:
        print(f"\nLoading atoms from {args.warm_from}")
        ckpt = torch.load(args.warm_from, map_location=device)
        if 'atoms' in ckpt:
            l1 = LoRAAtoms()
            l1 = l1.to(device=device, dtype=torch.bfloat16)
            l1.load_state_dict(ckpt['atoms'])
            llama.bake_lora(l1, scale=0.46)
            print(f"  L1 atoms: baked (scale=0.46)")
            del l1
        if 'atoms2' in ckpt:
            l2 = LoRAAtoms()
            l2 = l2.to(device=device, dtype=torch.bfloat16)
            l2.load_state_dict(ckpt['atoms2'])
            llama.bake_lora(l2, scale=0.46)
            print(f"  L2 atoms: baked (universal blend, scale=0.46)")
            del l2
    else:
        print("WARNING: No --warm_from specified. Llama has no baked math mode.")

    # Build thinking controller
    controller = ThinkingController(hidden_dim=llama.d_model, num_llama_layers=llama.num_layers)
    controller = controller.to(device=device, dtype=torch.bfloat16)

    # Load controller weights if available
    if args.warm_from and 'thinking_controller' in ckpt:
        controller.load_state_dict(ckpt['thinking_controller'])
        print(f"  controller: loaded")
    else:
        print(f"  controller: FRESH INIT (322M)")

    ctrl_params = list(controller.parameters())
    trainable = sum(p.numel() for p in ctrl_params)
    print(f"\nTrainable params: {trainable:,} (controller only)")
    print(f"GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Data
    level = args.level
    train_path = os.path.join(args.data_dir, f'{level}_train.jsonl')
    eval_path = os.path.join(args.data_dir, f'{level}_eval.jsonl')

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

    # Single optimizer — only the controller trains
    optimizer = torch.optim.AdamW(ctrl_params, lr=args.lr, weight_decay=0.01)

    # Baseline eval
    best_final = 0.0
    if eval_dataset:
        per_cycle_acc, final_acc, st_cos = evaluate(
            llama, controller, eval_dataset, device, num_passes=args.num_passes,
        )
        print(f"\nBaseline: final={final_acc:.1f}%  per_cycle={[f'{a:.1f}%' for a in per_cycle_acc]}")
        print(f"  soft_token_xproblem_cos={st_cos:.4f}")
        best_final = final_acc

    # Training
    patience_counter = 0

    for epoch in range(args.epochs):
        controller.train()
        train_dataset.set_epoch(epoch)
        t0 = time.time()

        ep_gen = 0.0
        ep_rl = 0.0
        ep_energy = 0.0
        ep_reward = 0.0
        nb = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            optimizer.zero_grad()

            total_loss, metrics = forward_pass(
                llama, controller, batch, device, num_passes=args.num_passes,
            )

            # Single backward — gen_loss flows through soft tokens to controller
            if total_loss.requires_grad:
                total_loss.backward()

            torch.nn.utils.clip_grad_norm_(ctrl_params, 1.0)
            optimizer.step()

            ep_gen += metrics['gen_loss']
            ep_rl += metrics['reinforce_loss']
            ep_energy += metrics['energy_loss']
            ep_reward += metrics['avg_reward']
            nb += 1

        elapsed = time.time() - t0

        # Gradient norms
        ctrl_grad = sum(p.grad.norm().item() for p in ctrl_params if p.grad is not None)
        ctrl_count = sum(1 for p in ctrl_params if p.grad is not None)
        ctrl_grad_avg = ctrl_grad / max(ctrl_count, 1)

        # Eval
        if eval_dataset:
            per_cycle_acc, final_acc, st_cos = evaluate(
                llama, controller, eval_dataset, device, num_passes=args.num_passes,
            )

            improved = final_acc > best_final
            if improved:
                best_final = final_acc
                patience_counter = 0
                ckpt_name = f'checkpoints/v3_{level}_best.pt'
                torch.save({
                    'epoch': epoch + 1,
                    'thinking_controller': controller.state_dict(),
                    'accuracy': final_acc,
                    'level': level,
                }, ckpt_name)
                print(f"  -> saved {ckpt_name} (final={final_acc:.1f}%)")
            else:
                patience_counter += 1

            print(
                f"Epoch {epoch+1}: gen={ep_gen/nb:.4f} rl={ep_rl/nb:.4f} "
                f"energy={ep_energy/nb:.4f} reward={ep_reward/nb:.3f} | "
                f"Final={final_acc:.1f}% best={best_final:.1f}% [{elapsed:.0f}s]"
            )
            print(f"  per_cycle_acc: {[f'{a:.1f}%' for a in per_cycle_acc]}")
            print(f"  ctrl_grad={ctrl_grad_avg:.4f} soft_token_cos={st_cos:.4f}")

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
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_passes', type=int, default=3)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--warm_from', type=str, default=None)
    p.add_argument('--augment', action='store_true')
    args = p.parse_args()
    train(args)
