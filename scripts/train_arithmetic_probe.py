#!/usr/bin/env python3
"""
Train State-Conditioned LoRA with SymPy Probe.

The probe head gives per-pass gradient: "your state should encode THIS value now."
SymPy automatically computes intermediate targets from the expression.

For "(48 / 2) + 48 =":
  Pass 1 target: 24 (result of first operation)
  Pass 2 target: 72 (final answer)

This breaks the 70% × 70% = 49% bottleneck by providing dense supervision.
"""

import random
import math
import re
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import argparse
import sys
sys.path.insert(0, '/home/ubuntu/mycelium')

from src.thinking_model import ThinkingModel


class ProbeHead(nn.Module):
    """
    Tiny probe that predicts intermediate value from state.
    Linear(64, 1) -> scalar prediction.
    """
    def __init__(self, state_size=64):
        super().__init__()
        self.fc = nn.Linear(state_size, 1)
    
    def forward(self, state):
        # state: (batch, 64)
        return self.fc(state).squeeze(-1)  # (batch,)


def compute_intermediate_targets(problem, answer, intermediate):
    """
    Compute target values for each thinking pass.
    
    For two-step arithmetic:
      Pass 1: intermediate result (e.g., 48/2 = 24)
      Pass 2: final answer (e.g., 24 + 48 = 72)
    
    We normalize targets to [-1, 1] range for stable training.
    """
    # For 2 passes: [intermediate, final]
    int_val = float(intermediate)
    final_val = float(answer)
    
    # Normalize: divide by 1000 to get roughly [-1, 1] range
    # Most arithmetic results are < 1000
    targets = [int_val / 1000.0, final_val / 1000.0]
    
    return targets


def generate_two_step_problem():
    """Generate a two-step arithmetic problem with solution."""
    ops = ['+', '-', '*', '/']
    
    # First operation
    op1 = random.choice(ops)
    if op1 == '/':
        b = random.randint(2, 10)
        a = b * random.randint(2, 12)
    elif op1 == '*':
        a = random.randint(2, 12)
        b = random.randint(2, 10)
    else:
        a = random.randint(10, 100)
        b = random.randint(1, min(a-1, 50)) if op1 == '-' else random.randint(1, 50)
    
    # Compute intermediate
    if op1 == '+':
        intermediate = a + b
    elif op1 == '-':
        intermediate = a - b
    elif op1 == '*':
        intermediate = a * b
    else:
        intermediate = a // b
    
    # Second operation
    op2 = random.choice(ops)
    if op2 == '/':
        divisors = [d for d in range(2, 13) if intermediate % d == 0 and intermediate // d > 0]
        if not divisors:
            c = 1
        else:
            c = random.choice(divisors)
    elif op2 == '*':
        c = random.randint(2, 10)
    elif op2 == '-':
        c = random.randint(1, max(1, intermediate - 1))
    else:
        c = random.randint(1, 50)
    
    # Compute final
    if op2 == '+':
        result = intermediate + c
    elif op2 == '-':
        result = intermediate - c
    elif op2 == '*':
        result = intermediate * c
    else:
        result = intermediate // c if c != 0 else intermediate
    
    problem = f"({a} {op1} {b}) {op2} {c} ="
    answer = str(result)
    
    return problem, answer, intermediate


def generate_dataset(n_samples, seed=42):
    """Generate n two-step arithmetic problems."""
    random.seed(seed)
    data = []
    for _ in range(n_samples):
        problem, answer, intermediate = generate_two_step_problem()
        data.append({
            "problem": problem,
            "answer": answer,
            "intermediate": intermediate,
        })
    return data


def compute_loss(model, probe, problem, gold_answer, intermediate, num_passes, device, probe_weight=0.5):
    """
    Compute loss with probe supervision at each pass.
    
    Total loss = answer_loss + probe_weight * probe_loss
    
    probe_loss is MSE between probe prediction and SymPy-computed intermediate.
    """
    # Get intermediate targets
    targets = compute_intermediate_targets(problem, gold_answer, intermediate)
    targets = torch.tensor(targets, device=device, dtype=torch.float32)
    
    # Tokenize
    prompt_ids = model.tokenizer(
        problem,
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids.to(device)
    
    answer_text = f" {gold_answer}"
    answer_ids = model.tokenizer(
        answer_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)
    
    full_ids = torch.cat([prompt_ids, answer_ids], dim=1)
    prompt_len = prompt_ids.shape[1]
    
    # Initialize state
    state = torch.zeros(1, model.state_size, device=device)
    state = state / (state.norm(dim=-1, keepdim=True) + 1e-8) * model.state_radius
    
    answer_loss = 0.0
    probe_loss = 0.0
    
    for pass_idx in range(num_passes):
        # Forward with LoRA
        model.apply_lora(state)
        outputs = model.transformer(
            input_ids=full_ids,
            output_hidden_states=True,
        )
        model.remove_lora()
        
        # Compress to state delta
        hidden_states = outputs.hidden_states[1:]
        delta = model.compressor(list(hidden_states), pass_num=pass_idx)
        
        # Update state
        state = state + delta
        state = state / (state.norm(dim=-1, keepdim=True) + 1e-8) * model.state_radius
        
        # PROBE LOSS: Does state encode the right intermediate?
        predicted_value = probe(state)  # (1,)
        target_value = targets[pass_idx]
        probe_loss = probe_loss + nn.functional.mse_loss(predicted_value, target_value.unsqueeze(0))
        
        # Answer loss (deep supervision)
        logits = outputs.logits
        answer_logits = logits[:, prompt_len-1:-1, :]
        answer_targets = full_ids[:, prompt_len:]
        
        loss = nn.functional.cross_entropy(
            answer_logits.reshape(-1, answer_logits.size(-1)),
            answer_targets.reshape(-1),
        )
        
        weight = (pass_idx + 1) / num_passes
        answer_loss = answer_loss + weight * loss
    
    # Normalize answer loss
    weight_sum = sum((i + 1) / num_passes for i in range(num_passes))
    answer_loss = answer_loss / weight_sum
    
    # Normalize probe loss
    probe_loss = probe_loss / num_passes
    
    # Total loss
    total_loss = answer_loss + probe_weight * probe_loss
    
    return total_loss, answer_loss.item(), probe_loss.item()


def evaluate(model, probe, eval_data, num_passes, device):
    """Evaluate accuracy and probe accuracy."""
    model.eval()
    probe.eval()
    correct = 0
    total = 0
    probe_errors = []
    
    with torch.no_grad():
        for sample in eval_data:
            problem = sample["problem"]
            gold = sample["answer"]
            intermediate = sample["intermediate"]
            
            targets = compute_intermediate_targets(problem, gold, intermediate)
            
            state = torch.zeros(1, model.state_size, device=device)
            state = state / (state.norm(dim=-1, keepdim=True) + 1e-8) * model.state_radius
            
            prompt_ids = model.tokenizer(
                problem,
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids.to(device)
            
            for pass_idx in range(num_passes):
                model.apply_lora(state)
                outputs = model.transformer(
                    input_ids=prompt_ids,
                    output_hidden_states=True,
                )
                model.remove_lora()
                
                hidden_states = outputs.hidden_states[1:]
                delta = model.compressor(list(hidden_states), pass_num=pass_idx)
                state = state + delta
                state = state / (state.norm(dim=-1, keepdim=True) + 1e-8) * model.state_radius
                
                # Check probe accuracy
                pred_val = probe(state).item()
                target_val = targets[pass_idx]
                error = abs(pred_val - target_val) * 1000  # Back to original scale
                probe_errors.append(error)
            
            # Generate answer
            model.apply_lora(state)
            gen_ids = model.transformer.generate(
                input_ids=prompt_ids,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=model.tokenizer.eos_token_id,
            )
            model.remove_lora()
            
            output = model.tokenizer.decode(gen_ids[0][prompt_ids.shape[1]:], skip_special_tokens=True).strip()
            pred = output.split()[0] if output else ""
            pred = pred.rstrip('.,;:')
            
            if pred == gold:
                correct += 1
            total += 1
    
    model.train()
    probe.train()
    
    acc = correct / total if total > 0 else 0.0
    avg_probe_error = sum(probe_errors) / len(probe_errors) if probe_errors else 0.0
    
    return acc, avg_probe_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-samples", type=int, default=1000)
    parser.add_argument("--eval-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-passes", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--template-lr", type=float, default=1e-3)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--probe-weight", type=float, default=0.5)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Probe weight: {args.probe_weight}")
    
    # Generate data
    print(f"\nGenerating {args.train_samples} training problems...")
    train_data = generate_dataset(args.train_samples, seed=42)
    print(f"Generating {args.eval_samples} eval problems...")
    eval_data = generate_dataset(args.eval_samples, seed=123)
    
    print("\nExample problems with intermediate targets:")
    for i in range(5):
        s = train_data[i]
        targets = compute_intermediate_targets(s['problem'], s['answer'], s['intermediate'])
        print(f"  {s['problem']} -> {s['answer']}")
        print(f"    Pass 1 target: {targets[0]*1000:.0f} (intermediate)")
        print(f"    Pass 2 target: {targets[1]*1000:.0f} (final)")
    
    # Create model
    print("\nLoading ThinkingModel with Llama-3.2-1B BASE...")
    model = ThinkingModel(
        model_name="meta-llama/Llama-3.2-1B",
        state_size=64,
        lora_rank=4,
        num_queries=4,
        num_perceiver_layers=7,
    )
    model.to(device)
    
    # Create probe head
    probe = ProbeHead(state_size=64).to(device)
    print(f"Probe head parameters: {sum(p.numel() for p in probe.parameters()):,}")
    
    # Freeze transformer
    for param in model.transformer.parameters():
        param.requires_grad = False
    
    # Parameter groups
    perceiver_params = list(model.compressor.parameters()) + list(model.confidence.parameters())
    template_params = list(model.lora.A_templates) + list(model.lora.B_templates)
    hyper_params = list(model.lora.state_to_scales.parameters())
    probe_params = list(probe.parameters())
    
    optimizer = AdamW([
        {"params": perceiver_params, "lr": args.lr},
        {"params": template_params, "lr": args.template_lr},
        {"params": hyper_params, "lr": args.template_lr},
        {"params": probe_params, "lr": args.probe_lr},
    ])
    
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_data) // args.batch_size)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable += sum(p.numel() for p in probe.parameters())
    print(f"Trainable parameters: {trainable:,}")
    
    # Evaluate baseline
    print("\nEvaluating baseline...")
    baseline_acc, baseline_probe_err = evaluate(model, probe, eval_data[:50], args.num_passes, device)
    print(f"Baseline: accuracy={baseline_acc*100:.1f}%, probe_error={baseline_probe_err:.1f}")
    
    # Training loop
    print(f"\nStarting training: {args.epochs} epochs, {args.num_passes} passes, probe_weight={args.probe_weight}")
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        probe.train()
        random.shuffle(train_data)
        
        epoch_loss = 0.0
        epoch_answer_loss = 0.0
        epoch_probe_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(range(0, len(train_data), args.batch_size), desc=f"Epoch {epoch+1}/{args.epochs}")
        for i in pbar:
            batch = train_data[i:i+args.batch_size]
            
            optimizer.zero_grad()
            batch_loss = 0.0
            batch_answer = 0.0
            batch_probe = 0.0
            
            for sample in batch:
                try:
                    loss, ans_loss, prb_loss = compute_loss(
                        model, probe,
                        sample["problem"], 
                        sample["answer"],
                        sample["intermediate"],
                        args.num_passes,
                        device,
                        args.probe_weight,
                    )
                    batch_loss = batch_loss + loss
                    batch_answer += ans_loss
                    batch_probe += prb_loss
                except Exception as e:
                    print(f"Error: {e}")
                    continue
            
            if batch_loss > 0:
                batch_loss = batch_loss / len(batch)
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(probe.parameters()), 1.0)
                optimizer.step()
                scheduler.step()
                
                epoch_loss += batch_loss.item()
                epoch_answer_loss += batch_answer / len(batch)
                epoch_probe_loss += batch_probe / len(batch)
                num_batches += 1
            
            pbar.set_postfix(loss=f"{batch_loss.item():.4f}" if isinstance(batch_loss, torch.Tensor) else "N/A")
        
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_answer = epoch_answer_loss / max(num_batches, 1)
        avg_probe = epoch_probe_loss / max(num_batches, 1)
        
        # Evaluate
        acc, probe_err = evaluate(model, probe, eval_data, args.num_passes, device)
        
        print(f"\nEpoch {epoch+1}: loss={avg_loss:.4f} (ans={avg_answer:.4f}, probe={avg_probe:.4f})")
        print(f"          accuracy={acc*100:.1f}%, probe_error={probe_err:.1f}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "probe_state_dict": probe.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "accuracy": acc,
            }, "checkpoints/arithmetic_probe_best.pt")
            print(f"  New best! Saved checkpoint.")
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best accuracy: {best_acc*100:.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
