"""
Day 1 Validation: Do Pythia-160M layers 0-3 improve with looping?

Core question: When we cycle the same 4 layers multiple times with
different loop embeddings, does accuracy increase (more thinking helps)
or degrade (representation corrupts)?

Setup:
  - Pythia-160M layers 0-3 (frozen, no training)
  - Simple loop embedding added to residual stream between loops
  - L3 problems (single-step arithmetic)
  - Generate "equation #### answer" via greedy decoding
  - Measure: accuracy @ 1, 2, 4, 8 loops

This is a SMOKE TEST. We don't expect high accuracy from a frozen model
on math — we're measuring whether MORE loops help or hurt. If accuracy
monotonically decreases with loops, looping is broken. If it stays flat
or increases, the architecture is viable.

Usage:
  python scripts/validate_looping.py --loops 1 2 4 8 --num_problems 200
"""

import argparse
import json
import os
import re
import sys
import time
import random

import torch
import torch.nn as nn
from transformers import AutoTokenizer, GPTNeoXForCausalLM

# Add project root for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.generate_per_cycle_data import L3_GENERATORS


# ---------------------------------------------------------------------------
# Data generation (inline, no file dependency)
# ---------------------------------------------------------------------------

def generate_l3_problems(num_problems, seed=42):
    """Generate L3 problems with answers."""
    rng = random.Random(seed)
    problems = []
    for _ in range(num_problems):
        gen_fn = rng.choice(L3_GENERATORS)
        problem, cycle_targets, final_answer, cycle_gen_targets = gen_fn(rng)
        problems.append({
            'problem': problem,
            'final_answer': final_answer,
            'gen_target': cycle_gen_targets[0],
        })
    return problems


# ---------------------------------------------------------------------------
# Looping model wrapper
# ---------------------------------------------------------------------------

class LoopingPythia(nn.Module):
    """
    Takes Pythia layers and loops them N times.

    Supports:
    - Layer selection (start_layer, num_layers)
    - Optional RMSNorm between loops to stabilize representations
    - Loop embeddings so the model knows which iteration it's on
    """

    def __init__(self, model, num_loops=1, use_loop_embed=True,
                 start_layer=0, num_layers=4, use_inter_loop_norm=False):
        super().__init__()
        self.model = model
        self.num_loops = num_loops
        self.use_loop_embed = use_loop_embed
        self.use_inter_loop_norm = use_inter_loop_norm
        self.start_layer = start_layer

        hidden_size = model.config.hidden_size

        if use_loop_embed:
            self.loop_embeds = nn.Parameter(
                torch.randn(8, hidden_size) * 0.01
            )

        self.num_layers = min(num_layers, len(model.gpt_neox.layers) - start_layer)

        # RMSNorm between loops to force representation back into stable range
        if use_inter_loop_norm:
            self.inter_loop_norm = nn.RMSNorm(hidden_size)

    def forward(self, input_ids, attention_mask=None):
        device = input_ids.device

        hidden_states = self.model.gpt_neox.embed_in(input_ids)
        seq_len = input_ids.shape[1]

        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        position_embeddings = self.model.gpt_neox.rotary_emb(hidden_states, position_ids)

        for loop_idx in range(self.num_loops):
            if self.use_loop_embed:
                loop_emb = self.loop_embeds[loop_idx].to(hidden_states.dtype).unsqueeze(0).unsqueeze(0)
                hidden_states = hidden_states + loop_emb

            for layer_idx in range(self.num_layers):
                layer = self.model.gpt_neox.layers[self.start_layer + layer_idx]
                outputs = layer(hidden_states, attention_mask=attention_mask,
                                position_embeddings=position_embeddings)
                hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs

            # Normalize between loops to prevent representation explosion/collapse
            if self.use_inter_loop_norm and loop_idx < self.num_loops - 1:
                hidden_states = self.inter_loop_norm(hidden_states.float()).to(hidden_states.dtype)

        hidden_states = self.model.gpt_neox.final_layer_norm(hidden_states)
        logits = self.model.embed_out(hidden_states)

        return logits

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=60):
        """Simple greedy generation."""
        device = input_ids.device
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # Only feed last 512 tokens to avoid OOM
            context = generated[:, -512:]
            logits = self.forward(context)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            # Stop on EOS
            if next_token.item() == self.model.config.eos_token_id:
                break

        return generated


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(text):
    """Extract number after #### marker, or last number in text."""
    match = re.search(r'####\s*([-]?\d+)', text)
    if match:
        return int(match.group(1))
    # Fallback: last number
    numbers = re.findall(r'[-]?\d+', text)
    return int(numbers[-1]) if numbers else None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_looping(model_wrapper, tokenizer, problems, device, verbose=False):
    """Run evaluation with the looping model."""
    correct = 0
    total = len(problems)
    format_correct = 0  # Has #### marker

    for i, prob in enumerate(problems):
        # Format input prompt
        prompt = f"Solve: {prob['problem']}\nAnswer: "
        inputs = tokenizer(prompt, return_tensors='pt').to(device)

        # Generate
        with torch.no_grad():
            output_ids = model_wrapper.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=60,
            )

        # Decode (skip prompt tokens)
        generated_text = tokenizer.decode(
            output_ids[0, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        # Extract answer
        predicted = extract_answer(generated_text)
        expected = prob['final_answer']

        if '####' in generated_text:
            format_correct += 1

        if predicted == expected:
            correct += 1

        if verbose and i < 5:
            print(f"  Problem: {prob['problem'][:60]}...")
            print(f"  Generated: {generated_text[:80]}")
            print(f"  Predicted={predicted}, Expected={expected}, {'✓' if predicted == expected else '✗'}")
            print()

    accuracy = correct / total if total > 0 else 0
    format_rate = format_correct / total if total > 0 else 0
    return accuracy, format_rate


# ---------------------------------------------------------------------------
# Baseline: run all 12 layers once (standard Pythia forward pass)
# ---------------------------------------------------------------------------

def evaluate_baseline(model, tokenizer, problems, device, verbose=False):
    """Standard single-pass through all 12 layers as baseline."""
    correct = 0
    total = len(problems)

    for i, prob in enumerate(problems):
        prompt = f"Solve: {prob['problem']}\nAnswer: "
        inputs = tokenizer(prompt, return_tensors='pt').to(device)

        with torch.no_grad():
            output_ids = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=60,
                do_sample=False,
            )

        generated_text = tokenizer.decode(
            output_ids[0, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        predicted = extract_answer(generated_text)
        expected = prob['final_answer']

        if predicted == expected:
            correct += 1

        if verbose and i < 5:
            print(f"  Problem: {prob['problem'][:60]}...")
            print(f"  Generated: {generated_text[:80]}")
            print(f"  Predicted={predicted}, Expected={expected}, {'✓' if predicted == expected else '✗'}")
            print()

    return correct / total if total > 0 else 0


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate looping on Pythia-160M")
    parser.add_argument('--loops', type=int, nargs='+', default=[1, 2, 4, 8],
                        help='Loop counts to test')
    parser.add_argument('--num_problems', type=int, default=200,
                        help='Number of L3 problems')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto/cpu/cuda)')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--no_baseline', action='store_true',
                        help='Skip full-model baseline')
    parser.add_argument('--few_shot', action='store_true',
                        help='Use few-shot prompting')
    parser.add_argument('--with_norm', action='store_true',
                        help='Also test RMSNorm between loops')
    parser.add_argument('--with_70m', action='store_true',
                        help='Also test Pythia-70M full-model looping')
    parser.add_argument('--start_layer', type=int, default=0,
                        help='First layer to include in loop (default: 0)')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of layers per loop (default: 4)')
    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load Pythia-160M
    print("Loading Pythia-160M...")
    model_name = "EleutherAI/pythia-160m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = GPTNeoXForCausalLM.from_pretrained(model_name)
    model = model.to(device).eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Layers: {len(model.gpt_neox.layers)}")
    print(f"  Hidden dim: {model.config.hidden_size}")
    print(f"  Heads: {model.config.num_attention_heads}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Generate problems
    print(f"\nGenerating {args.num_problems} L3 problems...")
    problems = generate_l3_problems(args.num_problems, seed=args.seed)
    print(f"  Sample: {problems[0]['problem']}")
    print(f"  Answer: {problems[0]['final_answer']}")

    # Optional few-shot prefix
    if args.few_shot:
        # We'll prepend exemplars to each prompt in evaluate_looping
        # For now, just note it
        print("  Mode: few-shot (3 exemplars)")

    results = {}

    # Baseline: full model (all 12 layers, single pass)
    if not args.no_baseline:
        print(f"\n{'='*60}")
        print(f"BASELINE: Full Pythia-160M (12 layers, 1 pass)")
        print(f"{'='*60}")
        t0 = time.time()
        baseline_acc = evaluate_baseline(model, tokenizer, problems, device, args.verbose)
        elapsed = time.time() - t0
        results['baseline_12L'] = baseline_acc
        print(f"  Accuracy: {baseline_acc:.1%}")
        print(f"  Time: {elapsed:.1f}s ({elapsed/len(problems)*1000:.0f}ms/problem)")

    # Looping experiments
    sl = args.start_layer
    nl = args.num_layers
    for num_loops in args.loops:
        print(f"\n{'='*60}")
        print(f"LOOPING: layers {sl}-{sl+nl-1} × {num_loops} loops = {nl*num_loops} effective layer passes")
        print(f"{'='*60}")

        looping_model = LoopingPythia(model, num_loops=num_loops, use_loop_embed=True,
                                       start_layer=sl, num_layers=nl)
        looping_model = looping_model.to(device).eval()

        t0 = time.time()
        accuracy, format_rate = evaluate_looping(
            looping_model, tokenizer, problems, device, args.verbose
        )
        elapsed = time.time() - t0

        results[f'loop_{num_loops}x'] = accuracy
        print(f"  Accuracy: {accuracy:.1%}")
        print(f"  Format (has ####): {format_rate:.1%}")
        print(f"  Time: {elapsed:.1f}s ({elapsed/len(problems)*1000:.0f}ms/problem)")

    # RMSNorm between loops experiment
    if args.with_norm:
        for num_loops in args.loops:
            if num_loops < 2:
                continue  # Norm only matters with 2+ loops
            print(f"\n{'='*60}")
            print(f"LOOPING + RMSNORM: layers {sl}-{sl+nl-1} × {num_loops} loops")
            print(f"{'='*60}")

            looping_model = LoopingPythia(model, num_loops=num_loops, use_loop_embed=True,
                                           start_layer=sl, num_layers=nl, use_inter_loop_norm=True)
            looping_model = looping_model.to(device).eval()

            t0 = time.time()
            accuracy, format_rate = evaluate_looping(
                looping_model, tokenizer, problems, device, args.verbose
            )
            elapsed = time.time() - t0

            results[f'norm_loop_{num_loops}x'] = accuracy
            print(f"  Accuracy: {accuracy:.1%}")
            print(f"  Format (has ####): {format_rate:.1%}")
            print(f"  Time: {elapsed:.1f}s ({elapsed/len(problems)*1000:.0f}ms/problem)")

    # Pythia-70M full-model looping experiment
    if args.with_70m:
        print(f"\n{'='*60}")
        print(f"LOADING Pythia-70M (6 layers — full model looping)")
        print(f"{'='*60}")
        model_70m = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-70m")
        model_70m = model_70m.to(device).eval()
        tok_70m = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
        if tok_70m.pad_token is None:
            tok_70m.pad_token = tok_70m.eos_token
        n_layers_70m = len(model_70m.gpt_neox.layers)
        print(f"  Layers: {n_layers_70m}, Hidden: {model_70m.config.hidden_size}")

        # Baseline: full 70M single pass
        print(f"\n{'='*60}")
        print(f"BASELINE: Full Pythia-70M ({n_layers_70m} layers, 1 pass)")
        print(f"{'='*60}")
        t0 = time.time()
        baseline_70m = evaluate_baseline(model_70m, tok_70m, problems, device, args.verbose)
        elapsed = time.time() - t0
        results['baseline_70m'] = baseline_70m
        print(f"  Accuracy: {baseline_70m:.1%}")
        print(f"  Time: {elapsed:.1f}s ({elapsed/len(problems)*1000:.0f}ms/problem)")

        for num_loops in [1, 2, 4]:
            print(f"\n{'='*60}")
            print(f"70M FULL MODEL × {num_loops} loops = {n_layers_70m*num_loops} effective layer passes")
            print(f"{'='*60}")

            looping_70m = LoopingPythia(model_70m, num_loops=num_loops, use_loop_embed=True,
                                         start_layer=0, num_layers=n_layers_70m,
                                         use_inter_loop_norm=True)
            looping_70m = looping_70m.to(device).eval()

            t0 = time.time()
            accuracy, format_rate = evaluate_looping(
                looping_70m, tok_70m, problems, device, args.verbose
            )
            elapsed = time.time() - t0

            results[f'70m_loop_{num_loops}x'] = accuracy
            print(f"  Accuracy: {accuracy:.1%}")
            print(f"  Format (has ####): {format_rate:.1%}")
            print(f"  Time: {elapsed:.1f}s ({elapsed/len(problems)*1000:.0f}ms/problem)")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Configuration':<40} {'Accuracy':>10}")
    print(f"{'-'*50}")
    for name, acc in results.items():
        print(f"{name:<40} {acc:>10.1%}")

    # Key diagnostic: is more loops helping?
    loop_accs = [(k, v) for k, v in results.items() if k.startswith('loop_')]
    if len(loop_accs) >= 2:
        first_acc = loop_accs[0][1]
        last_acc = loop_accs[-1][1]
        trend = last_acc - first_acc
        print(f"\n{'='*60}")
        print("VERDICT")
        print(f"{'='*60}")
        if trend > 0.02:
            print(f"  ✓ MORE LOOPS HELPS (+{trend:.1%})")
            print(f"  → Looping architecture is VIABLE. Proceed with π cycling.")
        elif trend < -0.02:
            print(f"  ✗ MORE LOOPS HURTS ({trend:.1%})")
            print(f"  → Representation degrades with reuse. Need revision.")
            print(f"  → Options: more layers per breath, layer norm between loops,")
            print(f"    different normalization, or abandon looping.")
        else:
            print(f"  ~ LOOPS NEUTRAL ({trend:+.1%})")
            print(f"  → Not helping but not hurting. π cycling may differentiate.")
            print(f"  → This is a WEAK positive — layers tolerate reuse.")

    # Save results
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'data', 'looping_validation_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({
            'results': results,
            'config': {
                'loops': args.loops,
                'num_problems': args.num_problems,
                'seed': args.seed,
                'device': str(device),
            }
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
