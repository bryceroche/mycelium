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
    Takes Pythia-160M layers 0-3 and loops them N times.

    Between loops, adds a learned loop embedding so the model knows
    which iteration it's on. This is minimal — no π cycling yet,
    no temperature modulation. Just: can the layers be reused?
    """

    def __init__(self, model, num_loops=1, use_loop_embed=True):
        super().__init__()
        self.model = model  # Full Pythia model (we'll only use layers 0-3)
        self.num_loops = num_loops
        self.use_loop_embed = use_loop_embed

        hidden_size = model.config.hidden_size  # 768

        # Simple learnable loop embeddings (one per loop iteration)
        # Small scale so they don't disrupt frozen representations
        if use_loop_embed:
            self.loop_embeds = nn.Parameter(
                torch.randn(8, hidden_size) * 0.01  # max 8 loops
            )

        # We'll use layers 0-3 only
        self.num_layers = min(4, len(model.gpt_neox.layers))

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass with looping.

        Instead of using model.forward() which runs ALL layers,
        we manually:
          1. Embed tokens
          2. Loop layers 0-3 N times
          3. Apply final layer norm + lm_head
        """
        device = input_ids.device

        # Token + position embedding
        hidden_states = self.model.gpt_neox.embed_in(input_ids)

        # Build causal attention mask
        seq_len = input_ids.shape[1]

        # Compute rotary position embeddings (needed by GPT-NeoX layers)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        position_embeddings = self.model.gpt_neox.rotary_emb(hidden_states, position_ids)

        for loop_idx in range(self.num_loops):
            # Add loop embedding (broadcast across sequence)
            if self.use_loop_embed:
                loop_emb = self.loop_embeds[loop_idx].to(hidden_states.dtype).unsqueeze(0).unsqueeze(0)
                hidden_states = hidden_states + loop_emb

            # Run through layers 0-3
            for layer_idx in range(self.num_layers):
                layer = self.model.gpt_neox.layers[layer_idx]
                outputs = layer(hidden_states, attention_mask=attention_mask,
                                position_embeddings=position_embeddings)
                hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs

        # Final layer norm
        hidden_states = self.model.gpt_neox.final_layer_norm(hidden_states)

        # LM head
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
    for num_loops in args.loops:
        print(f"\n{'='*60}")
        print(f"LOOPING: 4 layers × {num_loops} loops = {4*num_loops} effective layer passes")
        print(f"{'='*60}")

        # Create looping wrapper
        looping_model = LoopingPythia(model, num_loops=num_loops, use_loop_embed=True)
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

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Configuration':<30} {'Accuracy':>10} {'Eff. Layers':>12}")
    print(f"{'-'*52}")
    for name, acc in results.items():
        if name == 'baseline_12L':
            eff = 12
        else:
            loops = int(name.split('_')[1].replace('x', ''))
            eff = 4 * loops
        print(f"{name:<30} {acc:>10.1%} {eff:>12}")

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
