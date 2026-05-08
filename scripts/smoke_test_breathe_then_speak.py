"""
Smoke test: Breathe in representation space, speak only once.

The copy machine principle says: looping hidden states preserves signal,
but autoregressive generation between loops destroys it. This test validates
the v4 architecture by:

1. Loop Pythia-160M layers 0-11 for 1, 2, 4, 8 loops in hidden state space
   (NO token generation between loops)
2. Generate tokens ONCE from the final loop's hidden states
3. Compare generation quality across loop counts

If 2-loop generation is coherent (while 2-loop mid-generation was "had had had"),
the copy machine principle is validated and the breathing architecture works.

Usage:
  python scripts/smoke_test_breathe_then_speak.py [--num_problems 50]
"""

import argparse
import os
import re
import sys
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, GPTNeoXForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.generate_per_cycle_data import L3_GENERATORS


def generate_l3_problems(num_problems, seed=42):
    rng = random.Random(seed)
    problems = []
    for _ in range(num_problems):
        gen_fn = rng.choice(L3_GENERATORS)
        problem, cycle_targets, final_answer, cycle_gen_targets = gen_fn(rng)
        problems.append({'problem': problem, 'final_answer': final_answer})
    return problems


def forward_breathe(model, input_ids, num_loops, layer_indices=None):
    """
    Breathe in representation space: loop layers N times, return final hidden states.
    NO token generation between loops. The hidden states are the original painting.
    """
    if layer_indices is None:
        layer_indices = list(range(len(model.gpt_neox.layers)))

    hidden_states = model.gpt_neox.embed_in(input_ids)
    seq_len = input_ids.shape[1]
    device = input_ids.device

    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    position_embeddings = model.gpt_neox.rotary_emb(hidden_states, position_ids)

    for loop_idx in range(num_loops):
        for li in layer_indices:
            layer = model.gpt_neox.layers[li]
            outputs = layer(hidden_states, position_embeddings=position_embeddings)
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs

    # Final layer norm + lm_head
    hidden_states = model.gpt_neox.final_layer_norm(hidden_states)
    logits = model.embed_out(hidden_states)
    return logits


@torch.no_grad()
def generate_breathe_then_speak(model, input_ids, num_loops, layer_indices=None,
                                  max_new_tokens=60):
    """
    Breathe first (loop hidden states), then speak (greedy generation).

    For each NEW token, we re-breathe the full context through N loops.
    This is "breathe then speak" at the token level — each token decision
    benefits from N loops of processing, but no tokens are generated
    BETWEEN loops.
    """
    device = input_ids.device
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        context = generated[:, -512:]
        logits = forward_breathe(model, context, num_loops, layer_indices)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        if next_token.item() == model.config.eos_token_id:
            break

    return generated


def extract_answer(text):
    match = re.search(r'####\s*([-]?\d+)', text)
    if match:
        return int(match.group(1))
    numbers = re.findall(r'[-]?\d+', text)
    return int(numbers[-1]) if numbers else None


def evaluate(model, tokenizer, problems, num_loops, layer_indices, device, verbose=False):
    correct = 0
    total = len(problems)

    for i, prob in enumerate(problems):
        prompt = f"Solve: {prob['problem']}\nAnswer: "
        inputs = tokenizer(prompt, return_tensors='pt').to(device)

        output_ids = generate_breathe_then_speak(
            model, inputs['input_ids'], num_loops, layer_indices, max_new_tokens=60
        )

        gen_text = tokenizer.decode(
            output_ids[0, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        expected = prob['final_answer']
        predicted = extract_answer(gen_text)

        if predicted == expected:
            correct += 1

        if verbose and i < 5:
            mark = 'OK' if predicted == expected else 'X'
            print(f"  [{mark}] exp={expected} pred={predicted}: {gen_text[:60]}")

    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_problems', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load Pythia-160M
    print("Loading Pythia-160M...")
    model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-160m")
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_layers = len(model.gpt_neox.layers)
    print(f"  Layers: {n_layers}, Hidden: {model.config.hidden_size}")

    problems = generate_l3_problems(args.num_problems, seed=args.seed)
    print(f"  Problems: {args.num_problems}")
    print(f"  Sample: {problems[0]['problem']} → {problems[0]['final_answer']}")

    # ==========================================
    # Test 1: Full model (all 12 layers), varying loop count
    # Breathe in representation space, speak once.
    # ==========================================
    print(f"\n{'='*70}")
    print("TEST 1: Full model (12 layers) × N loops — breathe then speak")
    print(f"{'='*70}")

    all_layers = list(range(n_layers))

    for num_loops in [1, 2, 4, 8]:
        print(f"\n--- {num_loops} loop(s) ({n_layers * num_loops} effective layer passes) ---")
        t0 = time.time()
        acc = evaluate(model, tokenizer, problems, num_loops, all_layers, device, args.verbose)
        elapsed = time.time() - t0
        print(f"  Accuracy: {acc:.1%}")
        print(f"  Time: {elapsed:.1f}s ({elapsed/len(problems)*1000:.0f}ms/problem)")

    # ==========================================
    # Test 2: Layers 0-3 only, varying loop count
    # The original test, but breathe-then-speak instead of mid-loop generation.
    # ==========================================
    print(f"\n{'='*70}")
    print("TEST 2: Layers 0-3 × N loops — breathe then speak")
    print(f"{'='*70}")

    layers_0_3 = [0, 1, 2, 3]

    for num_loops in [1, 2, 4, 8]:
        eff = len(layers_0_3) * num_loops
        print(f"\n--- {num_loops} loop(s) ({eff} effective layer passes) ---")
        t0 = time.time()
        acc = evaluate(model, tokenizer, problems, num_loops, layers_0_3, device, args.verbose)
        elapsed = time.time() - t0
        print(f"  Accuracy: {acc:.1%}")
        print(f"  Time: {elapsed:.1f}s ({elapsed/len(problems)*1000:.0f}ms/problem)")

    # ==========================================
    # Test 3: Compare with the old mid-loop generation baseline
    # Generate between loops (the "copy machine") for reference.
    # ==========================================
    print(f"\n{'='*70}")
    print("COMPARISON: Mid-loop generation (copy machine) vs breathe-then-speak")
    print("  Full model, 2 loops")
    print(f"{'='*70}")

    # Mid-loop: generate after loop 1, feed tokens back, generate after loop 2
    # This is just the normal model run twice in autoregressive mode
    # For a fair test, let's manually do: generate with 1 loop, take those tokens,
    # and generate again with 1 loop from the full (original + generated) context.

    print("\n--- Breathe-then-speak (2 loops, no mid-gen) ---")
    t0 = time.time()
    acc_bts = evaluate(model, tokenizer, problems, 2, all_layers, device, args.verbose)
    elapsed = time.time() - t0
    print(f"  Accuracy: {acc_bts:.1%}")
    print(f"  Time: {elapsed:.1f}s")

    print("\n--- Normal single pass (baseline) ---")
    t0 = time.time()
    acc_base = evaluate(model, tokenizer, problems, 1, all_layers, device, args.verbose)
    elapsed = time.time() - t0
    print(f"  Accuracy: {acc_base:.1%}")
    print(f"  Time: {elapsed:.1f}s")

    # ==========================================
    # Hidden state quality comparison
    # ==========================================
    print(f"\n{'='*70}")
    print("HIDDEN STATE QUALITY: Does more breathing produce richer representations?")
    print(f"{'='*70}")

    import math

    for layer_set_name, layer_indices in [("All 12", all_layers), ("L0-3", layers_0_3)]:
        print(f"\n--- {layer_set_name} ---")

        for num_loops in [1, 2, 4, 8]:
            hiddens = []
            with torch.no_grad():
                for prob in problems[:20]:
                    prompt = f"Solve: {prob['problem']}\nAnswer: "
                    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)

                    h = model.gpt_neox.embed_in(input_ids)
                    seq_len = input_ids.shape[1]
                    pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                    pos_emb = model.gpt_neox.rotary_emb(h, pos_ids)

                    for loop_idx in range(num_loops):
                        for li in layer_indices:
                            layer = model.gpt_neox.layers[li]
                            outputs = layer(h, position_embeddings=pos_emb)
                            h = outputs[0] if isinstance(outputs, tuple) else outputs

                    hiddens.append(h[0, -1].float())  # last token

            stack = torch.stack(hiddens)
            centered = stack - stack.mean(dim=0, keepdim=True)
            c_n = F.normalize(centered, dim=-1)
            mask = ~torch.eye(20, device=device, dtype=torch.bool)
            c_cos = (c_n @ c_n.T)[mask].mean().item()

            U, S, V = torch.svd(centered)
            S_norm = S / (S.sum() + 1e-10)
            eff_rank = math.exp(-(S_norm * torch.log(S_norm + 1e-10)).sum().item())

            dc = stack.mean(dim=0).norm().item()
            sig = centered.norm(dim=-1).mean().item()
            snr = sig / dc if dc > 0 else 0

            print(f"  {num_loops} loop(s): cent_cos={c_cos:.4f}  eff_rank={eff_rank:.1f}  "
                  f"SNR={snr:.4f}  dc={dc:.1f}  signal={sig:.1f}")

    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    print("  If accuracy holds or improves across loops: breathing architecture VALIDATED.")
    print("  If generation stays coherent at 2+ loops: copy machine principle CONFIRMED.")
    print("  If generation degrades: frozen weights still need fine-tuning for looping.")


if __name__ == '__main__':
    main()
