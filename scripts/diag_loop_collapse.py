"""
Diagnostic: WHY does looping collapse representations?

Runs a few problems through Pythia-70M with 1-4 loops and measures:
1. Hidden state norms per loop (is it exploding/collapsing in scale?)
2. Cross-problem cosine similarity per loop (is per-problem info being erased?)
3. Cross-loop cosine similarity (is the representation changing between loops?)
4. Attention entropy per layer per loop (is attention collapsing to one position?)
5. Effective rank of hidden states per loop (are dimensions being lost?)

If cross-problem cos → 1.0, the layers are contractive maps erasing input.
If attention entropy → 0, attention is collapsing to a single position.
Both confirm the fixed-point attractor hypothesis.

Usage:
  python scripts/diag_loop_collapse.py [--num_problems 20] [--max_loops 4]
"""

import argparse
import os
import sys
import random
import math

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


def run_with_hooks(model, input_ids, num_loops, position_embeddings):
    """Run through layers with hooks to capture attention weights."""
    hidden_states = model.gpt_neox.embed_in(input_ids)
    n_layers = len(model.gpt_neox.layers)

    loop_hidden_states = []  # hidden state after each loop
    loop_attn_entropies = []  # per-layer attention entropy after each loop

    for loop_idx in range(num_loops):
        layer_entropies = []

        for layer_idx in range(n_layers):
            layer = model.gpt_neox.layers[layer_idx]

            # Hook to capture attention weights
            attn_weights_captured = []

            def capture_attn(module, args, kwargs, output, _captured=attn_weights_captured):
                # GPTNeoXAttention returns (attn_output, attn_weights)
                # But with SDPA, weights may be None. We'll compute them manually.
                pass

            # Run layer - get hidden states
            outputs = layer(hidden_states, position_embeddings=position_embeddings)
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs

        # Record hidden state after this loop (mean over sequence, before norm)
        loop_hidden_states.append(hidden_states.clone())

    return loop_hidden_states


def compute_attention_entropy(model, input_ids, num_loops, position_embeddings):
    """Manually compute attention and measure entropy per loop."""
    hidden_states = model.gpt_neox.embed_in(input_ids)
    n_layers = len(model.gpt_neox.layers)
    cos_pe, sin_pe = position_embeddings

    loop_entropies = []

    for loop_idx in range(num_loops):
        layer_entropies = []

        for layer_idx in range(n_layers):
            layer = model.gpt_neox.layers[layer_idx]
            attn = layer.attention

            # Manually compute QKV to get attention weights
            qkv = attn.query_key_value(layer.input_layernorm(hidden_states))
            q, k, v = qkv.chunk(3, dim=-1)

            # Reshape to [batch, heads, seq, head_dim]
            batch, seq, _ = q.shape
            num_heads = attn.config.num_attention_heads
            head_dim = attn.head_size
            q = q.view(batch, seq, num_heads, head_dim).transpose(1, 2)
            k = k.view(batch, seq, num_heads, head_dim).transpose(1, 2)

            # Apply rotary embeddings (only to rotary_ndims portion)
            rotary_ndims = attn.rotary_ndims
            q_rot, q_pass = q[..., :rotary_ndims], q[..., rotary_ndims:]
            k_rot, k_pass = k[..., :rotary_ndims], k[..., rotary_ndims:]

            cos = cos_pe.unsqueeze(1)  # [batch, 1, seq, dim]
            sin = sin_pe.unsqueeze(1)

            # rotate_half
            def rotate_half(x):
                x1 = x[..., :x.shape[-1] // 2]
                x2 = x[..., x.shape[-1] // 2:]
                return torch.cat((-x2, x1), dim=-1)

            q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
            k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)
            q = torch.cat([q_rot, q_pass], dim=-1)
            k = torch.cat([k_rot, k_pass], dim=-1)

            # Compute attention scores
            scale = 1.0 / math.sqrt(head_dim)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

            # Causal mask
            causal_mask = torch.triu(torch.ones(seq, seq, device=input_ids.device), diagonal=1).bool()
            attn_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

            attn_probs = F.softmax(attn_scores.float(), dim=-1)

            # Entropy: -sum(p * log(p)), averaged over heads and positions
            log_probs = torch.log(attn_probs + 1e-10)
            entropy = -(attn_probs * log_probs).sum(dim=-1)  # [batch, heads, seq]
            # Max possible entropy for each position = log(position+1)
            max_entropy = torch.log(torch.arange(1, seq + 1, device=input_ids.device, dtype=torch.float))
            # Normalize: entropy / max_entropy
            norm_entropy = entropy[0].mean(dim=0) / (max_entropy + 1e-10)  # [seq]
            # Average over positions (skip first which always has entropy 0)
            avg_entropy = norm_entropy[1:].mean().item() if seq > 1 else 0.0

            layer_entropies.append(avg_entropy)

            # Actually run the layer for hidden state update
            outputs = layer(hidden_states, position_embeddings=position_embeddings)
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs

        loop_entropies.append(layer_entropies)

    return hidden_states, loop_entropies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_problems', type=int, default=20)
    parser.add_argument('--max_loops', type=int, default=4)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device if args.device != 'auto' else 'cpu')
    print(f"Device: {device}")

    # Load Pythia-70M
    print("Loading Pythia-70M...")
    model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-70m")
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_layers = len(model.gpt_neox.layers)
    hidden_size = model.config.hidden_size
    print(f"  Layers: {n_layers}, Hidden: {hidden_size}")

    # Generate problems
    problems = generate_l3_problems(args.num_problems)
    print(f"  Problems: {args.num_problems}")

    # Collect per-loop hidden states for all problems
    print(f"\nRunning {args.max_loops} loops on {args.num_problems} problems...")
    print(f"{'='*70}")

    all_loop_hiddens = []  # [problem_idx][loop_idx] = hidden_state tensor
    all_loop_entropies = []  # [problem_idx][loop_idx][layer_idx] = entropy

    with torch.no_grad():
        for i, prob in enumerate(problems):
            prompt = f"Solve: {prob['problem']}\nAnswer: "
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            input_ids = inputs['input_ids']
            seq_len = input_ids.shape[1]

            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            hidden_states = model.gpt_neox.embed_in(input_ids)
            position_embeddings = model.gpt_neox.rotary_emb(hidden_states, position_ids)

            # Run with attention entropy tracking
            final_h, loop_entropies = compute_attention_entropy(
                model, input_ids, args.max_loops, position_embeddings
            )

            # Also collect hidden states per loop via separate run
            hidden_states = model.gpt_neox.embed_in(input_ids)
            loop_hiddens = []

            for loop_idx in range(args.max_loops):
                for layer_idx in range(n_layers):
                    layer = model.gpt_neox.layers[layer_idx]
                    outputs = layer(hidden_states, position_embeddings=position_embeddings)
                    hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs

                # Mean-pool over sequence for a single vector per problem per loop
                h_mean = hidden_states.float().mean(dim=1)  # [1, hidden]
                loop_hiddens.append(h_mean)

            all_loop_hiddens.append(loop_hiddens)
            all_loop_entropies.append(loop_entropies)

    # === DIAGNOSTIC 1: Hidden state norms per loop ===
    print(f"\n{'='*70}")
    print("DIAGNOSTIC 1: Hidden state L2 norm per loop")
    print(f"{'='*70}")
    for loop_idx in range(args.max_loops):
        norms = [all_loop_hiddens[p][loop_idx].norm().item() for p in range(args.num_problems)]
        mean_norm = sum(norms) / len(norms)
        std_norm = (sum((n - mean_norm)**2 for n in norms) / len(norms)) ** 0.5
        print(f"  Loop {loop_idx}: norm = {mean_norm:.2f} ± {std_norm:.2f}")

    # === DIAGNOSTIC 2: Cross-problem cosine similarity per loop ===
    print(f"\n{'='*70}")
    print("DIAGNOSTIC 2: Cross-problem cosine similarity per loop")
    print("  (1.0 = all problems identical = input info erased)")
    print(f"{'='*70}")
    for loop_idx in range(args.max_loops):
        # Stack all problem hidden states for this loop
        hiddens = torch.cat([all_loop_hiddens[p][loop_idx] for p in range(args.num_problems)], dim=0)  # [N, hidden]
        hiddens_norm = F.normalize(hiddens, dim=-1)
        cos_matrix = hiddens_norm @ hiddens_norm.T  # [N, N]
        # Mean of off-diagonal elements
        mask = ~torch.eye(args.num_problems, device=device, dtype=torch.bool)
        mean_cos = cos_matrix[mask].mean().item()
        min_cos = cos_matrix[mask].min().item()
        max_cos = cos_matrix[mask].max().item()
        print(f"  Loop {loop_idx}: cos = {mean_cos:.4f} (min={min_cos:.4f}, max={max_cos:.4f})")

    # === DIAGNOSTIC 3: Cross-loop cosine similarity (same problem) ===
    print(f"\n{'='*70}")
    print("DIAGNOSTIC 3: Cross-loop cosine similarity (same problem)")
    print("  (1.0 = representation unchanged by loop = fixed point)")
    print(f"{'='*70}")
    for loop_idx in range(1, args.max_loops):
        cos_vals = []
        for p in range(args.num_problems):
            h_prev = F.normalize(all_loop_hiddens[p][loop_idx - 1], dim=-1)
            h_curr = F.normalize(all_loop_hiddens[p][loop_idx], dim=-1)
            cos_vals.append((h_prev * h_curr).sum().item())
        mean_cos = sum(cos_vals) / len(cos_vals)
        std_cos = (sum((c - mean_cos)**2 for c in cos_vals) / len(cos_vals)) ** 0.5
        print(f"  Loop {loop_idx-1} → {loop_idx}: cos = {mean_cos:.4f} ± {std_cos:.4f}")

    # === DIAGNOSTIC 4: Attention entropy per layer per loop ===
    print(f"\n{'='*70}")
    print("DIAGNOSTIC 4: Attention entropy per layer per loop")
    print("  (normalized: 1.0 = uniform attention, 0.0 = attend to one token)")
    print(f"{'='*70}")
    for loop_idx in range(args.max_loops):
        entropies = []
        for layer_idx in range(n_layers):
            vals = [all_loop_entropies[p][loop_idx][layer_idx] for p in range(args.num_problems)]
            mean_e = sum(vals) / len(vals)
            entropies.append(mean_e)
        layer_str = "  ".join(f"L{i}={e:.3f}" for i, e in enumerate(entropies))
        avg_e = sum(entropies) / len(entropies)
        print(f"  Loop {loop_idx}: {layer_str}  avg={avg_e:.3f}")

    # === DIAGNOSTIC 5: Effective rank per loop ===
    print(f"\n{'='*70}")
    print("DIAGNOSTIC 5: Effective rank of hidden states per loop")
    print("  (measures how many dimensions carry information)")
    print(f"{'='*70}")
    for loop_idx in range(args.max_loops):
        hiddens = torch.cat([all_loop_hiddens[p][loop_idx] for p in range(args.num_problems)], dim=0)  # [N, hidden]
        # SVD to get singular values
        U, S, V = torch.svd(hiddens.float())
        # Effective rank = exp(entropy of normalized singular values)
        S_norm = S / S.sum()
        entropy = -(S_norm * torch.log(S_norm + 1e-10)).sum().item()
        eff_rank = math.exp(entropy)
        # Also: fraction of variance in top-1, top-3, top-10
        var_total = (S ** 2).sum().item()
        var_top1 = (S[0] ** 2).item() / var_total
        var_top3 = (S[:3] ** 2).sum().item() / var_total
        var_top10 = (S[:min(10, len(S))] ** 2).sum().item() / var_total
        print(f"  Loop {loop_idx}: eff_rank = {eff_rank:.1f}/{args.num_problems}, "
              f"top1={var_top1:.1%}, top3={var_top3:.1%}, top10={var_top10:.1%}")

    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")
    # Auto-interpret
    loop0_cos = sum(
        F.cosine_similarity(
            all_loop_hiddens[i][0], all_loop_hiddens[j][0]
        ).item()
        for i in range(args.num_problems) for j in range(i+1, args.num_problems)
    ) / (args.num_problems * (args.num_problems - 1) / 2)

    last_loop = args.max_loops - 1
    loopN_cos = sum(
        F.cosine_similarity(
            all_loop_hiddens[i][last_loop], all_loop_hiddens[j][last_loop]
        ).item()
        for i in range(args.num_problems) for j in range(i+1, args.num_problems)
    ) / (args.num_problems * (args.num_problems - 1) / 2)

    if loopN_cos > 0.95 and loop0_cos < 0.95:
        print("  → CONTRACTIVE MAP CONFIRMED: Cross-problem similarity increases with loops.")
        print("    The layers erase per-problem information, converging to a shared attractor.")
        print("    Fine-tuning must teach layers to PRESERVE representational diversity across loops.")
    elif loopN_cos > 0.95 and loop0_cos > 0.95:
        print("  → ALREADY SIMILAR: Even after one loop, representations are nearly identical.")
        print("    The model's 6 layers map most inputs to similar regions of representation space.")
    else:
        print("  → REPRESENTATIONS STAY DIVERSE across loops. Collapse is not the issue.")
        print("    Investigate attention patterns or specific layer bottlenecks instead.")


if __name__ == '__main__':
    main()
