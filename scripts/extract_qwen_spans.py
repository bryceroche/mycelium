#!/usr/bin/env python3
"""Extract span boundaries using Qwen hidden states + attention.

Qwen uses causal (autoregressive) attention, which makes raw attention
connectivity too smooth for boundary detection. Instead, we use hidden
state similarity drops to find clause boundaries:

Within a clause, adjacent token representations are similar (processing
the same semantic unit). At clause boundaries, representations shift
(new operation/context). This is a clean, attention-derived signal.

We also extract the symmetrized attention matrix for computing
cross-attention between spans (needed for graph composition).

Usage:
    python scripts/extract_qwen_spans.py --num-problems 7473
    python scripts/extract_qwen_spans.py --test  # quick test with 10 problems
"""

import argparse
import json
import sys
import time
import numpy as np
import torch
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from mycelium.attention_graph import AttentionGraphBuilder


def load_qwen_model(model_name: str = "Qwen/Qwen2.5-Math-7B-Instruct"):
    """Load Qwen model with attention and hidden state output."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",  # Need full attention matrices
    )
    model.eval()
    print(f"Model loaded on {model.hf_device_map if hasattr(model, 'hf_device_map') else 'auto'}")
    return model, tokenizer


def extract_qwen_features(
    model, tokenizer, text: str, max_length: int = 256
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract attention matrix AND hidden states from Qwen.

    Returns:
        attention_matrix: (seq_len, seq_len) symmetrized, middle-layer averaged
        hidden_states: (seq_len, hidden_dim) middle-layer token representations
        tokens: List of token strings
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"]

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True)

    # --- Hidden states (middle layer, best structural signal) ---
    all_hidden = outputs.hidden_states  # tuple of (batch, seq, hidden)
    num_hl = len(all_hidden)
    mid_layer = num_hl // 2
    hidden_states = all_hidden[mid_layer][0].cpu().float().numpy()  # (seq, hidden)

    # --- Attention matrix (middle layers, NaN-safe) ---
    attentions = outputs.attentions
    num_al = len(attentions)
    layer_start = max(0, num_al // 4)
    layer_end = min(num_al, 3 * num_al // 4)

    valid_layers = []
    for i in range(layer_start, layer_end):
        if not torch.isnan(attentions[i]).any():
            valid_layers.append(attentions[i].float())

    if not valid_layers:
        for a in attentions:
            if not torch.isnan(a).any():
                valid_layers.append(a.float())

    if not valid_layers:
        raise ValueError("All attention layers contain NaN")

    all_attn = torch.stack(valid_layers, dim=0)
    avg_attn = all_attn.mean(dim=(0, 2))[0]  # [seq, seq]
    attention_matrix = avg_attn.cpu().numpy()
    # Symmetrize causal attention for cross-span computation
    attention_matrix = (attention_matrix + attention_matrix.T) / 2

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu())

    return attention_matrix, hidden_states, tokens


def detect_spans_from_hidden_states(
    hidden_states: np.ndarray,
    tokens: List[str],
    min_span_size: int = 4,
    percentile_threshold: float = 25.0,
) -> List[Tuple[int, int]]:
    """Detect span boundaries using cosine similarity between adjacent hidden states.

    Within a clause, adjacent token representations are similar.
    At clause boundaries, they shift. We find positions where the
    similarity drops into the bottom Nth percentile.

    Args:
        hidden_states: (seq_len, hidden_dim) token representations
        tokens: Token strings
        min_span_size: Minimum tokens per span
        percentile_threshold: Bottom percentile for boundary detection
    """
    n = len(tokens)
    if n <= min_span_size:
        return [(0, n)]

    # Normalize hidden states
    norms = np.linalg.norm(hidden_states, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normalized = hidden_states / norms

    # Cosine similarity between adjacent tokens
    adj_sims = np.array([
        np.dot(normalized[i], normalized[i + 1])
        for i in range(n - 1)
    ])

    # Boundary threshold: bottom percentile of similarities
    threshold = np.percentile(adj_sims, percentile_threshold)

    # Find boundary candidates (positions where similarity drops)
    boundary_candidates = []
    for i in range(n - 1):
        if adj_sims[i] < threshold:
            boundary_candidates.append(i + 1)  # boundary AFTER position i

    # Build spans, enforcing min_span_size
    spans = []
    start = 0
    for boundary in boundary_candidates:
        if boundary - start >= min_span_size and n - boundary >= min_span_size:
            spans.append((start, boundary))
            start = boundary

    # Final span
    if start < n:
        if spans and (n - start) < min_span_size:
            spans[-1] = (spans[-1][0], n)
        else:
            spans.append((start, n))

    return spans if spans else [(0, n)]


def tokens_to_text(tokens: List[str], start: int, end: int) -> str:
    """Convert token range to readable text."""
    span_tokens = tokens[start:end]
    text = ""
    for t in span_tokens:
        if t in ("<s>", "</s>", "<|endoftext|>", "<|im_start|>", "<|im_end|>",
                  "[CLS]", "[SEP]", "[PAD]", "<pad>"):
            continue
        if t.startswith("##"):
            text += t[2:]
        elif t.startswith("Ġ") or t.startswith("▁"):
            text += " " + t[1:]
        else:
            if text and not text.endswith(" "):
                text += " "
            text += t
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Extract Qwen attention spans")
    parser.add_argument("--num-problems", type=int, default=7473)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--output", default="data/qwen_coarse_spans.json")
    parser.add_argument("--min-span-size", type=int, default=4, help="Min tokens per span")
    parser.add_argument("--percentile", type=float, default=25.0,
                        help="Boundary detection: bottom Nth percentile of similarity")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--test", action="store_true", help="Quick test (10 problems)")
    args = parser.parse_args()

    if args.test:
        args.num_problems = 10

    model, tokenizer = load_qwen_model(args.model)

    print("Loading GSM8K...")
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    print(f"Loaded {len(ds)} problems, processing {args.num_problems}")

    builder = AttentionGraphBuilder()

    all_spans = []
    span_lengths = []
    errors = 0

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    for i in tqdm(range(min(args.num_problems, len(ds))), desc="Extracting"):
        problem = ds[i]
        question = problem["question"]

        try:
            attention_matrix, hidden_states, tokens = extract_qwen_features(
                model, tokenizer, question, max_length=args.max_length
            )

            # Detect boundaries from hidden state similarity
            boundaries = detect_spans_from_hidden_states(
                hidden_states, tokens,
                min_span_size=args.min_span_size,
                percentile_threshold=args.percentile,
            )

            for span_idx, (start, end) in enumerate(boundaries):
                span_text = tokens_to_text(tokens, start, end)
                num_tokens = end - start

                # Cross-attention to earlier spans
                backward_attn = 0.0
                if span_idx > 0:
                    for prev_start, prev_end in boundaries[:span_idx]:
                        cross = attention_matrix[start:end, prev_start:prev_end].sum()
                        backward_attn += float(cross)

                # Span connectivity from attention
                conn = float(builder.compute_span_connectivity(attention_matrix, start, end))

                all_spans.append({
                    "problem_id": f"gsm8k_train_{i}",
                    "span_idx": span_idx,
                    "span_text": span_text,
                    "start_token": start,
                    "end_token": end,
                    "num_tokens": num_tokens,
                    "connectivity": conn,
                    "backward_attention": backward_attn,
                    "span_position": span_idx / max(1, len(boundaries) - 1) if len(boundaries) > 1 else 0.0,
                })
                span_lengths.append(num_tokens)

            if args.test:
                print(f"\nProblem {i}: {question[:80]}...")
                for span_idx, (start, end) in enumerate(boundaries):
                    txt = tokens_to_text(tokens, start, end)
                    conn = builder.compute_span_connectivity(attention_matrix, start, end)
                    print(f"  [{span_idx}] ({end-start} tok) conn={conn:.4f}: {txt[:80]}")

        except Exception as e:
            errors += 1
            if args.test:
                print(f"\nERROR on problem {i}: {e}")
                import traceback
                traceback.print_exc()

        if (i + 1) % args.checkpoint_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (args.num_problems - i - 1) / rate / 60

            checkpoint_path = output_dir / f"qwen_spans_checkpoint_{i+1}.json"
            with open(checkpoint_path, "w") as f:
                json.dump({"spans": all_spans, "stats": {
                    "problems_processed": i + 1,
                    "total_spans": len(all_spans),
                    "errors": errors,
                }}, f)

            ctr = Counter(span_lengths)
            print(f"\n  Checkpoint {i+1}: {len(all_spans)} spans, {errors} errors, "
                  f"{rate:.1f} prob/s, ETA: {eta:.0f} min")
            for length, count in sorted(ctr.items(), key=lambda x: -x[1])[:10]:
                print(f"    {length} tokens: {count} ({100*count/len(span_lengths):.1f}%)")

    elapsed = time.time() - t0
    ctr = Counter(span_lengths)

    result = {
        "spans": all_spans,
        "stats": {
            "problems_processed": min(args.num_problems, len(ds)),
            "total_spans": len(all_spans),
            "avg_spans_per_problem": len(all_spans) / max(1, min(args.num_problems, len(ds)) - errors),
            "avg_span_length": float(np.mean(span_lengths)) if span_lengths else 0,
            "median_span_length": float(np.median(span_lengths)) if span_lengths else 0,
            "errors": errors,
            "elapsed_seconds": elapsed,
            "model": args.model,
            "percentile_threshold": args.percentile,
            "min_span_size": args.min_span_size,
            "span_length_distribution": {str(k): v for k, v in sorted(ctr.items())},
        },
    }

    with open(args.output, "w") as f:
        json.dump(result, f)

    print(f"\n{'=' * 70}")
    print(f"DONE: {len(all_spans)} spans from {min(args.num_problems, len(ds))} problems")
    print(f"Avg spans/problem: {result['stats']['avg_spans_per_problem']:.1f}")
    print(f"Avg span length: {result['stats']['avg_span_length']:.1f} tokens")
    print(f"Median span length: {result['stats']['median_span_length']:.0f} tokens")
    print(f"Errors: {errors}")
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Saved to: {args.output}")

    print(f"\nSpan length distribution:")
    for length, count in sorted(ctr.items(), key=lambda x: -x[1])[:15]:
        print(f"  {length} tokens: {count} ({100*count/len(span_lengths):.1f}%)")


if __name__ == "__main__":
    main()
