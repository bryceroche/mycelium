#!/usr/bin/env python3
"""
Generalize GSM8K spans using Qwen-7B (one-time batch job).

Qwen generalizes each span into a pattern:
  - Names/pronouns → [ENTITY]
  - Items/objects → [ENTITY]
  - Numbers → [N]
  - Verbs and structural words → kept as-is

Patterns are then clustered at 95% cosine similarity to produce canonical
span templates. Custom sub-graph DSLs are written per template AFTER clustering.

Run on GPU VM. Output: gsm8k_generalized.json
Used ONLY for training template creation. NOT needed at inference time.
At inference, raw spans are embedded and matched to templates by cosine similarity.

USAGE:
    # Full run on VM with GPU:
    python scripts/generalize_with_qwen.py

    # Quick test (10 problems):
    python scripts/generalize_with_qwen.py --num-samples 10

    # Custom output:
    python scripts/generalize_with_qwen.py --output my_output.json
"""

import argparse
import json
import re
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# =============================================================================
# Span Extraction (sentence segmentation)
# =============================================================================

def extract_spans(problem_text: str) -> List[str]:
    """Extract spans from a problem by sentence segmentation.

    No filtering — every sentence is a span. Let the system learn
    what's important rather than hardcoding heuristics.
    """
    parts = re.split(r'(?<=[.!?])\s+', problem_text)

    spans = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        spans.append(part)

    return spans


# =============================================================================
# Qwen Generalization
# =============================================================================

GENERALIZATION_PROMPT = """Replace person names/pronouns with [PERSON1],[PERSON2],etc. Replace object nouns (apples/dollars/bags/hours) with [ITEM1],[ITEM2],etc. Replace numbers/$amounts with [N]. Keep verbs and structure. Output ONLY JSON.

{{"input": "John has 5 apples", "output": "[PERSON1] has [N] [ITEM1]"}}
{{"input": "She gave 2 cookies to Mary", "output": "[PERSON1] gave [N] [ITEM1] to [PERSON2]"}}
{{"input": "Each bag contains 5 items", "output": "each [ITEM1] contains [N] [ITEM2]"}}
{{"input": "She earned $12 per hour for 8 hours", "output": "[PERSON1] earned [N] per [ITEM1] for [N] [ITEM1]"}}
{{"input": "He has twice as many apples as Bob has oranges", "output": "[PERSON1] has twice as many [ITEM1] as [PERSON2] has [ITEM2]"}}
{{"input": "How many apples does John have?", "output": "how many [ITEM1] does [PERSON1] have?"}}
{{"input": "{span}", "output": """


def _clean_pattern(raw: str) -> str:
    """Strip common artifacts from extracted patterns."""
    p = raw.strip()
    # Strip trailing JSON artifacts: "}, }, etc.
    p = re.sub(r'["\s}]+$', '', p)
    # Strip leading quotes
    p = p.lstrip('"').lstrip("'")
    # Strip trailing period
    p = p.rstrip('.')
    return p.strip()


def parse_qwen_response(response: str) -> Optional[Dict]:
    """Parse Qwen's JSON response to extract the generalized pattern.

    The prompt asks for JSON like: {"output": "[PERSON1] has [N] [ITEM1]"}
    Since we provide the opening of the JSON, Qwen should complete it.

    Strategies (in order):
    1. Parse as JSON (ideal case)
    2. Extract quoted string after "output":
    3. Find any line with [PERSON*]/[ITEM*]/[N] placeholders
    """
    cleaned = response.strip()

    # Strategy 1: Try to parse as JSON or complete partial JSON
    # The prompt ends with {"input": "...", "output": so response is typically: "value"}
    for attempt in [
        '{"output": ' + cleaned,          # most common: response is "value"}
        cleaned,                          # raw response (already valid JSON)
        '{"output": ' + cleaned + '}',    # wrap as JSON value
        '{"output": "' + cleaned,         # wrap as JSON string
    ]:
        try:
            # Fix common JSON issues: single quotes, trailing comma
            fixed = attempt.replace("'", '"').rstrip(',').rstrip()
            if not fixed.endswith('}'):
                fixed = fixed.rstrip('"') + '"}'
            obj = json.loads(fixed)
            pattern = _clean_pattern(obj.get('output', ''))
            if pattern and len(pattern) > 3:
                return {'pattern': pattern}
        except (json.JSONDecodeError, AttributeError):
            continue

    # Strategy 2: Find "output": "..." in the response
    match = re.search(r'"output"\s*:\s*"([^"]+)"', cleaned)
    if match:
        pattern = _clean_pattern(match.group(1))
        if pattern and len(pattern) > 3:
            return {'pattern': pattern}

    # Strategy 3: Look for any line with [PERSON*]/[ITEM*]/[N] placeholders (fallback)
    placeholder_re = re.compile(r'\[(?:PERSON|ITEM)\d+\]|\[N\]', re.IGNORECASE)
    for line in cleaned.split('\n'):
        line = _clean_pattern(line)
        if placeholder_re.search(line) and len(line) > 5:
            # Skip if it looks like instructions rather than a pattern
            if any(x in line.lower() for x in ['replace', 'should be', 'we need']):
                continue
            return {'pattern': line}

    return None


def generalize_batch_qwen(
    spans: List[str],
    model,
    tokenizer,
    batch_size: int = 16,
    checkpoint_dir: str = ".",
    checkpoint_every: int = 50,
) -> List[Optional[Dict]]:
    """Generalize spans using Qwen with true batched inference.

    Saves checkpoint every `checkpoint_every` batches so we can test
    partial results while the full batch runs.

    Args:
        spans: List of raw span texts
        model: Loaded Qwen model
        tokenizer: Loaded Qwen tokenizer
        batch_size: Number of spans per GPU batch
        checkpoint_dir: Directory for checkpoint files
        checkpoint_every: Save checkpoint every N batches

    Returns:
        List of parsed results (or None for failures)
    """
    import torch

    # Check for existing checkpoint to resume from
    checkpoint_path = os.path.join(checkpoint_dir, "qwen_checkpoint.json")
    results = []
    start_batch = 0

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        results = checkpoint["results"]
        start_batch = checkpoint["next_batch_idx"]
        print(f"Resuming from checkpoint: {len(results)} spans done, "
              f"starting at batch index {start_batch}")

    # Pad from left for batched generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total_batches = (len(spans) + batch_size - 1) // batch_size
    batch_num = 0

    for i in tqdm(range(0, len(spans), batch_size), desc="Generalizing"):
        # Skip already-processed batches (from checkpoint)
        if i < start_batch:
            batch_num += 1
            continue
        batch = spans[i:i + batch_size]

        # Build all prompts for this batch
        texts = []
        for span in batch:
            prompt = GENERALIZATION_PROMPT.format(span=span)
            messages = [
                {"role": "system", "content": "Complete the JSON. Output ONLY the pattern value and closing brace. No explanation."},
                {"role": "user", "content": prompt},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)

        # Tokenize entire batch at once with padding
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
            )

        # Decode each output, maintaining order via placeholder slots
        batch_results = [None] * len(batch)
        retry_indices = []
        input_len = inputs.input_ids.shape[1]

        for j, span in enumerate(batch):
            response = tokenizer.decode(
                outputs[j][input_len:],
                skip_special_tokens=True,
            ).strip()

            if not response:
                # Empty response — pad_token==eos_token can cause this in batches
                retry_indices.append(j)
                continue

            parsed = parse_qwen_response(response)
            if parsed:
                parsed['raw_span'] = span
                parsed['qwen_response'] = response
                batch_results[j] = parsed
            else:
                fallback = regex_fallback_generalize(span)
                fallback['raw_span'] = span
                fallback['qwen_response'] = response
                fallback['qwen_failed'] = True
                batch_results[j] = fallback

        # Retry empty responses individually (no padding → no pad/eos confusion)
        for j in retry_indices:
            span = batch[j]
            prompt = GENERALIZATION_PROMPT.format(span=span)
            messages = [
                {"role": "system", "content": "Complete the JSON. Output ONLY the pattern value and closing brace. No explanation."},
                {"role": "user", "content": prompt},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            single_input = tokenizer(
                text, return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                single_output = model.generate(
                    **single_input,
                    max_new_tokens=100,
                    do_sample=False,
                )

            single_len = single_input.input_ids.shape[1]
            response = tokenizer.decode(
                single_output[0][single_len:],
                skip_special_tokens=True,
            ).strip()

            parsed = parse_qwen_response(response) if response else None
            if parsed:
                parsed['raw_span'] = span
                parsed['qwen_response'] = response
                batch_results[j] = parsed
            else:
                fallback = regex_fallback_generalize(span)
                fallback['raw_span'] = span
                fallback['qwen_response'] = response
                fallback['qwen_failed'] = True
                batch_results[j] = fallback

        results.extend(batch_results)
        batch_num += 1

        # Save checkpoint every N batches
        if batch_num % checkpoint_every == 0:
            checkpoint_data = {
                "results": results,
                "raw_spans": spans,  # Save all raw spans so generic_method can be re-run
                "next_batch_idx": i + batch_size,
                "total_spans": len(spans),
                "batches_done": batch_num,
                "total_batches": total_batches,
            }
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f)
            print(f"\n  Checkpoint saved: {len(results)}/{len(spans)} spans "
                  f"({batch_num}/{total_batches} batches)")

        # Clear GPU cache periodically
        if i % (batch_size * 10) == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Remove checkpoint file on completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Checkpoint file removed (batch complete)")

    return results


def regex_fallback_generalize(span: str) -> Dict:
    """Fallback generalization when Qwen fails. Numbers only."""
    pattern = re.sub(r'\$[\d,]+\.?\d*', '[N]', span)
    pattern = re.sub(r'\b\d{1,3}(?:,\d{3})+\.?\d*\b', '[N]', pattern)
    pattern = re.sub(r'\b\d+\.?\d*\b', '[N]', pattern)

    return {
        'pattern': pattern.lower(),
    }




# =============================================================================
# Main Pipeline
# =============================================================================

def load_gsm8k(data_dir: Path, num_samples: Optional[int] = None) -> List[Dict]:
    """Load GSM8K problems from local jsonl."""
    problems = []
    train_path = data_dir / "train.jsonl"

    with open(train_path) as f:
        for i, line in enumerate(f):
            if num_samples and i >= num_samples:
                break
            item = json.loads(line)
            problems.append({
                'id': f'gsm8k_train_{i}',
                'question': item['question'],
                'answer': item.get('answer'),
                'num_map': item.get('num_map', {}),
            })

    return problems


def run_generalization(
    problems: List[Dict],
    model,
    tokenizer,
    batch_size: int = 16,
) -> List[Dict]:
    """Run full generalization pipeline.

    Returns list of generalized span records with:
    - problem_id, raw_span, pattern, problem_text
    """
    all_records = []

    # Extract all spans with their problem context
    span_data = []
    for problem in problems:
        spans = extract_spans(problem['question'])
        for span in spans:
            span_data.append({
                'span': span,
                'problem_id': problem['id'],
                'problem_text': problem['question'],
            })

    print(f"Extracted {len(span_data)} spans from {len(problems)} problems")

    # Generalize all spans
    raw_spans = [s['span'] for s in span_data]
    generalized = generalize_batch_qwen(raw_spans, model, tokenizer, batch_size)

    # Merge results
    for sd, gen in zip(span_data, generalized):
        record = {
            'problem_id': sd['problem_id'],
            'problem_text': sd['problem_text'],
            'raw_span': sd['span'],
            'pattern': gen['pattern'],
            'qwen_failed': gen.get('qwen_failed', False),
            'qwen_response': gen.get('qwen_response', ''),
        }
        all_records.append(record)

    return all_records


def compute_template_embeddings(
    records: List[Dict],
    similarity_threshold: float = 0.95,
) -> List[Dict]:
    """Compute MiniLM embeddings and cluster templates.

    Two-stage deduplication:
    1. GROUP BY exact pattern (like SQL GROUP BY)
    2. Cluster by embedding cosine similarity across ALL templates

    Template centroids are computed from RAW span examples (not generalized
    patterns), so they live in the same embedding space as inference spans.

    Args:
        records: Generalized span records from Qwen
        similarity_threshold: Cosine threshold for merging similar templates.
            Higher = more templates (stricter merging). Default 0.95.
    """
    from mycelium.dual_signal_templates import SpanDetector
    import numpy as np

    print("\nComputing template embeddings from raw span examples...")
    detector = SpanDetector()

    # Stage 1: GROUP BY exact pattern
    groups = defaultdict(list)
    for r in records:
        groups[r['pattern']].append(r)

    print(f"  Exact pattern groups: {len(groups)}")

    # Build initial templates from groups
    raw_templates = []
    for pattern, group in tqdm(groups.items(), desc="Embedding"):
        if len(pattern) < 5:
            continue

        # Compute centroid from RAW span embeddings
        embeddings = []
        for r in group[:20]:
            emb, _, _ = detector.extract_features(r['raw_span'])
            embeddings.append(emb)

        centroid = np.mean(embeddings, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

        raw_templates.append({
            'pattern': pattern,
            'centroid': centroid,
            'count': len(group),
            'examples': [r['raw_span'] for r in group[:10]],
            'qwen_fail_rate': sum(1 for r in group if r.get('qwen_failed')) / len(group),
        })

    print(f"  Templates before clustering: {len(raw_templates)}")

    # Stage 2: Cluster by embedding similarity across ALL templates
    # No operation-type partitioning — spans can contain multiple ops,
    # so cosine similarity alone determines what merges.
    raw_templates.sort(key=lambda t: -t['count'])

    centroids = np.array([t['centroid'] for t in raw_templates])
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms[norms == 0] = 1
    centroids_norm = centroids / norms

    used = set()
    clusters = []

    for i in range(len(raw_templates)):
        if i in used:
            continue

        cluster = [i]
        used.add(i)

        for j in range(i + 1, len(raw_templates)):
            if j in used:
                continue
            sim = np.dot(centroids_norm[i], centroids_norm[j])
            if sim >= similarity_threshold:
                cluster.append(j)
                used.add(j)

        clusters.append(cluster)

    print(f"  Clustered: {len(raw_templates)} → {len(clusters)} templates")

    final_templates = []
    for cluster in clusters:
        # Representative = highest count (already sorted)
        rep_idx = cluster[0]
        rep = raw_templates[rep_idx]

        # Merge counts and examples from cluster
        total_count = sum(raw_templates[i]['count'] for i in cluster)
        all_examples = []
        for i in cluster:
            all_examples.extend(raw_templates[i]['examples'])

        # Recompute centroid from merged cluster
        cluster_centroids = centroids[cluster]
        merged_centroid = np.mean(cluster_centroids, axis=0)
        merged_centroid = merged_centroid / (np.linalg.norm(merged_centroid) + 1e-8)

        template_id = re.sub(r'[^a-z0-9]', '_', rep['pattern'].lower())[:50]

        final_templates.append({
            'template_id': template_id,
            'pattern': rep['pattern'],
            'embedding_centroid': merged_centroid.tolist(),
            'count': total_count,
            'cluster_size': len(cluster),
            'pattern_examples': all_examples[:10],
            'qwen_fail_rate': rep['qwen_fail_rate'],
        })

    final_templates.sort(key=lambda t: -t['count'])
    return final_templates


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generalize GSM8K spans using Qwen-7B (one-time batch job)"
    )
    parser.add_argument("--data-dir", default="data/gsm8k_gts",
                        help="Path to GSM8K data directory")
    parser.add_argument("--output", "-o", default="gsm8k_generalized.json",
                        help="Output path for generalized spans")
    parser.add_argument("--templates-output", default="qwen_templates.json",
                        help="Output path for deduplicated templates")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Limit number of problems (for testing)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for Qwen inference")
    parser.add_argument("--quantize", choices=["4bit", "8bit", "none"],
                        default="4bit", help="Quantization for Qwen")
    parser.add_argument("--skip-qwen", action="store_true",
                        help="Skip Qwen, use regex fallback only (for testing)")
    parser.add_argument("--similarity-threshold", type=float, default=0.95,
                        help="Cosine threshold for merging templates (higher=more templates)")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip MiniLM embedding computation")

    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / args.data_dir
    output_path = project_root / args.output
    templates_path = project_root / args.templates_output

    # Load problems
    print(f"Loading GSM8K from {data_dir}...")
    problems = load_gsm8k(data_dir, args.num_samples)
    print(f"Loaded {len(problems)} problems")

    if args.skip_qwen:
        # Regex-only fallback (for local testing without GPU)
        print("\nUsing regex fallback (no Qwen)...")
        all_records = []
        for problem in tqdm(problems, desc="Processing"):
            spans = extract_spans(problem['question'])
            for span in spans:
                fb = regex_fallback_generalize(span)
                fb['raw_span'] = span
                fb['problem_id'] = problem['id']
                fb['problem_text'] = problem['question']
                fb['qwen_failed'] = True
                all_records.append(fb)
    else:
        # Load Qwen
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("\nLoading Qwen-7B...")
        model_name = "Qwen/Qwen2.5-7B-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        model_kwargs = {"trust_remote_code": True}
        if args.quantize == "4bit":
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model_kwargs["device_map"] = "auto"
        elif args.quantize == "8bit":
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True
            )
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        model.eval()
        print(f"Loaded {model_name}")

        # Run generalization
        all_records = run_generalization(
            problems, model, tokenizer, args.batch_size
        )

    # Save raw generalized spans
    print(f"\nSaving {len(all_records)} generalized spans to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(all_records, f, indent=2)

    # Stats
    fail_count = sum(1 for r in all_records if r.get('qwen_failed'))
    print(f"\nQwen failures: {fail_count}/{len(all_records)} ({100*fail_count/max(len(all_records),1):.1f}%)")

    # Compute template embeddings and deduplicate
    if not args.skip_embeddings:
        templates = compute_template_embeddings(all_records, args.similarity_threshold)

        print(f"\nDeduplicated to {len(templates)} templates")
        print(f"\nTop 20 templates:")
        for t in templates[:20]:
            print(f"  {t['pattern'][:60]:60} (n={t['count']:4}, cluster={t['cluster_size']})")

        print(f"\nSaving templates to {templates_path}...")
        with open(templates_path, 'w') as f:
            json.dump(templates, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
