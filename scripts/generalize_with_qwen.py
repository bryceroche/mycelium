#!/usr/bin/env python3
"""
Generalize GSM8K spans using Qwen-7B (one-time batch job).

Qwen generalizes each span:
  - Names/pronouns → [ENTITY]
  - Items/objects → [ENTITY]
  - Numbers → [N]
  - Verbs and structural words → kept as-is

Also classifies operation type (SET/ADD/SUB/MUL/DIV) and infers DSL.

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
    """Extract operational spans from a problem.

    Splits on sentence boundaries. Filters out questions.
    Keeps spans with numbers OR operational keywords (twice, half, etc).
    """
    # Split on sentence-ending punctuation (but not abbreviations like Dr.)
    parts = re.split(r'(?<=[.!?])\s+', problem_text)

    # Operational keywords that don't require explicit numbers
    OP_KEYWORDS = [
        'twice', 'thrice', 'double', 'triple', 'half', 'quarter',
        'more than', 'less than', 'fewer than', 'times as',
        'each', 'every', 'per',
    ]

    spans = []
    for part in parts:
        part = part.strip().rstrip('.!?')
        if not part:
            continue

        # Skip questions
        part_lower = part.lower()
        if any(q in part_lower for q in [
            'how many', 'how much', 'what is', 'what are', 'what was',
            'what does', 'what did', 'how long', 'how far',
        ]):
            continue

        # Keep if has a number OR operational keywords
        has_number = bool(re.search(r'\d+', part))
        has_op_keyword = any(kw in part_lower for kw in OP_KEYWORDS)

        if has_number or has_op_keyword:
            spans.append(part)

    return spans


# =============================================================================
# Qwen Generalization
# =============================================================================

GENERALIZATION_PROMPT = """You are analyzing math word problem sentences. For each sentence, output exactly three lines:

PATTERN: Replace person names and pronouns (he/she/they/it) with [ENTITY], object/item nouns (apples/dollars/cookies/bags/etc) with [ENTITY], and numbers with [N]. Keep verbs, prepositions, and structural words exactly as they are.
OPERATION: One of: SET (establishing initial quantity), ADD (gaining/receiving), SUB (losing/giving away), MUL (multiplying/rate×time), DIV (dividing/splitting)
DSL: The computation expression using: value, entity + value, entity - value, entity * value, entity / value, ref + value, ref - value, ref * 2, ref * 3, ref * value

Examples:
Sentence: "John has 5 apples"
PATTERN: [ENTITY] has [N] [ENTITY]
OPERATION: SET
DSL: value

Sentence: "He gave 2 to Mary"
PATTERN: [ENTITY] gave [N] to [ENTITY]
OPERATION: SUB
DSL: entity - value

Sentence: "She bought 3 more oranges"
PATTERN: [ENTITY] bought [N] more [ENTITY]
OPERATION: ADD
DSL: entity + value

Sentence: "Each bag contains 5 items"
PATTERN: each [ENTITY] contains [N] [ENTITY]
OPERATION: MUL
DSL: entity * value

Sentence: "They split it equally among 4 friends"
PATTERN: [ENTITY] split [ENTITY] equally among [N] [ENTITY]
OPERATION: DIV
DSL: entity / value

Sentence: "She earned $12 per hour for 8 hours"
PATTERN: [ENTITY] earned [N] per [ENTITY] for [N] [ENTITY]
OPERATION: MUL
DSL: entity * value

Sentence: "He has 5 more than Mary"
PATTERN: [ENTITY] has [N] more than [ENTITY]
OPERATION: ADD
DSL: ref + value

Sentence: "She has twice as many as Bob"
PATTERN: [ENTITY] has twice as many as [ENTITY]
OPERATION: MUL
DSL: ref * 2

Now analyze this sentence:
Sentence: "{span}"
"""


def parse_qwen_response(response: str) -> Optional[Dict]:
    """Parse Qwen's response into a dict.

    Qwen2.5-Math-Instruct does chain-of-thought reasoning, so the structured
    PATTERN/OPERATION/DSL lines may appear anywhere in verbose output, or
    in a \\boxed{} block. We search flexibly rather than requiring exact format.

    Key Qwen quirks handled:
    - Bold markdown headers: **PATTERN:**, ** operation:**
    - Abbreviates "DSL" as "sl": ** sl:**, sl:
    - LaTeX wrappers: \\text{OPERATION: ADD}
    - Truncated CoT where operation is mentioned in reasoning
    """
    result = {}

    # Pre-process: strip markdown bold markers and normalize whitespace
    cleaned = re.sub(r'\*\*\s*', '', response)
    # Qwen consistently writes "sl" instead of "DSL" — normalize
    cleaned = re.sub(r'\bsl\b', 'DSL', cleaned, flags=re.IGNORECASE)
    # Strip LaTeX \text{} wrappers
    cleaned = re.sub(r'\\text\{([^}]*)\}', r'\1', cleaned)

    # Strategy 1: Look for PATTERN:/OPERATION:/DSL: lines
    # Qwen outputs multiple PATTERN: lines — first is often a CoT header
    # ("Identify the key elements..."), actual pattern has [ENTITY]/[N] placeholders.
    # Find ALL matches and prefer the one with placeholders.
    pattern_matches = re.findall(r'PATTERN:\s*(.+)', cleaned, re.IGNORECASE)
    if pattern_matches:
        # Prefer match containing [ENTITY] or [N] placeholders
        best = next(
            (m.strip() for m in pattern_matches if '[ENTITY]' in m.upper() or '[N]' in m.upper()),
            pattern_matches[-1].strip()  # fallback to last match (most likely the actual answer)
        )
        result['pattern'] = best

    # For OPERATION, find all matches and take the last (actual answer, not CoT header)
    op_matches = re.findall(r'OPERATION:\s*(SET|ADD|SUB|MUL|DIV)', cleaned, re.IGNORECASE)
    if op_matches:
        result['operation'] = op_matches[-1].strip().upper()

    # For DSL, prefer matches with actual expressions (entity/ref/value/operators)
    dsl_matches = re.findall(r'DSL:\s*(.+)', cleaned, re.IGNORECASE)
    if dsl_matches:
        best_dsl = next(
            (m.strip() for m in dsl_matches
             if re.search(r'entity|ref|value|[\+\-\*/]', m, re.IGNORECASE)),
            dsl_matches[-1].strip()
        )
        result['dsl'] = best_dsl

    if 'pattern' in result and 'operation' in result and 'dsl' in result:
        return _cleanup_result(result)

    # Strategy 2: Parse from chain-of-thought (Qwen says "the operation is ADD/MUL/etc")
    if 'operation' not in result:
        op_context = re.search(
            r'(?:operation\s+(?:is|here is|would be)|operation:\s*)\s*"?\s*(SET|ADD|SUB|MUL|DIV|AD|addition|subtraction|multiplication|division|setting|multiply|divid)',
            cleaned, re.IGNORECASE
        )
        if op_context:
            op_word = op_context.group(1).strip().upper()
            OP_MAP = {
                'ADDITION': 'ADD', 'ADD': 'ADD', 'AD': 'ADD',
                'SUBTRACTION': 'SUB', 'SUB': 'SUB',
                'MULTIPLICATION': 'MUL', 'MUL': 'MUL', 'MULTIPLY': 'MUL',
                'DIVISION': 'DIV', 'DIV': 'DIV', 'DIVID': 'DIV',
                'SETTING': 'SET', 'SET': 'SET',
            }
            result['operation'] = OP_MAP.get(op_word, 'SET')

    # Strategy 3: Infer operation from DSL expression found anywhere
    if 'operation' not in result:
        dsl_anywhere = re.search(
            r'(entity\s*[\+\-\*/]\s*value|ref\s*[\+\-\*/]\s*(?:value|\d+)|value)',
            cleaned, re.IGNORECASE
        )
        if dsl_anywhere:
            dsl_text = dsl_anywhere.group(1).strip().lower()
            if 'dsl' not in result:
                result['dsl'] = dsl_text
            if '+' in dsl_text:
                result['operation'] = 'ADD'
            elif '-' in dsl_text:
                result['operation'] = 'SUB'
            elif '*' in dsl_text:
                result['operation'] = 'MUL'
            elif '/' in dsl_text:
                result['operation'] = 'DIV'
            else:
                result['operation'] = 'SET'

    # Strategy 4: Infer operation from explicit math operators in CoT
    # (conservative — only match actual operator symbols, not descriptive words
    # which appear in Qwen's analysis text and cause false positives)
    if 'operation' not in result:
        # Look for actual computation expressions in the response
        if re.search(r'entity\s*/\s*value|ref\s*/\s*\d+|\bDIV\b', cleaned):
            result['operation'] = 'DIV'
        elif re.search(r'entity\s*\*\s*value|ref\s*\*\s*\d+|\bMUL\b', cleaned):
            result['operation'] = 'MUL'
        elif re.search(r'entity\s*\+\s*value|ref\s*\+\s*\d+|\bADD\b', cleaned):
            result['operation'] = 'ADD'
        elif re.search(r'entity\s*-\s*value|ref\s*-\s*\d+|\bSUB\b', cleaned):
            result['operation'] = 'SUB'
        else:
            # Last resort: default to SET
            result['operation'] = 'SET'

    # Strategy 5: Parse from \boxed{} (Qwen math format)
    boxed = re.search(r'\\boxed\{(.+?)\}', response)
    if boxed:
        boxed_text = boxed.group(1)
        if 'pattern' not in result:
            parts = re.split(r'[.;]', boxed_text)
            if parts:
                result['pattern'] = parts[0].strip()

    # Fill in pattern from ENTITY/N substitution in response
    if 'pattern' not in result:
        entity_pattern = re.search(
            r'(?:pattern\s+is|pattern:)\s*(.+?)(?:\.|$)',
            cleaned, re.IGNORECASE
        )
        if entity_pattern:
            result['pattern'] = entity_pattern.group(1).strip()

    # Default DSL based on operation
    if 'operation' in result and 'dsl' not in result:
        DSL_DEFAULTS = {
            'SET': 'value', 'ADD': 'entity + value', 'SUB': 'entity - value',
            'MUL': 'entity * value', 'DIV': 'entity / value',
        }
        result['dsl'] = DSL_DEFAULTS.get(result['operation'], 'value')

    if 'operation' in result and 'dsl' in result:
        if 'pattern' not in result:
            result['pattern'] = result['dsl']
        return _cleanup_result(result)
    return None


def _cleanup_result(result: Dict) -> Dict:
    """Clean up parsed result — strip CoT prefixes and markdown artifacts."""
    DSL_DEFAULTS = {
        'SET': 'value', 'ADD': 'entity + value', 'SUB': 'entity - value',
        'MUL': 'entity * value', 'DIV': 'entity / value',
    }

    for key in ('pattern', 'dsl'):
        if key in result:
            val = result[key]
            # Remove bullet prefixes like "- The pattern is:"
            val = re.sub(r'^[-*\d.]+\s*(?:The\s+)?(?:pattern|dsl|operation)\s+(?:is|would be):?\s*', '', val, flags=re.IGNORECASE)
            # Remove trailing markdown, LaTeX fragments
            val = re.sub(r'\s*\\?\[?\s*$', '', val)
            val = re.sub(r'\s*\*+\s*$', '', val)
            result[key] = val.strip().rstrip('.')

    # Validate DSL — if it's a garbage CoT sentence, replace with default
    if 'dsl' in result:
        dsl = result['dsl']
        # Valid DSLs are short expressions like "value", "entity + value", "ref * 2"
        is_garbage = (
            len(dsl) > 60
            or 'computation' in dsl.lower()
            or 'expression' in dsl.lower()
            or 'sentence' in dsl.lower()
            or dsl.startswith('- ')
            or dsl.startswith('The ')
            or dsl.startswith('Since ')
            or dsl.startswith('Let')
            or dsl.startswith('Write')
        )
        if is_garbage and 'operation' in result:
            result['dsl'] = DSL_DEFAULTS.get(result['operation'], 'value')

    return result


def generalize_batch_qwen(
    spans: List[str],
    model,
    tokenizer,
    batch_size: int = 16,
) -> List[Optional[Dict]]:
    """Generalize spans using Qwen with true batched inference.

    Args:
        spans: List of raw span texts
        model: Loaded Qwen model
        tokenizer: Loaded Qwen tokenizer
        batch_size: Number of spans per GPU batch

    Returns:
        List of parsed results (or None for failures)
    """
    import torch

    # Pad from left for batched generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []

    for i in tqdm(range(0, len(spans), batch_size), desc="Generalizing"):
        batch = spans[i:i + batch_size]

        # Build all prompts for this batch
        texts = []
        for span in batch:
            prompt = GENERALIZATION_PROMPT.format(span=span)
            messages = [
                {"role": "system", "content": "You analyze math word problem sentences. Always respond with exactly PATTERN, OPERATION, and DSL lines."},
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
                max_new_tokens=300,
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
                {"role": "system", "content": "You analyze math word problem sentences. Always respond with exactly PATTERN, OPERATION, and DSL lines."},
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
                    max_new_tokens=300,
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

        # Clear GPU cache periodically
        if i % (batch_size * 10) == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def regex_fallback_generalize(span: str) -> Dict:
    """Fallback generalization when Qwen fails. Numbers only."""
    pattern = re.sub(r'\$[\d,]+\.?\d*', '[N]', span)
    pattern = re.sub(r'\b\d{1,3}(?:,\d{3})+\.?\d*\b', '[N]', pattern)
    pattern = re.sub(r'\b\d+\.?\d*\b', '[N]', pattern)

    return {
        'pattern': pattern.lower(),
        'operation': 'SET',
        'dsl': 'value',
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
    - problem_id, raw_span, pattern, operation, dsl, problem_text
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
            'operation': gen['operation'],
            'dsl': gen['dsl'],
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
    2. Cluster by embedding similarity within same operation type

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

        # Majority vote for operation and DSL
        ops = [r['operation'] for r in group]
        dsls = [r['dsl'] for r in group]
        majority_op = max(set(ops), key=ops.count)
        majority_dsl = max(set(dsls), key=dsls.count)

        raw_templates.append({
            'pattern': pattern,
            'operation': majority_op,
            'dsl': majority_dsl,
            'centroid': centroid,
            'count': len(group),
            'examples': [r['raw_span'] for r in group[:5]],
            'qwen_fail_rate': sum(1 for r in group if r.get('qwen_failed')) / len(group),
        })

    print(f"  Templates before clustering: {len(raw_templates)}")

    # Stage 2: Cluster by embedding similarity WITHIN same operation type
    # Biased towards MORE templates (high threshold)
    by_op = defaultdict(list)
    for t in raw_templates:
        by_op[t['operation']].append(t)

    final_templates = []
    for op, op_templates in by_op.items():
        # Sort by count descending (most common = best representative)
        op_templates.sort(key=lambda t: -t['count'])

        centroids = np.array([t['centroid'] for t in op_templates])
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms[norms == 0] = 1
        centroids_norm = centroids / norms

        used = set()
        clusters = []

        for i in range(len(op_templates)):
            if i in used:
                continue

            cluster = [i]
            used.add(i)

            for j in range(i + 1, len(op_templates)):
                if j in used:
                    continue
                sim = np.dot(centroids_norm[i], centroids_norm[j])
                if sim >= similarity_threshold:
                    cluster.append(j)
                    used.add(j)

            clusters.append(cluster)

        print(f"  {op}: {len(op_templates)} → {len(clusters)} templates")

        for cluster in clusters:
            # Representative = highest count
            rep_idx = cluster[0]  # Already sorted by count
            rep = op_templates[rep_idx]

            # Merge counts and examples from cluster
            total_count = sum(op_templates[i]['count'] for i in cluster)
            all_examples = []
            for i in cluster:
                all_examples.extend(op_templates[i]['examples'])

            # Recompute centroid from merged cluster
            cluster_centroids = centroids[cluster]
            merged_centroid = np.mean(cluster_centroids, axis=0)
            merged_centroid = merged_centroid / (np.linalg.norm(merged_centroid) + 1e-8)

            template_id = re.sub(r'[^a-z0-9]', '_', rep['pattern'].lower())[:50]
            template_id = f"{rep['operation'].lower()}_{template_id}"

            final_templates.append({
                'template_id': template_id,
                'pattern': rep['pattern'],
                'operation': rep['operation'],
                'base_dsl': rep['dsl'],
                'embedding_centroid': merged_centroid.tolist(),
                'count': total_count,
                'cluster_size': len(cluster),
                'pattern_examples': all_examples[:5],
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
        model_name = "Qwen/Qwen2.5-Math-7B-Instruct"

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
    op_counts = defaultdict(int)
    for r in all_records:
        op_counts[r['operation']] += 1
    print(f"\nOperation distribution:")
    for op, count in sorted(op_counts.items()):
        print(f"  {op}: {count}")

    fail_count = sum(1 for r in all_records if r.get('qwen_failed'))
    print(f"Qwen failures: {fail_count}/{len(all_records)} ({100*fail_count/max(len(all_records),1):.1f}%)")

    # Compute template embeddings and deduplicate
    if not args.skip_embeddings:
        templates = compute_template_embeddings(all_records, args.similarity_threshold)

        print(f"\nDeduplicated to {len(templates)} templates")
        print(f"\nTop 20 templates:")
        for t in templates[:20]:
            print(f"  [{t['operation']:3}] {t['pattern'][:50]:50} (n={t['count']:4}) DSL: {t['base_dsl']}")

        print(f"\nSaving templates to {templates_path}...")
        with open(templates_path, 'w') as f:
            json.dump(templates, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
