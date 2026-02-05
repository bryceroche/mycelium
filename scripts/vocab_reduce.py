#!/usr/bin/env python3
"""
Vocabulary reduction + summarization pipeline for span templates.

1. Build word frequency table from all 25K raw spans
2. Select top-N vocabulary (structural/math words that carry operation semantics)
3. Ask Qwen to rewrite each span using ONLY vocab words + N for numbers
   - Summarize: shorten while preserving the arithmetic operation
   - Normalize: use only common words so patterns collapse
4. Cluster reduced patterns → templates
5. Generate SubGraphDSL per template

USAGE:
    python scripts/vocab_reduce.py --tp-size 4 --vocab-size 300
    python scripts/vocab_reduce.py --tp-size 4 --skip-to-cluster  # resume
    python scripts/vocab_reduce.py --tp-size 4 --skip-to-dsl      # resume
"""

import argparse
import json
import re
import os
import sys
import time
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

OUTPUT_DIR = Path(__file__).parent.parent


# =============================================================================
# Step 1: Build vocabulary
# =============================================================================

def build_vocab(records: List[Dict], vocab_size: int = 300) -> List[str]:
    """Build top-N vocabulary from raw spans.

    Keeps structural/math words that carry operation semantics.
    Filters out stopwords that don't help distinguish operations.
    """
    word_counts = Counter()
    for r in records:
        words = re.findall(r'[a-z]+', r.get('raw_span', '').lower())
        word_counts.update(words)

    # Always include these math-structural words even if rare
    FORCE_INCLUDE = {
        'has', 'have', 'had', 'gave', 'gives', 'give', 'got', 'gets', 'get',
        'bought', 'buys', 'buy', 'sold', 'sells', 'sell',
        'spent', 'spends', 'spend', 'paid', 'pays', 'pay',
        'earned', 'earns', 'earn', 'makes', 'made', 'make',
        'ate', 'eats', 'eat', 'drinks', 'drank', 'drink',
        'lost', 'loses', 'lose', 'found', 'finds', 'find',
        'took', 'takes', 'take', 'added', 'adds', 'add',
        'left', 'remaining', 'rest',
        'each', 'every', 'per', 'total', 'all',
        'more', 'less', 'fewer', 'extra', 'additional',
        'twice', 'half', 'double', 'triple', 'third', 'quarter',
        'times', 'percent', 'fraction',
        'cost', 'costs', 'price', 'worth', 'value',
        'how', 'many', 'much', 'what', 'number',
        'than', 'as', 'if', 'then', 'after', 'before',
        'first', 'second', 'third', 'last', 'next',
        'day', 'week', 'month', 'year', 'hour', 'minute',
        'split', 'divide', 'divided', 'share', 'shared',
        'together', 'combined', 'equal', 'equally',
        'needs', 'need', 'wants', 'want',
        'from', 'to', 'for', 'with', 'on', 'in', 'at', 'by',
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'will', 'be',
        'and', 'or', 'but', 'not', 'no',
        'he', 'she', 'it', 'they', 'his', 'her', 'their', 'its',
        'of', 'that', 'this', 'there', 'does', 'did', 'do',
        'can', 'one', 'two', 'three', 'four', 'five',
        'six', 'seven', 'eight', 'nine', 'ten',
        'some', 'same', 'other', 'another', 'both',
        'now', 'still', 'also', 'only', 'already',
        'person', 'people', 'items', 'things',
        'N',  # placeholder for numbers
    }

    # Take top vocab_size by frequency, then add forced words
    top_words = [w for w, _ in word_counts.most_common(vocab_size)]
    vocab = list(set(top_words) | FORCE_INCLUDE)
    vocab.sort(key=lambda w: word_counts.get(w, 0), reverse=True)

    return vocab


# =============================================================================
# Step 2: Qwen rewrite with constrained vocabulary
# =============================================================================

def load_qwen(tp_size: int = 4):
    """Load Qwen via vLLM."""
    from vllm import LLM, SamplingParams
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
    )
    return llm


def unload_qwen(engine):
    """Free GPU memory."""
    import gc, torch
    del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def batch_qwen(llm, prompts: List[str], max_tokens: int = 100) -> List[str]:
    """Run batch inference."""
    from vllm import SamplingParams
    params = SamplingParams(
        temperature=0.1,
        max_tokens=max_tokens,
        stop=["\n\n", "```", "\n"],  # Single newline stop — one line output only
    )
    outputs = llm.generate(prompts, params)
    return [o.outputs[0].text.strip() for o in outputs]


def build_rewrite_prompt(vocab_words: str) -> str:
    """Build the rewrite prompt template with the vocab list baked in."""
    return f"""Rewrite this math problem span as a short summary using ONLY common words.
Rules:
- Replace all names with "person"
- Replace all specific objects with "items"
- Replace all numbers/amounts with N
- Replace locations with "place"
- Keep the math operation words (bought, sold, gave, earned, split, each, per, more, less, twice, half, etc.)
- Make it as SHORT as possible while preserving what arithmetic operation happens
- Use ONLY these allowed words: {vocab_words}
- Output ONLY the rewritten span, nothing else

Examples:
"Janet's ducks lay 16 eggs per day" → person has N items per day
"She eats three for breakfast every morning" → person eats N each day
"He buys 3 sodas at $2 each" → person buys N items at N each
"The cost of the dessert is 25% of the price of the second course" → items cost N percent of items
"Josh decides to try flipping a house" → person buys items
"How many apples does John have left?" → how many items does person have left
"On Tuesday, she bought 4 boxes at the grocery store" → person bought N items at place
"If each pack weighs 250 grams and has 20 grams left" → each items has N and has N left
"Sam is serving spaghetti and meatballs for dinner" → person has items

Span: "{{span}}"
Summary: """


def clean_rewrite(raw: str) -> str:
    """Clean Qwen's rewrite output."""
    p = raw.strip()
    # Take first line only
    p = p.split('\n')[0].strip()
    # Remove quotes and artifacts
    p = p.strip('"\'')
    p = re.sub(r'^(Summary:\s*|Output:\s*|Rewrite:\s*)', '', p, flags=re.IGNORECASE)
    p = p.strip('"\'')
    # Normalize whitespace
    p = ' '.join(p.split())
    # Lowercase
    p = p.lower()
    return p.strip()


def is_valid_rewrite(rewrite: str, raw_span: str) -> bool:
    """Check if rewrite is valid."""
    if not rewrite or len(rewrite) < 3:
        return False
    # Should be shorter than or roughly equal to raw span
    if len(rewrite) > len(raw_span) * 1.5:
        return False
    # Should not be identical to raw span
    if rewrite == raw_span.lower():
        return False
    # Should contain at least one word
    if not re.search(r'[a-z]', rewrite):
        return False
    return True


def rewrite_spans(llm, records: List[Dict], vocab: List[str],
                  batch_size: int = 2000) -> List[Dict]:
    """Rewrite all spans using vocab-constrained summarization."""
    # Build vocab string (show top 200 to keep prompt short)
    vocab_words = ", ".join(vocab[:200])
    prompt_template = build_rewrite_prompt(vocab_words)

    col = "reduced"
    prompts_needed = [(i, r) for i, r in enumerate(records) if col not in r]

    if not prompts_needed:
        print("  All records already have reduced column — skipping")
        return records

    print(f"  Rewriting {len(prompts_needed)} spans with vocab reduction...")
    failures = 0

    for batch_start in range(0, len(prompts_needed), batch_size):
        batch = prompts_needed[batch_start:batch_start + batch_size]
        prompts = [prompt_template.format(span=r['raw_span']) for _, r in batch]

        responses = batch_qwen(llm, prompts, max_tokens=80)

        for (idx, record), response in zip(batch, responses):
            rewrite = clean_rewrite(response)
            if is_valid_rewrite(rewrite, record['raw_span']):
                records[idx][col] = rewrite
            else:
                # Fallback: basic programmatic reduction
                records[idx][col] = programmatic_reduce(record['raw_span'], set(vocab))
                failures += 1

        done = min(batch_start + batch_size, len(prompts_needed))
        pct_fail = failures / done * 100 if done > 0 else 0
        print(f"  [{done}/{len(prompts_needed)}] failures: {failures} ({pct_fail:.1f}%)")

    # Stats
    patterns = Counter(r.get(col, '') for r in records)
    singletons = sum(1 for c in patterns.values() if c == 1)
    print(f"\n  Rewrite results:")
    print(f"    Unique patterns: {len(patterns)}")
    print(f"    Singletons: {singletons} ({singletons/len(patterns)*100:.1f}%)")
    print(f"    Failures: {failures}")
    print(f"    Top 10 patterns:")
    for pat, count in patterns.most_common(10):
        print(f"      [{count:4d}] {pat}")

    return records


def programmatic_reduce(span: str, vocab_set: set) -> str:
    """Fallback: programmatic vocab reduction without Qwen."""
    words = re.findall(r'[a-z]+', span.lower())
    # Replace numbers with N
    result = re.sub(r'\d+[\d,.]*', 'N', span.lower())
    result = re.sub(r'\$\s*N', 'N', result)
    # Replace non-vocab words
    tokens = result.split()
    reduced = []
    for t in tokens:
        clean_t = re.sub(r'[^a-z]', '', t)
        if clean_t in vocab_set or t == 'N' or not clean_t:
            reduced.append(t)
        # Skip non-vocab words entirely (they're noise)
    return ' '.join(reduced) if reduced else 'items'


# =============================================================================
# Step 3: Embed + Cluster
# =============================================================================

def embed_and_cluster(records: List[Dict], col: str = "reduced",
                      target_count: int = 1000) -> List[Dict]:
    """Embed reduced patterns with MiniLM and cluster to target_count."""
    print(f"\n{'='*60}")
    print(f"EMBEDDING + CLUSTERING: {col} → ~{target_count} templates")
    print(f"{'='*60}")

    from mycelium.dual_signal_templates import SpanDetector

    detector = SpanDetector(model_path=None, device="cuda")

    # Group by pattern
    pattern_groups = defaultdict(list)
    for r in records:
        pattern = r.get(col, r.get('raw_span', ''))
        pattern_groups[pattern].append(r)

    patterns = list(pattern_groups.keys())
    print(f"  Unique patterns to embed: {len(patterns)}")

    # Embed each unique pattern
    embeddings = []
    for i, p in enumerate(patterns):
        emb, _, _ = detector.extract_features(p)
        embeddings.append(emb / (np.linalg.norm(emb) + 1e-8))
        if (i + 1) % 1000 == 0:
            print(f"  Embedded {i+1}/{len(patterns)}")

    embeddings = np.array(embeddings)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Binary search for threshold
    lo, hi = 0.5, 0.99
    best_thresh, best_count = lo, len(patterns)

    for _ in range(30):
        mid = (lo + hi) / 2
        assigned = [False] * len(patterns)
        cluster_count = 0
        for i in range(len(patterns)):
            if assigned[i]:
                continue
            cluster_count += 1
            for j in range(i + 1, len(patterns)):
                if not assigned[j]:
                    sim = np.dot(embeddings[i], embeddings[j])
                    if sim >= mid:
                        assigned[j] = True
        if cluster_count > target_count:
            lo = mid
        else:
            hi = mid
        best_thresh = mid
        best_count = cluster_count
        if abs(cluster_count - target_count) < 50:
            break

    print(f"  Threshold: {best_thresh:.4f} → {best_count} clusters")

    # Full clustering
    assigned = [-1] * len(patterns)
    cluster_id = 0
    clusters = {}

    for i in range(len(patterns)):
        if assigned[i] >= 0:
            continue
        cluster_members = [i]
        assigned[i] = cluster_id
        for j in range(i + 1, len(patterns)):
            if assigned[j] < 0:
                sim = np.dot(embeddings[i], embeddings[j])
                if sim >= best_thresh:
                    assigned[j] = cluster_id
                    cluster_members.append(j)
        clusters[cluster_id] = cluster_members
        cluster_id += 1

    print(f"  Final clusters: {cluster_id}")

    # Build templates
    templates = []
    for cid, member_indices in clusters.items():
        rep_idx = max(member_indices, key=lambda i: len(pattern_groups[patterns[i]]))
        rep_pattern = patterns[rep_idx]

        all_raw_spans = []
        all_patterns = []
        for mi in member_indices:
            p = patterns[mi]
            all_patterns.append(p)
            for r in pattern_groups[p]:
                all_raw_spans.append(r.get('raw_span', ''))

        centroid = np.mean([embeddings[mi] for mi in member_indices], axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

        templates.append({
            "template_id": f"tpl_{cid:04d}",
            "pattern": rep_pattern,
            "all_patterns": all_patterns[:10],
            "centroid": centroid.tolist(),
            "span_examples": all_raw_spans[:20],
            "member_count": len(all_raw_spans),
            "subgraph": None,
        })

    print(f"  Templates created: {len(templates)}")

    output_path = OUTPUT_DIR / "templates_vocab_reduced.json"
    with open(output_path, "w") as f:
        json.dump(templates, f, indent=2)
    print(f"  Saved to {output_path}")

    return templates


# =============================================================================
# Step 4: SubGraphDSL generation
# =============================================================================

DSL_PROMPT = """You are writing a SubGraphDSL for a math word problem pattern.

A SubGraphDSL defines the computation a span performs as JSON:
- "params": values extracted from the span text (numbers: n1, n2, ...)
- "inputs": values from upstream spans (use "upstream" for running entity value)
- "steps": ordered computation steps, each with "var", "op", "args"
- "output": which variable is exposed downstream

Allowed operators: SET (1 arg), ADD (2 args), SUB (2 args), MUL (2 args), DIV (2 args), NEG (1 arg)
Args can be variable names (strings) or literal numbers (floats).

Output ONLY valid JSON. No explanation.

Examples:

Pattern: "person has N items"
{{"params": {{"n1": "quantity"}}, "inputs": {{}}, "steps": [{{"var": "out", "op": "SET", "args": ["n1"]}}], "output": "out"}}

Pattern: "person gave N items to person"
{{"params": {{"n1": "quantity given"}}, "inputs": {{"upstream": "giver total"}}, "steps": [{{"var": "out", "op": "SUB", "args": ["upstream", "n1"]}}], "output": "out"}}

Pattern: "person buys N items at N each"
{{"params": {{"n1": "quantity", "n2": "price each"}}, "inputs": {{}}, "steps": [{{"var": "out", "op": "MUL", "args": ["n1", "n2"]}}], "output": "out"}}

Pattern: "person earns N percent more"
{{"params": {{"n1": "percentage"}}, "inputs": {{"upstream": "base amount"}}, "steps": [{{"var": "pct", "op": "DIV", "args": ["n1", 100]}}, {{"var": "bonus", "op": "MUL", "args": ["upstream", "pct"]}}, {{"var": "out", "op": "ADD", "args": ["upstream", "bonus"]}}], "output": "out"}}

Pattern: "N items split equally N"
{{"params": {{"n1": "total", "n2": "groups"}}, "inputs": {{}}, "steps": [{{"var": "out", "op": "DIV", "args": ["n1", "n2"]}}], "output": "out"}}

Pattern: "how many items does person have left"
{{"params": {{}}, "inputs": {{"upstream": "current total"}}, "steps": [{{"var": "out", "op": "SET", "args": ["upstream"]}}], "output": "out"}}

Pattern: "{pattern}"
Example spans:
{examples}
"""


def parse_subgraph_json(response: str) -> Optional[Dict]:
    """Parse Qwen's JSON response."""
    cleaned = response.strip()
    brace_start = cleaned.find('{')
    if brace_start < 0:
        return None

    depth = 0
    for i in range(brace_start, len(cleaned)):
        if cleaned[i] == '{':
            depth += 1
        elif cleaned[i] == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[brace_start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def generate_dsl(llm, templates: List[Dict], batch_size: int = 500) -> List[Dict]:
    """Generate SubGraphDSL JSON for each template."""
    from mycelium.subgraph_dsl import SubGraphDSL

    print(f"\n{'='*60}")
    print(f"DSL GENERATION (SubGraphDSL): {len(templates)} templates")
    print(f"{'='*60}")

    prompts_needed = [(i, t) for i, t in enumerate(templates) if not t.get('subgraph')]

    if not prompts_needed:
        print("  All templates already have subgraph DSL — skipping")
        return templates

    print(f"  Generating SubGraphDSL for {len(prompts_needed)} templates...")
    valid_count = 0
    fallback_count = 0

    for batch_start in range(0, len(prompts_needed), batch_size):
        batch = prompts_needed[batch_start:batch_start + batch_size]
        prompts = []

        for idx, tpl in batch:
            examples = tpl.get('span_examples', [])[:5]
            examples_str = "\n".join(f'  "{ex}"' for ex in examples)
            prompts.append(DSL_PROMPT.format(
                pattern=tpl['pattern'],
                examples=examples_str or '  (no examples)',
            ))

        responses = batch_qwen(llm, prompts, max_tokens=300)

        for (idx, tpl), response in zip(batch, responses):
            parsed = parse_subgraph_json(response)

            if parsed:
                parsed['template_id'] = tpl['template_id']
                parsed['pattern'] = tpl['pattern']
                try:
                    dsl = SubGraphDSL.from_dict(parsed)
                    errors = dsl.validate()
                    if not errors:
                        templates[idx]['subgraph'] = dsl.to_dict()
                        valid_count += 1
                        continue
                except Exception:
                    pass

            # Fallback
            fb = {
                "template_id": tpl['template_id'],
                "pattern": tpl['pattern'],
                "params": {"n1": "value"},
                "inputs": {},
                "steps": [{"var": "out", "op": "SET", "args": ["n1"]}],
                "output": "out",
            }
            templates[idx]['subgraph'] = fb
            fallback_count += 1

        done = min(batch_start + batch_size, len(prompts_needed))
        print(f"  [{done}/{len(prompts_needed)}] valid={valid_count} fallback={fallback_count}")

    # Stats
    def structure_key(tpl):
        sg = tpl.get('subgraph', {})
        steps = sg.get('steps', [])
        return tuple((s['op'], len(s['args'])) for s in steps)

    structure_counter = Counter(structure_key(t) for t in templates)
    print(f"\n  Valid DSLs: {valid_count} ({valid_count/len(templates)*100:.1f}%)")
    print(f"  Fallbacks: {fallback_count} ({fallback_count/len(templates)*100:.1f}%)")
    print(f"  Unique sub-graph structures: {len(structure_counter)}")
    print(f"  Top 10 structures:")
    for struct, count in structure_counter.most_common(10):
        ops_str = " → ".join(f"{op}({n})" for op, n in struct)
        print(f"    [{count:4d}] {ops_str}")

    # Save
    output_path = OUTPUT_DIR / "templates_final.json"
    with open(output_path, "w") as f:
        json.dump(templates, f, indent=2)
    print(f"\n  Saved to {output_path}")

    dsls_path = OUTPUT_DIR / "subgraph_dsls.json"
    dsls = [t['subgraph'] for t in templates if t.get('subgraph')]
    with open(dsls_path, "w") as f:
        json.dump(dsls, f, indent=2)
    print(f"  Standalone DSLs saved to {dsls_path}")

    return templates


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Vocab Reduction Pipeline")
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=300)
    parser.add_argument("--target-templates", type=int, default=1000)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--skip-to-cluster", action="store_true")
    parser.add_argument("--skip-to-dsl", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("Vocabulary Reduction Pipeline")
    print("=" * 60)

    # Load records
    input_path = Path(args.input) if args.input else OUTPUT_DIR / "generalized_v2.json"
    # Check for checkpoint
    reduced_path = OUTPUT_DIR / "vocab_reduced_records.json"
    if reduced_path.exists() and (args.skip_to_cluster or args.skip_to_dsl):
        input_path = reduced_path

    print(f"\nLoading from: {input_path}")
    with open(input_path) as f:
        records = json.load(f)
    print(f"Records: {len(records)}")

    if args.skip_to_dsl:
        tpl_path = OUTPUT_DIR / "templates_vocab_reduced.json"
        print(f"\nLoading templates from: {tpl_path}")
        with open(tpl_path) as f:
            templates = json.load(f)
        llm = load_qwen(args.tp_size)
        templates = generate_dsl(llm, templates)
        unload_qwen(llm)
        return

    if not args.skip_to_cluster:
        # Step 1: Build vocabulary
        print(f"\n{'='*60}")
        print(f"STEP 1: Build vocabulary (top {args.vocab_size})")
        print(f"{'='*60}")
        vocab = build_vocab(records, vocab_size=args.vocab_size)
        print(f"  Vocabulary size: {len(vocab)}")
        print(f"  Sample: {vocab[:30]}")

        # Save vocab
        with open(OUTPUT_DIR / "vocab.json", "w") as f:
            json.dump(vocab, f, indent=2)

        # Step 2: Rewrite spans
        print(f"\n{'='*60}")
        print(f"STEP 2: Rewrite spans with vocab reduction")
        print(f"{'='*60}")
        llm = load_qwen(args.tp_size)
        records = rewrite_spans(llm, records, vocab)
        unload_qwen(llm)

        # Save checkpoint
        with open(reduced_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"\n  Checkpoint saved to {reduced_path}")

    # Step 3: Embed + Cluster
    templates = embed_and_cluster(records, col="reduced",
                                  target_count=args.target_templates)

    # Step 4: DSL Generation
    llm = load_qwen(args.tp_size)
    templates = generate_dsl(llm, templates)
    unload_qwen(llm)

    # Final summary
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Input records: {len(records)}")
    print(f"  Final templates: {len(templates)}")
    valid = sum(1 for t in templates if t.get('subgraph', {}).get('steps'))
    print(f"  Valid SubGraphDSLs: {valid}")
    unique_structs = len(set(
        tuple((s['op'], len(s['args'])) for s in t.get('subgraph', {}).get('steps', []))
        for t in templates
    ))
    print(f"  Unique DSL structures: {unique_structs}")
    print(f"  Output: {OUTPUT_DIR / 'templates_final.json'}")


if __name__ == "__main__":
    main()
