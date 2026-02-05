#!/usr/bin/env python3
"""
Iterative generalization pipeline for template creation.

Each pass targets singletons from the previous clustering:
  raw_span → gen_1 (all) → cluster → singletons
             gen_2 (singletons only, ref gen_1) → cluster → singletons
             gen_3 (singletons only, ref gen_2) → cluster → singletons
             ...stop when singleton count stabilizes

Non-singletons carry forward their previous generalization.
Each pass generates a new column, preserving the audit trail.

After generalization converges, embeds final patterns with MiniLM,
clusters to ~1K templates, and generates free-form DSL sub-graphs.

USAGE:
    # Full pipeline on VM with 4 GPUs:
    python scripts/iterative_generalize.py --tp-size 4

    # Resume from a specific pass:
    python scripts/iterative_generalize.py --tp-size 4 --resume-from 2

    # Skip to DSL generation (after generalization is done):
    python scripts/iterative_generalize.py --tp-size 4 --skip-to-dsl
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
CHECKPOINT_FILE = OUTPUT_DIR / "iterative_gen_checkpoint.json"


# =============================================================================
# Qwen interaction (vLLM)
# =============================================================================

def load_qwen(tp_size: int = 4):
    """Load Qwen via vLLM."""
    from vllm import LLM, SamplingParams
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
        max_model_len=512,
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


def batch_qwen(llm, prompts: List[str], max_tokens: int = 150) -> List[str]:
    """Run batch inference on Qwen."""
    from vllm import SamplingParams
    params = SamplingParams(
        temperature=0.1,
        max_tokens=max_tokens,
        stop=["\n\n", "```"],
    )
    outputs = llm.generate(prompts, params)
    return [o.outputs[0].text.strip() for o in outputs]


# =============================================================================
# Generalization prompts — one per pass
# =============================================================================

PASS_1_PROMPT = """Replace specific details with generic tokens. Rules:
- Names/pronouns → [PERSON1],[PERSON2],etc.
- Objects/nouns → [ITEM1],[ITEM2],etc.
- Numbers/amounts → [N]
- Locations/places → [LOC1],[LOC2],etc.
- Time references → [TIME1],[TIME2],etc.
Keep verbs and sentence structure. Output ONLY the generalized pattern, nothing else.

Examples:
"John has 5 apples" → [PERSON1] has [N] [ITEM1]
"She gave 2 cookies to Mary at the store" → [PERSON1] gave [N] [ITEM1] to [PERSON2] at [LOC1]
"He walks 3 miles to school every morning" → [PERSON1] walks [N] [ITEM1] to [LOC1] every [TIME1]
"On Tuesday, she bought 4 boxes at the grocery store" → on [TIME1], [PERSON1] bought [N] [ITEM1] at [LOC1]
"The farmers market sells oranges for $2 each" → [LOC1] sells [ITEM1] for [N] each
"Last week he spent $50 at the gym" → [TIME1] [PERSON1] spent [N] at [LOC1]
"Each bag contains 5 items" → each [ITEM1] contains [N] [ITEM2]

Input: "{span}"
Output: """

PASS_N_PROMPT = """Generalize this math problem span by replacing specifics with tokens.

Rules:
- Names/pronouns → [PERSON1],[PERSON2],etc.
- Objects/nouns → [ITEM1],[ITEM2],etc.
- Numbers/amounts → [N]
- Locations/places → [LOC1],[LOC2],etc.
- Time references → [TIME1],[TIME2],etc.
- Units of measure (feet, miles, dollars, percent) → [UNIT1]
- Containers (bag, box, jar, can) → [ITEM1]
Keep verbs and sentence structure. Output ONLY the generalized pattern, nothing else.

Previous attempt (too specific, try harder): "{prev_pattern}"
Raw span: "{raw_span}"
Better generalization: """


# =============================================================================
# Pattern cleaning
# =============================================================================

def clean_pattern(raw: str) -> str:
    """Clean Qwen output into a usable pattern.

    Qwen often repeats itself or adds "Output:" prefixes.
    Take only the FIRST meaningful line.
    """
    p = raw.strip()

    # Take first line only (Qwen repeats on subsequent lines)
    first_line = p.split('\n')[0].strip()

    # If first line contains "Output:", split on it and take last part
    if 'Output:' in first_line or 'output:' in first_line:
        parts = re.split(r'[Oo]utput:\s*', first_line)
        first_line = parts[-1].strip()

    # Also handle inline repetition: take text before first "Output:" or duplicate
    # Pattern: real output followed by "Output: <repeat>"
    first_line = re.split(r'\s+[Oo]utput:\s*', first_line)[0]

    # Remove leading/trailing artifacts
    p = first_line.strip('"\'')
    p = re.sub(r'^(Output:\s*|Pattern:\s*|Result:\s*|→\s*)', '', p, flags=re.IGNORECASE)
    p = p.strip('"\'')

    # Fix common bracket errors
    p = re.sub(r'\[([A-Z]+)(\d*)\s*\]', r'[\1\2]', p)  # Fix spaces inside brackets
    p = re.sub(r'\[\s+', '[', p)
    p = re.sub(r'\s+\]', ']', p)

    # Truncate runaway token sequences (e.g., [TIME1] [TIME2] [TIME3]...)
    # If we see more than 3 consecutive generic tokens, truncate
    p = re.sub(r'(\s*\[[A-Z]+\d*\]){4,}', '', p)

    # Remove trailing punctuation artifacts
    p = re.sub(r'[}\s"\']+$', '', p)
    p = p.rstrip('.')

    # Normalize whitespace
    p = ' '.join(p.split())
    return p.strip()


def is_valid_pattern(pattern: str, raw_span: str) -> bool:
    """Check if a generalized pattern is valid."""
    if not pattern or len(pattern) < 3:
        return False
    # Should have at least one generic token
    if not re.search(r'\[(?:PERSON|ITEM|N|LOC|TIME|UNIT|ROLE)\d*\]', pattern):
        return False
    # Should not be longer than 3x the raw span (patterns can be slightly longer)
    if len(pattern) > len(raw_span) * 3:
        return False
    # Should not be the raw span itself
    if pattern.lower() == raw_span.lower():
        return False
    return True


# =============================================================================
# Clustering (pattern-level, no embeddings needed)
# =============================================================================

def cluster_patterns(records: List[Dict], gen_col: str) -> Tuple[Counter, Dict[str, List[int]]]:
    """Cluster records by their pattern in gen_col. Returns (counts, pattern→record_indices)."""
    pattern_groups = defaultdict(list)
    for idx, r in enumerate(records):
        pattern = r.get(gen_col, r.get('raw_span', ''))
        pattern_groups[pattern].append(idx)

    counts = Counter({p: len(idxs) for p, idxs in pattern_groups.items()})
    return counts, dict(pattern_groups)


def get_small_group_indices(records: List[Dict], gen_col: str, min_count: int = 2) -> List[int]:
    """Get indices of records whose pattern group size < min_count.

    Like SQL: SELECT gen_col, COUNT(*) FROM records GROUP BY gen_col HAVING COUNT(*) < min_count
    """
    counts, groups = cluster_patterns(records, gen_col)
    small_patterns = {p for p, c in counts.items() if c < min_count}
    small_indices = []
    for p in small_patterns:
        small_indices.extend(groups[p])
    return small_indices


# =============================================================================
# Generalization passes
# =============================================================================

def seed_from_existing(records: List[Dict]) -> List[Dict]:
    """Seed gen_0 from the existing 'pattern' column (previous Qwen run).

    Only call Qwen on singletons — never re-generalize spans that already
    collapsed into good clusters.
    """
    col = "gen_0"
    print(f"\n{'='*60}")
    print(f"SEED: Using existing 'pattern' column as {col}")
    print(f"{'='*60}")

    for r in records:
        # Use existing pattern if available, otherwise raw span
        existing = r.get('pattern', r.get('pattern_v1', r.get('raw_span', '')))
        r[col] = existing

    counts, _ = cluster_patterns(records, col)
    singletons = sum(1 for c in counts.values() if c == 1)
    non_singletons = len(counts) - singletons

    print(f"  Records: {len(records)}")
    print(f"  Unique patterns: {len(counts)}")
    print(f"  Already collapsed (count>1): {non_singletons} patterns")
    print(f"  Singletons to target: {singletons} ({singletons/len(counts)*100:.1f}%)")

    return records


def run_pass_n(llm, records: List[Dict], pass_num: int, min_group_size: int = 2,
               batch_size: int = 2000) -> List[Dict]:
    """Pass N (N>=2): Target small groups (< min_group_size) from previous pass.

    Like: SELECT gen_N-1, COUNT(*) GROUP BY gen_N-1 HAVING COUNT(*) < min_group_size
    Those records get re-generalized. The rest carry forward.
    """
    prev_col = f"gen_{pass_num - 1}"
    curr_col = f"gen_{pass_num}"

    # Get records in small groups from previous pass
    singleton_indices = get_small_group_indices(records, prev_col, min_count=min_group_size)

    print(f"\n{'='*60}")
    print(f"PASS {pass_num}: Target {len(singleton_indices)} records with group_size < {min_group_size} from {prev_col}")
    print(f"{'='*60}")

    if not singleton_indices:
        print("  No singletons — nothing to do")
        return records

    # Carry forward non-singletons
    for i, r in enumerate(records):
        if i not in set(singleton_indices):
            r[curr_col] = r.get(prev_col, r.get('raw_span', ''))

    # Build prompts for singletons
    # Group singletons by rough similarity for context
    prompts_needed = [(i, records[i]) for i in singleton_indices if curr_col not in records[i]]

    if not prompts_needed:
        print(f"  Already done — {curr_col} column exists for all singletons")
        return records

    print(f"  Processing {len(prompts_needed)} singleton patterns...")
    failures = 0

    for batch_start in range(0, len(prompts_needed), batch_size):
        batch = prompts_needed[batch_start:batch_start + batch_size]
        prompts = []

        for idx, record in batch:
            prev_pattern = record.get(prev_col, record.get('raw_span', ''))
            raw_span = record.get('raw_span', '')

            # If prev pattern IS the raw span (no prior generalization), use Pass 1 prompt
            if prev_pattern == raw_span or not re.search(r'\[', prev_pattern):
                prompt = PASS_1_PROMPT.format(span=raw_span)
            else:
                prompt = PASS_N_PROMPT.format(
                    prev_pattern=prev_pattern,
                    raw_span=raw_span,
                )
            prompts.append(prompt)

        responses = batch_qwen(llm, prompts, max_tokens=150)

        for (idx, record), response in zip(batch, responses):
            pattern = clean_pattern(response)
            raw = record.get('raw_span', '')
            if is_valid_pattern(pattern, raw):
                records[idx][curr_col] = pattern
            else:
                # Carry forward previous
                records[idx][curr_col] = record.get(prev_col, raw)
                failures += 1

        done = min(batch_start + batch_size, len(prompts_needed))
        print(f"  [{done}/{len(prompts_needed)}] failures so far: {failures}")

    # Stats
    counts, _ = cluster_patterns(records, curr_col)
    singletons = sum(1 for c in counts.values() if c == 1)
    print(f"\n  Pass {pass_num} results:")
    print(f"    Unique patterns: {len(counts)}")
    print(f"    Singletons: {singletons} ({singletons/len(counts)*100:.1f}%)")
    print(f"    Failures: {failures}")

    return records


# =============================================================================
# Embedding + Clustering to ~1K
# =============================================================================

def embed_and_cluster(records: List[Dict], final_col: str, target_count: int = 1000) -> List[Dict]:
    """Embed final patterns with MiniLM and cluster to target_count templates."""
    print(f"\n{'='*60}")
    print(f"EMBEDDING + CLUSTERING: {final_col} → ~{target_count} templates")
    print(f"{'='*60}")

    from mycelium.dual_signal_templates import SpanDetector

    detector = SpanDetector(model_path=None, device="cuda")

    # Group by pattern
    pattern_groups = defaultdict(list)
    for r in records:
        pattern = r.get(final_col, r.get('raw_span', ''))
        pattern_groups[pattern].append(r)

    print(f"  Unique patterns to embed: {len(pattern_groups)}")

    # Embed each unique pattern
    patterns = list(pattern_groups.keys())
    embeddings = []
    batch_size = 64

    for i in range(0, len(patterns), batch_size):
        batch = patterns[i:i + batch_size]
        for p in batch:
            emb, _, _ = detector.extract_features(p)
            embeddings.append(emb / (np.linalg.norm(emb) + 1e-8))
        if (i + batch_size) % 1000 == 0:
            print(f"  Embedded {min(i + batch_size, len(patterns))}/{len(patterns)}")

    embeddings = np.array(embeddings)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Find threshold for target count via binary search
    lo, hi = 0.5, 0.99
    best_thresh, best_count = lo, len(patterns)

    for _ in range(30):
        mid = (lo + hi) / 2
        # Quick count: greedy clustering
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

    # Full clustering at best threshold
    assigned = [-1] * len(patterns)
    cluster_id = 0
    clusters = {}  # cluster_id → list of pattern indices

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
        # Representative = pattern with most raw span examples
        rep_idx = max(member_indices, key=lambda i: len(pattern_groups[patterns[i]]))
        rep_pattern = patterns[rep_idx]

        # Collect all raw spans from all patterns in this cluster
        all_raw_spans = []
        all_patterns = []
        for mi in member_indices:
            p = patterns[mi]
            all_patterns.append(p)
            for r in pattern_groups[p]:
                all_raw_spans.append(r.get('raw_span', ''))

        # Centroid = mean of member embeddings
        centroid = np.mean([embeddings[mi] for mi in member_indices], axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

        templates.append({
            "template_id": f"tpl_{cid:04d}",
            "pattern": rep_pattern,
            "all_patterns": all_patterns[:10],  # Keep up to 10 for DSL context
            "centroid": centroid.tolist(),
            "span_examples": all_raw_spans[:20],  # Keep up to 20 for DSL context
            "member_count": len(all_raw_spans),
            "dsl_expr": "",  # To be filled by DSL generation
        })

    print(f"  Templates created: {len(templates)}")

    # Save
    output_path = OUTPUT_DIR / "templates_clustered.json"
    with open(output_path, "w") as f:
        json.dump(templates, f, indent=2)
    print(f"  Saved to {output_path}")

    return templates


# =============================================================================
# DSL Generation — free-form, unique per template
# =============================================================================

DSL_PROMPT = """You are writing a SubGraphDSL for a math word problem pattern.

A SubGraphDSL defines the computation a span performs:
- "params": values extracted from the span text (numbers in order: n1, n2, ...)
- "inputs": values wired from upstream spans (use "upstream" for the running entity value)
- "steps": ordered computation steps, each with "var", "op", "args"
- "output": which variable is exposed downstream

Allowed operators: SET (1 arg), ADD (2 args), SUB (2 args), MUL (2 args), DIV (2 args), MOD (2 args), NEG (1 arg)
Args can be variable names (strings) or literal numbers (floats).

Output ONLY valid JSON matching this format. No explanation.

Examples:

Pattern: "[PERSON1] has [N] [ITEM1]"
{{"params": {{"n1": "quantity"}}, "inputs": {{}}, "steps": [{{"var": "out", "op": "SET", "args": ["n1"]}}], "output": "out"}}

Pattern: "[PERSON1] gives [N] [ITEM1] to [PERSON2]"
{{"params": {{"n1": "quantity given"}}, "inputs": {{"upstream": "giver current total"}}, "steps": [{{"var": "out", "op": "SUB", "args": ["upstream", "n1"]}}], "output": "out"}}

Pattern: "[PERSON1] buys [N] [ITEM1] at [N] each"
{{"params": {{"n1": "quantity", "n2": "price per item"}}, "inputs": {{}}, "steps": [{{"var": "out", "op": "MUL", "args": ["n1", "n2"]}}], "output": "out"}}

Pattern: "[PERSON1] earns [N]% more than [PERSON2]"
{{"params": {{"n1": "percentage"}}, "inputs": {{"upstream": "reference amount"}}, "steps": [{{"var": "pct", "op": "DIV", "args": ["n1", 100]}}, {{"var": "bonus", "op": "MUL", "args": ["upstream", "pct"]}}, {{"var": "out", "op": "ADD", "args": ["upstream", "bonus"]}}], "output": "out"}}

Pattern: "[N] [ITEM1] are split equally among [N] [ITEM2]"
{{"params": {{"n1": "total items", "n2": "number of groups"}}, "inputs": {{}}, "steps": [{{"var": "out", "op": "DIV", "args": ["n1", "n2"]}}], "output": "out"}}

Pattern: "[PERSON1] spent [N] on [ITEM1] and [N] on [ITEM2]"
{{"params": {{"n1": "first amount", "n2": "second amount"}}, "inputs": {{}}, "steps": [{{"var": "out", "op": "ADD", "args": ["n1", "n2"]}}], "output": "out"}}

Pattern: "how many [ITEM1] does [PERSON1] have?"
{{"params": {{}}, "inputs": {{"upstream": "current total"}}, "steps": [{{"var": "out", "op": "SET", "args": ["upstream"]}}], "output": "out"}}

Pattern: "{pattern}"
Example spans:
{examples}
"""


def parse_subgraph_json(response: str) -> Optional[Dict]:
    """Parse Qwen's JSON response into a SubGraphDSL dict."""
    cleaned = response.strip()
    # Find the JSON object in the response
    # Sometimes Qwen adds text before/after the JSON
    brace_start = cleaned.find('{')
    if brace_start < 0:
        return None

    # Find matching closing brace
    depth = 0
    for i in range(brace_start, len(cleaned)):
        if cleaned[i] == '{':
            depth += 1
        elif cleaned[i] == '}':
            depth -= 1
            if depth == 0:
                json_str = cleaned[brace_start:i + 1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    return None
    return None


def make_fallback_dsl(pattern: str) -> Dict:
    """Create a simple SET(n1) fallback DSL."""
    return {
        "params": {"n1": "value from span"},
        "inputs": {},
        "steps": [{"var": "out", "op": "SET", "args": ["n1"]}],
        "output": "out",
    }


def generate_dsl(llm, templates: List[Dict], batch_size: int = 500) -> List[Dict]:
    """Generate SubGraphDSL JSON for each template using Qwen."""
    from mycelium.subgraph_dsl import SubGraphDSL, SubGraphStep

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
            examples_str = "\n".join(f"  \"{ex}\"" for ex in examples)
            prompt = DSL_PROMPT.format(
                pattern=tpl['pattern'],
                examples=examples_str or "  (no examples)",
            )
            prompts.append(prompt)

        responses = batch_qwen(llm, prompts, max_tokens=300)

        for (idx, tpl), response in zip(batch, responses):
            parsed = parse_subgraph_json(response)

            if parsed:
                # Add template_id and pattern for SubGraphDSL
                parsed['template_id'] = tpl['template_id']
                parsed['pattern'] = tpl['pattern']

                # Validate using the dataclass
                try:
                    dsl = SubGraphDSL.from_dict(parsed)
                    errors = dsl.validate()
                    if not errors:
                        templates[idx]['subgraph'] = dsl.to_dict()
                        valid_count += 1
                        continue
                except Exception:
                    pass

            # Fallback: simple SET(n1)
            fb = make_fallback_dsl(tpl['pattern'])
            fb['template_id'] = tpl['template_id']
            fb['pattern'] = tpl['pattern']
            templates[idx]['subgraph'] = fb
            fallback_count += 1

        done = min(batch_start + batch_size, len(prompts_needed))
        print(f"  [{done}/{len(prompts_needed)}] valid={valid_count} fallback={fallback_count}")

    # Stats: count unique sub-graph structures (by step ops)
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
        ops_str = " → ".join(f"{op}({nargs})" for op, nargs in struct)
        print(f"    [{count:4d}] {ops_str}")

    # Save templates with embedded subgraph
    output_path = OUTPUT_DIR / "templates_final.json"
    with open(output_path, "w") as f:
        json.dump(templates, f, indent=2)
    print(f"\n  Saved to {output_path}")

    # Also save standalone SubGraphDSL file
    dsls_path = OUTPUT_DIR / "subgraph_dsls.json"
    dsls = [t['subgraph'] for t in templates if 'subgraph' in t]
    with open(dsls_path, "w") as f:
        json.dump(dsls, f, indent=2)
    print(f"  Standalone DSLs saved to {dsls_path}")

    return templates


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(records: List[Dict], pass_num: int, phase: str):
    """Save progress."""
    checkpoint = {
        "pass_num": pass_num,
        "phase": phase,
        "record_count": len(records),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)

    # Save records
    records_path = OUTPUT_DIR / "iterative_gen_records.json"
    with open(records_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"  Checkpoint saved: pass={pass_num}, phase={phase}")


def load_checkpoint() -> Tuple[Optional[int], Optional[str]]:
    """Load checkpoint if exists."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            cp = json.load(f)
        return cp.get("pass_num"), cp.get("phase")
    return None, None


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Iterative Generalization Pipeline")
    parser.add_argument("--tp-size", type=int, default=4, help="Tensor parallel size for vLLM")
    parser.add_argument("--max-passes", type=int, default=5, help="Max generalization passes")
    parser.add_argument("--min-group-size", type=int, default=2,
                        help="Re-generalize patterns with fewer than this many members (default 2)")
    parser.add_argument("--min-drop-pct", type=float, default=5.0,
                        help="Stop if small-group count drops < this %% between passes (default 5%%)")
    parser.add_argument("--target-templates", type=int, default=1000, help="Target template count")
    parser.add_argument("--resume-from", type=int, default=None, help="Resume from pass N")
    parser.add_argument("--skip-to-dsl", action="store_true", help="Skip to DSL generation")
    parser.add_argument("--input", type=str, default=None,
                        help="Input JSON (default: generalized_v2.json or iterative_gen_records.json)")
    args = parser.parse_args()

    print("=" * 60)
    print("Iterative Generalization Pipeline")
    print("=" * 60)

    # Load records
    records_path = OUTPUT_DIR / "iterative_gen_records.json"
    if args.input:
        input_path = Path(args.input)
    elif records_path.exists() and args.resume_from:
        input_path = records_path
    else:
        input_path = OUTPUT_DIR / "generalized_v2.json"

    print(f"\nLoading records from: {input_path}")
    with open(input_path) as f:
        records = json.load(f)
    print(f"Records loaded: {len(records)}")

    if args.skip_to_dsl:
        # Load clustered templates and go straight to DSL
        templates_path = OUTPUT_DIR / "templates_clustered.json"
        print(f"\nLoading templates from: {templates_path}")
        with open(templates_path) as f:
            templates = json.load(f)
        print(f"Templates: {len(templates)}")

        llm = load_qwen(args.tp_size)
        templates = generate_dsl(llm, templates)
        unload_qwen(llm)
        return

    # =========================================================================
    # Generalization passes
    # =========================================================================

    # Seed gen_0 from existing patterns (no Qwen calls — just use what we have)
    records = seed_from_existing(records)
    save_checkpoint(records, 0, "seeded")

    start_pass = args.resume_from or 1
    llm = load_qwen(args.tp_size)
    prev_small_count = None
    pass_num = start_pass

    for pass_num in range(start_pass, args.max_passes + 1):
        # Every pass targets singletons/small groups from previous pass
        records = run_pass_n(llm, records, pass_num, min_group_size=args.min_group_size)

        # Save checkpoint
        save_checkpoint(records, pass_num, "generalization")

        # Check stopping criterion: how many records are in small groups?
        col = f"gen_{pass_num}"
        counts, _ = cluster_patterns(records, col)
        current_small = sum(c for c in counts.values() if c < args.min_group_size)
        total_unique = len(counts)
        singletons = sum(1 for c in counts.values() if c == 1)

        print(f"\n  Summary after pass {pass_num}:")
        print(f"    Unique patterns: {total_unique}")
        print(f"    Singletons (count=1): {singletons}")
        print(f"    Small groups (count<{args.min_group_size}): {current_small} records")

        if prev_small_count is not None:
            drop = (prev_small_count - current_small) / max(prev_small_count, 1)
            print(f"    Small group drop: {prev_small_count} → {current_small} ({drop*100:.1f}%)")
            if drop * 100 < args.min_drop_pct:
                print(f"    Drop < {args.min_drop_pct:.0f}% — stopping generalization")
                break

        prev_small_count = current_small

    final_col = f"gen_{pass_num}"
    print(f"\n  Final generalization column: {final_col}")

    # Unload Qwen for MiniLM embedding
    unload_qwen(llm)

    # =========================================================================
    # Embed + Cluster
    # =========================================================================

    templates = embed_and_cluster(records, final_col, target_count=args.target_templates)
    save_checkpoint(records, pass_num, "clustered")

    # =========================================================================
    # DSL Generation
    # =========================================================================

    # Reload Qwen for DSL
    llm = load_qwen(args.tp_size)
    templates = generate_dsl(llm, templates)
    unload_qwen(llm)

    save_checkpoint(records, pass_num, "dsl_done")

    # Final summary
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Input records: {len(records)}")
    print(f"  Generalization passes: {pass_num}")
    print(f"  Final templates: {len(templates)}")
    unique_dsl = len(set(t['dsl_expr'] for t in templates))
    print(f"  Unique DSL expressions: {unique_dsl}")
    print(f"  Output: {OUTPUT_DIR / 'templates_final.json'}")


if __name__ == "__main__":
    main()
