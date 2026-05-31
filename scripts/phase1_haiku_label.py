"""Phase 1: Haiku teacher-labeling script.

Runs Claude Haiku (claude-haiku-4-5-20251001) on GSM8K problems to produce
factor-graph JSON labels.  Two modes:

  Stage 1 (--stage 1 --n 50 --seed 42):
    Runs on 50 train problems, checks computability, saves prompt + stats.
    Reports go to stdout; human reviews before Stage 2.

  Stage 2 (--stage 2):
    Runs on 5000 train + 1000 test problems (seeds 42/43).
    Saves filtered records to .cache/gsm8k_factor_graphs_{train,val}.jsonl.
    Saves rejected records to .cache/gsm8k_factor_graphs_rejected.jsonl.

Usage:
  python scripts/phase1_haiku_label.py --stage 1 [--n 50] [--seed 42]
  python scripts/phase1_haiku_label.py --stage 2
  python scripts/phase1_haiku_label.py --stage 2 --resume  # skip already-done

Requirements:
  ANTHROPIC_API_KEY env var, or api_key via --api-key flag / key1.txt default.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import anthropic
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
CACHE_DIR = REPO_ROOT / ".cache"
GSM8K_DIR = CACHE_DIR / "gsm8k"
RAW_DIR   = CACHE_DIR / "gsm8k_raw"

GSM8K_TRAIN_PARQUET = (
    GSM8K_DIR
    / "datasets--openai--gsm8k"
    / "snapshots"
    / "740312add88f781978c0658806c59bc2815b9866"
    / "main"
    / "train-00000-of-00001.parquet"
)
GSM8K_TEST_PARQUET = (
    GSM8K_DIR
    / "datasets--openai--gsm8k"
    / "snapshots"
    / "740312add88f781978c0658806c59bc2815b9866"
    / "main"
    / "test-00000-of-00001.parquet"
)

PROMPT_SAVE_PATH  = CACHE_DIR / "phase1_haiku_prompt.txt"
TRAIN_OUT_PATH    = CACHE_DIR / "gsm8k_factor_graphs_train.jsonl"
VAL_OUT_PATH      = CACHE_DIR / "gsm8k_factor_graphs_val.jsonl"
REJECTED_OUT_PATH = CACHE_DIR / "gsm8k_factor_graphs_rejected.jsonl"

# ---------------------------------------------------------------------------
# Haiku model
# ---------------------------------------------------------------------------

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 1500
SLEEP_BETWEEN_CALLS = 0.05   # seconds (gentle throttle — Haiku is fast)
MAX_RETRIES = 3
RETRY_SLEEP = 2.0

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a math problem parser. Convert NL math problems to factor graph JSON. Output ONLY a JSON object."""

TASK_PROMPT = """Parse the math problem below into a factor graph JSON.

## Schema

{
  "n_vars": <int>,
  "n_factors": <int>,
  "domain": [0, 10000],
  "factor_types": ["add"|"sub"|"mul"|"div", ...],
  "factor_args": [[arg1, arg2, result], ...],
  "observed_mask": [1|0, ...],
  "observed_values": [int|null, ...],
  "query_idx": <int>,
  "var_descriptions": ["...", ...]
}

## Method (follow exactly)

1. Write out all OBSERVED variables (given numbers in the problem) with their values.
   Give each a sequential index starting at 0.
2. Write out all COMPUTED variables (quantities you must calculate).
   Continue the index sequence after the observed ones.
3. Set n_vars = total count of observed + computed variables.
4. Write out each arithmetic step as: op(arg1_idx, arg2_idx) -> result_idx
   Use only: add, sub, mul, div.
5. Set n_factors = number of steps.
6. Fill in the JSON arrays. Every array must have exactly n_vars (or n_factors) elements.
7. Verify: n_vars == len(observed_mask) == len(observed_values) == len(var_descriptions).
   Verify: n_factors == len(factor_types) == len(factor_args).

## Strict rules
- observed_mask[i]=1 ONLY for variables you explicitly listed as observed.
- observed_values[i] = the integer for observed vars, null for computed vars.
- NEVER use the same index as both observed AND as a result of a factor.
- Every result index must appear in observed_mask as 0 (not 1).
- query_idx MUST be a computed variable (observed_mask[query_idx] = 0).
- For fractions like "1/3": add a variable for the denominator (3) as observed.
- For percents like "40%": add 40 and 100 as observed variables. Compute: (val * 40) / 100.

## Example 1

Problem: "Janet has 16 eggs. She eats 3 for breakfast. She bakes muffins with 4. She sells remaining eggs at $2 each. How much money?"

Method:
Observed: v0=16 (total eggs), v1=3 (eaten), v2=4 (baking), v3=2 (price/egg)
Computed: v4=eggs after breakfast, v5=eggs after baking, v6=money
n_vars=7
Steps: sub(0,1)->4, sub(4,2)->5, mul(5,3)->6
n_factors=3

{
  "n_vars": 7,
  "n_factors": 3,
  "domain": [0, 10000],
  "factor_types": ["sub", "sub", "mul"],
  "factor_args": [[0, 1, 4], [4, 2, 5], [5, 3, 6]],
  "observed_mask": [1, 1, 1, 1, 0, 0, 0],
  "observed_values": [16, 3, 4, 2, null, null, null],
  "query_idx": 6,
  "var_descriptions": ["total eggs", "eggs eaten", "eggs for baking", "price per egg", "eggs after breakfast", "eggs after baking", "total money"]
}

## Example 2

Problem: "A store sells apples for $3 and oranges for $2. Tom buys 5 apples and 4 oranges. Total cost?"

Method:
Observed: v0=3 (apple price), v1=2 (orange price), v2=5 (apples), v3=4 (oranges)
Computed: v4=apple cost, v5=orange cost, v6=total
n_vars=7
Steps: mul(0,2)->4, mul(1,3)->5, add(4,5)->6
n_factors=3

{
  "n_vars": 7,
  "n_factors": 3,
  "domain": [0, 10000],
  "factor_types": ["mul", "mul", "add"],
  "factor_args": [[0, 2, 4], [1, 3, 5], [4, 5, 6]],
  "observed_mask": [1, 1, 1, 1, 0, 0, 0],
  "observed_values": [3, 2, 5, 4, null, null, null],
  "query_idx": 6,
  "var_descriptions": ["apple price", "orange price", "apples bought", "oranges bought", "apple subtotal", "orange subtotal", "total"]
}

## Example 3

Problem: "48 students. 1/3 are in science club. Half of those join math club. How many are in both?"

Method:
Observed: v0=48 (students), v1=3 (divisor for 1/3), v2=2 (divisor for half)
Computed: v3=science club, v4=both clubs
n_vars=5
Steps: div(0,1)->3, div(3,2)->4
n_factors=2

{
  "n_vars": 5,
  "n_factors": 2,
  "domain": [0, 10000],
  "factor_types": ["div", "div"],
  "factor_args": [[0, 1, 3], [3, 2, 4]],
  "observed_mask": [1, 1, 1, 0, 0],
  "observed_values": [48, 3, 2, null, null],
  "query_idx": 4,
  "var_descriptions": ["total students", "third divisor", "half divisor", "science club", "both clubs"]
}

## Example 4

Problem: "Betty picked 16 strawberries. Matthew picked 20 more than Betty. Nancy picked half of Matthew's amount. How many did all three pick together?"

Method:
Observed: v0=16 (Betty), v1=20 (extra for Matthew), v2=2 (Nancy divisor)
Computed: v3=Matthew count, v4=Nancy count, v5=Betty+Matthew, v6=total
n_vars=7
Steps: add(0,1)->3, div(3,2)->4, add(0,3)->5, add(5,4)->6
n_factors=4

{
  "n_vars": 7,
  "n_factors": 4,
  "domain": [0, 10000],
  "factor_types": ["add", "div", "add", "add"],
  "factor_args": [[0, 1, 3], [3, 2, 4], [0, 3, 5], [5, 4, 6]],
  "observed_mask": [1, 1, 1, 0, 0, 0, 0],
  "observed_values": [16, 20, 2, null, null, null, null],
  "query_idx": 6,
  "var_descriptions": ["Betty count", "extra for Matthew", "Nancy divisor", "Matthew count", "Nancy count", "Betty plus Matthew", "total all three"]
}

## Example 5

Problem: "Josh has 100 pounds of sugar. He uses 40 pounds for candy. Then uses 60% of remaining for cake. How much left?"

Method:
Observed: v0=100 (initial sugar), v1=40 (candy), v2=60 (percent numerator), v3=100 (percent denominator)
Computed: v4=after candy, v5=cake percent times remaining, v6=cake amount, v7=final
n_vars=8
Steps: sub(0,1)->4, mul(4,2)->5, div(5,3)->6, sub(4,6)->7
n_factors=4

{
  "n_vars": 8,
  "n_factors": 4,
  "domain": [0, 10000],
  "factor_types": ["sub", "mul", "div", "sub"],
  "factor_args": [[0, 1, 4], [4, 2, 5], [5, 3, 6], [4, 6, 7]],
  "observed_mask": [1, 1, 1, 1, 0, 0, 0, 0],
  "observed_values": [100, 40, 60, 100, null, null, null, null],
  "query_idx": 7,
  "var_descriptions": ["initial sugar", "candy sugar", "percent 60", "divisor 100", "after candy", "sugar times pct", "cake sugar", "final sugar"]
}

## Now parse this problem

Problem: {PROBLEM}

Remember: Show the Method steps first (Observed, Computed, n_vars, Steps, n_factors), then output the JSON."""


def build_prompt(problem: str) -> str:
    """Build the full prompt for a single problem."""
    return TASK_PROMPT.replace("{PROBLEM}", problem.strip())


# ---------------------------------------------------------------------------
# GSM8K loading
# ---------------------------------------------------------------------------

def load_gsm8k_parquet(path: Path) -> list[dict]:
    """Load GSM8K from parquet. Returns list of {question, answer} dicts."""
    t = pq.read_table(str(path))
    d = t.to_pydict()
    records = []
    for q, a in zip(d["question"], d["answer"]):
        records.append({"question": q, "answer": a})
    return records


def extract_gold_answer(gsm8k_answer: str) -> float | None:
    """Extract numeric gold answer from GSM8K answer string (ends with '#### N')."""
    parts = gsm8k_answer.strip().split("####")
    if len(parts) < 2:
        return None
    raw = parts[-1].strip().replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {
    "n_vars", "n_factors", "domain", "factor_types", "factor_args",
    "observed_mask", "observed_values", "query_idx", "var_descriptions",
}
VALID_OPS = {"add", "sub", "mul", "div"}


def autocorrect_record(rec: dict[str, Any]) -> dict[str, Any]:
    """Attempt light auto-correction of common Haiku mistakes.

    Corrections applied (in order):
    1. If n_vars disagrees with array lengths but all three arrays agree → fix n_vars.
    2. If n_factors disagrees with list lengths but both agree → fix n_factors.
    3. If observed_values has None strings → convert to Python None.
    4. Remove 'gold_values' field if present (not part of the schema we emit).

    Returns a (possibly modified) copy. Does NOT mutate in place.
    """
    rec = dict(rec)  # shallow copy

    # Fix n_vars
    obs_mask = rec.get("observed_mask", [])
    obs_vals = rec.get("observed_values", [])
    var_desc = rec.get("var_descriptions", [])
    if isinstance(obs_mask, list) and isinstance(obs_vals, list) and isinstance(var_desc, list):
        lengths = [len(obs_mask), len(obs_vals), len(var_desc)]
        if len(set(lengths)) == 1:
            # All three arrays agree; trust them over n_vars
            rec["n_vars"] = lengths[0]

    # Fix n_factors
    ft_list = rec.get("factor_types", [])
    fa_list = rec.get("factor_args", [])
    if isinstance(ft_list, list) and isinstance(fa_list, list):
        if len(ft_list) == len(fa_list):
            rec["n_factors"] = len(ft_list)

    # Fix "None" strings → Python None in observed_values
    if isinstance(rec.get("observed_values"), list):
        fixed_vals = []
        for v in rec["observed_values"]:
            if isinstance(v, str) and v.lower() in ("none", "null", ""):
                fixed_vals.append(None)
            else:
                fixed_vals.append(v)
        rec["observed_values"] = fixed_vals

    # If observed_mask=1 but observed_values=None and there's a float in var_descriptions,
    # we can't recover — leave it to fail validation.

    # Convert integer-valued floats to ints in observed_values
    if isinstance(rec.get("observed_values"), list):
        fixed_vals = []
        for v in rec["observed_values"]:
            if isinstance(v, float) and v.is_integer():
                fixed_vals.append(int(v))
            else:
                fixed_vals.append(v)
        rec["observed_values"] = fixed_vals

    return rec


def validate_record(rec: dict[str, Any]) -> tuple[bool, str]:
    """Structural validation of a parsed factor-graph record.

    Returns (is_valid, error_message). error_message is empty string if valid.
    """
    # Required keys
    missing = REQUIRED_KEYS - set(rec.keys())
    if missing:
        return False, f"Missing keys: {missing}"

    n_vars    = rec.get("n_vars")
    n_factors = rec.get("n_factors")

    if not isinstance(n_vars, int) or n_vars <= 0:
        return False, f"n_vars must be a positive int, got {n_vars!r}"
    if not isinstance(n_factors, int) or n_factors < 0:
        return False, f"n_factors must be a non-negative int, got {n_factors!r}"

    # Length checks
    if len(rec["factor_types"]) != n_factors:
        return False, f"factor_types length {len(rec['factor_types'])} != n_factors {n_factors}"
    if len(rec["factor_args"]) != n_factors:
        return False, f"factor_args length {len(rec['factor_args'])} != n_factors {n_factors}"
    if len(rec["observed_mask"]) != n_vars:
        return False, f"observed_mask length {len(rec['observed_mask'])} != n_vars {n_vars}"
    if len(rec["observed_values"]) != n_vars:
        return False, f"observed_values length {len(rec['observed_values'])} != n_vars {n_vars}"
    if len(rec["var_descriptions"]) != n_vars:
        return False, f"var_descriptions length {len(rec['var_descriptions'])} != n_vars {n_vars}"

    # Op types
    for i, op in enumerate(rec["factor_types"]):
        if op not in VALID_OPS:
            return False, f"factor_types[{i}] = {op!r} not in {VALID_OPS}"

    # factor_args validity
    for i, fa in enumerate(rec["factor_args"]):
        if len(fa) != 3:
            return False, f"factor_args[{i}] has {len(fa)} elements, expected 3"
        for vi in fa:
            if not (0 <= int(vi) < n_vars):
                return False, f"factor_args[{i}] index {vi} out of range [0, {n_vars})"

    # observed_mask values
    for i, m in enumerate(rec["observed_mask"]):
        if m not in (0, 1):
            return False, f"observed_mask[{i}] = {m!r}, must be 0 or 1"

    # observed_values types
    for i, v in enumerate(rec["observed_values"]):
        if rec["observed_mask"][i] == 1:
            if v is None:
                return False, f"observed_values[{i}] is null but mask=1"
            try:
                float(v)
            except (TypeError, ValueError):
                return False, f"observed_values[{i}] = {v!r} is not numeric"
        # mask=0: value should be null (we allow non-null for gold values)

    # query_idx
    qi = rec.get("query_idx")
    if not isinstance(qi, int) or not (0 <= qi < n_vars):
        return False, f"query_idx {qi!r} out of range [0, {n_vars})"
    if rec["observed_mask"][qi] == 1:
        return False, f"query_idx {qi} has observed_mask=1 (must be unobserved)"

    # Leaf check: result variables of factors should not be observed
    result_idxs = {int(fa[2]) for fa in rec["factor_args"]}
    for ri in result_idxs:
        if rec["observed_mask"][ri] == 1:
            return False, f"Variable {ri} is the result of a factor but observed_mask=1"

    # domain shape
    dom = rec.get("domain")
    if not (isinstance(dom, list) and len(dom) == 2):
        return False, f"domain must be [min, max], got {dom!r}"

    return True, ""


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def extract_json_from_response(text: str) -> dict | None:
    """Extract the first valid JSON object from model response text."""
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON block in markdown fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find first { ... } block
    brace_match = re.search(r"\{[\s\S]+\}", text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Haiku API call
# ---------------------------------------------------------------------------

def call_haiku(
    client: anthropic.Anthropic,
    problem: str,
    prompt_template: str,
) -> tuple[str | None, dict]:
    """Call Haiku with retry logic.

    Returns (raw_text, usage_dict).
    usage_dict has 'input_tokens' and 'output_tokens'.
    """
    prompt = build_prompt(problem)
    usage_total = {"input_tokens": 0, "output_tokens": 0}

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            usage = resp.usage
            usage_total["input_tokens"]  += usage.input_tokens
            usage_total["output_tokens"] += usage.output_tokens
            text = resp.content[0].text if resp.content else ""
            return text, usage_total

        except anthropic.RateLimitError:
            wait = RETRY_SLEEP * (2 ** attempt)
            print(f"  [rate limit] sleeping {wait:.1f}s ...", flush=True)
            time.sleep(wait)
        except anthropic.APIError as e:
            print(f"  [api error attempt {attempt+1}] {e}", flush=True)
            time.sleep(RETRY_SLEEP)

    return None, usage_total


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

# Haiku 4.5 pricing (per 1M tokens, as of late 2025)
INPUT_COST_PER_1M  = 0.80   # $0.80 / 1M input tokens
OUTPUT_COST_PER_1M = 4.00   # $4.00 / 1M output tokens


def compute_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1_000_000 * INPUT_COST_PER_1M
            + output_tokens / 1_000_000 * OUTPUT_COST_PER_1M)


# ---------------------------------------------------------------------------
# Process one problem
# ---------------------------------------------------------------------------

def process_problem(
    client: anthropic.Anthropic,
    gsm_rec: dict,
    idx: int,
    prompt_template: str,
) -> dict:
    """Run Haiku on one GSM8K problem.  Returns a result dict."""
    question = gsm_rec["question"]
    answer   = gsm_rec["answer"]
    gold     = extract_gold_answer(answer)

    raw_text, usage = call_haiku(client, question, prompt_template)
    time.sleep(SLEEP_BETWEEN_CALLS)

    result = {
        "idx": idx,
        "question": question,
        "gold_answer": gold,
        "raw_response": raw_text,
        "usage": usage,
    }

    if raw_text is None:
        result["status"] = "api_failure"
        result["error"] = "All retries exhausted"
        return result

    parsed = extract_json_from_response(raw_text)
    if parsed is None:
        result["status"] = "json_parse_failure"
        result["error"] = "Could not extract JSON from response"
        return result

    # Apply light auto-corrections before validation
    parsed = autocorrect_record(parsed)
    is_valid, err_msg = validate_record(parsed)
    if not is_valid:
        result["status"] = "schema_invalid"
        result["error"] = err_msg
        result["parsed"] = parsed
        return result

    # Topological eval vs gold
    from phase1_topological_eval import topo_eval, _floats_match
    eval_res = topo_eval(parsed)
    if not eval_res["success"]:
        result["status"] = "topo_eval_failure"
        result["error"] = eval_res["error"]
        result["parsed"] = parsed
        result["topo_result"] = eval_res
        return result

    computed = eval_res["computed_answer"]
    match = (gold is not None and computed is not None
             and _floats_match(float(computed), float(gold)))

    result["status"] = "ok" if match else "answer_mismatch"
    result["computed_answer"] = computed
    result["match"] = match
    result["parsed"] = parsed
    result["topo_result"] = eval_res
    if not match:
        result["error"] = f"computed={computed}, gold={gold}"

    return result


# ---------------------------------------------------------------------------
# Stage 1: 50-problem validation
# ---------------------------------------------------------------------------

def run_stage1(client, n: int = 50, seed: int = 42) -> None:
    """Stage 1: Run on n train problems, report computability rate."""
    print(f"\n{'='*60}", flush=True)
    print(f"STAGE 1: {n}-problem validation (seed={seed})", flush=True)
    print(f"{'='*60}\n", flush=True)

    # Load GSM8K train
    train_recs = load_gsm8k_parquet(GSM8K_TRAIN_PARQUET)
    rng = random.Random(seed)
    sample_indices = rng.sample(range(len(train_recs)), min(n, len(train_recs)))
    sample = [train_recs[i] for i in sample_indices]

    print(f"Loaded {len(train_recs)} train problems, sampled {len(sample)}.\n",
          flush=True)

    results = []
    total_input  = 0
    total_output = 0

    for i, rec in enumerate(sample):
        print(f"[{i+1:3d}/{len(sample)}] ", end="", flush=True)
        res = process_problem(client, rec, sample_indices[i], TASK_PROMPT)
        results.append(res)
        total_input  += res["usage"]["input_tokens"]
        total_output += res["usage"]["output_tokens"]

        status = res["status"]
        if status == "ok":
            print(f"OK  (computed={res['computed_answer']:.4g}, gold={res['gold_answer']})",
                  flush=True)
        else:
            print(f"FAIL [{status}]: {res.get('error','')[:80]}", flush=True)

    # Stats
    n_ok       = sum(1 for r in results if r["status"] == "ok")
    n_mismatch = sum(1 for r in results if r["status"] == "answer_mismatch")
    n_topo     = sum(1 for r in results if r["status"] == "topo_eval_failure")
    n_schema   = sum(1 for r in results if r["status"] == "schema_invalid")
    n_json_err = sum(1 for r in results if r["status"] == "json_parse_failure")
    n_api_err  = sum(1 for r in results if r["status"] == "api_failure")

    computable_rate = n_ok / len(results) if results else 0.0
    cost = compute_cost(total_input, total_output)

    print(f"\n{'='*60}", flush=True)
    print(f"STAGE 1 RESULTS ({len(results)} problems)", flush=True)
    print(f"  Correct (computable):   {n_ok:3d}  ({computable_rate:.1%})", flush=True)
    print(f"  Answer mismatch:        {n_mismatch:3d}", flush=True)
    print(f"  Topo eval failure:      {n_topo:3d}", flush=True)
    print(f"  Schema invalid:         {n_schema:3d}", flush=True)
    print(f"  JSON parse failure:     {n_json_err:3d}", flush=True)
    print(f"  API failure:            {n_api_err:3d}", flush=True)
    print(f"  Tokens: in={total_input}, out={total_output}", flush=True)
    print(f"  Estimated cost: ${cost:.4f}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # Save prompt
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PROMPT_SAVE_PATH.write_text(SYSTEM_PROMPT + "\n\n---\n\n" + TASK_PROMPT)
    print(f"Prompt saved to {PROMPT_SAVE_PATH}", flush=True)

    # Save Stage 1 raw results for inspection
    stage1_results_path = CACHE_DIR / "phase1_stage1_results.jsonl"
    with open(stage1_results_path, "w") as f:
        for res in results:
            # Serialize without the full raw_response to keep it readable
            out = {k: v for k, v in res.items() if k != "raw_response"}
            f.write(json.dumps(out) + "\n")
    print(f"Stage 1 raw results saved to {stage1_results_path}", flush=True)

    # Print 5 sample outputs (mix of ok and fail)
    print("\n--- SAMPLE OUTPUTS (3 OK + 2 FAIL) ---\n", flush=True)
    ok_samples   = [r for r in results if r["status"] == "ok"][:3]
    fail_samples = [r for r in results if r["status"] != "ok"][:2]
    for r in ok_samples + fail_samples:
        print(f"Problem [{r['idx']}]: {r['question'][:100]}...")
        print(f"  Status: {r['status']}")
        if r.get("parsed"):
            print(f"  n_vars={r['parsed'].get('n_vars')}, "
                  f"n_factors={r['parsed'].get('n_factors')}, "
                  f"factor_types={r['parsed'].get('factor_types')}")
        print(f"  computed={r.get('computed_answer')}, gold={r.get('gold_answer')}")
        if r.get("error"):
            print(f"  error: {r['error']}")
        print()

    # Gate check
    target = 0.85
    if computable_rate >= target:
        print(f"GATE PASSED: {computable_rate:.1%} >= {target:.0%}  — ready for Stage 2.",
              flush=True)
    else:
        print(f"GATE FAILED: {computable_rate:.1%} < {target:.0%}  — iterate prompt before Stage 2.",
              flush=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Stage 2: Full labeling
# ---------------------------------------------------------------------------

def run_stage2(
    client,
    n_train: int = 5000,
    n_val: int = 1000,
    seed_train: int = 42,
    seed_val: int = 43,
    budget_usd: float = 60.0,
    resume: bool = False,
) -> None:
    """Stage 2: Full labeling run."""
    print(f"\n{'='*60}", flush=True)
    print(f"STAGE 2: Full labeling (train={n_train}, val={n_val})", flush=True)
    print(f"{'='*60}\n", flush=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    all_train = load_gsm8k_parquet(GSM8K_TRAIN_PARQUET)
    all_test  = load_gsm8k_parquet(GSM8K_TEST_PARQUET)

    rng_train = random.Random(seed_train)
    rng_val   = random.Random(seed_val)

    train_indices = rng_train.sample(range(len(all_train)), min(n_train, len(all_train)))
    val_indices   = rng_val.sample(range(len(all_test)),  min(n_val,   len(all_test)))

    train_sample = [(i, all_train[i]) for i in train_indices]
    val_sample   = [(i, all_test[i])  for i in val_indices]

    print(f"Train: {len(train_sample)} problems, Val: {len(val_sample)} problems", flush=True)

    # Resume: find already-processed indices
    done_train_idxs: set[int] = set()
    done_val_idxs:   set[int] = set()
    if resume:
        for path, done_set in [
            (TRAIN_OUT_PATH, done_train_idxs),
            (REJECTED_OUT_PATH, done_train_idxs),
        ]:
            if path.exists():
                with open(path) as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                            if "gsm8k_idx" in rec:
                                done_set.add(rec["gsm8k_idx"])
                        except json.JSONDecodeError:
                            pass
        if VAL_OUT_PATH.exists():
            with open(VAL_OUT_PATH) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        if "gsm8k_idx" in rec:
                            done_val_idxs.add(rec["gsm8k_idx"])
                    except json.JSONDecodeError:
                        pass
        print(f"Resume: {len(done_train_idxs)} train + {len(done_val_idxs)} val already done",
              flush=True)

    total_input  = 0
    total_output = 0
    n_accepted   = 0
    n_rejected   = 0

    def _process_split(
        split_name: str,
        sample: list[tuple[int, dict]],
        out_path: Path,
        done_idxs: set[int],
    ) -> None:
        nonlocal total_input, total_output, n_accepted, n_rejected

        mode = "a" if resume and out_path.exists() else "w"
        with (
            open(out_path, mode) as out_f,
            open(REJECTED_OUT_PATH, "a") as rej_f,
        ):
            for pos, (gsm_idx, gsm_rec) in enumerate(sample):
                if gsm_idx in done_idxs:
                    continue

                # Budget check
                current_cost = compute_cost(total_input, total_output)
                if current_cost > budget_usd * 0.95:
                    print(f"\nBUDGET WARNING: cost ${current_cost:.2f} approaching "
                          f"limit ${budget_usd:.0f}. Stopping.", flush=True)
                    return

                print(f"[{split_name} {pos+1}/{len(sample)}] idx={gsm_idx} ", end="",
                      flush=True)

                res = process_problem(client, gsm_rec, gsm_idx, TASK_PROMPT)
                total_input  += res["usage"]["input_tokens"]
                total_output += res["usage"]["output_tokens"]
                cost_so_far  = compute_cost(total_input, total_output)

                if res["status"] == "ok":
                    record = res["parsed"]
                    record["gsm8k_idx"]   = gsm_idx
                    record["gold_answer"] = res["gold_answer"]
                    record["question"]    = gsm_rec["question"]
                    out_f.write(json.dumps(record) + "\n")
                    out_f.flush()
                    n_accepted += 1
                    print(f"OK (cost=${cost_so_far:.2f})", flush=True)
                else:
                    rejected_rec = {
                        "gsm8k_idx":   gsm_idx,
                        "question":    gsm_rec["question"],
                        "gold_answer": res.get("gold_answer"),
                        "status":      res["status"],
                        "error":       res.get("error"),
                        "parsed":      res.get("parsed"),
                        "raw_response": res.get("raw_response"),
                    }
                    rej_f.write(json.dumps(rejected_rec) + "\n")
                    rej_f.flush()
                    n_rejected += 1
                    print(f"FAIL [{res['status']}] (cost=${cost_so_far:.2f})", flush=True)

    # Run train split
    _process_split("train", train_sample, TRAIN_OUT_PATH, done_train_idxs)

    # Run val split
    _process_split("val",   val_sample,   VAL_OUT_PATH,   done_val_idxs)

    # Final stats
    final_cost = compute_cost(total_input, total_output)
    print(f"\n{'='*60}", flush=True)
    print(f"STAGE 2 COMPLETE", flush=True)
    print(f"  Accepted: {n_accepted}", flush=True)
    print(f"  Rejected: {n_rejected}", flush=True)
    if n_accepted + n_rejected > 0:
        print(f"  Accept rate: {n_accepted/(n_accepted+n_rejected):.1%}", flush=True)
    print(f"  Tokens: in={total_input}, out={total_output}", flush=True)
    print(f"  Total cost: ${final_cost:.4f}", flush=True)
    print(f"{'='*60}\n", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Phase 1 Haiku labeling script")
    ap.add_argument("--stage", type=int, choices=[1, 2], default=1,
                    help="Stage to run (1=validation, 2=full labeling)")
    ap.add_argument("--n", type=int, default=50,
                    help="Number of problems for Stage 1 (default 50)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for Stage 1 sampling (default 42)")
    ap.add_argument("--n-train", type=int, default=5000,
                    help="Train problems for Stage 2 (default 5000)")
    ap.add_argument("--n-val", type=int, default=1000,
                    help="Val problems for Stage 2 (default 1000)")
    ap.add_argument("--resume", action="store_true",
                    help="Stage 2: skip already-processed indices")
    ap.add_argument("--api-key", type=str, default=None,
                    help="Anthropic API key (default: read from ~/Desktop/keys/key1.txt)")
    ap.add_argument("--budget", type=float, default=60.0,
                    help="Stage 2 cost budget in USD (default 60)")
    args = ap.parse_args()

    # Resolve API key
    api_key = args.api_key
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key is None:
        key_path = Path.home() / "Desktop" / "keys" / "key1.txt"
        if key_path.exists():
            api_key = key_path.read_text().strip()
        else:
            print("ERROR: No API key found. Provide via --api-key, "
                  "ANTHROPIC_API_KEY env var, or ~/Desktop/keys/key1.txt", file=sys.stderr)
            sys.exit(1)

    # Add scripts dir to path for phase1_topological_eval import
    scripts_dir = str(Path(__file__).parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    client = anthropic.Anthropic(api_key=api_key)

    if args.stage == 1:
        run_stage1(client, n=args.n, seed=args.seed)
    else:
        run_stage2(
            client,
            n_train=args.n_train,
            n_val=args.n_val,
            seed_train=args.seed,
            seed_val=43,
            budget_usd=args.budget,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
