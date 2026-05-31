"""Phase 1 full labeling run — 5000 train + 1000 val using v2 prompt.

Uses claude-haiku-4-5-20251001 with the v2 prompt (.cache/phase1_haiku_prompt_v2.txt).
Saves:
  .cache/gsm8k_factor_graphs_train.jsonl       — accepted train records
  .cache/gsm8k_factor_graphs_val.jsonl         — accepted val records
  .cache/gsm8k_factor_graphs_train_rejected.jsonl — rejected train
  .cache/gsm8k_factor_graphs_val_rejected.jsonl   — rejected val

Usage:
  python scripts/phase1_full_run.py [--resume] [--budget 35]
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

GSM8K_TRAIN_PARQUET = (
    CACHE_DIR / "gsm8k"
    / "datasets--openai--gsm8k"
    / "snapshots"
    / "740312add88f781978c0658806c59bc2815b9866"
    / "main"
    / "train-00000-of-00001.parquet"
)
GSM8K_TEST_PARQUET = (
    CACHE_DIR / "gsm8k"
    / "datasets--openai--gsm8k"
    / "snapshots"
    / "740312add88f781978c0658806c59bc2815b9866"
    / "main"
    / "test-00000-of-00001.parquet"
)

PROMPT_V2_PATH        = CACHE_DIR / "phase1_haiku_prompt_v2.txt"
TRAIN_OUT_PATH        = CACHE_DIR / "gsm8k_factor_graphs_train.jsonl"
VAL_OUT_PATH          = CACHE_DIR / "gsm8k_factor_graphs_val.jsonl"
TRAIN_REJECTED_PATH   = CACHE_DIR / "gsm8k_factor_graphs_train_rejected.jsonl"
VAL_REJECTED_PATH     = CACHE_DIR / "gsm8k_factor_graphs_val_rejected.jsonl"

# ---------------------------------------------------------------------------
# Model + API settings
# ---------------------------------------------------------------------------

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 1500
SLEEP_BETWEEN_CALLS = 0.1   # seconds
MAX_RETRIES = 3
RETRY_SLEEP = 2.0
COST_PRINT_INTERVAL = 500   # print cumulative cost every N problems

# Haiku 4.5 pricing (per 1M tokens)
INPUT_COST_PER_1M  = 0.80
OUTPUT_COST_PER_1M = 4.00


def compute_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1_000_000 * INPUT_COST_PER_1M
            + output_tokens / 1_000_000 * OUTPUT_COST_PER_1M)


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_v2_prompt() -> tuple[str, str]:
    """Load v2 prompt file, return (system_prompt, task_prompt_template)."""
    raw = PROMPT_V2_PATH.read_text()
    # File format: system prompt, then "---", then task template
    if "\n---\n" in raw:
        parts = raw.split("\n---\n", 1)
        return parts[0].strip(), parts[1].strip()
    # If no separator, treat entire file as task template
    return "You are a math problem parser. Convert NL math problems to factor graph JSON. Output ONLY a JSON object.", raw.strip()


# ---------------------------------------------------------------------------
# GSM8K loading
# ---------------------------------------------------------------------------

def load_gsm8k_parquet(path: Path) -> list[dict]:
    t = pq.read_table(str(path))
    d = t.to_pydict()
    return [{"question": q, "answer": a} for q, a in zip(d["question"], d["answer"])]


def extract_gold_answer(gsm8k_answer: str) -> float | None:
    parts = gsm8k_answer.strip().split("####")
    if len(parts) < 2:
        return None
    raw = parts[-1].strip().replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {
    "n_vars", "n_factors", "domain", "factor_types", "factor_args",
    "observed_mask", "observed_values", "query_idx", "var_descriptions",
}
VALID_OPS = {"add", "sub", "mul", "div"}


def autocorrect_record(rec: dict[str, Any]) -> dict[str, Any]:
    rec = dict(rec)
    obs_mask = rec.get("observed_mask", [])
    obs_vals = rec.get("observed_values", [])
    var_desc = rec.get("var_descriptions", [])
    if isinstance(obs_mask, list) and isinstance(obs_vals, list) and isinstance(var_desc, list):
        lengths = [len(obs_mask), len(obs_vals), len(var_desc)]
        if len(set(lengths)) == 1:
            rec["n_vars"] = lengths[0]
    ft_list = rec.get("factor_types", [])
    fa_list = rec.get("factor_args", [])
    if isinstance(ft_list, list) and isinstance(fa_list, list):
        if len(ft_list) == len(fa_list):
            rec["n_factors"] = len(ft_list)
    if isinstance(rec.get("observed_values"), list):
        fixed_vals = []
        for v in rec["observed_values"]:
            if isinstance(v, str) and v.lower() in ("none", "null", ""):
                fixed_vals.append(None)
            else:
                fixed_vals.append(v)
        rec["observed_values"] = fixed_vals
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
    missing = REQUIRED_KEYS - set(rec.keys())
    if missing:
        return False, f"Missing keys: {missing}"
    n_vars    = rec.get("n_vars")
    n_factors = rec.get("n_factors")
    if not isinstance(n_vars, int) or n_vars <= 0:
        return False, f"n_vars must be a positive int, got {n_vars!r}"
    if not isinstance(n_factors, int) or n_factors < 0:
        return False, f"n_factors must be a non-negative int, got {n_factors!r}"
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
    for i, op in enumerate(rec["factor_types"]):
        if op not in VALID_OPS:
            return False, f"factor_types[{i}] = {op!r} not in {VALID_OPS}"
    for i, fa in enumerate(rec["factor_args"]):
        if len(fa) != 3:
            return False, f"factor_args[{i}] has {len(fa)} elements, expected 3"
        for vi in fa:
            if not (0 <= int(vi) < n_vars):
                return False, f"factor_args[{i}] index {vi} out of range [0, {n_vars})"
    for i, m in enumerate(rec["observed_mask"]):
        if m not in (0, 1):
            return False, f"observed_mask[{i}] = {m!r}, must be 0 or 1"
    for i, v in enumerate(rec["observed_values"]):
        if rec["observed_mask"][i] == 1:
            if v is None:
                return False, f"observed_values[{i}] is null but mask=1"
            try:
                float(v)
            except (TypeError, ValueError):
                return False, f"observed_values[{i}] = {v!r} is not numeric"
    qi = rec.get("query_idx")
    if not isinstance(qi, int) or not (0 <= qi < n_vars):
        return False, f"query_idx {qi!r} out of range [0, {n_vars})"
    if rec["observed_mask"][qi] == 1:
        return False, f"query_idx {qi} has observed_mask=1 (must be unobserved)"
    result_idxs = {int(fa[2]) for fa in rec["factor_args"]}
    for ri in result_idxs:
        if rec["observed_mask"][ri] == 1:
            return False, f"Variable {ri} is the result of a factor but observed_mask=1"
    dom = rec.get("domain")
    if not (isinstance(dom, list) and len(dom) == 2):
        return False, f"domain must be [min, max], got {dom!r}"
    return True, ""


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def extract_json_from_response(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass
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
    system_prompt: str,
    task_template: str,
) -> tuple[str | None, dict]:
    prompt = task_template.replace("{PROBLEM}", problem.strip())
    usage_total = {"input_tokens": 0, "output_tokens": 0}

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=system_prompt,
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
# Process one problem
# ---------------------------------------------------------------------------

def process_problem(
    client: anthropic.Anthropic,
    gsm_rec: dict,
    idx: int,
    system_prompt: str,
    task_template: str,
) -> dict:
    question = gsm_rec["question"]
    answer   = gsm_rec["answer"]
    gold     = extract_gold_answer(answer)

    raw_text, usage = call_haiku(client, question, system_prompt, task_template)
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

    parsed = autocorrect_record(parsed)
    is_valid, err_msg = validate_record(parsed)
    if not is_valid:
        result["status"] = "schema_invalid"
        result["error"] = err_msg
        result["parsed"] = parsed
        return result

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
# Main run
# ---------------------------------------------------------------------------

def run(
    n_train: int = 5000,
    n_val: int = 1000,
    seed_train: int = 42,
    seed_val: int = 43,
    budget_usd: float = 35.0,
    resume: bool = False,
    api_key: str | None = None,
) -> None:
    # Load v2 prompt
    system_prompt, task_template = load_v2_prompt()
    print(f"Loaded v2 prompt ({len(task_template)} chars task template)", flush=True)

    # API client
    client = anthropic.Anthropic(api_key=api_key)

    # Load data
    all_train = load_gsm8k_parquet(GSM8K_TRAIN_PARQUET)
    all_test  = load_gsm8k_parquet(GSM8K_TEST_PARQUET)
    print(f"GSM8K: {len(all_train)} train, {len(all_test)} test problems loaded", flush=True)

    rng_train = random.Random(seed_train)
    rng_val   = random.Random(seed_val)

    train_indices = rng_train.sample(range(len(all_train)), min(n_train, len(all_train)))
    val_indices   = rng_val.sample(range(len(all_test)),  min(n_val,   len(all_test)))

    train_sample = [(i, all_train[i]) for i in train_indices]
    val_sample   = [(i, all_test[i])  for i in val_indices]

    print(f"Sampled: {len(train_sample)} train (seed={seed_train}), "
          f"{len(val_sample)} val (seed={seed_val})", flush=True)

    # Resume: find already-processed indices
    done_train_idxs: set[int] = set()
    done_val_idxs:   set[int] = set()
    if resume:
        for path, done_set in [
            (TRAIN_OUT_PATH, done_train_idxs),
            (TRAIN_REJECTED_PATH, done_train_idxs),
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
        for path in (VAL_OUT_PATH, VAL_REJECTED_PATH):
            if path.exists():
                with open(path) as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                            if "gsm8k_idx" in rec:
                                done_val_idxs.add(rec["gsm8k_idx"])
                        except json.JSONDecodeError:
                            pass
        print(f"Resume: {len(done_train_idxs)} train + {len(done_val_idxs)} val already done",
              flush=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Global counters
    total_input  = 0
    total_output = 0
    n_accepted   = 0
    n_rejected   = 0
    problems_done = 0

    # Failure mode counters
    fail_counts: dict[str, int] = {}

    start_time = time.time()

    def _process_split(
        split_name: str,
        sample: list[tuple[int, dict]],
        out_path: Path,
        rej_path: Path,
        done_idxs: set[int],
    ) -> bool:
        """Returns False if budget exceeded."""
        nonlocal total_input, total_output, n_accepted, n_rejected, problems_done

        mode = "a" if resume and out_path.exists() else "w"
        rej_mode = "a" if resume and rej_path.exists() else "w"

        with open(out_path, mode) as out_f, open(rej_path, rej_mode) as rej_f:
            for pos, (gsm_idx, gsm_rec) in enumerate(sample):
                if gsm_idx in done_idxs:
                    continue

                # Budget check (stop at 95% to avoid overshoot)
                current_cost = compute_cost(total_input, total_output)
                if current_cost > budget_usd * 0.95:
                    print(f"\nBUDGET WARNING: ${current_cost:.2f} approaching "
                          f"limit ${budget_usd:.0f}. Stopping.", flush=True)
                    return False

                problems_done += 1
                print(f"[{split_name} {pos+1}/{len(sample)}] idx={gsm_idx} ", end="",
                      flush=True)

                res = process_problem(client, gsm_rec, gsm_idx, system_prompt, task_template)
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
                    fail_counts[res["status"]] = fail_counts.get(res["status"], 0) + 1
                    rejected_rec = {
                        "gsm8k_idx":    gsm_idx,
                        "question":     gsm_rec["question"],
                        "gold_answer":  res.get("gold_answer"),
                        "status":       res["status"],
                        "error":        res.get("error"),
                        "parsed":       res.get("parsed"),
                        "raw_response": res.get("raw_response"),
                    }
                    rej_f.write(json.dumps(rejected_rec) + "\n")
                    rej_f.flush()
                    n_rejected += 1
                    print(f"FAIL [{res['status']}] (cost=${cost_so_far:.2f})", flush=True)

                # Print cumulative cost every COST_PRINT_INTERVAL problems
                if problems_done % COST_PRINT_INTERVAL == 0:
                    elapsed = time.time() - start_time
                    total_done = n_accepted + n_rejected
                    rate = n_accepted / total_done if total_done > 0 else 0.0
                    print(f"\n--- CHECKPOINT at {problems_done} problems ---", flush=True)
                    print(f"  Accepted: {n_accepted}, Rejected: {n_rejected}, "
                          f"Accept rate: {rate:.1%}", flush=True)
                    print(f"  Cumulative cost: ${cost_so_far:.2f}", flush=True)
                    print(f"  Elapsed: {elapsed/60:.1f} min", flush=True)
                    print(f"  Failure modes: {fail_counts}", flush=True)
                    print(f"---\n", flush=True)

        return True

    print(f"\n{'='*60}", flush=True)
    print(f"FULL RUN: 5000 train + 1000 val, budget=${budget_usd}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # Train split
    ok = _process_split("train", train_sample, TRAIN_OUT_PATH, TRAIN_REJECTED_PATH,
                        done_train_idxs)
    if not ok:
        _print_summary(n_accepted, n_rejected, total_input, total_output, fail_counts,
                       start_time, budget_usd)
        return

    # Val split
    _process_split("val", val_sample, VAL_OUT_PATH, VAL_REJECTED_PATH, done_val_idxs)

    _print_summary(n_accepted, n_rejected, total_input, total_output, fail_counts,
                   start_time, budget_usd)

    # Spot-check 5 random valid records from train
    _spot_check()


def _print_summary(n_accepted, n_rejected, total_input, total_output, fail_counts,
                   start_time, budget_usd):
    final_cost = compute_cost(total_input, total_output)
    elapsed = time.time() - start_time
    total = n_accepted + n_rejected
    rate = n_accepted / total if total > 0 else 0.0

    print(f"\n{'='*60}", flush=True)
    print(f"FULL RUN COMPLETE", flush=True)
    print(f"  Accepted:      {n_accepted}", flush=True)
    print(f"  Rejected:      {n_rejected}", flush=True)
    print(f"  Total:         {total}", flush=True)
    print(f"  Accept rate:   {rate:.1%}", flush=True)
    print(f"  Failure modes: {fail_counts}", flush=True)
    print(f"  Tokens: in={total_input}, out={total_output}", flush=True)
    print(f"  Total cost:    ${final_cost:.4f} (budget=${budget_usd})", flush=True)
    print(f"  Wall-clock:    {elapsed/60:.1f} min ({elapsed/3600:.2f} hr)", flush=True)

    # Per-split accepted counts
    train_ok = sum(1 for _ in open(TRAIN_OUT_PATH)) if TRAIN_OUT_PATH.exists() else 0
    val_ok   = sum(1 for _ in open(VAL_OUT_PATH))   if VAL_OUT_PATH.exists()   else 0
    train_rej = sum(1 for _ in open(TRAIN_REJECTED_PATH)) if TRAIN_REJECTED_PATH.exists() else 0
    val_rej   = sum(1 for _ in open(VAL_REJECTED_PATH))   if VAL_REJECTED_PATH.exists()   else 0
    print(f"  Train accepted: {train_ok} / {train_ok+train_rej}", flush=True)
    print(f"  Val accepted:   {val_ok}  / {val_ok+val_rej}", flush=True)
    print(f"{'='*60}\n", flush=True)


def _spot_check():
    if not TRAIN_OUT_PATH.exists():
        return
    records = []
    with open(TRAIN_OUT_PATH) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not records:
        return
    rng = random.Random(99)
    samples = rng.sample(records, min(5, len(records)))
    print("\n--- SPOT CHECK: 5 random valid train records ---\n", flush=True)
    for i, rec in enumerate(samples):
        print(f"[{i+1}] gsm8k_idx={rec.get('gsm8k_idx')}", flush=True)
        q = rec.get("question", "")
        print(f"  Q: {q[:120]}...", flush=True)
        print(f"  gold_answer={rec.get('gold_answer')}", flush=True)
        print(f"  n_vars={rec.get('n_vars')}, n_factors={rec.get('n_factors')}", flush=True)
        print(f"  factor_types={rec.get('factor_types')}", flush=True)
        print(f"  var_descriptions={rec.get('var_descriptions')}", flush=True)
        print(flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Phase 1 full labeling run (v2 prompt)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip already-processed indices")
    ap.add_argument("--budget", type=float, default=35.0,
                    help="Cost budget in USD (default 35)")
    ap.add_argument("--n-train", type=int, default=5000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--api-key", type=str, default=None)
    args = ap.parse_args()

    api_key = args.api_key
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key is None:
        key_path = Path.home() / "Desktop" / "keys" / "key1.txt"
        if key_path.exists():
            api_key = key_path.read_text().strip()
        else:
            print("ERROR: No API key found.", file=sys.stderr)
            sys.exit(1)

    # Add scripts dir to path for phase1_topological_eval import
    scripts_dir = str(Path(__file__).parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    run(
        n_train=args.n_train,
        n_val=args.n_val,
        budget_usd=args.budget,
        resume=args.resume,
        api_key=api_key,
    )


if __name__ == "__main__":
    main()
