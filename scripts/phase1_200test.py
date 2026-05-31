"""200-problem Phase 1 labeling test.

Runs Haiku on 200 GSM8K train problems (seed=42, same as Stage 1),
saves accepted records to .cache/gsm8k_factor_graphs_200test.jsonl
and rejected to .cache/gsm8k_factor_graphs_200test_rejected.jsonl.

Hard budget cap: $5.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from collections import Counter

# Add scripts dir for phase1_topological_eval import
SCRIPTS_DIR = Path(__file__).parent
REPO_ROOT   = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import anthropic

# Reuse everything from phase1_haiku_label
from phase1_haiku_label import (
    load_gsm8k_parquet,
    process_problem,
    compute_cost,
    TASK_PROMPT,
    GSM8K_TRAIN_PARQUET,
    CACHE_DIR,
)

import random
import time

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_PROBLEMS    = 200
SEED          = 42          # same as Stage 1
BUDGET_USD    = 5.0
OUT_PATH      = CACHE_DIR / "gsm8k_factor_graphs_200test.jsonl"
REJECTED_PATH = CACHE_DIR / "gsm8k_factor_graphs_200test_rejected.jsonl"


def main():
    # Resolve API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key is None:
        key_path = Path.home() / "Desktop" / "keys" / "key1.txt"
        if key_path.exists():
            api_key = key_path.read_text().strip()
        else:
            print("ERROR: No API key found.", file=sys.stderr)
            sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Load and sample
    train_recs = load_gsm8k_parquet(GSM8K_TRAIN_PARQUET)
    rng = random.Random(SEED)
    sample_indices = rng.sample(range(len(train_recs)), N_PROBLEMS)
    sample = [(i, train_recs[i]) for i in sample_indices]

    print(f"\n{'='*60}", flush=True)
    print(f"200-PROBLEM PHASE 1 TEST (seed={SEED})", flush=True)
    print(f"  Output: {OUT_PATH}", flush=True)
    print(f"  Budget: ${BUDGET_USD:.2f}", flush=True)
    print(f"{'='*60}\n", flush=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    total_input  = 0
    total_output = 0
    n_ok         = 0
    n_mismatch   = 0
    n_topo       = 0
    n_schema     = 0
    n_json_err   = 0
    n_api_err    = 0
    op_counts    = Counter()
    var_counts   = []

    with open(OUT_PATH, "w") as out_f, open(REJECTED_PATH, "w") as rej_f:
        for pos, (gsm_idx, gsm_rec) in enumerate(sample):
            # Budget check
            current_cost = compute_cost(total_input, total_output)
            if current_cost > BUDGET_USD * 0.95:
                print(f"\nBUDGET STOP: cost=${current_cost:.2f} approaching ${BUDGET_USD:.2f}",
                      flush=True)
                break

            print(f"[{pos+1:3d}/200] idx={gsm_idx} ", end="", flush=True)

            res = process_problem(client, gsm_rec, gsm_idx, TASK_PROMPT)
            total_input  += res["usage"]["input_tokens"]
            total_output += res["usage"]["output_tokens"]

            status = res["status"]
            if status == "ok":
                record = res["parsed"]
                record["gsm8k_idx"]   = gsm_idx
                record["gold_answer"] = res["gold_answer"]
                record["question"]    = gsm_rec["question"]
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                n_ok += 1
                # Accumulate op/var stats
                for op in record.get("factor_types", []):
                    op_counts[op] += 1
                var_counts.append(record.get("n_vars", 0))
                print(f"OK  computed={res['computed_answer']:.4g}  gold={res['gold_answer']}",
                      flush=True)
            else:
                rejected_rec = {
                    "gsm8k_idx":    gsm_idx,
                    "question":     gsm_rec["question"],
                    "gold_answer":  res.get("gold_answer"),
                    "status":       status,
                    "error":        res.get("error"),
                    "parsed":       res.get("parsed"),
                    "raw_response": res.get("raw_response"),
                }
                rej_f.write(json.dumps(rejected_rec) + "\n")
                rej_f.flush()
                if status == "answer_mismatch": n_mismatch += 1
                elif status == "topo_eval_failure": n_topo += 1
                elif status == "schema_invalid": n_schema += 1
                elif status == "json_parse_failure": n_json_err += 1
                elif status == "api_failure": n_api_err += 1
                print(f"FAIL [{status}]: {str(res.get('error',''))[:70]}",
                      flush=True)

    total = pos + 1
    final_cost = compute_cost(total_input, total_output)
    n_total_fail = total - n_ok
    computable_rate = n_ok / total if total > 0 else 0.0

    print(f"\n{'='*60}", flush=True)
    print(f"200-PROBLEM TEST RESULTS ({total} labeled)", flush=True)
    print(f"  Computable (ok):      {n_ok:3d}  ({computable_rate:.1%})", flush=True)
    print(f"  Answer mismatch:      {n_mismatch:3d}", flush=True)
    print(f"  Topo eval failure:    {n_topo:3d}", flush=True)
    print(f"  Schema invalid:       {n_schema:3d}", flush=True)
    print(f"  JSON parse failure:   {n_json_err:3d}", flush=True)
    print(f"  API failure:          {n_api_err:3d}", flush=True)
    print(f"\n  Op distribution: {dict(op_counts)}", flush=True)
    if var_counts:
        print(f"  Var count: min={min(var_counts)}, max={max(var_counts)}, "
              f"avg={sum(var_counts)/len(var_counts):.1f}", flush=True)
    print(f"\n  Tokens: in={total_input}, out={total_output}", flush=True)
    print(f"  Estimated cost: ${final_cost:.4f}", flush=True)
    print(f"\n  Accepted JSONL: {OUT_PATH}", flush=True)
    print(f"  Rejected JSONL: {REJECTED_PATH}", flush=True)

    # Gate
    target = 0.85
    if computable_rate >= target:
        print(f"\n  GATE PASSED: {computable_rate:.1%} >= {target:.0%}  — PROCEED to 6000-run.",
              flush=True)
    else:
        print(f"\n  GATE FAILED: {computable_rate:.1%} < {target:.0%}  — iterate prompt.",
              flush=True)

    # Sample outputs
    print(f"\n--- SAMPLE OUTPUTS (accepted JSONL) ---", flush=True)
    try:
        accepted = []
        with open(OUT_PATH) as f:
            for line in f:
                accepted.append(json.loads(line))
        for r in accepted[:3]:
            print(f"\n  [idx={r['gsm8k_idx']}] {r['question'][:90]}...")
            print(f"    factor_types={r['factor_types']}")
            print(f"    n_vars={r['n_vars']}, n_factors={r['n_factors']}")
            print(f"    gold_answer={r['gold_answer']}")
    except Exception as e:
        print(f"  (could not read samples: {e})", flush=True)

    print(f"\n--- SAMPLE FAILURES (rejected JSONL) ---", flush=True)
    try:
        rejected = []
        with open(REJECTED_PATH) as f:
            for line in f:
                rejected.append(json.loads(line))
        for r in rejected[:2]:
            print(f"\n  [idx={r['gsm8k_idx']}] status={r['status']}")
            print(f"    {r['question'][:80]}...")
            print(f"    error: {str(r.get('error',''))[:120]}")
    except Exception as e:
        print(f"  (could not read failures: {e})", flush=True)

    print(f"\n{'='*60}\n", flush=True)


if __name__ == "__main__":
    main()
