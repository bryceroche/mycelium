"""200-problem Phase 1 labeling test — v2 prompt.

Same 200 GSM8K train problems (seed=42), same pipeline, but uses
.cache/phase1_haiku_prompt_v2.txt instead of the default TASK_PROMPT.

Saves:
  .cache/gsm8k_factor_graphs_200test_v2.jsonl
  .cache/gsm8k_factor_graphs_200test_v2_rejected.jsonl

Hard budget cap: $3.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from collections import Counter

SCRIPTS_DIR = Path(__file__).parent
REPO_ROOT   = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import anthropic

# Reuse infrastructure from phase1_haiku_label
from phase1_haiku_label import (
    load_gsm8k_parquet,
    process_problem,
    compute_cost,
    extract_json_from_response,
    autocorrect_record,
    validate_record,
    call_haiku,
    extract_gold_answer,
    GSM8K_TRAIN_PARQUET,
    CACHE_DIR,
    MODEL,
    SYSTEM_PROMPT,
    SLEEP_BETWEEN_CALLS,
)

import random
import time

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_PROBLEMS    = 200
SEED          = 42
BUDGET_USD    = 3.0
OUT_PATH      = CACHE_DIR / "gsm8k_factor_graphs_200test_v2.jsonl"
REJECTED_PATH = CACHE_DIR / "gsm8k_factor_graphs_200test_v2_rejected.jsonl"
PROMPT_V2_PATH = CACHE_DIR / "phase1_haiku_prompt_v2.txt"


def process_problem_with_prompt(client, gsm_rec, idx, prompt_template):
    """Run Haiku on one problem using the given prompt template."""
    question = gsm_rec["question"]
    answer   = gsm_rec["answer"]
    gold     = extract_gold_answer(answer)

    prompt = prompt_template.replace("{PROBLEM}", question.strip())
    usage_total = {"input_tokens": 0, "output_tokens": 0}

    import time as _time
    raw_text = None
    MAX_RETRIES = 3
    RETRY_SLEEP = 2.0

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=1500,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            usage = resp.usage
            usage_total["input_tokens"]  += usage.input_tokens
            usage_total["output_tokens"] += usage.output_tokens
            raw_text = resp.content[0].text if resp.content else ""
            break
        except anthropic.RateLimitError:
            wait = RETRY_SLEEP * (2 ** attempt)
            print(f"  [rate limit] sleeping {wait:.1f}s ...", flush=True)
            _time.sleep(wait)
        except anthropic.APIError as e:
            print(f"  [api error attempt {attempt+1}] {e}", flush=True)
            _time.sleep(RETRY_SLEEP)

    _time.sleep(SLEEP_BETWEEN_CALLS)

    result = {
        "idx": idx,
        "question": question,
        "gold_answer": gold,
        "raw_response": raw_text,
        "usage": usage_total,
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

    # Load v2 prompt
    if not PROMPT_V2_PATH.exists():
        print(f"ERROR: v2 prompt not found at {PROMPT_V2_PATH}", file=sys.stderr)
        sys.exit(1)
    # The prompt file is formatted as: "SYSTEM_PROMPT\n\n---\n\nTASK_PROMPT"
    # We need just the task portion (everything after "---\n\n")
    raw_prompt = PROMPT_V2_PATH.read_text()
    if "---\n\n" in raw_prompt:
        task_prompt = raw_prompt.split("---\n\n", 1)[1]
    else:
        task_prompt = raw_prompt
    print(f"Loaded v2 prompt ({len(task_prompt)} chars)", flush=True)

    # Load and sample — same seed=42 as v1
    train_recs = load_gsm8k_parquet(GSM8K_TRAIN_PARQUET)
    rng = random.Random(SEED)
    sample_indices = rng.sample(range(len(train_recs)), N_PROBLEMS)
    sample = [(i, train_recs[i]) for i in sample_indices]

    print(f"\n{'='*60}", flush=True)
    print(f"200-PROBLEM PHASE 1 TEST v2 (seed={SEED})", flush=True)
    print(f"  Prompt: {PROMPT_V2_PATH}", flush=True)
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
            current_cost = compute_cost(total_input, total_output)
            if current_cost > BUDGET_USD * 0.95:
                print(f"\nBUDGET STOP: cost=${current_cost:.2f} approaching ${BUDGET_USD:.2f}",
                      flush=True)
                break

            print(f"[{pos+1:3d}/200] idx={gsm_idx} ", end="", flush=True)

            res = process_problem_with_prompt(client, gsm_rec, gsm_idx, task_prompt)
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
    computable_rate = n_ok / total if total > 0 else 0.0

    print(f"\n{'='*60}", flush=True)
    print(f"200-PROBLEM TEST v2 RESULTS ({total} labeled)", flush=True)
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

    # Compare v1 baseline
    v1_rate = 0.835
    print(f"\n  A/B comparison: v1={v1_rate:.1%}  v2={computable_rate:.1%}  "
          f"delta={computable_rate - v1_rate:+.1%}", flush=True)

    target = 0.87
    if computable_rate >= target:
        print(f"\n  TARGET MET: {computable_rate:.1%} >= {target:.0%}  — adopt v2.",
              flush=True)
    else:
        print(f"\n  TARGET MISSED: {computable_rate:.1%} < {target:.0%}  — review failures.",
              flush=True)

    print(f"\n{'='*60}\n", flush=True)


if __name__ == "__main__":
    main()
