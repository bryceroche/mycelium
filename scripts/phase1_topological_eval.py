"""Phase 1 topological evaluator.

Given a factor-graph record (dict matching the v100 schema), topologically
evaluates the DAG and returns the computed value at query_idx.

Also provides a batch helper that compares against the GSM8K gold answer.

Factor graph schema (v1 — for Phase 1 NL parsing):
  {
    "n_vars": int,
    "n_factors": int,
    "domain": [min, max],
    "factor_types": ["add"|"sub"|"mul"|"div", ...],   # length n_factors
    "factor_args": [[arg1_idx, arg2_idx, result_idx], ...],
    "observed_mask": [0|1, ...],                       # length n_vars
    "observed_values": [int or null, ...],             # length n_vars
    "query_idx": int,
    "var_descriptions": [str, ...],                    # length n_vars
  }

Usage (standalone):
  python scripts/phase1_topological_eval.py --record '{"n_vars":7,...}'
  python scripts/phase1_topological_eval.py --file .cache/gsm8k_factor_graphs_train.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------

def apply_op(op: str, a: float, b: float) -> float | None:
    """Apply op(a, b).  Returns float result or None if invalid."""
    if op in ("add", "+"):
        return a + b
    if op in ("sub", "-"):
        return a - b
    if op in ("mul", "*", "x"):
        return a * b
    if op in ("div", "/"):
        if b == 0:
            return None
        return a / b
    return None


def topo_eval(record: dict[str, Any]) -> dict[str, Any]:
    """Topologically evaluate a factor-graph record.

    Returns a dict with keys:
      "success": bool — True if eval completed (no missing operands)
      "values":  dict {var_idx: float} — computed values for all vars
      "computed_answer": float | None — value at query_idx, or None on failure
      "error":   str | None — human-readable failure message
    """
    n_vars     = int(record["n_vars"])
    n_factors  = int(record["n_factors"])
    obs_mask   = record["observed_mask"]       # list[int]
    obs_vals   = record["observed_values"]     # list[int|None]
    ft_list    = record["factor_types"]        # list[str]
    fa_list    = record["factor_args"]         # list[[int,int,int]]
    query_idx  = int(record["query_idx"])

    # Basic structural checks
    if len(obs_mask) != n_vars:
        return {"success": False, "values": {}, "computed_answer": None,
                "error": f"observed_mask length {len(obs_mask)} != n_vars {n_vars}"}
    if len(obs_vals) != n_vars:
        return {"success": False, "values": {}, "computed_answer": None,
                "error": f"observed_values length {len(obs_vals)} != n_vars {n_vars}"}
    if len(ft_list) != n_factors:
        return {"success": False, "values": {}, "computed_answer": None,
                "error": f"factor_types length {len(ft_list)} != n_factors {n_factors}"}
    if len(fa_list) != n_factors:
        return {"success": False, "values": {}, "computed_answer": None,
                "error": f"factor_args length {len(fa_list)} != n_factors {n_factors}"}

    # Validate factor_args indices
    for fi, fa in enumerate(fa_list):
        if len(fa) != 3:
            return {"success": False, "values": {}, "computed_answer": None,
                    "error": f"factor_args[{fi}] has {len(fa)} elements, expected 3"}
        for vi in fa:
            if not (0 <= int(vi) < n_vars):
                return {"success": False, "values": {}, "computed_answer": None,
                        "error": f"factor_args[{fi}] contains out-of-range index {vi} (n_vars={n_vars})"}

    # Validate query_idx
    if not (0 <= query_idx < n_vars):
        return {"success": False, "values": {}, "computed_answer": None,
                "error": f"query_idx {query_idx} out of range [0, {n_vars})"}

    # Seed values from observed mask
    values: dict[int, float] = {}
    for vi in range(n_vars):
        if obs_mask[vi] == 1:
            ov = obs_vals[vi]
            if ov is None:
                return {"success": False, "values": {}, "computed_answer": None,
                        "error": f"var {vi} has observed_mask=1 but observed_values=null"}
            values[vi] = float(ov)

    # Topological fixed-point (iterate until no new values computed or stuck)
    max_rounds = n_factors + 1
    for _round in range(max_rounds):
        changed = False
        for fi in range(n_factors):
            fa  = fa_list[fi]
            a1i = int(fa[0])
            a2i = int(fa[1])
            res_i = int(fa[2])

            if res_i in values:
                continue  # already computed
            if a1i not in values or a2i not in values:
                continue  # operands not yet available

            result = apply_op(ft_list[fi], values[a1i], values[a2i])
            if result is None:
                return {"success": False, "values": dict(values), "computed_answer": None,
                        "error": f"factor {fi} ({ft_list[fi]}): invalid operation "
                                 f"({values[a1i]} {ft_list[fi]} {values[a2i]})"}
            values[res_i] = result
            changed = True

        if not changed:
            break

    # Check if query_idx was computed
    if query_idx not in values:
        missing = [vi for vi in range(n_vars) if vi not in values]
        return {"success": False, "values": dict(values), "computed_answer": None,
                "error": f"query_idx {query_idx} could not be computed; "
                         f"unresolved vars: {missing}"}

    return {
        "success": True,
        "values": dict(values),
        "computed_answer": values[query_idx],
        "error": None,
    }


# ---------------------------------------------------------------------------
# Gold answer extraction
# ---------------------------------------------------------------------------

def extract_gold_answer(gsm8k_answer: str) -> float | None:
    """Extract the numeric answer from a GSM8K answer string.

    GSM8K answers end with '#### NUMBER'.  Returns float or None.
    """
    parts = gsm8k_answer.strip().split("####")
    if len(parts) < 2:
        return None
    raw = parts[-1].strip().replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------

def _floats_match(a: float, b: float, rtol: float = 1e-4, atol: float = 0.01) -> bool:
    """True if a and b are close enough (handles integer answers + small rounding)."""
    return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)


def check_record_against_gold(
    record: dict[str, Any],
    gold_answer: float,
) -> dict[str, Any]:
    """Run topo_eval and compare to gold_answer.

    Returns dict:
      "computable": bool — factor graph evaluates correctly to gold answer
      "eval_result": the topo_eval result dict
      "gold": float
      "computed": float | None
      "match": bool — computed ≈ gold
    """
    eval_res = topo_eval(record)
    computed = eval_res.get("computed_answer")
    match = (
        eval_res["success"]
        and computed is not None
        and _floats_match(float(computed), float(gold_answer))
    )
    return {
        "computable": match,
        "eval_result": eval_res,
        "gold": gold_answer,
        "computed": computed,
        "match": match,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    ap = argparse.ArgumentParser(description="Topologically evaluate factor graphs")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--record", type=str, help="JSON string of one factor graph record")
    g.add_argument("--file", type=str, help="JSONL file of records (with gold_answer field)")
    ap.add_argument("--gold", type=float, help="Gold answer for --record mode")
    args = ap.parse_args()

    if args.record:
        rec = json.loads(args.record)
        result = topo_eval(rec)
        print(json.dumps(result, indent=2))
        if args.gold is not None:
            check = check_record_against_gold(rec, args.gold)
            print(f"\nMatch against gold {args.gold}: {check['match']}")
    else:
        # File mode: each line has a factor graph + optionally a gold_answer
        total = correct = failed_parse = failed_topo = 0
        with open(args.file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    failed_parse += 1
                    continue
                total += 1
                gold = rec.get("gold_answer")
                if gold is not None:
                    check = check_record_against_gold(rec, float(gold))
                    if check["computable"]:
                        correct += 1
                    else:
                        failed_topo += 1
                        if total <= 5:
                            print(f"  FAIL: computed={check['computed']}, gold={gold}, "
                                  f"error={check['eval_result'].get('error')}")
                else:
                    result = topo_eval(rec)
                    if result["success"]:
                        correct += 1
                    else:
                        failed_topo += 1

        print(f"\nTotal: {total}, Correct: {correct}, "
              f"Failed topo: {failed_topo}, Failed parse: {failed_parse}")
        if total > 0:
            print(f"Computability rate: {correct/total:.1%}")


if __name__ == "__main__":
    _cli()
