"""B-SWEEP diagnostic for the perceiver JIT step.

For B in {8, 16, 32, 64} (back off on OOM): compile the perceiver JIT step
(per_constraint, K=8), run 3 warm-up steps to steady state, then capture
one timed step + one DEBUG=2 step (separate run per B because DEBUG=2 output
must come from a subprocess to be capturable as text).

This top-level script orchestrates per-B subprocess calls to
perceiver_bsweep_worker.py which does the actual compilation + timing + DEBUG=2
capture for a single B value.

Usage (from /home/bryce/mycelium):
  PERCEIVER_TASK=1 \\
  KENKEN_TRAIN=.cache/kenken_train_curriculum.jsonl \\
  KENKEN_TEST=.cache/kenken_test_curriculum.jsonl \\
  .venv/bin/python scripts/perceiver_bsweep.py
"""
from __future__ import annotations

import json
import os
import sys
import subprocess
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tinygrad.helpers import getenv

KENKEN_TRAIN = getenv("KENKEN_TRAIN", ".cache/kenken_train_curriculum.jsonl")
KENKEN_TEST = getenv("KENKEN_TEST", ".cache/kenken_test_curriculum.jsonl")
K = int(getenv("K", "8"))
SEED = int(getenv("SEED", "42"))
BALL_PATH = "per_constraint"

B_VALUES = [8, 16, 32, 64]
PER_B_TIMEOUT = 900  # 15 min per B — generous for compile + 4 warmup + 2 measure steps


def main():
    assert int(getenv("PERCEIVER_TASK", 0)) > 0, "PERCEIVER_TASK=1 must be set"

    print(f"=== Perceiver B-SWEEP orchestrator ===")
    print(f"B values: {B_VALUES}  K={K}  ball_path={BALL_PATH}")
    print(f"Worker timeout: {PER_B_TIMEOUT}s per B")
    print()

    results = {}
    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "perceiver_bsweep_worker.py")

    for B in B_VALUES:
        print(f"\n{'='*60}")
        print(f"=== B={B} ===")
        print(f"{'='*60}", flush=True)

        out_json = f"/tmp/perceiver_bsweep_B{B}.json"
        out_log = f"/tmp/perceiver_bsweep_B{B}.log"

        env = os.environ.copy()
        env["PERCEIVER_TASK"] = "1"
        env["PERCEIVER_HOIST_BIAS"] = "0"
        env["PERCEIVER_FP16_THINK"] = "0"
        env["PERCEIVER_DEFUSE_BREATH"] = "0"
        env["KENKEN_TRAIN"] = KENKEN_TRAIN
        env["KENKEN_TEST"] = KENKEN_TEST
        env["K"] = str(K)
        env["B"] = str(B)
        env["SEED"] = str(SEED)
        env["BALL_PATH"] = BALL_PATH
        env["OUT_JSON"] = out_json
        env["LOG_FILE"] = out_log
        env["DEBUG"] = "2"

        t0 = time.perf_counter()
        try:
            with open(out_log, "w") as logf:
                proc = subprocess.run(
                    [sys.executable, worker],
                    env=env,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    timeout=PER_B_TIMEOUT,
                    cwd="/home/bryce/mycelium",
                )
            elapsed = time.perf_counter() - t0
            print(f"  Worker done in {elapsed:.1f}s (rc={proc.returncode})", flush=True)

            if os.path.exists(out_json):
                with open(out_json) as f:
                    r = json.load(f)
                results[B] = r
                if r.get("oom"):
                    print(f"  OOM: {r.get('error','')[:80]}")
                else:
                    print(f"  s/step={r.get('s_per_step','?'):.3f}  "
                          f"ex/s={r.get('examples_per_sec','?'):.2f}  "
                          f"eater_occ={r.get('eater_occ','N/A')}  "
                          f"eater_GFLOP/s={r.get('eater_gflops','N/A')}")
            else:
                print(f"  ERROR: no output JSON found; rc={proc.returncode}")
                print(f"  Log tail:")
                with open(out_log) as f:
                    lines = f.readlines()
                for line in lines[-20:]:
                    print(f"    {line.rstrip()}")
                results[B] = {"oom": True, "error": f"no output JSON, rc={proc.returncode}"}

        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - t0
            print(f"  TIMEOUT after {elapsed:.0f}s")
            results[B] = {"oom": False, "timeout": True}
        except Exception as e:
            print(f"  ERROR: {e}")
            results[B] = {"oom": True, "error": str(e)}

    # =========================================================================
    # FINAL REPORT
    # =========================================================================
    print("\n" + "=" * 72)
    print(f"B-SWEEP FINAL REPORT  (per_constraint, K={K}, all perf toggles OFF)")
    print("=" * 72)
    hdr = (f"{'B':>4}  {'eater_occ':>9}  {'eater_GF/s':>10}  "
           f"{'s/step':>7}  {'ex/s':>7}  {'eater_%':>7}")
    print(hdr)
    print("-" * 72)
    for B in B_VALUES:
        r = results.get(B, {})
        if r.get("oom"):
            err = r.get("error", "")[:30]
            print(f"{B:>4}  OOM ({err})")
        elif r.get("timeout"):
            print(f"{B:>4}  TIMEOUT")
        else:
            occ = r.get("eater_occ")
            gf = r.get("eater_gflops")
            sps = r.get("s_per_step", float("nan"))
            eps = r.get("examples_per_sec", float("nan"))
            epct = r.get("eater_pct")
            occ_s = str(occ) if occ is not None else "N/A"
            gf_s = str(gf) if gf is not None else "N/A"
            epct_s = f"{epct:.1f}%" if epct is not None else "N/A"
            print(f"{B:>4}  {occ_s:>9}  {gf_s:>10}  {sps:>7.3f}  {eps:>7.2f}  {epct_s:>7}")
    print()

    # Best throughput
    valid = [(B, results[B]) for B in B_VALUES
             if results.get(B) and not results[B].get("oom")
             and not results[B].get("timeout")
             and results[B].get("examples_per_sec") is not None]
    if valid:
        best_B, best_r = max(valid, key=lambda x: x[1]["examples_per_sec"])
        baseline = results.get(8, {}).get("examples_per_sec")
        ratio = best_r["examples_per_sec"] / baseline if baseline else float("nan")
        print(f"Best throughput: B={best_B}  {best_r['examples_per_sec']:.2f} ex/s  "
              f"({ratio:.2f}x over B=8={baseline:.2f} ex/s if valid)")

    # Save combined results
    out_combined = "/tmp/perceiver_bsweep_results.json"
    with open(out_combined, "w") as f:
        json.dump({str(B): r for B, r in results.items()}, f, indent=2)
    print(f"\nCombined results: {out_combined}")
    print("Per-B logs: /tmp/perceiver_bsweep_B{8,16,32,64}.log")


if __name__ == "__main__":
    main()
