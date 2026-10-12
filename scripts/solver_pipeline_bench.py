"""solver_pipeline_bench.py — CPU-only timing for THE SOLVER PIPELINE (perf
lever A). No GPU, no training fired. Three parts:

  [1] rack_release_rows' per-row solve (the ONE sequential, unpooled solve
      site left in the consult path — see solver_pool.py's module docstring
      for why the other two sites are already pooled): sequential / pool(4) /
      pool(8) / memo(cold) / memo(warm) / budget(500) / budget(5000), on 8
      real diet rows (.cache/form_tiny64.jsonl) and 64 real wild rows
      (.cache/wild_admitted_holdout.jsonl) — their GOLD factor graphs stand
      in for "a consult's committed subgraph" (the actual decode needs a
      trained checkpoint + a forward pass, i.e. the GPU, which this script
      never touches; gold graphs are a reasonable, reproducible proxy for
      the complexity a real commit has — stated, not hidden).

  [2] the mask-rebuild cost (_rack_masked_by_slot) in isolation, timed on
      synthetic but correctly-shaped logits (the decode step is pure numpy
      either way — gold vs synthetic logits cost the same to mask-rebuild).

  [3] the ALT3 fixture's end-to-end CPU step time (BATCH=2 STEPS=4): unset /
      pool(4) / pool(4)+memo(1) — RACK is OFF in this exact fixture (found by
      the gate, not assumed), so this part's three numbers are a NULL-RESULT
      check (the new code is inert here) with the honest expectation they
      land within noise of each other; .cache/solverpipe_gate_rack*.log (PART
      2 of the gate) carries the timing comparison where ALG_RACK_RELEASE is
      actually on.

Run: DEV=CPU .venv/bin/python3 scripts/solver_pipeline_bench.py
"""
import json
import os
import statistics as _st
import sys
import time

os.environ.setdefault("DEV", "CPU")
os.environ.setdefault("ALG_WIDE", "1")
os.environ.setdefault("ALG_FTYPES", "9")
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")

import numpy as np

import phase1_algebra_head as H
import solver_pool as SP

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_rows(path, n):
    rows = []
    with open(os.path.join(_ROOT, path)) as f:
        for line in f:
            if len(rows) >= n:
                break
            rows.append(json.loads(line))
    return rows


def _row_to_task(row):
    """A real diet/wild row's GOLD factors -> a rack_release_rows-style task:
    (by_slot, dry_slots, m, n_vars), treating the WHOLE gold graph as the
    committed subgraph (the worst case for a single consult — most real
    commits are a subset — so this times an upper bound on per-row cost)."""
    by_slot = {j: f for j, f in enumerate(row["factors"])}
    dry = list(range(len(row["factors"])))
    return (by_slot, dry, row["m"], row["n_vars"])


def _time_many(fn, n_repeat=3):
    best = None
    for _ in range(n_repeat):
        t0 = time.time()
        fn()
        dt = time.time() - t0
        best = dt if best is None else min(best, dt)
    return best


def bench_release(tag, tasks):
    print(f"\n[1] rack_release_rows' per-row solve — {tag} ({len(tasks)} rows, gold graphs)")
    n = len(tasks)

    def seq():
        return SP.solve_release_rows(list(tasks), workers=0, budget=5000)
    t_seq = _time_many(seq)
    print(f"    sequential         {t_seq * 1000:9.2f} ms total  {t_seq / n * 1000:7.3f} ms/row")

    for w in (4, 8):
        def pooled():
            return SP.solve_release_rows(list(tasks), workers=w, budget=5000)
        t_pool = _time_many(pooled, n_repeat=2)
        print(f"    pool(N={w})          {t_pool * 1000:9.2f} ms total  {t_pool / n * 1000:7.3f} ms/row "
              f"  speedup {t_seq / t_pool:4.2f}x")

    SP._SEQ_MEMO.clear()

    def memo_cold():
        return SP.solve_release_rows(list(tasks), workers=0, budget=5000, use_memo=True)
    t_cold = _time_many(memo_cold, n_repeat=1)
    print(f"    memo cold (1st pass) {t_cold * 1000:9.2f} ms total  {t_cold / n * 1000:7.3f} ms/row "
          f"  (memo size now {len(SP._SEQ_MEMO)})")

    def memo_warm():
        return SP.solve_release_rows(list(tasks), workers=0, budget=5000, use_memo=True)
    t_warm = _time_many(memo_warm, n_repeat=3)
    print(f"    memo warm (repeat)   {t_warm * 1000:9.2f} ms total  {t_warm / n * 1000:7.3f} ms/row "
          f"  speedup {t_cold / max(t_warm, 1e-9):5.1f}x  hits={SP._memo_stats[0]} misses={SP._memo_stats[1]}")
    SP._SEQ_MEMO.clear(); SP._memo_stats[0] = SP._memo_stats[1] = 0

    for b in (500, 5000):
        def budgeted():
            return SP.solve_release_rows(list(tasks), workers=0, budget=b)
        t_b = _time_many(budgeted, n_repeat=2)
        statuses = budgeted()
        n_budget = sum(1 for s in statuses if s == "budget")
        print(f"    budget={b:<5d}        {t_b * 1000:9.2f} ms total  {t_b / n * 1000:7.3f} ms/row "
              f"  (exhausted/rejected: {n_budget}/{n})")


def bench_mask_rebuild(tag, rows):
    """Part [2]: _rack_masked_by_slot in isolation, on synthetic (but
    correctly-shaped) logits — the OTHER host cost at a consult besides the
    solve, so the solve's share of the total host phase can be read off."""
    print(f"\n[2] mask-rebuild (_rack_masked_by_slot) — {tag} ({len(rows)} rows, synthetic logits)")
    rng = np.random.RandomState(0)
    L, K, ND, NFT = H.L_FAC, H.K_VARS, H.N_DIG, int(os.environ.get("ALG_FTYPES", "9"))
    synth_rows = []
    for row in rows:
        synth_rows.append({
            "pres": rng.randn(L).astype(np.float32),
            "ftype": rng.randn(L, NFT).astype(np.float32),
            "op": rng.randn(L, 2).astype(np.float32),
            "dig": rng.randn(L, ND, 10).astype(np.float32),
            "args": rng.randn(L, K).astype(np.float32),
            "res": rng.randn(L, K).astype(np.float32),
        })
    texts = [r["text"] for r in rows]

    def run():
        for r, t in zip(synth_rows, texts):
            H._rack_masked_by_slot(r, t)
    t = _time_many(run, n_repeat=5)
    print(f"    {t * 1000:9.2f} ms total  {t / len(rows) * 1000:7.3f} ms/row")
    return t / len(rows)


def bench_device_copy_proxy():
    """Part [2b]: DEV=CPU's own realize()+.numpy() marshal, as the ONLY
    stand-in for host<->device copy cost this script can measure without a
    GPU — on tinygrad's CPU backend this is NOT the same latency as a real
    AMD/AM round trip (likely a severe underestimate: no PCIe, no driver
    call), so it is reported as a proxy only, explicitly unverified for the
    real number (the report says so again)."""
    from tinygrad import Tensor, dtypes
    shapes = [(8, H.L_FAC, H.K_VARS), (64, H.L_FAC, H.K_VARS)]
    print("\n[2b] host<->device copy PROXY (DEV=CPU realize()+.numpy(); NOT the real GPU number)")
    for b, l, k in shapes:
        def run():
            x = Tensor(np.random.randn(b, l, k).astype(np.float32))
            return (x * 2).realize().numpy()
        t = _time_many(run, n_repeat=5)
        print(f"    shape ({b},{l},{k}): {t * 1000:7.3f} ms/call")


def bench_end_to_end():
    print("\n[3] end-to-end CPU step time, ALT3 fixture (BATCH=2 STEPS=4) — RACK OFF in this exact "
          "fixture (found by the gate: ALG_RACK_RELEASE=0 means rack_release_rows is never called "
          "here, so pool/memo are inert by construction; expect near-noise agreement, not a speedup)")
    import subprocess
    s8 = ("ALG_ROUTER=2 R_GAIN_INIT=1.0 ALG_FREEZE=r_gain ALG_ROUTER_PTR=0.0 ALG_SPAN_ALL=1 "
          "ALG_SPAN_ARGS=1 ALG_SPAN_OP=1 ALG_SPAN_RCUE=1 ALG_SPAN_ARCUE=1 ALG_PTR_SURF=role:add:2.0 "
          "BIND_CODES=.cache/bindbus_codes512r.npz")
    alt3 = s8 + " ALG_ALT3=1 ALG_CERT=2.0 ALG_FACTS_WORKERS=2"
    base = (
        "ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_BREATH=7 ALG_NOTEBOOK=1 "
        "ALG_SIXWAVE=1 NB_PERSLOT=1 ALG_BINDBUS=7 ALG_BIND_D=512 BIND_CODES=.cache/bindbus_codes512.npz "
        "ALG_BUSGARAGE=2 ALG_SHELF_CIRCLE=2 ALG_ALTMASK=1 ALG_ALT21=1 ALG_ALT2=1 ALG_MASKHEAD=1 "
        "ALG_FED=1 ALG_POLAR=1 ALG_POLAR_D=128 ALG_POLAR_EM=0.1 ALG_POLAR_D_INIT=.cache/polar_waist_init_d128u.npz "
        "ALG_PRUNE=pforms,s4,fednl0,lane2 ALG_SLOT_ALL=1 ALG_STELLAR=2 ALG_CLOCK_CANON=1 SC_EVAL=0 "
        "ALG_ALLOW_PEN_TRAIN=1 ALG_TRAIN=.cache/form_tiny64.jsonl ALG_TRAIN_NAME=tiny64 "
        "ALG_TEST=.cache/test_tiny64.jsonl ALG_TEST_NAME=testtiny64 ALG_MASKPREP_CACHE=1 "
        "ALG_MASKPREP_IGNORE=ALG_MASK_COOK,ALG_MASK_COOK_SKEL,ALG_TOK_COOK,ALG_TOK_COOK_S3,ALG_TG_INIT,"
        "ALG_BAL_COOK,ALG_NB2,ALG_ALT5,ALG_PRUNE,ALG_MASKPREP_JIT,ALG_MASKPREP_B,ALG_FACTS_POOL,"
        "ALG_FACTS_FLUSH,ALG_JIT_VAL LR=1e-4 SEED=242 WARM_FROM=.cache/sharp_balV242.safetensors "
        "SNAP_EVERY=4000"
    )
    for name, extra in (("unset", alt3), ("pool4", alt3 + " ALG_SOLVER_POOL=4"),
                        ("pool4_memo1", alt3 + " ALG_SOLVER_POOL=4 ALG_SOLVER_MEMO=1")):
        env = dict(os.environ)
        for kv in (base + " " + extra).split():
            k, _, v = kv.partition("=")
            env[k] = v
        env["BATCH"] = "2"; env["STEPS"] = "4"
        env["ALG_CKPT"] = ".cache/sharp_solverpipebench.safetensors"
        t0 = time.time()
        r = subprocess.run([".venv/bin/python3", "scripts/phase1_algebra_head.py", "--train"],
                           cwd=_ROOT, env=env, capture_output=True, text=True, timeout=600)
        wall = time.time() - t0
        steps = [l for l in r.stdout.splitlines() if l.strip().startswith("step")]
        print(f"    {name:12s} wall={wall:6.1f}s  " + " | ".join(s.strip()[:60] for s in steps[-2:]))
        if r.returncode != 0:
            print(f"      FAILED rc={r.returncode}: {r.stderr[-400:]}")
    try:
        os.remove(os.path.join(_ROOT, ".cache/sharp_solverpipebench.safetensors"))
    except OSError:
        pass


def main():
    diet = _load_rows(".cache/form_tiny64.jsonl", 8)
    wild = _load_rows(".cache/wild_admitted_holdout.jsonl", 64)
    diet_tasks = [_row_to_task(r) for r in diet]
    wild_tasks = [_row_to_task(r) for r in wild]

    bench_release("diet x8", diet_tasks)
    bench_release("wild x64", wild_tasks)
    bench_mask_rebuild("diet x8", diet)
    bench_mask_rebuild("wild x64", wild)
    bench_device_copy_proxy()
    bench_end_to_end()


if __name__ == "__main__":
    main()
