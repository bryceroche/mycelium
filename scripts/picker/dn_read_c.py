"""dn_read_c.py -- READ C: THE TABU REPLAY (ledger 2026-10-12 12:35, denoising-breath reads).

Zero-GPU (DEV=CPU asserted; .cache/gpu.lock never touched). Uses the replay gym
(scripts/phase1_algebra_head.py's replay()/snap_load(), docs/replay_gym.md) on PMS8_241, 64 wild
rows, snapshotting at kb=2 (ALG_SNAP_AT=2) under the body's own family env (adaptive_stop.py's
FAMILY_ENVS["PMS8_241"]) + ALG_MINE_BREATHS=1 (so the SAME forward() call that makes the snapshot
also returns breath-2's raw decode heads, out["heads_all"][2], for the oracle flag -- no second
GPU/CPU pass).

THE MECHANISM, stated once (the task's "keep it to ONE mechanism" rule): the hard tabu enters
through ctx["anchor_bias0"] -- THE BREATH-0 ANCHOR's own pbias port (phase1_algebra_head.py
~7378-7387: `_anch_b0 = ctx.get("anchor_bias0"); if _anch_b0 is not None: _pb_kb = ... + _anch_b0`),
which is added, UNCONDITIONALLY and IDENTICALLY, to the factor bank's per-breath attention bias
(`_pb_kb`, fed into `bank(p["fq"], ..., pbias=_pb_kb, ...)` at line ~7513, before the softmax at
line ~6234 `sc = sc + pbias`) at EVERY breath of replay()'s resumed loop (3,4,5,6) -- exactly the
"anchor" semantics ALG_ANCHOR already uses for a keep-bias, run here in reverse (-1e4, a tabu) via
replay()'s own `overrides["ctx"]` merge (applied once, before the resumed loop, so it is identical
at every resumed breath, matching how anchor_bias0 is normally constant-across-breaths by
construction). No code in phase1_algebra_head.py is touched; this is a pure replay-time override.

Per flagged (row, slot): the "previously-attended mention" = argmax_t state["fat_cur"][row, slot, t]
(breath 2's own slots<-tokens attention, present in every snapshot unconditionally --
phase1_algebra_head.py:7528 `state["fat_cur"] = fat_cur`, never gated), restricted to real tokens
via tokmask. tabu_bias[row, 0, slot, t*] = -1e4, else 0; one flag source -> one additive tensor.

Flag sources: (i) ORACLE = slot wrong at the breath-2 decode (_decode_slots on
out["heads_all"][2], scored against the custody gold, dn_common.score_row_slots). (ii) SOLVER =
the best READ B flagger (unsat_core when the breath-2 parse is unsat; dead_facts/minimality when
solved+unique; free_vars is READ B's weakest flagger by recall and is skipped here for the solver
flag, stated). (iii) PLACEBO = the same per-row COUNT of flags as oracle, at uniformly random
present slots.

Output: .cache/dnread_read_c_PMS8_241.json + .txt ; new snapshot artifacts under
.cache/dnread_replay_snap_PMS8_241_kb2_b<N>.npz (deleted at the end -- scratch, not banked).
"""
import os
import sys
import json
import random
import signal
import collections

os.environ.setdefault("DEV", "CPU")
assert os.environ.get("DEV") == "CPU", f"DEV={os.environ.get('DEV')!r} -- this script is CPU-only, never GPU"
assert not os.path.exists(".cache/gpu.lock") or True, "never touches .cache/gpu.lock regardless"

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/picker")
import numpy as np

from adaptive_stop import FAMILY_ENVS
FAM = dict(FAMILY_ENVS["PMS8_241"])
FAM["DEV"] = "CPU"
FAM["ALG_MINE_BREATHS"] = "1"
for k, v in FAM.items():
    os.environ[k] = v

SNAP_KB = 2
N_ROWS = int(os.environ.get("DNC_NROWS", "64"))
BATCH = 8
CKPT = ".cache/sharp_PMS8_241.safetensors"
SNAP_TMP = ".cache/dnread_replay_snap_PMS8_241_kb2_tmp.npz"
OUT_JSON = ".cache/dnread_read_c_PMS8_241.json"
OUT_TXT = ".cache/dnread_read_c_PMS8_241.txt"

from dn_common import load_gold, score_row_slots

random.seed(0)


class _T(Exception):
    pass


def _alarm(signum, frame):
    raise _T()


def _walled(fn, wall):
    old = signal.signal(signal.SIGALRM, _alarm); signal.setitimer(signal.ITIMER_REAL, wall)
    try:
        return fn()
    except _T:
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, old)


def solver_flag(parse, q, wall=3.0):
    """READ B's best flagger, combined by status, run on a (possibly mid-loop) decoded parse.
    Returns a set of flagged `_slot` values, or empty if no flagger applies / everything walls."""
    import minimality as MIN
    from mycelium.csp_domains import problem_from_algebra3
    from mycelium.csp_core import solve_symbolic
    from alternator_bridge import refuse_and_core
    gv, nvv = MIN._build_gv_nvv(parse, q)
    gmax = max([int(v) for v in gv.values()] + [1])
    m = int(min(10001, max(300, 2 * gmax)))

    def _solve():
        return solve_symbolic(problem_from_algebra3(nvv, parse, gv, m), budget=5000, seed=0)
    r = _walled(_solve, wall)
    if r is None:
        return set()
    if r["status"] == "unsat":
        rc = _walled(lambda: refuse_and_core(nvv, parse, m, budget=1500, seed=0), wall)
        if rc and rc.get("status") == "unsat":
            return {parse[idx]["_slot"] for idx in rc.get("core", [])}
        return set()
    if r["status"] == "solved":
        val = int(r["assignment"][q])
        uniq = _walled(lambda: bool(__import__("mycelium.doors", fromlist=["certify_unique"])
                                    .certify_unique(problem_from_algebra3(nvv, parse, gv, m), q, val, budget=5000)), wall)
        if uniq:
            cert = _walled(lambda: MIN.minimality(parse, q, val, m, wall=wall), wall)
            if cert:
                return {parse[idx]["_slot"] for idx, _ft in cert["dead"]}
    return set()


def tabu_tensor(flags_per_row, fat_cur, tokmask_np, B, L_TOT, T):
    """flags_per_row: {row_bi: [slot_j,...]}. fat_cur: (B,L_TOT,T) breath-2 attention.
    tokmask_np: (B,T). Returns (B,1,L_TOT,T) float32 additive bias, -1e4 at each flagged
    slot's breath-2 attention-argmax real token, 0 elsewhere."""
    bias = np.zeros((B, 1, L_TOT, T), np.float32)
    for bi, slots in flags_per_row.items():
        for j in slots:
            att = fat_cur[bi, j] * (tokmask_np[bi] > 0.5)
            if att.sum() <= 0:
                continue
            t_star = int(att.argmax())
            bias[bi, 0, j, t_star] = -1e4
    return bias


def decode_fields_equal(a, b, j):
    """Discrete-decode equality at slot j between two row-head dicts (replay_gym.md's decode-exact
    rule: argmax-level comparison, never raw float array_equal)."""
    eq = True
    eq &= int(np.argmax(a["ftype"][j])) == int(np.argmax(b["ftype"][j]))
    eq &= int(np.argmax(a["res"][j])) == int(np.argmax(b["res"][j]))
    eq &= (a["pres"][j] > 0) == (b["pres"][j] > 0)
    eq &= int(np.argmax(a["op"][j])) == int(np.argmax(b["op"][j]))
    eq &= tuple(np.argsort(-a["args"][j])[:2].tolist()) == tuple(np.argsort(-b["args"][j])[:2].tolist())
    eq &= bool((np.argmax(a["dig"][j], axis=-1) == np.argmax(b["dig"][j], axis=-1)).all())
    return eq


def main():
    import time
    t0 = time.time()
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    import phase1_algebra_head as H
    from phase1_algebra_head import (build_params, forward, load_alg, build_slot_masks,
                                     alt2_fact_buf, K_VARS, L_FAC, replay, snap_load, _decode_slots)

    vg = load_gold(".cache/phase1_alg_states_wildhold.npz")
    vs, vst, vtk, _vg2, vse = load_alg("test")
    n = min(N_ROWS, len(vs))
    print(f"[dn-read-c] PMS8_241, {n} wild rows, snapshot at kb={SNAP_KB}, DEV={os.environ['DEV']}", flush=True)

    p = build_params(0)
    sd = safe_load(CKPT)
    assert set(sd) == set(p), (sorted(set(sd) - set(p))[:4], sorted(set(p) - set(sd))[:4])
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

    per_slot_rows = {"baseline_vs_live": [], "oracle": [], "solver": [], "placebo": []}
    decode_diff_baseline = 0; decode_total_baseline = 0
    oracle_flag_counts, solver_flag_counts, placebo_flag_counts = [], [], []

    os.environ["ALG_SNAP_AT"] = str(SNAP_KB)
    os.environ["ALG_SNAP_OUT"] = SNAP_TMP

    for s0 in range(0, n, BATCH):
        sl = np.arange(s0, min(s0 + BATCH, n))
        pad = BATCH - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        B = len(sl_p); T = vse.shape[1]
        nv = np.array([vs[int(i)].get("n_vars", K_VARS) for i in sl_p])
        ma = np.array([vs[int(i)].get("m", 0) for i in sl_p])
        se_np = vse[sl_p].astype(np.int32)
        ts = Tensor(np.ascontiguousarray(vst[sl_p]), dtype=dtypes.half)
        tk = Tensor(vtk[sl_p].astype(np.float32))
        se = Tensor(se_np, dtype=dtypes.int)

        o0 = forward(p, ts, tk, se)
        onp0 = {k: o0[k].numpy() for k in ("fat", "args", "res")}
        mk = build_slot_masks(onp0, se_np)
        _ka = ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())
        _oa = {**onp0, **{k: o0[k].numpy() for k in _ka}}
        fb = alt2_fact_buf(_oa, se_np, nv, ma)
        fact_t = Tensor(fb, dtype=dtypes.float)

        o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float), fact_buf=fact_t)
        KEYS = ("pres", "ftype", "op", "dig", "args", "res") + (("dup",) if "dup" in o else ())
        live_final = {k: (o[k].numpy() if k in o else o["heads_all"][-1][k].numpy()) for k in KEYS}
        heads2 = {k: o["heads_all"][SNAP_KB][k].numpy() for k in KEYS}

        snap = snap_load(SNAP_TMP)
        fat_cur2 = snap["state"]["fat_cur"]
        fat_cur2 = fat_cur2.numpy() if hasattr(fat_cur2, "numpy") else np.asarray(fat_cur2)
        tokmask_np = vtk[sl_p].astype(np.float32)
        L_TOT = fat_cur2.shape[1]

        # baseline (unmodified) replay -- the decode-exact gate
        out_base = replay(p, snap, SNAP_KB)
        base_final = {k: (out_base[k].numpy() if k in out_base else None) for k in KEYS}

        flags_oracle, flags_solver, flags_placebo = {}, {}, {}
        for bi, i in enumerate(sl):
            i = int(i)
            row2 = {k: heads2[k][bi] for k in KEYS}
            s2 = score_row_slots(row2, vg, i)
            wrong2 = [j for j, d in s2.items() if not d["ok"]]
            present2 = sorted(s2.keys())
            flags_oracle[bi] = wrong2
            oracle_flag_counts.append(len(wrong2))
            parse2 = _decode_slots(row2)
            sf = sorted(solver_flag(parse2, vs[i]["query_var"]))
            flags_solver[bi] = sf
            solver_flag_counts.append(len(sf))
            flags_placebo[bi] = random.sample(present2, min(len(wrong2), len(present2))) if present2 else []
            placebo_flag_counts.append(len(flags_placebo[bi]))

        for name, flagset in (("oracle", flags_oracle), ("solver", flags_solver), ("placebo", flags_placebo)):
            tb = tabu_tensor(flagset, fat_cur2, tokmask_np, B, L_TOT, int(T))
            ov = {"ctx": {"anchor_bias0": Tensor(tb, dtype=dtypes.float)}}
            out_t = replay(p, snap, SNAP_KB, overrides=ov)
            t_final = {k: (out_t[k].numpy() if k in out_t else None) for k in KEYS}
            for bi, i in enumerate(sl):
                i = int(i)
                s_base = score_row_slots({k: base_final[k][bi] for k in KEYS}, vg, i)
                s_t = score_row_slots({k: t_final[k][bi] for k in KEYS}, vg, i)
                for j in sorted(set(s_base) & set(s_t)):
                    per_slot_rows[name].append((i, j, s_base[j]["ok"], s_t[j]["ok"]))

        for bi, i in enumerate(sl):
            i = int(i)
            s_live = score_row_slots({k: live_final[k][bi] for k in KEYS}, vg, i)
            s_base = score_row_slots({k: base_final[k][bi] for k in KEYS}, vg, i)
            for j in sorted(set(s_live) & set(s_base)):
                decode_total_baseline += 1
                row_live = {k: live_final[k][bi] for k in KEYS}
                row_base = {k: base_final[k][bi] for k in KEYS}
                if not decode_fields_equal(row_live, row_base, j):
                    decode_diff_baseline += 1

        print(f"[dn-read-c] batch {s0}-{s0+len(sl)-1} done ({time.time()-t0:.0f}s elapsed)", flush=True)

    if os.path.exists(SNAP_TMP):
        os.remove(SNAP_TMP)

    def net_corrections(pairs):
        wr = sum(1 for _, _, b, t in pairs if (not b) and t)
        rw = sum(1 for _, _, b, t in pairs if b and (not t))
        return dict(n=len(pairs), WR=wr, RW=rw, net=wr - rw)

    report = {
        "tag": "PMS8_241", "fixture": "wildhold", "n_rows": n, "snap_kb": SNAP_KB,
        "decode_exact_baseline": {"n_slots": decode_total_baseline, "n_diff": decode_diff_baseline,
                                  "frac_diff": decode_diff_baseline / max(decode_total_baseline, 1)},
        "flag_counts": {"oracle_mean": float(np.mean(oracle_flag_counts)) if oracle_flag_counts else 0.0,
                       "solver_mean": float(np.mean(solver_flag_counts)) if solver_flag_counts else 0.0,
                       "placebo_mean": float(np.mean(placebo_flag_counts)) if placebo_flag_counts else 0.0,
                       "oracle_rows_fired": sum(1 for c in oracle_flag_counts if c),
                       "solver_rows_fired": sum(1 for c in solver_flag_counts if c)},
        "net_corrections": {name: net_corrections(pairs) for name, pairs in per_slot_rows.items() if name != "baseline_vs_live"},
    }
    json.dump(report, open(OUT_JSON, "w"), indent=1)

    lines = []
    lines.append(f"READ C -- THE TABU REPLAY (PMS8_241, wildhold, n_rows={n}, snapshot kb={SNAP_KB})")
    lines.append("")
    d = report["decode_exact_baseline"]
    lines.append(f"DECODE-EXACT BASELINE CHECK (unmodified replay vs the live forward() call's own final decode):")
    lines.append(f"  slots compared {d['n_slots']}, differing {d['n_diff']} ({d['frac_diff']:.4f})")
    lines.append("")
    fc = report["flag_counts"]
    lines.append(f"FLAG COUNTS (mean per row): oracle={fc['oracle_mean']:.2f} (fired {fc['oracle_rows_fired']}/{n}) "
                 f"solver={fc['solver_mean']:.2f} (fired {fc['solver_rows_fired']}/{n}) placebo={fc['placebo_mean']:.2f}")
    lines.append("")
    lines.append("NET CORRECTIONS vs the UNMODIFIED replay (same kb=2 snapshot, breaths 3..6):")
    for name, v in report["net_corrections"].items():
        lines.append(f"  {name:8s} n_slots={v['n']:5d}  wrong->right={v['WR']:4d}  right->wrong={v['RW']:4d}  NET={v['net']:+5d}  "
                     f"[PREDICTION: positive net under oracle = hard alternation helps untrained]")
    lines.append("")
    lines.append("MECHANISM: ctx['anchor_bias0'] (THE BREATH-0 ANCHOR's pbias port, phase1_algebra_head.py "
                 "~7378-7387), added into the factor bank's per-breath pbias (~7513) before the softmax clip "
                 "(~6234 `sc = sc + pbias`); -1e4 at the flagged slot's breath-2 attention-argmax real token.")
    txt = "\n".join(lines)
    open(OUT_TXT, "w").write(txt + "\n")
    print(txt)
    print(f"\n[dn-read-c] wrote {OUT_JSON} and {OUT_TXT} ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
