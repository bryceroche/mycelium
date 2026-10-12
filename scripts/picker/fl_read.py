"""fl_read.py -- THE FLAGGED LOOP's CAPABILITY READ (ARM 1, ledger 2026-10-12 12:35/13:17;
THE DIR2 RULE: oracle flags are a capability test, NEVER a claim, stated on every log line).

Replays a trained FL_241 checkpoint from a kb=2 snapshot (dn_read_c.py's own proven mechanism:
ALG_SNAP_AT=2, replay()'s ctx["anchor_bias0"] override as the hard tabu, reused verbatim --
see dn_read_c.py's own docstring for the port's exact line numbers) under THREE read modes,
each applying the TRAINED flag_emb vector to state["cur"] at the snapshot point (breath 2) --
"applied at breath 2 with the tabu for breaths 3..6", per the spec -- and the tabu on each
flagged slot's own breath-2 attention-argmax token for the resumed breaths 3..6:

  ORACLE  = the slot is WRONG at the breath-2 decode, scored against the custody gold AT READ
            TIME (score_row_slots, dn_common.py) -- an oracle, never fed into the live wild
            run itself; a capability ceiling, not a deployable signal. THE DIR2 RULE.
  PLACEBO = the SAME per-row COUNT of flags as oracle, at uniformly random PRESENT slots
            (regardless of whether they are right or wrong at breath 2).
  NOFLAG  = no flags, no cur injection, no tabu -- the unmodified replay (= the body's plain
            unroll; also the decode-exact baseline check against the live forward() call).

METRICS (the exact lines the chain greps):
  HELD-OUT FIX RATE  = among ORACLE-flagged slots (wrong at breath 2, by construction), the
                       fraction right at breath 6.
  PLACEBO FIX RATE   = among PLACEBO-flagged slots that ALSO happen to be wrong at breath 2
                       (the apples-to-apples subset -- placebo flags random slots, most of
                       which are already right and have nothing to fix), the fraction right
                       at breath 6.
  FLIP RATIO         = P(decode changes after breath 2 | flagged) / P(changes | unflagged),
                       measured WITHIN the oracle run (the training objective's own promise:
                       flagged slots are the only road to change).

THE KNOB CENSUS (printed every run): flag_emb's row-1 std vs state["cur"]'s own std at the
snapshot point (the pre/post knob law). The tabu's own exactness is NOT re-derived here --
out_t["fat"] (replay()'s return) is _forward_tail's `_fed_core(fat)` of the GROUNDING fat
forward() closed over, not state["fat_cur"] at the final breath, and replay() has no
ALG_SNAP_AT hook of its own to re-snapshot it mid-resume; that check is flagloop_gate.sh's
TABU EXACT (fl_gate_checks.py), which reads state["fat_cur"] directly off a real snapshot and
is the one place this road's exactness is actually certified.

Usage: DEV=CPU FL_CKPT=.cache/sharp_FL_241.safetensors FL_NROWS=311 FL_TAG=FL_241 \\
       FL_OUT=.cache/flread_FL_241 .venv/bin/python3 scripts/picker/fl_read.py
(env exactly as the arm's own training recipe, PMS8_241's FAMILY_ENVS + ALG_FLAG=1 -- the
chain sources this; ALG_TEST/ALG_TEST_NAME select the split, default wildhold.)
"""
import os
import sys
import json
import random

os.environ.setdefault("DEV", "CPU")
assert os.environ.get("DEV") == "CPU", f"DEV={os.environ.get('DEV')!r} -- this script is CPU-only, never GPU"

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/picker")
import numpy as np

from dn_common import load_gold, score_row_slots

CKPT = os.environ.get("FL_CKPT", ".cache/sharp_FL_241.safetensors")
N_ROWS = int(os.environ.get("FL_NROWS", "311"))
TAG = os.environ.get("FL_TAG", "FL_241")
GOLD_NPZ = os.environ.get("FL_GOLD_NPZ", ".cache/phase1_alg_states_wildhold.npz")
OUT = os.environ.get("FL_OUT", f".cache/flread_{TAG}")
SNAP_KB = 2
BATCH = 8

random.seed(0)


def main():
    import time
    t0 = time.time()
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    import phase1_algebra_head as H
    from phase1_algebra_head import (build_params, forward, load_alg, build_slot_masks,
                                     alt2_fact_buf, K_VARS, L_FAC, replay, snap_load, _decode_slots)

    assert H.ALG_FLAG, "fl_read.py needs ALG_FLAG=1 in the environment (FL_241's own recipe)"

    vg = load_gold(GOLD_NPZ)
    vs, vst, vtk, _vg2, vse = load_alg("test")
    n = min(N_ROWS, len(vs))
    print(f"[fl-read] {TAG}, {n} rows ({os.environ.get('ALG_TEST_NAME', '?')}), "
          f"snapshot at kb={SNAP_KB}, DEV={os.environ['DEV']}", flush=True)

    p = build_params(0)
    sd = safe_load(CKPT)
    missing = sorted(set(p) - set(sd))
    extra = sorted(set(sd) - set(p))
    assert not extra, f"{CKPT} carries keys build_params(0) does not expect: {extra[:8]}"
    for k in p:
        if k in sd:
            assert tuple(sd[k].shape) == tuple(p[k].shape), (k, tuple(sd[k].shape), tuple(p[k].shape))
            p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    print(f"[fl-read] loaded {len(sd)}/{len(p)} params from {CKPT}"
          + (f" (missing, stayed init: {missing})" if missing else ""), flush=True)

    os.environ["ALG_SNAP_AT"] = str(SNAP_KB)
    SNAP_TMP = os.environ.get("FL_SNAP_TMP", f".cache/flread_snap_{TAG}_tmp.npz")
    os.environ["ALG_SNAP_OUT"] = SNAP_TMP

    per_slot = {"oracle": [], "placebo": [], "noflag": []}         # (row, slot, ok2, ok6, changed)
    flag_counts = {"oracle": [], "placebo": []}
    decode_diff_baseline = 0; decode_total_baseline = 0
    knob_std_cur = []; knob_std_flagrow = None

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
        cur2 = snap["state"]["cur"]

        knob_std_cur.append(float((cur2.numpy() if hasattr(cur2, "numpy") else np.asarray(cur2))[tokmask_np.sum(-1) > 0].std()))
        if knob_std_flagrow is None:
            knob_std_flagrow = float(p["flag_emb"][1].numpy().std()) if "flag_emb" in p else float("nan")

        out_base = replay(p, snap, SNAP_KB)
        base_final = {k: (out_base[k].numpy() if k in out_base else None) for k in KEYS}

        flags_oracle, flags_placebo = {}, {}
        for bi, i in enumerate(sl):
            i = int(i)
            row2 = {k: heads2[k][bi] for k in KEYS}
            s2 = score_row_slots(row2, vg, i)
            wrong2 = [j for j, d in s2.items() if not d["ok"]]
            present2 = sorted(s2.keys())
            flags_oracle[bi] = wrong2
            flag_counts["oracle"].append(len(wrong2))
            flags_placebo[bi] = random.sample(present2, min(len(wrong2), len(present2))) if present2 else []
            flag_counts["placebo"].append(len(flags_placebo[bi]))

        for name, flagset in (("oracle", flags_oracle), ("placebo", flags_placebo)):
            flag_idx = np.zeros((B, L_TOT), np.float32)
            for bi, slots in flagset.items():
                for j in slots:
                    flag_idx[bi, j] = 1.0
            flag_idx_t = Tensor(flag_idx, dtype=dtypes.int)
            cur2_t = cur2 if hasattr(cur2, "shape") else Tensor(cur2)
            cur_mod = cur2_t + p["flag_emb"][flag_idx_t]
            tb = _tabu_tensor(flagset, fat_cur2, tokmask_np, B, L_TOT, int(T))
            ov = {"state": {"cur": cur_mod}, "ctx": {"anchor_bias0": Tensor(tb, dtype=dtypes.float)}}
            out_t = replay(p, snap, SNAP_KB, overrides=ov)
            # NOTE: out_t["fat"] is NOT breath 6's attention -- _forward_tail sets
            # out["fat"] = _fed_core(fat) from forward()'s OWN top-level `fat` argument,
            # which is always the BREATH-0 GROUNDING attention regardless of how many loop
            # breaths ran (verified 2026-10-12: _forward_tail's call site passes the
            # grounding `fat`, not state["fat_cur"]). replay() has no ALG_SNAP_AT hook of
            # its own (grep: the hook exists exactly once, inside forward()'s inline loop) --
            # so this read cannot re-derive the tabu's breath-6 exactness from here. That
            # exactness is certified once, rigorously, by flagloop_gate.sh's TABU EXACT check
            # (fl_gate_checks.py, via a real ALG_SNAP_AT=1 snapshot reading state["fat_cur"]
            # directly) -- not re-measured, and not claimed, in this read.
            t_final = {k: (out_t[k].numpy() if k in out_t else None) for k in KEYS}
            for bi, i in enumerate(sl):
                i = int(i)
                s_base = score_row_slots({k: base_final[k][bi] for k in KEYS}, vg, i)
                s_t = score_row_slots({k: t_final[k][bi] for k in KEYS}, vg, i)
                row2 = {k: heads2[k][bi] for k in KEYS}
                s2 = score_row_slots(row2, vg, i)
                for j in sorted(set(s_t) & set(s2)):
                    flagged = bool(flag_idx[bi, j] > 0.5)
                    ok2 = s2[j]["ok"]; ok6 = s_t[j]["ok"]
                    changed = not _decode_fields_equal(row2, {k: t_final[k][bi] for k in KEYS}, j)
                    per_slot[name].append((i, j, flagged, ok2, ok6, changed))

        for bi, i in enumerate(sl):
            i = int(i)
            s_live = score_row_slots({k: live_final[k][bi] for k in KEYS}, vg, i)
            s_base = score_row_slots({k: base_final[k][bi] for k in KEYS}, vg, i)
            row2 = {k: heads2[k][bi] for k in KEYS}
            s2 = score_row_slots(row2, vg, i)
            for j in sorted(set(s_live) & set(s_base)):
                decode_total_baseline += 1
                if not _decode_fields_equal({k: live_final[k][bi] for k in KEYS},
                                            {k: base_final[k][bi] for k in KEYS}, j):
                    decode_diff_baseline += 1
            # THE NOFLAG READ (the body's plain unroll = the unmodified replay, no cur
            # injection, no tabu): every slot is "unflagged" by construction.
            for j in sorted(set(s_base) & set(s2)):
                changed = not _decode_fields_equal(row2, {k: base_final[k][bi] for k in KEYS}, j)
                per_slot["noflag"].append((i, j, False, s2[j]["ok"], s_base[j]["ok"], changed))

        print(f"[fl-read] batch {s0}-{s0 + len(sl) - 1} done ({time.time() - t0:.0f}s elapsed)", flush=True)

    if os.path.exists(SNAP_TMP):
        os.remove(SNAP_TMP)

    def rate(pairs, pred):
        sel = [x for x in pairs if pred(x)]
        return (sum(1 for x in sel if x[4]) / len(sel) if sel else float("nan")), len(sel)

    oracle_fix, n_oracle_fix = rate(per_slot["oracle"], lambda x: x[2] and not x[3])          # flagged & wrong@2 -> right@6?
    placebo_fix, n_placebo_fix = rate(per_slot["placebo"], lambda x: x[2] and not x[3])
    # NOFLAG has no flagged slots by construction (x[2] is always False) -- its informative
    # number is the NATURAL correction rate among slots already wrong at breath 2, with no
    # intervention at all (the body's own plain-unroll drift; dn_read_c.py READ A's quantity).
    noflag_fix, n_noflag_fix = rate(per_slot["noflag"], lambda x: not x[3]) if per_slot["noflag"] else (float("nan"), 0)

    def flip_rate(pairs, flagged):
        sel = [x for x in pairs if x[2] == flagged]
        return (sum(1 for x in sel if x[5]) / len(sel) if sel else float("nan")), len(sel)

    flip_flagged, n_ff = flip_rate(per_slot["oracle"], True)
    flip_unflagged, n_fu = flip_rate(per_slot["oracle"], False)
    flip_ratio = (flip_flagged / flip_unflagged) if (flip_unflagged and flip_unflagged > 0) else float("inf")

    knob_cur_std = float(np.mean(knob_std_cur)) if knob_std_cur else float("nan")

    report = {
        "tag": TAG, "n_rows": n, "snap_kb": SNAP_KB,
        "decode_exact_baseline": {"n_slots": decode_total_baseline, "n_diff": decode_diff_baseline,
                                  "frac_diff": decode_diff_baseline / max(decode_total_baseline, 1)},
        "flag_counts": {k: (float(np.mean(v)) if v else 0.0) for k, v in flag_counts.items()},
        "NOFLAG_NATURAL_CORRECTION_RATE": noflag_fix, "n_noflag_wrong": n_noflag_fix,
        "HELD_OUT_FIX_RATE": oracle_fix, "n_oracle_wrong_flagged": n_oracle_fix,
        "PLACEBO_FIX_RATE": placebo_fix, "n_placebo_wrong_flagged": n_placebo_fix,
        "FLIP_RATIO": flip_ratio, "flip_flagged": flip_flagged, "flip_unflagged": flip_unflagged,
        "n_flip_flagged": n_ff, "n_flip_unflagged": n_fu,
        "knob_census": {"flag_emb_row1_std": knob_std_flagrow, "cur_std": knob_cur_std,
                        },
    }
    json.dump(report, open(OUT + ".json", "w"), indent=1)

    lines = []
    lines.append(f"FL READ -- THE FLAGGED LOOP CAPABILITY READ ({TAG}, n_rows={n}, snapshot kb={SNAP_KB}) "
                 f"-- DIR2 RULE: oracle flags are a capability test, NEVER a claim")
    lines.append("")
    d = report["decode_exact_baseline"]
    lines.append(f"DECODE-EXACT BASELINE (unmodified replay vs the live forward() call): "
                 f"slots {d['n_slots']} differing {d['n_diff']} ({d['frac_diff']:.4f})")
    lines.append("")
    lines.append(f"FLAG COUNTS (mean per row): oracle={report['flag_counts'].get('oracle',0):.2f} "
                 f"placebo={report['flag_counts'].get('placebo',0):.2f}")
    lines.append("")
    lines.append(f"HELD-OUT FIX RATE (oracle) = {oracle_fix:.4f} (n={n_oracle_fix}) [BAR >= 0.40, KILL < 0.15]")
    lines.append(f"PLACEBO FIX RATE          = {placebo_fix:.4f} (n={n_placebo_fix}) [BAR: >= 0.05 lower than oracle]")
    lines.append(f"NOFLAG NATURAL CORRECTION = {noflag_fix:.4f} (n={n_noflag_fix}) [no intervention; context, not a bar]")
    lines.append(f"FLIP RATIO (oracle run)   = {flip_ratio:.4f} "
                 f"(flagged {flip_flagged:.4f} n={n_ff} / unflagged {flip_unflagged:.4f} n={n_fu}) [BAR >= 50:1]")
    lines.append("")
    lines.append(f"KNOB CENSUS: flag_emb[1].std={knob_std_flagrow:.4f} vs state['cur'].std@breath2={knob_cur_std:.4f} "
                 f"(ratio {knob_std_flagrow/knob_cur_std if knob_cur_std else float('nan'):.4f}) | "
                 f"tabu exactness certified separately by flagloop_gate.sh's TABU EXACT check (not re-derived here)")
    lines.append("")
    lines.append("MECHANISM: dn_read_c.py's own road -- ctx['anchor_bias0'] (THE BREATH-0 ANCHOR's pbias "
                 "port, phase1_algebra_head.py ~7378-7387), folded into the factor bank's per-breath pbias "
                 "(~7513) before the softmax clip (~6234); flag_emb added to state['cur'] at the snapshot "
                 "point (breath 2), the tabu -1e4 at each flagged slot's breath-2 attention-argmax token, "
                 "held constant for breaths 3..6.")
    txt = "\n".join(lines)
    open(OUT + ".txt", "w").write(txt + "\n")
    print(txt)
    print(f"\n[fl-read] wrote {OUT}.json and {OUT}.txt ({time.time()-t0:.0f}s total)", flush=True)


def _tabu_tensor(flags_per_row, fat_cur, tokmask_np, B, L_TOT, T):
    """Fallback copy of dn_read_c.tabu_tensor (kept import-independent: fl_read.py must not
    require dn_read_c.py to be on the path under every caller's sys.path setup)."""
    bias = np.zeros((B, 1, L_TOT, T), np.float32)
    for bi, slots in flags_per_row.items():
        for j in slots:
            att = fat_cur[bi, j] * (tokmask_np[bi] > 0.5)
            if att.sum() <= 0:
                continue
            t_star = int(att.argmax())
            bias[bi, 0, j, t_star] = -1e4
    return bias


def _decode_fields_equal(a, b, j):
    """dn_read_c.py's decode_fields_equal, copied verbatim (argmax-level comparison)."""
    eq = True
    eq &= int(np.argmax(a["ftype"][j])) == int(np.argmax(b["ftype"][j]))
    eq &= int(np.argmax(a["res"][j])) == int(np.argmax(b["res"][j]))
    eq &= (a["pres"][j] > 0) == (b["pres"][j] > 0)
    eq &= int(np.argmax(a["op"][j])) == int(np.argmax(b["op"][j]))
    eq &= tuple(np.argsort(-a["args"][j])[:2].tolist()) == tuple(np.argsort(-b["args"][j])[:2].tolist())
    eq &= bool((np.argmax(a["dig"][j], axis=-1) == np.argmax(b["dig"][j], axis=-1)).all())
    return eq


if __name__ == "__main__":
    main()
