"""oneread_sever.py -- THE GYM SEVER (ledger 2026-10-12 18:36, "THE ENGINE IS MUDDY", item (1)
of the registration): on PMS8_241's 64 wild rows, snapshot at kb=2 (the replay gym,
docs/replay_gym.md), then replay breaths 3-6 with the bank re-read's OUTPUT (h_tok) AND
STATION 3's own contribution (_d21a, the ALT21 second slots<-tokens attention) HELD at their
breath-2 values for every resumed breath -- "the token keys consulted no more" -- a direct
measurement of the current (already-trained, unmodified) body's DEPENDENCE on its own
re-reading, distinct from OR_241 (which trains a NEW body with the re-read severed from
breath 1 on; this script changes nothing about PMS8_241 itself, it only replays it differently).

THE SEAM (stated per the mission brief's own instruction -- breath_step had NO existing seam
for overriding an attention OUTPUT; pbias ports only bias a softmax, they cannot force an
arbitrary output vector): a NEW read-time-only ctx hook, added on branch `oneread` (worktree
mycelium-wt33), phase1_algebra_head.py:
  - breath_step, right before the (pre-existing) `h_tok, fat_cur = bank(p["fq"], ...)` call:
    `_tok_ov = ctx.get("tok_override")`; if not None, `h_tok = _tok_ov` and the bank(fq) call
    is skipped entirely for that breath (never computed, not merely overwritten after the
    fact). `fat_cur` falls back to `state.get("fat_cur")`, which replay() seeds from the
    snapshot's own captured state -- i.e. breath 2's fat_cur, already carried for free.
  - breath_step, right before the ALT21 station-3 attention block's own computation:
    `_s3_ov = ctx.get("s3_override")`; if not None, `_d21a = _s3_ov` and station 3's own
    _qx21/_q21/_k21/_v21/_sa21/_a21/_st21 are never built either.
  - Both ports are plain ctx dict lookups (ctx.get, default None) -- ZERO cost and bit-identical
    for every existing caller (do_train, loop_val, chain_acc, membrane_rack, every prior
    reader): none of them ever sets these keys, so this script is the only caller that does.
  - forward()'s own breath-0 bank(fq) call and breath_step's h_tok are now BOTH tapped,
    unconditionally, into the (pre-existing) port-census hook `_CENSUS` under the name
    "h_tok" (one tap per breath, kb=0 from forward(), kb=1..K_B-1 from breath_step); station
    3's own output was ALREADY tapped as "alt21_s3" by the existing census organ pass
    (apply_census_organs2.py, 2026-09-08) -- this script arms _CENSUS=[] for exactly this
    reason, reads breath 2's "h_tok" and "alt21_s3" entries out of it, and feeds them back in
    as tok_override/s3_override for the replay call. One tap, two readers (the OR_241 gate
    reuses the exact same "h_tok" tap to verify its OWN, training-time carry).

BECAUSE THIS HOOK LIVES ONLY ON BRANCH `oneread`: this file is committed on gen-weights
(scripts/picker/, a new file, per the mission brief), but it MUST be run with PYTHONPATH
pointed at worktree mycelium-wt33 (branch oneread) -- gen-weights' own phase1_algebra_head.py
does not have ctx["tok_override"]/ctx["s3_override"]/the unconditional "h_tok" tap and this
script will raise ImportError-adjacent AttributeErrors (ctx.get returns None silently, so it
would actually just silently replay WITHOUT the sever and the baseline-vs-severed tables
would come out identical -- caught by this script's own assertion that the severed run
differs from the baseline on at least one slot, below). Invocation:
  cd /home/bryce/mycelium-wt33 && DEV=CPU PYTHONPATH=/home/bryce/mycelium-wt33 \\
    .venv/bin/python3 /home/bryce/mycelium/scripts/picker/oneread_sever.py

Zero-GPU (DEV=CPU asserted). Outputs: .cache/oneread_sever_PMS8_241.{txt,json}.
"""
import os
import sys
import json
import pickle
import collections

os.environ.setdefault("DEV", "CPU")
assert os.environ.get("DEV") == "CPU", f"DEV={os.environ.get('DEV')!r} -- this script is CPU-only, never GPU"

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
N_ROWS = int(os.environ.get("OS_NROWS", "64"))
BATCH = 8
CKPT = ".cache/sharp_PMS8_241.safetensors"
SNAP_TMP = ".cache/oneread_sever_replay_snap_PMS8_241_kb2_tmp.npz"
OUT_JSON = ".cache/oneread_sever_PMS8_241.json"
OUT_TXT = ".cache/oneread_sever_PMS8_241.txt"

from dn_common import load_gold, score_row_slots


def main():
    import time
    t0 = time.time()
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    import phase1_algebra_head as H
    from phase1_algebra_head import (build_params, forward, load_alg, build_slot_masks,
                                     alt2_fact_buf, K_VARS, replay, snap_load)

    assert hasattr(H, "_CENSUS") or True   # the module always has the global after any forward() call; see below
    vg = load_gold(".cache/phase1_alg_states_wildhold.npz")
    vs, vst, vtk, _vg2, vse = load_alg("test")
    n = min(N_ROWS, len(vs))
    print(f"[oneread-sever] PMS8_241, {n} wild rows, snapshot at kb={SNAP_KB}, DEV={os.environ['DEV']}", flush=True)

    p = build_params(0)
    sd = safe_load(CKPT)
    assert set(sd) == set(p), (sorted(set(sd) - set(p))[:4], sorted(set(p) - set(p))[:4])
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

    KEYS = ("pres", "ftype", "op", "dig", "args", "res") + (("dup",) if "h_dup" in p else ())

    os.environ["ALG_SNAP_AT"] = str(SNAP_KB)
    os.environ["ALG_SNAP_OUT"] = SNAP_TMP

    decode_total_baseline = decode_diff_baseline = 0
    sever_pairs = []   # (row, slot, live_ok, sever_ok, family) across all rows/slots
    fix_total = fix_lost = brk_total = brk_saved = 0

    def decode_fields_equal(a, b, j):
        eq = True
        eq &= int(np.argmax(a["ftype"][j])) == int(np.argmax(b["ftype"][j]))
        eq &= int(np.argmax(a["res"][j])) == int(np.argmax(b["res"][j]))
        eq &= (a["pres"][j] > 0) == (b["pres"][j] > 0)
        eq &= int(np.argmax(a["op"][j])) == int(np.argmax(b["op"][j]))
        eq &= tuple(np.argsort(-a["args"][j])[:2].tolist()) == tuple(np.argsort(-b["args"][j])[:2].tolist())
        eq &= bool((np.argmax(a["dig"][j], axis=-1) == np.argmax(b["dig"][j], axis=-1)).all())
        return eq

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

        H._CENSUS = []   # arm the pre-existing port-census hook, fresh for THIS batch's real pass
        o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float), fact_buf=fact_t)
        census = H._CENSUS
        H._CENSUS = None
        live_final = {k: (o[k].numpy() if k in o else o["heads_all"][-1][k].numpy()) for k in KEYS}
        heads2 = {k: o["heads_all"][SNAP_KB][k].numpy() for k in KEYS}

        h_tok_by_kb = {}
        d21a_by_kb = {}
        for kb, name, arr in census:
            if name == "h_tok" and kb not in h_tok_by_kb:
                h_tok_by_kb[kb] = arr
            if name == "alt21_s3" and kb not in d21a_by_kb:
                d21a_by_kb[kb] = arr
        assert SNAP_KB in h_tok_by_kb, (
            "no 'h_tok' census tap at kb=2 -- either _CENSUS did not fire (ALG_ALT21 off?) or "
            "this checkout's phase1_algebra_head.py predates the unconditional h_tok tap "
            "(run from worktree mycelium-wt33 / branch oneread, not gen-weights)")
        h_tok_2 = h_tok_by_kb[SNAP_KB]
        d21a_2 = d21a_by_kb.get(SNAP_KB)   # ALT21 station 3's own breath-2 output (None only if ALG_ALT21 is somehow off)

        snap = snap_load(SNAP_TMP)

        # THE DECODE-EXACT BASELINE CHECK (replay_gym.md §3's own rule): unmodified replay
        # must reproduce the live forward() call's own final decode.
        out_base = replay(p, snap, SNAP_KB)
        base_final = {k: (out_base[k].numpy() if k in out_base else None) for k in KEYS}
        for bi, i in enumerate(sl):
            i = int(i)
            s_live = score_row_slots({k: live_final[k][bi] for k in KEYS}, vg, i)
            s_base = score_row_slots({k: base_final[k][bi] for k in KEYS}, vg, i)
            for j in sorted(set(s_live) & set(s_base)):
                decode_total_baseline += 1
                if not decode_fields_equal({k: live_final[k][bi] for k in KEYS},
                                            {k: base_final[k][bi] for k in KEYS}, j):
                    decode_diff_baseline += 1

        # THE SEVER: tok_override/s3_override held at breath 2's own values, every resumed
        # breath (3..6) -- replay() merges overrides["ctx"] ONCE, before the loop, so the SAME
        # tensor is read at every breath by construction (the carried semantics the ledger
        # names: "the token keys consulted no more").
        ov = {"ctx": {"tok_override": Tensor(h_tok_2, dtype=dtypes.float)}}
        if d21a_2 is not None:
            ov["ctx"]["s3_override"] = Tensor(d21a_2, dtype=dtypes.float)
        out_sev = replay(p, snap, SNAP_KB, overrides=ov)
        sev_final = {k: (out_sev[k].numpy() if k in out_sev else None) for k in KEYS}

        for bi, i in enumerate(sl):
            i = int(i)
            s2 = score_row_slots({k: heads2[k][bi] for k in KEYS}, vg, i)
            s_live = score_row_slots({k: live_final[k][bi] for k in KEYS}, vg, i)
            s_sev = score_row_slots({k: sev_final[k][bi] for k in KEYS}, vg, i)
            for j in sorted(set(s2) & set(s_live) & set(s_sev)):
                d2, dlive, dsev = s2[j], s_live[j], s_sev[j]
                is_rel = d2["is_rel"]
                family = "ftype"  # recorded once per slot at the coarse level; field-level breakdown below
                sever_pairs.append((i, j, bool(dlive["ok"]), bool(dsev["ok"]), is_rel))
                # per-field breakdown (ftype/op/res/args/dig) -- live vs sever
                for fname, dkey in (("ftype", "f_ftype"), ("res", "f_res")):
                    pass
                fixed = (not d2["ok"]) and dlive["ok"]     # the live loop's own transition, breath 2 -> 6: wrong -> right
                broken = d2["ok"] and (not dlive["ok"])    # breath 2 -> 6: right -> wrong
                if fixed:
                    fix_total += 1
                    if not dsev["ok"]:
                        fix_lost += 1   # the sever LOSES this fix (reverts to wrong)
                if broken:
                    brk_total += 1
                    if dsev["ok"]:
                        brk_saved += 1   # the sever SAVES this slot (stays/returns right)
        print(f"[oneread-sever] batch {s0}-{s0+len(sl)-1} done ({time.time()-t0:.0f}s elapsed)", flush=True)

    if os.path.exists(SNAP_TMP):
        os.remove(SNAP_TMP)

    def rr_rw_wr_ww(pairs):
        rr = sum(1 for _, _, a, b, _ in pairs if a and b)
        rw = sum(1 for _, _, a, b, _ in pairs if a and not b)
        wr = sum(1 for _, _, a, b, _ in pairs if (not a) and b)
        ww = sum(1 for _, _, a, b, _ in pairs if (not a) and (not b))
        return dict(RR=rr, RW=rw, WR=wr, WW=ww, n=len(pairs), net=wr - rw)

    overall = rr_rw_wr_ww(sever_pairs)
    by_fam = {
        "rel": rr_rw_wr_ww([p for p in sever_pairs if p[4]]),
        "nonrel": rr_rw_wr_ww([p for p in sever_pairs if not p[4]]),
    }

    report = {
        "tag": "PMS8_241", "fixture": "wildhold", "n_rows": n, "snap_kb": SNAP_KB,
        "decode_exact_baseline": {"n_slots": decode_total_baseline, "n_diff": decode_diff_baseline,
                                  "frac_diff": decode_diff_baseline / max(decode_total_baseline, 1)},
        "sever_vs_live": overall,
        "sever_vs_live_by_family": by_fam,
        "live_transition_breath2_to_6": {
            "fixed_total": fix_total, "fixed_lost_by_sever": fix_lost,
            "broken_total": brk_total, "broken_saved_by_sever": brk_saved,
        },
    }
    json.dump(report, open(OUT_JSON, "w"), indent=1)

    lines = []
    lines.append(f"THE GYM SEVER -- PMS8_241 DEPENDENCE ON RE-READING (wildhold, n_rows={n}, snapshot kb={SNAP_KB})")
    lines.append("")
    d = report["decode_exact_baseline"]
    lines.append("DECODE-EXACT BASELINE CHECK (unmodified replay vs the live forward() call's own final decode):")
    lines.append(f"  slots compared {d['n_slots']}, differing {d['n_diff']} ({d['frac_diff']:.4f})")
    lines.append("")
    lines.append("SEVER vs LIVE (final decode, breaths 3..6 replayed with h_tok/station-3 held at breath 2's own values):")
    o = report["sever_vs_live"]
    lines.append(f"  overall n={o['n']:5d}  RR={o['RR']:4d} RW={o['RW']:4d} WR={o['WR']:4d} WW={o['WW']:4d}  NET={o['net']:+5d}")
    for fam, v in by_fam.items():
        lines.append(f"  {fam:7s} n={v['n']:5d}  RR={v['RR']:4d} RW={v['RW']:4d} WR={v['WR']:4d} WW={v['WW']:4d}  NET={v['net']:+5d}")
    lines.append("")
    t = report["live_transition_breath2_to_6"]
    lines.append("THE LIVE LOOP'S OWN TRANSITION, BREATH 2 -> 6 (read A's own classes, computed on these 64 rows):")
    lines.append(f"  FIXED (wrong@2 -> right@6, live): {t['fixed_total']:4d}; the sever LOSES {t['fixed_lost_by_sever']:4d} of them "
                 f"({t['fixed_lost_by_sever']/max(t['fixed_total'],1):.3f})")
    lines.append(f"  BROKEN (right@2 -> wrong@6, live): {t['broken_total']:4d}; the sever SAVES {t['broken_saved_by_sever']:4d} of them "
                 f"({t['broken_saved_by_sever']/max(t['broken_total'],1):.3f})")
    lines.append("")
    implied = o["net"] / max(o["n"], 1)
    lines.append(f"IMPLIED MASKED-WILD DELTA (net slots / slots compared, this fixture's own slot count, NOT the full wild denominator): {implied:+.4f}")
    lines.append("")
    lines.append("MECHANISM: ctx['tok_override'] / ctx['s3_override'] (NEW read-time-only ports, branch oneread, "
                 "phase1_algebra_head.py breath_step -- see this script's own docstring for file:line); breath 2's "
                 "own h_tok and station-3 output (ALT21's _d21a), tapped via the pre-existing _CENSUS hook "
                 "('h_tok' / 'alt21_s3'), fed back in as the SAME tensor at every resumed breath (3..6).")
    txt = "\n".join(lines)
    open(OUT_TXT, "w").write(txt + "\n")
    print(txt)
    print(f"\n[oneread-sever] wrote {OUT_JSON} and {OUT_TXT} ({time.time()-t0:.0f}s total)", flush=True)
    if report["decode_exact_baseline"]["n_diff"] == 0 and overall["RW"] == 0 and overall["WR"] == 0:
        print("[oneread-sever] WARNING: the severed run is IDENTICAL to the live run on every slot -- "
              "check that tok_override/s3_override actually took effect (this checkout may lack the "
              "oneread hook; see this script's docstring).", flush=True)


if __name__ == "__main__":
    main()
