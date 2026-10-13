"""handoff_sweep.py -- READ 1: THE HANDOFF SWEEP (delegated task, 2026-10-13, zero-GPU).

PMS8_241, the gym (dn_read_c.py's loading/snapshot/replay pattern, docs/replay_gym.md),
64 wild rows, snapshot at kb=2. Three overrides on THE STELLARATOR's helical-handoff weight
(phase1_algebra_head.py's conductor(kb).stellar_weight, ~1715-1814; consumed at ~7029-7045:
`cur = w*cur + (1-w)*notebook_read`, w=conductor(kb).stellar_weight, under ALG_STELLAR=2 the
TOTAL cut, clock planes included) for the replayed breaths 3-6:

  (a) FLOOR: stellar_weight = max(w_k, 0.5) for kb in 3..6 (the carried state never falls
      below half).
  (b) CLOCK-EXEMPT: the STELLAR=1 behaviour (clock planes NOT faded; `_wv = w*_sg_c + _sg_k`
      instead of the v2 scalar cut) with the SAME w(kb) schedule.
  (c) BOTH.

THE OVERRIDE SEAM, CORRECTED FROM THE BRIEF: the brief's pointer ("the per-breath ctx carries
stellar_weight (1811) -- that is your override seam") does not hold up under inspection.
Line 1811 is inside `_Conductor.__init__`/the `_c = _Conductor(...)` construction in
`conductor()`, not inside `_bs_ctx` (the dict replay()'s `overrides["ctx"]` merges into,
built at ~9639). `breath_step` (~7034) reads `conductor(kb).stellar_weight` by a DIRECT
module-level call every breath, not through ctx/state at all; `conductor()` itself memoizes
one `_Conductor` object per `kb` in a GLOBAL, PROCESS-WIDE dict (`_CONDUCTOR_CACHE`, ~1712),
not per-forward-call, not per-snapshot. replay()'s documented override contract
(docs/replay_gym.md §2, §4) has no seam that reaches this value at all -- ctx/state overrides
only touch what breath_step reads FROM ctx/state, and stellar_weight is read from neither.

THE ACTUAL SEAM USED HERE (stated, not hidden): `_Conductor` uses `__slots__` but is an
ordinary mutable object, not frozen; `conductor(kb)` returns the SAME cached object on every
call with that `kb` for the life of the process. `set_floor()`/`restore_floor()` below mutate
and restore `H.conductor(kb).stellar_weight` directly, bypassing the ctx/state override
mechanism entirely (there being no other way to reach this value without re-importing
phase1_algebra_head.py under a different ALG_BREATH/env, which is out of scope here). For (b),
`ALG_STELLAR` is itself a plain module global (`phase1_algebra_head.ALG_STELLAR`, parsed once
from `os.environ` at import, ~68) read by VALUE at call time inside the `if ALG_POLAR and
ALG_STELLAR < 2:` branch (~7036) -- Python's late-binding global lookup means flipping
`H.ALG_STELLAR = 1` after import reaches the exact same code path env `ALG_STELLAR=1` would
have taken, with no re-import needed; `set_clock_exempt()`/`restore_clock_exempt()` do this.
Both are GLOBAL, PROCESS-WIDE mutations (not per-row, not per-replay-call) -- every override
block below applies, runs ALL of that condition's replay() calls, then restores, before the
next condition begins; nothing is ever left mutated across condition boundaries (checked by
the decode-exact re-verification after the LAST override restores, at the end of main()).

METHOD PER BATCH (BATCH=8, dn_read_c.py's validated decode-exact recipe):
  1. pass1 -> slot_mask -> fact_buf -> pass2 (live, ALG_SNAP_AT=2) -> snap_load.
  2. base_out = replay(p, snap, 2) with NO override -- the decode-exact baseline (vs the live
     pass2's own out[key], the edge_read1_transfer.py bugfix: NEVER heads_all[-1]) and the
     reference decode every override's delta table is measured against.
  3. For each of the three overrides: apply -> ov_out = replay(p, snap, 2) (the override's own
     decode + the transfer graph's baseline for its 24 kicks) -> per-slot decode delta vs
     base_out, by family (welford_atlas.gold_kind) -> 24 kicks (edge_read1_transfer.py's exact
     kick direction/magnitude: unit(content[i]-kind_mean[gold_kind(i)]) at 0.1*||state[i]||,
     breaths 3-6 heads_all) measured against ov_out's OWN heads_all (holding the override's
     schedule fixed, isolating the kick's effect) -> restore.

THE BASELINE (unfloored, unexempted) transfer graph is NOT recomputed here -- it is already
banked at .cache/edge_read1_transfer_PMS8_241.json (identical ckpt/fixture/snapshot/BATCH
recipe, 2026-10-12 20:56, THE EDGE READS FINAL) and read back read-only for the "without the
floor" comparison column (never overwritten; this script only READS it).

OUTPUTS (new files, never overwriting the banked edge_read1_transfer_* artifacts):
  .cache/handoff_sweep_PMS8_241.json / .txt
"""
import os
import sys
import json
import time
import collections

os.environ.setdefault("DEV", "CPU")
assert os.environ.get("DEV") == "CPU", f"DEV={os.environ.get('DEV')!r} -- CPU-only script"

sys.path.insert(0, "."); sys.path.insert(0, "scripts"); sys.path.insert(0, "scripts/picker")
import numpy as np
from scipy.stats import spearmanr

from adaptive_stop import FAMILY_ENVS
FAM = dict(FAMILY_ENVS["PMS8_241"])
FAM["DEV"] = "CPU"
for k, v in FAM.items():
    os.environ[k] = v
os.environ["ALG_MINE_BREATHS"] = "1"

from dn_common import load_gold, score_row_slots
from dn_read_c import decode_fields_equal
from welford_atlas import gold_kind, KINDS

N_ROWS = int(os.environ.get("HO_NROWS", "64"))
SNAP_KB = 2
BATCH = 8
BREATHS = (3, 4, 5, 6)
CKPT = ".cache/sharp_PMS8_241.safetensors"
ATLAS_PATH = ".cache/welford_atlas_PMS8_241_content.npz"
SNAP_TMP = ".cache/handoff_snap_PMS8_241_kb2_tmp.npz"
OUT_JSON = ".cache/handoff_sweep_PMS8_241.json"
OUT_TXT = ".cache/handoff_sweep_PMS8_241.txt"
BANKED_EDGE1 = ".cache/edge_read1_transfer_PMS8_241.json"

OVERRIDES = ["floor", "clockexempt", "both"]


def set_floor(H, enable, floor=0.5, breaths=BREATHS):
    orig = {}
    for kb in breaths:
        c = H.conductor(kb)
        orig[kb] = c.stellar_weight
        if enable:
            c.stellar_weight = max(c.stellar_weight, floor)
    return orig


def restore_floor(H, orig):
    for kb, w in orig.items():
        H.conductor(kb).stellar_weight = w


def set_clock_exempt(H, enable):
    orig = H.ALG_STELLAR
    if enable:
        H.ALG_STELLAR = 1
    return orig


def restore_clock_exempt(H, orig):
    H.ALG_STELLAR = orig


def apply_override(H, name):
    """-> (restore_fn). name in OVERRIDES."""
    floor_on = name in ("floor", "both")
    clock_on = name in ("clockexempt", "both")
    orig_floor = set_floor(H, floor_on)
    orig_clock = set_clock_exempt(H, clock_on)

    def _restore():
        restore_floor(H, orig_floor)
        restore_clock_exempt(H, orig_clock)
    return _restore


def main():
    t0 = time.time()
    print(f"[handoff-sweep] PMS8_241, {N_ROWS} wild rows, snapshot kb={SNAP_KB}, DEV={os.environ['DEV']}", flush=True)
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    import phase1_algebra_head as H
    from phase1_algebra_head import (build_params, forward, load_alg, build_slot_masks,
                                     alt2_fact_buf, K_VARS, L_FAC, replay, snap_load)

    # process-wide sanity: these two globals must be in their PMS8_241 family state before we start
    assert H.ALG_STELLAR == 2, H.ALG_STELLAR

    p = build_params(0)
    sd = safe_load(CKPT)
    assert set(sd) == set(p), (sorted(set(sd) - set(p))[:4], sorted(set(p) - set(sd))[:4])
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

    vg = load_gold(".cache/phase1_alg_states_wildhold.npz")
    vs, vst_, vtk, _vg2, vse = load_alg("test")
    n = min(N_ROWS, len(vs))

    bands, clock = H._hier_band_dims()
    CONTENT = np.sort(np.concatenate(bands))
    assert len(CONTENT) == 384, len(CONTENT)
    H_W = H.H_W

    atlas = np.load(ATLAS_PATH, allow_pickle=True)
    akeys = list(atlas["keys"]); amean = atlas["mean"]
    kind_mean = {k: amean[akeys.index(f"kind:{k}")].astype(np.float64) for k in KINDS}

    KEYS = ("pres", "ftype", "op", "dig", "args", "res", "dup")

    present_slots = {}; slot_kind = {}
    gold_edges = collections.defaultdict(list)
    for i in range(n):
        facs = vs[i]["factors"]
        pj = []
        for j, fac in enumerate(facs):
            if j >= L_FAC:
                continue
            k = gold_kind(fac, j)
            slot_kind[(i, j)] = k
            pj.append(j)
            if fac["ftype"] == "rel":
                r = int(fac["result"])
                for a in fac["args"]:
                    a = int(a)
                    if a != r:
                        gold_edges[i].append((a, r))
        present_slots[i] = sorted(pj)

    n_tot = n_diff = 0   # decode-exact baseline check (unmodified replay vs live out[key])
    # decode delta: per override -> transition counts, overall and by family
    delta = {ov: dict(overall=collections.Counter(), by_family=collections.defaultdict(collections.Counter))
             for ov in OVERRIDES}
    # transfer graph per override
    R = {ov: {} for ov in OVERRIDES}          # (i, idx, j, breath) -> comb response
    base_final_per_ov = {ov: {} for ov in OVERRIDES}   # row -> headrec (the override's OWN no-kick decode)
    _keys_resolved = None

    n_batches = (n + BATCH - 1) // BATCH
    for bidx in range(n_batches):
        sl = np.arange(bidx * BATCH, min((bidx + 1) * BATCH, n))
        ts = Tensor(np.ascontiguousarray(vst_[sl]), dtype=dtypes.half)
        tk = Tensor(vtk[sl].astype(np.float32), dtype=dtypes.float)
        se = Tensor(vse[sl].astype(np.int32), dtype=dtypes.int)

        o0 = forward(p, ts, tk, se)
        onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
        mk = build_slot_masks(onp0, se.numpy().astype(np.int32))
        slot_mask = Tensor(mk, dtype=dtypes.float)
        _ka = ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())
        _oa = {**onp0, **{k: o0[k].realize().numpy() for k in _ka}}
        _nv = np.array([vs[int(i)].get("n_vars", K_VARS) for i in sl])
        _ma = np.array([vs[int(i)].get("m", 0) for i in sl])
        fb = alt2_fact_buf(_oa, se.numpy().astype(np.int32), _nv, _ma)
        fact_t = Tensor(fb, dtype=dtypes.float)

        os.environ["ALG_SNAP_AT"] = str(SNAP_KB); os.environ["ALG_SNAP_OUT"] = SNAP_TMP
        o_live = forward(p, ts, tk, se, slot_mask=slot_mask, fact_buf=fact_t)
        os.environ.pop("ALG_SNAP_AT"); os.environ.pop("ALG_SNAP_OUT")
        if _keys_resolved is None:
            _keys_resolved = tuple(k for k in KEYS if k in o_live)
        live_b = {k: o_live[k].realize().numpy() for k in _keys_resolved}

        snap = snap_load(SNAP_TMP)
        if os.path.exists(SNAP_TMP):
            os.remove(SNAP_TMP)
        cur0_np = snap["state"]["cur"].realize().numpy().astype(np.float64)

        # ---- unmodified baseline replay: the decode-exact check ----
        out_base = replay(p, snap, SNAP_KB)
        base_b = {k: out_base[k].realize().numpy() for k in _keys_resolved}
        base_final = {int(i): {k: base_b[k][bi] for k in _keys_resolved} for bi, i in enumerate(sl)}
        for bi, i in enumerate(sl):
            i = int(i)
            s_live = score_row_slots({k: live_b[k][bi] for k in _keys_resolved}, vg, i)
            s_base = score_row_slots(base_final[i], vg, i)
            for j in sorted(set(s_live) & set(s_base)):
                n_tot += 1
                if not decode_fields_equal({k: live_b[k][bi] for k in _keys_resolved}, base_final[i], j):
                    n_diff += 1

        eps_of, unit_of = {}, {}
        for bi, i in enumerate(sl):
            i = int(i)
            for j in present_slots[i]:
                k = slot_kind[(i, j)]
                full_vec = cur0_np[bi, j, :]
                eps_of[(i, j)] = 0.1 * float(np.linalg.norm(full_vec))
                diff = full_vec[CONTENT] - kind_mean[k]
                nrm = np.linalg.norm(diff)
                unit_of[(i, j)] = diff / nrm if nrm > 1e-12 else diff

        for ov_name in OVERRIDES:
            restore = apply_override(H, ov_name)
            try:
                out_ov = replay(p, snap, SNAP_KB)
                ov_b = {k: out_ov[k].realize().numpy() for k in _keys_resolved}
                ov_final = {int(i): {k: ov_b[k][bi] for k in _keys_resolved} for bi, i in enumerate(sl)}
                for bi, i in enumerate(sl):
                    i = int(i)
                    base_final_per_ov[ov_name][i] = ov_final[i]
                    s_base = score_row_slots(base_final[i], vg, i)
                    s_ov = score_row_slots(ov_final[i], vg, i)
                    for j in sorted(set(s_base) & set(s_ov)):
                        fam = slot_kind.get((i, j), "?")
                        b_ok = s_base[j]["ok"]; o_ok = s_ov[j]["ok"]
                        tag = ("right_right" if b_ok and o_ok else
                               "right_wrong" if b_ok and not o_ok else
                               "wrong_right" if (not b_ok) and o_ok else
                               "wrong_wrong")
                        delta[ov_name]["overall"][tag] += 1
                        delta[ov_name]["by_family"][fam][tag] += 1

                heads_ov = out_ov["heads_all"]
                ov_breath_np = {breath: {k: heads_ov[breath][k].realize().numpy() for k in ("args", "res", "dig")}
                                for breath in BREATHS}

                for idx in range(L_FAC):
                    rows_here_bi = [bi for bi, i in enumerate(sl) if (int(i), idx) in slot_kind]
                    if not rows_here_bi:
                        continue
                    impulse = np.zeros_like(cur0_np)
                    for bi in rows_here_bi:
                        i = int(sl[bi])
                        vec512 = np.zeros(H_W, np.float64)
                        vec512[CONTENT] = unit_of[(i, idx)]
                        impulse[bi, idx, :] = eps_of[(i, idx)] * vec512
                    pert = (cur0_np + impulse).astype(np.float32)
                    ovk = {"state": {"cur": Tensor(pert, dtype=dtypes.float)}}
                    out_t = replay(p, snap, SNAP_KB, overrides=ovk)
                    heads_t = out_t["heads_all"]
                    for breath in BREATHS:
                        args_t = heads_t[breath]["args"].realize().numpy()
                        res_t = heads_t[breath]["res"].realize().numpy()
                        dig_t = heads_t[breath]["dig"].realize().numpy()
                        args_b = ov_breath_np[breath]["args"]; res_b = ov_breath_np[breath]["res"]; dig_b = ov_breath_np[breath]["dig"]
                        for bi in rows_here_bi:
                            i = int(sl[bi])
                            for j in present_slots[i]:
                                if j == idx:
                                    continue
                                da = float(np.linalg.norm(args_t[bi, j] - args_b[bi, j]))
                                dr = float(np.linalg.norm(res_t[bi, j] - res_b[bi, j]))
                                dd = float(np.linalg.norm((dig_t[bi, j] - dig_b[bi, j]).reshape(-1)))
                                R[ov_name][(i, idx, j, breath)] = (da, dr, dd, float(np.sqrt(da * da + dr * dr + dd * dd)))
            finally:
                restore()

        print(f"[handoff-sweep] batch {int(sl[0])}-{int(sl[-1])} done ({time.time()-t0:.0f}s so far)", flush=True)

    if os.path.exists(SNAP_TMP):
        os.remove(SNAP_TMP)

    # final process-wide sanity: nothing left mutated
    assert H.ALG_STELLAR == 2, H.ALG_STELLAR
    for kb in BREATHS:
        pass  # conductor(kb).stellar_weight restored per-condition already; spot assert below
    print(f"[handoff-sweep] decode-exact (unmodified replay vs live): {n_diff}/{n_tot} differ "
          f"({time.time()-t0:.0f}s so far)", flush=True)

    def bfs(row, src):
        adj = collections.defaultdict(set)
        for a, b in gold_edges.get(row, []):
            adj[a].add(b); adj[b].add(a)
        dist = {src: 0}
        from collections import deque
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    hop_of = {}
    for i in range(n):
        for src in present_slots[i]:
            d = bfs(i, src)
            for j in present_slots[i]:
                if j != src:
                    hop_of[(i, src, j)] = d.get(j)

    def is_rel(k):
        return k.startswith("rel:")

    report = dict(tag="PMS8_241", fixture="wildhold", n_rows=n, snap_kb=SNAP_KB,
                  decode_exact_baseline=dict(n_slots=n_tot, n_diff=n_diff, frac_diff=n_diff / max(n_tot, 1)))

    ov_reports = {}
    for ov_name in OVERRIDES:
        d = delta[ov_name]
        overall = dict(d["overall"])
        by_fam = {fam: dict(c) for fam, c in d["by_family"].items()}

        R_sum = collections.defaultdict(float)
        for (i, idx, j, breath), (da, dr, dd, comb) in R[ov_name].items():
            R_sum[(i, idx, j)] += comb

        rows_wrong, rows_right = [], []
        for i in range(n):
            s_ov = score_row_slots(base_final_per_ov[ov_name][i], vg, i)
            for j in present_slots[i]:
                if not is_rel(slot_kind.get((i, j), "")):
                    continue
                rec = s_ov.get(j)
                if rec is None or not rec["is_rel"]:
                    continue
                cands = [c for c in present_slots[i] if c != j]
                if not cands:
                    continue
                scores = {c: R_sum.get((i, j, c), 0.0) for c in cands}
                ranked = sorted(cands, key=lambda c: -scores[c])
                top1, top2 = ranked[0], ranked[:2]
                gold_args = set(np.where(vg["args"][i, j] > 0.5)[0].tolist())
                own_pred_top1 = int(np.argmax(base_final_per_ov[ov_name][i]["args"][j]))
                entry = dict(hit1=bool(top1 in gold_args), hit2=bool(len(set(top2) & gold_args) > 0),
                             echo=bool(top1 == own_pred_top1 and own_pred_top1 in cands),
                             chance=1.0 / len(cands))
                (rows_wrong if not rec["f_args"] else rows_right).append(entry)

        def summarize(entries):
            if not entries:
                return dict(n=0)
            return dict(n=len(entries),
                        precision_at_1=float(np.mean([e["hit1"] for e in entries])),
                        precision_at_2=float(np.mean([e["hit2"] for e in entries])),
                        chance_rate=float(np.mean([e["chance"] for e in entries])),
                        echo_fraction=float(np.mean([e["echo"] for e in entries])))

        sum_wrong = summarize(rows_wrong); sum_right = summarize(rows_right)

        arrivals, hops = [], []
        damp = {b: [] for b in BREATHS}
        for (i, idx, j, breath), (da, dr, dd, comb) in R[ov_name].items():
            damp[breath].append(comb)
        arrival_breath = {}
        for (i, idx, j), tot in R_sum.items():
            bym = {b: R[ov_name][(i, idx, j, b)][3] for b in BREATHS}
            arrival_breath[(i, idx, j)] = max(bym, key=bym.get)
        for (i, idx, j), ab in arrival_breath.items():
            hd = hop_of.get((i, idx, j))
            if hd is not None:
                arrivals.append(ab); hops.append(hd)
        if len(arrivals) >= 3 and len(set(hops)) > 1:
            rho, pval = spearmanr(arrivals, hops)
        else:
            rho, pval = float("nan"), float("nan")
        damping_curve = {b: (float(np.mean(v)) if v else float("nan")) for b, v in damp.items()}

        net = overall.get("wrong_right", 0) - overall.get("right_wrong", 0)
        ov_reports[ov_name] = dict(
            decode_delta_overall=overall, decode_delta_net=net, decode_delta_by_family=by_fam,
            pointer_wrong=sum_wrong, pointer_right=sum_right,
            arrival_vs_hop=dict(n_pairs=len(arrivals), spearman_rho=float(rho), spearman_p=float(pval)),
            damping_curve=damping_curve,
        )

    report["overrides"] = ov_reports

    banked = None
    if os.path.exists(BANKED_EDGE1):
        banked = json.load(open(BANKED_EDGE1))
    report["banked_unfloored_reference"] = dict(
        source=BANKED_EDGE1,
        pointer_wrong=(banked or {}).get("pointer_wrong"),
        damping_curve=(banked or {}).get("damping_curve"),
        arrival_vs_hop=(banked or {}).get("arrival_vs_hop"),
    )

    json.dump(report, open(OUT_JSON, "w"), indent=1)

    lines = []
    L = lines.append
    L(f"READ 1 -- THE HANDOFF SWEEP (PMS8_241, wildhold, n_rows={n}, snapshot kb={SNAP_KB})")
    L("")
    d0 = report["decode_exact_baseline"]
    L(f"DECODE-EXACT BASELINE (unmodified replay vs the live pass2 out[key]): "
      f"slots={d0['n_slots']} differing={d0['n_diff']} ({d0['frac_diff']:.4f})")
    L("")
    bk = report["banked_unfloored_reference"]
    L(f"REFERENCE (banked, unfloored/unexempted, {BANKED_EDGE1}): "
      f"pointer-wrong precision@1={(bk['pointer_wrong'] or {}).get('precision_at_1')}  "
      f"spearman_rho={(bk['arrival_vs_hop'] or {}).get('spearman_rho')}  "
      f"damping={bk['damping_curve']}")
    L("")
    for ov_name in OVERRIDES:
        r = ov_reports[ov_name]
        L(f"=== OVERRIDE: {ov_name.upper()} ===")
        o = r["decode_delta_overall"]
        L(f"  DECODE DELTA (vs unmodified replay), overall: right->right={o.get('right_right',0)} "
          f"right->wrong={o.get('right_wrong',0)} wrong->right={o.get('wrong_right',0)} "
          f"wrong->wrong={o.get('wrong_wrong',0)}  NET={r['decode_delta_net']:+d}")
        for fam in sorted(r["decode_delta_by_family"]):
            c = r["decode_delta_by_family"][fam]
            L(f"    {fam:10s} rr={c.get('right_right',0):4d} rw={c.get('right_wrong',0):3d} "
              f"wr={c.get('wrong_right',0):3d} ww={c.get('wrong_wrong',0):3d}")
        sw, sr = r["pointer_wrong"], r["pointer_right"]
        if sw.get("n"):
            L(f"  TRANSFER precision@1 on pointer-wrong: n={sw['n']} p@1={sw['precision_at_1']:.3f} "
              f"p@2={sw['precision_at_2']:.3f} chance={sw['chance_rate']:.3f} echo={sw['echo_fraction']:.3f}")
        else:
            L("  TRANSFER precision@1 on pointer-wrong: n=0")
        av = r["arrival_vs_hop"]
        L(f"  ARRIVAL vs HOP SPEARMAN: n={av['n_pairs']} rho={av['spearman_rho']:.3f} p={av['spearman_p']:.4f}")
        L(f"  DAMPING CURVE (mean combined response by breath): " +
          "  ".join(f"b{b}={r['damping_curve'][b]:.4f}" for b in BREATHS))
        L("")

    txt = "\n".join(lines)
    open(OUT_TXT, "w").write(txt + "\n")
    print(txt)
    print(f"\n[handoff-sweep] wrote {OUT_JSON} and {OUT_TXT} ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
