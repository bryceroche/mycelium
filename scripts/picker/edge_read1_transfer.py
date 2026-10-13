"""edge_read1_transfer.py -- READ 1: THE TRANSFER GRAPH (geophones across the field).

Delegated task (2026-10-12, zero-GPU parser-head reads). Builds on the replay gym
(docs/replay_gym.md; scripts/picker/dn_read_c.py's loading/snapshot/replay/scoring pattern,
scripts/resonance_read.py's kick-and-listen convention) and dn_common.py's custody-gold scoring.

METHOD. PMS8_241, 64 wild rows (.cache/wild_admitted_holdout.jsonl, same ordering as every other
gym read), snapshot at kb=2 (ALG_SNAP_AT=2 -- NOT 1, so a kick at the save point has three full
breaths (3,4,5,6) to propagate rather than persisting un-metabolized, per the 2026-10-08 resonance
read's own note). For every GOLD-PRESENT slot i (dn_common.score_row_slots / welford_atlas.gold_kind
-- the slot-index == own-variable-index identity holds on this body, K_VARS==L_FAC==24, so "slot i"
and "variable i" are the same bank position; there is no separate variable-slot state to kick), kick
slot i's CONTENT-plane state (the 384 dims outside H._hier_band_dims()'s clock bands; the 128 clock
dims and the radius channel are NEVER perturbed, per BRYCE'S RULING 2026-10-12 and the clock-is-time
law) along its VALUE/IDENTITY direction, replay breaths 3-6, and record at every OTHER slot j the
per-head L2 of the (args, res, dig) logit delta, combined as one scalar response
R(i->j, breath) = sqrt(||d_args||^2 + ||d_res||^2 + ||d_dig||^2).

THE KICK DIRECTION (stated choice): unit(content[i] - kind_mean[gold_kind(i)]), magnitude
eps = 0.1 * ||state[i]||_2 (the FULL 512-dim norm, matching resonance_read.py's convention; applied
only on the 384 content dims). CHOSEN (not the resonance read's own kind-mean direction, and not a
raw ||content||-scaled self-direction) because this read needs a SLOT-SPECIFIC identity signal, not
a kind-generic one: resonance_read.py (2026-10-08) already measured "kick along the kind mean" and
found right/wrong slots ring the same -- a kind axis carries no per-instance identity. The residual
from the slot's own kind mean is exactly the part of its state that is NOT shared by every slot of
that kind, i.e. the closest zero-GPU proxy for "which instance (which argument) this slot IS" that
the existing welford atlas (.cache/welford_atlas_PMS8_241_content.npz) supports without a gradient.

THE MECHANISM FOR GETTING PER-BREATH HEADS CHEAPLY: `_forward_tail`'s own ALG_MINE_BREATHS=1 branch
(phase1_algebra_head.py ~8769) does `out["heads_all"] = [heads_of(_b9) for _b9 in out_breaths]` where
`out_breaths = breaths = state["breaths"]` -- the SAME list `replay()` resumes and appends to
(breath_step's `breaths.append(cur)`, ~8293). Setting os.environ["ALG_MINE_BREATHS"]="1" BEFORE
calling `replay()` therefore returns `out["heads_all"][kb]` for every kb in 0..K_B-1, including the
replayed breaths 3..6 -- no bespoke per-breath loop needed (unlike resonance_read.py, which predates
this env hook's availability from replay() and so re-implements breath_step's body by hand). This
is the SAME module-level `_heads_of(p, s, vst, B, dirgold=None)` forward() itself calls; nothing in
phase1_algebra_head.py is edited to get this (a read-only env flag, already wired for exactly this).

ONE replay() call per GLOBAL SLOT INDEX (0..23), not per (row, slot) pair: rows are batch-independent
(no cross-row attention), so a single batched call kicks slot index `idx` in EVERY row where it is
gold-present simultaneously (zero elsewhere) -- 24 replay() calls total (+1 shared baseline) cover
all ~419 (row, slot) kicks, each call running breaths 3-6 on the full 64-row batch.

DECISIVE CELL / BARS: see the bottom of main() and the printed report.

Outputs: .cache/edge_read1_transfer_PMS8_241.json / .txt. DEV=CPU asserted; .cache/gpu.lock never
touched. Reads (does not write) .cache/sharp_PMS8_241.safetensors, the welford atlas, and the wild
holdout fixture -- all pre-existing banked artifacts.
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
os.environ["ALG_MINE_BREATHS"] = "1"   # the one env flag this whole read leans on (see docstring)

from dn_common import load_gold, score_row_slots
from dn_read_c import decode_fields_equal   # reused verbatim (replay_gym.md's decode-exact rule)
from welford_atlas import gold_kind, KINDS

N_ROWS = int(os.environ.get("EDGE_NROWS", "64"))
SNAP_KB = 2
CKPT = ".cache/sharp_PMS8_241.safetensors"
ATLAS_PATH = ".cache/welford_atlas_PMS8_241_content.npz"
SNAP_TMP = ".cache/edge_read1_snap_PMS8_241_kb2_tmp.npz"
OUT_JSON = ".cache/edge_read1_transfer_PMS8_241.json"
OUT_TXT = ".cache/edge_read1_transfer_PMS8_241.txt"
BREATHS_MEASURED = (3, 4, 5, 6)


def main():
    t0 = time.time()
    print(f"[edge-read1] PMS8_241, {N_ROWS} wild rows, snapshot kb={SNAP_KB}, DEV={os.environ['DEV']}", flush=True)
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    import phase1_algebra_head as H
    from phase1_algebra_head import (build_params, forward, load_alg, build_slot_masks,
                                     alt2_fact_buf, K_VARS, L_FAC, replay, snap_load)

    p = build_params(0)
    sd = safe_load(CKPT)
    assert set(sd) == set(p), (sorted(set(sd) - set(p))[:4], sorted(set(p) - set(sd))[:4])
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

    vg = load_gold(".cache/phase1_alg_states_wildhold.npz")
    vs, vst_, vtk, _vg2, vse = load_alg("test")
    n = min(N_ROWS, len(vs))
    KEYS = ("pres", "ftype", "op", "dig", "args", "res", "dup")   # "dup" dropped below if absent
    BATCH = 8   # dn_read_c.py's validated recipe: BATCH=8 gives decode-exact 419/419; a single
                # B=64 batch measurably does NOT (136/419 differ against dn_read_c.py's own
                # fresh B=8 recheck on the SAME checkpoint/fixture/breath) -- a real batch-size-
                # dependent divergence in forward()/replay() on this body, not explained further
                # here (out of this read's scope; flagged in the deliverable). This script pays
                # the 8x call-count cost to stay on the validated side of that line.
    print(f"[edge-read1] BATCH={BATCH} (dn_read_c.py's validated decode-exact recipe)", flush=True)

    # ---- content dims (H._hier_band_dims' own CONTENT ordering) ----
    bands, clock = H._hier_band_dims()
    CONTENT = np.sort(np.concatenate(bands))
    assert len(CONTENT) == 384, len(CONTENT)
    H_W = H.H_W

    # ---- kind means (the welford atlas; resonance_read.py's own source) ----
    atlas = np.load(ATLAS_PATH, allow_pickle=True)
    akeys = list(atlas["keys"]); amean = atlas["mean"]
    kind_mean = {k: amean[akeys.index(f"kind:{k}")].astype(np.float64) for k in KINDS}

    # ---- per-row gold facts: present slots, kind, args, result (the gold graph too) ----
    present_slots = {}        # row -> sorted [j,...]
    slot_kind = {}             # (row,j) -> kind string
    gold_edges = collections.defaultdict(list)   # row -> [(a,b), ...] undirected variable-level edges
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
    n_slots_total = sum(len(v) for v in present_slots.values())
    print(f"[edge-read1] {n_slots_total} gold-present slots across {n} rows", flush=True)

    # ---- per-batch pipeline: pass1->mask->factbuf->pass2+snapshot->baseline replay->24 kicks ----
    # (dn_read_c.py's validated BATCH=8 recipe, applied row-batch by row-batch; rows across
    # batches never interact, so R/decode-exact/live/base accumulate identically to a single
    # big batch WOULD if that batch size were itself decode-exact, which B=64 measurably is not.)
    live_final = {}; base_final = {}
    n_tot = n_diff = 0
    R = {}   # (row, idx, j, breath) -> (da, dr, dd, comb)
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
        for bi, i in enumerate(sl):
            live_final[int(i)] = {k: live_b[k][bi] for k in _keys_resolved}

        snap = snap_load(SNAP_TMP)
        if os.path.exists(SNAP_TMP):
            os.remove(SNAP_TMP)
        cur0_np = snap["state"]["cur"].realize().numpy().astype(np.float64)   # (B, L_TOT, H_W)

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

        out_base = replay(p, snap, SNAP_KB)
        heads_base = out_base["heads_all"]   # list, index == breath number -- RAW heads_of(breath
                                             # state), i.e. BEFORE the top-level post-loop fusions
                                             # (router pointer prior beta_ptr*(...), busreg terms,
                                             # self-match, direction-bit masking -- all applied to
                                             # out["args"]/out["res"] AFTER heads_all is built,
                                             # phase1_algebra_head.py ~8769-8910). The TRUE final
                                             # decode is out_base[key] (top-level), NOT
                                             # heads_all[-1][key] -- using the latter was THE BUG
                                             # behind this script's first cut showing 136/419
                                             # decode-exact differences (batch size was a red
                                             # herring: a plain-forward() B=64-vs-B=8 check found
                                             # 0/51 differ). Fixed here for the baseline/decode-
                                             # exact check; the kick-response measurement (R,
                                             # below) deliberately keeps using heads_all uniformly
                                             # across breaths 3-6 for an apples-to-apples transfer
                                             # signal (stated in the deliverable).
        base_b = {k: out_base[k].realize().numpy() for k in _keys_resolved}
        for bi, i in enumerate(sl):
            base_final[int(i)] = {k: base_b[k][bi] for k in _keys_resolved}

        for i in (int(x) for x in sl):
            s_live = score_row_slots(live_final[i], vg, i)
            s_base = score_row_slots(base_final[i], vg, i)
            for j in sorted(set(s_live) & set(s_base)):
                n_tot += 1
                if not decode_fields_equal(live_final[i], base_final[i], j):
                    n_diff += 1

        base_np = {breath: {k: heads_base[breath][k].realize().numpy() for k in ("args", "res", "dig")}
                   for breath in BREATHS_MEASURED}

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
            ov = {"state": {"cur": Tensor(pert, dtype=dtypes.float)}}
            out_t = replay(p, snap, SNAP_KB, overrides=ov)
            heads_t = out_t["heads_all"]
            for breath in BREATHS_MEASURED:
                args_t = heads_t[breath]["args"].realize().numpy()
                res_t = heads_t[breath]["res"].realize().numpy()
                dig_t = heads_t[breath]["dig"].realize().numpy()
                args_b = base_np[breath]["args"]; res_b = base_np[breath]["res"]; dig_b = base_np[breath]["dig"]
                for bi in rows_here_bi:
                    i = int(sl[bi])
                    for j in present_slots[i]:
                        if j == idx:
                            continue
                        da = float(np.linalg.norm(args_t[bi, j] - args_b[bi, j]))
                        dr = float(np.linalg.norm(res_t[bi, j] - res_b[bi, j]))
                        dd = float(np.linalg.norm((dig_t[bi, j] - dig_b[bi, j]).reshape(-1)))
                        R[(i, idx, j, breath)] = (da, dr, dd, float(np.sqrt(da * da + dr * dr + dd * dd)))
        print(f"[edge-read1] batch {int(sl[0])}-{int(sl[-1])} done ({time.time()-t0:.0f}s so far)", flush=True)

    KEYS = _keys_resolved
    print(f"[edge-read1] decode-exact baseline: {n_diff}/{n_tot} slots differ ({time.time()-t0:.0f}s so far)", flush=True)
    print(f"[edge-read1] all {n_batches} batches x 24 kick passes done, {len(R)} (row,i,j,breath) cells "
          f"({time.time()-t0:.0f}s so far)", flush=True)

    # ---- R_sum over breaths 3-6, per (row, i, j) ----
    R_sum = collections.defaultdict(float)
    for (i, idx, j, breath), (da, dr, dd, comb) in R.items():
        R_sum[(i, idx, j)] += comb

    # ---- gold hop distance (BFS over the gold variable graph), per row ----
    def bfs(row, src):
        adj = collections.defaultdict(set)
        for a, b in gold_edges.get(row, []):
            adj[a].add(b); adj[b].add(a)
        dist = {src: 0}
        q = collections.deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    hop_of = {}   # (row, i, j) -> hop distance (None if disconnected)
    for i in range(n):
        for src in present_slots[i]:
            d = bfs(i, src)
            for j in present_slots[i]:
                if j != src:
                    hop_of[(i, src, j)] = d.get(j)

    # ---- decisive cell: pointer-wrong vs pointer-right relation slots ----
    def is_rel(k):
        return k.startswith("rel:")

    rows_wrong, rows_right = [], []
    for i in range(n):
        s_base = score_row_slots(base_final[i], vg, i)
        for j in present_slots[i]:
            if not is_rel(slot_kind[(i, j)]):
                continue
            rec = s_base.get(j)
            if rec is None or not rec["is_rel"]:
                continue
            cands = [c for c in present_slots[i] if c != j]
            if not cands:
                continue
            scores = {c: R_sum.get((i, j, c), 0.0) for c in cands}
            ranked = sorted(cands, key=lambda c: -scores[c])
            top1, top2 = ranked[0], ranked[:2]
            gold_args = set(np.where(vg["args"][i, j] > 0.5)[0].tolist())
            own_pred_top1 = int(np.argmax(base_final[i]["args"][j]))
            # THE GOLD-FREE predicted arg set (coordinator's scope extension, 2026-10-12): the
            # SAME rule score_row_slots/decode use for f_args -- dup-predicted single-arg (top-1
            # only) else the model's own unrestricted top-2 of args[j] -- never gold, never
            # restricted to `cands` (the system's real decode isn't restricted to present slots
            # either; `cands` itself IS gold-presence-derived from read 1's own design, a
            # pre-existing choice, caveat stated in the deliverable, not changed here).
            dup_pred = bool("dup" in base_final[i] and base_final[i]["dup"][j] > 0)
            if dup_pred:
                pred_args_set = {int(np.argmax(base_final[i]["args"][j]))}
            else:
                pred_args_set = set(np.argsort(-base_final[i]["args"][j])[:2].tolist())
            chance = 1.0 / len(cands)
            entry = dict(row=i, slot=j, n_cand=len(cands), chance=chance,
                         hit1=bool(top1 in gold_args), hit2=bool(len(set(top2) & gold_args) > 0),
                         echo=bool(top1 == own_pred_top1 and own_pred_top1 in cands),
                         top1=int(top1), top2=[int(x) for x in top2],
                         pred_args_set=sorted(int(x) for x in pred_args_set),
                         gold_args=sorted(int(x) for x in gold_args),
                         pointer_wrong=bool(not rec["f_args"]))
            (rows_wrong if not rec["f_args"] else rows_right).append(entry)

    def summarize(entries):
        if not entries:
            return dict(n=0)
        return dict(n=len(entries),
                    precision_at_1=float(np.mean([e["hit1"] for e in entries])),
                    precision_at_2=float(np.mean([e["hit2"] for e in entries])),
                    chance_rate=float(np.mean([e["chance"] for e in entries])),
                    echo_fraction=float(np.mean([e["echo"] for e in entries])))

    sum_wrong = summarize(rows_wrong)
    sum_right = summarize(rows_right)

    # ---- arrival breath vs gold hop distance (Spearman), damping curve ----
    arrivals, hops = [], []
    damp = {b: [] for b in BREATHS_MEASURED}
    for (i, idx, j, breath), (da, dr, dd, comb) in R.items():
        damp[breath].append(comb)
    arrival_breath = {}
    for (i, idx, j), tot in R_sum.items():
        bym = {b: R[(i, idx, j, b)][3] for b in BREATHS_MEASURED}
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

    report = dict(
        tag="PMS8_241", fixture="wildhold", n_rows=n,
        decode_exact_baseline=dict(n_slots=n_tot, n_diff=n_diff, frac_diff=n_diff / max(n_tot, 1)),
        n_gold_slots=n_slots_total, n_RxJ_cells=len(R_sum),
        pointer_wrong=sum_wrong, pointer_right=sum_right,
        arrival_vs_hop=dict(n_pairs=len(arrivals), spearman_rho=float(rho), spearman_p=float(pval)),
        damping_curve=damping_curve,
        bars=dict(precision_at_1_bar=0.30, kill_le=sum_wrong.get("chance_rate", float("nan")) + 0.05 if sum_wrong else None,
                  echo_kill_frac=0.80),
    )
    json.dump(report, open(OUT_JSON, "w"), indent=1)

    import pickle
    ENTRIES_PKL = ".cache/edge_read1_entries_PMS8_241.pkl"
    pickle.dump(dict(entries=rows_wrong + rows_right, tag="PMS8_241", n_rows=n),
               open(ENTRIES_PKL, "wb"))
    print(f"[edge-read1] wrote {ENTRIES_PKL} ({len(rows_wrong) + len(rows_right)} per-relation entries)", flush=True)

    lines = []
    L = lines.append
    L("READ 1 -- THE TRANSFER GRAPH (PMS8_241, wildhold, n_rows=%d, snapshot kb=%d)" % (n, SNAP_KB))
    L("")
    d = report["decode_exact_baseline"]
    L(f"DECODE-EXACT BASELINE: slots compared {d['n_slots']}, differing {d['n_diff']} ({d['frac_diff']:.4f})")
    L("")
    L(f"GOLD SLOTS: {n_slots_total} across {n} rows; (row,i,j) cells with a measured transfer: {len(R_sum)}")
    L("")
    L("PRECISION TABLE (strongest-responding variable slot vs gold args of the relation):")
    for name, s in (("pointer-WRONG", sum_wrong), ("pointer-RIGHT", sum_right)):
        if s.get("n"):
            L(f"  {name:14s} n={s['n']:4d}  precision@1={s['precision_at_1']:.3f}  precision@2={s['precision_at_2']:.3f}  "
              f"chance={s['chance_rate']:.3f}  echo_frac={s['echo_fraction']:.3f}")
        else:
            L(f"  {name:14s} n=0 (no decisions)")
    L("")
    av = report["arrival_vs_hop"]
    L(f"ARRIVAL BREATH vs GOLD HOP DISTANCE: n_pairs={av['n_pairs']}  spearman_rho={av['spearman_rho']:.3f}  p={av['spearman_p']:.4f}")
    L("")
    L("DAMPING CURVE (mean combined response ||d_args,d_res,d_dig|| by breath):")
    L("  " + "  ".join(f"b{b}={damping_curve[b]:.4f}" for b in BREATHS_MEASURED))
    L("")
    bar_hit = sum_wrong.get("n", 0) > 0 and sum_wrong["precision_at_1"] >= 0.30
    kill_hit = sum_wrong.get("n", 0) > 0 and (sum_wrong["precision_at_1"] <= sum_wrong["chance_rate"] + 0.05
                                              or sum_wrong["echo_fraction"] > 0.80)
    L(f"VERDICT: BAR (precision@1 >= 0.30 on pointer-wrong) {'MET' if bar_hit else 'NOT MET'}; "
      f"KILL (<=chance+0.05 or echo>80%) {'TRIGGERED' if kill_hit else 'not triggered'}.")
    txt = "\n".join(lines)
    open(OUT_TXT, "w").write(txt + "\n")
    print(txt)
    print(f"\n[edge-read1] wrote {OUT_JSON} and {OUT_TXT} ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
