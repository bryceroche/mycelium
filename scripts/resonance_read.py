"""resonance_read.py -- THE GYM's SECOND USE: THE RESONANCE READ (2026-10-08, worktree
mycelium-wt8, branch replay; Bryce: "ring the problem with an impulse to see what resonates").

Registered 2026-10-08 11:49 (docs/phase1_skeleton_spec.md, gen-weights): "RINGING: impulse v2
read the STABILITY (a b1 kick never contracts -- the first look is the fixed point; b2-b4 kicks
contract to 0.2-0.4); the new question is the echo's DIRECTION: resonance = a perturbation decays
ALONG the slot's kind direction and persists; isotropic decay = no resonance. ... PREDICTION: the
echo contracts as before, aligns with the own-kind direction more on RIGHT slots than wrong (a
perceiver channel), and the cross-kind impulse is NOT absorbed (the state has no restoring force
toward its kind: the radius read's 'sharpen by separating, not by contracting')."

Method: PMS8_241's SURF8 env (no consults -- a single ALT2 pass is the whole read), 64 wild rows,
snapshot at kb=2 (THE SAVE POINT, scripts/replay_gate_check.py's own recipe). For every GOLD slot
(gold_kind() ported from scripts/welford_atlas.py:104, which the task names directly) across the
64 rows: an impulse of magnitude eps = 0.1 * ||cur_slot|| (the FULL 512-dim per-slot state norm at
the save point) added, in the 384 content dims only (H._hier_band_dims(), zero in the 128 clock
dims -- the polar waist's own partition), along three unit directions -- (a) the slot's own gold
kind's Welford mean (.cache/welford_atlas_PMS8_241_content.npz), (b) a seeded random unit vector,
(c) the NEAREST OTHER kind's mean (by cosine among the 5 wild kinds: given/rel:add/rel:mul/
rel:sub/rel:div -- a fixed per-kind pairing, not per-row). All gold slots in the 64-row batch are
perturbed SIMULTANEOUSLY per condition (one replay() call per condition, not one per slot) --
STATED SIMPLIFICATION: within a row with >1 gold slot, breath_step's attention can let one
perturbed slot's echo leak into another perturbed slot's own read; the aggregate (mean relative
norm / cosine over ~400 slots) is the intended deliverable, not a single-slot isolation.

replay() returns only the FINAL out; the per-breath RESPONSE needs intermediate states, so this
script inlines replay()'s own loop body with a per-breath capture (the same bespoke-probe pattern
scripts/chalk_dryrun.py used for its attention-mass census, not a second generic entry point).

DEV=CPU asserted; .cache/gpu.lock never touched (no flock in this file).
"""
import os
import sys
import time

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("DEV", "CPU")
assert os.environ.get("DEV") == "CPU"

import numpy as np

N_ROWS = 64
SNAP_KB = 2
SEED = 1234
KINDS = ["given", "rel:add", "rel:mul", "rel:sub", "rel:div"]
CONDITIONS = ["own", "random", "other"]
CKPT = ".cache/sharp_PMS8_241.safetensors"
ATLAS_PATH = ".cache/welford_atlas_PMS8_241_content.npz"
PS_LEGAL_PATH = ".cache/ps_legal_wild_PMS8_241.npz"
OUT_TXT = ".cache/resonance_read_PMS8_241.txt"

_FAM = {
    "DEV": "CPU", "ALG2": "1", "ALG_FTYPES": "9", "ALG_DUP": "1",
    "ALG_HW": "512", "ALG_WIDE": "1", "ALG_BREATH": "7",
    "ALG_NOTEBOOK": "1", "ALG_SIXWAVE": "1", "NB_PERSLOT": "1",
    "ALG_BINDBUS": "7", "ALG_BIND_D": "512",
    "BIND_CODES": ".cache/bindbus_codes512r.npz",
    "ALG_BUSGARAGE": "2", "ALG_SHELF_CIRCLE": "2", "ALG_ALTMASK": "1",
    "ALG_ALT21": "1", "ALG_ALT2": "1", "ALG_MASKHEAD": "1",
    "ALG_FED": "1", "ALG_POLAR": "1", "ALG_POLAR_D": "128",
    "ALG_POLAR_EM": "0.1",
    "ALG_POLAR_D_INIT": ".cache/polar_waist_init_d128u.npz",
    "ALG_PRUNE": "pforms,s4,fednl0,lane2", "ALG_SLOT_ALL": "1",
    "ALG_STELLAR": "2", "ALG_CLOCK_CANON": "1", "SC_EVAL": "0",
    "ALG_TEST": ".cache/wild_admitted_holdout.jsonl", "ALG_TEST_NAME": "wildhold",
    "ALG_ROUTER": "2", "R_GAIN_INIT": "1.0", "ALG_FREEZE": "r_gain", "ALG_ROUTER_PTR": "0.0",
    "ALG_SPAN_ALL": "1", "ALG_SPAN_ARGS": "1", "ALG_SPAN_OP": "1", "ALG_SPAN_RCUE": "1",
    "ALG_SPAN_ARCUE": "1", "ALG_PTR_SURF": "role:add:2.0",
}


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    P = []
    def log(s):
        print(s, flush=True); P.append(s)

    t0 = time.time()
    for k, v in _FAM.items():
        os.environ[k] = v
    import phase1_algebra_head as H
    from tinygrad.nn.state import safe_load
    from tinygrad import Tensor, dtypes
    sys.path.insert(0, "scripts")
    from welford_atlas import gold_kind

    p = H.build_params(0)
    sd = safe_load(CKPT)
    assert set(sd.keys()) == set(p.keys()), (sorted(set(sd) - set(p))[:4], sorted(set(p) - set(sd))[:4])
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

    vs, vst_, vtk, vg, vse = H.load_alg("test")
    n = min(N_ROWS, len(vs))
    sl = np.arange(n)
    ts = Tensor(np.ascontiguousarray(vst_[sl]), dtype=dtypes.half)
    tk = Tensor(vtk[sl].astype(np.float32), dtype=dtypes.float)
    se = Tensor(vse[sl].astype(np.int32), dtype=dtypes.int)

    # pass 1 -> slot_mask -> ALT2 fact_buf (PMS8_241's own single-pass recipe, no consults)
    o0 = H.forward(p, ts, tk, se)
    onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
    mk = H.build_slot_masks(onp0, se.numpy().astype(np.int32))
    slot_mask = Tensor(mk, dtype=dtypes.float)
    _ka = ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())
    _oa = {**onp0, **{k: o0[k].realize().numpy() for k in _ka}}
    _nv = np.array([vs[int(i)].get("n_vars", H.K_VARS) for i in sl])
    _ma = np.array([vs[int(i)].get("m", 0) for i in sl])
    fb = H.alt2_fact_buf(_oa, se.numpy().astype(np.int32), _nv, _ma)
    fact_t = Tensor(fb, dtype=dtypes.float)

    snap_path = ".cache/resonance_snap_PMS8_241_kb2.npz"
    os.environ["ALG_SNAP_AT"] = str(SNAP_KB); os.environ["ALG_SNAP_OUT"] = snap_path
    _ = H.forward(p, ts, tk, se, slot_mask=slot_mask, fact_buf=fact_t)
    os.environ.pop("ALG_SNAP_AT"); os.environ.pop("ALG_SNAP_OUT")
    snap = H.snap_load(snap_path)
    log(f"[resonance] snapshot at kb={SNAP_KB} loaded ({time.time() - t0:.0f}s so far)")

    # ---- content dims (the atlas's own CONTENT ordering: welford_atlas.py:171 etc) ----
    bands, clock = H._hier_band_dims()
    CONTENT = np.sort(np.concatenate(bands))
    assert len(CONTENT) == 384, len(CONTENT)

    # ---- kind means + the nearest-other-kind pairing ----
    atlas = np.load(ATLAS_PATH, allow_pickle=True)
    keys = list(atlas["keys"]); means = atlas["mean"]
    kind_mean = {}
    for k in KINDS:
        idx = keys.index(f"kind:{k}")
        kind_mean[k] = means[idx].astype(np.float64)
    nearest_other = {}
    for k in KINDS:
        best, bestc = None, -2.0
        for k2 in KINDS:
            if k2 == k:
                continue
            c = cos(kind_mean[k], kind_mean[k2])
            if c > bestc:
                bestc, best = c, k2
        nearest_other[k] = best
    log(f"[resonance] nearest-other-kind pairing: " + ", ".join(f"{k}->{nearest_other[k]}({cos(kind_mean[k],kind_mean[nearest_other[k]]):.3f})" for k in KINDS))

    # ---- ps_legal right/wrong lookup ----
    ps = np.load(PS_LEGAL_PATH)
    ok_lookup = {(int(r), int(c)): bool(o) for r, c, o in zip(ps["rows"], ps["slots"], ps["ok"])}

    # ---- gold slots ----
    gold_slots = []
    for i in range(n):
        facs = vs[i]["factors"]
        for j, fac in enumerate(facs):
            if j >= H.L_FAC:
                continue
            k = gold_kind(fac, j)
            if k not in KINDS:
                continue
            right = ok_lookup.get((i, j))
            if right is None:
                continue
            gold_slots.append((i, j, k, bool(right)))
    n_right = sum(1 for *_, r in gold_slots if r)
    log(f"[resonance] {len(gold_slots)} gold slots across {n} rows (right={n_right}, wrong={len(gold_slots) - n_right})")

    cur0_t = snap["state"]["cur"]
    cur0_np = cur0_t.realize().numpy().astype(np.float64)
    H_W = H.H_W

    rng = np.random.RandomState(SEED)
    impulses = {c: np.zeros_like(cur0_np) for c in CONDITIONS}
    eps_of = {}
    unit_dirs = {}   # (i, j, cond) -> 384-dim unit vector applied
    own_unit = {}    # (i, j) -> 384-dim own-kind unit
    other_unit = {}  # (i, j) -> 384-dim nearest-other-kind unit
    for (i, j, k, right) in gold_slots:
        full_vec = cur0_np[i, j, :]
        eps = 0.1 * float(np.linalg.norm(full_vec))
        eps_of[(i, j)] = eps
        o_u = kind_mean[k] / (np.linalg.norm(kind_mean[k]) + 1e-12)
        x_u = kind_mean[nearest_other[k]] / (np.linalg.norm(kind_mean[nearest_other[k]]) + 1e-12)
        r_u = rng.standard_normal(len(CONTENT)); r_u = r_u / np.linalg.norm(r_u)
        own_unit[(i, j)] = o_u; other_unit[(i, j)] = x_u
        for cond, d384 in (("own", o_u), ("random", r_u), ("other", x_u)):
            vec512 = np.zeros(H_W, np.float64)
            vec512[CONTENT] = d384
            impulses[cond][i, j, :] += eps * vec512
            unit_dirs[(i, j, cond)] = d384
    log(f"[resonance] impulses built for {len(gold_slots)} slots x {len(CONDITIONS)} conditions ({time.time() - t0:.0f}s so far)")

    # ---- the bespoke per-breath loop (replay()'s own body, instrumented) ----
    def run_breaths(cur_override_np):
        ctx = dict(snap["ctx"])
        state = {kk: (list(v) if isinstance(v, list) else v) for kk, v in snap["state"].items()}
        if cur_override_np is not None:
            state["cur"] = Tensor(cur_override_np.astype(np.float32), dtype=dtypes.float)
        bank = H._make_bank(p, snap["waist"], snap["tokmask"], n, sent=ctx["mc_sent"], tree=snap["tree"])
        ctx["bank"] = bank; ctx["rot2"] = H.rot2_interleaved
        vst = ctx["vst"]
        facts3 = snap["facts3"]; facts5 = snap["facts5"]
        per_breath = {}
        for kb in range(SNAP_KB + 1, int(snap["K_B"])):
            H.breath_step(p, state, kb, ctx)
            if H.conductor(kb).facts3_inject and facts3 is not None:
                vst = H._fact_inject(p, vst, facts3); ctx["vst"] = vst
            if H.conductor(kb).facts5_inject and facts5 is not None:
                vst = H._fact_inject(p, vst, facts5); ctx["vst"] = vst
            per_breath[kb] = state["cur"].realize().numpy().astype(np.float64).copy()
        return per_breath

    base_breaths = run_breaths(None)
    pert_breaths = {}
    for cond in CONDITIONS:
        pert_breaths[cond] = run_breaths(cur0_np + impulses[cond])
        log(f"[resonance] condition={cond} breaths replayed ({time.time() - t0:.0f}s so far)")

    # ---- measure: per condition x breath x right/wrong ----
    agg = {}   # (cond, kb, right) -> list of (relnorm, cos_own, cos_inj, cos_other)
    for cond in CONDITIONS:
        for kb in sorted(base_breaths):
            b = base_breaths[kb]; pr = pert_breaths[cond][kb]
            for (i, j, k, right) in gold_slots:
                resp = pr[i, j, CONTENT] - b[i, j, CONTENT]
                eps = eps_of[(i, j)]
                rn = float(np.linalg.norm(resp))
                rel = rn / (eps + 1e-12)
                c_own = cos(resp, own_unit[(i, j)]) if rn > 1e-12 else float("nan")
                c_inj = cos(resp, unit_dirs[(i, j, cond)]) if rn > 1e-12 else float("nan")
                c_oth = cos(resp, other_unit[(i, j)]) if rn > 1e-12 else float("nan")
                agg.setdefault((cond, kb, right), []).append((rel, c_own, c_inj, c_oth))

    lines = []
    L = lines.append
    L("=" * 86)
    L("THE RESONANCE READ -- PMS8_241, wild, kb=2 save point, 64 rows (2026-10-08)")
    L("=" * 86)
    L(f"gold slots: {len(gold_slots)} (right={n_right}, wrong={len(gold_slots) - n_right}); "
      f"nearest-other-kind pairing: " + ", ".join(f"{k}->{nearest_other[k]}" for k in KINDS))
    L("")
    for cond in CONDITIONS:
        L("-" * 86)
        L(f"CONDITION = {cond}")
        L("-" * 86)
        L("  breath  split    n   rel-norm  cos(own-kind)  cos(injected)  cos(other-kind)")
        for kb in sorted(base_breaths):
            for right, label in ((True, "right"), (False, "wrong")):
                rows = agg.get((cond, kb, right), [])
                if not rows:
                    continue
                arr = np.array(rows)
                L(f"    b{kb}   {label:5s}  {len(rows):4d}  {np.nanmean(arr[:,0]):8.4f}  "
                  f"{np.nanmean(arr[:,1]):13.4f}  {np.nanmean(arr[:,2]):13.4f}  {np.nanmean(arr[:,3]):14.4f}")
    lines_measure_done_at = time.time()

    # ---- decode agreement + custody-key proxy, per condition (row-level; reuses replay()) ----
    L(""); L("-" * 86); L("DECODE AGREEMENT (row-level; reuses replay(), not the per-breath loop above)"); L("-" * 86)
    out_base = H.replay(p, snap, SNAP_KB)
    need = ["pres", "res", "ftype", "op", "dig", "args", "query"] + \
           [k for k in ("sgn", "dup", "dargs") if k in out_base]
    def _row_dicts(out):
        npd = {k: out[k].realize().numpy() for k in need if k in out}
        return [{k: npd[k][i] for k in npd} for i in range(n)]
    rows_base = _row_dicts(out_base)
    gold_pg = {"presence": vg["presence"][sl], "ftype": vg["ftype"][sl], "digits": vg["digits"][sl]}
    def _key_ok(dec, row_i):
        facs_, _qv = dec
        ok_all = True
        for f in facs_:
            if f["ftype"] != "given":
                continue
            v = int(f["var"])
            if v >= gold_pg["presence"].shape[1] or gold_pg["presence"][row_i, v] <= 0 or int(gold_pg["ftype"][row_i, v]) != 1:
                ok_all = False; continue
            gval = int("".join(str(int(x)) for x in gold_pg["digits"][row_i, v]))
            if f["value"] != gval:
                ok_all = False
        return ok_all
    for cond in CONDITIONS:
        out_pert = H.replay(p, snap, SNAP_KB, overrides={"state": {"cur": Tensor((cur0_np + impulses[cond]).astype(np.float32), dtype=dtypes.float)}})
        rows_pert = _row_dicts(out_pert)
        changed = [i for i in range(n) if H.decode(rows_base[i]) != H.decode(rows_pert[i])]
        kb_ = sum(_key_ok(H.decode(rows_base[i]), i) for i in changed)
        ka_ = sum(_key_ok(H.decode(rows_pert[i]), i) for i in changed)
        L(f"  condition={cond:7s} decode changed on {len(changed)}/{n} rows {changed[:10]}{'...' if len(changed)>10 else ''}; "
          f"custody-key-ok before={kb_}/{len(changed)} after={ka_}/{len(changed)}")

    # ---- the six-line reading ----
    L(""); L("-" * 86); L("THE SIX-LINE READING"); L("-" * 86)

    def _mean_at(cond, kb, right=None):
        if right is None:
            rows = agg.get((cond, kb, True), []) + agg.get((cond, kb, False), [])
        else:
            rows = agg.get((cond, kb, right), [])
        return np.array(rows) if rows else np.zeros((0, 4))

    kbs = sorted(base_breaths)
    contr = {cond: [float(np.nanmean(_mean_at(cond, kb)[:, 0])) if len(_mean_at(cond, kb)) else float("nan") for kb in kbs] for cond in CONDITIONS}
    L(f"1. CONTRACTION: relative norm by breath (b{kbs[0]}..b{kbs[-1]}) -- "
      + "; ".join(f"{c}: " + " ".join(f"{x:.2f}" for x in contr[c]) for c in CONDITIONS))
    own_cos_by_breath = {cond: [float(np.nanmean(_mean_at(cond, kb)[:, 1])) if len(_mean_at(cond, kb)) else float("nan") for kb in kbs] for cond in CONDITIONS}
    L(f"2. ALIGNMENT TO OWN KIND: cos(response, own-kind mean) by breath -- "
      + "; ".join(f"{c}: " + " ".join(f"{x:.3f}" for x in own_cos_by_breath[c]) for c in CONDITIONS))
    inj_cos_by_breath = {cond: [float(np.nanmean(_mean_at(cond, kb)[:, 2])) if len(_mean_at(cond, kb)) else float("nan") for kb in kbs] for cond in CONDITIONS}
    L(f"3. PERSISTENCE OF THE KICK: cos(response, injected direction) by breath -- "
      + "; ".join(f"{c}: " + " ".join(f"{x:.3f}" for x in inj_cos_by_breath[c]) for c in CONDITIONS))
    kb_last = kbs[-1]
    right_own = float(np.nanmean(_mean_at("own", kb_last, True)[:, 1])) if len(_mean_at("own", kb_last, True)) else float("nan")
    wrong_own = float(np.nanmean(_mean_at("own", kb_last, False)[:, 1])) if len(_mean_at("own", kb_last, False)) else float("nan")
    L(f"4. RIGHT vs WRONG at the last breath (b{kb_last}, own-kind condition): "
      f"cos(response, own-kind mean) right={right_own:.3f} wrong={wrong_own:.3f} "
      f"(diff={right_own - wrong_own:+.3f}; positive = the predicted perceiver channel)")
    oth_cos_by_breath = {cond: [float(np.nanmean(_mean_at(cond, kb)[:, 3])) if len(_mean_at(cond, kb)) else float("nan") for kb in kbs] for cond in CONDITIONS}
    L(f"5. THE CROSS-KIND IMPULSE (condition=other): cos(response, nearest-other-kind mean) by breath -- "
      + " ".join(f"{x:.3f}" for x in oth_cos_by_breath["other"])
      + f" (not absorbed/pulled back toward own kind would show as this staying positive and cos-to-own staying low)")
    L("6. VERDICT: see the numbers above against the 2026-10-08 11:49 prediction (contracts; "
      "aligns more on right than wrong; cross-kind not absorbed) -- stated per-number, not re-asserted here.")

    txt = "\n".join(lines) + "\n"
    open(OUT_TXT, "w").write(txt)
    log(f"[resonance] wrote {OUT_TXT} ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
