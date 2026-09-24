"""THE CLOCK-BAND PROBE (2026-09-24, the clock-planes question): what do the polar
band's 64 clocked planes (128 of 512 dims: breath-hand 32 / parity 16 / pass 16,
.cache/polar_bands.json) actually carry in a trained body? Zero training. Two
modes:

  collect (GPU, under the lock): the family read on the wild holdout (the masked
    pass, the row's own live facts — the criticality meter's collect verbatim),
    the port census hook armed, the state ENTERING each loop breath kb=1..6
    banked per factor slot (n, 6, L_FAC, 512) with the slot's gold metadata:
    presence, ftype class, derivation depth (given 0; relation 1 + max over its
    argument variables' introducing factors), and the anchor sentence (a given's
    literal sentence; a relation's clause sentence; -1 = none).

  probe (CPU): by band — CLOCK (128 dims), CONTENT (384), each WHEEL, and five
    size-matched random 128-dim CONTENT subsets (the control the clock band must
    beat) —
    A. BREATH TIME: 6-way logistic regression (row-split 5-fold) breath <- band;
       for the clock band, the accuracy from the top-k planes by F-statistic
       (k = 1, 2, 4, 8, 16, 32, 64): how FEW dials name the breath.
    B. STORY TIME: ridge regression (row-split 5-fold) depth <- band and
       sentence <- band, per breath and pooled, reported as held-out R^2 with
       the slot index j (one-hot) as the baseline predictor: the number that
       matters is the INCREMENT over j alone (under the positional law j is
       story order already), and whether the clock band's increment beats the
       size-matched content control.

BARS (pinned before the read, in the chain header): see .cache/clock_reads_chain.sh.
usage:  CB_MODE=collect CKPT=... python3 scripts/clock_band_probe.py   (family env)
        CB_MODE=probe  CB_TAG=... python3 scripts/clock_band_probe.py
"""
import json
import os
import sys

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
import numpy as np

MODE = os.environ.get("CB_MODE", "collect")
CKPT = os.environ.get("CKPT", ".cache/sharp_PMS8_241.safetensors")
TAG = os.environ.get("CB_TAG", os.path.basename(CKPT).replace("sharp_", "").replace(".safetensors", ""))
STATES = f".cache/clock_band_states_{TAG}.npz"
OUT = os.environ.get("OUT", f".cache/clock_band_probe_{TAG}.txt")
BANDS = os.environ.get("POLAR_BANDS", ".cache/polar_bands.json")
FIXTURE = (".cache/wild_admitted_holdout.jsonl", "wildhold")


# ---------------------------------------------------------------- targets
def _row_targets(row):
    """per factor idx: (ftype class, depth, sentence). ftype class: 0 given, 1 relation, 2 other."""
    import stamp_arg_mentions as SAM
    import lexical_identity_census as LIC
    (text, factors, solution, bounds, sspans, intro_map, memo,
     cand_key, cand_sent) = LIC.build_candidate_tables(row)
    depth = {}

    def _d(i, stack=()):
        if i in depth:
            return depth[i]
        if i in stack:
            return 0
        fac = factors[i]
        if fac["ftype"] == "given":
            depth[i] = 0
            return 0
        best = 0
        for a in SAM.recursion_args(fac):
            k = intro_map.get(a)
            if k is None or k == i:
                continue
            best = max(best, 1 + _d(k, stack + (i,)))
        depth[i] = best
        return best

    out = []
    for i, fac in enumerate(factors):
        ft = 0 if fac["ftype"] == "given" else (1 if fac["ftype"] == "rel" else 2)
        d = _d(i)
        sent = -1
        if fac["ftype"] == "given":
            ss = cand_sent.get(i) or set()
        else:
            try:
                cl = SAM.clause_of(i, factors, text, bounds, sspans, solution, intro_map, memo)
                ss = SAM.sent_set_of(cl, bounds, sspans) if cl is not None else set()
            except Exception:
                ss = set()
        if ss:
            sent = int(min(ss))
        out.append((ft, d, sent))
    return out


# ---------------------------------------------------------------- collect
def collect():
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    os.environ["ALG_TEST"], os.environ["ALG_TEST_NAME"] = FIXTURE
    import phase1_algebra_head as H
    from phase1_algebra_head import (build_params, forward, load_alg,
                                     build_slot_masks, alt2_fact_buf, K_VARS, L_FAC)
    assert int(os.environ.get("ALG_POLAR", "0")) == 1 and int(os.environ.get("ALG_WIDE", "0")) == 1, \
        "run under the family env (ALG_POLAR=1 ALG_WIDE=1 ...)"
    vs, vst, vtk, vg, vse = load_alg("test")
    n = len(vs)
    K_B = int(os.environ.get("ALG_BREATH", "1"))
    N_LOOP = K_B - 1
    print(f"[clock-band] collect n={n} K_B={K_B} ckpt={CKPT} tag={TAG}", flush=True)
    p = build_params(0)
    sd = safe_load(CKPT)
    assert set(sd.keys()) == set(p.keys()), (sorted(set(sd) - set(p))[:4], sorted(set(p) - set(sd))[:4])
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

    states = np.zeros((n, N_LOOP, L_FAC, H.H_W), np.float16)
    meta = np.full((n, L_FAC, 4), -1, np.int16)      # presence, ftype class, depth, sentence
    rows = [json.loads(l) for l in open(FIXTURE[0])]
    assert len(rows) == n, (len(rows), n)
    for i in range(n):
        tg = _row_targets(rows[i])
        for j in range(min(L_FAC, len(tg))):
            meta[i, j] = (int(vg["presence"][i, j] > 0.5), tg[j][0], tg[j][1], tg[j][2])

    for s0 in range(0, n, 8):
        sl = np.arange(s0, min(s0 + 8, n))
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        ts = Tensor(np.ascontiguousarray(vst[sl_p]), dtype=dtypes.half)
        tk = Tensor(vtk[sl_p].astype(np.float32), dtype=dtypes.float)
        se = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)
        o0 = forward(p, ts, tk, se)
        onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
        mk = build_slot_masks(onp0, vse[sl_p].astype(np.int32))
        _ka = ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())
        _oa = {**onp0, **{k: o0[k].realize().numpy() for k in _ka}}
        _nv = np.array([vs[int(i)].get("n_vars", K_VARS) for i in sl_p])
        _ma = np.array([vs[int(i)].get("m", 0) for i in sl_p])
        fb = alt2_fact_buf(_oa, vse[sl_p].astype(np.int32), _nv, _ma)
        H._CENSUS = []
        o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float),
                    fact_buf=Tensor(fb, dtype=dtypes.float))
        o["fat"].realize()
        got = {kb: arr for (kb, tag, arr) in H._CENSUS if tag == "state"}
        H._CENSUS = None
        assert sorted(got) == list(range(1, K_B)), sorted(got)
        for kb in range(1, K_B):
            states[sl, kb - 1] = got[kb][:len(sl), :L_FAC].astype(np.float16)
        if s0 % 64 == 0:
            print(f"[clock-band] {s0 + len(sl)}/{n}", flush=True)
    np.savez_compressed(STATES, states=states, meta=meta)
    print(f"[clock-band] wrote {STATES} states {states.shape}", flush=True)


# ---------------------------------------------------------------- probe
def _bands():
    pb = json.load(open(BANDS))
    P = int(pb["n_planes"])
    wheel_of = np.full(P, -1)
    names = []
    for w in pb["wheels"]:
        names.append(w["name"])
        for pl in w["planes"]:
            wheel_of[int(pl)] = int(w["wheel"])
    dims = lambda planes: np.concatenate([[2 * q, 2 * q + 1] for q in planes]).astype(int)
    out = {"CLOCK": dims(np.flatnonzero(wheel_of >= 0)),
           "CONTENT": dims(np.flatnonzero(wheel_of < 0))}
    for wi, nm in enumerate(names):
        out[f"wheel:{nm}"] = dims(np.flatnonzero(wheel_of == wi))
    rng = np.random.RandomState(7)
    cp = np.flatnonzero(wheel_of < 0)
    for r in range(5):
        out[f"content-rand64-{r}"] = dims(np.sort(rng.choice(cp, 64, replace=False)))
    return out, wheel_of


def probe():
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import spearmanr
    d = np.load(STATES)
    S = d["states"].astype(np.float32)          # (n, N_LOOP, L_FAC, H)
    M = d["meta"]                               # (n, L_FAC, 4)
    n, NL, LF, HW = S.shape
    bands, wheel_of = _bands()
    lines = []

    def P(s=""):
        print(s, flush=True)
        lines.append(s)

    P(f"THE CLOCK-BAND PROBE — {TAG} on the wild holdout (n={n} rows, {NL} loop breaths, {LF} factor slots, H={HW})")
    P(f"bands: clock {len(bands['CLOCK'])} dims, content {len(bands['CONTENT'])} dims; "
      f"wheels {[ (k, len(v)) for k, v in bands.items() if k.startswith('wheel:')]}; five random 64-plane content controls")
    pres = M[:, :, 0] > 0
    ri, ji = np.nonzero(pres)
    P(f"present factor slots: {len(ri)}; givens {int((M[:, :, 1][pres] == 0).sum())}, relations {int((M[:, :, 1][pres] == 1).sum())}, other {int((M[:, :, 1][pres] == 2).sum())}")
    groups = ri
    gkf = GroupKFold(n_splits=5)

    # A. BREATH TIME ---------------------------------------------------
    P("\nA. BREATH TIME: breath <- band (6-way logistic, row-split 5-fold, standardized). chance 0.167")
    X_all = S[ri, :, ji, :]                     # (N, NL, H)
    N = len(ri)
    yb = np.tile(np.arange(NL), N)              # breath label
    Xf = X_all.reshape(N * NL, HW)
    gb = np.repeat(groups, NL)

    def cv_logit(cols):
        acc = []
        Xc = Xf[:, cols]
        for tr, te in gkf.split(Xc, yb, gb):
            sc = StandardScaler().fit(Xc[tr])
            clf = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(Xc[tr]), yb[tr])
            acc.append(float((clf.predict(sc.transform(Xc[te])) == yb[te]).mean()))
        return float(np.mean(acc))

    for name in ["CLOCK", "CONTENT"] + [k for k in bands if k.startswith("wheel:")] + [k for k in bands if k.startswith("content-rand")]:
        P(f"  {name:22s} dims={len(bands[name]):4d}  breath acc {cv_logit(bands[name]):.3f}")
    # how few dials: F-statistic ranking of the clock PLANES (a plane = 2 dims)
    clock_planes = np.flatnonzero(wheel_of >= 0)
    F = []
    for q in clock_planes:
        cols = [2 * q, 2 * q + 1]
        x = Xf[:, cols]
        mu = x.mean(0)
        between = sum(((x[yb == b].mean(0) - mu) ** 2).sum() * (yb == b).sum() for b in range(NL)) / (NL - 1)
        within = sum(((x[yb == b] - x[yb == b].mean(0)) ** 2).sum() for b in range(NL)) / (len(x) - NL)
        F.append(between / (within + 1e-9))
    order = clock_planes[np.argsort(F)[::-1]]
    P("  top clock planes by F-statistic: " + ", ".join(f"p{int(q)}({'bh' if wheel_of[q]==0 else ('par' if wheel_of[q]==1 else 'pass')}) F={F[list(clock_planes).index(q)]:.1f}" for q in order[:8]))
    for k in (1, 2, 4, 8, 16, 32, 64):
        cols = np.concatenate([[2 * q, 2 * q + 1] for q in order[:k]])
        P(f"  top-{k:2d} clock planes ({2*k:3d} dims): breath acc {cv_logit(cols):.3f}")
    # the same for content planes (the control: how many content planes name the breath)
    content_planes = np.flatnonzero(wheel_of < 0)
    Fc = []
    for q in content_planes:
        x = Xf[:, [2 * q, 2 * q + 1]]
        mu = x.mean(0)
        between = sum(((x[yb == b].mean(0) - mu) ** 2).sum() * (yb == b).sum() for b in range(NL)) / (NL - 1)
        within = sum(((x[yb == b] - x[yb == b].mean(0)) ** 2).sum() for b in range(NL)) / (len(x) - NL)
        Fc.append(between / (within + 1e-9))
    orderc = content_planes[np.argsort(Fc)[::-1]]
    for k in (1, 4, 16):
        cols = np.concatenate([[2 * q, 2 * q + 1] for q in orderc[:k]])
        P(f"  top-{k:2d} CONTENT planes ({2*k:3d} dims): breath acc {cv_logit(cols):.3f}  (max content F {max(Fc):.1f} vs max clock F {max(F):.1f})")

    # B. STORY TIME ----------------------------------------------------
    P("\nB. STORY TIME: depth / sentence <- band (ridge alpha=10, row-split 5-fold, standardized), held-out R^2;")
    P("   baseline = slot index j one-hot (the positional law's story order); the read is the INCREMENT over j.")
    depth = M[:, :, 2][pres].astype(np.float32)
    sent = M[:, :, 3][pres].astype(np.float32)
    J = np.eye(LF, dtype=np.float32)[ji]
    P(f"   depth: mean {depth.mean():.2f} max {int(depth.max())}; sentence known on {int((sent >= 0).sum())}/{N} slots (max {int(sent.max())})")

    def cv_ridge(Xb, y, keep):
        r2, rho = [], []
        Xb, y, g = Xb[keep], y[keep], groups[keep]
        for tr, te in gkf.split(Xb, y, g):
            sc = StandardScaler().fit(Xb[tr])
            m = Ridge(alpha=10.0).fit(sc.transform(Xb[tr]), y[tr])
            pr = m.predict(sc.transform(Xb[te]))
            ss = ((y[te] - y[te].mean()) ** 2).sum()
            r2.append(1.0 - ((y[te] - pr) ** 2).sum() / (ss + 1e-9))
            rho.append(spearmanr(pr, y[te]).correlation if len(set(y[te])) > 1 else 0.0)
        return float(np.mean(r2)), float(np.nanmean(rho))

    for tname, y, keep in (("depth", depth, np.ones(N, bool)), ("sentence", sent, sent >= 0)):
        base_r2, base_rho = cv_ridge(J, y, keep)
        P(f"\n   target {tname}: j-only baseline R^2 {base_r2:.3f} (rho {base_rho:.3f})")
        for kb in list(range(NL)) + ["pooled"]:
            if kb == "pooled":
                Xk = X_all.mean(1)
            else:
                Xk = X_all[:, kb]
            cells = []
            for name in ("CLOCK", "CONTENT", "wheel:breath_hand", "wheel:parity", "wheel:pass_wheel"):
                r2, rho = cv_ridge(np.concatenate([J, Xk[:, bands[name]]], 1), y, keep)
                cells.append(f"{name.replace('wheel:', '')} +{r2 - base_r2:+.3f}")
            rc = [cv_ridge(np.concatenate([J, Xk[:, bands[f'content-rand64-{r}']]], 1), y, keep)[0] - base_r2 for r in range(5)]
            cells.append(f"content-rand64 +{np.mean(rc):+.3f}±{np.std(rc):.3f}")
            P(f"     breath {str(kb):6s}: " + " | ".join(cells))
    open(OUT, "w").write("\n".join(lines) + "\n")
    print(f"[clock-band] wrote {OUT}")


if __name__ == "__main__":
    collect() if MODE == "collect" else probe()
