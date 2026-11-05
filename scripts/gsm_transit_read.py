"""TRANSIT IS PHASE BINDING THE GLUE BETWEEN CHARTS? (ledger 2026-11-04 20:02, word given). Zero training.

The atoms C (step types (op, kind a, kind b); 32 complex planes, (re, im) = (0::2, 1::2) as in rot) and the charts' local axes (fit as in
scripts/gsm_chart_read.py). Along each factor axis, the phase shift between two values = the radius-weighted circular mean over shared contexts
of the per-plane angle difference: theta = angle(sum_k z_v,k * conj(z_u,k)).
  TR-1   LOO: rot(source, theta from the other contexts) snapped to the nearest valid atom == the target?   (pooled hit@1)
  TR-1b  phase hit@1 - offset hit@1 (source + the mean difference); beside: orthogonal Procrustes, the target-value mean atom
  TR-2   the cocycle: theta12 + theta23 - theta13 on DISJOINT context sets, radius-weighted mean |wrapped residual| (random: pi/2)
  TR-3   charts' axes: overlap(P_t, rot(P_s, theta)) - overlap(P_t, P_s), overlap = ||A B^T||^2 / R
  TR-4   (sizing) residual ||C - C_hat||^2 / ||C - mean atom||^2 per atom vs its Welford count; per-plane resultant lengths
TR_SYNTH=planted | random: the instrument check on synthetic atoms (no model loaded). Else BR.LOAD (BT4M8) -> .cache/gsmb_transit_{TAG}.json
"""
import itertools, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

SYN = os.environ.get("TR_SYNTH", "")
NV = (4, 3, 3); NT = 36; R_ = 8; NMIN = 50; DB = 64
tid = lambda o, a, b: o * 9 + a * 3 + b
cx = lambda X: X[..., 0::2] + 1j * X[..., 1::2]


def re64(Z):
    out = np.empty(Z.shape[:-1] + (2 * Z.shape[-1],)); out[..., 0::2] = Z.real; out[..., 1::2] = Z.imag; return out


def rot(X, th):
    return re64(cx(X) * np.exp(1j * th))


def atoms_of(axis, u, ctx):
    """the type id with value u on `axis` and the other two factor values = ctx"""
    f = list(ctx); f.insert(axis, u); return tid(*f)


def contexts(axis):
    return list(itertools.product(*[range(NV[i]) for i in range(3) if i != axis]))


def theta(C, pairs):
    """pairs [(src id, tgt id)] -> per-plane phase shift and resultant length"""
    s = sum(cx(C[t]) * np.conj(cx(C[u])) for u, t in pairs)
    w = sum(np.abs(cx(C[t])) * np.abs(cx(C[u])) for u, t in pairs)
    return np.angle(s), np.abs(s) / np.maximum(w, 1e-12)


def run(C, valid, charts, cnt):
    V = np.flatnonzero(valid)
    snap = lambda x: int(V[np.argmin(((C[V] - x) ** 2).sum(1))])
    hits = {"phase": [], "offset": [], "procrustes": [], "mean": []}; res = {t: [] for t in V}; rbar = []; tr3 = []
    for axis in range(3):
        for u, v in itertools.permutations(range(NV[axis]), 2):
            ks = [k for k in contexts(axis) if valid[atoms_of(axis, u, k)] and valid[atoms_of(axis, v, k)]]
            if len(ks) < 3: continue
            P = [(atoms_of(axis, u, k), atoms_of(axis, v, k)) for k in ks]
            rbar.append(theta(C, P)[1])
            for i, (s, t) in enumerate(P):
                oth = P[:i] + P[i + 1:]
                th, _ = theta(C, oth)
                pr = {"phase": rot(C[s], th),
                      "offset": C[s] + np.mean([C[b] - C[a] for a, b in oth], 0),
                      "mean": np.mean([C[b] for _, b in oth], 0)}
                A = np.array([C[a] for a, _ in oth]); B = np.array([C[b] for _, b in oth])
                U_, _, Vt_ = np.linalg.svd(B.T @ A); pr["procrustes"] = (U_ @ Vt_) @ C[s]
                for m, x in pr.items(): hits[m].append(snap(x) == t)
                res[t].append(((C[t] - pr["phase"]) ** 2).sum() / ((C[t] - C[V].mean(0)) ** 2).sum())
                if axis == 0 and s in charts and t in charts:
                    Ps, Pt = charts[s], charts[t]
                    ov = lambda X: float(((Pt @ X.T) ** 2).sum() / R_)
                    tr3.append(ov(rot(Ps, th)) - ov(Ps))
    co, cw = [], []
    for o1, o2, o3 in itertools.permutations(range(4), 3):
        ks = [k for k in contexts(0) if all(valid[tid(o, *k)] for o in (o1, o2, o3))]
        if len(ks) < 3: continue
        g = [ks[i::3] for i in range(3)]
        t12 = theta(C, [(tid(o1, *k), tid(o2, *k)) for k in g[0]])[0]
        t23 = theta(C, [(tid(o2, *k), tid(o3, *k)) for k in g[1]])[0]
        t13 = theta(C, [(tid(o1, *k), tid(o3, *k)) for k in g[2]])[0]
        r = np.abs(np.angle(np.exp(1j * (t12 + t23 - t13))))
        w = sum(np.abs(cx(C[tid(o1, *k)])) * np.abs(cx(C[tid(o3, *k)])) for k in ks)
        co.append((r * w).sum() / w.sum()); cw.append(1)
    out = {f"hit_{m}": float(np.mean(h)) for m, h in hits.items()}
    out.update(n_loo=len(hits["phase"]), chance=1 / len(V), valid=int(len(V)), tr2=float(np.mean(co)) if co else float("nan"),
               n_tr2=len(co), tr3=float(np.mean(tr3)) if tr3 else float("nan"), n_tr3=len(tr3),
               rbar_median=float(np.median(np.concatenate(rbar))) if rbar else float("nan"))
    r4 = {int(t): float(np.mean(r)) for t, r in res.items() if r}
    out["tr4"] = r4
    if cnt is not None and r4:
        ts = sorted(r4); c_ = np.array([cnt[t] for t in ts]); r_ = np.array([r4[t] for t in ts])
        rk = lambda a: np.argsort(np.argsort(a))
        out["tr4_spearman_count"] = float(np.corrcoef(rk(c_), rk(r_))[0, 1])
        med = np.median(c_); out["tr4_resid_frequent"] = float(np.median(r_[c_ >= med])); out["tr4_resid_rare"] = float(np.median(r_[c_ < med]))
    return out


def report(tag, o):
    print(f"[tr] {tag} valid atoms {o['valid']} (chance {o['chance']:.3f}) | LOO n {o['n_loo']} | hit@1 phase {o['hit_phase']:.3f} offset "
          f"{o['hit_offset']:.3f} procrustes {o['hit_procrustes']:.3f} mean {o['hit_mean']:.3f}", flush=True)
    print(f"[tr] {tag} TR-2 cocycle residual {o['tr2']:.3f} rad (n {o['n_tr2']}; pi/4 {math.pi / 4:.3f}, random {math.pi / 2:.3f}) | TR-3 axes "
          f"overlap gain {o['tr3']:+.3f} (n {o['n_tr3']}) | resultant median {o['rbar_median']:.3f}", flush=True)
    if "tr4_spearman_count" in o:
        print(f"[tr] {tag} TR-4 residual median frequent {o['tr4_resid_frequent']:.3f} rare {o['tr4_resid_rare']:.3f} | Spearman(count, residual) "
              f"{o['tr4_spearman_count']:+.3f}", flush=True)


if SYN:
    rng = np.random.default_rng(0)
    if SYN == "planted":
        Go, Ga, Gb = (rng.uniform(0, 2 * np.pi, (n, 32)) for n in NV)
        base = rng.normal(size=32) + 1j * rng.normal(size=32)
        P0 = np.linalg.qr(rng.normal(size=(DB, R_)))[0].T
        C = np.zeros((NT, DB)); charts = {}
        for o, a, b in itertools.product(*map(range, NV)):
            ph = Go[o] + Ga[a] + Gb[b]
            C[tid(o, a, b)] = re64(base * np.exp(1j * ph)) + 0.05 * rng.normal(size=DB)
            charts[tid(o, a, b)] = rot(P0, ph)
    else:
        C = rng.normal(size=(NT, DB)); charts = {t: np.linalg.qr(rng.normal(size=(DB, R_)))[0].T for t in range(NT)}
    report(f"SYNTH-{SYN}", run(C, np.ones(NT, bool), charts, rng.integers(10, 1000, NT)))
    sys.exit(0)

import torch
import gsm_breath_reader as BR          # FIRST (the 11-04 import rule): it sets RD_ARM=TICK, which gsm_trunk_reader reads at import
import gsm_beam_read as GB

assert BR.TR.ARM == "TICK" and BR.TR.TICK and BR.TR.NOTE and BR.CYCLOCK and BR.T6OFF and not BR.FACT
assert BR.TYPES[tid(2, 1, 0)] == ("*", "s", "t") and len(BR.TYPES) == NT, "type layout"
C = GB.net.C.detach().numpy()[:NT].astype(np.float64); cnt = GB.net.Cn.detach().numpy()[:NT]; valid = cnt >= NMIN
rho = float(json.load(open(f".cache/gsmb_fbeam_{GB.TAG}_nt2.json"))["rho0"])
_last, _rec = {}, {}
_breath, _sc = GB.net.breath, GB.slot_cands
GB.net.breath = lambda *a, **k: (lambda o: (_last.__setitem__("z", o[3].numpy()), o)[1])(_breath(*a, **k))
GB.slot_cands = lambda b, k, dec, tv, ti, alt: (_rec.__setitem__((b, k), _last["z"][b, k].copy()), _sc(b, k, dec, tv, ti, alt))[1]
Z, G = [], []
its = GB.Dt["TRAIN"]
for s0 in range(0, len(its), 128):
    part = its[s0:s0 + 128]; _rec.clear()
    t, E, xq, L, progs, cands = GB.gate_batch(part, rho, s0)
    _, gt = BR.gold(E)
    for b, e in enumerate(E):
        for k in range(min(e["m"], GB.K)):
            if gt[b, k] != GB.STOP and (b, k) in _rec: Z.append(_rec[(b, k)]); G.append(int(gt[b, k]))
Z, G = np.array(Z, np.float64), np.array(G)
charts = {}
for t in range(NT):
    X = Z[G == t] - C[t]
    if len(X) >= NMIN: charts[t] = np.linalg.svd(X, full_matrices=False)[2][:R_]
print(f"[tr] {GB.TAG} atoms valid (Cn >= {NMIN}) {int(valid.sum())} / {NT} | charts {len(charts)}", flush=True)
o = run(C, valid, charts, cnt); o.update(tag=GB.TAG, cn=cnt.tolist()); report(GB.TAG, o)
json.dump(o, open(f".cache/gsmb_transit_{GB.TAG}.json", "w"), indent=1, default=float)
