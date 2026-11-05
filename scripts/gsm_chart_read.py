"""CHART THE LOCAL CHARTS READ (ledger 2026-11-04, word given; CORE_THEORIES: "connecting local charts which are centroids in the Tree").

A zero-training read on banked BT weights (BR.LOAD). The BT runs its BT6-gate commit loop as in BEAM-1 (rho0 = the arm's fitted value);
at each slot's commit cycle the silhouette z (DB-d) and the type logits are recorded. Gold real steps only.
  the charts   per gold type t (n_fit >= 50 on TRAIN, all of it by default): origin = the BT's own atom C[t]; local axes = the top-R principal
               directions of (z - C[t]); sigma_t = the in-fit RMS off-chart residual (the part of z - C[t] outside the R-plane)
  CH-1  charts are low-dimensional: median over charts of the variance fraction in the top R (isotropic would be R/DB)
  CH-2  the junctions: among wrong-typed steps, the share whose gold type is the commit distribution's rank 2 (the overlap of two charts)
  CH-3  off the atlas: median off-chart distance to the GOLD chart in units of sigma_t, TEST / CFNAME (names changed, nothing else)
  CH-4  the chart classifier: type by the best PPCA log-likelihood over charts (in-plane Mahalanobis + off-plane isotropic sigma_t^2,
        + log prior) vs the BT's committed type
-> .cache/gsmb_chart_{TAG}.json
"""
import json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import gsm_breath_reader as BR          # FIRST (the 11-04 import rule): it sets RD_ARM=TICK, which gsm_trunk_reader reads at import
import gsm_beam_read as GB

assert BR.TR.ARM == "TICK" and BR.TR.TICK and BR.TR.NOTE, f"trunk reader arm {BR.TR.ARM}"
assert BR.CYCLOCK and BR.T6OFF, "the T6 clock must be on (BT_CYCLOCK=1 BT_T6OFF=1)"
assert not BR.FACT, "CHART reads the joint atoms C (a FACT arm's type logits are not distances to C)"
R_ = int(os.environ.get("CH_R", "8")); NFIT = int(os.environ.get("CH_FIT", "0")) or None; NMIN = 50   # NFIT default: all of TRAIN
SM = int(os.environ.get("CH_SMOKE", "0")) or None
rho = float(json.load(open(f".cache/gsmb_fbeam_{GB.TAG}_nt2.json"))["rho0"])
STOP = GB.STOP

_last = {}
_breath = GB.net.breath


def breath(*a, **k):                                     # stash this cycle's silhouette + logits for the commit hook
    out = _breath(*a, **k); _last["z"] = out[3].numpy(); _last["logit"] = out[0].log_softmax(-1).numpy(); return out


GB.net.breath = breath
_rec = {}
_slot_cands = GB.slot_cands


def slot_cands(b, k, dec, tv, ti, alt):                  # called once per slot, at its commit cycle
    _rec[(b, k)] = (_last["z"][b, k].copy(), _last["logit"][b, k].copy())
    return _slot_cands(b, k, dec, tv, ti, alt)


GB.slot_cands = slot_cands


def collect(its):
    """-> z (N, DB), logp (N, NT), gold type (N,) over gold real steps that committed"""
    Z, LP, G = [], [], []
    for s0 in range(0, len(its), 128):
        part = its[s0:s0 + 128]; _rec.clear()
        t, E, xq, L, progs, cands = GB.gate_batch(part, rho, s0)
        _, gt = BR.gold(E)
        for b, e in enumerate(E):
            for k in range(min(e["m"], GB.K)):
                if gt[b, k] == STOP or (b, k) not in _rec: continue
                z, lp = _rec[(b, k)]; Z.append(z); LP.append(lp); G.append(int(gt[b, k]))
    return np.array(Z), np.array(LP), np.array(G)


C = GB.net.C.detach().numpy()
Zf, LPf, Gf = collect(GB.Dt["TRAIN"][:SM or NFIT])
NF = len(GB.Dt["TRAIN"][:SM or NFIT])
charts = {}
for t in range(STOP):
    X = Zf[Gf == t] - C[t]
    if len(X) < NMIN: continue
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    P = Vt[:R_]; off = X - X @ P.T @ P
    lam = (S[:R_] ** 2) / len(X)
    charts[t] = dict(n=len(X), P=P, lam=lam, sig2=float((off ** 2).sum(1).mean() / (X.shape[1] - R_)),
                     frac=float((S[:R_] ** 2).sum() / (S ** 2).sum()), prior=math.log(len(X) / len(Zf)))
DB = Zf.shape[1]
print(f"[ch] {GB.TAG} fit TRAIN items {NF} steps {len(Zf)} | charts {len(charts)} (n >= {NMIN}) | R {R_} of {DB} (isotropic {R_ / DB:.3f})", flush=True)


def offd(z, t):
    """off-chart distance of z to chart t, in units of the chart's per-dimension off-plane sigma"""
    x = z - C[t]; ch = charts[t]; o = x - x @ ch["P"].T @ ch["P"]
    return float(np.sqrt((o ** 2).sum() / (DB - R_) / ch["sig2"]))


def ppca(z):
    """the chart classifier: argmax_t log N(z; C[t], P^T diag(lam) P + sig2 (I - P^T P)) + log prior"""
    best, bt = -1e30, -1
    for t, ch in charts.items():
        x = z - C[t]; y = ch["P"] @ x; o = x - ch["P"].T @ y
        ll = -0.5 * ((y ** 2 / ch["lam"]).sum() + np.log(ch["lam"]).sum() + (o ** 2).sum() / ch["sig2"] + (DB - R_) * math.log(ch["sig2"])) + ch["prior"]
        if ll > best: best, bt = ll, t
    return bt


out = dict(tag=GB.TAG, rho0=rho, R=R_, fit_steps=len(Zf), charts=len(charts),
           CH1_median_frac=float(np.median([c["frac"] for c in charts.values()])), iso=R_ / DB)
print(f"[ch] CH-1 median variance fraction in the top {R_}: {out['CH1_median_frac']:.3f} (isotropic {R_ / DB:.3f}; "
      f"range {min(c['frac'] for c in charts.values()):.3f}..{max(c['frac'] for c in charts.values()):.3f})", flush=True)
for split in ("CFNAME", "TEST"):
    Z, LP, G = collect(GB.Dt[split][:SM])
    pred = LP.argmax(1); wrong = pred != G
    rank = (LP > LP[np.arange(len(G)), G][:, None]).sum(1) + 1                      # the gold type's rank in the commit distribution
    has = np.array([g in charts for g in G])
    q = np.array([offd(z, g) if g in charts else np.nan for z, g in zip(Z, G)])
    cp = np.array([ppca(z) for z in Z])
    o = out[split] = dict(n=len(G), type_acc=float((~wrong).mean()), wrong=int(wrong.sum()),
                          rank2_of_wrong=float((rank[wrong] == 2).mean()), rank_le4_of_wrong=float((rank[wrong] <= 4).mean()),
                          gold_charted=float(has.mean()), q_median=float(np.nanmedian(q)),
                          q_median_right=float(np.nanmedian(q[~wrong])), q_median_wrong=float(np.nanmedian(q[wrong])),
                          chart_acc=float((cp == G).mean()))
    print(f"[ch] {GB.TAG} {split} steps {o['n']} | BT type {o['type_acc']:.4f} | wrong {o['wrong']}: gold at rank 2 {o['rank2_of_wrong']:.3f}, "
          f"rank <= 4 {o['rank_le4_of_wrong']:.3f} | off-chart q median {o['q_median']:.2f} (right {o['q_median_right']:.2f}, wrong "
          f"{o['q_median_wrong']:.2f}) | chart classifier {o['chart_acc']:.4f} | gold charted {o['gold_charted']:.3f}", flush=True)
out["CH3_ratio"] = out["TEST"]["q_median"] / out["CFNAME"]["q_median"]
print(f"[ch] {GB.TAG} CH-3 off-chart TEST / CFNAME {out['CH3_ratio']:.2f}", flush=True)
json.dump(out, open(f".cache/gsmb_chart_{GB.TAG}{'_smoke' if SM else ''}.json", "w"), indent=1, default=float)
