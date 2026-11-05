"""CHART-IN THE CHARTS IN THE LOOP (ledger 2026-11-04, word given): the chart classifier becomes the BT's type decision. Zero training.

The charts are fit exactly as in scripts/gsm_chart_read.py: the BT's own commit loop on all of TRAIN, gold real steps; per type t with
n >= 50: origin C[t], top-R principal axes, off-plane sigma; PPCA log-likelihood + log prior. Under CHIN=1 every breath's type
distribution is replaced:
  log p'(STOP) = log p_BT(STOP)                                         (no chart for stop: the BT keeps that decision)
  log p'(t)    = log(1 - p_BT(STOP)) + log softmax_charts(LL)(t)        (charted step types)
  log p'(t)    = -1e4                                                    (uncharted types: cannot be chosen, the CH-4 handicap)
and the pointers are recomputed under the new argmax type (net.ptrs, no zoom fallback; stated: the pointer head's radius input stays the
BT's own p(type)). The commit gate, the notebook write-back (atom x radius of the CHART's type), BEAM-1's top-2 types x 2 x 2 and the
solver all follow. rho0 and lam are fit on TRAIN[:600] per arm (BEAM-1's procedure). CHIN=0 = the control through the same code.
Reads: TEST / CFNAME type at commit (gold real steps), greedy ans, beam ans, oracle@8 -> .cache/gsmb_chartin_{TAG}_c{CHIN}.json
"""
import json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import gsm_breath_reader as BR          # FIRST (the 11-04 import rule): it sets RD_ARM=TICK, which gsm_trunk_reader reads at import
import gsm_beam_read as GB

assert BR.TR.ARM == "TICK" and BR.TR.TICK and BR.TR.NOTE, f"trunk reader arm {BR.TR.ARM}"
assert BR.CYCLOCK and BR.T6OFF, "the T6 clock must be on (BT_CYCLOCK=1 BT_T6OFF=1)"
assert not BR.FACT
CHIN = os.environ.get("CHIN", "1") == "1"; R_ = 8; NMIN = 50
SM = int(os.environ.get("CH_SMOKE", "0")) or None
STOP = GB.STOP
C = GB.net.C.detach().numpy(); DB = C.shape[1]
rho_bt = float(json.load(open(f".cache/gsmb_fbeam_{GB.TAG}_nt2.json"))["rho0"])

# ---- the charts (the BT's own loop, unchanged, at its fitted rho0)
_last, _rec = {}, {}
_breath = GB.net.breath
_slot_cands = GB.slot_cands


def rec_breath(*a, **k):
    out = _breath(*a, **k); _last["z"] = out[3].numpy(); return out


def rec_cands(b, k, dec, tv, ti, alt):
    _rec[(b, k)] = _last["z"][b, k].copy(); return _slot_cands(b, k, dec, tv, ti, alt)


def committed(its, rho):
    """-> z, gold type, committed type over gold real steps"""
    Z, G, P_ = [], [], []
    for s0 in range(0, len(its), 128):
        part = its[s0:s0 + 128]; _rec.clear(); _last.clear()
        t, E, xq, L, progs, cands = GB.gate_batch(part, rho, s0)
        _, gt = BR.gold(E)
        for b, e in enumerate(E):
            for k in range(min(e["m"], GB.K)):
                if gt[b, k] == STOP or (b, k) not in _rec: continue
                Z.append(_rec[(b, k)]); G.append(int(gt[b, k])); P_.append(_rec_type[(b, k)])
    return np.array(Z), np.array(G), np.array(P_)


_rec_type = {}


def rec_cands_t(b, k, dec, tv, ti, alt):
    _rec_type[(b, k)] = int(ti[b, k, 0]); return rec_cands(b, k, dec, tv, ti, alt)


GB.net.breath, GB.slot_cands = rec_breath, rec_cands_t
Zf, Gf, _ = committed(GB.Dt["TRAIN"][:SM], rho_bt)
CT = [t for t in range(STOP) if (Gf == t).sum() >= NMIN]
Pm, Lm, S2, PR = [], [], [], []
for t in CT:
    X = Zf[Gf == t] - C[t]; U, S, Vt = np.linalg.svd(X, full_matrices=False); P = Vt[:R_]; off = X - X @ P.T @ P
    Pm.append(P); Lm.append(S[:R_] ** 2 / len(X)); S2.append((off ** 2).sum(1).mean() / (DB - R_)); PR.append(math.log(len(X) / len(Zf)))
Pm, Lm, S2, PR = (torch.as_tensor(np.array(a), dtype=torch.float32) for a in (Pm, Lm, S2, PR))
CTt = torch.as_tensor(CT); Ct = torch.as_tensor(C[CT], dtype=torch.float32)
print(f"[ci] {GB.TAG} charts {len(CT)} fit on {len(Zf)} TRAIN steps (BT loop, rho0 {rho_bt})", flush=True)


def chart_ll(z):
    """z (B, K, DB) -> PPCA log-likelihood + log prior per chart (B, K, NC)"""
    x = z[:, :, None] - Ct                                           # (B, K, NC, DB)
    y = torch.einsum("bkcd,crd->bkcr", x, Pm)
    o2 = (x ** 2).sum(-1) - (y ** 2).sum(-1)
    return -0.5 * ((y ** 2 / Lm).sum(-1) + Lm.log().sum(-1) + o2 / S2 + (DB - R_) * S2.log()) + PR


def chart_breath(*a, **k):
    logit, pa, pb, z, p = _breath(*a, **k)
    if not CHIN: _last["z"] = z.numpy(); return logit, pa, pb, z, p
    ps = p[..., STOP].clamp(1e-6, 1 - 1e-6)
    lc = chart_ll(z).log_softmax(-1)
    nl = torch.full_like(logit, -1e4)
    nl[..., STOP] = ps.log()
    nl[..., CTt] = (1 - ps).log()[..., None] + lc
    np_ = nl.softmax(-1)
    pa, pb = GB.net.ptrs(nl.argmax(-1))
    _last["z"] = z.numpy()
    return nl, pa, pb, z, np_


GB.net.breath = chart_breath
_rec.clear(); _rec_type.clear()

# ---- the read: BEAM-1's fit, then the splits
fit = {rho: float(np.mean([r["greedy"] for r in GB.collect(GB.Dt["TRAIN"][:SM or 600], rho)])) for rho in (GB.GRID[-3:-1] if SM else GB.GRID)}
rho = max(fit, key=lambda r: fit[r])
RT = GB.collect(GB.Dt["TRAIN"][:SM or 600], rho)
lfit = {(li, ln): float(GB.pick(RT, li, ln).mean()) for li in GB.LAMS for ln in GB.LAMS}
lam = max(lfit, key=lambda x: (lfit[x], -x[0] - x[1]))
out = dict(tag=GB.TAG, chin=CHIN, charts=len(CT), rho0=rho, lam=lam, lam_fit_best=lfit[lam])
print(f"[ci] {GB.TAG} CHIN {int(CHIN)} rho0 {rho} lam {lam} TRAIN ans {lfit[lam]:.4f}", flush=True)
for split in ("CFNAME", "TEST"):
    its = GB.Dt[split][:SM]
    R = GB.collect(its, rho)
    g = np.array([r["greedy"] for r in R], float); bm = GB.pick(R, *lam).astype(float)
    orc = np.array([any(x[2] for x in r["beam"]) or r["greedy"] for r in R], float)
    _, G, P_ = committed(its, rho)
    o = out[split] = dict(n=len(R), steps=len(G), type=float((G == P_).mean()), greedy=g.mean(), beam=bm.mean(), oracle=orc.mean(),
                          per_item=bm.tolist())
    print(f"[ci] {GB.TAG} CHIN {int(CHIN)} {split} n {o['n']} | type at commit {o['type']:.4f} ({o['steps']} steps) | ans greedy {o['greedy']:.4f} "
          f"| beam+solver {o['beam']:.4f} | oracle@8 {o['oracle']:.4f}", flush=True)
json.dump(out, open(f".cache/gsmb_chartin_{GB.TAG}_c{int(CHIN)}{'_smoke' if SM else ''}.json", "w"), default=float)
