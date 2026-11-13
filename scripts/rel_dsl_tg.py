"""rel_dsl_tg.py -- THE REL-DSL DEPTH-SPLIT TRAINING ON TINYGRAD (arms NOMLP and NOTEBOOK) + THE UNITARY BREATH (RD_UNIT=1).
A port of rel_dsl_breath.make_net (NOMLP) and rel_dsl_organs.make_net (NOTEBOOK) so the depth split trains on the GPU.

  - the torch modules stay the source of truth for data, encoding, executor, scoring (imported, never re-implemented); env is set
    BEFORE the import and the configuration is ASSERTED after it (rel_dsl_organs reads RB.ARM at import)
  - the parameter dict uses the torch state_dict names, so a torch init (same seed) is copied in and the weights are copied back
    into the torch net for RB.score (one scorer for every engine); nn.MultiheadAttention is ported by hand (in_proj, 1/sqrt(DH),
    out_proj, additive -1e4 key mask); gathers are one-hot matmuls, masks are where()
  - NOTEBOOK: the notebook state (known, kt) and the fire rule (argmax of each breath's logits) live IN-GRAPH as non-differentiable
    int/float tensors (argmax / comparisons carry no grad), so one TinyJit per shape bucket covers the whole step
  - batches are padded to multiples of TG_BUCKET (8) in (vars, factors): pads are masked keys / masked pointers / dv-masked rows
    (same math as the trimmed torch batch), one JIT per bucket; rng draws replicate the torch loop exactly (same batches)
  - params the torch graph never touches (grad None -> AdamW skips them) are found by one throwaway torch step; left out here too
  - RD_UNIT=1 (THE UNITARY BREATH): every additive write in the breath loop (T6 stamp, notebook known/kproj rows, attention
    residual, snap) becomes the sphere exponential map per token (norm preserved exactly); rows not written stay as they are
Modes (TG_MODE): parity | smoke | unitcheck (DEV=CPU gates) | train (the full run; scored every 2000 steps by RB.score on copied-back
weights; writes .cache/rel_dsl_{TAG}.json with engine/unit/norm_census). Env: RD_ARM (NOMLP|NOTEBOOK) RD_UNIT RD_SEED RD_COV RD_SPLIT=depth2
RD_NB (10) RD_STEPS RD_TAG TG_MODE TG_BUCKET."""
import json, math, os, sys, time

os.environ.setdefault("RD_SPLIT", "depth2"); os.environ.setdefault("RD_NB", "10"); os.environ.setdefault("RD_COV", "0.25")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
import numpy as np
import rel_dsl_breath as RB          # configures from env (RD_ARM etc.) -- imported FIRST
import rel_dsl_organs as RO          # reads RB.ARM
from tinygrad import Tensor, TinyJit, dtypes

ARM = os.environ.get("RD_ARM", "NOMLP"); UNIT = int(os.environ.get("RD_UNIT", "0")); COV = float(os.environ["RD_COV"])
DAMP = int(os.environ.get("RD_DAMP", "0"))   # (DAMP, ledger 11-13) a learned pole per complex plane on the carried state, each breath
assert ARM in ("NOMLP", "NOTEBOOK"), f"arm {ARM} not ported"
assert RB.ARM == ARM and RO.ARM == ARM and RO.SPLIT == "depth2" and RB.NB == int(os.environ["RD_NB"]) and RB.COV == COV, "config mismatch"
D, NH, NB, NID, PB = RB.D, RB.NH, RB.NB, RB.NID, RO.PB
DH = D // NH; NOTE = ARM == "NOTEBOOK"; BUCKET = int(os.environ.get("TG_BUCKET", "8"))
TAG = os.environ.get("RD_TAG", f"TG{ARM}{'U' if UNIT else ''}{'D' if DAMP else ''}_depth2c{COV:g}_s{RB.SEED}")
assert not (DAMP and UNIT), "DAMP and UNIT are separate forms"


# ---- net ----------------------------------------------------------------------------------------------------------------------
def from_torch(sd):
    return {k: Tensor(v.detach().cpu().numpy().astype(np.float32)).contiguous().realize() for k, v in sd.items()}


def to_torch(P, tn):
    import torch
    tn.load_state_dict({k: torch.as_tensor(P[k].numpy().copy()) for k in tn.state_dict()})


def lin(P, x, n, bias=True):
    y = x @ P[n + ".weight"].T
    return y + P[n + ".bias"] if bias else y


def lnorm(P, x, n):
    return x.layernorm(axis=-1, eps=1e-5) * P[n + ".weight"] + P[n + ".bias"]


def emb(P, idx, n, size):
    return idx.one_hot(size).cast(dtypes.float) @ P[n + ".weight"]


def expmap(x, d):
    """sphere exp map per token: x -> cos(th) x + sin(th)/th dp, dp = d projected tangent to x, th = |dp|/|x| (norm preserved)"""
    xn = (x * x).sum(-1, keepdim=True).sqrt().maximum(1e-6); xh = x / xn
    dp = d - (d * xh).sum(-1, keepdim=True) * xh
    th = ((dp * dp).sum(-1, keepdim=True) / (xn * xn) + 1e-12).sqrt()
    return x - 2 * (th * 0.5).sin() ** 2 * x + (th.sin() / th) * dp          # cos(th) x as x + (cos-1) x: exact at d = 0 (a float32 cos(1e-6) is 1 ulp off)


def damp(P, x):
    """(DAMP) the carried state times a learned pole per complex plane (adjacent dims paired): p_k = sigmoid(a_k) e^{i phi_k}"""
    r = P["damp_a"].sigmoid(); c, sn = r * P["damp_phi"].cos(), r * P["damp_phi"].sin(); re, im = x[..., 0::2], x[..., 1::2]
    return Tensor.stack(re * c - im * sn, re * sn + im * c, dim=-1).reshape(*x.shape)


def add_damp(P):
    P["damp_a"] = Tensor.full((D // 2,), 3.0).contiguous().realize(); P["damp_phi"] = Tensor.zeros(D // 2).contiguous().realize()
    return P


def write(x, d, unit):
    return expmap(x, d) if unit else x + d


def mha(P, y, mask):
    """nn.MultiheadAttention (self, batch_first) by hand; mask additive, broadcastable to (B, NH, L, L)"""
    B, L, _ = y.shape
    qkv = y @ P["blocks.0.att.in_proj_weight"].T + P["blocks.0.att.in_proj_bias"]
    q, k, v = (qkv[..., i * D:(i + 1) * D].reshape(B, L, NH, DH).transpose(1, 2) for i in range(3))
    sc = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(DH)) + mask
    return lin(P, (sc.softmax(-1) @ v).transpose(1, 2).reshape(B, L, D), "blocks.0.att.out_proj")


def forward(P, bt, ids, unit=False, ex=False):
    """-> dict(lg (B, NB, V, F) [, known/kt per breath, token norms per breath]); bt = the 10 arrays of RB.batch (as Tensors)"""
    vf, vm, fo, fm, A, Rz, Mz, pm, tg, dv = bt
    B, nv, nf = vf.shape[0], vf.shape[1], fo.shape[1]; L = nv + nf; pad = Tensor.cat(vm, fm, dim=1).logical_not()
    if NOTE:
        th = ids.one_hot(NID).cast(dtypes.float) @ P["theta"]; Er, Ei = th.cos(), th.sin(); Rr, Ri = P["rho"].cos(), P["rho"].sin()
        zr = zi = 0
        for r, c in enumerate((A, Rz, Mz)):                                      # z = sum_r (c @ E) * R[r]  (complex, as re/im)
            cr, ci = c @ Er, c @ Ei; zr = zr + cr * Rr[r] - ci * Ri[r]; zi = zi + cr * Ri[r] + ci * Rr[r]
        us = []
        for r in range(3):                                                       # u_r = Re(z conj(R_r) . conj(E)) / PB
            wr, wi = zr * Rr[r] + zi * Ri[r], zi * Rr[r] - zr * Ri[r]
            us.append((Er @ wr.transpose(1, 2) + Ei @ wi.transpose(1, 2)) * (1 / PB))
        u = Tensor.stack(*us, dim=-1)                                            # B x V x F x 3
        V = lin(P, Tensor.cat(Er, Ei, dim=-1), "vin") + emb(P, vf, "flag", 4) + P["kind.weight"][0]
        Fz = lin(P, Tensor.cat(zr, zi, dim=-1), "fin") + emb(P, fo, "op", 4) + P["kind.weight"][1]
        vfb, fvb = lin(P, u, "vf_bias").permute(0, 3, 1, 2), lin(P, u, "fv_bias").permute(0, 3, 2, 1)
        bias = Tensor.cat(Tensor.cat(Tensor.zeros(B, NH, nv, nv), vfb, dim=3), Tensor.cat(fvb, Tensor.zeros(B, NH, nf, nf), dim=3), dim=2)
        mask = pad[:, None, None, :].where(-1e4, bias)
        inc = ((A + Rz + Mz) > 0).cast(dtypes.float).transpose(1, 2)             # B x V x F incidence
        known = ((vf % 2) == 1).cast(dtypes.float); kt = known.cast(dtypes.int) * 0
        Tk = P["kst"] @ P["kproj.weight"].T + P["kproj.bias"]                    # kproj(kst[kt]) as a table
    else:
        S = emb(P, ids, "sym", NID)
        V = S + emb(P, vf, "flag", 4) + P["kind.weight"][0]
        Fz = emb(P, fo, "op", 4) + lin(P, A @ S, "w_arg", False) + lin(P, Rz @ S, "w_res", False) + lin(P, Mz @ S, "w_min", False) + P["kind.weight"][1]
        mask = pad[:, None, None, :].where(-1e4, 0.0)
    x = Tensor.cat(V, Fz, dim=1); valid = pad.logical_not().cast(dtypes.float)
    out, kn, ktl, nrm = [], [], [], [(x * x).sum(-1).sqrt()] if ex else []; xs = [x] if ex else []
    for b in range(NB):
        if DAMP: x = damp(P, x)
        x = write(x, lin(P, P["bst"][b], "bproj").reshape(1, 1, D), unit)
        if NOTE:                                                                  # THE NOTEBOOK: what is derived so far, and when
            kd = emb(P, known.cast(dtypes.int), "known", 2) + (kt.one_hot(NB + 1).cast(dtypes.float) @ Tk) * known.unsqueeze(-1)
            x = Tensor.cat(write(x[:, :nv], kd, unit), x[:, nv:], dim=1)
        x = write(x, mha(P, lnorm(P, x, "blocks.0.ln1"), mask), unit)
        z = lnorm(P, x, "ln"); lg = (lin(P, z[:, :nv], "pq") @ lin(P, z[:, nv:], "pk").transpose(1, 2)) * (1 / math.sqrt(D))
        lg = pm.where(lg, -1e4); out.append(lg)
        p = lg.softmax(-1); ch = lg.argmax(-1); oh = ch.one_hot(nf).cast(dtypes.float)                    # THE SNAP (straight-through)
        st = (oh + (p - p.detach())) * dv.cast(dtypes.float).unsqueeze(-1)
        x = Tensor.cat(write(x[:, :nv], lin(P, st @ x[:, nv:], "snap"), unit), x[:, nv:], dim=1)
        if NOTE:                                                                  # fire: the chosen factor's other vars all known
            unk = 1 - known; miss = (inc.transpose(1, 2) @ unk.unsqueeze(-1)).squeeze(-1)
            missc = (oh * miss.unsqueeze(1)).sum(-1) - unk
            fire = dv & (known == 0) & (missc <= 0.5)
            kt = fire.where(b + 1, kt); known = fire.where(1.0, known)
            if ex: kn.append(known); ktl.append(kt)
        if ex: nrm.append((x * x).sum(-1).sqrt()); xs.append(x)
    r = dict(lg=Tensor.stack(*out, dim=1))
    if ex: r.update(known=kn, kt=ktl, nrm=nrm, valid=valid, xs=xs)
    return r


def loss_fn(lg, tg, dv):
    """RB.loss_fn: set-valued log-marginal (logsumexp over targets) via the exact exp-sum form (no where().logsumexp backward)"""
    lg = lg.contiguous(); m = lg.max(-1, keepdim=True).detach(); e = (lg - m).exp(); dvf = dv.cast(dtypes.float)
    num = m.squeeze(-1) + (e * tg.cast(dtypes.float).unsqueeze(1)).sum(-1).maximum(1e-30).log()
    den = m.squeeze(-1) + e.sum(-1).log()
    return (((den - num) * dvf.unsqueeze(1)).contiguous().sum() / (dvf.sum() * lg.shape[1])).contiguous()


# ---- data ---------------------------------------------------------------------------------------------------------------------
def to_np(bt, ids):
    return [t.numpy() for t in bt], ids.numpy()


def pad_np(arrs, ids, bk=BUCKET):
    """pad (vars, factors) up to a multiple of bk: pads are masked (vm/fm/pm/dv False), ids 0"""
    nv, nf = arrs[0].shape[1], arrs[2].shape[1]; NV, NF = -(-nv // bk) * bk, -(-nf // bk) * bk
    dims = ["V", "V", "F", "F", "FV", "FV", "FV", "VF", "VF", "V"]; tgt = {"V": NV, "F": NF}
    out = [np.pad(a, [(0, 0)] + [(0, tgt[c] - a.shape[i + 1]) for i, c in enumerate(d)]) for a, d in zip(arrs, dims)]
    return out, np.pad(ids, [(0, 0), (0, NV - nv)])


def tens(arrs, ids):
    cv = lambda a: Tensor(a.astype(np.int32) if a.dtype in (np.int64, np.int32) else a)
    return [cv(a) for a in arrs], cv(ids)


def make_batch(its, rng, pad=True):
    bt, ids = RB.batch(its, rng); arrs, ids = to_np(bt, ids)
    return pad_np(arrs, ids) if pad else (arrs, ids)


# ---- training -----------------------------------------------------------------------------------------------------------------
def unused_params(tn, its, rng):
    """names the torch loss leaves grad None (one throwaway step on a deepcopy; torch rng untouched)"""
    import copy
    c = copy.deepcopy(tn); bt, ids = RB.batch(its, rng); RB.loss_fn(c(bt, ids), bt).backward()
    return {n for n, v in c.named_parameters() if v.grad is None}


class Trainer:
    def __init__(s, P, lr, unit, unused=()):
        s.P = P; s.unit = unit; s.params = [v for k, v in P.items() if k not in ("bst", "kst") and k not in unused]
        for v in s.params: v.requires_grad = True
        s.lr = lr; s.t = 0; s.jits = {}; s.bad = 0
        s.m = [Tensor.zeros(*v.shape).contiguous().realize() for v in s.params]; s.v = [Tensor.zeros(*v.shape).contiguous().realize() for v in s.params]

    def adamw(s, ss, bc2s):
        """torch's single-tensor AdamW, op for op (decay first, lerp, addcmul, sqrt/bc2_sqrt + eps, addcdiv by -lr/bc1; the bias
        corrections are host float64 -> float32 inputs): tinygrad's AdamW folds decay into one expression and rounds differently, an
        ulp per step on the phase params (|theta| up to 2 pi) -- enough to flip a notebook argmax tie by step 25 (diagnosed 11-13)"""
        for t, m, v in zip(s.params, s.m, s.v):
            g = t.grad
            t.assign(t * (1 - s.lr * 0.01)); m.assign(m + (g - m) * (1 - 0.9)); v.assign(v * 0.999 + g * g * (1 - 0.999))
            t.assign(t + (m / ((v.sqrt() / bc2s) + 1e-8)) * (-ss))
        return s.params + s.m + s.v

    def _step(s, *a):
        bt, ids, ss, bc2s = list(a[:10]), a[10], a[11], a[12]
        with Tensor.train():
            for v in s.params: v.grad = None
            lo = loss_fn(forward(s.P, bt, ids, s.unit)["lg"], bt[8], bt[9]); lo.backward()
            gp = [v for v in s.params if v.grad is not None]; assert len(gp) == len(s.params), "a trained param has no grad"
            gn = Tensor.stack(*[(v.grad ** 2).sum() for v in gp]).sum().sqrt(); coef = (1.0 / (gn + 1e-6)).minimum(1.0)
            ok = (lo.isfinite()).contiguous()
            for v in gp: v.grad = ok.where(v.grad * coef, 0.0)
            Tensor.realize(lo, *[v.grad for v in gp])                            # pre-update loss (torch's order)
            Tensor.realize(*s.adamw(ss, bc2s))
            return lo

    def step(s, arrs, ids):
        key = (arrs[0].shape[1], arrs[2].shape[1])
        if key not in s.jits: s.jits[key] = TinyJit(s._step)
        s.t += 1; ss = Tensor([np.float32(s.lr / (1 - 0.9 ** s.t))]); bc2s = Tensor([np.float32(math.sqrt(1 - 0.999 ** s.t))])
        bt, idt = tens(arrs, ids); lo = float(s.jits[key](*bt, idt, ss, bc2s).numpy())
        s.bad = 0 if math.isfinite(lo) else s.bad + 1
        if s.bad >= 50: raise RuntimeError("50 consecutive non-finite losses (every update zeroed)")
        return lo


def census(P, items, unit):
    """mean token L2 norm over real var / factor / all tokens after each breath (one fixed batch, eager)"""
    rng = np.random.default_rng(777); its = [items[i % len(items)] for i in range(256)]
    arrs, ids = make_batch(its, rng); bt, idt = tens(arrs, ids); r = forward(P, bt, idt, unit, ex=True)
    nv = arrs[0].shape[1]; val = r["valid"].numpy() > 0; res = dict(all=[], var=[], fac=[])
    for n in r["nrm"][1:]:
        n = n.numpy(); res["all"].append(float(n[val].mean())); res["var"].append(float(n[:, :nv][val[:, :nv]].mean()))
        res["fac"].append(float(n[:, nv:][val[:, nv:]].mean()))
    return res


def pole_census(P, items, unit):
    """(POLE, ledger 11-13) per-breath states of the real tokens (256 DEEP items, the census batch), centered per breath, pooled least
    squares x_{b+1} ~ x_b A over breaths 1..NB-1 -> eigenvalues of A: a local linearization around the trajectory, R^2 reported"""
    rng = np.random.default_rng(777); its = [items[i % len(items)] for i in range(256)]
    arrs, ids = make_batch(its, rng); bt, idt = tens(arrs, ids); r = forward(P, bt, idt, unit, ex=True); val = r["valid"].numpy() > 0
    X = [x.numpy()[val].astype(np.float64) for x in r["xs"]]; X = [x - x.mean(0) for x in X]
    Xa, Xb = np.concatenate(X[1:-1]), np.concatenate(X[2:]); A, *_ = np.linalg.lstsq(Xa, Xb, rcond=None)
    r2 = 1 - ((Xb - Xa @ A) ** 2).sum() / (Xb ** 2).sum(); lam = np.abs(np.linalg.eigvals(A))
    out = dict(r2=float(r2), q=[float(x) for x in np.quantile(lam, [0.25, 0.5, 0.75])], inside=float((lam < 0.99).mean()),
               on=float(((lam >= 0.99) & (lam <= 1.01)).mean()), outside=float((lam > 1.01).mean()), top=[float(x) for x in np.sort(lam)[::-1][:5]])
    if DAMP:
        rad = 1 / (1 + np.exp(-P["damp_a"].numpy().astype(np.float64))); ph = np.degrees(np.abs(P["damp_phi"].numpy()))
        out["learned"] = dict(radius_q=[float(x) for x in np.quantile(rad, [0.25, 0.5, 0.75])], share_lt_0p9=float((rad < 0.9).mean()),
                              angle_deg_q=[float(x) for x in np.quantile(ph, [0.25, 0.5, 0.75])])
    return out


def tg_score(P, items, unit, seed=0):
    """RB.score on the tinygrad forward (same batches: RB.batch with the same rng; the last breath's argmax plan, the executor grades);
    needed for DAMP (the torch net has no poles); checked equal to RB.score on the control path"""
    rng = np.random.default_rng(10_000 + seed); ok = []
    for s0 in range(0, len(items), 256):
        its = items[s0:s0 + 256]; arrs, ids = make_batch(its, rng); bt, idt = tens(arrs, ids); lg = forward(P, bt, idt, unit)["lg"].numpy()[:, -1]
        for k, t in enumerate(its): ok.append(RB.RC.execute(t["facs"], {v: int(lg[k, v].argmax()) for v in t["der"]}, t["query"]) == t["gold"])
    return np.asarray(ok)


def torch_flags(lgT, vf, A, Rz, Mz, dv):
    """replay the torch notebook's fire rule (rel_dsl_organs.make_net) from the torch logits -> per-breath known, kt"""
    import torch
    inc = ((A + Rz + Mz) > 0).float().transpose(1, 2); known = ((vf % 2) == 1).float(); kt = torch.zeros_like(known, dtype=torch.long); K, T = [], []
    for b in range(NB):
        ch = lgT[:, b].argmax(-1); miss = torch.einsum("bvf,bv->bf", inc, 1 - known); missc = torch.gather(miss, 1, ch) - (1 - known)
        fire = dv & (known == 0) & (missc <= 0.5); kt = torch.where(fire, torch.full_like(kt, b + 1), kt); known = torch.where(fire, torch.ones_like(known), known)
        K.append(known.clone()); T.append(kt.clone())
    return K, T


if __name__ == "__main__":
    import torch
    MODE = os.environ.get("TG_MODE", "parity"); SEED, STEPS = RB.SEED, RB.STEPS
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "4")))
    train, shallow, mid, deep = RO.load_depth(); mk = RO.make_net
    if MODE == "parity":
        assert not UNIT; torch.manual_seed(0); tn = mk(); tn.train(); P = from_torch(tn.state_dict()); rng = np.random.default_rng(5); worst = 0.0
        for nm, items in (("train", train), ("deep", deep)):
            its = [items[int(i)] for i in rng.integers(0, len(items), 64)]; bt, ids = RB.batch(its, rng)
            with torch.no_grad(): lgT = tn(bt, ids)
            arrs, idn = pad_np(*to_np(bt, ids)); tb, ti = tens(arrs, idn); r = forward(P, tb, ti, False, ex=True)
            nv, nf = bt[0].shape[1], bt[2].shape[1]; lgG = r["lg"].numpy()[:, :, :nv, :nf]; pm = bt[7].numpy()[:, None].repeat(NB, 1)
            d = np.abs(lgG - lgT.numpy())[pm]; worst = max(worst, float(d.max()))
            per = [float(np.abs(lgG[:, b] - lgT[:, b].numpy())[pm[:, b]].max()) for b in range(NB)]
            msg = f"[tg] parity {ARM} {nm}: pad {tuple(arrs[0].shape[1:2])}x{arrs[2].shape[1]} (trim {nv}x{nf}) max |d| {d.max():.2e} | per-breath " + " ".join(f"{x:.1e}" for x in per)
            if NOTE:
                K, T = torch_flags(lgT, bt[0], bt[4], bt[5], bt[6], bt[9])
                kd = max(int((r["known"][b].numpy()[:, :nv] != K[b].numpy()).sum()) for b in range(NB)); td = max(int((r["kt"][b].numpy()[:, :nv] != T[b].numpy()).sum()) for b in range(NB))
                msg += f" | notebook flags mismatches: known {kd} kt {td} (fired total {int(K[-1].sum())})"
            print(msg + f" | loss torch {float(RB.loss_fn(lgT, bt)):.6f} tg {float(loss_fn(Tensor(lgG), Tensor(bt[8].numpy()), Tensor(bt[9].numpy())).numpy()):.6f}", flush=True)
        print(f"[tg] parity {ARM} GATE (max |d| <= 1e-3): worst {worst:.2e} -> {'PASS' if worst <= 1e-3 else 'FAIL'}", flush=True)
    elif MODE == "smoke":
        n = int(os.environ.get("TG_STEPS", "40")); torch.manual_seed(SEED); tn = mk(); P = from_torch(tn.state_dict())
        rng = np.random.default_rng(SEED); un = unused_params(tn, [train[int(i)] for i in rng.integers(0, len(train), 8)], np.random.default_rng(12345))
        tr = Trainer(P, RB.LR, UNIT, un); rng = np.random.default_rng(SEED); cg = []; t0 = time.time()
        for step in range(n):
            its = [train[int(i)] for i in rng.integers(0, len(train), RB.BATCH)]; cg.append(tr.step(*make_batch(its, rng)))
        tg_t = time.time() - t0; torch.manual_seed(SEED); tn = mk(); opt = torch.optim.AdamW(tn.parameters(), lr=RB.LR); rng = np.random.default_rng(SEED); ct = []; t0 = time.time()
        for step in range(n):
            its = [train[int(i)] for i in rng.integers(0, len(train), RB.BATCH)]; bt, ids = RB.batch(its, rng); loss = RB.loss_fn(tn(bt, ids), bt)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(tn.parameters(), 1.0); opt.step(); ct.append(float(loss))
        for s_ in sorted({1, 2, 5, 10, 20, 30, n}): print(f"[tg] smoke {ARM} step {s_}: tg {cg[s_ - 1]:.5f} torch {ct[s_ - 1]:.5f} (d {cg[s_ - 1] - ct[s_ - 1]:+.5f})", flush=True)
        d = abs(cg[-1] - ct[-1]); dm = abs(np.mean(cg[-10:]) - np.mean(ct[-10:]))
        print(f"[tg] smoke {ARM} GATE step {n} to 3 decimals: |d| {d:.5f} (last-10 mean |d| {dm:.5f}) -> {'PASS' if round(cg[-1], 3) == round(ct[-1], 3) or d < 5e-4 else 'FAIL'} | "
              f"tg {tg_t / n:.2f}s/step (incl. JIT compiles, buckets {sorted(tr.jits)}) torch {(time.time() - t0) / n:.2f}s/step | unused {sorted(un)}", flush=True)
    elif MODE == "unitcheck":
        assert UNIT; torch.manual_seed(0); tn = mk(); P = from_torch(tn.state_dict()); x = Tensor(np.random.default_rng(0).normal(size=(4, 7, D)).astype(np.float32))
        e0 = float(np.abs(expmap(x, Tensor.zeros(1, 1, D)).numpy() - x.numpy()).max())   # compared on the host (a fused x - x reassociates)
        print(f"[tg] unit (a) expmap(x,0)==x: max |d| {e0:.1e} -> {'PASS' if e0 == 0 else 'FAIL'}", flush=True)
        rng = np.random.default_rng(3); its = [deep[int(i)] for i in rng.integers(0, len(deep), 64)]; arrs, ids = make_batch(its, rng); bt, idt = tens(arrs, ids)
        ru = forward(P, bt, idt, True, ex=True); r0 = forward(P, bt, idt, False, ex=True); val = ru["valid"].numpy() > 0; n0 = ru["nrm"][0].numpy()
        dr = max(float((np.abs(n.numpy() - n0) / n0)[val].max()) for n in ru["nrm"][1:]); d0 = max(float((np.abs(n.numpy() - n0) / n0)[val].max()) for n in r0["nrm"][1:])
        print(f"[tg] unit (b) norm drift over NB={NB}: UNIT=1 max rel {dr:.2e} -> {'PASS' if dr <= 1e-4 else 'FAIL'} (UNIT=0 for contrast: {d0:.2e})", flush=True)
        un = unused_params(tn, its, np.random.default_rng(12345)); tr = Trainer(P, RB.LR, 1, un)
        with Tensor.train():
            lo = loss_fn(ru["lg"], bt[8], bt[9]); lo.backward()
        none = [k for k, v in zip([k for k in P if k not in ("bst", "kst") and k not in un], tr.params) if v.grad is None]
        bad = [k for k, v in zip([k for k in P if k not in ("bst", "kst") and k not in un], tr.params) if v.grad is not None and not np.isfinite(v.grad.numpy()).all()]
        gn = float(np.sqrt(sum(float((v.grad ** 2).sum().numpy()) for v in tr.params if v.grad is not None)))
        print(f"[tg] unit (c) grads: {len(tr.params)} trained params, None {none}, non-finite {bad}, loss {float(lo.numpy()):.4f}, grad norm {gn:.3f} -> {'PASS' if not none and not bad else 'FAIL'}", flush=True)
        dd = float(np.abs(ru["lg"].numpy() - r0["lg"].numpy())[bt[7].numpy()[:, None].repeat(NB, 1)].max())
        print(f"[tg] unit (d) UNIT=1 vs UNIT=0 logits: max |d| {dd:.3e} -> {'PASS' if dd > 1e-3 else 'FAIL'}", flush=True)
    elif MODE == "train":
        torch.manual_seed(SEED); rng = np.random.default_rng(SEED); tn = mk(); P = from_torch(tn.state_dict()); npar = sum(p.numel() for p in tn.parameters())
        if DAMP: P = add_damp(P); npar += D
        SC = (lambda its: tg_score(P, its, UNIT)) if DAMP else (lambda its: (to_torch(P, tn), tn.eval(), RB.score(tn, its), tn.train())[2])
        un = unused_params(tn, [train[int(i)] for i in np.random.default_rng(1).integers(0, len(train), 8)], np.random.default_rng(12345))
        tr = Trainer(P, RB.LR, UNIT, un)
        rp = lambda its: float(np.mean([RB.RC.execute(t["facs"], RB.RC.result_plan(t), t["query"]) == t["gold"] for t in its]))
        print(f"[tg] {TAG} arm {ARM} unit {UNIT} | train {len(train)} | shallow {len(shallow)} mid {len(mid)} deep {len(deep)} | params {npar} | tinygrad "
              f"{Tensor.zeros(1).device} (unused {sorted(un)})", flush=True)
        t0 = time.time(); curve = []; tstep = 0.0; t20 = None
        for step in range(1, STEPS + 1):
            its = [train[int(i)] for i in rng.integers(0, len(train), RB.BATCH)]; ts = time.time(); lo = tr.step(*make_batch(its, rng)); tstep += time.time() - ts
            if step == 20: t20 = tstep
            if step % 2000 == 0 or step == STEPS:
                s_ = SC(shallow).mean(); d_ = SC(deep).mean()
                curve.append(dict(step=step, loss=lo, shallow=float(s_), deep=float(d_)))
                print(f"[tg] {TAG} step {step} loss {lo:.4f} SHALLOW {s_:.3f} DEEP {d_:.3f} ({time.time() - t0:.0f}s)", flush=True)
        sps = tstep / STEPS; steady = (tstep - t20) / (STEPS - 20) if t20 is not None and STEPS > 20 else float("nan")
        print(f"[tg] {TAG} train s/step {sps:.3f} (steps only, incl. JIT compiles over buckets {sorted(tr.jits)}); steady (21..{STEPS}) {steady:.3f}", flush=True)
        S = {k: SC(v) for k, v in (("shallow", shallow), ("mid", mid), ("deep", deep))}
        agree = None if DAMP else float((tg_score(P, deep, UNIT) == S["deep"]).mean())   # the tinygrad scorer's witness on the control path
        pc = pole_census(P, deep, UNIT)
        bydepth = {d: float(np.mean([o for o, t in zip(S["deep"], deep) if t["depth"] == d])) for d in range(5, 10) if any(t["depth"] == d for t in deep)}
        nc = census(P, deep, UNIT)
        json.dump(dict(tag=TAG, arm=ARM, seed=SEED, split="depth2", n_train=len(train), params=npar, steps=STEPS, curve=curve,
                       **{k: float(v.mean()) for k, v in S.items()}, **{k + "_per": [int(x) for x in v] for k, v in S.items()},
                       deep_by_depth=bydepth, cov=COV, tree=None, result=dict(shallow=rp(shallow), mid=rp(mid), deep=rp(deep)),
                       engine="tinygrad", unit=UNIT, damp=DAMP, norm_census=nc, s_per_step=sps, pole=pc, tg_score_agree=agree), open(f".cache/rel_dsl_{TAG}.json", "w"))
        print(f"[tg] {TAG} FINAL SHALLOW {S['shallow'].mean():.3f} MID {S['mid'].mean():.3f} DEEP {S['deep'].mean():.3f} | by depth "
              + " ".join(f"d{d} {v:.3f}" for d, v in bydepth.items()) + " | norms(all) " + " ".join(f"{x:.2f}" for x in nc["all"]), flush=True)
        print(f"[tg] {TAG} POLE R2 {pc['r2']:.3f} | |lambda| q25/50/75 " + "/".join(f"{x:.3f}" for x in pc["q"]) + f" | inside {pc['inside']:.3f} on {pc['on']:.3f} "
              f"outside {pc['outside']:.3f} | top " + " ".join(f"{x:.3f}" for x in pc["top"]) + (f" | learned {pc['learned']}" if DAMP else f" | tg-score agree {agree}"), flush=True)
    else: raise SystemExit(f"unknown TG_MODE {MODE}")
