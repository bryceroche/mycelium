"""gsm_breath_tg.py -- THE BT ON TINYGRAD (ledger 2026-11-03, the roadmap's practical lever: Bryce "lets move towards our long term
vision"). A port of gsm_breath_reader.py's network (read + breath + ptrs) and train_step to tinygrad, so the BT trains on the GPU.

  - the torch module stays the source of truth for data, notebook, executor, gold and scoring rules (imported, never edited here);
    the parameter dict uses the torch state_dict's names, so banked .pt weights load directly and a torch init can be copied in
  - gathers are one-hot matmuls (exact: one nonzero term), masks are where(); the executor/notebook loop stays host-side (numpy)
  - training pads every batch's question block to a 32-bucket (pads are masked keys: exp(-1e4) = 0 in fp32, so padding is exact
    up to reduction order) and runs one TinyJit per bucket; the rng sequence replicates the torch train_step exactly, so a run
    sees the torch run's batches and masks
  - ported: BT4 (CYCLOCK, ZOOM, ROLE, T6OFF) + SOFT and KINDCE in train_step; not ported: FACT (BT8), ORACLE, GATE (score = the schedule)
Modes (TG_MODE): parity (DEV=CPU; banked weights vs torch: tensors per cycle + score metrics), smoke (same init + same data, both
frameworks, TG_STEPS steps), time (the GPU timing smoke: s/step at BATCH 64). Env as gsm_breath_reader (BT_*), TG_STEPS, TG_LOAD.
"""
import math, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import gsm_breath_reader as BR
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.nn.optim import AdamW

D, NH, DB, NLE, NLB, NCYC = BR.D, BR.NH, BR.DB, BR.NLE, BR.NLB, BR.NCYC
K, NC, NT, STOP, NOP = BR.K, BR.NC, BR.NT, BR.STOP, BR.NOP
DH = D // NH; NPA = D // 2; NTK = NPA // 3; NNAME = BR.TR.NNAME
assert not BR.FACT and not BR.ORACLE and not BR.GATE, "not ported: FACT / ORACLE / GATE"
BUFFERS = ("C", "Cn", "Cc", "Ccn")

_W = np.radians(BR.CK.t6_plane_speeds(DH // 2)).astype(np.float32)
_WH = np.radians(BR.CK.t6_plane_speeds(DH // 4)).astype(np.float32)
_WT = np.radians(BR.CK.t6_plane_speeds(NTK)).astype(np.float32)
_KM = np.zeros((NT, 2, 3), np.float32)
for _j, (_, _a, _b) in enumerate(BR.TYPES):
    _KM[_j, 0, BR.KINDS.index(_a)] = 1; _KM[_j, 1, BR.KINDS.index(_b)] = 1
_KM[STOP] = 1
_PARM = np.zeros((NOP, NT), np.float32); _PARM[BR.PARENT, np.arange(NT)] = 1               # (lse @ PARM)[t] = lse[parent(t)]
_KMOP = np.stack([(_KM[_PARM[g] > 0].max(0)) for g in range(NOP)])                          # the op-level mask (union of its types)
_KMOPT = _KMOP[BR.PARENT]                                                                    # per type: its op's mask
W, WH, WT = Tensor(_W).realize(), Tensor(_WH).realize(), Tensor(_WT).realize()
KM6, KMOP6 = Tensor(_KM.reshape(NT, 6)).realize(), Tensor(_KMOPT.reshape(NT, 6)).realize()
PARM, GRP = Tensor(_PARM).realize(), Tensor(_PARM > 0).realize()
TRIL = Tensor(np.tril(np.ones((K, K), bool), -1)).realize()
FIELDS = ("x", "theta", "nid", "isn", "isnum", "g", "off", "sid", "pad", "role")


def from_torch(sd):
    """a torch state_dict -> the parameter dict (same names), realized on the default device"""
    return {k: Tensor(v.detach().cpu().numpy().astype(np.float32)).contiguous().realize() for k, v in sd.items()}


def lin(P, x, n, bias=True):
    y = x @ P[n + ".weight"].T
    return y + P[n + ".bias"] if bias else y


def lnorm(P, x, n):
    return x.layernorm(axis=-1, eps=1e-5) * P[n + ".weight"] + P[n + ".bias"]


def emb(P, idx, n, size):
    return idx.one_hot(size).cast(dtypes.float) @ P[n + ".weight"]


def rot(z, a):
    """z (..., 2P) as P interleaved planes; a (..., P)"""
    sh = z.shape; zz = z.reshape(*sh[:-1], sh[-1] // 2, 2); re, im = zz[..., 0], zz[..., 1]; c, s = a.cos(), a.sin()
    return Tensor.stack(re * c - im * s, re * s + im * c, dim=-1).reshape(sh)


def ang(sid, off):
    if not BR.T6OFF: return sid.cast(dtypes.float).unsqueeze(-1) * W
    return Tensor.cat(sid.cast(dtypes.float).unsqueeze(-1) * WH, off.cast(dtypes.float).unsqueeze(-1) * WH, dim=-1)


def attn(P, n, xq, xk, aq, ak, mask):
    B, Lq, Lk = xq.shape[0], xq.shape[1], xk.shape[1]
    q = rot(lin(P, lnorm(P, xq, n + ".lnq"), n + ".q").reshape(B, Lq, NH, DH).transpose(1, 2), aq.unsqueeze(1))
    kv = lin(P, lnorm(P, xk, n + ".lnk"), n + ".kv")
    k = rot(kv[..., :D].reshape(B, Lk, NH, DH).transpose(1, 2), ak.unsqueeze(1)); v = kv[..., D:].reshape(B, Lk, NH, DH).transpose(1, 2)
    sc = ((q @ k.transpose(-2, -1)) * (1 / math.sqrt(DH)) + mask).clip(-1e4, 1e4)
    return xq + lin(P, (sc.softmax(-1) @ v).transpose(1, 2).reshape(B, Lq, D), n + ".o")


def read(P, t):
    x = lin(P, lnorm(P, t["x"], "inp.0"), "inp.1")
    th = t["nid"].maximum(0).one_hot(NNAME).cast(dtypes.float) @ t["theta"]
    x = t["isn"].unsqueeze(-1).where(lin(P, Tensor.cat(th.cos(), th.sin(), dim=-1), "idp"), x)
    x = x + emb(P, t["isn"].cast(dtypes.int), "isn", 2) + emb(P, t["isnum"].cast(dtypes.int), "isnum", 2) + lin(P, t["g"], "gp")
    if not BR.T6OFF: x = x + emb(P, t["off"], "off", BR.OMAX)
    if BR.ROLE: x = x + lin(P, t["role"], "role", bias=False)
    Lq = x.shape[1] - K; xq = x[:, :Lq]; aq = ang(t["sid"][:, :Lq], t["off"][:, :Lq])
    mask = t["pad"][:, None, None, :Lq].where(-1e4, 0.0)
    for i in range(NLE): xq = attn(P, f"enc.{i}", xq, xq, aq, aq, mask)
    return xq


def breath(P, t, xq, nb, cyc, sp=None):
    """one cycle -> dict(logit, p, z, lfine, ctx) ; ptrs(P, out, c, zoom) gives the pointer logits under types c"""
    typ, r, cf, lev, cons = nb
    B, Lq = xq.shape[0], xq.shape[1]; L = Lq + K
    com = typ >= 0
    re = lin(P, r.unsqueeze(-1) * (typ.maximum(0).one_hot(NT).cast(dtypes.float) @ P["C"].detach()), "reent")
    unc = P["maskv"] if sp is None else P["maskv"] + lin(P, sp @ P["C"].detach(), "sre", bias=False)
    h = P["slotk.weight"].unsqueeze(0) + com.unsqueeze(-1).where(re, unc) + lin(P, cf, "cfp") + emb(P, lev, "lev", BR.LEVMAX + 1) \
        + emb(P, cons[:, Lq:].cast(dtypes.int), "cons", 2)
    xk_q = xq + emb(P, cons[:, :Lq].cast(dtypes.int), "cons", 2)
    sa = ang(t["sid"][:, Lq:], t["off"][:, Lq:]); ka = ang(t["sid"], t["off"])
    if BR.CYCLOCK: sa = sa + cyc.cast(dtypes.float).reshape(B, 1, 1) * W
    mask = t["pad"][:, None, None, :].where(-1e4, 0.0)
    for i in range(NLB): h = attn(P, f"core.{i}", h, Tensor.cat(xk_q, h, dim=1), sa, ka, mask)
    z = lin(P, lnorm(P, h, "down.0"), "down.1")
    logit = -((z.unsqueeze(2) - P["C"].detach()) ** 2).sum(-1) * (1 / math.sqrt(DB)) * P["tau"].exp()
    lfine = None
    if BR.ZOOM:
        lc_ = (-((z.unsqueeze(2) - P["Cc"].detach()) ** 2).sum(-1) * (1 / math.sqrt(DB)) * P["tauc"].exp()).log_softmax(-1)
        lse = Tensor.stack(*[GRP[g].where(logit, -1e9).logsumexp(-1) for g in range(NOP)], dim=-1)
        lfine = logit - lse @ PARM; logit = lc_ @ PARM + lfine
    return dict(logit=logit, p=logit.softmax(-1), z=z, lfine=lfine, xk_q=xk_q, cons=cons, t=t, Lq=Lq)


def ptrs(P, o, c, zoom=False, free=False):
    """the pointer heads under types c (B, K) int -> pa, pb in the torch layout [L positions | K slots | NC consts]"""
    t, xk_q, cons, Lq, z, p = o["t"], o["xk_q"], o["cons"], o["Lq"], o["z"], o["p"]
    B = z.shape[0]; L = Lq + K
    oc = c.one_hot(NT).cast(dtypes.float)
    rr = (p * oc).sum(-1, keepdim=True)
    u = lin(P, z, "up") + lin(P, rr * (oc @ P["C"].detach()), "upc"); hu = lnorm(P, u, "ln")
    tick = t["sid"][:, Lq:].cast(dtypes.float).unsqueeze(-1) * WT

    def bind(q, role):
        return Tensor.cat(q[..., :2 * NTK], rot(q[..., 2 * NTK:], oc @ P["phi"][:, role, :]), dim=-1)

    def tickrot(z_):
        return Tensor.cat(rot(z_[..., :2 * NTK], tick), z_[..., 2 * NTK:], dim=-1)
    kt = lin(P, xk_q, "kt"); ksl = tickrot(lin(P, hu, "ks")); cnt = cons.cast(dtypes.float)
    km = (oc @ KM6).reshape(B, K, 2, 3) > 0.5
    if free: km = Tensor.ones(B, K, 2, 3, dtype=dtypes.bool)
    if zoom:
        unsure = ((o["lfine"] * oc).sum(-1)).exp() < BR.ZOOM_TAU
        km = unsure.reshape(B, K, 1, 1).where((oc @ KMOP6).reshape(B, K, 2, 3) > 0.5, km)
    outs = []
    for role, n in ((0, "qa"), (1, "qb")):
        q = bind(lin(P, hu, n), role)
        lt = q @ kt.transpose(-1, -2) + P["pc"] * cnt[:, None, :Lq] * math.sqrt(D)
        lt = (t["isnum"][:, None, :Lq].logical_not() | km[:, :, role, 0:1].logical_not()).where(-1e4, lt)
        ls = tickrot(q) @ ksl.transpose(-1, -2) + P["pc"] * cnt[:, None, Lq:] * math.sqrt(D)
        ls = (TRIL.logical_not().unsqueeze(0) | km[:, :, role, 1:2].logical_not()).where(-1e4, ls)
        lc = (km[:, :, role, 2:3].logical_not()).where(-1e4, q @ P["kc"].T)
        outs.append(Tensor.cat(lt, ls, lc, dim=-1) * (1 / math.sqrt(D)))
    padb = Tensor.full((B, K, L - Lq), -1e4)
    return tuple(Tensor.cat(x[..., :Lq], padb, x[..., Lq:], dim=-1) for x in outs)


# ---- data ----------------------------------------------------------------------------------------------------------------------
def to_np(t):
    return {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in t.items()}


def repad(t, Lq_new):
    """insert pad positions so the question block is Lq_new long (pads are masked keys; exact up to reduction order)"""
    L = t["pad"].shape[1]; Lq = L - K; ex = Lq_new - Lq
    if ex == 0: return t
    out = dict(t)
    for k in ("x", "isn", "isnum", "sid", "off", "nid", "g", "pad", "role"):
        if k not in t: continue
        a = t[k]; fill = np.ones if k == "pad" else (lambda s, dtype: np.full(s, -1, dtype)) if k == "nid" else np.zeros
        mid = fill((a.shape[0], ex) + a.shape[2:], dtype=a.dtype)
        out[k] = np.concatenate([a[:, :Lq], mid, a[:, Lq:]], 1)
    for k in ("va", "vb"):
        a = t[k]; out[k] = np.concatenate([a[..., :Lq], np.zeros(a.shape[:2] + (ex,), bool), a[..., Lq:]], -1)
    return out


def tens(t):
    out = {}
    for k in FIELDS:
        if k not in t: continue
        a = t[k]
        out[k] = Tensor(a.astype(np.int32)) if a.dtype in (np.int64, np.int32) else Tensor(a.astype(np.float32)) if a.dtype != bool else Tensor(a)
    return out


def nb_t(arrs):
    typ, r, cf, lev, cons = arrs
    return Tensor(typ.astype(np.int32)), Tensor(r.astype(np.float32)), Tensor(cf.astype(np.float32)), Tensor(lev.astype(np.int32)), Tensor(cons)


# ---- training -------------------------------------------------------------------------------------------------------------------
class Trainer:
    def __init__(s, P, lr):
        assert BR.ROLE, "the train step's signature carries the role feats (BT_ROLE=1)"
        s.P = P; s.params = [v for k, v in P.items() if k not in BUFFERS]
        for v in s.params: v.requires_grad = True
        s.opt = AdamW(s.params, lr=lr, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.01); s.jits = {}

    def _step(s, x, theta, nid, isn, isnum, g, off, sid, pad, role, typ, r, cf, lev, cons, gt, mk, am, va, vb, cyc, sp):
        P = s.P; t = dict(x=x, theta=theta, nid=nid, isn=isn, isnum=isnum, g=g, off=off, sid=sid, pad=pad, role=role)
        with Tensor.train():
            s.opt.zero_grad()
            xq = read(P, t); o = breath(P, t, xq, (typ, r, cf, lev, cons), cyc, sp if BR.SOFT else None)
            pa, pb = ptrs(P, o, gt)
            ce = -(o["logit"].log_softmax(-1) * gt.one_hot(NT).cast(dtypes.float)).sum(-1).contiguous()
            mw = (mk / mk.sum()).contiguous(); aw = (am / am.sum().maximum(1)).contiguous()
            lo = (ce * mw).sum()
            heads = [(pa, va), (pb, vb)]
            if BR.KINDCE: heads += list(zip(ptrs(P, o, gt, free=True), (va, vb)))
            for q_, V in heads:
                term = (q_.logsumexp(-1) - V.where(q_, -1e4).logsumexp(-1)).contiguous()
                lo = lo + (term * aw).sum()
            lo.backward()
            gp = [v for v in s.params if v.grad is not None]
            gn = Tensor.stack(*[(v.grad ** 2).sum() for v in gp]).sum().sqrt()
            coef = (1.0 / (gn + 1e-6)).minimum(1.0)
            ok = lo.isfinite().cast(dtypes.float)
            for v in gp: v.grad = v.grad * coef * ok
            s.opt.step()
            # Welford: the atoms follow the (pre-update) silhouettes of the masked gold slots; the coarse atoms over the op
            zz = o["z"].detach().reshape(-1, DB); oh = (gt.one_hot(NT).cast(dtypes.float) * mk.unsqueeze(-1)).reshape(-1, NT)
            for Cn_, Ck_, ohx in (("Cn", "C", oh),) + ((("Ccn", "Cc", oh @ PARM.T),) if BR.ZOOM else ()):
                cnt = ohx.sum(0); sm = ohx.T @ zz; nn_ = P[Cn_] + cnt; upd = (cnt > 0).unsqueeze(-1)
                P[Ck_].assign(upd.where(P[Ck_] + (sm - cnt.unsqueeze(-1) * P[Ck_]) / nn_.maximum(1).unsqueeze(-1), P[Ck_]))
                P[Cn_].assign(nn_.minimum(BR.NCAP))
            Tensor.realize(lo, *[P[b] for b in BUFFERS if b in P])   # the Welford assigns land in this step (else the next backward meets them)
            return lo

    def step(s, its, states, tree, rng):
        """train_step with the torch rng sequence (BR.train_step's draws, in order) -> loss"""
        t, E = BR.batch(its, states, tree, rng); t = to_np(t); L = t["pad"].shape[1]
        for b, it in enumerate(its): E[b]["vals"] = it["vals"]
        progs, gt = BR.gold(E)
        m = np.array([e["m"] for e in E]); rr = rng.random(len(E))[:, None]
        masked = rng.random((len(E), K)) < rr
        masked[np.arange(len(E)), rng.integers(0, np.minimum(m + 1, K))] = True
        cyc = np.clip(np.floor(NCYC * (1 - rr[:, 0])), 0, NCYC - 1).astype(np.int32)
        soft = BR.SOFT and rng.random() < 0.5
        Lq = L - K; Lb = max(64, int(math.ceil(Lq / 32) * 32)); t = repad(t, Lb)
        nb = BR.notebook(E, Lb + K, progs, ~masked, gt, np.ones((len(E), K), np.float32))
        tt = tens(t); nbt = nb_t(nb); gtt = Tensor(gt.astype(np.int32)); cyct = Tensor(cyc)
        sp = Tensor.zeros(len(E), K, NT).contiguous()
        if soft:   # the self-conditioning pre-pass (no grad), as torch's: the model's own belief at this mask level
            tau = (2.0 ** (1.0 - 2.0 * cyc / (NCYC - 1))).astype(np.float32)
            o0 = breath(s.P, tt, read(s.P, tt), nbt, cyct); sp = (o0["logit"] / Tensor(tau).reshape(-1, 1, 1)).softmax(-1).detach().realize()
        key = (Lb, soft)
        if key not in s.jits: s.jits[key] = TinyJit(s._step)
        return float(s.jits[key](*[tt[k] for k in FIELDS if k in tt], *nbt, gtt, Tensor(masked.astype(np.float32)),
                                 Tensor((t["am"] & masked).astype(np.float32)), Tensor(t["va"]), Tensor(t["vb"]), cyct, sp).numpy())


# ---- inference ------------------------------------------------------------------------------------------------------------------
def score(P, its, states, tree, seed=0):
    """BR.score's schedule loop on the tinygrad net (eager) -> the same metric dict"""
    res = dict(ans=[], step_op=[], step_ptr=[], step_all=[], type_ok=[])
    for s0 in range(0, len(its), 128):
        part = its[s0:s0 + 128]
        t, E = BR.batch(part, states, tree, np.random.default_rng(10_000 * seed + s0)); t = to_np(t); L = t["pad"].shape[1]; B = len(part)
        for b, it in enumerate(part): E[b]["vals"] = it["vals"]
        tt = tens(t); xq = read(P, tt).realize()
        com = np.zeros((B, K), bool); typ = np.zeros((B, K), np.int64); rad = np.zeros((B, K), np.float32)
        IA = np.zeros((B, K), np.int64); IB = np.zeros((B, K), np.int64)
        progs = [[(".", None, None)] * K for _ in range(B)]
        dec = lambda i: ("t", int(i)) if i < L else (("s", int(i - L)) if i < L + K else ("c", int(i - L - K)))
        sp = None
        for c in range(NCYC):
            o = breath(P, tt, xq, nb_t(BR.notebook(E, L, progs, com, typ, rad)), Tensor(np.full(B, c, np.int32)), sp)
            lg = o["logit"].numpy()
            if BR.SOFT and c + 1 < NCYC:
                tau = float(2.0 ** (1.0 - 2.0 * (c + 1) / (NCYC - 1))); sp = Tensor(lg / tau).softmax(-1)
            ty = lg.argmax(-1); pa, pb = ptrs(P, o, Tensor(ty.astype(np.int32)), zoom=BR.ZOOM)
            pa, pb = pa.numpy(), pb.numpy(); pt = o["p"].numpy().max(-1)
            ia, ib = pa.argmax(-1), pb.argmax(-1)
            sm = lambda a: np.exp(a - a.max(-1, keepdims=True)) / np.exp(a - a.max(-1, keepdims=True)).sum(-1, keepdims=True)
            conf = np.where(ty == STOP, pt, pt * sm(pa).max(-1) * sm(pb).max(-1))
            target = math.ceil(K * (c + 1) / NCYC)
            for b in range(B):
                need = target - com[b].sum()
                if need <= 0: continue
                for k in [k for k in np.argsort(-conf[b]) if not com[b, k]][:need]:
                    com[b, k] = True; typ[b, k] = ty[b, k]; rad[b, k] = conf[b, k]; IA[b, k], IB[b, k] = ia[b, k], ib[b, k]
                    op = BR.TYPES[ty[b, k]][0] if ty[b, k] != STOP else "."
                    progs[b][k] = (op, dec(ia[b, k]), dec(ib[b, k])) if op != "." else (".", None, None)
        gprogs, gt = BR.gold(E)
        for b, (it, e) in enumerate(zip(part, E)):
            prog = []
            for st in progs[b]:
                if st[0] == ".": break
                prog.append(st)
            v = BR.GD.execute(prog, it["vals"] + [None] * (L - len(it["vals"])))
            res["ans"].append(v is not None and BR.GD.V1.near(v, it["final"]))
            for k in range(e["m"]):
                o_ = progs[b][k][0] == gprogs[b][k][0]; pk = bool(t["va"][b, k, IA[b, k]] and t["vb"][b, k, IB[b, k]])
                res["step_op"].append(o_); res["step_ptr"].append(pk); res["step_all"].append(o_ and pk); res["type_ok"].append(typ[b, k] == gt[b, k])
    return {k: np.asarray(v) for k, v in res.items()}


if __name__ == "__main__":
    import torch
    MODE = os.environ.get("TG_MODE", "parity")
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "4")))
    Dt, states = BR.TR.load(); tree = BR.TR.load_tree()
    BR.RF = np.load(BR.GD.V1.C(BR.rolef_name()), mmap_mode="r") if BR.ROLE else None
    if MODE == "parity":
        tn = BR.make_net(); tn.load_state_dict(torch.load(os.environ.get("TG_LOAD", ".cache/gsmb_breath_BT4T9NS_s0.pt"))); tn.eval()
        P = from_torch(tn.state_dict())
        part = Dt["TEST"][:16]
        t, E = BR.batch(part, states, tree, np.random.default_rng(5)); L = t["pad"].shape[1]; B = len(part)
        for b, it in enumerate(part): E[b]["vals"] = it["vals"]
        tt = tens(to_np(t))
        with torch.no_grad():
            xq_t = tn.read(t)
        xq = read(P, tt).realize(); print(f"[tg] parity read: max |d| {np.abs(xq.numpy() - xq_t.numpy()).max():.2e} (|x| max {xq_t.abs().max():.2f})", flush=True)
        com = np.zeros((B, K), bool); typ = np.zeros((B, K), np.int64); rad = np.zeros((B, K), np.float32)
        progs = [[(".", None, None)] * K for _ in range(B)]
        dec = lambda i: ("t", int(i)) if i < L else (("s", int(i - L)) if i < L + K else ("c", int(i - L - K)))
        for c in range(3):
            nb = BR.notebook(E, L, progs, com, typ, rad)
            with torch.no_grad():
                lg_t, pa_t, pb_t, z_t, p_t = tn.breath(t, xq_t, BR.nb_tensors(nb), None, torch.full((B,), c), None)
            o = breath(P, tt, xq, nb_t(nb), Tensor(np.full(B, c, np.int32)))
            ty = lg_t.argmax(-1); pa, pb = ptrs(P, o, Tensor(ty.numpy().astype(np.int32)), zoom=BR.ZOOM)
            d = {n: float(np.abs(a.numpy() - b_.numpy()).max()) for n, a, b_ in (("logit", o["logit"], lg_t), ("p", o["p"], p_t), ("z", o["z"], z_t), ("pa", pa, pa_t), ("pb", pb, pb_t))}
            print(f"[tg] parity cycle {c}: " + " ".join(f"{n} {v:.2e}" for n, v in d.items()) + f" | argmax type agree {float((o['logit'].numpy().argmax(-1) == ty.numpy()).mean()):.3f}", flush=True)
            conf = p_t.max(-1).values.numpy()
            for b in range(B):
                for k in [k for k in np.argsort(-conf[b]) if not com[b, k]][:2]:
                    com[b, k] = True; typ[b, k] = int(ty[b, k]); rad[b, k] = conf[b, k]
                    op = BR.TYPES[typ[b, k]][0] if typ[b, k] != STOP else "."
                    progs[b][k] = (op, dec(int(pa_t[b, k].argmax())), dec(int(pb_t[b, k].argmax()))) if op != "." else (".", None, None)
        N = int(os.environ.get("TG_PARN", "128"))
        for split in ("ORIG", "TEST"):
            Rt = BR.score(tn, Dt[split][:N], states, tree); Rg = score(P, Dt[split][:N], states, tree)
            print(f"[tg] parity score {split} n {N}: " + " | ".join(f"{m} torch {Rt[m].mean():.4f} tg {Rg[m].mean():.4f} (d {Rg[m].mean() - Rt[m].mean():+.4f}; "
                  f"item agree {float((Rg[m] == Rt[m]).mean()):.3f})" for m in ("ans", "step_all", "type_ok")), flush=True)
    else:
        SEED = BR.SEED; STEPS = int(os.environ.get("TG_STEPS", "60"))
        torch.manual_seed(SEED); tn = BR.make_net(); P = from_torch(tn.state_dict())
        tr = Trainer(P, BR.LR); train = Dt["TRAIN"]; rng = np.random.default_rng(SEED); curve = []; t0 = time.time()
        for step in range(1, STEPS + 1):
            its = [train[int(i)] for i in rng.integers(0, len(train), BR.BATCH)]
            lo = tr.step(its, states, tree, rng); curve.append(lo)
            if step % 10 == 0 or step == STEPS:
                print(f"[tg] {MODE} step {step} loss {np.mean(curve[-10:]):.4f} | {(time.time() - t0) / step:.3f}s/step (cumulative; buckets {len(tr.jits)})", flush=True)
            if step == 20: t20 = time.time()
        if STEPS > 20: print(f"[tg] {MODE} steady s/step (steps 21..{STEPS}): {(time.time() - t20) / (STEPS - 20):.3f} | buckets {sorted(tr.jits)}", flush=True)
        if MODE == "smoke":   # the torch curve from the same init on the same rng sequence
            torch.manual_seed(SEED); tn = BR.make_net(); opt = torch.optim.AdamW(tn.parameters(), lr=BR.LR); rng = np.random.default_rng(SEED); ct = []; t0 = time.time()
            for step in range(1, STEPS + 1):
                its = [train[int(i)] for i in rng.integers(0, len(train), BR.BATCH)]
                ct.append(BR.train_step(tn, opt, its, states, tree, rng))
                if step % 10 == 0 or step == STEPS: print(f"[tg] torch step {step} loss {np.mean(ct[-10:]):.4f} | {(time.time() - t0) / step:.3f}s/step", flush=True)
