"""gsm_breath_reader.py -- THE BT ARM (ledger 2026-10-30, word given: "the diffusion reader should be a true Breathing Transformer
honoring all our ideas such as no MLP"). The GSM8K pointer-program bench (gsm_trunk_reader.py's data, splits, executor and scoring
definitions), read by a breathing transformer under every ruling:
  NO MLP          attention, LayerNorm and linear maps only (the U-Net projections are linear)
  READER          run once per problem: TICK's input chassis (trunk state, ID phases, tree gender, offset, T6 sentence clock on q/k)
  THE BREATH      ONE weight-tied set of NLB cross+self attention blocks, applied every cycle; slot rows query [question ; all slots]
                  (every step sees every step: the diffusion ruling); the T6 step tick rotates slot q/k
  U-NET           per breath, per step: D -> DB (the SILHOUETTE) -> classified against 37 ATOM centroids (op x operand kinds + stop),
                  Welford running means of gold-typed silhouettes (no gradient, count capped) -> snapped -> expanded back to D
  GRAVITY         type loss = softmax over -distance to every centroid (contrastive: own atom vs the other 36)
  RADIUS          confidence r = p(type) scales the snapped centroid's re-entry; commit order = the largest radius
  MICRO KEYS      pointer queries come ONLY from the expanded silhouette (no full-state skip: the bypass guard), bound with the micro
                  keys: slot id, T6 step tick, tree level (the executor's derivation depth)
  ROTOR PLANES    pointer space = D/2 complex planes: [0:NTK) the T6 step tick (relative rotation of q and slot keys),
                  [NTK:) the BUS: q bound to its type's role phase e^{i phi(type, operand)}, read by the conjugate (Re q e^{i phi} conj k)
  STRUCTURE       the type's operand kinds mask each pointer's domain (token / earlier slot / constant): the silhouette conditions keys
  DIFFUSION       training = masked discrete diffusion on the gold program (loss on masked slots); inference = 6 cycles from all-masked,
                  committing the most confident slots each cycle
  ALTERNATION     after every cycle the executor runs every committed slot whose operands are known and writes back certificates
                  (executed / integer / non-negative -- never values), consumed marks and tree levels; the next cycle reads them
Env: BT_SEED, BT_TAG, BT_STEPS (6000), BT_BATCH (64), BT_LR (3e-4), TOY_THREADS. Output .cache/gsmb_breath_{TAG}.json."""
import json, math, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
os.environ["RD_ARM"] = "TICK"                     # the chassis's encoder features (ID phases, tree gender, sentence clock, step tick)
import gsm_trunk_reader as TR
import gsm_bench_data as GD
from mycelium import rotor_clock as CK

SEED = int(os.environ.get("BT_SEED", "0")); TAG = os.environ.get("BT_TAG", f"BT_s{SEED}")
ORACLE = os.environ.get("BT_ORACLE") == "1"          # the oracle-type read: the gold step type is the silhouette's class (inference only)
LOAD = os.environ.get("BT_LOAD", "")                 # score saved weights (.cache/gsmb_breath_{TAG}.pt) instead of training
STEPS, BATCH, LR = int(os.environ.get("BT_STEPS", "6000")), int(os.environ.get("BT_BATCH", "64")), float(os.environ.get("BT_LR", "3e-4"))
D, NH, NLE, NLB, DB, NCYC = 192, 4, 2, 2, 32, 6
K, NC, H, OMAX, PB = TR.K, TR.NC, TR.H, TR.OMAX, TR.PB
KINDS = "tsc"; TYPES = [(o, a, b) for o in GD.OPS for a in KINDS for b in KINDS]; STOP = len(TYPES); NT = STOP + 1
LEVMAX, NCAP = 8, 2000
SPLITS = TR.SPLITS


def step_type(op, a, b):
    return STOP if op == "." else TYPES.index((op, a[0], b[0]))


def notebook(E, L, progs, committed, types, radius):
    """the executor's write-back for the committed slots -> slot_in arrays: typ (K; -1 masked), r, cf (K x 3), lev, cons (L)."""
    B = len(E); typ = np.full((B, K), -1); r = np.zeros((B, K), np.float32); cf = np.zeros((B, K, 3), np.float32)
    lev = np.zeros((B, K), np.int64); cons = np.zeros((B, L), bool)
    for b, e in enumerate(E):
        n, vals = e["n"], e["vals"]; sv = [None] * K; sl = [0] * K
        for k in range(K):
            if not committed[b, k]: continue
            typ[b, k] = types[b, k]; r[b, k] = radius[b, k]; op, A, Bo = progs[b][k]
            if op == ".": continue
            av, known, lv = [], True, 0
            for kind, i in (A, Bo):
                if kind == "t": cons[b, i] = True; av.append(vals[i] if i < n else None)
                elif kind == "s":
                    cons[b, L - K + i] = True; av.append(sv[i] if committed[b, i] else None)
                    if committed[b, i] and sl[i] > 0: lv = max(lv, sl[i])
                    else: known = False                                                    # level unknown until its operands are
                else: av.append(GD.CONSTS[i])
            if known: sl[k] = min(lv + 1, LEVMAX); lev[b, k] = sl[k]
            x, y = av
            if x is not None and y is not None and not (op == "/" and y == 0):
                try: sv[k] = x + y if op == "+" else x - y if op == "-" else x * y if op == "*" else x / y
                except OverflowError: sv[k] = None
            cf[b, k] = TR.cert(sv[k])
    return typ, r, cf, lev, cons


def gold(E):
    """per item: the gold program per slot (steps, then stops) and its step types."""
    progs, types = [], np.full((len(E), K), STOP)
    for b, e in enumerate(E):
        p = list(e["teach"]) + [(".", None, None)] * (K - len(e["teach"]))
        progs.append(p)
        for k, (op, A, Bo) in enumerate(p):
            if op != ".": types[b, k] = step_type(op, A, Bo)
    return progs, types


def make_net():
    import torch, torch.nn as nn, torch.nn.functional as F
    DH = D // NH; NPA = D // 2; NTK = NPA // 3
    w = torch.as_tensor(np.radians(CK.t6_plane_speeds(DH // 2)), dtype=torch.float32)      # attention space: the T6 clock, per head
    wt = torch.as_tensor(np.radians(CK.t6_plane_speeds(NTK)), dtype=torch.float32)        # pointer space: the step tick's planes
    kmask = torch.zeros(NT, 2, 3, dtype=torch.bool)                                       # type x operand -> allowed kind (t, s, c)
    for j, (_, a, b) in enumerate(TYPES):
        kmask[j, 0, KINDS.index(a)] = True; kmask[j, 1, KINDS.index(b)] = True
    kmask[STOP] = True

    def rot(z, a):                                                                         # z (..., 2P) as P planes; a (..., P)
        c, s_ = a.cos(), a.sin(); re, im = z[..., 0::2], z[..., 1::2]
        return torch.stack([re * c - im * s_, re * s_ + im * c], -1).flatten(-2)

    class Attn(nn.Module):
        def __init__(s):
            super().__init__(); s.lnq = nn.LayerNorm(D); s.lnk = nn.LayerNorm(D)
            s.q = nn.Linear(D, D); s.kv = nn.Linear(D, 2 * D); s.o = nn.Linear(D, D)

        def forward(s, xq, xk, aq, ak, mask):
            B, Lq, _ = xq.shape; Lk = xk.shape[1]
            q = rot(s.q(s.lnq(xq)).reshape(B, Lq, NH, DH).transpose(1, 2), aq[:, None])
            k, v = (t.reshape(B, Lk, NH, DH).transpose(1, 2) for t in s.kv(s.lnk(xk)).chunk(2, -1))
            k = rot(k, ak[:, None])
            return xq + s.o(F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(B, Lq, D))

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.inp = nn.Sequential(nn.LayerNorm(H), nn.Linear(H, D)); s.isn = nn.Embedding(2, D); s.isnum = nn.Embedding(2, D)
            s.off = nn.Embedding(OMAX, D); s.idp = nn.Linear(2 * PB, D); s.gp = nn.Linear(3, D)
            s.enc = nn.ModuleList([Attn() for _ in range(NLE)])                                # THE READER (once per problem)
            s.cons = nn.Embedding(2, D)                                                        # the notebook's consumed marks
            s.slotk = nn.Embedding(K, D); s.maskv = nn.Parameter(torch.zeros(D)); s.cfp = nn.Linear(3, D)
            s.lev = nn.Embedding(LEVMAX + 1, D); s.reent = nn.Linear(DB, D)
            s.core = nn.ModuleList([Attn() for _ in range(NLB)])                               # THE BREATH (weight-tied over cycles)
            s.down = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, DB)); s.up = nn.Linear(DB, D); s.upc = nn.Linear(DB, D)
            s.tau = nn.Parameter(torch.zeros(1))
            s.register_buffer("C", torch.randn(NT, DB) * 0.1); s.register_buffer("Cn", torch.zeros(NT))   # the atoms (Welford)
            s.phi = nn.Parameter(torch.rand(NT, 2, NPA - NTK) * 2 * math.pi)                   # type-role phases on the bus planes
            s.ln = nn.LayerNorm(D); s.qa = nn.Linear(D, D); s.qb = nn.Linear(D, D); s.kt = nn.Linear(D, D); s.ks = nn.Linear(D, D)
            s.kc = nn.Parameter(torch.randn(NC, D) * 0.02); s.pc = nn.Parameter(torch.zeros(1))

        def read(s, t):
            x = s.inp(t["x"])
            th = torch.gather(t["theta"], 1, t["nid"].clamp(min=0)[..., None].expand(-1, -1, PB))
            x = torch.where(t["isn"][..., None], s.idp(torch.cat([th.cos(), th.sin()], -1)), x)
            x = x + s.isn(t["isn"].long()) + s.isnum(t["isnum"].long()) + s.gp(t["g"]) + s.off(t["off"])
            Lq = x.shape[1] - K; xq = x[:, :Lq]; aq = t["sid"][:, :Lq, None].float() * w
            mask = torch.zeros(x.shape[0], 1, Lq, Lq).masked_fill(t["pad"][:, None, None, :Lq], -1e4)
            for blk in s.enc: xq = blk(xq, xq, aq, aq, mask)
            return xq

        def breath(s, t, xq, nb, cond):
            """one cycle: every slot proposed in parallel. nb = notebook tensors; cond (B, K) = the step type the keys condition on
            (gold in training, the snapped prediction at inference; None = the snapped prediction)."""
            B, Lq, _ = xq.shape; L = Lq + K
            typ, r, cf, lev, cons = nb
            com = typ >= 0
            re = s.reent(r[..., None] * s.C[typ.clamp(min=0)])                                  # committed: atom x radius
            h = s.slotk.weight[None] + torch.where(com[..., None], re, s.maskv) + s.cfp(cf) + s.lev(lev) + s.cons(cons[:, Lq:].long())
            xk_q = xq + s.cons(cons[:, :Lq].long())
            sa = t["sid"][:, Lq:, None].float() * w; ka = t["sid"][..., None].float() * w
            mask = torch.zeros(B, 1, K, L).masked_fill(t["pad"][:, None, None, :], -1e4)
            for blk in s.core: h = blk(h, torch.cat([xk_q, h], 1), sa, ka, mask)
            z = s.down(h)                                                                      # THE SILHOUETTE
            logit = -((z[:, :, None] - s.C[None, None]) ** 2).sum(-1) / math.sqrt(DB) * s.tau.exp()
            p = logit.softmax(-1)
            c = logit.argmax(-1) if cond is None else cond
            rr = p.gather(-1, c[..., None])                                                    # the radius: type confidence
            u = s.up(z) + s.upc(rr * s.C[c])                                                   # expand: silhouette + snapped atom x r
            hu = s.ln(u)
            tick = t["sid"][:, Lq:, None].float() * wt                                         # (B, K, NTK)

            def bind(q, role):                                                                 # bus planes: q * e^{i phi(type, role)}
                ph = s.phi[c, role]
                return torch.cat([q[..., :2 * NTK], rot(q[..., 2 * NTK:], ph)], -1)

            def tickrot(z_):
                return torch.cat([rot(z_[..., :2 * NTK], tick), z_[..., 2 * NTK:]], -1)

            kt = s.kt(xk_q); ksl = tickrot(s.ks(hu)); cnt = cons.float()
            km = kmask.to(c.device)[c]                                                         # (B, K, 2, 3)
            outs = []
            for role, proj in ((0, s.qa), (1, s.qb)):
                q = bind(proj(hu), role)
                lt = torch.einsum("bkd,bld->bkl", q, kt) + s.pc * cnt[:, None, :Lq] * math.sqrt(D)
                lt = lt.masked_fill(~t["isnum"][:, None, :Lq] | ~km[:, :, role, 0:1], -1e4)
                ls = torch.einsum("bkd,bjd->bkj", tickrot(q), ksl) + s.pc * cnt[:, None, Lq:] * math.sqrt(D)
                ls = ls.masked_fill(~torch.ones(K, K, dtype=torch.bool).tril(-1) | ~km[:, :, role, 1:2], -1e4)
                lc = torch.einsum("bkd,cd->bkc", q, s.kc).masked_fill(~km[:, :, role, 2:3], -1e4)
                outs.append(torch.cat([lt, ls, lc], -1) / math.sqrt(D))
            # pointer layout = TR.batch's valid-set layout: [L positions (question + pad + slots) | K slots | NC consts]
            pad = torch.full((B, K, L - Lq), -1e4)
            pa, pb = (torch.cat([o[..., :Lq], pad, o[..., Lq:]], -1) for o in outs)
            return logit, pa, pb, z, p

    return Net()


def nb_tensors(arrs):
    import torch
    typ, r, cf, lev, cons = arrs
    return (torch.as_tensor(typ), torch.as_tensor(r), torch.as_tensor(cf), torch.as_tensor(lev), torch.as_tensor(cons))


def train_step(net, opt, its, states, tree, rng):
    import torch, torch.nn.functional as F
    t, E = TR.batch(its, states, tree, rng); L = t["pad"].shape[1]
    for b, it in enumerate(its): E[b]["vals"] = it["vals"]
    progs, gt = gold(E)
    m = np.array([e["m"] for e in E]); rr = rng.random(len(E))[:, None]
    masked = rng.random((len(E), K)) < rr
    masked[np.arange(len(E)), rng.integers(0, np.minimum(m + 1, K))] = True                   # at least one supervised slot masked
    nb = nb_tensors(notebook(E, L, progs, ~masked, gt, np.ones((len(E), K), np.float32)))
    xq = net.read(t); gtt = torch.as_tensor(gt)
    logit, pa, pb, z, _ = net.breath(t, xq, nb, gtt)
    mk = torch.as_tensor(masked); am = t["am"] & mk
    lo = F.cross_entropy(logit[mk], gtt[mk])                                                  # gravity: own atom vs the other 36
    for p, V in ((pa, t["va"]), (pb, t["vb"])):
        lo = lo + (p.logsumexp(-1) - p.masked_fill(~V, -1e4).logsumexp(-1))[am].mean()
    opt.zero_grad(); lo.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    with torch.no_grad():                                                                     # Welford: the atoms follow the silhouettes
        zz, tt = z[mk].detach(), gtt[mk]
        cnt = torch.bincount(tt, minlength=NT).float(); sm = torch.zeros(NT, DB).index_add_(0, tt, zz)
        nn_ = net.Cn + cnt; upd = cnt > 0                                                     # Chan's merge of a batch into the mean
        net.C[upd] += (sm[upd] - cnt[upd, None] * net.C[upd]) / nn_[upd, None]
        net.Cn.copy_(nn_.clamp(max=NCAP))                                                     # capped: the atom keeps following
    return float(lo.detach())


def score(net, its, states, tree, seed=0):
    import torch
    res = dict(ans=[], prog=[], step_op=[], step_ptr=[], step_all=[], first_all=[], type_ok=[], num_for_slot=[], commit_cycle=[])
    with torch.no_grad():
        for s0 in range(0, len(its), 128):
            part = its[s0:s0 + 128]
            t, E = TR.batch(part, states, tree, np.random.default_rng(10_000 * seed + s0)); L = t["pad"].shape[1]; B = len(part)
            for b, it in enumerate(part): E[b]["vals"] = it["vals"]
            xq = net.read(t)
            com = np.zeros((B, K), bool); typ = np.zeros((B, K), np.int64); rad = np.zeros((B, K), np.float32)
            IA = np.zeros((B, K), np.int64); IB = np.zeros((B, K), np.int64); cyc = np.full((B, K), -1)
            progs = [[(".", None, None)] * K for _ in range(B)]
            dec = lambda i: ("t", int(i)) if i < L else (("s", int(i - L)) if i < L + K else ("c", int(i - L - K)))
            gt0 = gold(E)[1]
            for c in range(NCYC):
                nb = nb_tensors(notebook(E, L, progs, com, typ, rad))
                logit, pa, pb, _, p = net.breath(t, xq, nb, torch.as_tensor(gt0) if ORACLE else None)
                ty = gt0 if ORACLE else logit.argmax(-1).numpy()
                pt = p.gather(-1, torch.as_tensor(ty)[..., None])[..., 0].numpy() if ORACLE else p.max(-1).values.numpy()
                ia, ib = pa.argmax(-1).numpy(), pb.argmax(-1).numpy()
                ca = pa.softmax(-1).max(-1).values.numpy(); cb = pb.softmax(-1).max(-1).values.numpy()
                conf = np.where(ty == STOP, pt, pt * ca * cb)                                     # the radius of each proposal
                target = math.ceil(K * (c + 1) / NCYC)
                for b in range(B):
                    need = target - com[b].sum()
                    if need <= 0: continue
                    cand = [k for k in np.argsort(-conf[b]) if not com[b, k]][:need]
                    for k in cand:
                        com[b, k] = True; typ[b, k] = ty[b, k]; rad[b, k] = conf[b, k]; IA[b, k], IB[b, k] = ia[b, k], ib[b, k]; cyc[b, k] = c
                        op = TYPES[ty[b, k]][0] if ty[b, k] != STOP else "."
                        progs[b][k] = (op, dec(ia[b, k]), dec(ib[b, k])) if op != "." else (".", None, None)
            gprogs, gt = gold(E)
            for b, (it, e) in enumerate(zip(part, E)):
                prog = []
                for st in progs[b]:
                    if st[0] == ".": break
                    prog.append(st)
                v = GD.execute(prog, it["vals"] + [None] * (L - len(it["vals"])))
                res["ans"].append(v is not None and GD.V1.near(v, it["final"]))
                mm = e["m"]; okall = len(prog) == mm
                for k in range(mm):
                    o = progs[b][k][0] == gprogs[b][k][0]
                    ptk = bool(t["va"][b, k, IA[b, k]] and t["vb"][b, k, IB[b, k]])
                    res["step_op"].append(o); res["step_ptr"].append(ptk); res["step_all"].append(o and ptk)
                    res["type_ok"].append(typ[b, k] == gt[b, k]); res["commit_cycle"].append(int(cyc[b, k]))
                    okall &= o and ptk
                    if k == 0: res["first_all"].append(o and ptk)
                    for V, I in ((t["va"], IA), (t["vb"], IB)):
                        Vb = V[b, k].numpy()
                        if Vb[L:L + K].any() and not Vb[:L].any(): res["num_for_slot"].append(bool(I[b, k] < L))
                res["prog"].append(bool(okall))
    return {k: np.asarray(v) for k, v in res.items()}


if __name__ == "__main__":
    import torch
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "6"))); torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    Dt, states = TR.load(); tree = TR.load_tree(); train = Dt["TRAIN"]
    net = make_net(); opt = torch.optim.AdamW(net.parameters(), lr=LR); npar = sum(p.numel() for p in net.parameters())
    print(f"[bt] {TAG} D {D} reader {NLE} + breath {NLB} (weight-tied x {NCYC} cycles) silhouette {DB} atoms {NT} | params {npar}", flush=True)
    t0 = time.time(); curve = []
    if LOAD: net.load_state_dict(torch.load(LOAD)); STEPS = 0; print(f"[bt] {TAG} loaded {LOAD} (oracle type {ORACLE})", flush=True)
    for step in range(1, STEPS + 1):
        its = [train[int(i)] for i in rng.integers(0, len(train), BATCH)]
        lo = train_step(net, opt, its, states, tree, rng)
        if step % 1000 == 0 or step == STEPS:
            net.eval(); r = {k: score(net, Dt[k][:600], states, tree) for k in ("ORIG", "TEST")}; net.train()
            curve.append(dict(step=step, loss=lo, **{f"{k}_{m}": float(v[m].mean()) for k, v in r.items() for m in ("ans", "step_all")}))
            print(f"[bt] {TAG} step {step} loss {lo:.4f} " + " ".join(f"{k} ans {v['ans'].mean():.3f} step {v['step_all'].mean():.3f} "
                  f"type {v['type_ok'].mean():.3f}" for k, v in r.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
    if os.environ.get("BT_SAVE") == "1": torch.save(net.state_dict(), f".cache/gsmb_breath_{TAG}.pt")
    net.eval(); R = {k: score(net, Dt[k], states, tree) for k in SPLITS}
    out = dict(tag=TAG, arm="BT", seed=SEED, d=D, params=npar, steps=STEPS, curve=curve, oracle=ORACLE, load=LOAD,
               **{f"{k}_{m}": float(v[m].mean()) for k, v in R.items() for m in v if m != "commit_cycle"},
               **{f"{k}_commit_share": np.bincount(v["commit_cycle"][v["commit_cycle"] >= 0], minlength=NCYC).tolist() for k, v in R.items()},
               **{f"{k}_per": [int(x) for x in v["ans"]] for k, v in R.items()},
               atoms_C=net.C.tolist(), atoms_n=net.Cn.tolist())
    json.dump(out, open(f".cache/gsmb_breath_{TAG}.json", "w"))
    print(f"[bt] {TAG} FINAL " + " | ".join(f"{k} ans {v['ans'].mean():.3f} step {v['step_all'].mean():.3f} op {v['step_op'].mean():.3f} "
          f"ptr {v['step_ptr'].mean():.3f} type {v['type_ok'].mean():.3f}" for k, v in R.items()), flush=True)
