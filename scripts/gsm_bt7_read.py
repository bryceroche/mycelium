"""BT7a THE BELIEF-PROPAGATION DIAL (ledger 2026-11-02 20:31): a zero-training read on banked BT weights.

Run with the weights' own BT_* env and BT_LOAD=.cache/gsmb_breath_{TAG}.pt. Per weights file:
  1. fit rho0 on TRAIN rows (Dt["TRAIN"][:600], the BT6 grid) for the BT6 gate AND for the BT7a gate (separately)
  2. score ORIG and TEST under both on the same weights -> .cache/gsmb_bt7_{TAG}.json + summary lines
The factor graph per cycle (3 mean-field rounds, unit temperature, no tunables):
  type<->pointer  A_k(t) = sum_role log P(pointer kind = t's kind | the pointer head under t with every kind open)
  prefix          slot k non-STOP needs k-1 non-STOP; slot k STOP needs k+1 STOP   (messages q_j = belief(j non-STOP))
  slot-ref        a pointer to an earlier slot j is weighted by q_j (log q_j on that column)
Committed slots are clamped (one-hot type, q = 1 / 0). Verdict: scripts/gsm_bt7_verdict.py.
"""
import json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import gsm_breath_reader as BR

GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
SMOKE = int(os.environ.get("BT7_SMOKE", "0"))   # >0: that many items per split, one grid point, no JSON
ROUNDS, EPS = 3, 1e-6
FACT = os.environ.get("BT7_FACTORS", "A,P").split(",")   # attribution reads only; the pinned form = both
TAG = os.path.basename(BR.LOAD).replace("gsmb_breath_", "").replace(".pt", "")
assert BR.LOAD and os.path.exists(BR.LOAD) and not BR.GATE and not BR.ORACLE
torch.set_num_threads(int(os.environ.get("TOY_THREADS", "3")))
Dt, states = BR.TR.load(); tree = BR.TR.load_tree()
if BR.ROLE: BR.RF = np.load(BR.GD.V1.C(BR.rolef_name()), mmap_mode="r")
net = BR.make_net(); net.load_state_dict(torch.load(BR.LOAD)); net.eval()
K, NT, STOP, NCYC = BR.K, BR.NT, BR.STOP, BR.NCYC
KIND = np.array([[BR.KINDS.index(a), BR.KINDS.index(b)] for _, a, b in BR.TYPES])     # (NT-1, 2) the kind each type claims per role


def kind_logmass(lg, Lq, L, slot_bias):
    """lg (B, K, L+K+NC) free pointer logits; slot_bias (B, K) = log q_j on slot columns -> (B, K, 3) log P(kind)"""
    lg = lg.clone(); lg[..., L:L + K] = lg[..., L:L + K] + slot_bias[:, None, :]
    lp = lg.log_softmax(-1)
    return torch.stack([lp[..., :Lq].logsumexp(-1), lp[..., L:L + K].logsumexp(-1), lp[..., L + K:].logsumexp(-1)], -1)


def bp_beliefs(p, free, Lq, L, com, typ):
    """p (B, K, NT) type probs; free[t] = (pa, pb) under type t with every kind open -> beliefs (B, K, NT), q (B, K)"""
    B = p.shape[0]; comt = torch.as_tensor(com); typt = torch.as_tensor(typ)
    clamp = torch.nn.functional.one_hot(typt, NT).float()
    b = torch.where(comt[..., None], clamp, p)
    lp = torch.log(p.clamp(min=EPS))
    for _ in range(ROUNDS):
        q = (1 - b[..., STOP]).clamp(EPS, 1 - EPS)                                            # (B, K) P(slot non-STOP)
        sb = torch.log(q)
        A = torch.zeros(B, K, NT)
        for t in range(NT - 1):
            for role in (0, 1):
                A[..., t] += kind_logmass(free[t][role], Lq, L, sb)[..., KIND[t, role]]
        prev = torch.cat([torch.ones(B, 1), q[:, :-1]], 1)                                    # slot 0 has no predecessor
        nxt_stop = torch.cat([1 - q[:, 1:], torch.ones(B, 1)], 1)                             # the last slot has no successor
        lb = lp + (A if "A" in FACT else 0)
        if "P" in FACT:
            lb[..., :STOP] = lb[..., :STOP] + torch.log(prev.clamp(min=EPS))[..., None]
            lb[..., STOP] = lb[..., STOP] + torch.log(nxt_stop.clamp(min=EPS))
        b = torch.where(comt[..., None], clamp, lb.softmax(-1))
    return b, (1 - b[..., STOP]).clamp(EPS, 1 - EPS)


def score(its, rho, bp):
    """BR.score with the BT6 gate (bp False; must reproduce BT6) or the BT7a beliefs + gate (bp True)"""
    res = dict(ans=[], step_all=[], step_op=[], type_ok=[], commit_cycle=[], op_right_type_wrong=[])
    with torch.no_grad():
        for s0 in range(0, len(its), 128):
            part = its[s0:s0 + 128]
            t, E = BR.batch(part, states, tree, np.random.default_rng(s0)); L = t["pad"].shape[1]; B = len(part); Lq = L - K
            for b, it in enumerate(part): E[b]["vals"] = it["vals"]
            xq = net.read(t)
            com = np.zeros((B, K), bool); typ = np.zeros((B, K), np.int64); rad = np.zeros((B, K), np.float32)
            IA = np.zeros((B, K), np.int64); IB = np.zeros((B, K), np.int64); cyc = np.full((B, K), -1)
            progs = [[(".", None, None)] * K for _ in range(B)]
            dec = lambda i: ("t", int(i)) if i < L else (("s", int(i - L)) if i < L + K else ("c", int(i - L - K)))
            sp = None
            for c in range(NCYC):
                nb = BR.nb_tensors(BR.notebook(E, L, progs, com, typ, rad))
                logit, pa, pb, _, p = net.breath(t, xq, nb, None, torch.full((B,), c), sp)
                if BR.SOFT and c + 1 < NCYC: sp = (logit / BR.TAU(torch.full((B,), c + 1))[:, None, None]).softmax(-1)
                if bp:
                    free = [net.ptrs(torch.full((B, K), tt, dtype=torch.long), free=True) for tt in range(NT - 1)]
                    bel, q = bp_beliefs(p, free, Lq, L, com, typ)
                    ty = bel.argmax(-1)
                    pa, pb = net.ptrs(ty)
                    sl = torch.log(q)[:, None, :]                                                # slot-ref: weight earlier slots by q_j
                    pa = pa.clone(); pb = pb.clone(); pa[..., L:L + K] += sl; pb[..., L:L + K] += sl
                    pt = bel.gather(-1, ty[..., None])[..., 0].numpy(); ty = ty.numpy()
                else:
                    ty = logit.argmax(-1).numpy(); pt = p.max(-1).values.numpy()
                ia, ib = pa.argmax(-1).numpy(), pb.argmax(-1).numpy()
                ca = pa.softmax(-1).max(-1).values.numpy(); cb = pb.softmax(-1).max(-1).values.numpy()
                conf = np.where(ty == STOP, pt, pt * ca * cb)
                for b in range(B):
                    for k in [k for k in np.argsort(-conf[b]) if not com[b, k] and conf[b, k] >= rho * (1 - c / (NCYC - 1))]:
                        com[b, k] = True; typ[b, k] = ty[b, k]; rad[b, k] = conf[b, k]; IA[b, k], IB[b, k] = ia[b, k], ib[b, k]; cyc[b, k] = c
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
                    o = progs[b][k][0] == gprogs[b][k][0]
                    ptk = bool(t["va"][b, k, IA[b, k]] and t["vb"][b, k, IB[b, k]])
                    res["step_op"].append(o); res["step_all"].append(o and ptk); res["type_ok"].append(typ[b, k] == gt[b, k])
                    res["commit_cycle"].append(int(cyc[b, k])); res["op_right_type_wrong"].append(o and typ[b, k] != gt[b, k])
    return {k: np.asarray(v) for k, v in res.items()}


out = dict(tag=TAG, load=BR.LOAD, soft=BR.SOFT, grid=GRID, rounds=ROUNDS)
for arm, bp in (("bt6", False), ("bt7", True)):
    fit = {r: float(score(Dt["TRAIN"][:SMOKE or 600], r, bp)["step_all"].mean()) for r in (GRID[-2:] if SMOKE else GRID)}
    out[arm] = dict(fit=fit, rho0=max(fit, key=lambda r: fit[r]))
    print(f"[bt7] {TAG} {arm} TRAIN fit " + " ".join(f"{r}:{v:.3f}" for r, v in fit.items()) + f" -> rho0 {out[arm]['rho0']}", flush=True)
for split in ("ORIG", "TEST"):
    R = {arm: score(Dt[split][:SMOKE or None], out[arm]["rho0"], arm == "bt7") for arm in ("bt6", "bt7")}
    sh = lambda X: np.bincount(X["commit_cycle"][X["commit_cycle"] >= 0], minlength=NCYC).tolist()
    d = {m: R["bt7"][m].astype(float) - R["bt6"][m].astype(float) for m in ("step_all", "type_ok")}
    out[split] = {**{arm: dict(**{m: float(X[m].mean()) for m in ("ans", "step_all", "step_op", "type_ok", "op_right_type_wrong")}, share=sh(X))
                     for arm, X in R.items()},
                  "diff_step": float(d["step_all"].mean()), "se_step": float(d["step_all"].std(ddof=1) / math.sqrt(len(d["step_all"]))),
                  "diff_type": float(d["type_ok"].mean()), "se_type": float(d["type_ok"].std(ddof=1) / math.sqrt(len(d["type_ok"])))}
    o = out[split]
    print(f"[bt7] {TAG} {split} step {o['bt6']['step_all']:.4f} -> {o['bt7']['step_all']:.4f} ({o['diff_step']:+.4f}, SE {o['se_step']:.4f}) | "
          f"type {o['bt6']['type_ok']:.4f} -> {o['bt7']['type_ok']:.4f} ({o['diff_type']:+.4f}, SE {o['se_type']:.4f}) | op {o['bt6']['step_op']:.3f} -> "
          f"{o['bt7']['step_op']:.3f} | op-right-type-wrong {o['bt6']['op_right_type_wrong']:.3f} -> {o['bt7']['op_right_type_wrong']:.3f} | "
          f"ans {o['bt6']['ans']:.3f} -> {o['bt7']['ans']:.3f} | share {o['bt6']['share']} -> {o['bt7']['share']}", flush=True)
if not SMOKE and FACT == ["A", "P"]: json.dump(out, open(f".cache/gsmb_bt7_{TAG}.json", "w"), indent=1)
