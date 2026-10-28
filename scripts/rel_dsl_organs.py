"""rel_dsl_organs.py -- BRINGING THE BREATHING TRANSFORMER ALIVE, organ by organ, on the relation-DSL task (ledger 2026-10-28; Bryce:
"word given for the breathing transformer -- lets bring it alive"; definition docs/breathing_transformer.md). CPU torch.
THE DEPTH SPLIT (RD_SPLIT=depth): TRAIN = train-fixture structures outside VAL with depth to query <= 3 (one item per structure);
SHALLOW = VAL structures depth <= 3 (unseen, in-range); MID = all structures depth 4 (read); DEEP = all structures depth 5..8 (never
trained at that depth: the chaining test). Grading, loss, data encoding, steps: rel_dsl_breath.py's.
ARMS (RD_ARM):
  VANILLA / NOMLP   rel_dsl_breath.make_net, unchanged
  PHASE             rel_dsl_phase's arm (fixed bind + conjugate-unbind attention bias), rebuilt here; with every organ off it is
                    bit-identical to rel_dsl_phase.make_net (checked)
  NOTEBOOK          PHASE + THE NOTEBOOK (organ 2): the executor in the loop. Every breath, each derived var's CHOSEN factor fires
                    iff all its other vars are already KNOWN (givens known at breath 0); a var that fires becomes known. The known
                    flag (and the breath it became known, as a T6 stamp) is written back into the var's token before the next
                    breath: neural picks, the fixed algebra marks what is now derived, the result returns as state. A road: the
                    flags replace nothing, they are added every breath with no gain. Known-ness is structural (which vars are
                    derived); values stay with the final executor. Non-differentiable by design (the notebook is a record).
Output .cache/rel_dsl_{TAG}.json (TAG default {ARM}_depth_s{SEED})."""
import json, math, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
import rel_dsl_breath as RB
from mycelium import rotor_clock as CK

ARM = RB.ARM; PB = int(os.environ.get("RD_PB", "256"))
TAG = os.environ.get("RD_TAG", f"{ARM}_depth_s{RB.SEED}")


def load_depth():
    tr, _ = [], None
    first = {}
    for l in open(".cache/rel_dsl_rows.jsonl"):
        t = json.loads(l)
        if t["split"] != "train": continue
        t["ok"] = {int(k): v for k, v in t["ok"].items()}; t["gold"] = RB.Fraction(t["gold"]); first.setdefault(t["sdig"], t)
    items = sorted(first.values(), key=lambda t: RB.h(t["sdig"]))
    isval = lambda t: RB.h(t["sdig"]) % 5 == 0
    train = [t for t in items if not isval(t) and t["depth"] <= 3]
    shallow = [t for t in items if isval(t) and t["depth"] <= 3]
    mid = [t for t in items if t["depth"] == 4]
    deep = [t for t in items if 5 <= t["depth"] <= 8]
    return train, shallow, mid, deep


def make_net():
    if ARM in ("VANILLA", "NOMLP"): return RB.make_net()
    import torch, torch.nn as nn, torch.nn.functional as F
    D, NH, NB = RB.D, RB.NH, RB.NB
    NOTE = ARM == "NOTEBOOK"

    class Block(nn.Module):
        def __init__(s):
            super().__init__()
            s.ln1 = nn.LayerNorm(D); s.att = nn.MultiheadAttention(D, NH, batch_first=True)

        def forward(s, x, mask):
            y = s.ln1(x); return x + s.att(y, y, y, attn_mask=mask, need_weights=False)[0]

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.theta = nn.Parameter(torch.rand(RB.NID, PB) * 2 * math.pi); s.rho = nn.Parameter(torch.rand(3, PB) * 2 * math.pi)
            s.vin = nn.Linear(2 * PB, D); s.fin = nn.Linear(2 * PB, D)
            s.op = nn.Embedding(4, D); s.flag = nn.Embedding(4, D); s.kind = nn.Embedding(2, D)
            s.vf_bias = nn.Linear(3, NH); s.fv_bias = nn.Linear(3, NH)
            s.blocks = nn.ModuleList([Block()])
            ph = np.stack([CK.t6_phases(b, 6) for b in range(NB)])
            s.register_buffer("bst", torch.as_tensor(np.concatenate([np.cos(ph), np.sin(ph)], -1), dtype=torch.float32))
            s.bproj = nn.Linear(12, D); s.snap = nn.Linear(D, D)
            s.ln = nn.LayerNorm(D); s.pq = nn.Linear(D, D); s.pk = nn.Linear(D, D)
            if NOTE:   # created last so the shared parameters draw the same init as rel_dsl_phase
                ph1 = np.stack([CK.t6_phases(b, 6) for b in range(NB + 1)])
                s.register_buffer("kst", torch.as_tensor(np.concatenate([np.cos(ph1), np.sin(ph1)], -1), dtype=torch.float32))
                s.known = nn.Embedding(2, D); s.kproj = nn.Linear(12, D)

        def forward(s, bt, ids):
            vf, vm, fo, fm, A, Rz, Mz, pm, tg, dv = bt
            th = s.theta[ids]; E = torch.polar(torch.ones_like(th), th)
            R = torch.polar(torch.ones_like(s.rho), s.rho)
            C = [A, Rz, Mz]
            z = sum(torch.einsum("bfv,bvp->bfp", c.to(E.dtype), E) * R[r] for r, c in enumerate(C))
            u = torch.stack([torch.einsum("bfp,bvp->bvf", z * R[r].conj(), E.conj()).real / PB for r in range(3)], -1)
            V = s.vin(torch.cat([E.real, E.imag], -1)) + s.flag(vf) + s.kind.weight[0]
            Fz = s.fin(torch.cat([z.real, z.imag], -1)) + s.op(fo) + s.kind.weight[1]
            x = torch.cat([V, Fz], 1); nv, nf = V.shape[1], Fz.shape[1]; B = x.shape[0]; L = nv + nf
            bias = x.new_zeros(B, NH, L, L)
            bias[:, :, :nv, nv:] = s.vf_bias(u).permute(0, 3, 1, 2); bias[:, :, nv:, :nv] = s.fv_bias(u).permute(0, 3, 2, 1)
            pad = ~torch.cat([vm, fm], 1)
            mask = bias.masked_fill(pad[:, None, None, :], -1e4).reshape(B * NH, L, L)
            if NOTE:
                inc = ((A + Rz + Mz) > 0).float().transpose(1, 2)                 # B x V x F incidence
                known = ((vf % 2) == 1).float()                                  # givens known at breath 0
                kt = torch.zeros_like(known, dtype=torch.long)                   # breath each var became known
            out = []
            for b in range(NB):
                x = x + s.bproj(s.bst[b])[None, None]
                if NOTE:                                                         # THE NOTEBOOK: what is derived so far, and when
                    x = torch.cat([x[:, :nv] + s.known(known.long()) + s.kproj(s.kst[kt]) * known[..., None], x[:, nv:]], 1)
                x = s.blocks[0](x, mask)
                zz = s.ln(x); lg = torch.einsum("bvd,bfd->bvf", s.pq(zz[:, :nv]), s.pk(zz[:, nv:])) / math.sqrt(D)
                lg = lg.masked_fill(~pm, -1e4); out.append(lg)
                p = lg.softmax(-1); ch = p.argmax(-1); oh = F.one_hot(ch, nf).float(); st = (oh - p.detach() + p) * dv[..., None]
                x = torch.cat([x[:, :nv] + s.snap(st @ x[:, nv:]), x[:, nv:]], 1)
                if NOTE:
                    with torch.no_grad():                                        # fire: the chosen factor's other vars all known
                        miss = torch.einsum("bvf,bv->bf", inc, 1 - known)        # unknown vars per factor
                        missc = torch.gather(miss, 1, ch) - (1 - known)          # minus the var itself
                        fire = dv & (known == 0) & (missc <= 0.5)
                        kt = torch.where(fire, torch.full_like(kt, b + 1), kt); known = torch.where(fire, torch.ones_like(known), known)
            return torch.stack(out, 1)
    return Net()


if __name__ == "__main__":
    import torch
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "16"))); torch.manual_seed(RB.SEED); rng = np.random.default_rng(RB.SEED)
    train, shallow, mid, deep = load_depth(); net = make_net(); opt = torch.optim.AdamW(net.parameters(), lr=RB.LR)
    npar = sum(p.numel() for p in net.parameters())
    rp = lambda its: float(np.mean([RB.RC.execute(t["facs"], RB.RC.result_plan(t), t["query"]) == t["gold"] for t in its]))
    print(f"[ro] {TAG} arm {ARM} | train {len(train)} (depth<=3) | shallow {len(shallow)} mid {len(mid)} deep {len(deep)} | params {npar} | "
          f"RESULT shallow {rp(shallow):.3f} mid {rp(mid):.3f} deep {rp(deep):.3f}", flush=True)
    t0 = time.time(); curve = []
    for step in range(1, RB.STEPS + 1):
        its = [train[i] for i in rng.integers(0, len(train), RB.BATCH)]
        bt, ids = RB.batch(its, rng); loss = RB.loss_fn(net(bt, ids), bt)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if step % 2000 == 0 or step == RB.STEPS:
            net.eval(); s_ = RB.score(net, shallow).mean(); d_ = RB.score(net, deep).mean(); net.train()
            curve.append(dict(step=step, loss=float(loss.detach()), shallow=float(s_), deep=float(d_)))
            print(f"[ro] {TAG} step {step} loss {float(loss.detach()):.4f} SHALLOW {s_:.3f} DEEP {d_:.3f} ({time.time() - t0:.0f}s)", flush=True)
    net.eval(); S = {k: RB.score(net, v) for k, v in (("shallow", shallow), ("mid", mid), ("deep", deep))}
    bydepth = {d: float(np.mean([o for o, t in zip(S["deep"], deep) if t["depth"] == d])) for d in range(5, 9)}
    json.dump(dict(tag=TAG, arm=ARM, seed=RB.SEED, split="depth", n_train=len(train), params=npar, steps=RB.STEPS, curve=curve,
                   **{k: float(v.mean()) for k, v in S.items()}, **{k + "_per": [int(x) for x in v] for k, v in S.items()},
                   deep_by_depth=bydepth, result=dict(shallow=rp(shallow), mid=rp(mid), deep=rp(deep))),
              open(f".cache/rel_dsl_{TAG}.json", "w"))
    print(f"[ro] {TAG} FINAL SHALLOW {S['shallow'].mean():.3f} MID {S['mid'].mean():.3f} DEEP {S['deep'].mean():.3f} | by depth "
          + " ".join(f"d{d} {v:.3f}" for d, v in bydepth.items()), flush=True)
