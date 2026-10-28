"""rel_dsl_phase.py -- THE PHASE-BINDING ARM on the relation-DSL task (ledger 2026-10-28, Bryce: "word given for the phase-binding arm").
PHASE = NOMLP (rel_dsl_breath.py: one shared block, no feed-forward, T6 breath stamp, the snap) with the LINEAR role binding swapped for
the fixed algebra (a separate file: the running grid re-reads rel_dsl_breath.py, which stays untouched):
  SYMBOLS  each anonymous symbol is a learned PHASE vector theta in R^P (P = RD_PB planes, default 256); fresh random assignment per read
  BIND     a factor is the phasor sum z_f = sum_roles sum_v count_r(f,v) * exp(i(theta_v + rho_r)), roles r = arg / result / minuend
           (rho_r learned role phases); the factor token = op + Linear([Re z, Im z]); a var token = Linear([cos theta, sin theta]) + flags
  UNBIND   u_r(v,f) = mean_p Re( z_f * conj(exp(i theta_v)) * conj(exp(i rho_r)) ) ~ count_r(f,v) + crosstalk ~ 1/sqrt(P): the role
           reading of every (var, factor) pair, by conjugate unbind, enters every breath's attention as a per-head additive bias
           (var->factor and factor->var each by its own Linear(3, heads)); the mandatory road (no gain)
Everything else (data, splits, loss, executor grading, steps, D, NB, lr) is rel_dsl_breath.py's. Output .cache/rel_dsl_{TAG}.json."""
import json, math, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
import rel_dsl_breath as RB
from mycelium import rotor_clock as CK

PB = int(os.environ.get("RD_PB", "256"))
assert RB.ARM == "PHASE", "run with RD_ARM=PHASE"


def make_net():
    import torch, torch.nn as nn, torch.nn.functional as F
    D, NH, NB = RB.D, RB.NH, RB.NB

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

        def forward(s, bt, ids):
            vf, vm, fo, fm, A, Rz, Mz, pm, tg, dv = bt
            th = s.theta[ids]; E = torch.polar(torch.ones_like(th), th)                     # B x V x P unit phasors
            R = torch.polar(torch.ones_like(s.rho), s.rho)                                  # 3 x P role phasors
            C = [A, Rz, Mz]
            z = sum(torch.einsum("bfv,bvp->bfp", c.to(E.dtype), E) * R[r] for r, c in enumerate(C))   # BIND
            u = torch.stack([torch.einsum("bfp,bvp->bvf", z * R[r].conj(), E.conj()).real / PB for r in range(3)], -1)  # UNBIND
            V = s.vin(torch.cat([E.real, E.imag], -1)) + s.flag(vf) + s.kind.weight[0]
            Fz = s.fin(torch.cat([z.real, z.imag], -1)) + s.op(fo) + s.kind.weight[1]
            x = torch.cat([V, Fz], 1); nv, nf = V.shape[1], Fz.shape[1]; B = x.shape[0]; L = nv + nf
            bias = x.new_zeros(B, NH, L, L)
            bias[:, :, :nv, nv:] = s.vf_bias(u).permute(0, 3, 1, 2); bias[:, :, nv:, :nv] = s.fv_bias(u).permute(0, 3, 2, 1)
            pad = ~torch.cat([vm, fm], 1)
            mask = bias.masked_fill(pad[:, None, None, :], -1e4).reshape(B * NH, L, L)
            out = []
            for b in range(NB):
                x = x + s.bproj(s.bst[b])[None, None]
                x = s.blocks[0](x, mask)
                zz = s.ln(x); lg = torch.einsum("bvd,bfd->bvf", s.pq(zz[:, :nv]), s.pk(zz[:, nv:])) / math.sqrt(D)
                lg = lg.masked_fill(~pm, -1e4); out.append(lg)
                p = lg.softmax(-1); oh = F.one_hot(p.argmax(-1), nf).float(); st = (oh - p.detach() + p) * dv[..., None]
                x = torch.cat([x[:, :nv] + s.snap(st @ x[:, nv:]), x[:, nv:]], 1)      # THE SNAP
            return torch.stack(out, 1)
    return Net()


if __name__ == "__main__":
    import torch
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "16"))); torch.manual_seed(RB.SEED); rng = np.random.default_rng(RB.SEED)
    train, val, ho = RB.load(); net = make_net(); opt = torch.optim.AdamW(net.parameters(), lr=RB.LR)
    npar = sum(p.numel() for p in net.parameters()); TAG = RB.TAG
    res_val = np.mean([RB.RC.execute(t["facs"], RB.RC.result_plan(t), t["query"]) == t["gold"] for t in val])
    print(f"[rd] {TAG} arm PHASE P {PB} cov {RB.COV} | train structures {len(train)} | val {len(val)} | params {npar} | RESULT val {res_val:.3f}", flush=True)
    t0 = time.time(); curve = []
    for step in range(1, RB.STEPS + 1):
        its = [train[i] for i in rng.integers(0, len(train), RB.BATCH)]
        bt, ids = RB.batch(its, rng); loss = RB.loss_fn(net(bt, ids), bt)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if step % 2000 == 0 or step == RB.STEPS:
            net.eval(); v = RB.score(net, val).mean(); tr_s = RB.score(net, train[:1000]).mean(); net.train()
            curve.append(dict(step=step, loss=float(loss.detach()), val=float(v), train=float(tr_s)))
            print(f"[rd] {TAG} step {step} loss {float(loss.detach()):.4f} train {tr_s:.3f} VAL {v:.3f} ({time.time() - t0:.0f}s)", flush=True)
    net.eval(); sv = RB.score(net, val); sh = RB.score(net, ho); un = np.asarray([not t["seen"] for t in ho])
    json.dump(dict(tag=TAG, arm="PHASE", pb=PB, seed=RB.SEED, cov=RB.COV, n_train=len(train), params=npar, steps=RB.STEPS, curve=curve,
                   val=float(sv.mean()), val_per=[int(x) for x in sv], val_sdig=[t["sdig"] for t in val],
                   hold=float(sh.mean()), hold_seen=float(sh[~un].mean()), hold_unseen=float(sh[un].mean()), hold_per=[int(x) for x in sh],
                   result_val=float(res_val)), open(f".cache/rel_dsl_{TAG}.json", "w"))
    print(f"[rd] {TAG} FINAL VAL {sv.mean():.3f} | holdout {sh.mean():.3f} (seen {sh[~un].mean():.3f} / unseen {sh[un].mean():.3f})", flush=True)
