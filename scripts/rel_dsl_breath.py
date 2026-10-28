"""rel_dsl_breath.py -- THE RELATION-DSL TASK, the breathing transformer behind the bottleneck (ledger 2026-10-28, word given; CPU
torch). Input = the GOLD factor graph as generic symbols (scripts/rel_dsl_census.py rows): every var is an anonymous symbol drawn
from a learned table by a fresh random assignment each time it is read (entity1, entity2 ... carry no meaning), given values are
WITHHELD, tokens are a set (no position). Var tokens = symbol + given/query flags; factor tokens = op + W_arg(sum of arg symbols)
+ W_res(result symbol) (+ W_min(first arg) for sub). Output = for every derived var, WHICH incident factor determines it (pointer
over the factors containing it). Graded by the EXECUTOR (rel_dsl_census.execute, exact Fractions, gold values): the plan must be
acyclic, each step solvable when it fires, and the query value = the custody key. Any valid plan scores; the loss is set-valued
(logsumexp over the factors that determine the var at its optimal round).
ARMS (the eq_slot forms):
  NOMLP    ONE shared block (self-attn over vars+factors, NO feed-forward) applied NB times, a T6 breath stamp each breath, and
           THE SNAP: a straight-through one-hot of each var's pointer argmax re-injects the chosen factor (Bryce's form)
  BREATH   NOMLP + the block's feed-forward
  VANILLA  NB DISTINCT blocks with feed-forward, no stamp, no snap (the compute-matched standard transformer)
SPLITS: VAL = the train-fixture STRUCTURES with sha1(sdig) % 5 == 0 (structure-disjoint; one item per structure, its first row);
TRAIN = the rest, one item per structure, capped to the first RD_COV share of structures in sha1 order (nested: 0.125 c 0.25 c 1).
HOLDOUT = the wild fixture (311 rows, per row), read split by structure seen / unseen in the full train fixture.
Fixed step count, final checkpoint (no selection). Each batch is trimmed to its widest item (pads are masked; same math). Envs RD_ARM, RD_SEED, RD_COV, RD_STEPS, RD_NB, RD_D, RD_BATCH, RD_LR, RD_TAG.
Output .cache/rel_dsl_{TAG}.json."""
import hashlib, json, math, os, sys, time
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
import rel_dsl_census as RC
from mycelium import rotor_clock as CK

ARM = os.environ.get("RD_ARM", "NOMLP"); SEED = int(os.environ.get("RD_SEED", "0")); COV = float(os.environ.get("RD_COV", "1.0"))
STEPS = int(os.environ.get("RD_STEPS", "8000")); NB = int(os.environ.get("RD_NB", "8")); D = int(os.environ.get("RD_D", "256"))
BATCH = int(os.environ.get("RD_BATCH", "64")); LR = float(os.environ.get("RD_LR", "3e-4")); NH = 8
TAG = os.environ.get("RD_TAG", f"{ARM}_c{COV:g}_s{SEED}")
VMAX, FMAX, NID = 24, 24, 32
OPS = {"given": 0, "add": 1, "mul": 2, "sub": 3}
CK.check_range(CK.T6_SPEEDS, NB + 1)


def h(s):
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def load():
    tr, ho = [], []
    for l in open(".cache/rel_dsl_rows.jsonl"):
        t = json.loads(l); t["ok"] = {int(k): v for k, v in t["ok"].items()}; t["gold"] = Fraction(t["gold"])
        (tr if t["split"] == "train" else ho).append(t)
    first = {}
    for t in tr: first.setdefault(t["sdig"], t)
    items = sorted(first.values(), key=lambda t: h(t["sdig"]))
    val = [t for t in items if h(t["sdig"]) % 5 == 0]; rest = [t for t in items if h(t["sdig"]) % 5 != 0]
    train = rest[:max(1, int(round(COV * len(rest))))]
    seen = set(first)
    for t in ho: t["seen"] = t["sdig"] in seen
    return train, val, ho


def encode(t):
    """-> arrays for one item: var flags, factor ops, arg/res/min incidence (factor x var counts), pointer mask, targets."""
    vf = np.zeros(VMAX, np.int64); vm = np.zeros(VMAX, bool); fo = np.zeros(FMAX, np.int64); fm = np.zeros(FMAX, bool)
    A = np.zeros((FMAX, VMAX), np.float32); Rz = np.zeros((FMAX, VMAX), np.float32); Mz = np.zeros((FMAX, VMAX), np.float32)
    for i, f in enumerate(t["facs"]):
        fm[i] = True; fo[i] = OPS["given"] if f["ftype"] == "given" else OPS[f["op"]]
        if f["ftype"] == "given": Rz[i, f["var"]] = 1; vf[f["var"]] |= 1; vm[f["var"]] = True; continue
        for a in f["args"]: A[i, a] += 1; vm[a] = True
        Rz[i, f["result"]] = 1; vm[f["result"]] = True
        if f["op"] == "sub": Mz[i, f["args"][0]] = 1
    vf[t["query"]] |= 2; vm[t["query"]] = True
    pm = np.zeros((VMAX, FMAX), bool); tg = np.zeros((VMAX, FMAX), bool); dv = np.zeros(VMAX, bool)
    for v in t["der"]:
        dv[v] = True
        for i, f in enumerate(t["facs"]):
            if f["ftype"] == "rel" and v in f["args"] + [f["result"]]: pm[v, i] = True
        for i in t["ok"][v]: tg[v, i] = True
    return vf, vm, fo, fm, A, Rz, Mz, pm, tg, dv


def batch(items, rng):
    import torch
    E = [encode(t) for t in items]
    ids = torch.as_tensor(np.stack([rng.permutation(NID)[:VMAX] for _ in items]))   # fresh anonymous symbols per read
    nv = int(max(np.nonzero(e[1])[0].max() for e in E)) + 1; nf = max(len(t["facs"]) for t in items)   # trim to the batch's
    cut = ((slice(None, nv),), (slice(None, nv),), (slice(None, nf),), (slice(None, nf),), (slice(None, nf), slice(None, nv)),   # widest
           (slice(None, nf), slice(None, nv)), (slice(None, nf), slice(None, nv)), (slice(None, nv), slice(None, nf)),   # (pads are
           (slice(None, nv), slice(None, nf)), (slice(None, nv),))                                                       # masked)
    out = [torch.as_tensor(np.stack([e[k][cut[k]] for e in E])) for k in range(10)]
    return out, ids[:, :nv]


def make_net():
    import torch, torch.nn as nn, torch.nn.functional as F

    class Block(nn.Module):
        def __init__(s, ff):
            super().__init__()
            s.ln1 = nn.LayerNorm(D); s.att = nn.MultiheadAttention(D, NH, batch_first=True)
            s.ff = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D)) if ff else None

        def forward(s, x, pad):
            y = s.ln1(x); x = x + s.att(y, y, y, key_padding_mask=pad, need_weights=False)[0]
            return x + s.ff(x) if s.ff is not None else x

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.sym = nn.Embedding(NID, D); s.op = nn.Embedding(4, D); s.flag = nn.Embedding(4, D); s.kind = nn.Embedding(2, D)
            s.w_arg = nn.Linear(D, D, bias=False); s.w_res = nn.Linear(D, D, bias=False); s.w_min = nn.Linear(D, D, bias=False)
            s.shared = ARM in ("NOMLP", "BREATH")
            s.blocks = nn.ModuleList([Block(ARM != "NOMLP")] if s.shared else [Block(True) for _ in range(NB)])
            if s.shared:
                ph = np.stack([CK.t6_phases(b, 6) for b in range(NB)])
                s.register_buffer("bst", torch.as_tensor(np.concatenate([np.cos(ph), np.sin(ph)], -1), dtype=torch.float32))
                s.bproj = nn.Linear(12, D); s.snap = nn.Linear(D, D)
            s.ln = nn.LayerNorm(D); s.pq = nn.Linear(D, D); s.pk = nn.Linear(D, D)

        def forward(s, bt, ids):
            vf, vm, fo, fm, A, Rz, Mz, pm, tg, dv = bt
            S = s.sym(ids)                                                  # B x V x D anonymous symbols
            V = S + s.flag(vf) + s.kind.weight[0]
            Fz = s.op(fo) + s.w_arg(A @ S) + s.w_res(Rz @ S) + s.w_min(Mz @ S) + s.kind.weight[1]
            x = torch.cat([V, Fz], 1); pad = ~torch.cat([vm, fm], 1); nv = V.shape[1]
            out = []
            for b in range(NB):
                if s.shared: x = x + s.bproj(s.bst[b])[None, None]
                x = s.blocks[0 if s.shared else b](x, pad)
                z = s.ln(x); lg = torch.einsum("bvd,bfd->bvf", s.pq(z[:, :nv]), s.pk(z[:, nv:])) / math.sqrt(D)
                lg = lg.masked_fill(~pm, -1e4); out.append(lg)
                if s.shared:                                                # THE SNAP: commit each var's chosen factor
                    p = lg.softmax(-1); oh = F.one_hot(p.argmax(-1), lg.shape[-1]).float(); st = oh - p.detach() + p
                    st = st * dv[..., None]
                    x = torch.cat([x[:, :nv] + s.snap(st @ x[:, nv:]), x[:, nv:]], 1)
            return torch.stack(out, 1)                                      # B x NB x V x F
    return Net()


def loss_fn(lg, bt):
    pm, tg, dv = bt[7], bt[8], bt[9]
    num = lg.masked_fill(~tg[:, None], -1e4).logsumexp(-1); den = lg.logsumexp(-1)
    l = (den - num) * dv[:, None]
    return l.sum() / (dv.sum() * lg.shape[1])


def score(net, items, seed=0):
    import torch
    rng = np.random.default_rng(10_000 + seed); ok = []
    with torch.no_grad():
        for s0 in range(0, len(items), 256):
            its = items[s0:s0 + 256]; bt, ids = batch(its, rng); lg = net(bt, ids)[:, -1]
            for k, t in enumerate(its):
                plan = {v: int(lg[k, v].argmax()) for v in t["der"]}
                ok.append(RC.execute(t["facs"], plan, t["query"]) == t["gold"])
    return np.asarray(ok)


if __name__ == "__main__":
    import torch
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "16"))); torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    train, val, ho = load(); net = make_net(); opt = torch.optim.AdamW(net.parameters(), lr=LR)
    npar = sum(p.numel() for p in net.parameters())
    res_val = np.mean([RC.execute(t["facs"], RC.result_plan(t), t["query"]) == t["gold"] for t in val])
    print(f"[rd] {TAG} arm {ARM} cov {COV} | train structures {len(train)} | val {len(val)} | holdout {len(ho)} (unseen "
          f"{sum(not t['seen'] for t in ho)}) | params {npar} | RESULT baseline val {res_val:.3f}", flush=True)
    t0 = time.time(); curve = []
    for step in range(1, STEPS + 1):
        its = [train[i] for i in rng.integers(0, len(train), BATCH)]
        bt, ids = batch(its, rng); loss = loss_fn(net(bt, ids), bt)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if step % 2000 == 0 or step == STEPS:
            net.eval(); v = score(net, val).mean(); tr_s = score(net, train[:1000]).mean(); net.train()
            curve.append(dict(step=step, loss=float(loss), val=float(v), train=float(tr_s)))
            print(f"[rd] {TAG} step {step} loss {float(loss):.4f} train {tr_s:.3f} VAL {v:.3f} ({time.time() - t0:.0f}s)", flush=True)
    net.eval(); sv = score(net, val); sh = score(net, ho)
    un = np.asarray([not t["seen"] for t in ho])
    json.dump(dict(tag=TAG, arm=ARM, seed=SEED, cov=COV, n_train=len(train), params=npar, steps=STEPS, curve=curve,
                   val=float(sv.mean()), val_per=[int(x) for x in sv], val_sdig=[t["sdig"] for t in val],
                   hold=float(sh.mean()), hold_seen=float(sh[~un].mean()), hold_unseen=float(sh[un].mean()), hold_per=[int(x) for x in sh],
                   result_val=float(res_val)), open(f".cache/rel_dsl_{TAG}.json", "w"))
    print(f"[rd] {TAG} FINAL VAL {sv.mean():.3f} | holdout {sh.mean():.3f} (seen {sh[~un].mean():.3f} / unseen {sh[un].mean():.3f})", flush=True)
