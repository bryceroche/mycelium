"""rel_dsl_nostamp.py -- THE NOSTAMP ABLATION (ledger 2026-10-29, Bryce: "word given for the NOSTAMP ablation"): a copy of
rel_dsl_organs.py (the running chain re-reads that file, so it stays untouched) with ONE change: arm NOSTAMP = NOTEBOOK without the
write-time T6 stamp (s.kproj(s.kst[kt]) is not added; kproj is still created so the init draws match NOTEBOOK's). The known bit stays;
the per-breath T6 clock stamp (bst) stays. Question: must the notebook be coupled with the T6 clock?
--- the original docstring follows ---
rel_dsl_organs.py -- BRINGING THE BREATHING TRANSFORMER ALIVE, organ by organ, on the relation-DSL task (ledger 2026-10-28; Bryce:
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
  ROTATE            NOTEBOOK + BREATH ROTATION (organ 7): the shared block's attention is computed by hand from the SAME
                    parameters (in_proj / out_proj); every query is rotated by the T6 gears x the current breath b, every key by
                    the gears x b too EXCEPT a known var's key, rotated by its write time (the breath it became known): the
                    var->known-fact score depends on b - write time alone (relative phase, T2u). Head planes take the T6 gears
                    cyclically (rotor_clock.t6_plane_speeds); the T6 stamp dims in the residual are never rotated.
  ROTQ              NOTEBOOK + QUERY-ONLY ROTATION (ledger 10-29, Bryce: "only rotate the queries not the entire attention space"):
                    the same hand attention, queries rotated by the T6 gears x the current breath b, keys never rotated (the
                    v109pi pi-cycled-Q form): each breath reads the keys through its own clock angle; no write-time relative phase.
  TREE              NOTEBOOK + THE TREE SNAP (organ 3; ledger 10-29, Bryce: "word given for the tree"): the snap no longer re-injects
                    the chosen factor's free, drifting state; it injects a FIXED POINT = the centroid of the step's TYPE (op of the
                    chosen factor x the var's role in it: arg / result / minuend; root -> op -> role, 12 leaves) + the factor's
                    breath-0 token (its ID: the fixed phase bind of its symbols + op, never drifted). The road has no bypass (the
                    free snap is gone). Centroids are BIRCH CFs (N, LS, SS) of the chosen factor's state at the breath a var FIRES
                    in the notebook, accumulated ONLY from chains the executor verified (final plan -> the gold answer), in
                    windows of RD_TWIN steps (the active centroid = the last closed window's mean; window 0 snaps on ID alone).
                    THE SQLITE TREE (mycelium/tree_db.py, .cache/tree_reldsl_{TAG}.db): the 12 step types are nodes; every
                    window's verified steps are a permanent batch of SYMBOLIC events (key = step type, vec = (breath, depth));
                    the model-space centroids go only to the generation-tagged centroids table (the space guard), and the final
                    scoring reloads them FROM SQLite (asserted equal to memory).
THE HARD DEPTH SPLIT (RD_SPLIT=depth2; the organs past the notebook need headroom): TRAIN = non-VAL structures depth <= 2,
capped to the first RD_COV share (default 0.25) in sha1 order; SHALLOW = VAL depth <= 2; MID = depth 3..4; DEEP = depth 5..9;
NB 10 (RD_NB) so every chain fits.
Output .cache/rel_dsl_{TAG}.json (TAG default {ARM}_{SPLIT}_s{SEED})."""
import json, math, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
import rel_dsl_breath as RB
from mycelium import rotor_clock as CK

ARM = RB.ARM; PB = int(os.environ.get("RD_PB", "256"))
SPLIT = os.environ.get("RD_SPLIT", "depth")
TAG = os.environ.get("RD_TAG", f"{ARM}_{SPLIT}_s{RB.SEED}")
TWIN = int(os.environ.get("RD_TWIN", "1000")); NTYPE = 12


def load_depth():
    tr, _ = [], None
    first = {}
    for l in open(".cache/rel_dsl_rows.jsonl"):
        t = json.loads(l)
        if t["split"] != "train": continue
        t["ok"] = {int(k): v for k, v in t["ok"].items()}; t["gold"] = RB.Fraction(t["gold"]); first.setdefault(t["sdig"], t)
    items = sorted(first.values(), key=lambda t: RB.h(t["sdig"]))
    isval = lambda t: RB.h(t["sdig"]) % 5 == 0
    if SPLIT == "depth2":
        tr = [t for t in items if not isval(t) and t["depth"] <= 2]
        cov = float(os.environ.get("RD_COV", "0.25"))
        return (tr[:max(1, int(round(cov * len(tr))))], [t for t in items if isval(t) and t["depth"] <= 2],
                [t for t in items if 3 <= t["depth"] <= 4], [t for t in items if 5 <= t["depth"] <= 9])
    train = [t for t in items if not isval(t) and t["depth"] <= 3]
    shallow = [t for t in items if isval(t) and t["depth"] <= 3]
    mid = [t for t in items if t["depth"] == 4]
    deep = [t for t in items if 5 <= t["depth"] <= 8]
    return train, shallow, mid, deep


def make_net():
    if ARM in ("VANILLA", "NOMLP"): return RB.make_net()
    import torch, torch.nn as nn, torch.nn.functional as F
    D, NH, NB = RB.D, RB.NH, RB.NB
    NOTE = ARM in ("NOTEBOOK", "ROTATE", "ROTQ", "TREE", "NOSTAMP"); ROT = ARM in ("ROTATE", "ROTQ"); TREE = ARM == "TREE"; DH = D // NH
    w = torch.as_tensor(np.radians(CK.t6_plane_speeds(DH // 2)), dtype=torch.float32)   # gear speed per head plane

    class Block(nn.Module):
        def __init__(s):
            super().__init__()
            s.ln1 = nn.LayerNorm(D); s.att = nn.MultiheadAttention(D, NH, batch_first=True)

        def forward(s, x, mask, aq=None, ak=None):
            y = s.ln1(x)
            if aq is None: return x + s.att(y, y, y, attn_mask=mask, need_weights=False)[0]
            B, L, _ = y.shape                                                   # the same attention, by hand, with rotation
            q, k, v = (t.reshape(B, L, NH, DH).transpose(1, 2) for t in F.linear(y, s.att.in_proj_weight, s.att.in_proj_bias).chunk(3, -1))
            q, k = rot(q, aq), rot(k, ak)
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.reshape(B, NH, L, L))
            return x + s.att.out_proj(o.transpose(1, 2).reshape(B, L, D))

    def rot(z, a):   # z: B x NH x L x DH (interleaved planes), a: B x L x DH/2 angles
        c, sn = a.cos()[:, None], a.sin()[:, None]; re, im = z[..., 0::2], z[..., 1::2]
        return torch.stack([re * c - im * sn, re * sn + im * c], -1).flatten(-2)

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
            if TREE:   # in-memory centroids of the step types (buffers: Welford/BIRCH, never trained)
                s.register_buffer("cent", torch.zeros(NTYPE, D)); s.collect = None

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
            if TREE:                                                             # step type of every (var, factor) pair
                role = torch.where(Rz.transpose(1, 2) > 0, 1, torch.where(Mz.transpose(1, 2) > 0, 2, 0))
                T = fo[:, None, :] * 3 + role; OH = F.one_hot(T, NTYPE).float()    # B x V x F x 12
                F0 = Fz                                                          # the factor's ID: its breath-0 token
            out = []
            for b in range(NB):
                x = x + s.bproj(s.bst[b])[None, None]
                if NOTE:                                                         # THE NOTEBOOK: what is derived so far, and when
                    x = torch.cat([x[:, :nv] + s.known(known.long()) + (0 if ARM == "NOSTAMP" else s.kproj(s.kst[kt]) * known[..., None]), x[:, nv:]], 1)
                if ROT:                                                          # queries at now; known keys at their write time
                    tk = torch.cat([torch.where(known > 0, kt, torch.full_like(kt, b)), torch.full((B, nf), b, dtype=kt.dtype)], 1)
                    ak = torch.zeros(B, L, DH // 2) if ARM == "ROTQ" else tk[..., None].float() * w
                    x = s.blocks[0](x, mask, aq=(w * b).expand(B, L, -1), ak=ak)
                else: x = s.blocks[0](x, mask)
                zz = s.ln(x); lg = torch.einsum("bvd,bfd->bvf", s.pq(zz[:, :nv]), s.pk(zz[:, nv:])) / math.sqrt(D)
                lg = lg.masked_fill(~pm, -1e4); out.append(lg)
                p = lg.softmax(-1); ch = p.argmax(-1); oh = F.one_hot(ch, nf).float(); st = (oh - p.detach() + p) * dv[..., None]
                if TREE:                                                         # THE TREE SNAP: type centroid + ID
                    xf = x[:, nv:]
                    x = torch.cat([x[:, :nv] + s.snap(torch.einsum("bvf,bvft->bvt", st, OH) @ s.cent + st @ F0), x[:, nv:]], 1)
                else: x = torch.cat([x[:, :nv] + s.snap(st @ x[:, nv:]), x[:, nv:]], 1)
                if NOTE:
                    with torch.no_grad():                                        # fire: the chosen factor's other vars all known
                        miss = torch.einsum("bvf,bv->bf", inc, 1 - known)        # unknown vars per factor
                        missc = torch.gather(miss, 1, ch) - (1 - known)          # minus the var itself
                        fire = dv & (known == 0) & (missc <= 0.5)
                        kt = torch.where(fire, torch.full_like(kt, b + 1), kt); known = torch.where(fire, torch.ones_like(known), known)
                        if TREE and s.collect is not None:                       # the step taken: type + chosen factor's state
                            s.collect.append((b, fire.clone(), torch.gather(T, 2, ch[..., None])[..., 0],
                                              torch.gather(xf, 1, ch[..., None].expand(-1, -1, D)).detach()))
            return torch.stack(out, 1)
    return Net()


if __name__ == "__main__":
    import torch
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "16"))); torch.manual_seed(RB.SEED); rng = np.random.default_rng(RB.SEED)
    train, shallow, mid, deep = load_depth(); net = make_net(); opt = torch.optim.AdamW(net.parameters(), lr=RB.LR)
    npar = sum(p.numel() for p in net.parameters())
    rp = lambda its: float(np.mean([RB.RC.execute(t["facs"], RB.RC.result_plan(t), t["query"]) == t["gold"] for t in its]))
    print(f"[ro] {TAG} arm {ARM} | train {len(train)} (depth<={2 if SPLIT == "depth2" else 3}) | shallow {len(shallow)} mid {len(mid)} deep {len(deep)} | params {npar} | "
          f"RESULT shallow {rp(shallow):.3f} mid {rp(mid):.3f} deep {rp(deep):.3f}", flush=True)
    t0 = time.time(); curve = []
    if ARM == "TREE":
        from mycelium import tree_db as TDB
        dbp = f".cache/tree_reldsl_{TAG}.db"; assert not os.path.exists(dbp), f"refusing to reuse {dbp}"
        tdb = TDB.TreeDB(dbp); tdb.register_space("step-type", "symbolic", 2, "rel-DSL verified step: (breath fired, chain depth)")
        tname = [f"{on}.{rn}" for on in ("given", "add", "mul", "sub") for rn in ("arg", "result", "minuend")]
        CF = [np.zeros(NTYPE), np.zeros((NTYPE, RB.D)), np.zeros(NTYPE)]; ev = []; tstat = []
    for step in range(1, RB.STEPS + 1):
        its = [train[i] for i in rng.integers(0, len(train), RB.BATCH)]
        bt, ids = RB.batch(its, rng)
        if ARM == "TREE": net.collect = []
        lg_all = net(bt, ids); loss = RB.loss_fn(lg_all, bt)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if ARM == "TREE":
            with torch.no_grad():                                                # verified chains only (the executor)
                lgl = lg_all[:, -1]; ver = torch.as_tensor([RB.RC.execute(t["facs"], {v: int(lgl[k, v].argmax()) for v in t["der"]},
                                                                            t["query"]) == t["gold"] for k, t in enumerate(its)])
                for b, fire, ty, xs in net.collect:
                    m = (fire & ver[:, None]).numpy()
                    if not m.any(): continue
                    tys, X = ty.numpy()[m], xs.numpy()[m].astype(np.float64); kk = np.nonzero(m)[0]
                    np.add.at(CF[0], tys, 1); np.add.at(CF[1], tys, X); np.add.at(CF[2], tys, (X * X).sum(1))
                    ev += [(tname[t_], np.array([b, its[k]["depth"]], np.float32), f"step{step}.item{k}") for t_, k in zip(tys, kk)]
            net.collect = None
            if step % TWIN == 0:                                                 # close the window: centroids + a batch
                live = CF[0] > 0; c = np.zeros((NTYPE, RB.D)); c[live] = CF[1][live] / CF[0][live, None]
                net.cent.copy_(torch.as_tensor(c, dtype=torch.float32)); tdb.save_centroids(TAG, *CF)
                if ev:
                    tdb.add_evidence(f"w{step}", ev, "permanent", "executor", TAG, "step-type")
                    if not tdb.nodes():                                          # the shape, certified by the first batch:
                        tdb.add_node(1000, None, 0, "step", f"w{step}")          # root -> op -> role; leaf id = type index
                        for o, on in enumerate(("given", "add", "mul", "sub")):  # (= the centroids table's node)
                            tdb.add_node(100 + o, 1000, 1, on, f"w{step}")
                            for r, rn in enumerate(("arg", "result", "minuend")): tdb.add_node(3 * o + r, 100 + o, 2, f"{on}.{rn}", f"w{step}")
                tstat.append(dict(step=step, n=[int(x) for x in CF[0]], ver=float(ver.float().mean())))
                print(f"[ro] {TAG} tree window {step}: verified steps {int(CF[0].sum())} | per type " +
                      " ".join(f"{tname[i]}:{int(CF[0][i])}" for i in range(NTYPE) if CF[0][i]), flush=True)
                CF = [np.zeros(NTYPE), np.zeros((NTYPE, RB.D)), np.zeros(NTYPE)]; ev = []
        if step % 2000 == 0 or step == RB.STEPS:
            net.eval(); s_ = RB.score(net, shallow).mean(); d_ = RB.score(net, deep).mean(); net.train()
            curve.append(dict(step=step, loss=float(loss.detach()), shallow=float(s_), deep=float(d_)))
            print(f"[ro] {TAG} step {step} loss {float(loss.detach()):.4f} SHALLOW {s_:.3f} DEEP {d_:.3f} ({time.time() - t0:.0f}s)", flush=True)
    if ARM == "TREE":                                                            # score with the centroids reloaded FROM SQLite
        N_, LS_, _ = tdb.load_centroids(TAG, TAG); c = np.zeros((NTYPE, RB.D)); c[N_ > 0] = LS_[N_ > 0] / N_[N_ > 0, None]
        assert np.abs(c - net.cent.numpy()).max() < 1e-5, "SQLite centroids differ from memory"
        net.cent.copy_(torch.as_tensor(c, dtype=torch.float32)); tdb.close()
    net.eval(); S = {k: RB.score(net, v) for k, v in (("shallow", shallow), ("mid", mid), ("deep", deep))}
    bydepth = {d: float(np.mean([o for o, t in zip(S["deep"], deep) if t["depth"] == d])) for d in range(5, 10) if any(t["depth"] == d for t in deep)}
    json.dump(dict(tag=TAG, arm=ARM, seed=RB.SEED, split=SPLIT, n_train=len(train), params=npar, steps=RB.STEPS, curve=curve,
                   **{k: float(v.mean()) for k, v in S.items()}, **{k + "_per": [int(x) for x in v] for k, v in S.items()},
                   deep_by_depth=bydepth, cov=float(os.environ.get("RD_COV", "0.25")) if SPLIT == "depth2" else None,
                   tree=tstat if ARM == "TREE" else None, result=dict(shallow=rp(shallow), mid=rp(mid), deep=rp(deep))),
              open(f".cache/rel_dsl_{TAG}.json", "w"))
    print(f"[ro] {TAG} FINAL SHALLOW {S['shallow'].mean():.3f} MID {S['mid'].mean():.3f} DEEP {S['deep'].mean():.3f} | by depth "
          + " ".join(f"d{d} {v:.3f}" for d, v in bydepth.items()), flush=True)
