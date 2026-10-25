"""tree_search_toy.py -- T2o, REAL T6 GEAR STAMPS + THE READER SEARCHES THE TREE (2026-10-24; Bryce's word; ledger "T2o"). CPU.
T2n's PLANE arm (plane_term_toy: the learned term reads the slot's composite key plane-wise) with two changes, one per ask:
  (1) THE CLOCK. The key's time factor time[j % 16] (the slot INDEX, i.e. creation order) is replaced by two REAL gear stamps,
      read relative to the mention's own tick (tick = sentence index; T2m's GEARS: periods 2,4,8,16,32,64, plane p rides gear p % 6):
          K_xs = node(n_s) + slot(j) + level(n_s) + gC(now_x - created_s) + gL(now_x - lastmention_s,x)
      so the key is per (mention, slot): the last-mention stamp moves as the story goes.
  (2) THE SEARCH. No gold node at test. The tree is permanent: interior nodes person / group(they) / business(it) at level 1,
      he / she under person at level 2, each with a search centroid = a decayed running mean of the reader's search vector
      v_s = normalize(W hs) over certified members (decayed because the reader's coordinates drift while it trains); below
      them, LITERAL LEAVES keyed by the exact name / head token (the SQL primary key). In training every slot is a certified
      write (the generator's node stands in for the answer-key gate), so every training name is a leaf and the resolver trains
      on exactly GEARS's input; the search head trains by a prototype loss against the (detached) centroids.
      At test, a slot's head token is looked up: HIT -> the leaf's node. MISS (an unseen name) -> the coarse-to-fine walk
      (level-1 snap, then level-2 snap) gives a posterior, and the stored procedure UPSERTS a provisional leaf: INSERT on first
      sight, then every later sighting UPDATES the leaf's exact running mean of walk posteriors (Welford, count-based); the
      leaf's node = argmax of that mean, or the undecided parent 'person' when the max is under the vigilance threshold tau.
      The eval split is streamed in episode order, so an unseen name's first sightings ride on the walk alone.
Generator, seeds, eval sets, reader, query hop and the other resolver terms are tree_type_toy's (the episode function is
tree_type_toy's own source with one mention-log line injected; asserted byte-identical on 500 episodes).
Arms:
  GOLD          the control: T2n's PLANE (gold node, slot-index time; asserted equal to plane_term_toy's PLANE at init)
  GEARS         gold node, real gear stamps
  SEARCH        GEARS + the search head; scored with searched nodes (leaf-upserted, tau 0.5)
  GEARS_NOHAND  GEARS with the hand recency rank AND the role flags removed (read: does the clock carry time on its own?)
  SLOT_NOHAND   GOLD with the same removal (its floor)"""
import inspect, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_argv = sys.argv[:]; sys.argv = [sys.argv[0]] + (["--smoke"] if "--smoke" in _argv else [])
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tree_type_toy as K
import plane_term_toy as P
sys.argv = _argv

ARMS = os.environ.get("TO_ARMS", "GOLD,GEARS,SEARCH,GEARS_NOHAND,SLOT_NOHAND").split(",")
OUT_TXT = ".cache/tree_search{}.txt"
PB, S = K.PB, K.S
_pg = np.random.default_rng(4053)
OMEGA = torch.tensor(2 * np.pi / np.array([2., 4., 8., 16., 32., 64.]), dtype=torch.float32)[torch.arange(PB) % 6]
OFF_C, OFF_L = (torch.tensor(_pg.uniform(0, 2 * np.pi, PB), dtype=torch.float32) for _ in range(2))
PERSON = 4                                                                  # the undecided interior node (level 1)
ANG_NODE = torch.cat([P.ANG["node"], torch.tensor(_pg.uniform(0, 2 * np.pi, (1, PB)), dtype=torch.float32)])
NODE_LEVEL = torch.tensor(K.NODE_LEVEL + [1])
L1 = [PERSON, 2, 3]                                                         # level-1 children of the root
DS, SC, ALPHA = 32, 10.0, 0.05                                              # search dims, prototype scale, centroid decay
HELD_TOK = {K.O_F + f for f in K.HELDOUT}

# ---- the generator, verbatim + one mention log line (each mention: entity, start position, sentence tick) ----
ORIG_EPISODE = K.episode
_src = inspect.getsource(ORIG_EPISODE)
for a, b in (("toks, slot_of, slot_span, st = [], {}, [], dict(P=None)", "toks, slot_of, slot_span, st = [], {}, [], dict(P=None); mlog = []"),
             ("def mention(e, allow_pron=True, push=False):\n        x = ents[e]\n",
              "def mention(e, allow_pron=True, push=False):\n        x = ents[e]; mlog.append((e, len(toks), toks.count(T_S)))\n"),
             ("return dict(toks=toks,", "return dict(mlog=mlog, toks=toks,")):
    assert _src.count(a) == 1, a
    _src = _src.replace(a, b)
exec(compile(_src, "tree_type_toy.episode+mlog", "exec"), K.__dict__)
LOGGED = []
_ep_mlog = K.episode


def _ep_logged(rng, split):
    x = _ep_mlog(rng, split); LOGGED.append(x); return x
K.episode = _ep_logged


def ages(x, a, now):
    """(S,2) for a mention starting at token a with tick now: (now - creation tick, now - last-mention tick before a)."""
    out = np.zeros((S, 2), np.float32)
    for e, j in x["slot_of"].items():
        ms = [(p, t) for (e2, p, t) in x["mlog"] if e2 == e]
        c = min(ms)[1]; prior = [t for p, t in ms if p < a]
        out[j] = (now - c, now - (max(prior) if prior else c))
    return out


def batch(rng, B, split, *args):
    LOGGED.clear(); bt = K.batch(rng, B, split, *args); exs = list(LOGGED); assert len(exs) == B
    R = bt["rpos"].shape[1]; ra = np.zeros((B, R, S, 2), np.float32); qa = np.zeros((B, 1, S, 2), np.float32)
    for b, x in enumerate(exs):
        cs = np.cumsum(np.array(x["toks"]) == K.T_S)
        for i, r in enumerate(x["recs"]):
            ra[b, i] = ages(x, r["ms"][0], int(cs[r["ms"][0]]))
        qa[b, 0] = ages(x, len(x["toks"]), int(cs[-1]) + 1)
    bt["rage"], bt["qage"] = torch.from_numpy(ra), torch.from_numpy(qa)
    return bt


class Net(P.Net):
    def __init__(self, arm):
        super().__init__("PLANE"); self.arm3 = arm
        self.gears = arm in ("GEARS", "SEARCH", "GEARS_NOHAND"); self.hand = not arm.endswith("NOHAND")
        if arm == "SEARCH":
            self.wsr = nn.Linear(K.D, DS)
            self.register_buffer("mu", torch.zeros(5, DS)); self.register_buffer("mu_n", torch.zeros(5))

    def keys(self, bt, ag):
        """(B,X,S,PB) composite-key phases (X = 1 broadcasts)."""
        ty = bt["stype"].clamp(min=0); B = ty.shape[0]; j = torch.arange(S)[None].expand(B, -1)
        base = ANG_NODE[ty] + P.ANG["slot"][j] + P.ANG["level"][NODE_LEVEL[ty]]
        if not self.gears:
            return (base + P.ANG["time"][j % 16])[:, None]
        return base[:, None] + OMEGA * ag[..., 0:1] + OFF_C + OMEGA * ag[..., 1:2] + OFF_L

    def term(self, span, h, Kx):
        """T2n's PLANE term: per-plane phase difference, one shared degree-2 Fourier f, pooled by a plain mean over planes."""
        hm = torch.einsum("bxl,bld->bxd", span, h); q = self.wph(hm)
        d = q[:, :, None, :] - Kx
        a1, b1, a2, b2 = self.fc
        return self.ls.exp() * (a1 * d.cos().mean(-1) + b1 * d.sin().mean(-1) + a2 * (2 * d).cos().mean(-1) + b2 * (2 * d).sin().mean(-1))

    def resolve(self, h, bt):
        """K.Net.resolve for HYBRID_ROLE_TREE (same term order), the learned term replaced by term() on keys()."""
        hs = torch.einsum("bsl,bld->bsd", bt["sspan"], h)
        B = h.shape[0]; z = torch.zeros(B, 1, dtype=torch.long)
        rank = bt["rank"] if self.hand else torch.full_like(bt["rank"], S)
        rl = self.exact(bt["mcont"], bt["mtitle"], bt["mpron"], rank, bt)
        ql = self.exact(bt["qcont"][:, None], z, z - 1, torch.full((B, 1, S), S), bt)
        if self.hand:
            rl = rl + (bt["mpron"] >= 0)[..., None] * (self.wr[0] * bt["rsubj"] + self.wr[1] * bt["rrecp"])
        rl = rl + self.term(bt["rspan"], h, self.keys(bt, bt["rage"]))
        ql = ql + self.term(bt["qspan"][:, None], h, self.keys(bt, bt["qage"]))
        if self.arm3 == "SEARCH":
            self._v = F.normalize(self.wsr(hs), dim=-1)
        m = ~bt["smask"][:, None, :]
        return rl.masked_fill(m, -1e9), ql.masked_fill(m, -1e9)[:, 0]

    def walk(self, v):
        """(..., DS) -> (..., 4) posterior over the tree's nodes he/she/they/it: level-1 snap x level-2 snap."""
        p1 = F.softmax(-SC * ((v[..., None, :] - self.mu[L1]) ** 2).sum(-1), -1)
        p2 = F.softmax(-SC * ((v[..., None, :] - self.mu[[0, 1]]) ** 2).sum(-1), -1)
        return torch.cat([p1[..., :1] * p2, p1[..., 1:]], -1)


def proto_loss(net, bt):
    v, m, ty = net._v, bt["smask"], bt["stype"]
    t1 = torch.where(ty < 2, 0, ty - 1)
    l = F.cross_entropy(-SC * ((v[..., None, :] - net.mu[L1]) ** 2).sum(-1)[m], t1[m])
    mp = m & (ty < 2)
    return l + F.cross_entropy(-SC * ((v[..., None, :] - net.mu[[0, 1]]) ** 2).sum(-1)[mp], ty[mp])


@torch.no_grad()
def update_centroids(net, bt):
    v, m, ty = net._v.detach(), bt["smask"], bt["stype"]
    for c in range(5):
        mm = m & ((ty < 2) if c == PERSON else (ty == c))
        if mm.any():
            bm = v[mm].mean(0)
            net.mu[c] = bm if net.mu_n[c] == 0 else net.mu[c] + ALPHA * (bm - net.mu[c]); net.mu_n[c] += int(mm.sum())


def search_nodes(net, bt, leaves, tau):
    """Stream the split in episode order: HIT -> leaf node; MISS -> walk + upsert (exact running mean). Returns the searched
    stype tensor, the per-episode walk node (no leaf), and per-slot flags."""
    net.eval()
    with torch.no_grad():
        net(bt)
    post = net.walk(net._v).numpy(); walk = post.argmax(-1)
    st, sm, tok = bt["stype"].numpy(), bt["smask"].numpy(), bt["sptok"].numpy()
    out = st.copy(); prov = {}
    for b in range(st.shape[0]):
        for j in range(S):
            if not sm[b, j]:
                continue
            t = int(tok[b, j])
            if t in leaves:
                out[b, j] = leaves[t]; continue
            n, mean = prov.get(t, (0, np.zeros(4)))
            n += 1; mean = mean + (post[b, j] - mean) / n; prov[t] = (n, mean)            # INSERT / UPDATE (Welford mean)
            k = int(mean.argmax())
            out[b, j] = k if mean[k] >= tau or mean[:2].sum() < max(mean[2], mean[3]) else PERSON
    net.train()
    return torch.from_numpy(out), torch.from_numpy(walk), prov


def run(arm, seed):
    """plane_term_toy.run (= tree_type_toy.run for HYBRID_ROLE_TREE), with this file's Net, batch, and the search head."""
    rng = np.random.default_rng(seed); erng = np.random.default_rng(91_000 + seed)
    nrng = np.random.default_rng(77_000 + seed); envg = np.random.default_rng(93_000 + seed)
    trng = np.random.default_rng(55_000 + seed); tev = np.random.default_rng(57_000 + seed)
    ev = {"indist": batch(erng, K.N_EVAL, "train", 0.0, envg, 0.0, False, tev),
          "heldout": batch(erng, K.N_EVAL, "heldout", 0.0, envg, 0.0, False, tev)}
    torch.manual_seed(seed); net = Net(arm); leaves = {}
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=K.LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=K.LR, total_steps=K.STEPS, pct_start=0.1)
    for step in range(K.STEPS):
        bt = batch(rng, K.BATCH, "train", 0.0, nrng, 0.0, False, trng); lp, aux = net(bt); tg = bt["tgt"]; m = tg >= 0
        loss = F.nll_loss(lp[m], tg[m]) + aux
        if arm == "SEARCH":
            loss = loss + proto_loss(net, bt)
            for t, n_ in zip(bt["sptok"][bt["smask"]].tolist(), bt["stype"][bt["smask"]].tolist()):
                assert leaves.setdefault(t, n_) == n_, "a literal leaf with two nodes"           # certified INSERT
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        if arm == "SEARCH":
            update_centroids(net, bt)
    if arm != "SEARCH":
        return K.evaluate(net, ev)
    r = {}
    gold = {sp: bt["stype"].clone() for sp, bt in ev.items()}
    for sp, bt in ev.items():                                               # the walk alone, every slot treated as a miss
        net.eval()
        with torch.no_grad():
            net(bt)
        wk = net.walk(net._v).argmax(-1); sm = bt["smask"]; held = torch.isin(bt["sptok"], torch.tensor(sorted(HELD_TOK))) & sm
        r[f"{sp}_walk_all"] = float((wk == gold[sp])[sm].float().mean())
        r[f"{sp}_walk_heldname"] = float((wk == gold[sp])[held].float().mean()) if held.any() else float("nan")
        r[f"{sp}_n_heldname"] = int(held.sum())
    net.train()
    r.update({f"goldnode_{k}": v for k, v in K.evaluate(net, {"heldout": ev["heldout"]}).items()})
    for tau in (0.5, 0.7, 0.9):
        evs = {}
        for sp, bt in (ev.items() if tau == 0.5 else [("heldout", ev["heldout"])]):
            s_, wk, prov = search_nodes(net, bt, leaves, tau)
            held = torch.isin(bt["sptok"], torch.tensor(sorted(HELD_TOK))) & bt["smask"]
            r[f"t{tau}_{sp}_leafacc_heldname"] = float((s_ == gold[sp])[held].float().mean()) if held.any() else float("nan")
            r[f"t{tau}_{sp}_undecided_heldname"] = float((s_ == PERSON)[held].float().mean()) if held.any() else float("nan")
            r[f"t{tau}_{sp}_names_right"] = (sum(int(v[1].argmax() == (0 if t - K.O_F < 8 else 1)) for t, v in prov.items()), len(prov))
            evs[sp] = dict(bt, stype=s_)
            if tau == 0.5:
                evs[sp + "_noleaf"] = dict(bt, stype=wk)
        res = K.evaluate(net, {k: v for k, v in evs.items() if not k.endswith("_noleaf")})
        if tau == 0.5:
            r.update(res)                                                   # the scored read: leaf-upserted, tau 0.5
            r.update({f"noleaf_{k}": v for k, v in K.evaluate(net, {"heldout": evs["heldout_noleaf"]}).items()})
        else:
            r.update({f"t{tau}_{k}": v for k, v in res.items()})
    return r


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "16")))
    OUT = OUT_TXT.format("_smoke" if K.mode == "smoke" else "")
    assert K.mode == "smoke" or not os.path.exists(OUT), "refusing to overwrite a banked read"
    out = open(OUT, "w"); t0 = time.time(); A = {}; allres = []

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    say(f"mode {K.mode} steps {K.STEPS} seeds {K.SEEDS} arms {ARMS}")
    # the generator is tree_type_toy's, byte for byte
    r1, r2 = np.random.default_rng(11), np.random.default_rng(11)
    for i in range(500):
        sp = "heldout" if i % 2 else "train"; x, y = ORIG_EPISODE(r1, sp), _ep_mlog(r2, sp)
        assert x["toks"] == y["toks"] and x["recs"] == y["recs"] and x["slot_of"] == y["slot_of"], "generator drifted"
    # the control is T2n's PLANE (same init, same forward up to float order)
    bt = batch(np.random.default_rng(3), 16, "heldout")
    torch.manual_seed(0); a = P.Net("PLANE"); torch.manual_seed(0); b = Net("GOLD"); a.eval(); b.eval()
    with torch.no_grad():
        assert torch.allclose(a(bt)[0], b(bt)[0], atol=1e-5), "control differs from T2n's PLANE"
    for arm in ARMS:
        rs = [run(arm, s) for s in K.SEEDS]
        ag = {kk: (float(np.nanmean([x[kk] for x in rs])) if not isinstance(rs[0][kk], tuple) else [x[kk] for x in rs]) for kk in rs[0]}
        A[arm] = ag; allres.append(dict(arm=arm, seeds=rs, agg=ag))
        for sp in ("indist", "heldout"):
            say(f"{arm:12s} {sp:7s} | role {ag[f'{sp}_res_pron_role']:.3f} (seeds {[round(x[f'{sp}_res_pron_role'], 3) for x in rs]}) | all "
                f"{ag[f'{sp}_res_all']:.3f} alias {ag[f'{sp}_res_alias']:.3f} | blind {ag[f'{sp}_acc_blind']:.3f} explicit {ag[f'{sp}_acc_explicit']:.3f}")
            say("        cells: " + " ".join(f"{c} {ag[f'{sp}_res_{c}']:.3f}" for c in K.CELLS) + f"  [{time.time() - t0:.0f}s]")
        if arm == "SEARCH":
            say(f"        WALK (every slot as a miss): in-dist all {ag['indist_walk_all']:.3f} | held-out names {ag['heldout_walk_heldname']:.3f} "
                f"(seeds {[round(x['heldout_walk_heldname'], 3) for x in rs]}; n {ag['heldout_n_heldname']:.0f}) | held-out all {ag['heldout_walk_all']:.3f}")
            for tau in (0.5, 0.7, 0.9):
                hr = ag["heldout_res_pron_role"] if tau == 0.5 else ag[f"t{tau}_heldout_res_pron_role"]
                say(f"        LEAF tau {tau}: held-out-name node acc {ag[f't{tau}_heldout_leafacc_heldname']:.3f} undecided "
                    f"{ag[f't{tau}_heldout_undecided_heldname']:.3f} names right {ag[f't{tau}_heldout_names_right']} | held-out role {hr:.3f}")
            say(f"        held-out role with: gold node {ag['goldnode_heldout_res_pron_role']:.3f} | walk, no leaf {ag['noleaf_heldout_res_pron_role']:.3f}")
    json.dump(allres, open(OUT.replace(".txt", ".json"), "w"), indent=1)
    if K.mode == "full" and all(x in A for x in ("GOLD", "GEARS", "SEARCH")):   # PINNED (ledger 2026-10-24, T2o)
        g, gr, se = A["GOLD"], A["GEARS"], A["SEARCH"]
        say("VERDICTS (3-seed means)")
        say(f"CONTROL: GOLD held-out role {g['heldout_res_pron_role']:.3f} (T2n PLANE banked 0.995) -> "
            f"{'OK' if abs(g['heldout_res_pron_role'] - 0.995) <= 0.02 else 'VOID'}")
        worst = min(gr[f"indist_res_{k}"] - g[f"indist_res_{k}"] for k in K.CELLS)
        g1 = gr["heldout_res_pron_role"] >= g["heldout_res_pron_role"] - 0.01 and worst >= -0.01
        say(f"G1 REAL CLOCK IN THE KEY: {'PASS' if g1 else 'FAIL'} (GEARS held-out role {gr['heldout_res_pron_role']:.3f} vs "
            f"{g['heldout_res_pron_role']:.3f}; worst in-dist cell delta {worst:+.3f})")
        w = se["heldout_walk_heldname"]
        say(f"S1 THE WALK FINDS THE NODE: {'PASS' if w >= 0.90 else 'FAIL'} (held-out-name slots, per episode, no leaf: {w:.3f})")
        s2 = se["heldout_res_pron_role"] >= 0.95 and se["heldout_res_pron_role"] >= gr["heldout_res_pron_role"] - 0.02
        say(f"S2 THE UPSERTED TREE: {'PASS' if s2 else 'FAIL'} (SEARCH held-out role {se['heldout_res_pron_role']:.3f} vs GEARS "
            f"{gr['heldout_res_pron_role']:.3f}; leaf node acc {se['t0.5_heldout_leafacc_heldname']:.3f})")
        if all(x in A for x in ("GEARS_NOHAND", "SLOT_NOHAND")):
            gn, sn = A["GEARS_NOHAND"], A["SLOT_NOHAND"]
            say("READ (no bar) NOHAND, in-dist / held-out: " + " | ".join(
                f"{c} gears {gn[f'indist_res_{c}']:.3f}/{gn[f'heldout_res_{c}']:.3f} slot {sn[f'indist_res_{c}']:.3f}/{sn[f'heldout_res_{c}']:.3f}"
                for c in ("pron_gender", "pron_role", "they", "it")))
    say(f"DONE in {time.time() - t0:.0f}s")
