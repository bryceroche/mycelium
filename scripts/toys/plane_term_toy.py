"""plane_term_toy.py -- T2n, THE LEARNED TERM READS THE TREE, PLANE BY PLANE (2026-10-24; Bryce's word; ledger "T2n"). CPU.
T2l's HYBRID_ROLE_TREE with its LEARNED term (the 12:02 autopsy's single culprit: a free bilinear score between the mention's
reader state and the slot's FIRST-MENTION span, i.e. the name's surface) replaced. Generator, seeds, eval sets, reader, query
hop and every other resolver term are tree_type_toy's, imported unchanged.
The slot side of the new terms reads ONLY the database: the slot's composite key, phases
    K_s = node(type) + slot(j) + time(j) + level(node)      (tree_type_toy's fixed phasor codebooks, PB = 64 planes)
the composite key T2l's TREE_BOUND arm bound; never the slot's tokens. The mention side is the reader (MHA) state.
Arms:
  TREE         the control: T2l's arm, free learned term on the name's surface (re-run)
  NOLEARN      the learned term removed (is it needed at all?)
  FREEKEY      the database key read FREELY: MLP([W hm, Re K_s, Im K_s]) -> score (mixes across planes)
  PLANE        the database key read PLANE-WISE: the reader emits a query phase per plane, theta_q = W hm; each plane
               scores its own phase difference d_p = theta_q,p - K_s,p with ONE shared learned function
               f(d) = a1 cos d + b1 sin d + a2 cos 2d + b2 sin 2d; the planes are pooled by a plain sum (T3d's rule:
               transform plane-wise, decide by symmetric pooling)
  PLANE_NOMLP  PLANE with an MHA-ONLY reader (every encoder layer's FFN zeroed and frozen; Bryce: "the breathing
               transformer is MHA without the MLP")"""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_argv = sys.argv[:]; sys.argv = [sys.argv[0]] + (["--smoke"] if "--smoke" in _argv else [])
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tree_type_toy as K
sys.argv = _argv

ARMS = os.environ.get("TN_ARMS", "TREE,NOLEARN,FREEKEY,PLANE,PLANE_NOMLP").split(",")
OUT_TXT = ".cache/plane_term{}.txt"
PB = K.PB
ANG = {k: torch.tensor(np.angle(v), dtype=torch.float32) for k, v in
       (("slot", K.PH_SLOT), ("time", K.PH_TIME), ("level", K.PH_LEVEL), ("node", K.PH_NODE))}
NODE_LEVEL = torch.tensor(K.NODE_LEVEL)


def slot_key(bt):
    """(B,S,PB) composite-key phases from the database only: node(type) + slot(j) + time(j) + level(node)."""
    ty = bt["stype"].clamp(min=0); B, S = ty.shape; j = torch.arange(S)[None].expand(B, -1)
    return ANG["node"][ty] + ANG["slot"][j] + ANG["time"][j % 16] + ANG["level"][NODE_LEVEL[ty]]


class Net(K.Net):
    def __init__(self, arm):
        super().__init__("HYBRID_ROLE_TREE"); self.arm2 = arm
        if arm == "FREEKEY":
            self.fk = nn.Sequential(nn.Linear(K.D + 2 * PB, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
            self.fm = nn.Linear(K.D, K.D)
        if arm.startswith("PLANE"):
            self.wph = nn.Linear(K.D, PB); self.fc = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        if arm == "PLANE_NOMLP":
            for layer in self.enc.layers:
                for lin in (layer.linear1, layer.linear2):
                    nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias); lin.weight.requires_grad_(False); lin.bias.requires_grad_(False)

    def resolve(self, h, bt):
        self._bt = bt
        return super().resolve(h, bt)

    def learned(self, span, h, hs):
        if self.arm2 == "TREE":
            return super().learned(span, h, hs)
        hm = torch.einsum("bxl,bld->bxd", span, h)                          # (B,X,D) the reader's mention state
        if self.arm2 == "NOLEARN":
            return torch.zeros(hm.shape[0], hm.shape[1], K.S)
        Ks = slot_key(self._bt)                                             # (B,S,PB) database only
        if self.arm2 == "FREEKEY":
            X, S = hm.shape[1], K.S
            a = self.fm(hm)[:, :, None].expand(-1, -1, S, -1)
            k = torch.cat([Ks.cos(), Ks.sin()], -1)[:, None].expand(-1, X, -1, -1)
            return self.fk(torch.cat([a, k], -1))[..., 0]
        q = self.wph(hm)                                                    # (B,X,PB) query phases
        eq1, ek1 = torch.polar(torch.ones_like(q), q), torch.polar(torch.ones_like(Ks), Ks)
        eq2, ek2 = torch.polar(torch.ones_like(q), 2 * q), torch.polar(torch.ones_like(Ks), 2 * Ks)
        s1 = torch.einsum("bxp,bsp->bxs", eq1, ek1.conj()) / PB             # sum over planes of e^{i d_p}: symmetric pooling
        s2 = torch.einsum("bxp,bsp->bxs", eq2, ek2.conj()) / PB
        a1, b1, a2, b2 = self.fc
        return self.ls.exp() * (a1 * s1.real + b1 * s1.imag + a2 * s2.real + b2 * s2.imag)


def run(arm, seed):
    """tree_type_toy.run for HYBRID_ROLE_TREE, verbatim, with this file's Net."""
    rng = np.random.default_rng(seed); erng = np.random.default_rng(91_000 + seed)
    nrng = np.random.default_rng(77_000 + seed); envg = np.random.default_rng(93_000 + seed)
    trng = np.random.default_rng(55_000 + seed); tev = np.random.default_rng(57_000 + seed)
    ev = {"indist": K.batch(erng, K.N_EVAL, "train", 0.0, envg, 0.0, False, tev),
          "heldout": K.batch(erng, K.N_EVAL, "heldout", 0.0, envg, 0.0, False, tev)}
    torch.manual_seed(seed); net = Net(arm)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=K.LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=K.LR, total_steps=K.STEPS, pct_start=0.1)
    for step in range(K.STEPS):
        bt = K.batch(rng, K.BATCH, "train", 0.0, nrng, 0.0, False, trng); lp, aux = net(bt); tg = bt["tgt"]; m = tg >= 0
        loss = F.nll_loss(lp[m], tg[m]) + aux
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    return K.evaluate(net, ev)


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "16")))
    OUT = OUT_TXT.format("_smoke" if K.mode == "smoke" else "")
    assert K.mode == "smoke" or not os.path.exists(OUT), "refusing to overwrite a banked read"
    out = open(OUT, "w"); t0 = time.time(); A = {}; allres = []

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    say(f"mode {K.mode} steps {K.STEPS} seeds {K.SEEDS} arms {ARMS}")
    # the control's equivalence: this Net('TREE') must be tree_type_toy's HYBRID_ROLE_TREE (same init, same forward)
    torch.manual_seed(0); a = K.Net("HYBRID_ROLE_TREE"); torch.manual_seed(0); b = Net("TREE"); a.eval(); b.eval()
    bt = K.batch(np.random.default_rng(3), 16, "heldout")
    with torch.no_grad():
        assert torch.equal(a(bt)[0], b(bt)[0]), "control differs from T2l's HYBRID_ROLE_TREE"
    for arm in ARMS:
        rs = [run(arm, s) for s in K.SEEDS]
        ag = {kk: float(np.nanmean([x[kk] for x in rs])) for kk in rs[0]}; A[arm] = ag; allres.append(dict(arm=arm, seeds=rs, agg=ag))
        for sp in ("indist", "heldout"):
            say(f"{arm:12s} {sp:7s} | role {ag[f'{sp}_res_pron_role']:.3f} (seeds {[round(x[f'{sp}_res_pron_role'], 3) for x in rs]}) | all "
                f"{ag[f'{sp}_res_all']:.3f} alias {ag[f'{sp}_res_alias']:.3f} | blind {ag[f'{sp}_acc_blind']:.3f} explicit {ag[f'{sp}_acc_explicit']:.3f}")
            say("        cells: " + " ".join(f"{c} {ag[f'{sp}_res_{c}']:.3f}" for c in K.CELLS) + f"  [{time.time() - t0:.0f}s]")
    json.dump(allres, open(OUT.replace(".txt", ".json"), "w"), indent=1)
    if K.mode == "full" and all(x in A for x in ("TREE", "FREEKEY", "PLANE")):   # PINNED (ledger 2026-10-24, T2n)
        c, fk, pl = A["TREE"], A["FREEKEY"], A["PLANE"]
        say("VERDICTS (3-seed means)")
        say(f"CONTROL: TREE held-out role {c['heldout_res_pron_role']:.3f} (banked 0.894) -> "
            f"{'OK' if abs(c['heldout_res_pron_role'] - 0.894) <= 0.03 else 'VOID'}")
        worst = min(pl[f"indist_res_{k}"] - c[f"indist_res_{k}"] for k in K.CELLS)
        p1 = pl["heldout_res_pron_role"] >= 0.95 and worst >= -0.01 and pl["heldout_res_all"] >= c["heldout_res_all"] - 0.02
        say(f"P1 PLANE-WISE TREE TERM: {'PASS' if p1 else 'FAIL'} (held-out role {pl['heldout_res_pron_role']:.3f}; worst in-dist cell "
            f"delta {worst:+.3f}; held-out all {pl['heldout_res_all']:.3f} vs {c['heldout_res_all']:.3f})")
        sd = [p["heldout_res_pron_role"] - f["heldout_res_pron_role"] for p, f in
              zip(allres[ARMS.index("PLANE")]["seeds"], allres[ARMS.index("FREEKEY")]["seeds"])]
        gap = pl["heldout_res_pron_role"] - fk["heldout_res_pron_role"]
        p2 = gap >= 0.03 and all(x > 0 for x in sd)
        say(f"P2 PLANE-WISE BEYOND THE KEY: {'PASS' if p2 else 'FAIL'} (PLANE - FREEKEY held-out role {gap:+.3f}; per seed {[round(x, 3) for x in sd]})")
        for nm in ("NOLEARN", "PLANE_NOMLP"):
            if nm in A:
                w = min(A[nm][f"indist_res_{k}"] - c[f"indist_res_{k}"] for k in K.CELLS)
                say(f"READ (no bar) {nm}: held-out role {A[nm]['heldout_res_pron_role']:.3f}; worst in-dist cell delta vs TREE {w:+.3f}; "
                    f"held-out all {A[nm]['heldout_res_all']:.3f}")
    say(f"DONE in {time.time() - t0:.0f}s")
