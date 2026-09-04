"""step_trainer.py — THE INNER-STEP TRAINER v1 (FINAL BOSS, 2026-09-03).

Implements alternator_v21_training_spec.md Section 1 on the FACTORED
forward (apply_step_trainer.py): forward = stage-0 (the real forward()
under _STEP_TAP hold) + K_B-1 dispatches of the real breath_step() with
solver pings between dispatches (alt2_fact_buf + alternator_bridge,
facts as DETACHED constants assigned into fixed buffers — the
dual-terminal contract); backward = the checkpointed reverse walk
(recompute segment k from banked state inside its own capture, param
grads accumulated into fixed buffers, dL/dstate threaded down); then
one optimizer capture. SINGLE SOURCE OF TRUTH: every tensor op runs
through phase1_algebra_head's own organs (forward preamble, breath_step,
_make_bank, _heads_of, _fact_inject, _loss_single) — this file contains
ZERO reimplemented model math.

DEVIATION FROM THE SPEC'S "THREE CAPTURES", DECLARED: the live loop is
shape-INHOMOGENEOUS across breaths (notebook shelf grows 1 -> K_B,
garage shelf 0 -> K_B-1, kb-conditional python branches: notebook birth
at kb==1, shelf-circle at kb==SC_KB), so ONE weight-tied step capture
cannot exist without surgery on the proven graph. The ladder becomes a
CAPTURE FAMILY keyed by breath index: 1 stage-0 + (K_B-1) fwd steps +
(K_B-1) reverse steps + 1 stage-0-reverse + 1 decode + 1 optimizer =
2*K_B + 2 captures (16 at K_B=7), each SMALLER than the fused do_train
step that runs today. Memory stays O(1) in depth (state banks + fact
bufs), which is the spec's real point.

FINDING (load-bearing, discovered by reading the loop): W_fact facts
inject into the VAR-slot states (vst), and vst is read ONLY by the
emission heads — the breath loop itself never touches vst. Per-seam
fact injection therefore sharpens every seam's decode and every
breath's ladder-loss emissions (vst_k = _fact_inject(vst_base,
fact_k)), and compounds through the ping chain — but does NOT enter the
slot-mixer dynamics in today's architecture. The channels that DO enter
the dynamics (per-seam masks, the mask head, the per-step atlas) plug
into exactly the seams this trainer creates; that is the final-boss
case as the ledger scoped it.

LOSS DECOMPOSITION (exact, verified by --cpuprobe grads): the fused
ladder loss sum_kb w_kb * _loss_single(dict(o, **heads_kb)) splits into
per-breath emission terms (attached at reverse step kb with the SHARED
keys detached) plus the shared terms (fat/vat/query/bind/...) attached
ONCE at stage-0 with coefficient sum(w) — _loss_single is additive
across keys and the shared terms are kb-independent. Gradients are
identical by linearity; --cpuprobe measures it.

THE TRUE CROSS-BREATH STATE (what the banks carry):
  gradient-carrying: cur (B,L,H) + the notebook shelf inks (grow 1->K_B)
  detached by law:   garage deposits, lattice snaps (only [-1] read)
  constants:         nb stamp table, slot_mask, fact bufs, waist (a
                     per-forward tensor; reverse steps RECOMPUTE stage-0
                     live inside their own capture, so no dL/dwaist or
                     dL/dvst threading exists — trunk states are frozen)

SUPPORTED CONFIG: the champion family (port242 envs). Every branch this
walker does not thread is REFUSED LOUDLY at start (see REFUSED) — a
true partial beats a false whole.

Envs: ST_CKPT (warm source, default .cache/sharp_port242.safetensors —
the gentle-continuation law's warm parse), ST_STEPS (default 6000),
ST_LR (default 1e-4, gentle), ST_BATCH (default 32, mega-batch),
ST_PING (1=pings live, 0=frozen fact_0 everywhere — rung 1), ST_JIT
(0=eager reference path, 1=TinyJit capture family), ST_THETA (0.9),
ST_OUT (default .cache/step_trainer.safetensors), ST_SNAP_EVERY,
ST_LOG_EVERY, ST_EQN (eq-mode batch rows, default 8), SEED, plus the
standard ALG_* stack (reads and training must carry the TRAINED env).

Modes:
  --train    the trainer (default)
  --eqfwd    rung-1 forward: step-partitioned vs fused, np.array_equal
  --eqbwd    rung-1 backward: reverse-walk grads vs fused grads,
             tolerance 1e-4 relative, printed before asserted
  --cpuprobe DEV=CPU, random params/data: eqfwd bitwise + eqbwd grads +
             (ST_JIT=1) capture-replay determinism — no ckpt, no GPU
  --selftest pure-python bookkeeping on numpy fakes: ladder weights,
             shelf plan, reverse threading vs finite differences,
             contract checks. Zero tinygrad compute.
"""
import math
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import numpy as np


def _envi(k, d="0"):
    return int(os.environ.get(k, d))


# every env whose loop/loss branch this walker does not thread; set any
# of these and the trainer refuses LOUDLY (no silent divergence)
REFUSED = ("ALG_RINGS", "ALG_XOUT", "ALG_CLOCK", "ALG_STELLAR",
           "ALG_CIRCLE", "NAZ_TRAIN", "ALG_CONSUME", "ALG_DEEPSUP",
           "ALG_TRUNK_LORA", "BREATH_DROPOUT", "ALG_MASK_GOLD",
           "CURRICULUM", "RATION_FILE", "ALG_STRAW", "ALG_LSENT",
           "ALG_SYNC", "ALG_MASKRE", "ALG_MINE_BREATHS", "ALG_ROUTER",
           "ALG_INV", "ALG_OPATT", "ALG_CMT_REG", "ALG_ROUTER_GRADED",
           "ALG_FREEZE_DUP", "ALG_TRAIN_ONLY", "ALG_POSCH",
           "ALG_OPCOUNT", "ALG_REF", "ALG_DIAL", "ALG_VALATT",
           "ALG_DUPPTR", "ALG_DETWAVE")

REQ_CTX = ("B", "K_B", "waist", "tokmask", "slot_mask", "bank", "rot2",
           "sync", "drop", "gmod", "revoke", "tail", "reg", "RINGS",
           "XOUT", "XARM", "XR_GRADED", "XR_ELASTIC")
REQ_STATE = ("cur", "breaths", "nb", "nb_st", "garage", "snaps",
             "snaps_g", "rb_last", "m_c", "anchor", "cmt_logits", "x_rel")


def refuse_unsupported():
    bad = [k for k in REFUSED
           if os.environ.get(k, "") not in ("", "0")]
    if bad:
        raise RuntimeError(
            f"step_trainer v1 supports the champion configuration only; "
            f"refused envs set: {bad} — extend the walker deliberately "
            f"(state threading + loss split) before lifting this fence")
    assert _envi("ALG_BREATH", "1") > 1, "step trainer needs ALG_BREATH > 1"


# ===========================================================================
# PURE BOOKKEEPING (selftest-covered; no tinygrad)
# ===========================================================================

def ladder_weights(K, breath_norm=False):
    """The v98 ladder exactly as loss_fn applies it: w_kb = 1 + kb/(K-1),
    normalized by K (or by sum(w) under BREATH_NORM). Returns (w list,
    shared_extra) where shared_extra = sum(w[1:]) — the coefficient the
    stage-0 reverse step applies to the SHARED loss terms beyond the
    live w[0] contribution (total shared coefficient = sum(w))."""
    w = [1.0 + kb / max(K - 1, 1) for kb in range(K)]
    norm = sum(w) if breath_norm else float(K)
    w = [x / norm for x in w]
    return w, sum(w[1:])


def shelf_plan(kb, notebook=True, garage=True, snaps=True):
    """Entering breath kb: (n_nb_in, n_garage_in, n_snap_in). The
    notebook shelf is BORN inside breath 1 (ink0 = ink(cur_0) is
    computed there), so nb_in is 0 at kb==1 and kb entries after."""
    n_nb = (0 if kb == 1 else kb) if notebook else 0
    n_gar = (kb - 1) if garage else 0
    n_sn = (kb - 1) if snaps else 0
    return n_nb, n_gar, n_sn


def reverse_schedule(K_B):
    """Segment indices walked backward: K_B-1 .. 1 (stage-0 is step 0)."""
    return list(range(K_B - 1, 0, -1))


def gold_feed(gold, idx):
    """do_train's feed dict, champion terminal set (the feed-door fence:
    every consumed gold key is asserted present, never silently zero)."""
    need = ["presence", "is_lit", "args", "fspan", "vspan", "ftype",
            "op", "res", "digits", "query", "sel", "is_rel", "is_mod",
            "is_sel", "is_pct", "is_fdiv"]
    ft = _envi("ALG_FTYPES", "4")
    if ft >= 7:
        need += ["is_macro", "digits2", "y"]
    if ft >= 8:
        need += ["is_frac"]
    if ft >= 9:
        need += ["is_chain"]
    if _envi("ALG_WIDE"):
        need += ["sign"]
    if _envi("ALG_BINDBUS") >= 3:
        need += ["bind_ids"]
    missing = [k for k in need if k not in gold]
    assert not missing, f"gold keys missing from the states npz: {missing} " \
        f"(the side-door fence) — re-precompute this cache"
    feed = {k: gold[k][idx] for k in need}
    feed["is_lit_f"] = feed.pop("is_lit")
    feed["arg_dup"] = (gold["arg_dup"][idx] if "arg_dup" in gold
                       else np.zeros_like(feed["is_rel"]))
    return feed


def gold_spec(H, p):
    """(key, per-row shape, is_int) for the champion gold buffer set."""
    L, K, T, ND = H.L_FAC, H.K_VARS, H.T_ALG, H.N_DIG
    ft = _envi("ALG_FTYPES", "4")
    spec = [("presence", (L,), 0), ("is_lit_f", (L,), 0),
            ("args", (L, K), 0), ("fspan", (L, T), 0),
            ("vspan", (K, T), 0), ("ftype", (L,), 1), ("op", (L,), 1),
            ("res", (L,), 1), ("digits", (L, ND), 1), ("query", (), 1),
            ("sel", (L,), 1), ("is_rel", (L,), 0), ("is_mod", (L,), 0),
            ("is_sel", (L,), 0), ("is_pct", (L,), 0),
            ("is_fdiv", (L,), 0), ("arg_dup", (L,), 0)]
    if ft >= 7:
        spec += [("is_macro", (L,), 0), ("digits2", (L, ND), 1),
                 ("y", (L,), 1)]
    if ft >= 8:
        spec += [("is_frac", (L,), 0)]
    if ft >= 9:
        spec += [("is_chain", (L,), 0)]
    if _envi("ALG_WIDE"):
        spec += [("sign", (L,), 0)]
    if _envi("ALG_BINDBUS") >= 3:
        spec += [("bind_ids", (L, 4), 1)]
    return spec


# ===========================================================================
# THE WALKER
# ===========================================================================

class StepWalker:
    """Owns the fixed buffers, the capture family, and the walk drivers.
    ST_JIT=0: the same closures run eagerly (the correctness reference —
    rung 1). ST_JIT=1: each closure is TinyJit-wrapped (zero-arg,
    closure-over-fixed-buffers — do_train's proven capture idiom; the
    driver's assigns between dispatches are the CPU stubs of spec S1)."""

    def __init__(self, H, p, B, jit=False, ping=False, theta=0.9):
        from tinygrad import Tensor, dtypes
        from tinygrad.engine.jit import TinyJit
        self.H, self.p, self.B = H, p, B
        self.Tensor, self.dt = Tensor, dtypes
        self.K_B = _envi("ALG_BREATH", "1")
        assert self.K_B > 1
        self.jit, self.ping, self.theta = bool(jit), bool(ping), theta
        self.inject = bool(_envi("ALG_ALT2")) and "W_fact" in p
        if self.ping and not self.inject:
            print("[walker] NOTICE: ST_PING=1 but no injection port "
                  "(ALG_ALT2 unset or ckpt lacks W_fact) — pings computed "
                  "as leak gauge only; vst stays fact_0-conditioned")
        self.notebook = bool(H.ALG_NOTEBOOK)
        self.garage = (_envi("ALG_BUSGARAGE") >= 2 and "W_gq" in p
                       and "W_bind2" in p)
        self.snaps_on = self.garage and ("alt_g" in p or "W_det" in p)
        self.perslot = bool(H.NB_PERSLOT)
        L, K, T = H.L_FAC, H.K_VARS, H.T_ALG
        HW, HT = H.H_W, H.H_TRUNK
        self.L = L

        def fix(shape, dt=dtypes.float, rg=False):
            npdt = np.float32 if dt == dtypes.float else np.int32
            return Tensor(np.zeros(shape, npdt), dtype=dt,
                          requires_grad=rg).contiguous().realize()
        self.fix = fix
        # inputs (assigned per batch)
        self.b_tr = fix((B, T, HT))
        self.b_tk = fix((B, T))
        self.b_se = fix((B, T), dtypes.int)
        self.b_mask = fix((B, L, L))
        self.b_facts = [fix((B, K, 4)) for _ in range(self.K_B)]
        # state banks (spec S1: entry state per step, thin tensors);
        # cur/nb are the gradient-crossing set -> requires_grad leaves
        self.cur_bank = [fix((B, L, HW), rg=True) for _ in range(self.K_B)]
        nb_shape = (B, L, HW) if self.perslot else (B, HW)
        self.nb_bank = ([fix(nb_shape, rg=True) for _ in range(self.K_B)]
                        if self.notebook else [])
        gd = int(p["W_bind2"].shape[1]) if self.garage else 0
        self.gar_bank = ([fix((B, L, gd)) for _ in range(self.K_B - 1)]
                         if self.garage else [])
        self.snap_bank = ({k: [fix((B, L, 24)), fix((B, L, 24)),
                               fix((B, L, 24)), fix((B, L))]
                           for k in range(1, self.K_B)}
                          if self.snaps_on else {})
        # per-forward constants recomputed at stage-0, then frozen for
        # the forward walk (reverse steps recompute their own, live)
        self.waist_bank = fix((B, T, HW))
        self.vst_base_bank = fix((B, K, HW))
        # grad threading buffers (dL/dstate down the ladder)
        self.G_cur = fix((B, L, HW))
        self.G_nb = ([fix(nb_shape) for _ in range(self.K_B)]
                     if self.notebook else [])
        # seam decode buffers
        self.cur_dec = fix((B, L, HW))
        self.fact_dec = fix((B, K, 4))
        self.dec_keys = (["pres", "ftype", "op", "dig", "args", "res"]
                         + (["dup"] if "h_dup" in p else []))
        # gold buffers + grad accumulators (fixed; the optimizer capture
        # reads the accumulators through p[.].grad bindings)
        self.bg = {k: fix((B,) + shp, dtypes.int if isint else dtypes.float)
                   for k, shp, isint in gold_spec(H, p)}
        self.names = sorted(p.keys())
        self.gbufs = {n: fix(tuple(p[n].shape)) for n in self.names}
        self.nb_st_const = (Tensor(H.NB_STAMPS, dtype=dtypes.float)
                            .contiguous().realize() if self.notebook else None)
        self.w, self.shared_extra = ladder_weights(
            self.K_B, bool(_envi("BREATH_NORM")))
        self.rot2 = None
        self.emit_keys = None
        self.rs_layout = {}
        wrap = (TinyJit if self.jit else (lambda f: f))
        self.s0_fn = wrap(self._mk_s0())
        self.fwd_fns = {k: wrap(self._mk_fwd(k))
                        for k in range(1, self.K_B)}
        self.rs_fns = {k: wrap(self._mk_rs(k)) for k in range(1, self.K_B)}
        self.rs0_fn = wrap(self._mk_rs0())
        self.dec_fn = wrap(self._mk_dec())

    # ---- plumbing -------------------------------------------------------
    def put(self, buf, arr):
        npdt = np.float32 if buf.dtype == self.dt.float else np.int32
        buf.assign(self.Tensor(np.ascontiguousarray(arr, dtype=npdt),
                               dtype=buf.dtype).contiguous()).realize()

    def _tap_call(self, fact_buf):
        """One stage-0 pass through the REAL forward under hold: builds
        waist/vst/fst/qst + ctx/state + heads_of, skips the fused loop."""
        H = self.H
        H._STEP_TAP = {"hold": True}
        try:
            o0 = H.forward(self.p, self.b_tr, self.b_tk, self.b_se,
                           slot_mask=self.b_mask, fact_buf=fact_buf)
            tap = H._STEP_TAP
        finally:
            H._STEP_TAP = None
        return o0, tap

    def prime(self):
        """One eager tap call (never jitted): captures the rot2 pure
        function + the emission key set, warms _SGC, checks contracts."""
        _o0, tap = self._tap_call(self.b_facts[0])
        for kk in REQ_CTX:
            assert kk in tap["ctx"], f"ctx contract broke: missing {kk}"
        for kk in REQ_STATE:
            assert kk in tap["state"], f"state contract broke: missing {kk}"
        self.rot2 = tap["ctx"]["rot2"]
        self.emit_keys = sorted(tap["heads_of"](tap["fst"]).keys())

    def _mk_ctx(self, waist):
        H = self.H
        return {"B": self.B, "K_B": self.K_B, "waist": waist,
                "tokmask": self.b_tk, "slot_mask": self.b_mask,
                "bank": H._make_bank(self.p, waist, self.b_tk, self.B),
                "rot2": self.rot2, "sync": None, "drop": None,
                "gmod": None, "revoke": None, "tail": None, "reg": None,
                "RINGS": False, "XOUT": False, "XARM": "dump",
                "XR_GRADED": 0.5, "XR_ELASTIC": 0.15}

    def _mk_state(self, k, cur):
        n_nb, n_gar, _ = shelf_plan(k, self.notebook, self.garage,
                                    self.snaps_on)
        return {"cur": cur, "breaths": [],
                "nb": ([self.nb_bank[j] for j in range(n_nb)]
                       if (self.notebook and k > 1) else None),
                "nb_st": (self.nb_st_const
                          if (self.notebook and k > 1) else None),
                "garage": ([self.gar_bank[j] for j in range(n_gar)]
                           if self.garage else None),
                "snaps": ([tuple(self.snap_bank[k - 1])]
                          if (self.snaps_on and k > 1) else []),
                "snaps_g": [], "rb_last": None, "m_c": None,
                "anchor": None, "cmt_logits": None, "x_rel": None}

    # ---- capture family -------------------------------------------------
    def _mk_s0(self):
        def s0():
            _o0, tap = self._tap_call(self.b_facts[0])
            # banks are PURE VALUES (the reverse walk recomputes live);
            # detach cuts any chance of cross-step graph chaining
            return [tap["waist"].detach(), tap["vst_base"].detach(),
                    tap["fst"].detach()]
        return s0

    def _mk_fwd(self, k):
        def fwd():
            state = self._mk_state(k, self.cur_bank[k - 1])
            self.H.breath_step(self.p, state, k, self._mk_ctx(self.waist_bank))
            outs = [state["cur"]]
            if self.notebook:
                outs += state["nb"][-2:] if k == 1 else [state["nb"][-1]]
            if self.garage:
                outs.append(state["garage"][-1])
            if self.snaps_on:
                outs += list(state["snaps"][-1])
            return [t.detach() for t in outs]   # banks are pure values
        return fwd

    def _mk_dec(self):
        def dec():
            vstk = (self.H._fact_inject(self.p, self.vst_base_bank,
                                        self.fact_dec)
                    if self.inject else self.vst_base_bank)
            o = self.H._heads_of(self.p, self.cur_dec, vstk, self.B)
            return [o[kk].detach() for kk in self.dec_keys]
        return dec

    def _mk_rs(self, k):
        n_nb, _, _ = shelf_plan(k, self.notebook, self.garage, self.snaps_on)

        def rs():
            for n in self.names:
                self.p[n].grad = None
            leaves = [self.cur_bank[k - 1]] + \
                [self.nb_bank[j] for j in range(n_nb)]
            for t in leaves:
                t.grad = None
            # stage-0 recomputed LIVE (waist/vst/bank reborn inside this
            # capture -> stage-0 params get this segment's contribution;
            # no dL/dwaist threading exists by construction)
            o0, tap = self._tap_call(self.b_facts[k])
            state = self._mk_state(k, self.cur_bank[k - 1])
            self.H.breath_step(self.p, state, k, tap["ctx"])
            cur_out = state["cur"]
            scalar = (cur_out * self.G_cur).sum()
            if self.notebook:
                for j, ink in enumerate(state["nb"]):
                    scalar = scalar + (ink * self.G_nb[j]).sum()
            # breath-k ladder term: emissions LIVE on vst_k, shared keys
            # DETACHED (they pay once, at stage-0, with sum(w))
            full = {kk: vv.detach() for kk, vv in o0.items()}
            full.update(tap["heads_of"](cur_out))
            term = self.H._loss_single(full, self.bg) * self.w[k]
            (scalar + term).backward()
            ret = [term.detach(), self.cur_bank[k - 1].grad]
            ret += [self.nb_bank[j].grad for j in range(n_nb)]
            gnames = []
            for n in self.names:
                g = self.p[n].grad
                if g is not None:
                    gnames.append(n)
                    ret.append(g)
            if k in self.rs_layout:
                assert self.rs_layout[k] == gnames, "rs layout drifted"
            self.rs_layout[k] = gnames
            return ret
        return rs

    def _mk_rs0(self):
        def rs0():
            for n in self.names:
                self.p[n].grad = None
            o0, tap = self._tap_call(self.b_facts[0])
            scalar = (tap["fst"] * self.G_cur).sum()
            term0 = self.H._loss_single(o0, self.bg) * self.w[0]
            shared = dict(o0)
            for ek in self.emit_keys:
                shared[ek] = o0[ek].detach()
            term_sh = (self.H._loss_single(shared, self.bg)
                       * self.shared_extra)
            (scalar + term0 + term_sh).backward()
            ret = [term0.detach()]
            gnames = []
            for n in self.names:
                g = self.p[n].grad
                if g is not None:
                    gnames.append(n)
                    ret.append(g)
            if 0 in self.rs_layout:
                assert self.rs_layout[0] == gnames, "rs0 layout drifted"
            self.rs_layout[0] = gnames
            return ret
        return rs0

    # ---- drivers --------------------------------------------------------
    def load_batch(self, states, tokmask, sent, masks, idx, feed=None):
        self.put(self.b_tr, states[idx])
        self.put(self.b_tk, tokmask[idx])
        self.put(self.b_se, sent[idx])
        self.put(self.b_mask, masks)
        if feed is not None:
            for kk, buf in self.bg.items():
                assert kk in feed, f"feed missing gold {kk} (feed door)"
                self.put(buf, feed[kk])

    def walk_forward(self, fact0, se_np=None, nv=None, ma=None):
        """Spec S1 forward: stage-0 + K_B-1 dispatches with CPU seam
        stubs. Returns (facts-per-seam counts, the seam fact bufs are
        left in b_facts for the reverse walk)."""
        H = self.H
        self.put(self.b_facts[0], fact0)
        r = self.s0_fn()
        self.waist_bank.assign(r[0])
        self.vst_base_bank.assign(r[1])
        self.cur_bank[0].assign(r[2])
        self.Tensor.realize(self.waist_bank, self.vst_base_bank,
                            self.cur_bank[0])
        fact_cur = fact0
        rates = []
        for k in range(1, self.K_B):
            outs = self.fwd_fns[k]()
            todo = [self.cur_bank[k]]
            self.cur_bank[k].assign(outs[0])
            i = 1
            if self.notebook:
                if k == 1:
                    self.nb_bank[0].assign(outs[1])
                    self.nb_bank[1].assign(outs[2])
                    todo += [self.nb_bank[0], self.nb_bank[1]]
                    i = 3
                else:
                    self.nb_bank[k].assign(outs[1])
                    todo.append(self.nb_bank[k])
                    i = 2
            if self.garage:
                self.gar_bank[k - 1].assign(outs[i])
                todo.append(self.gar_bank[k - 1])
                i += 1
            if self.snaps_on:
                for jj in range(4):
                    self.snap_bank[k][jj].assign(outs[i + jj])
                todo += self.snap_bank[k]
            self.Tensor.realize(*todo)
            if self.ping:
                # the seam stub: decode confident slots on vst(fact_{k-1}),
                # ping the organ, pack the fixed (B,24,4) buffer
                self.put(self.fact_dec, fact_cur)
                self.cur_dec.assign(self.cur_bank[k]).realize()
                dec = self.dec_fn()
                onp = {kk: t.numpy() for kk, t in zip(self.dec_keys, dec)}
                fact_cur = H.alt2_fact_buf(onp, se_np, nv, ma,
                                           theta=self.theta)
                rates.append(int((fact_cur[:, :, 0] > 0).sum()))
            self.put(self.b_facts[k], fact_cur)
        return rates

    def walk_backward(self):
        """Spec S1 backward: reverse capture per segment, dL/dstate
        threaded through G_cur/G_nb, param grads accumulated into the
        fixed gbufs. Returns (walk loss = the TRUE fused-ladder value,
        set of param names that received grads)."""
        zero = [self.G_cur] + self.G_nb
        for g in zero:
            g.assign(g * 0.0)
        self.Tensor.realize(*zero)
        seen = set()
        loss = 0.0

        def acc(gnames, gs):
            todo = []
            for n, g in zip(gnames, gs):
                if n in seen:
                    self.gbufs[n].assign(self.gbufs[n] + g)
                else:
                    self.gbufs[n].assign(g)
                    seen.add(n)
                todo.append(self.gbufs[n])
            self.Tensor.realize(*todo)

        for k in reverse_schedule(self.K_B):
            n_nb, _, _ = shelf_plan(k, self.notebook, self.garage,
                                    self.snaps_on)
            ret = self.rs_fns[k]()
            # PIN every output BEFORE touching the G buffers: the lazy
            # grad graphs READ G_cur/G_nb, and assigning the new thread
            # values first makes them recompute against the mutated
            # buffers (measured: exactly-doubled segment grads at K=2 —
            # the probe's second lesson). realize() is a no-op under JIT.
            self.Tensor.realize(*ret)
            loss += float(ret[0].numpy())
            i = 2 + n_nb
            gnames = self.rs_layout[k]
            todo = [self.G_cur]
            self.G_cur.assign(ret[1])
            for j in range(n_nb):
                self.G_nb[j].assign(ret[2 + j])
                todo.append(self.G_nb[j])
            self.Tensor.realize(*todo)
            acc(gnames, ret[i:i + len(gnames)])
        ret0 = self.rs0_fn()
        self.Tensor.realize(*ret0)     # same pin (G_cur read by the dot)
        loss += float(ret0[0].numpy())
        acc(self.rs_layout[0], ret0[1:1 + len(self.rs_layout[0])])
        return loss, seen

    def read_final(self):
        """Assembled final output dict (shared + final-breath heads on
        the last seam's vst) — the eq-mode read; eager only."""
        o0, tap = self._tap_call(self.b_facts[self.K_B - 1])
        heads = tap["heads_of"](self.cur_bank[self.K_B - 1])
        return {**o0, **heads}


# ===========================================================================
# SHARED SETUP HELPERS
# ===========================================================================

def _import_head():
    import phase1_algebra_head as H
    for fn in ("breath_step", "_make_bank", "_heads_of", "_fact_inject"):
        assert hasattr(H, fn), \
            (f"phase1_algebra_head lacks {fn} — run "
             f"scripts/apply_step_trainer.py first (rung 0)")
    return H


def load_ckpt_into(H, p, path):
    from tinygrad.nn.state import safe_load
    sd = safe_load(path)
    assert set(sd.keys()) == set(p.keys()), \
        (sorted(set(sd) - set(p))[:4], sorted(set(p) - set(sd))[:4])
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()


def prep_masks_facts(H, p, samples, states, tokmask, sent):
    """do_train's mask-prep pass on the factored module: masks + fact_0
    from the WARM head's own breath-0 parse (the warm-parse law — the
    warm source's parse feeds the first seam, never a newborn's)."""
    from tinygrad import Tensor, dtypes
    n = states.shape[0]
    L, K = H.L_FAC, H.K_VARS
    alt2 = bool(_envi("ALG_ALT2"))
    MASKS = np.zeros((n, L, L), np.float32)
    FACTS = np.zeros((n, K, 4), np.float32) if alt2 else None
    for s0 in range(0, n, 8):
        sl = np.arange(s0, min(s0 + 8, n))
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        out0 = H.forward(
            p, Tensor(states[sl_p].astype(np.float32), dtype=dtypes.float),
            Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float),
            Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int))
        o0 = {k: out0[k].realize().numpy() for k in ("fat", "args", "res")}
        MASKS[sl] = H.build_slot_masks(o0, sent[sl_p])[:len(sl)]
        if FACTS is not None:
            ka = (("pres", "ftype", "op", "dig")
                  + (("dup",) if "dup" in out0 else ()))
            oa = {**o0, **{k: out0[k].realize().numpy() for k in ka}}
            nv = np.array([samples[int(i)].get("n_vars", K) for i in sl_p])
            ma = np.array([samples[int(i)].get("m", 0) for i in sl_p])
            FACTS[sl] = H.alt2_fact_buf(oa, sent[sl_p], nv, ma)[:len(sl)]
    return MASKS, FACTS


def _row_meta(H, samples, idx):
    nv = np.array([samples[int(i)].get("n_vars", H.K_VARS) for i in idx])
    ma = np.array([samples[int(i)].get("m", 0) for i in idx])
    return nv, ma


# ===========================================================================
# RUNG-1 EQUIVALENCE MODES (real ckpt + real fixture; run on the box)
# ===========================================================================

def _eq_setup():
    refuse_unsupported()
    H = _import_head()
    from tinygrad import Tensor, dtypes
    ckpt = os.environ.get("ST_CKPT", ".cache/sharp_port242.safetensors")
    p = H.build_params(0)
    load_ckpt_into(H, p, ckpt)
    samples, states, tokmask, gold, sent = H.load_alg("train")
    nrows = int(os.environ.get("ST_EQN", "8"))
    idx = np.arange(nrows)
    MASKS, FACTS = prep_masks_facts(H, p, [samples[int(i)] for i in idx],
                                    states[idx], tokmask[idx], sent[idx])
    fact0 = FACTS if FACTS is not None else np.zeros(
        (nrows, H.K_VARS, 4), np.float32)
    w = StepWalker(H, p, nrows, jit=bool(_envi("ST_JIT")), ping=False,
                   theta=float(os.environ.get("ST_THETA", "0.9")))
    w.load_batch(states, tokmask, sent, MASKS, idx,
                 feed=gold_feed(gold, idx))
    w.put(w.b_facts[0], fact0)   # BEFORE any fused baseline: vst is
    # fact-injected — a zeros-vs-facts mismatch here masquerades as a
    # walker bug on every vst-keyed emission (the probe's own lesson)
    w.prime()
    print(f"[eq] ckpt={ckpt} rows={nrows} jit={int(w.jit)} "
          f"inject={int(w.inject)}")
    return H, p, w, fact0


def _fused_out(H, p, w):
    return H.forward(p, w.b_tr, w.b_tk, w.b_se, slot_mask=w.b_mask,
                     fact_buf=w.b_facts[0])


def run_eqfwd():
    H, p, w, fact0 = _eq_setup()
    w.walk_forward(fact0)
    o_p = w.read_final()
    o_f = _fused_out(H, p, w)
    keys = sorted(set(w.emit_keys) | {"query", "fat", "vat", "bind"})
    keys = [k for k in keys if k in o_f]
    tol = float(os.environ.get("ST_EQ_TOL", "0"))
    bad = []
    for k in keys:
        a = o_f[k].realize().numpy()
        b = o_p[k].realize().numpy()
        eq = np.array_equal(a, b)
        d = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
        print(f"  {k:8s} array_equal={eq}  maxabs={d:.3e}")
        if not eq and d > tol:
            bad.append(k)
    print(f"[eqfwd] {'PASS' if not bad else f'FAIL {bad}'} "
          f"(bar: bitwise, ST_EQ_TOL={tol:g}; the realize boundary "
          f"between segments changes kernel fusion — measured ~1e-7 "
          f"single-ulp diffs on CPU; a re-pin is a REGISTERED call, "
          f"never a silent one)")
    assert not bad, "rung-1 forward equivalence FAILED (loudly, per the bar)"


def _fused_grads(H, p, w):
    for n in p:
        p[n].grad = None
    o = _fused_out(H, p, w)
    l = H.loss_fn(o, w.bg)
    l.backward()
    lv = float(l.realize().numpy())
    g = {n: p[n].grad.realize().numpy().copy() for n in p
         if p[n].grad is not None}
    for n in p:
        p[n].grad = None
    return lv, g


def _grad_compare(fused_g, walker, seen, tol):
    """Tolerance pinned BEFORE comparison: per-param PASS iff
    |d|max <= tol * |fused|max  OR  |d|max <= 1e-6 * global_scale.
    The absolute floor exists for the softmax-shift-invariant params
    (K-projection biases: a constant shift of attention scores cancels
    in softmax, so their TRUE gradient is identically zero and both
    sides are rounding dust — relative-only would divide dust by dust).
    Any structural error still trips the floor. Everything prints
    BEFORE the assert."""
    assert set(fused_g) == set(seen), \
        (f"grad SETS differ: fused-only "
         f"{sorted(set(fused_g) - set(seen))[:4]}, walk-only "
         f"{sorted(set(seen) - set(fused_g))[:4]} (two-terminal law)")
    gscale = max(float(np.abs(a).max()) for a in fused_g.values())
    floor = 1e-6 * max(gscale, 1e-3)
    rows = []
    for n in sorted(fused_g):
        a = fused_g[n]
        b = walker.gbufs[n].numpy()
        d = float(np.abs(a - b).max())
        am = float(np.abs(a).max())
        ok = (d <= tol * am) or (d <= floor)
        rows.append((d / (am + 1e-12), d, am, n, ok))
    rows.sort(reverse=True)
    print(f"[eqbwd] param-grad diffs (rel tol {tol:g}, dust floor "
          f"{floor:.1e} = 1e-6 x grad scale {gscale:.2e}); worst first:")
    for rel, d, am, n, ok in rows[:12]:
        print(f"    rel={rel:.3e} abs={d:.3e} |fused|={am:.3e} "
              f"{'ok' if ok else 'FAIL'}  {n}")
    bad = [n for _, _, _, n, ok in rows if not ok]
    worst = rows[0][0] if rows else 0.0
    print(f"[eqbwd] {len(rows) - len(bad)}/{len(rows)} params pass"
          + (f"; FAILING: {bad}" if bad else ""))
    assert not bad, f"rung-1 backward tolerance EXCEEDED on {bad}"
    return worst


def run_eqbwd():
    H, p, w, fact0 = _eq_setup()
    lf, fused_g = _fused_grads(H, p, w)
    w.walk_forward(fact0)
    lw, seen = w.walk_backward()
    print(f"[eqbwd] loss fused={lf:.6f} walk={lw:.6f} "
          f"delta={abs(lf - lw):.3e}")
    _grad_compare(fused_g, w, seen, 1e-4)
    print("[eqbwd] PASS within pinned tolerance")


# ===========================================================================
# THE TRAINER
# ===========================================================================

def run_train():
    refuse_unsupported()
    H = _import_head()
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.optim import AdamW
    from tinygrad.nn.state import safe_save
    from tinygrad.engine.jit import TinyJit

    steps = int(os.environ.get("ST_STEPS", "6000"))
    lr = float(os.environ.get("ST_LR", "1e-4"))        # gentle continuation
    B = int(os.environ.get("ST_BATCH", "32"))          # the mega-batch
    seed = int(os.environ.get("SEED", "242"))
    ping = bool(_envi("ST_PING", "1"))
    jit = bool(_envi("ST_JIT", "1"))
    out_ckpt = os.environ.get("ST_OUT", ".cache/step_trainer.safetensors")
    ckpt = os.environ.get("ST_CKPT", ".cache/sharp_port242.safetensors")
    log_every = int(os.environ.get("ST_LOG_EVERY", "50"))
    snap_every = int(os.environ.get("ST_SNAP_EVERY", "0"))

    p = H.build_params(seed)
    load_ckpt_into(H, p, ckpt)     # warm source: the warm-parse law
    print(f"[train] warm from {ckpt} steps={steps} lr={lr} B={B} "
          f"ping={int(ping)} jit={int(jit)}", flush=True)

    samples, states, tokmask, gold, sent = H.load_alg("train")
    n = states.shape[0]
    print("[train] mask/fact prep pass (the warm head's own parse) ...",
          flush=True)
    MASKS, FACTS = prep_masks_facts(H, p, samples, states, tokmask, sent)
    print(f"[train] masks ready (mean degree "
          f"{MASKS.sum(-1).mean():.1f}/{H.L_FAC})"
          + (f", fact_0 rows carrying facts: "
             f"{int((FACTS[:, :, 0] > 0).any(1).sum())}/{n}"
             if FACTS is not None else ""), flush=True)

    w = StepWalker(H, p, B, jit=jit, ping=ping,
                   theta=float(os.environ.get("ST_THETA", "0.9")))
    w.prime()
    opt = AdamW(list(p.values()), lr=lr, weight_decay=0.01)
    for nm in w.names:
        p[nm].grad = w.gbufs[nm]   # the optimizer capture reads the
        # fixed accumulators; rebound before every step (rs fns None them)

    def opt_step():
        opt.step()
        return list(p.values())    # closure-assign quirk: return targets
    opt_fn = TinyJit(opt_step) if jit else opt_step

    rng = np.random.RandomState(seed)
    lr_min = lr / 30.0
    rate_sum = np.zeros(w.K_B - 1, np.float64)
    rate_n = 0
    skipped = 0
    t0 = time.time()
    for s in range(steps):
        cur_lr = lr_min + 0.5 * (lr - lr_min) * (
            1 + math.cos(math.pi * s / steps))
        opt.lr.assign(Tensor([cur_lr], dtype=dtypes.float)).realize()
        idx = rng.choice(n, B, replace=False)         # flat mix, always
        w.load_batch(states, tokmask, sent, MASKS[idx], idx,
                     feed=gold_feed(gold, idx))
        if _envi("ALG_SHELF_CIRCLE") >= 2 and not os.environ.get("SC_EVAL"):
            sev = getattr(H, "_SEV", None)
            if sev is not None:                       # the pulse, per step
                sev.assign(Tensor(
                    [1.0 if np.random.rand()
                     < float(os.environ.get("SC_P", "0.5")) else 0.0],
                    dtype=sev.dtype)).realize()
        fact0 = (FACTS[idx] if FACTS is not None
                 else np.zeros((B, H.K_VARS, 4), np.float32))
        nv, ma = _row_meta(H, samples, idx)
        rates = w.walk_forward(fact0, sent[idx].astype(np.int32), nv, ma)
        if rates:
            rate_sum += np.array(rates, np.float64) / B
            rate_n += 1
        loss, seen = w.walk_backward()
        if s == 0:
            missing = [nm for nm in w.names if nm not in seen]
            assert not missing, \
                f"params with NO grad across the whole walk: {missing}"
        if not np.isfinite(loss):
            skipped += 1
            print(f"  step {s}: NON-FINITE walk loss — optimizer SKIPPED "
                  f"({skipped} total)", flush=True)
        else:
            for nm in w.names:
                p[nm].grad = w.gbufs[nm]
            opt_fn()
        if s % log_every == 0 or s == steps - 1:
            rr = (rate_sum / max(rate_n, 1)).round(2).tolist()
            print(f"  step {s:5d} loss={loss:.4f} lr={cur_lr:.1e} "
                  f"({(time.time() - t0) / (s + 1):.2f}s/step) "
                  f"facts/item/breath={rr}", flush=True)
            rate_sum[:] = 0.0
            rate_n = 0
        if snap_every and (s + 1) % snap_every == 0:
            sp = out_ckpt.replace(".safetensors", f"_s{s + 1}.safetensors")
            safe_save(p, sp)
            print(f"  [snap @{s + 1}] -> {sp}", flush=True)
    safe_save(p, out_ckpt)
    print(f"[train] saved {out_ckpt} (final-step params; selection is "
          f"external reads — the wild-val template)", flush=True)


# ===========================================================================
# CPU PROBE — the full rung-1 logic on random params/data, no ckpt/GPU
# ===========================================================================

def run_cpuprobe():
    os.environ["DEV"] = "CPU"
    CHAMP = dict(ALG2="1", ALG_FTYPES="9", ALG_DUP="1", ALG_HW="512",
                 ALG_WIDE="1", ALG_BREATH="7", ALG_NOTEBOOK="1",
                 ALG_SIXWAVE="1", NB_PERSLOT="1", ALG_BINDBUS="7",
                 ALG_BIND_D="512", BIND_CODES=".cache/bindbus_codes512.npz",
                 ALG_BUSGARAGE="2", ALG_SHELF_CIRCLE="2", SC_EVAL="0",
                 ALG_ALTMASK="1", ALG_ALT21="1", ALG_ALT2="1")
    os.environ.update(CHAMP)
    refuse_unsupported()
    H = _import_head()
    from tinygrad import Tensor, dtypes
    jit = bool(_envi("ST_JIT"))
    B = 2
    rng = np.random.RandomState(7)
    p = H.build_params(0)
    for k in p:      # perturb: no exact-zero gates -> every grad nonzero
        p[k].assign(p[k].detach() + Tensor(
            (rng.randn(*[int(x) for x in p[k].shape]) * 0.01)
            .astype(np.float32), dtype=p[k].dtype)).realize()

    w = StepWalker(H, p, B, jit=jit, ping=False)
    # synthetic inputs
    ts = (rng.randn(B, H.T_ALG, H.H_TRUNK) * 0.1).astype(np.float32)
    tk = np.zeros((B, H.T_ALG), np.float32)
    tk[:, :200] = 1.0
    se = np.stack([np.minimum(np.arange(H.T_ALG) // 16, 31)] * B
                  ).astype(np.int32)
    mk = (rng.rand(B, H.L_FAC, H.L_FAC) < 0.4).astype(np.float32)
    for b in range(B):
        np.fill_diagonal(mk[b], 1.0)
    fact0 = np.zeros((B, H.K_VARS, 4), np.float32)
    fact0[0, 3] = (1.0, 0.1, 0.4, 0.7)
    fact0[1, 5] = (1.0, 0.0, 0.2, 0.9)
    # synthetic gold (in-range; semantics irrelevant for grad equality)
    L, K, T, ND = H.L_FAC, H.K_VARS, H.T_ALG, H.N_DIG
    feed = {}
    for key, shp, isint in gold_spec(H, p):
        full = (B,) + shp
        if isint:
            hi = {"ftype": 3, "op": 2, "res": K, "digits": 10, "query": K,
                  "sel": 2, "digits2": 10, "y": K, "bind_ids": 24}[key]
            feed[key] = rng.randint(0, hi, full).astype(np.int32)
        else:
            feed[key] = (rng.rand(*full) < 0.5).astype(np.float32)
    feed["presence"][:, :8] = 1.0     # nonzero masks everywhere
    feed["is_rel"][:, :4] = 1.0
    feed["fspan"][:, :, :40] = (rng.rand(B, L, 40) < 0.3)
    feed["vspan"][:, :, :40] = (rng.rand(B, K, 40) < 0.3)
    args2 = np.zeros((B, L, K), np.float32)
    for b in range(B):
        for j in range(L):
            args2[b, j, rng.choice(K, 2, replace=False)] = 1.0
    feed["args"] = args2

    class _S:                        # tiny row-indexable shims
        def __init__(self, a):
            self.a = a

        def __getitem__(self, i):
            return self.a[i]
    idx = np.arange(B)
    w.load_batch(_S(ts), _S(tk), _S(se), mk, idx, feed=feed)
    w.put(w.b_facts[0], fact0)       # before the baseline: vst is
    w.prime()                        # fact-injected (the probe's lesson)

    print(f"[cpuprobe] jit={int(jit)} B={B} — fused baseline ...")
    lf, fused_g = _fused_grads(H, p, w)
    o_f = _fused_out(H, p, w)
    keys = sorted(set(w.emit_keys) | {"query", "fat", "vat", "bind"})
    keys = [k for k in keys if k in o_f]
    o_f_np = {k: o_f[k].realize().numpy() for k in keys}

    n_rep = 3 if jit else 1          # jit: eager-trace / capture / replay
    prev = None
    for rep in range(n_rep):
        w.put(w.b_facts[0], fact0)   # identical inputs every rep
        w.walk_forward(fact0)
        if rep == 0:
            o_p = w.read_final()
            worst_fwd = 0.0
            n_bit = 0
            for k in keys:
                a, b = o_f_np[k], o_p[k].realize().numpy()
                d = float(np.abs(a.astype(np.float64)
                                 - b.astype(np.float64)).max())
                worst_fwd = max(worst_fwd, d)
                n_bit += int(np.array_equal(a, b))
                if d > 1e-5:
                    print(f"[cpuprobe] eqfwd STRUCTURAL diff {k}: {d:.3e}")
            print(f"[cpuprobe] eqfwd: {n_bit}/{len(keys)} keys bitwise, "
                  f"worst maxabs {worst_fwd:.3e} (segment-realize fusion "
                  f"noise is ~1e-7; anything above 1e-5 is structural)")
            assert worst_fwd <= 1e-5, "cpuprobe forward equivalence FAILED"
        lw, seen = w.walk_backward()
        print(f"[cpuprobe] rep {rep}: loss fused={lf:.6f} walk={lw:.6f} "
              f"delta={abs(lf - lw):.3e}")
        assert abs(lf - lw) <= 1e-4 * max(abs(lf), 1.0), "loss value drifted"
        worst = _grad_compare(fused_g, w, seen, 1e-4)
        cur = {nm: w.gbufs[nm].numpy().copy() for nm in w.names}
        if prev is not None:
            rep_bad = [nm for nm in w.names
                       if not np.array_equal(prev[nm], cur[nm])]
            print(f"[cpuprobe] rep {rep} vs rep {rep - 1} grad "
                  f"determinism: "
                  f"{'IDENTICAL' if not rep_bad else f'DRIFT {rep_bad[:5]}'}")
            assert not rep_bad, "capture replay is not deterministic"
        prev = cur
    print(f"[cpuprobe] PASS (jit={int(jit)}): forward within fusion "
          f"noise, loss matched, 78-param grad walk passes the pinned "
          f"dual criterion"
          + (", replays deterministic" if jit else ""))


# ===========================================================================
# SELFTEST — pure-python bookkeeping on fakes (zero tinygrad compute)
# ===========================================================================

def selftest():
    import py_compile
    py_compile.compile(os.path.abspath(__file__), doraise=True)

    # --- ladder weights: the v98 ladder + the shared coefficient ---
    w, extra = ladder_weights(7)
    assert abs(sum(w) - 1.5) < 1e-12          # sum(1 + k/6)/7 = 10.5/7
    assert abs(w[0] - 1.0 / 7) < 1e-12 and abs(w[6] - 2.0 / 7) < 1e-12
    assert abs(extra - (sum(w) - w[0])) < 1e-12
    wn, extran = ladder_weights(7, breath_norm=True)
    assert abs(sum(wn) - 1.0) < 1e-12          # BREATH_NORM: true scale
    assert abs(extran - (1.0 - wn[0])) < 1e-12

    # --- shelf plan: notebook born inside breath 1; garage lags one ---
    assert shelf_plan(1) == (0, 0, 0)
    assert shelf_plan(2) == (2, 1, 1)
    assert shelf_plan(6) == (6, 5, 5)
    assert shelf_plan(4, notebook=False) == (0, 3, 3)
    assert shelf_plan(4, garage=False, snaps=False) == (4, 0, 0)
    assert reverse_schedule(7) == [6, 5, 4, 3, 2, 1]

    # --- the reverse-threading equations vs finite differences on a toy
    # mirror of the walk structure: scalar state s_k, notebook-style
    # shelf born inside segment 1 (ink0 = c*s_0), per-breath ladder
    # terms, shelf read by every later segment. Same banking, same
    # dot-seeding, same passthrough-grad threading as StepWalker. ---
    Kb = 7
    rng = np.random.RandomState(3)
    a = rng.randn(Kb)
    b = rng.randn(Kb) * 0.3
    c = rng.randn() * 0.5
    t = rng.randn(Kb)
    wl, _ = ladder_weights(Kb)

    def fwd(x, a, b, c):
        s = [x]
        shelf = []
        for k in range(1, Kb):
            if k == 1:
                shelf = [c * s[0]]
            sk = a[k] * s[k - 1] + b[k] * sum(shelf)
            shelf.append(c * sk)
            s.append(sk)
        return s, shelf

    def loss_of(x, a, b, c):
        s, _ = fwd(x, a, b, c)
        return sum(wl[k] * (s[k] - t[k]) ** 2 for k in range(Kb))

    x0 = 0.7
    s_bank, _ = fwd(x0, a, b, c)               # the forward walk banks
    G_s = 0.0
    G_nb = np.zeros(Kb)
    dA = np.zeros(Kb)
    dB = np.zeros(Kb)
    dC = 0.0
    for k in range(Kb - 1, 0, -1):             # the reverse walk
        # recompute segment k from banked leaves (mirror of rs_k);
        # at k==1 the shelf is BORN inside (ink0 = c*s_0 — a function
        # of the segment input, exactly the real notebook's birth)
        sm1 = s_bank[k - 1]
        shelf_in = ([c * sm1] if k == 1
                    else [c * s_bank[j] for j in range(k)])
        read = sum(shelf_in)
        sk = a[k] * sm1 + b[k] * read
        assert abs(sk - s_bank[k]) < 1e-12     # bit-faithful recompute
        # scalar = sk*G_s + sum_j shelf_out[j]*G_nb[j] + w_k*(sk-t_k)^2
        # shelf_out = shelf_in passthrough + ink_k (+ ink0 inside k==1)
        A = G_s + 2 * wl[k] * (sk - t[k]) + c * G_nb[k]   # grad reaching sk
        dA[k] += A * sm1
        dB[k] += A * read
        dC += G_nb[k] * sk                     # ink_k = c*s_k, born here
        if k == 1:
            g_ink0 = A * b[1] + G_nb[0]        # read + dot, born here too
            G_s = A * a[1] + g_ink0 * c
            dC += g_ink0 * sm1
        else:
            G_s = A * a[k]
            G_nb[:k] = [A * b[k] + G_nb[j] for j in range(k)]
    # stage 0: G_s reaches s_0 = x; breath-0 ladder term
    dX = G_s + 2 * wl[0] * (s_bank[0] - t[0])
    eps = 1e-6

    def num(f, v):
        return (f(v + eps) - f(v - eps)) / (2 * eps)
    dX_num = num(lambda v: loss_of(v, a, b, c), x0)
    dC_num = num(lambda v: loss_of(x0, a, b, v), c)
    assert abs(dX - dX_num) < 1e-5, (dX, dX_num)
    for k in range(1, Kb):
        ak = a.copy()

        def fa(v, k=k, ak=ak):
            ak[k] = v
            return loss_of(x0, ak, b, c)
        assert abs(dA[k] - num(fa, a[k])) < 1e-5, (k, dA[k])
        bk = b.copy()

        def fb(v, k=k, bk=bk):
            bk[k] = v
            return loss_of(x0, a, bk, c)
        assert abs(dB[k] - num(fb, b[k])) < 1e-5, (k, dB[k])
    assert abs(dC - dC_num) < 1e-4, (dC, dC_num)

    # --- contract checks against the patched head (import-only, no GPU)
    os.environ.setdefault("ALG_BREATH", "7")
    H = _import_head()
    assert not hasattr(H, "Tensor"), "head grew a module-level Tensor"
    assert hasattr(H, "_STEP_TAP") and H._STEP_TAP is None
    import inspect
    src = inspect.getsource(H.breath_step)
    for kk in REQ_CTX:
        assert f'ctx["{kk}"]' in src, f"breath_step lost ctx[{kk!r}]"
    for kk in REQ_STATE:
        assert f'state["{kk}"]' in src, f"breath_step lost state[{kk!r}]"

    # --- the commit adapter is importable and packs facts (CPU organs)
    Bp, L, K, ND = 1, H.L_FAC, H.K_VARS, H.N_DIG
    onp = {"pres": np.full((Bp, L), -9.0), "ftype": np.zeros((Bp, L, 3)),
           "op": np.zeros((Bp, L, 2)), "dig": np.zeros((Bp, L, ND, 10)),
           "args": np.full((Bp, L, K), -9.0), "res": np.zeros((Bp, L, K))}
    onp["pres"][0, 0] = 9.0
    onp["ftype"][0, 0, 1] = 99.0               # a confident given
    onp["res"][0, 0, 2] = 99.0
    onp["dig"][0, 0, :, 0] = 9.0
    onp["dig"][0, 0, ND - 1, 7] = 99.0         # value 7 -> var 2
    fb = H.alt2_fact_buf(onp, None, np.array([3]), np.array([50]))
    assert fb.shape == (Bp, K, 4) and fb[0, 2, 0] == 1.0, fb[0, :4]

    print("[step-trainer] selftest PASS: ladder weights (+BREATH_NORM), "
          "shelf plan, reverse schedule, reverse-threading equations vs "
          "finite differences (dX/dA/dB/dC), ctx/state contract vs the "
          "patched head, commit-adapter fact packing; zero GPU")


def main(argv):
    if "--selftest" in argv:
        selftest()
    elif "--cpuprobe" in argv:
        run_cpuprobe()
    elif "--eqfwd" in argv:
        run_eqfwd()
    elif "--eqbwd" in argv:
        run_eqbwd()
    else:
        run_train()


if __name__ == "__main__":
    main(sys.argv[1:])
