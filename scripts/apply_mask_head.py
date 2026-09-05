"""apply_mask_head.py — THE MASK HEAD, staged patch (2026-09-05, word
given: "be sure they have plenty of capacity — enough heads, storage,
and access to learnable data"). The FOURTH trained organ
(docs/mask_head_spec.md): a dedicated MH_HEADS-head attention bank
(env MH_HEADS, default 4; 8 supported — head count is a free reshape,
the registered scale axis) that turns the MASKRE reflex into a LEARNER.
It reads the metadata the >0.5 threshold throws away — the GRADED snap
adjacency _A5, the solver fact_buf, the previous breath's adjacency
(persistent state storage), breath phase, plus documented ports for the
domain-mass matryoshka and the step-atlas page — and emits a soft
OPEN-ONLY mask bias _mb (B, L_FAC, L_FAC) into the slot-mixer scores
sc2 and the alt21 station-4 mixer, BEFORE their -1e4 closes.

RUN ONLY AFTER any running chain exits (the running-chain law — a live
systemd unit has this module imported). --check mode loads the file,
asserts every anchor, ast-parses the WOULD-BE result, runs the symtable
free-variable audit, prints the exact new-param count at MH_HEADS=4 and
8, and writes NOTHING.

Equivalence contract (rung 1 of the bring-up ladder, BY CONSTRUCTION):
  ALG_MASKHEAD unset -> zero behavior change (every new computation is
                        env-guarded; the ctx/state dict keys added are
                        plumbing only — no compute touches them; the
                        chain's eq pre/post A/B/C dumps verify
                        bit-identity on banked ckpts)
  ALG_MASKHEAD=1     -> at birth the organ's TWO output doors are
                        ZERO-INIT (mh_wo — the output projection, the
                        ResNet law — and mh_headmix, the head-score
                        combiner), so _raw == exact zeros and
                        _mb = mh_gain * (softplus(_raw) -
                        softplus(_raw*0)) * open == exact zeros
                        (identical kernels on identical inputs cancel
                        bitwise); forward equals baseline exactly.

THE BIRTH-PLATEAU NOTE (registered here, honestly): softplus(0) = ln2
!= 0, so the spec's literal `mb = softplus(raw) * open` cannot be both
nonnegative-by-construction AND exactly zero at birth AND alive in
gradient (f >= 0 with f(0) = 0 forces f'(0) = 0 — a theorem, not a
choice). Resolution: the bias is measured RELATIVE to its own birth
plateau, `mb = gain * (softplus(raw) - softplus(raw*0)) * open`. It is
bounded BELOW by -gain*ln2 (a whisper; at gain 0.02 that is 0.014
logits) and unbounded-open above: the base mask SUPPORT is untouchable
(outside the open region _mb is exactly 0 and the -1e4 close is
byte-for-byte the baseline's) — A0's grave stays honored; a bounded
signed bias WITHIN the open region is the standing alt_g precedent
(`sc2 + _A5sym * alt_g`, signed, unbounded). The ajar gain (0.02, the
law) keeps d_mb/d_raw = gain * sigmoid(raw) * open != 0 from step one,
so both zero doors receive gradient at birth — no gate deadlock.

TRAINING ACCESS (the "learnable data" emphasis): NO new loss, no mask
target, no mask term anywhere — the Goodhart fence
(mycelium/diagnostic_register.py's law; a supervised mask teaches
concealment, not precision; assert_not_supervised in spirit — there is
no mask signal to register because none enters any loss). THE GRADIENT
PATH (two-terminal, verified by construction): _mb -> sc2 (and _sm21)
-> softmax -> h_slot -> cur (breath-gate blend) -> breaths[-1] ->
_s_final -> emission heads (pres/ftype/op/args/res/dig/query) -> parse
CE, which the fused trainer backprops; emission AND gold both feed, so
the grads are defined, never None. Every metadata terminal (_A5, snap
one-hots, fact_buf, mh_mass, mh_atlas, mh_prev) enters DETACHED — the
dual-terminal contract: dL/dp through solver/atlas = 0.

STORAGE: state["mh_prev"] threads the graded adjacency the organ
consumed this breath into the next breath's read (the notebook-shelf
contract: initialized in forward's _bs_state, mutated in breath_step,
carried by the step trainer's held state untouched) — the organ sees
the commitment FLOW, not just the level.

JIT discipline: no dtypes.float32 literals; every score tensor carries
clip(-1e4, 1e4); _raw is clipped to (-30, 30) before exp (single-kernel
finite softplus — NaN-safe). All new params live INSIDE the env guard
and inside the breath-loop graph in every training mode: absent ports
feed zero-tensors built from cur*0 slices, so every parameter owns a
DEFINED (possibly zero) gradient at the optimizer (the None-grad law).
"""
import ast
import builtins
import symtable
import sys

fn = 'scripts/phase1_algebra_head.py'
CHECK = '--check' in sys.argv
s = open(fn).read()
n_lines0 = s.count('\n')

assert 'ALG_MASKHEAD' not in s and '"mh_wo"' not in s, \
    "mask head already present — patch was applied; refuse (idempotence)"

PATCHES = []


def patch(num, desc, old, new):
    PATCHES.append((num, desc, old, new))


# 1. module constants: the head-count dial (env MH_HEADS, default 4;
#    the registered scale axis — 4 -> 8 is a free reshape, not a
#    rebuild) and the mask-context feature width. Loud door on
#    divisibility.
patch(1, "module consts: MH_HEADS dial + MH_CTX_F (env MH_HEADS)",
      '''N_HEADS = 8''',
      '''N_HEADS = 8
MH_HEADS = int(os.environ.get("MH_HEADS", "4"))  # MASK HEAD bank width
                                                 # (4 default; 8 = the
                                                 # registered scale axis)
assert H_W % MH_HEADS == 0, \\
    f"MH_HEADS={MH_HEADS} must divide H_W={H_W} (head reshape)"
MH_CTX_F = 22   # mask-context features: 12 fact (arg1/arg2/res x 4) +
                # 3 domain-mass port + 1 given-flag + 2 adjacency
                # row/col mass + 2 prev-breath row/col + 2 breath phase''')

# 2. build_params: the organ (env ALG_MASKHEAD), inside the K_B > 1
#    block after the alt21 station pair. ~1.97M params at MH_HEADS=4 —
#    sized UP per the word (capacity), against the single-headed
#    frugality the other organs were audit-flagged for. TWO zero-init
#    doors (mh_wo, mh_headmix) = silent birth; mh_gain AJAR (0.02).
patch(2, "build_params: mask-head organ params (env ALG_MASKHEAD)",
      '''            p["alt21_W_bo"] = t(np.zeros((H_W, H_W)))      # ZERO: silent birth
            p["alt21_W_bo_b"] = t(np.zeros(H_W))
        pass''',
      '''            p["alt21_W_bo"] = t(np.zeros((H_W, H_W)))      # ZERO: silent birth
            p["alt21_W_bo_b"] = t(np.zeros(H_W))
        if int(os.environ.get("ALG_MASKHEAD", "0")):
            # THE MASK HEAD (2026-09-05, word given): the fourth trained
            # organ — the learned PRECISION channel (mask_head_spec.md).
            # A dedicated MH_HEADS-head attention bank over H_W with its
            # own Wq/Wk/Wv, a gelu integration layer (mh_wu), a pair-key
            # projection (mh_wp), a mask-context encoder (facts +
            # graded adjacency + mass port + breath phase -> kv space),
            # and an atlas-page port projection (mh_atlas_w). TWO
            # ZERO-INIT output doors: mh_wo (output projection — the
            # ResNet law) and mh_headmix (per-head score combiner), so
            # the emitted bias is EXACTLY zero at birth; mh_gain is
            # AJAR (0.02, gate-deadlock corollary) — the softplus slope
            # gain*sigmoid(raw)*open is nonzero from step one, so both
            # doors self-open. Sized UP per the word: ~1.97M at 4 heads
            # (compute), state["mh_prev"] (storage), parse-CE-only
            # training through the re-masked pass (learnable data).
            p["mh_wq"], p["mh_wq_b"] = lin(H_W, H_W)
            p["mh_wk"], p["mh_wk_b"] = lin(H_W, H_W)
            p["mh_wv"], p["mh_wv_b"] = lin(H_W, H_W)
            p["mh_wu"], p["mh_wu_b"] = lin(H_W, H_W)
            p["mh_wo"] = t(np.zeros((H_W, H_W)))    # ZERO door 1: birth
            p["mh_wo_b"] = t(np.zeros(H_W))         # is bit-identical
            p["mh_wp"] = t(rng.randn(H_W, H_W) / math.sqrt(H_W))
            p["mh_enc1"], p["mh_enc1_b"] = lin(MH_CTX_F, 256)
            p["mh_enc2"], p["mh_enc2_b"] = lin(256, H_W)
            p["mh_atlas_w"] = t(rng.randn(H_W, H_W) / math.sqrt(H_W))
            p["mh_headmix"] = t(np.zeros(MH_HEADS)) # ZERO door 2
            p["mh_gain"] = t(np.full(1, 0.02))      # AJAR (the law)
        pass''')

# 3. forward: fact_buf rides into the breath-step ctx (plumbing only —
#    a dict key, no compute reads it unless ALG_MASKHEAD; the step
#    trainer re-enters forward per seam, so its per-step fact bufs
#    reach the mask head with no extra wiring).
patch(3, "forward: fact_buf -> _bs_ctx (mask-head metadata plumbing)",
      '''                   "revoke": revoke, "tail": tail, "reg": reg,''',
      '''                   "revoke": revoke, "tail": tail, "reg": reg,
                   # MASK HEAD metadata (2026-09-05; plumbing only — a
                   # dict key, zero compute when ALG_MASKHEAD unset).
                   # ctx also serves the OPTIONAL per-seam ports read
                   # via ctx.get: "mh_mass" (per-var domain-mass from
                   # the solver ping) and "mh_atlas" (step_atlas
                   # consult page) — populated by seam drivers only.
                   "fact_buf": fact_buf,''')

# 4. forward: the persistent mask-context storage slot, threaded like
#    the notebook shelf (state crosses breath boundaries; the step
#    trainer holds and re-walks the same dict, so the buffer persists
#    under the partitioned walk too).
patch(4, "forward: mh_prev storage slot in _bs_state",
      '''        _bs_state = {"cur": cur, "breaths": breaths, "nb": None,''',
      '''        _bs_state = {"cur": cur, "breaths": breaths, "nb": None,
                     # MASK HEAD storage (2026-09-05): the graded
                     # adjacency the organ consumed at the previous
                     # breath_step (detached) — Δ-visibility into the
                     # commitment FLOW; the notebook-threading contract
                     "mh_prev": None,''')

# 5. breath_step: THE ORGAN — after the MASKRE mask re-formation,
#    before the sc2 -1e4 close (the injection site). Emits _mb and adds
#    it to sc2; also leaves _mb in scope for station 4 (patch 6).
patch(5, "breath_step: mask-head organ + sc2 injection (env ALG_MASKHEAD)",
      '''    sc2 = sc2.clip(-1e4, 1e4) + (1.0 - _sm_kb) * -1e4''',
      '''    _mb = None
    if int(os.environ.get("ALG_MASKHEAD", "0")) and "mh_wo" in p:
        # THE MASK HEAD (2026-09-05): the learned precision channel at
        # the RELATE seam. Reads what the >0.5 reflex throws away — the
        # GRADED _A5 (confidences), solver fact_buf, the previous
        # breath's adjacency (state storage), breath phase, and the
        # domain-mass / atlas-page ports — and emits a soft OPEN-ONLY
        # bias over the slot mixer. ALL metadata enters DETACHED (the
        # dual-terminal law); the live terminal is cur (queries + kv
        # stream). NO mask loss exists anywhere — trained ONLY by
        # downstream parse CE through sc2 -> softmax -> h_slot -> cur
        # -> emissions (Goodhart fence: a supervised mask teaches
        # concealment; assert_not_supervised in spirit — no mask
        # signal enters any loss, ever). EQUIVALENCE AT BIRTH: mh_wo
        # and mh_headmix are ZERO-INIT, so _raw == 0 everywhere and
        # _mb = gain*(softplus(_raw) - softplus(_raw*0))*open == exact
        # zeros (identical kernels cancel bitwise). OPEN-ONLY: _mb is
        # gated to the already-open region _sm_kb (committed MASKRE
        # edges included) and bounded below by -gain*ln2 (the birth-
        # plateau reference) — the -1e4 close and the base-mask
        # SUPPORT are untouchable (A0's grave honored); a bounded
        # signed bias within the open region is the alt_g precedent.
        _z1 = (cur[:, :, :1] * 0.0).detach()
        if _A5 is not None:
            _A5s = (_A5 + _A5.transpose(-2, -1)).detach()
            _mh_a = _snaps[-1][0]           # detached snap one-hots
            _mh_b = _snaps[-1][1]
            _mh_r = _snaps[-1][2]
            _mh_g = _snaps[-1][3].unsqueeze(-1)
            _mh_row = _A5s.mean(-1, keepdim=True)
            _mh_col = _A5s.transpose(-2, -1).mean(-1, keepdim=True)
        else:
            _A5s = None
            _mh_a = _mh_b = _mh_r = None
            _mh_g = _z1; _mh_row = _z1; _mh_col = _z1
        _mh_f = ctx.get("fact_buf")         # (B, K_VARS, 4) solver
        if _mh_f is not None and _mh_a is not None:   # facts, detached
            _mh_ff = Tensor.cat(_mh_a @ _mh_f, _mh_b @ _mh_f,
                                _mh_r @ _mh_f, dim=-1)   # (B, L, 12):
            # what the solver knows about MY args and MY result
        else:
            _mh_ff = Tensor.cat(*([_z1] * 12), dim=-1)
        # DOMAIN-MASS PORT (documented, 2026-09-05): (B, K_VARS, 1)
        # per-var matryoshka radius (alternator_bridge.ping returns
        # mass; not yet threaded into the fused graph — only fact_buf
        # is in-graph today). A seam driver may set ctx["mh_mass"];
        # absent -> zeros, graph shape unchanged, grads stay defined.
        _mh_m = ctx.get("mh_mass")
        if _mh_m is not None and _mh_a is not None:
            _mh_fm = Tensor.cat(_mh_a @ _mh_m, _mh_b @ _mh_m,
                                _mh_r @ _mh_m, dim=-1)   # (B, L, 3)
        else:
            _mh_fm = Tensor.cat(*([_z1] * 3), dim=-1)
        _mh_p = state.get("mh_prev")        # STORAGE READ: the organ
        if _mh_p is not None:               # sees the commitment FLOW
            _mh_pr = _mh_p.mean(-1, keepdim=True)
            _mh_pc = _mh_p.transpose(-2, -1).mean(-1, keepdim=True)
        else:
            _mh_pr = _z1; _mh_pc = _z1
        _mh_bs = _z1 + math.sin(kb * math.pi / 3.0)   # breath phase
        _mh_bc = _z1 + math.cos(kb * math.pi / 3.0)   # (60-deg clock)
        _mh_cf = Tensor.cat(_mh_ff, _mh_fm, _mh_g, _mh_row, _mh_col,
                            _mh_pr, _mh_pc, _mh_bs, _mh_bc,
                            dim=-1)          # (B, L, MH_CTX_F) DETACHED
        _mh_ce = ((_mh_cf @ p["mh_enc1"] + p["mh_enc1_b"]).gelu()
                  @ p["mh_enc2"] + p["mh_enc2_b"])    # (B, L, H_W)
        # ATLAS-PAGE PORT (documented, 2026-09-05): (B, H_W) or
        # (B, L, H_W) detached page(s) from mycelium/step_atlas.consult
        # at a seam (the fused loop cannot consult mid-graph — consult
        # is numpy); a seam driver may set ctx["mh_atlas"]; absent ->
        # zeros from cur*0 keep mh_atlas_w in-graph (defined zero
        # grads — the None-grad law; degrade gracefully).
        _mh_ap = ctx.get("mh_atlas")
        if _mh_ap is None:
            _mh_ap = (cur * 0.0).detach()
        _mh_ce = _mh_ce + _mh_ap.reshape(B, -1, H_W) @ p["mh_atlas_w"]
        _mh_kv = cur + _mh_ce      # LIVE stream + detached context
        _mh_q = cur @ p["mh_wq"] + p["mh_wq_b"]
        _mh_k = _mh_kv @ p["mh_wk"] + p["mh_wk_b"]
        _mh_v = _mh_kv @ p["mh_wv"] + p["mh_wv_b"]
        _mh_hd = H_W // MH_HEADS
        _mh_qh = _mh_q.reshape(B, L_FAC, MH_HEADS, _mh_hd).permute(0, 2, 1, 3)
        _mh_kh = _mh_k.reshape(B, L_FAC, MH_HEADS, _mh_hd).permute(0, 2, 1, 3)
        _mh_vh = _mh_v.reshape(B, L_FAC, MH_HEADS, _mh_hd).permute(0, 2, 1, 3)
        _mh_sc = ((_mh_qh @ _mh_kh.transpose(-2, -1))
                  / math.sqrt(_mh_hd)).clip(-1e4, 1e4)   # (B, M, L, L)
        _mh_at = (_mh_sc
                  + (1.0 - _sm_kb.unsqueeze(1)) * -1e4).softmax(-1)
        _mh_gt = (_mh_at @ _mh_vh).permute(0, 2, 1, 3) \\
            .reshape(B, L_FAC, H_W)
        _mh_u = (_mh_gt @ p["mh_wu"] + p["mh_wu_b"]).gelu()
        _mh_o = _mh_u @ p["mh_wo"] + p["mh_wo_b"]     # ZERO door 1
        _mh_rp = (_mh_o @ (_mh_kv @ p["mh_wp"]).transpose(-2, -1)) \\
            / math.sqrt(H_W)                # value-informed pair logits
        _mh_rh = (_mh_sc * p["mh_headmix"].reshape(1, MH_HEADS, 1, 1)) \\
            .sum(1)                         # ZERO door 2: direct head
        _raw = (_mh_rp + _mh_rh).clip(-30.0, 30.0)    # finite softplus
        _mh_sp = (1.0 + _raw.exp()).log()             # softplus(raw)
        _mh_sp0 = (1.0 + (_raw * 0.0).exp()).log()    # birth plateau
        _mb = (p["mh_gain"].reshape(1, 1, 1)
               * (_mh_sp - _mh_sp0) * _sm_kb)
        if _A5s is not None:                # STORAGE WRITE: this
            state["mh_prev"] = _A5s         # breath's consumed
                                            # adjacency, detached
        sc2 = sc2 + _mb        # the injection site: BEFORE the close
    sc2 = sc2.clip(-1e4, 1e4) + (1.0 - _sm_kb) * -1e4''')

# 6. breath_step: the same bias into the alt21 station-4 mixer, before
#    ITS -1e4 close (one geometry per breath — both mixers see the
#    same learned precision; env unset or organ absent -> _mb is None
#    -> byte-identical).
patch(6, "breath_step: mask-head bias -> alt21 station-4 mixer",
      '''        _sm21 = _sm21.clip(-1e4, 1e4) + (1.0 - _sm_kb) * -1e4''',
      '''        if _mb is not None:        # MASK HEAD: the same open-only
            _sm21 = _sm21 + _mb    # bias, station-4 mixer (before
                                   # ITS close — one geometry/breath)
        _sm21 = _sm21.clip(-1e4, 1e4) + (1.0 - _sm_kb) * -1e4''')


for num, desc, old, new in PATCHES:
    assert old in s, f"anchor {num} MISSING ({desc}) — read the file, adjust"
    assert s.count(old) == 1, f"anchor {num} NOT UNIQUE ({desc})"
    s = s.replace(old, new, 1)

tree = ast.parse(s)                       # the would-be result must parse

# structural asserts on the would-be module (cheap, no import, no GPU)
for key in ("mh_wq", "mh_wk", "mh_wv", "mh_wu", "mh_wo", "mh_wp",
            "mh_enc1", "mh_enc2", "mh_atlas_w", "mh_headmix", "mh_gain"):
    assert f'p["{key}"]' in s, f"patched tree lost {key}"
assert s.count('os.environ.get("ALG_MASKHEAD", "0")') == 2, \
    "expected exactly 2 ALG_MASKHEAD guards (params + organ)"
assert s.index('_mb = None') < s.index('sc2 = sc2 + _mb') \
    < s.index('sc2 = sc2.clip(-1e4, 1e4)'), \
    "organ must land before the sc2 -1e4 close (the injection site)"
assert s.index('if _mb is not None:') \
    < s.index('_sm21 = _sm21.clip(-1e4, 1e4)'), \
    "station-4 bias must land before ITS -1e4 close"
for probe in ('state["mh_prev"]', 'ctx.get("fact_buf")',
              'ctx.get("mh_mass")', 'ctx.get("mh_atlas")',
              '"mh_prev": None'):
    assert probe in s, f"patched tree lost the {probe} plumbing"

# the symtable free-variable audit (the apply_step_trainer.py idiom):
# every global name breath_step/build_params reads must be module-level
# (or a known dynamic global, or a builtin) — the loud door against a
# missed local
mod_tbl = symtable.symtable(s, fn, 'exec')
module_names = set(mod_tbl.get_identifiers())
DYNAMIC_OK = {'_CENSUS', '_IMP', '_SEV', '_SGC', '_BINDC'}
BUILTIN = set(dir(builtins))


def audit(tbl, fname):
    bad = set()
    for sym in tbl.get_symbols():
        n = sym.get_name()
        if sym.is_global() and n not in module_names \
                and n not in DYNAMIC_OK and n not in BUILTIN:
            bad.add(n)
    for ch in tbl.get_children():
        bad |= audit(ch, fname)
    assert not bad, f"{fname}: unresolved free variables {sorted(bad)}"
    return set()


for child in mod_tbl.get_children():
    if child.get_name() in ('breath_step', 'build_params'):
        audit(child, child.get_name())

# the exact new-param count (at ALG_HW=512, MH_CTX_F=22), printed per
# the word — capacity is a registered claim, not a vibe
H, F = 512, 22


def mh_count(m):
    linpb = H * H + H                    # lin(H, H) weight + bias
    return (4 * linpb                    # mh_wq/wk/wv/wu
            + linpb                      # mh_wo (+bias), ZERO door 1
            + H * H                      # mh_wp (no bias)
            + (F * 256 + 256)            # mh_enc1
            + (256 * H + H)              # mh_enc2
            + H * H                      # mh_atlas_w (no bias)
            + m                          # mh_headmix, ZERO door 2
            + 1)                         # mh_gain (AJAR)


print(f"[mask-head] {len(PATCHES)} anchors OK "
      f"(+{s.count(chr(10)) - n_lines0} lines):")
for num, desc, _o, _n in PATCHES:
    print(f"  {num}. {desc}")
print(f"[mask-head] symtable free-var audit PASS (breath_step, "
      f"build_params)")
print(f"[mask-head] new params: {mh_count(4):,} at MH_HEADS=4 | "
      f"{mh_count(8):,} at MH_HEADS=8 (head count is a free reshape; "
      f"only mh_headmix grows)")
if CHECK:
    print("[mask-head] --check: ast OK on the would-be result; "
          "NOTHING written")
else:
    open(fn, 'w').write(s)
    print("[mask-head] APPLIED; ast OK — run the eq pre/post A/B/C "
          "gate before trusting (equivalence contract, rung 1)")
