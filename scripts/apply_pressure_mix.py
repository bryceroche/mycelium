"""apply_pressure_mix.py — THE PRESSURE MIX, staged patch (2026-09-06,
word given; ledger: "WORD GIVEN: THE PRESSURE-COOKER CAMPAIGN").
MIXED-SEAL TRAINING: a flat per-row mix — a constant share of training
rows (ALG_PC_MIX, dose 0.30 in the chain) runs the forward with the
pressure-cooker seal ENGAGED (mode-1-style severance at breath SC_KB:
residual dead, only the shelf crossing survives) beside open rows that
run exactly as today. Env family: ALG_PC_MIX (float share; unset/0 =
off = byte-identical — every new line lives behind the guard);
ALG_PC_LIVE (default 1 inside the family) = the live-wire sub-dial.

WHY (the forensic, docs/silent_organ_audit.md item 0): sealed-mode wild
= 0.0000/2051 on fedon242 (vs 0.2613 open; post-severance commit buffer
6 facts vs 1067) — and the champion NEVER trained under the seal:
the FED chain exports SC_EVAL=0 process-wide, which collapses the
mode-2 blend to the open identity at JIT capture (cur*(1-0)+seal*0 =
cur, exactly), and the _SEV pulse buffer is only *created* inside the
SC_EVAL-unset branch, so the trainer's per-step reseal no-ops forever.
Training was always open; the committed channel never felt pressure.

THE GRADIENT-PATH FINDING (the crux, investigated before design):
what crosses the seal is _cur_seal = cur*0.0 + _inj4, where _inj4 =
(sum_j softmax((cur@W_gq). d_j)_j * garage_j, per-role conj-unbound)
@ W_busr. The garage deposits d_j are DETACHED BY LAW (head line
"_wg4 = (_canon4 / _cn4 * _wn4).detach()"; "the deposit is a FACT")
AND their canonical content is built through an argmax one-hot snap
(_oh4 = (_lg4 == _lg4.max()).float()) — gradient-dead in DIRECTION
even if un-detached. So under the seal, post-seal loss reaches:
  (a) post-seal breaths + emission heads (live) — they must learn to
      parse from surviving committed content: the main pressure;
  (b) pre-seal breaths via the attention QUERY (cur@W_gq is live at
      the crossing) — pre-seal state learns to SELECT which facts are
      read; W_gq/W_busr train;
  (c) pre-seal breaths via the NOTEBOOK: post-seal breaths re-read
      lane-1/lane-2 ink (_nb[j] = breath j's cur@W_sil, NOT detached)
      — an honest leak in the severance that the 0.0000 meter shares
      (same seal, same leak: bar comparability holds);
  (d) NOT the deposit content: the committer (W_bind1/W_bind1_b/
      W_bind2 and upstream cur through the deposit) gets ZERO gradient
      from post-seal loss — its only teacher is the bind emission
      (out["bind"], the breath-0 tap, upstream of the cut).
THE CHOSEN DESIGN: keep (a)-(c) as the primary mechanism, and open
one lawful extra wire — THE LIVE WIRE (ALG_PC_LIVE, default 1): for
SEALED training rows only, the deposit crossing is un-detached
per-row (value-identical; detach never changes values). Because of
the argmax snap, the surviving live channel into the committer is
exactly the CONFIDENCE STAMP _wn4 (the pre-snap magnitude): the
committer learns to modulate commitment AMPLITUDE by downstream
usefulness ("wrong wires run quiet" becomes trainable). Dual-terminal
contract preserved: the SOLVER facts (_snaps, the 2-of-3 field's
inputs) stay detached; open rows' deposits stay detached; val and all
reads stay detached (SC_EVAL gate). The fact IDENTITY (which lattice
point) remains gradient-free — the argmax stands.

PER-ROW MECHANISM (option (a) of the build order — one graph,
JIT-stable): _PCV, a (BATCH,1,1) float data buffer (the _SEV /
MASK_GOLD idiom: created before capture, assigned in place per step),
blends per row: cur' = cur*(1-v) + _cur_seal*v, q_extra' =
_q_open*(1-v) + _q_seal*v, v in {0,1} per row. Sealed rows reproduce
the mode-1 constant severance bit-for-bit (1.0*x = x; 0.0*finite = 0;
x+0 = x); open rows reproduce today's open path likewise. ASSIGNMENT
IS STABLE (flat-mix law, never re-rolled per epoch): row i is sealed
iff hash(i) < share, hash = Knuth multiplicative on the DATASET row
index ((i * 2654435761) mod 2^32 / 2^32).

VAL/READ HYGIENE: the per-row branch requires SC_EVAL to be UNSET.
The trainer's existing val push (SC_EVAL="0" around _quick_val for
mode >= 2) therefore excludes val automatically — val compares OPEN
mode, unchanged, comparable across arms. The chain's PRESSURE arm
must NOT export SC_EVAL (asserted loudly at arm time); reads carry
their own SC_EVAL per meter (sealed = unset, open = 0). Outside
do_train, _PCV never exists, so ALG_PC_MIX in a read env is inert by
construction (trained-env law satisfied cheaply).

--check: loads the file, asserts every anchor unique, ast-parses the
would-be result, runs the symtable free-variable audit (the
apply_mask_head.py idiom; _PCV joins the dynamic-globals allowlist),
writes NOTHING. PM_TARGET env may point at a copy (rehearsal);
default scripts/phase1_algebra_head.py.
"""
import ast
import builtins
import os
import symtable
import sys

fn = os.environ.get("PM_TARGET", 'scripts/phase1_algebra_head.py')
CHECK = '--check' in sys.argv
s = open(fn).read()
n_lines0 = s.count('\n')

assert 'ALG_PC_MIX' not in s and '_PCV' not in s, \
    "pressure mix already present — patch was applied; refuse (idempotence)"
assert 'ALG_SHELF_CIRCLE' in s and '_SEV = Tensor([1.0])' in s, \
    "shelf-circle/pulse machinery missing — wrong vintage of the head"
assert '_wg4 = (_canon4 / _cn4 * _wn4).detach()' in s, \
    "canonical-shelf deposit anchor missing — wrong vintage of the head"

PATCHES = []


def patch(num, desc, old, new):
    PATCHES.append((num, desc, old, new))


# 1. breath_step: the per-row mixed seal. Takes precedence over the
#    SC_EVAL/_SEV branches ONLY when the trainer has armed _PCV and
#    SC_EVAL is unset (val's push lands in the elif, open, unchanged).
patch(1, "breath_step: per-row mixed seal (_PCV data buffer)",
      '''            if _scm >= 2:
                _sce = os.environ.get("SC_EVAL", "")
                if _sce:
                    _sv = float(_sce)
                    cur = cur * (1.0 - _sv) + _cur_seal * _sv
                    q_extra = _q_open * (1.0 - _sv) + _q_seal * _sv
                else:
                    global _SEV
                    try: _SEV
                    except NameError: _SEV = None
                    if _SEV is None:
                        _SEV = Tensor([1.0]).contiguous().realize()
                    _svt = _SEV.reshape(1, 1, 1)
                    cur = cur * (1.0 - _svt) + _cur_seal * _svt
                    q_extra = _q_open * (1.0 - _svt) + _q_seal * _svt
            else:
                cur = _cur_seal
                q_extra = _q_seal''',
      '''            _pcv4 = (globals().get("_PCV")
                     if float(os.environ.get("ALG_PC_MIX", "0")) > 0.0
                     and not os.environ.get("SC_EVAL", "") else None)
            if _pcv4 is not None:
                # THE PRESSURE MIX (apply_pressure_mix.py, 2026-09-06):
                # per-row mixed seal — _PCV is a (B,1,1) data buffer
                # (the _SEV/MASK_GOLD idiom: one JIT graph, dynamic
                # value). 1.0 rows get the mode-1 constant severance
                # (residual dead, shelf crossing the only road), 0.0
                # rows the open path — same blend arithmetic as the
                # SC_EVAL forms, per row (exact at v in {0,1}: 1.0*x =
                # x, 0.0*finite = 0, x+0 = x). The trainer assigns it
                # per step from the STABLE index-hash assignment (flat
                # mix, never re-rolled); _quick_val is excluded by the
                # existing SC_EVAL="0" push (val compares OPEN mode).
                _pvt4 = _pcv4.reshape(-1, 1, 1)
                cur = cur * (1.0 - _pvt4) + _cur_seal * _pvt4
                q_extra = _q_open * (1.0 - _pvt4) + _q_seal * _pvt4
            elif _scm >= 2:
                _sce = os.environ.get("SC_EVAL", "")
                if _sce:
                    _sv = float(_sce)
                    cur = cur * (1.0 - _sv) + _cur_seal * _sv
                    q_extra = _q_open * (1.0 - _sv) + _q_seal * _sv
                else:
                    global _SEV
                    try: _SEV
                    except NameError: _SEV = None
                    if _SEV is None:
                        _SEV = Tensor([1.0]).contiguous().realize()
                    _svt = _SEV.reshape(1, 1, 1)
                    cur = cur * (1.0 - _svt) + _cur_seal * _svt
                    q_extra = _q_open * (1.0 - _svt) + _q_seal * _svt
            else:
                cur = _cur_seal
                q_extra = _q_seal''')

# 2. breath_step: THE LIVE WIRE — sealed rows' deposit crossing is
#    un-detached per-row (value-identical; gradient only). Given the
#    argmax snap upstream, the live channel into the committer is the
#    confidence stamp _wn4 alone (amplitude, never identity).
patch(2, "breath_step: live-wire deposit crossing for sealed rows",
      '''            _wn4 = _wg4.pow(2).sum(-1, keepdim=True).sqrt() + 1e-6
            _cn4 = _canon4.pow(2).sum(-1, keepdim=True).sqrt() + 1e-6
            _wg4 = (_canon4 / _cn4 * _wn4).detach()''',
      '''            _wn4 = _wg4.pow(2).sum(-1, keepdim=True).sqrt() + 1e-6
            _cn4 = _canon4.pow(2).sum(-1, keepdim=True).sqrt() + 1e-6
            _dep4 = _canon4 / _cn4 * _wn4
            _wg4 = _dep4.detach()
            _pcl4 = (globals().get("_PCV")
                     if float(os.environ.get("ALG_PC_MIX", "0")) > 0.0
                     and int(os.environ.get("ALG_PC_LIVE", "1"))
                     and not os.environ.get("SC_EVAL", "") else None)
            if _pcl4 is not None:
                # THE LIVE WIRE (apply_pressure_mix.py, 2026-09-06):
                # for SEALED training rows the shelf crossing carries
                # gradient (per-row blend of live/detached — VALUES
                # identical either way; detach only cuts the tape).
                # Because _canon4 rides an argmax one-hot (grad-dead
                # direction) the surviving live channel into the
                # committer (W_bind1/2, upstream cur) is exactly the
                # confidence stamp _wn4: commitment AMPLITUDE learns
                # from downstream use; fact IDENTITY stays discrete.
                # Dual-terminal contract: _snaps (the solver facts,
                # detached above) untouched; open rows detached as
                # today; val/reads detached via the SC_EVAL gate.
                _plv4 = _pcl4.reshape(-1, 1, 1)
                _wg4 = _dep4 * _plv4 + _dep4.detach() * (1.0 - _plv4)''')

# 3. do_train: arm the mix — stable per-row assignment + the _PCV
#    buffer, created BEFORE the first step() capture (JIT law).
patch(3, "do_train: arm _PCV + stable index-hash assignment",
      '''    t0 = time.time()
    for s in range(steps):''',
      '''    _pc_mix = float(os.environ.get("ALG_PC_MIX", "0"))
    _pc_assign = None
    if _pc_mix > 0.0:
        # THE PRESSURE MIX (2026-09-06): per-row seal assignment is a
        # DETERMINISTIC hash of the dataset row index (Knuth
        # multiplicative) — stable across epochs/steps/restarts (flat
        # mix law: a constant sealed subpopulation, not a per-epoch
        # coin). The seal needs the shelf road to exist and SC_EVAL
        # unset (else the mode-2 branch bakes OPEN at JIT capture —
        # the exact silent no-op the forensic caught in the champion).
        assert (int(os.environ.get("ALG_BUSGARAGE", "0")) >= 2
                and int(os.environ.get("ALG_SHELF_CIRCLE", "0")) >= 2), (
            "ALG_PC_MIX needs ALG_BUSGARAGE>=2 + ALG_SHELF_CIRCLE>=2 (the per-row "
            "blend rides the mode-2 SC_EVAL forms; _quick_val's OPEN push is "
            "mode>=2 only — at mode 1 a stale _PCV would leak into val)")
        assert not os.environ.get("SC_EVAL", ""), (
            "ALG_PC_MIX with SC_EVAL set would bake the seal shut at "
             "JIT capture (the champion's silent no-op) — unset SC_EVAL; "
             "val forces OPEN by itself")
        _pc_h = ((np.arange(n, dtype=np.uint64) * np.uint64(2654435761))
                 % np.uint64(4294967296)).astype(np.float64) / 4294967296.0
        _pc_assign = (_pc_h < _pc_mix).astype(np.float32)
        globals()["_PCV"] = Tensor(
            np.zeros((batch, 1, 1), np.float32)).contiguous().realize()
        print(f"[pressure] mixed-seal armed: share={_pc_mix} -> "
              f"{int(_pc_assign.sum())}/{n} rows sealed (stable "
              f"index-hash); live-wire="
              f"{os.environ.get('ALG_PC_LIVE', '1')}", flush=True)
    t0 = time.time()
    for s in range(steps):''')

# 4. do_train loop: per-step per-row seal feed (beside the _SEV
#    reseal; assign-in-place keeps the one captured graph).
patch(4, "do_train loop: per-step _PCV assign from the batch indices",
      '''        lv = step()''',
      '''        if _pc_assign is not None:
            globals()["_PCV"].assign(Tensor(
                _pc_assign[idx].reshape(-1, 1, 1),
                dtype=globals()["_PCV"].dtype)).realize()
        lv = step()''')

for num, desc, old, new in PATCHES:
    assert old in s, f"anchor {num} MISSING ({desc}) — read the file, adjust"
    assert s.count(old) == 1, f"anchor {num} NOT UNIQUE ({desc})"
    s = s.replace(old, new, 1)

tree = ast.parse(s)                       # the would-be result must parse

# structural asserts on the would-be module (cheap, no import, no GPU)
assert s.count('_pcv4 = ') == 1 and s.count('_pcl4 = ') == 1
assert s.count('globals().get("_PCV")') == 2, "both reads present"
assert s.count('globals()["_PCV"]') == 3, "arm + per-step assign (2 uses)"
assert s.index('_pc_assign = (_pc_h < _pc_mix)') \
    < s.index('t0 = time.time()'), "arming must precede the step loop"
assert s.index('_pc_assign[idx]') < s.index('lv = step()'), \
    "per-step assign must precede step()"
assert s.index('the pulse: reseal per step') < s.index('_pc_assign[idx]'), \
    "_PCV assign rides beside (after) the _SEV reseal block"
assert '_wg4 = _dep4.detach()' in s and '_dep4 * _plv4' in s, \
    "live-wire blend lost"
assert 'cur = cur * (1.0 - _pvt4) + _cur_seal * _pvt4' in s, \
    "per-row severance blend lost"
# the seal-order invariant: the per-row branch must sit inside the
# SC_KB block, ahead of the mode-2 elif
_i_kb = s.index('if _scm and kb == int(os.environ.get("SC_KB", "4")):')
assert _i_kb < s.index('_pcv4 = ') < s.index('elif _scm >= 2:'), \
    "per-row branch must be the first arm of the seal dispatch"

# the symtable free-variable audit (the apply_mask_head.py idiom)
mod_tbl = symtable.symtable(s, fn, 'exec')
module_names = set(mod_tbl.get_identifiers())
DYNAMIC_OK = {'_CENSUS', '_IMP', '_SEV', '_SGC', '_BINDC', '_PCV'}
BUILTIN = set(dir(builtins))


def audit(tbl, fname):
    bad = set()
    for sym in tbl.get_symbols():
        n_ = sym.get_name()
        if sym.is_global() and n_ not in module_names \
                and n_ not in DYNAMIC_OK and n_ not in BUILTIN:
            bad.add(n_)
    for ch in tbl.get_children():
        bad |= audit(ch, fname)
    assert not bad, f"{fname}: unresolved free variables {sorted(bad)}"
    return set()


for child in mod_tbl.get_children():
    if child.get_name() in ('breath_step', 'do_train', 'forward',
                            'build_params'):
        audit(child, child.get_name())

print(f"[pressure mix] {len(PATCHES)} anchors OK "
      f"(+{s.count(chr(10)) - n_lines0} lines):")
for num, desc, _o, _n in PATCHES:
    print(f"  {num:2d}. {desc}")
print("[pressure mix] symtable free-var audit PASS (breath_step, "
      "do_train, forward, build_params)")
print("[pressure mix] NEW params: 0 (a data buffer and a gradient "
      "route, not capacity); ALG_PC_MIX unset = byte-identical "
      "(eq A/B/C gate proves it)")
print("[pressure mix] gradient contract: sealed rows' pressure = "
      "post-seal heads + query shaping + notebook road + live-wire "
      "confidence stamp; deposit IDENTITY stays gradient-free "
      "(argmax); solver facts stay detached")
if CHECK:
    print("[pressure mix] --check: ast OK on the would-be result; "
          "NOTHING written")
else:
    open(fn, 'w').write(s)
    print(f"[pressure mix] APPLIED ({fn}); ast OK — run the eq "
          "pre/post A/B/C gate + .cache/pc_row_smoke.py before "
          "trusting (equivalence contract)")
