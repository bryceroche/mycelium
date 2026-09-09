"""read_batch.py — THE ONE-PROCESS READ (2026-09-08).

THE PROBLEM IT SOLVES: every read in a chain was a fresh process — head
import + fixture load + a full kernel compile of the two-pass forward
before the first number arrives (~60 s on the GPU box, per read), and a
read block runs ~20 of them over a handful of checkpoints and two
fixtures. Within ONE fixture the fixed cost is paid once: load the head
once, build the params once, compile once, then for each checkpoint
assign the weights in place (the loop_val idiom,
`p[k].assign(sd[k]...).realize()`) and re-run the same read.

SAME ORGAN, NOT A REIMPLEMENTATION (the meter-divergence law: a check
must CALL its organ). Every number here comes out of the standalone
script's own function:

  loopval           -> loop_val.main()            (its read AND its print)
  sealed            -> step_engine_read.main()    (its exact body)
  clock:<planes>    -> clock_read.run(fixture)    (its exact body)

read_batch owns exactly two things: the loop over checkpoints, and the
env each organ is handed. It computes nothing itself, and it prints no
number of its own that an organ did not hand it.

ONE ARCHITECTURE PER PROCESS. The head's shape dials (ALG_HW,
ALG_FTYPES, ALG_POLAR_D, ALG_POLAR_EM, ...) are read at IMPORT, so
every checkpoint in RB_CKPTS must fit one build_params(0). A preflight
checks key sets AND shapes for all of them before the first read, so a
mismatched arm dies in seconds instead of four checkpoints into a
chain. Two arms with different dials (e.g. the polar chain's arm A vs
arm B) get one read_batch process each.

THE FIXTURE IS THE PROCESS. ALG_TEST / ALG_TEST_NAME are read at
phase1_algebra_head IMPORT time, so one process holds exactly one
fixture — the same rule clock_read states for CR_TEST=both. RB_TEST is
this script's declared door; it is set BEFORE the head is imported and
then cross-checked against the head's own module constants.

THE ENV IS THE METER. Each meter runs under the env its standalone runs
under in the chains, and read_batch RESTORES the whole environment after
every meter (os.environ snapshot/restore) so no meter can inherit
another's dials:

  loopval   the CALLER's SC_EVAL, untouched — and asserted PRESENT at
            startup, because the chains pass SC_EVAL=0 on every loop_val
            line and an unset SC_EVAL silently makes it a SEALED read
  sealed    SC_EVAL UNSET (the chains' sealed line passes none; unset is
            what arms the mode-2 seal). SE_R passed through untouched.
  clock     SC_EVAL UNSET (the chains' clock lines pass none, and
            clock_read leaves SC_EVAL exactly as the caller left it)

Note step_engine_read.main() sets ALG_INV=1 process-wide when the atlas
file exists — exactly the kind of leak the snapshot/restore exists for.

WHY MIXING SC_EVAL MODES IN ONE PROCESS IS ALLOWED HERE — AND THE
TRIPWIRE THAT PROVES IT. The ledger's SC_EVAL capture collapse (THE
UNLIT STOVE, 2026-09-06) is a JIT-CAPTURE specimen: SC_EVAL exported
process-wide collapsed the mode-2 blend to open identity at the
TRAINER's TinyJit capture and _SEV was never created, so the reseal
no-opped forever. The read path has no capture — the only @TinyJit in
phase1_algebra_head.py is inside do_train(), forward() is called
un-captured, and breath_step reads os.environ["SC_EVAL"] while it
BUILDS the graph, i.e. on every forward() call. So the toggle is live
per call. That is an argument, not a measurement, so read_batch does not
trust it: at startup it fires THE SC_EVAL TRIPWIRE — one batch through
forward() with SC_EVAL=0 and one with SC_EVAL unset, on the first
checkpoint — and REFUSES to run if the two agree bit-for-bit while the
shelf-circle seal is armed. A baked seal cannot reach a meter here
without the tripwire firing first. Set RB_ONE_MODE=1 to skip the mixing
entirely and serve one SC_EVAL mode per process instead (then run one
read_batch per meter group); correctness beats speed, and a wrong meter
is worse than a slow one.

ENV CONTRACT
  RB_TEST     wild | mint            (required; the fixture, from
                                      clock_read's own _FIXTURES table)
  RB_CKPTS    comma-separated checkpoint paths (required)
  RB_METERS   comma-separated from: loopval | sealed |
              clock:<plane>[+<plane>...]   e.g.
              RB_METERS=clock:breath_hand+parity+content,loopval,sealed
  RB_ONE_MODE 1 = refuse to mix SC_EVAL modes in this process
  SE_R        passed through to step_engine_read (chains use SE_R=1)
  CR_*        passed through to clock_read (CR_KEY/CR_N/CR_RANK/CR_SEED);
              CR_CKPT / CR_PLANES / CR_TEST are set per read here
  everything else: the caller's ALG_* env — reads carry the TRAINED env,
  always. read_batch sets no ALG_* dial of its own and never sets DEV.

  RB_LV_MOD / RB_SE_MOD / RB_CR_MOD override the module names of the
  three organs (default loop_val / step_engine_read / clock_read). They
  exist so a preview copy can be exercised before its patch is applied;
  the module actually loaded is printed in the banner, every run.

INVOCATION (the polar-chain read block, one process per fixture per
architecture; the ALG_* stack is the arm's TRAINED env, verbatim, with
$X the arm's extra dials):

  env DEV=PCI+AMD $ALG_STACK $X SC_EVAL=0 SE_R=1 \
      CR_KEY=breaths_u CR_RANK=64 CR_N=256 \
      RB_TEST=wild RB_CKPTS=.cache/sharp_polar242.safetensors \
      RB_METERS=clock:breath_hand+parity+content,loopval,sealed \
      .venv/bin/python3 scripts/read_batch.py

  env DEV=PCI+AMD $ALG_STACK $X SC_EVAL=0 \
      CR_KEY=breaths_u CR_RANK=64 CR_N=256 \
      RB_TEST=mint RB_CKPTS=.cache/sharp_polar242.safetensors \
      RB_METERS=clock:breath_hand+parity+content,loopval \
      .venv/bin/python3 scripts/read_batch.py

Checkpoints that share an architecture ride the same process: put them
all in RB_CKPTS (comma-separated) and the setup is paid once for the
lot.

OUTPUT: the organs' own lines, verbatim, plus one [read-batch] line per
(checkpoint, meter) with the wall time, and a closing timing summary.
"""
import importlib
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

T_START = time.time()

# clock_read is CPU-safe to import (module level = env reads + chdir) and
# it owns the fixture table — one source of truth for what 'wild' means.
_CR_MOD = os.environ.get("RB_CR_MOD", "clock_read")
clock_read = importlib.import_module(_CR_MOD)
FIXTURES = clock_read._FIXTURES

RB_TEST = os.environ.get("RB_TEST", "").strip().lower()
assert RB_TEST in FIXTURES, (
    f"RB_TEST={RB_TEST!r} — one process holds exactly one fixture "
    f"(ALG_TEST is read at head import); want {'|'.join(FIXTURES)}")
CKPTS = [c for c in os.environ.get("RB_CKPTS", "").split(",") if c.strip()]
assert CKPTS, "RB_CKPTS is empty (comma-separated checkpoint paths)"
CKPTS = [c.strip() for c in CKPTS]
for c in CKPTS:
    assert os.path.exists(c), f"RB_CKPTS: no such checkpoint {c}"
METERS = [m.strip() for m in os.environ.get("RB_METERS", "").split(",")
          if m.strip()]
assert METERS, ("RB_METERS is empty (loopval | sealed | "
                "clock:<plane>[+<plane>...])")
for m in METERS:
    assert m in ("loopval", "sealed") or m.startswith("clock:"), \
        f"RB_METERS: unknown meter {m!r} (loopval|sealed|clock:<planes>)"
    if m.startswith("clock:"):
        pls = [x for x in m[len("clock:"):].split("+") if x]
        assert pls, f"RB_METERS: {m!r} names no planes"
        for pl in pls:
            assert pl in clock_read._CR_PLANE_SELS, (
                f"RB_METERS: {m!r} plane {pl!r} unknown "
                f"(want {'|'.join(clock_read._CR_PLANE_SELS)})")
ONE_MODE = int(os.environ.get("RB_ONE_MODE", "0"))

# THE FIXTURE DOOR — set before the head is imported, and never silently
# over a caller who asked for a different one.
_PATH, _NAME = FIXTURES[RB_TEST]
for _k, _v in (("ALG_TEST", _PATH), ("ALG_TEST_NAME", _NAME)):
    _had = os.environ.get(_k)
    assert _had in (None, "", _v), (
        f"{_k}={_had!r} in the env contradicts RB_TEST={RB_TEST} "
        f"({_v!r}) — refusing to relabel a fixture silently")
    os.environ[_k] = _v

import numpy as np                                          # noqa: E402
import phase1_algebra_head as HEAD                          # noqa: E402
from tinygrad import Tensor, dtypes                         # noqa: E402
from tinygrad.nn.state import safe_load                     # noqa: E402

assert HEAD.ALG_TEST == _PATH and HEAD.TEST_NAME == _NAME, (
    f"the head bound fixture {HEAD.ALG_TEST}/{HEAD.TEST_NAME} but "
    f"RB_TEST={RB_TEST} means {_PATH}/{_NAME} — import order broken")

_LV_MOD = os.environ.get("RB_LV_MOD", "loop_val")
_SE_MOD = os.environ.get("RB_SE_MOD", "step_engine_read")


def _import_organ(name):
    """Check the SOURCE for the refactored entry point BEFORE importing.
    An unpatched loop_val runs its whole read at import time (the body is
    top level) — importing it to discover it is unpatched would fire a
    stray read first. Refuse from the text instead."""
    spec = importlib.util.find_spec(name)
    assert spec is not None and spec.origin, f"no module {name!r} on sys.path"
    src = open(spec.origin).read()
    assert "def main(ckpt=None, data=None, p=None)" in src, (
        f"{spec.origin} has no `def main(ckpt=None, data=None, p=None)` — "
        f"apply the read_batch refactor patch first "
        f"(scratchpad: {name}_readbatch.patch). read_batch CALLS the "
        f"organ; it does not reimplement it, and it will not run against "
        f"an organ it cannot hand a checkpoint to. NOTE: an unpatched "
        f"loop_val executes its entire read at import, so this check "
        f"deliberately reads the file rather than importing it.")
    return importlib.import_module(name)


import importlib.util                                       # noqa: E402
loop_val = _import_organ(_LV_MOD)
step_engine_read = _import_organ(_SE_MOD)
import inspect                                              # noqa: E402
for _m, _nm in ((loop_val, _LV_MOD), (step_engine_read, _SE_MOD)):
    assert set(("ckpt", "data", "p")) <= set(
        inspect.signature(_m.main).parameters), (
        f"{_nm}.main(ckpt=, data=, p=) is missing — apply the read_batch "
        f"refactor patch.")

# THE NO-CAPTURE FENCE: the read path must not be JIT-captured, or the
# first meter's SC_EVAL would be baked into every later one (THE UNLIT
# STOVE). forward() is a plain function; assert it, loudly, here.
from tinygrad.engine.jit import TinyJit                     # noqa: E402
assert not isinstance(HEAD.forward, TinyJit), (
    "phase1_algebra_head.forward is TinyJit-wrapped — a captured graph "
    "bakes SC_EVAL at capture; read_batch must not share a graph across "
    "meters. Run one meter mode per process.")

# THE JIT'D READ DOOR (mycelium/jit_read.py, 2026-09-08). The fence
# above still stands and still means exactly what it said: forward()
# itself is never wrapped, so no graph is shared across meters by
# accident. jit_read captures graphs BESIDE forward() and keys every
# one of them on the value of every env name the head's source reads —
# SC_EVAL, SC_KB, ALG_SHELF_CIRCLE, ALG_PC_MIX included — so a mode
# change gets its own graph instead of a stale one. THE UNLIT STOVE,
# keyed instead of argued. Assert it here rather than trusting prose.
from mycelium import jit_read                               # noqa: E402
_JRN = jit_read.env_names(os.path.abspath(HEAD.__file__))
if jit_read.enabled():
    for _req in ("SC_EVAL", "SC_KB", "ALG_SHELF_CIRCLE", "ALG_PC_MIX",
                 "ALG_BREATH", "ALG_ALT2", "ALG_INV"):
        assert _req in _JRN, (
            f"ALG_JIT_READ=1 but {_req} is not in jit_read's env key — a "
            f"captured graph would outlive the mode that built it")

# hooks that a miner/trainer arms and a read must never inherit
_HOOKS = ("_PCV", "_SEV", "_CENSUS", "_IMP", "_STEP_TAP")


def hook_state():
    return {h: HEAD.__dict__.get(h) for h in _HOOKS}


def assert_hooks_clean(where, allow=("_SEV",)):
    """_SEV is the seal's own lazily-built constant Tensor([1.0]) and is
    checkpoint-independent; every other hook must be None in a read."""
    for h, v in hook_state().items():
        if h in allow:
            continue
        assert v is None, (
            f"{where}: phase1_algebra_head.{h} is armed ({type(v)}) — a "
            f"miner/trainer hook must never be live inside a read")
    assert not Tensor.training, (
        f"{where}: Tensor.training is True — only do_train sets it; a "
        f"read must never inherit it")


class env_scope:
    """Whole-environment snapshot/restore around one meter: no meter can
    leak a dial (SC_EVAL, ALG_INV, ALG_MINE_BREATHS, CR_*) into another."""

    def __init__(self, **kw):
        self.kw = kw

    def __enter__(self):
        self.saved = dict(os.environ)
        for k, v in self.kw.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        return self

    def __exit__(self, *a):
        os.environ.clear()
        os.environ.update(self.saved)
        return False


def banner():
    print(f"[read-batch] fixture={RB_TEST} ({_NAME}) rows-file={_PATH} "
          f"dev={os.environ.get('DEV')} one_mode={ONE_MODE}", flush=True)
    print(f"[read-batch] organs: loop_val={_LV_MOD} "
          f"step_engine_read={_SE_MOD} clock_read={_CR_MOD}", flush=True)
    print(f"[read-batch] jit-read={'ON' if jit_read.enabled() else 'off'}"
          + (f" (env key spans {len(_JRN)} names; slots capped at "
             f"{os.environ.get('ALG_JIT_READ_MAX', '32')})"
             if jit_read.enabled() else " (forward runs eager)"),
          flush=True)
    print(f"[read-batch] ckpts={CKPTS}", flush=True)
    print(f"[read-batch] meters={METERS}", flush=True)


def preflight(p):
    """Every checkpoint must fit THIS param dict — one process serves one
    architecture (ALG_HW / ALG_POLAR_D / ALG_FTYPES ... are read at head
    import). Fail here, not four checkpoints into a chain."""
    for c in CKPTS:
        sd = safe_load(c)
        assert set(sd.keys()) == set(p.keys()), (
            f"{c}: key mismatch vs build_params under this env "
            f"({sorted(set(sd) - set(p))[:4]} / "
            f"{sorted(set(p) - set(sd))[:4]}) — different architecture "
            f"dials; give it its own read_batch process")
        bad = [k for k in p if tuple(sd[k].shape) != tuple(p[k].shape)]
        assert not bad, (
            f"{c}: shape mismatch on {bad[:4]} — different architecture "
            f"dials; give it its own read_batch process")
    print(f"[read-batch] preflight OK: {len(CKPTS)} ckpts match "
          f"build_params ({len(p)} tensors)", flush=True)


def sc_eval_tripwire(p, data):
    """THE TRIPWIRE (the meter-divergence law: the check CALLS the organ
    — forward/build_slot_masks, the same functions loop_val composes).
    One batch through loop_val's two-pass cycle under SC_EVAL=0 and
    under SC_EVAL unset. With the shelf-circle seal armed the two modes
    are different computations, so the outputs MUST differ; if they
    agree bit-for-bit the seal is baked (a captured graph, or the seal
    branch is dark) and mixing modes in one process would hand back a
    wrong meter. Refuse in that case. This produces no number that
    leaves the function — it is a sensitivity probe, never a meter."""
    scm = int(os.environ.get("ALG_SHELF_CIRCLE", "0") or 0)
    if scm < 2:
        print(f"[read-batch] SC_EVAL tripwire SKIPPED: "
              f"ALG_SHELF_CIRCLE={scm} (<2 — the seal branch is not "
              f"armed in this env, so SC_EVAL cannot change a number)",
              flush=True)
        return
    vs, vst, vtk, vg, vse = data
    sl = np.arange(0, min(8, len(vs)))
    pad = 8 - len(sl)
    sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
    ts = Tensor(vst[sl_p].astype(np.float32), dtype=dtypes.float)
    tk = Tensor(vtk[sl_p].astype(np.float32), dtype=dtypes.float)
    se = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)

    def probe():
        """loop_val's own cycle on one batch: unmasked pass 1 ->
        build_slot_masks -> MASKED pass 2. The mask matters — the breath
        loop only runs when slot_mask is given (the loop-free-val
        finding), and the shelf-circle seal lives INSIDE the loop at
        SC_KB, so an unmasked forward is insensitive to SC_EVAL by
        construction and would make this tripwire cry wolf."""
        # Through jit_read's door, so that under ALG_JIT_READ=1 the
        # tripwire probes THE PATH THE METERS TAKE (a check must call
        # its organ). Door shut, these two lines are the eager calls
        # they replace, argument for argument.
        o0 = jit_read.read_forward(HEAD.forward, p, ts, tk, se,
                                   keys=("fat", "args", "res"))
        onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
        mk = HEAD.build_slot_masks(onp0, vse[sl_p].astype(np.int32))
        o = jit_read.read_forward(HEAD.forward, p, ts, tk, se,
                                  keys=("res",),
                                  slot_mask=Tensor(mk, dtype=dtypes.float))
        return o["res"].realize().numpy()

    with env_scope(SC_EVAL="0"):
        a = probe()
    with env_scope(SC_EVAL=None):
        b = probe()
    same = bool(np.array_equal(a, b))
    assert not same, (
        "SC_EVAL TRIPWIRE FAILED: SC_EVAL=0 and SC_EVAL-unset produced "
        "bit-identical outputs while ALG_SHELF_CIRCLE>=2 — the seal is "
        "baked in this process (THE UNLIT STOVE). Refusing to serve two "
        "SC_EVAL modes from one process; run one meter mode per "
        "read_batch (RB_ONE_MODE=1) instead.")
    print(f"[read-batch] SC_EVAL tripwire PASS: open vs sealed differ "
          f"(max|d|={float(np.abs(a - b).max()):.6g}) — the toggle is "
          f"live per forward() call in this process", flush=True)


def modes_of(meters):
    """Which SC_EVAL mode each meter needs, for the RB_ONE_MODE check."""
    m = set()
    for x in meters:
        m.add("0" if x == "loopval" else "unset")
    return m


def run_loopval(ckpt, p, data):
    """loop_val.main() — its read AND its print, so the [loop-val] line
    has exactly one author. SC_EVAL is the CALLER's (the chains set
    SC_EVAL=0 on every loop_val line); it is asserted present at
    startup, because an unset SC_EVAL silently turns loop_val into a
    SEALED read — a different meter with the same name."""
    with env_scope():
        assert_hooks_clean("loopval")
        n_ok, n_tot = loop_val.main(ckpt=ckpt, data=data, p=p)
        return {"fac_exact": n_ok / max(n_tot, 1), "n": n_tot}


def run_sealed(ckpt, p, data):
    """step_engine_read.main() with SC_EVAL UNSET — the chains' sealed
    line passes no SC_EVAL, and unset is what arms the mode-2 seal.
    SE_R is passed through untouched: the organ's own default (3)
    governs when the caller sets none, exactly as the standalone."""
    with env_scope(SC_EVAL=None):
        assert_hooks_clean("sealed")
        r = step_engine_read.main(ckpt=ckpt, data=data, p=p)
        return {"fac_exact": r["final"], "n": r["n_tot"], "R": r["R"]}


def run_clock(ckpt, planes):
    """clock_read's dials are module-level (read at import), so the organ
    is re-imported per read with the dials in the env — the module body
    is env reads and a chdir, nothing expensive, and phase1_algebra_head
    stays loaded. run(fixture) is called directly rather than main()
    because main() with CR_TEST=both forks a process per fixture, and
    this process already IS one fixture."""
    out = {}
    for pl in planes:
        # SC_EVAL UNSET: the chains' clock lines pass none, and clock_read
        # documents that it leaves SC_EVAL exactly as the caller left it —
        # so a base SC_EVAL=0 (which the loopval meter needs) would make
        # this a different read from the standalone one.
        with env_scope(SC_EVAL=None, CR_CKPT=ckpt, CR_PLANES=pl,
                       CR_TEST=RB_TEST):
            assert_hooks_clean("clock")
            mod = importlib.reload(clock_read)
            assert mod.CKPT == ckpt and mod.CR_PLANES == pl \
                and mod.CR_TEST == RB_TEST, (
                    "clock_read did not pick up the dials on reload "
                    f"({mod.CKPT} {mod.CR_PLANES} {mod.CR_TEST})")
            mod.run(RB_TEST)
            assert HEAD.ALG_TEST == _PATH and HEAD.TEST_NAME == _NAME, (
                "clock_read._install_envs moved ALG_TEST after the head "
                "was already bound — the printed fixture would be a lie")
            out[pl] = True
    return out


def main():
    banner()
    if "loopval" in METERS:
        assert os.environ.get("SC_EVAL", "") != "", (
            "RB_METERS includes loopval but SC_EVAL is unset. loop_val "
            "reads the caller's SC_EVAL; unset means the mode-2 SEAL, "
            "not the open read the chains take (they pass SC_EVAL=0 on "
            "every loop_val line). Set it explicitly — a mode this "
            "meter picks up by accident is a wrong number, not a slow "
            "one.")
    if "sealed" in METERS:
        _ser = os.environ.get("SE_R", "(unset -> the organ's default 3)")
        print(f"[read-batch] sealed meter: SE_R={_ser} "
              f"— the chains pass SE_R=1", flush=True)
    t0 = time.time()
    data = HEAD.load_alg("test")
    p = HEAD.build_params(0)
    t_setup_a = time.time() - t0
    preflight(p)

    if ONE_MODE:
        ms = modes_of(METERS)
        assert len(ms) == 1, (
            f"RB_ONE_MODE=1 but RB_METERS spans SC_EVAL modes {sorted(ms)} "
            f"— split them into one read_batch per mode")
        print("[read-batch] RB_ONE_MODE=1: single SC_EVAL mode "
              f"({ms.pop()}) in this process", flush=True)
    elif len(modes_of(METERS)) > 1:
        # first checkpoint's weights, so the tripwire probes a real organ
        sd = safe_load(CKPTS[0])
        for k in p:
            p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
        sc_eval_tripwire(p, data)
    else:
        # nothing to mix: every meter here wants the same SC_EVAL mode, so
        # the hazard the tripwire guards cannot arise — and the probe would
        # otherwise compile the OTHER mode's graph for nothing.
        print(f"[read-batch] SC_EVAL tripwire not needed: all meters run "
              f"one mode ({sorted(modes_of(METERS))[0]})", flush=True)
    t_setup = time.time() - t0
    print(f"[read-batch] setup {t_setup:.1f}s "
          f"(fixture+params {t_setup_a:.1f}s) — paid ONCE for "
          f"{len(CKPTS)} ckpt x {len(METERS)} meters", flush=True)

    results = {}
    times = []
    for ci, ckpt in enumerate(CKPTS):
        t_ck = time.time()
        for meter in METERS:
            tm = time.time()
            if meter == "loopval":
                r = run_loopval(ckpt, p, data)
            elif meter == "sealed":
                r = run_sealed(ckpt, p, data)
            else:
                r = run_clock(ckpt, meter[len("clock:"):].split("+"))
            dt = time.time() - tm
            results[(ckpt, meter)] = r
            val = (f"fac-exact={r['fac_exact']:.4f} (n={r['n']})"
                   if "fac_exact" in r else "see [clock] lines above")
            print(f"[read-batch] {os.path.basename(ckpt)} :: {meter} :: "
                  f"{val}  [{dt:.1f}s]", flush=True)
        times.append(time.time() - t_ck)

    print(f"[read-batch] per-ckpt wall: "
          + " ".join(f"{os.path.basename(c)}:{t:.1f}s"
                     for c, t in zip(CKPTS, times)), flush=True)
    if len(times) > 1:
        print(f"[read-batch] setup {t_setup:.1f}s amortized over "
              f"{len(CKPTS)} ckpts; incremental per extra ckpt "
              f"{sum(times[1:]) / (len(times) - 1):.1f}s vs "
              f"{times[0] + t_setup:.1f}s for a cold single read",
              flush=True)
    print(f"[read-batch] TOTAL {time.time() - T_START:.1f}s for "
          f"{len(CKPTS)} ckpt x {len(METERS)} meters on {RB_TEST}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
