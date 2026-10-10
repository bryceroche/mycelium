"""THE CLOCK PROBE for ALG_CLOCK_SEP (2026-10-09/10; CPU by default -- pass DEV=PCI+AMD to
read on the GPU for a real trained checkpoint, same as any other short read in this chain's
family).

At INIT (build_params(0), no checkpoint, random init): can breath time be read from the
REGISTER (state["clk"], surfaced as out["clk_all"] -- scripts/phase1_algebra_head.py's
breath_step/forward, the same ALG_MINE_BREATHS tap convention as breaths_u/breaths_r) the
same way clock_read.ridge_probe reads it from the bus's clock planes today ("the clock (no
gain, probe 1.0)", ledger 2026-09-08/09)? BAR (the build's own spec): probe >= 0.95 at init
-- the register's turn is the SAME deterministic six-wave code at full width, so it should
be at least as readable as today's narrower in-bus band, with no training and no gain.

ON THE TRAINED BODY (CSEP_CKPT set): does breath time survive 48k steps in the register?
STRICT key match against the checkpoint -- which means THE CALLER MUST SUPPLY THE EXACT
FAMILY ENV THE CHECKPOINT WAS TRAINED UNDER (copy verbatim from the chain log's own
"== ARM <TAG> (...)" line), the same way clocksep_grad_probe.py already required. THE BUG
THIS FIXES (2026-10-10, CS_241's own chain): this script used to carry a large block of its
own os.environ.setdefault(...) calls approximating "the plain recipe" -- close enough for a
fresh-init read (no checkpoint, nothing to key-match), but WRONG for a body trained under
SURF8 (ALG_ROUTER=2/ALG_PTR_SURF=role:../ALG_SPAN_*/the role BIND_CODES) or any other
family variant, because build_params()'s key SET depends on exactly which flags are set --
a checkpoint trained with SURF8 + ALG_POLAR_D=128 has W_role/W_rq2/W_rk2/r_gain/polar_wd/
polar_wu that a probe run under this script's OLD partial defaults never allocated, and
the resulting STRICT key-match assert correctly refused to silently load a mismatched
state (killed CS_241's chain at this exact step, released by hand after every verdict read
had already landed -- the arm's own result was unaffected). FIX: this script no longer sets
ANY recipe-defining env itself -- only ALG_POLAR/ALG_CLOCK_SEP/ALG_MINE_BREATHS (the probe's
OWN measurement conditions, not a recipe choice) are forced; everything else (ALG2, ALG_HW,
SURF8's ALG_ROUTER/ALG_PTR_SURF/ALG_SPAN_*/BIND_CODES, ALG_POLAR_D, ALG_ALT21, ALG_PRUNE,
ALG_TEST, DEV, ...) must come from the caller's own environment, exactly as the chain's own
"== ARM" line states it (clocksep_grad_probe.py's existing convention, now matched here).

Run plain for the POSITIVE read; run with ALG_SEVER=clockband (already set by the caller,
this script never touches ALG_SEVER itself) for the SEVER CHECK -- the register zeroed
leaving each breath -- which must read a LOWER probe (the register is where breath time
lives now; if the sever did not move it, nothing material happened). The chain/gate script
runs this twice (plain, then with ALG_SEVER=clockband in the environment) and compares.

Rows: CSEP_ROWS (default 8 with no checkpoint -- a quick init smoke; "all" with a
checkpoint -- the chain's own wild holdout is 311 rows, forward-only, no training).
Batched by CSEP_BATCH (default 32) to bound memory.

Usage (AT INIT, no checkpoint -- only the two forced flags matter):
  DEV=CPU .venv/bin/python3 scripts/clocksep_probe.py
  DEV=CPU ALG_SEVER=clockband .venv/bin/python3 scripts/clocksep_probe.py

Usage (ON A TRAINED BODY -- copy the FULL family env verbatim from the chain's own
"== ARM <TAG> (...)" line; DEV defaults to CPU, pass DEV=PCI+AMD under flock .cache/gpu.lock
for a real GPU read):
  env <FAM> <the arm's own X string> ALG_CLOCK_SEP=1 CSEP_CKPT=.cache/sharp_CS_241.safetensors \
      .venv/bin/python3 scripts/clocksep_probe.py
"""
import os
import sys

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
os.environ.setdefault("DEV", "CPU")       # default (never forced): pass DEV=PCI+AMD yourself for a GPU read
os.environ["ALG_POLAR"] = "1"             # THE SEPARATED CLOCK needs its bus -- the probe's own condition
os.environ["ALG_CLOCK_SEP"] = "1"         # ditto -- not a recipe choice, this probe's whole subject
os.environ["ALG_MINE_BREATHS"] = "1"      # arms the clk_all tap -- ditto
# EVERYTHING ELSE (ALG2/ALG_FTYPES/ALG_HW/.../SURF8's ALG_ROUTER etc./ALG_POLAR_D/ALG_TEST/...)
# must already be set by the caller -- see the module docstring. No setdefault here on
# purpose: a partial internal default was exactly the bug (silently wrong, not silently
# absent) that killed CS_241's chain at this step.

import numpy as np                                               # noqa: E402
from tinygrad import Tensor, dtypes                              # noqa: E402
import phase1_algebra_head as M                                  # noqa: E402
import clock_read as CR                                          # noqa: E402

_ckpt = os.environ.get("CSEP_CKPT", "")
_rows_arg = os.environ.get("CSEP_ROWS", "" if not _ckpt else "all") or "8"
_batch = int(os.environ.get("CSEP_BATCH", "32"))

vs, vst, vtk, vg, vse = M.load_alg("test")
N_total = len(vs)
n_rows = N_total if _rows_arg == "all" else min(int(_rows_arg), N_total)
print(f"[clocksep-probe] rows={n_rows}/{N_total} (CSEP_ROWS={_rows_arg!r}) batch={_batch} "
      f"ckpt={_ckpt or '(none -- fresh init)'} DEV={os.environ['DEV']}")

P = M.build_params(0)
if _ckpt:
    # THE CLOCK PROBE ON THE TRAINED BODY: load a real checkpoint (e.g. CS_241's own) in
    # place of the fresh-init read above -- does breath time survive 48k steps of training
    # in the register, or does it collapse/get overwritten? STRICT key match (the checkpoint
    # must have been trained under this exact family env -- see the module docstring).
    from tinygrad.nn.state import safe_load
    _sd = safe_load(_ckpt)
    assert set(_sd.keys()) == set(P.keys()), (
        "STRICT key mismatch loading CSEP_CKPT -- the caller's env does not match the "
        "checkpoint's own training recipe; copy the family env VERBATIM from the chain "
        "log's '== ARM <TAG> (...)' line (see this script's module docstring)",
        sorted(set(_sd) - set(P))[:6], sorted(set(P) - set(_sd))[:6])
    for _k in P:
        P[_k].assign(_sd[_k].to(P[_k].device).cast(P[_k].dtype)).realize()
    print(f"[clocksep-probe] loaded CSEP_CKPT={_ckpt} (strict key match)")

clk_pages = None   # list of K accumulator arrays, concatenated across batches
for lo in range(0, n_rows, _batch):
    sl = np.arange(lo, min(lo + _batch, n_rows))
    TS = Tensor(np.asarray(vst[sl], dtype=np.float32), dtype=dtypes.float)
    TK = Tensor(np.asarray(vtk[sl], dtype=np.float32), dtype=dtypes.float)
    SE = Tensor(np.asarray(vse[sl], dtype=np.int32), dtype=dtypes.int)
    SENT = np.asarray(vse[sl], dtype=np.int32)
    o0 = M.forward(P, TS, TK, SE)
    onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
    mk = M.build_slot_masks(onp0, SENT)
    o = M.forward(P, TS, TK, SE, slot_mask=Tensor(mk, dtype=dtypes.float))
    assert "clk_all" in o, "ALG_CLOCK_SEP did not surface out['clk_all'] -- check the build"
    batch_pages = [b.realize().numpy() for b in o["clk_all"]]
    K = len(batch_pages)
    assert K == 6, f"expected 6 register pages (breaths 1..6, no breath-0 register); got {K}"
    for j, a in enumerate(batch_pages):
        assert np.isfinite(a).all(), f"clk_all[{j}] non-finite (rows {sl[0]}..{sl[-1]})"
    if clk_pages is None:
        clk_pages = [[] for _ in range(K)]
    for j, a in enumerate(batch_pages):
        clk_pages[j].append(a)
    print(f"[clocksep-probe] rows {sl[0]}..{sl[-1]} done", flush=True)

clk_all = [np.concatenate(pages, axis=0) for pages in clk_pages]
D = clk_all[0].shape[-1]
assert D == 128, f"register width {D} != 128"
X = np.stack([a.reshape(-1, D) for a in clk_all], axis=0)        # (K, N, D), per-slot samples
pr = CR.ridge_probe(X, 242)
_sev = os.environ.get("ALG_SEVER", "") or "(none)"
_tag = "ON THE TRAINED BODY" if _ckpt else "AT BIRTH"
print(f"[clocksep-probe] ALG_SEVER={_sev} REGISTER breath probe {_tag}: "
      f"acc={pr['acc']:.4f} norm-only control={pr['norm_acc']:.4f} "
      f"(K={len(clk_all)} N={X.shape[1]} D={D})")
