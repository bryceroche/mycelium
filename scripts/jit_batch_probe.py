#!/usr/bin/env python3
"""jit_batch_probe.py — THE JIT GRAPH-BATCH gate (perf lever B, 2026-10-11).

THE BUG this gates: tinygrad's CPU HCQ backend (tinygrad/engine/jit.py's
graph_split_rewrite -> tinygrad/runtime/graph/hcq.py's HCQGraph, running on
tinygrad/runtime/ops_cpu.py's thread-pool queue emulation) can SEGFAULT on the
REPLAY call (the first real execution of a captured TinyJit graph, i.e. training
step 1) once a training step's forward+backward graph gets batched into large
enough HCQGraph calls. First found on THE LOOPED TRANSFORMER (ALG_DEPTH=N,
2026-10-09) and reproduced independently on THE MIXER family (mixoff, 2026-10-10).
See memory/reference_tinygrad_am_quirks.md (2026-10-09 entry) and
.cache/loopedtf_gate.sh / .cache/mixer_gate.sh for the organ-level precedent.

WHAT WE LEARNED BISECTING IT (2026-10-11, ledger: THE JIT GRAPH-BATCH HANG):
  - graph_split_rewrite() (tinygrad/engine/jit.py) doubles its own batch-size
    cap on EVERY successful flush (`max_batch_size *= 2`), regardless of
    whether that flush was triggered by hitting the cap or by a forced
    device/op boundary. For a large enough total kernel count this runs away:
    an unset JIT_BATCH_SIZE (default 32) grew 2 -> 64 -> 128 -> 256 -> 512 ->
    1024 -> 2048 on the looped-transformer (ALG_DEPTH=4) fixture before the
    1024-kernel graph crashed.
  - It is NOT a simple "one graph too big" ceiling: pinning the cap FLAT (a
    local, uncommitted tinygrad patch adding JITBATCH_NODOUBLE=1 to skip the
    `*= 2`) still crashed — a flat JIT_BATCH_SIZE=768 ran ONE 768-kernel graph
    fine and crashed building/running the SECOND one; a flat 256 ran FIVE
    256-kernel graphs fine and crashed on the SIXTH. The failure tracks
    something CUMULATIVE across HCQGraph objects within one captured step,
    empirically in the ~1000-2000-total-graphed-kernels range for the
    ALG_DEPTH=4 fixture — NOT simply "total kernels this step" either: THE
    ROLE SIGNATURE's role8 chassis (no ALG_DEPTH) safely built a 1024-kernel
    graph and reached a cumulative 2891 graphed kernels in the SAME step
    shape with zero crash. The two workloads differ in kernel content (larger
    self-attn/cross-attn/FFN tensors under ALG_DEPTH), not just count, and we
    did not pin the exact C-level invariant that overflows (no gdb/core
    backtrace obtained — flagged unverified). Under -X faulthandler the crash
    printed TWO overlapping "Fatal Python error: Segmentation fault" headers
    at once, i.e. two OS threads faulted close together — consistent with a
    concurrency bug in ops_cpu.py's CPUWorker/push_task thread-pool queue
    emulation (peak observed thread count during one crash: 50, i.e. not a
    thread-exhaustion blowup by itself).
  - ONLY JIT_BATCH_SIZE=1 (which makes graph_split_rewrite never merge more
    than one kernel per HCQGraph, so the doubling path never fires) is
    confirmed safe on CPU for the looped-transformer and mixer-severed organs,
    with role8/unset verified BIT-IDENTICAL under it (loopedtf_gate.sh,
    mixer_gate.sh). JIT_BATCH_SIZE=4 reproduces the SAME crash (verified here);
    16/64 were not independently run (same unconditional-doubling mechanism;
    treat as equally unsafe until shown otherwise).
  - UNVERIFIED: the real device (DEV=PCI+AMD) manifests the analogous failure
    as "RuntimeError: Device hang detected" rather than a segfault (per
    memory/reference_tinygrad_am_quirks.md) — presumably the same oversized-
    single-graph root cause hitting a different resource limit in the real
    HCQ hardware-queue path, but this has NOT been confirmed on-device by this
    probe (never touch .cache/gpu.lock / the live GPU from here). Any arm that
    newly crosses this gate on CPU must still re-smoke on the device before
    firing a production run (rack_chain_LT.sh's pattern: a tiny 2-step GPU
    smoke under the JIT_BATCH_SIZE=1 workaround BEFORE the real arm).

WHAT THIS SCRIPT DOES: runs a tiny (BATCH=2 STEPS=2) DEV=CPU training probe of
a candidate organ's env recipe against the tiny64 fixture, with DEBUG=2, and
parses tinygrad's own "JIT GRAPHing batch with N kernels" / "*** CPU ... batched
N" lines to report: every HCQGraph batch size tinygrad tried to build, the max
single batch, the cumulative graphed-kernel count, and whether the process
crashed. It then applies the conservative empirical rule above (max batch >=
512 OR cumulative >= 1200 OR an observed crash => FLAG) and, if flagged (or if
--always-confirm is passed), re-runs the SAME recipe once more with
JIT_BATCH_SIZE=1 to confirm the workaround actually clears it, reporting both
runs' step-1 timings so a caller can see the performance tax paid for safety.

USAGE:
  .venv/bin/python3 scripts/jit_batch_probe.py --extra-env "ALG_DEPTH=4"
  .venv/bin/python3 scripts/jit_batch_probe.py --extra-env "ALG_MIXER_OFF=1" --warm .cache/sharp_balV242.safetensors

This NEVER sets DEV to anything but CPU (hardcoded) and never touches
.cache/gpu.lock — it is a CPU-only gate, matching how the bug was found. Any
throwaway checkpoint it writes is deleted on exit (success, failure, or crash).
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The role8 chassis (THE ROLE SIGNATURE) — the FAM/S8/TFIX recipe shared by
# loopedtf_gate.sh and mixer_gate.sh, reused here so a new organ's probe is
# apples-to-apples with the two organs that already bisected this bug.
_S8 = (
    "ALG_ROUTER=2 R_GAIN_INIT=1.0 ALG_FREEZE=r_gain ALG_ROUTER_PTR=0.0 "
    "ALG_SPAN_ALL=1 ALG_SPAN_ARGS=1 ALG_SPAN_OP=1 ALG_SPAN_RCUE=1 ALG_SPAN_ARCUE=1 "
    "ALG_PTR_SURF=role:add:2.0 BIND_CODES=.cache/bindbus_codes512r.npz"
)
_FAM = (
    "ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_BREATH=7 ALG_NOTEBOOK=1 "
    "ALG_SIXWAVE=1 NB_PERSLOT=1 ALG_BINDBUS=7 ALG_BIND_D=512 BIND_CODES=.cache/bindbus_codes512.npz "
    "ALG_BUSGARAGE=2 ALG_SHELF_CIRCLE=2 ALG_ALTMASK=1 ALG_ALT21=1 ALG_ALT2=1 ALG_MASKHEAD=1 "
    "ALG_FED=1 ALG_POLAR=1 ALG_POLAR_D=128 ALG_POLAR_EM=0.1 ALG_POLAR_D_INIT=.cache/polar_waist_init_d128u.npz "
    "ALG_PRUNE=pforms,s4,fednl0,lane2 ALG_SLOT_ALL=1 ALG_STELLAR=2 ALG_CLOCK_CANON=1 SC_EVAL=0"
)
_TFIX = (
    "ALG_ALLOW_PEN_TRAIN=1 ALG_TRAIN=.cache/form_tiny64.jsonl ALG_TRAIN_NAME=tiny64 "
    "ALG_TEST=.cache/test_tiny64.jsonl ALG_TEST_NAME=testtiny64 ALG_MASKPREP_CACHE=1 "
    "ALG_MASKPREP_IGNORE=ALG_MASK_COOK,ALG_MASK_COOK_SKEL,ALG_TOK_COOK,ALG_TOK_COOK_S3,ALG_TG_INIT,"
    "ALG_BAL_COOK,ALG_NB2,ALG_ALT5,ALG_PRUNE,ALG_MASKPREP_JIT,ALG_MASKPREP_B,ALG_FACTS_POOL,"
    "ALG_FACTS_FLUSH,ALG_JIT_VAL"
)

# Conservative thresholds from the 2026-10-11 bisection (see module docstring).
# role8 (safe): max single batch 1024, cumulative 2891. ALG_DEPTH=4 (crashes):
# flat caps of 768 and 256 both eventually crashed; the exponential default
# crashed building a 1024-kernel graph after a 962-kernel cumulative. We flag
# anything that reaches the lower edge of the observed-unsafe band.
MAX_BATCH_WARN = 512
CUMULATIVE_WARN = 1200


def _parse_env_string(s: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for tok in s.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            env[k] = v
    return env


def run_probe(extra_env: dict[str, str], warm: str, ckpt_path: str, jit_batch_size: str | None,
               timeout: int) -> dict:
    env = dict(os.environ)
    env.update(_parse_env_string(_FAM))
    env.update(_parse_env_string(_TFIX))
    env.update(_parse_env_string(_S8))
    env.update(extra_env)
    env["DEV"] = "CPU"  # hardcoded -- this probe never touches the GPU
    env["PYTHONUNBUFFERED"] = "1"
    env["DEBUG"] = "2"
    env["LR"] = env.get("LR", "1e-4")
    env["SEED"] = env.get("SEED", "242")
    env["WARM_FROM"] = warm
    env["SNAP_EVERY"] = "4000"
    env["BATCH"] = "2"
    env["STEPS"] = "2"
    env["ALG_CKPT"] = ckpt_path
    if jit_batch_size is not None:
        env["JIT_BATCH_SIZE"] = jit_batch_size
    else:
        env.pop("JIT_BATCH_SIZE", None)

    cmd = [os.path.join(_ROOT, ".venv", "bin", "python3"), "-u",
           os.path.join(_ROOT, "scripts", "phase1_algebra_head.py"), "--train"]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=_ROOT, env=env, capture_output=True, text=True,
                               timeout=timeout)
        rc, out = proc.returncode, proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as e:
        rc, out = -9, (e.stdout or "") + (e.stderr or "") + "\n[jit_batch_probe] TIMEOUT"
    wall = time.time() - t0

    batches = [int(m) for m in re.findall(r"JIT GRAPHing batch with (\d+) kernels", out)]
    step_lines = re.findall(r"step\s+(\d+)\s+loss=([0-9.eE+-]+).*?\(([0-9.]+)s/step\)", out)
    crashed = rc < 0 or rc == 139 or "Segmentation fault" in out or "Fatal Python error" in out
    return {
        "rc": rc, "wall_s": wall, "batches": batches,
        "max_batch": max(batches) if batches else 0,
        "cumulative": sum(batches),
        "steps": step_lines,
        "crashed": crashed,
        "raw_tail": "\n".join(out.splitlines()[-15:]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extra-env", default="", help="extra ALG_*/env overrides for the candidate organ, "
                                                      "e.g. \"ALG_DEPTH=4\" or \"ALG_MIXER_OFF=1\"")
    ap.add_argument("--warm", default=os.path.join(".cache", "sharp_balV242.safetensors"),
                     help="WARM_FROM checkpoint (default: the role8 chassis checkpoint)")
    ap.add_argument("--timeout", type=int, default=400, help="seconds before a run is declared hung")
    ap.add_argument("--always-confirm", action="store_true",
                     help="also run the JIT_BATCH_SIZE=1 confirmation even if the unprotected run looks safe")
    args = ap.parse_args()

    extra_env = _parse_env_string(args.extra_env)
    tmpdir = tempfile.mkdtemp(prefix="jit_batch_probe_")
    ckpt = os.path.join(tmpdir, "probe.safetensors")
    try:
        print(f"[jit_batch_probe] unprotected run (JIT_BATCH_SIZE unset, default doubling), "
              f"extra_env={extra_env or '{}'}")
        unprotected = run_probe(extra_env, args.warm, ckpt, jit_batch_size=None, timeout=args.timeout)
        _report("unprotected", unprotected)

        near_danger = (unprotected["max_batch"] >= MAX_BATCH_WARN
                       or unprotected["cumulative"] >= CUMULATIVE_WARN)
        if unprotected["crashed"]:
            verdict = "CRASHED -- this organ hits THE JIT GRAPH-BATCH HANG on CPU"
        elif near_danger:
            # role8 itself reaches max_batch=1024, cumulative=2891 with ZERO crash -- these
            # thresholds are NOT a hard predictive rule (content, not just size, matters; see
            # the module docstring), so a clean run past them is NOT itself a failure. Treat it
            # as "ran clean, but in the size range where another organ (ALG_DEPTH=4) has crashed
            # before" -- worth a confirmation pass, not an alarm.
            verdict = "ran clean, but past the sizes where another organ has crashed before -- confirm"
        else:
            verdict = "within the observed-safe range (role8-class)"
        print(f"\n[jit_batch_probe] VERDICT: {verdict} "
              f"(max_batch={unprotected['max_batch']}, cumulative={unprotected['cumulative']}, "
              f"crashed={unprotected['crashed']})")

        flagged = unprotected["crashed"]
        if flagged or near_danger or args.always_confirm:
            print("\n[jit_batch_probe] confirming JIT_BATCH_SIZE=1 (the only verified-safe CPU workaround)...")
            protected = run_probe(extra_env, args.warm, ckpt, jit_batch_size="1", timeout=args.timeout)
            _report("JIT_BATCH_SIZE=1", protected)
            if protected["crashed"]:
                print("\n[jit_batch_probe] ALARM: JIT_BATCH_SIZE=1 ALSO crashed -- this organ needs a fresh "
                      "bisection, the known workaround does not clear it. Do not fire this on the device.")
                return 2
            print("\n[jit_batch_probe] JIT_BATCH_SIZE=1 clears it (role8/unset numbers stay bit-identical "
                  "under it per loopedtf_gate.sh/mixer_gate.sh precedent -- verify that separately for a "
                  "brand-new organ, this probe only confirms it runs without crashing). NOT YET VERIFIED ON "
                  "THE DEVICE (DEV=PCI+AMD) -- re-smoke there before firing a production arm.")
        return 1 if flagged else 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _report(label: str, r: dict) -> None:
    print(f"  [{label}] rc={r['rc']} wall={r['wall_s']:.1f}s crashed={r['crashed']}")
    print(f"  [{label}] HCQ graph batches built: {r['batches']} "
          f"(max={r['max_batch']}, cumulative={r['cumulative']})")
    for step, loss, secs in r["steps"]:
        print(f"  [{label}] step {step} loss={loss} ({secs}s/step)")
    if r["crashed"]:
        print(f"  [{label}] tail of output:\n" + "\n".join(f"    {l}" for l in r["raw_tail"].splitlines()))


if __name__ == "__main__":
    sys.exit(main())
