"""Per-B worker for the perceiver B-sweep.

Run with DEBUG=2 set in env before import. All kernel output goes to stdout/stderr.
We use sentinel prints around the final measured step to extract only that step's kernels.

Env inputs: PERCEIVER_TASK, K, B, BALL_PATH, KENKEN_TRAIN, KENKEN_TEST, SEED, OUT_JSON
Perf toggles (all OFF): PERCEIVER_HOIST_BIAS=0 PERCEIVER_FP16_THINK=0 PERCEIVER_DEFUSE_BREATH=0
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv, GlobalCounters
from tinygrad.engine.jit import TinyJit
from tinygrad.nn.optim import AdamW

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.kenken_data import N_CELLS, N_MAX, load_jsonl
from mycelium.kenken import kenken_constraint_energy
from mycelium.perceiver_poincare import (
    PERCEIVER_K_MAX, PERCEIVER_N_GLOBAL,
    PERCEIVER_HOIST_BIAS, PERCEIVER_FP16_THINK, PERCEIVER_DEFUSE_BREATH,
    attach_perceiver_params,
    perceiver_breathing_forward,
    perceiver_gphi_parameters, perceiver_active_cell_coords,
)
from mycelium.perceiver_poincare_data import PerceiverLoader, latent_capacity
from scripts.perceiver_train import collect_params, cast_layers_fp32


def parse_debug2_text(text: str):
    """Parse DEBUG=2 kernel lines from text block. Returns list of (ms, knum, name, gflops, occ, bw)."""
    rows = []
    for line in text.splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = re.search(
            r"AMD\s+(\d+)\s+(.*?)\s+arg\s+\d+\s+mem.*?tm\s+([\d.]+)(us|ms|s)\b.*?"
            r"\(\s*(\d+)\s+GFLOPS\s+(\d+)\|(\d+)\s+GB/s",
            line
        )
        if not m:
            continue
        num = int(m.group(1))
        name = m.group(2).strip()
        t = float(m.group(3))
        u = m.group(4)
        gf = int(m.group(5))
        occ = int(m.group(6))
        bw = int(m.group(7))
        ms = t * 1000 if u == "s" else (t if u == "ms" else t / 1000)
        rows.append((ms, num, name, gf, occ, bw))
    return rows


def find_eater(rows):
    """Return the dominant slow kernel (the EATER).
    Primary: largest occ=0 kernel (the original eater target).
    If no occ=0 kernels exist or they are all trivially small (<1ms),
    fall back to the slowest kernel overall — which in a JIT-batched step
    will be the largest 'batched N' group containing the backward pass."""
    if not rows:
        return None
    occ0 = [r for r in rows if r[4] == 0]
    occ0_big = [r for r in occ0 if r[0] > 1.0]  # > 1ms
    if occ0_big:
        return max(occ0_big, key=lambda r: r[0])
    # Fallback: slowest kernel overall (typically the large batched backward group)
    return max(rows, key=lambda r: r[0])


def build_jit_step(model, opt, K: int, B: int, L: int, ball_path: str):
    cw = 0.3
    gc_val = 1.0
    jit_params = opt.params

    @TinyJit
    def _step(input_cells: Tensor, gold: Tensor, cell_valid: Tensor,
              value_domain_mask: Tensor, latent_membership: Tensor,
              latent_valid: Tensor, latent_type: Tensor):
        opt.zero_grad()

        class _B:
            pass
        batch = _B()
        batch.input_cells = input_cells
        batch.gold = gold
        batch.cell_valid = cell_valid
        batch.value_domain_mask = value_domain_mask
        batch.latent_membership = latent_membership
        batch.latent_valid = latent_valid
        batch.latent_type = latent_type

        cell_logits_history, _, _, _, _ = perceiver_breathing_forward(
            model, batch, K=K, ball_path=ball_path, collect_engagement=False)

        observed = (input_cells > 0).cast(dtypes.float)
        supervise = cell_valid * (1.0 - observed)
        sup_sum = supervise.sum() + 1e-6
        gold_idx = (gold - 1).clip(0, N_MAX - 1).reshape(B * N_CELLS)
        supervise_flat = supervise.reshape(B * N_CELLS)

        cell_loss_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
        weight_sum = 0.0
        for k, logits in enumerate(cell_logits_history):
            weight_k = 1.0 + float(k) / float(K - 1) if K > 1 else 1.0
            ce_elems = logits.reshape(B * N_CELLS, N_MAX).sparse_categorical_crossentropy(
                gold_idx, reduction="none")
            ce_k = (ce_elems * supervise_flat).sum() / sup_sum
            cell_loss_sum = cell_loss_sum + ce_k * weight_k
            weight_sum += weight_k
        cell_loss = cell_loss_sum / float(weight_sum)

        final_probs = cell_logits_history[-1].softmax(axis=-1)
        energy = kenken_constraint_energy(final_probs, batch).mean()
        total = cell_loss + cw * energy
        total.backward()

        healthy_b = total.isfinite()
        for p in jit_params:
            if p.grad is not None:
                p.grad = healthy_b.where(p.grad, Tensor.zeros_like(p.grad))

        sq_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
        for p in jit_params:
            if p.grad is not None:
                sq_sum = sq_sum + p.grad.cast(dtypes.float).square().sum()
        gnorm = (sq_sum + 1e-12).sqrt()
        scale = (gc_val / gnorm).clip(max_=1.0)
        for p in jit_params:
            if p.grad is not None:
                p.grad = p.grad.cast(dtypes.float) * scale

        opt.step()
        return (total.detach().realize(),)

    return _step


def run_step(step_fn, loader):
    batch = loader.sample_batch()
    t0 = time.perf_counter()
    step_fn(
        batch.input_cells, batch.gold, batch.cell_valid,
        batch.value_domain_mask, batch.latent_membership,
        batch.latent_valid, batch.latent_type,
    )
    Device[Device.DEFAULT].synchronize()
    return time.perf_counter() - t0


def _write_result(path: str, data: dict) -> None:
    def _clean(v):
        if isinstance(v, (np.float32, np.float64)):
            return float(v)
        if isinstance(v, (np.int32, np.int64)):
            return int(v)
        if isinstance(v, list):
            return [_clean(x) for x in v]
        if isinstance(v, dict):
            return {k: _clean(vv) for k, vv in v.items()}
        return v
    with open(path, "w") as f:
        json.dump(_clean(data), f, indent=2)


def main():
    assert int(getenv("PERCEIVER_TASK", 0)) > 0, "PERCEIVER_TASK=1 must be set"

    K = int(getenv("K", "8"))
    B = int(getenv("B", "8"))
    BALL_PATH = getenv("BALL_PATH", "per_constraint")
    KENKEN_TRAIN = getenv("KENKEN_TRAIN", ".cache/kenken_train_curriculum.jsonl")
    KENKEN_TEST = getenv("KENKEN_TEST", ".cache/kenken_test_curriculum.jsonl")
    SEED = int(getenv("SEED", "42"))
    OUT_JSON = getenv("OUT_JSON", f"/tmp/perceiver_bsweep_B{B}.json")
    # Path to the log file that the orchestrator is capturing our output to
    # We parse this file after the measure step.
    LOG_FILE = getenv("LOG_FILE", f"/tmp/perceiver_bsweep_B{B}.log")

    debug_level = int(os.environ.get("DEBUG", "0"))
    print(f"[worker B={B}] K={K} ball_path={BALL_PATH} DEBUG={debug_level} "
          f"HOIST={PERCEIVER_HOIST_BIAS} FP16={PERCEIVER_FP16_THINK} "
          f"DEFUSE={PERCEIVER_DEFUSE_BREATH}", flush=True)

    np.random.seed(SEED)
    Tensor.manual_seed(SEED)

    train_recs = load_jsonl(KENKEN_TRAIN)
    test_recs = load_jsonl(KENKEN_TEST)
    corpus_n_cages_max = max(
        max(len(r["cages"]) for r in train_recs),
        max(len(r["cages"]) for r in test_recs),
    )
    N_CAGES_MAX = int(getenv("KENKEN_N_CAGES_MAX", str(corpus_n_cages_max)))
    L_MAX = latent_capacity(N_CAGES_MAX, PERCEIVER_N_GLOBAL)
    print(f"[worker B={B}] n_cages_max={N_CAGES_MAX}  L_max={L_MAX}", flush=True)

    cfg = Config()
    print(f"[worker B={B}] Loading Pythia-410M ...", flush=True)
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    del sd
    cast_layers_fp32(model)
    attach_perceiver_params(model, hidden=cfg.hidden, n_heads=cfg.n_heads,
                            L_max=L_MAX, k_max=K)
    Device[Device.DEFAULT].synchronize()

    loader = PerceiverLoader(KENKEN_TRAIN, batch_size=B, seed=SEED,
                             n_cages_max=N_CAGES_MAX)
    params = collect_params(model, ball_path=BALL_PATH)
    opt = AdamW(params, lr=3e-5, weight_decay=0.0)
    n_params = sum(int(np.prod(t.shape)) for t in params)
    print(f"[worker B={B}] Trainable params: {n_params/1e6:.1f}M", flush=True)

    Tensor.training = True
    step_fn = build_jit_step(model, opt, K=K, B=B, L=L_MAX, ball_path=BALL_PATH)

    # First call: triggers actual kernel compilation
    print(f"[worker B={B}] First call (kernel compilation)...", flush=True)
    try:
        t_first = run_step(step_fn, loader)
        print(f"[worker B={B}] First call done: {t_first:.2f}s", flush=True)
    except Exception as e:
        print(f"[worker B={B}] OOM on first call: {e}", flush=True)
        _write_result(OUT_JSON, {"oom": True, "error": str(e), "B": B, "K": K})
        return

    # Warmup steps
    print(f"[worker B={B}] Warmup (3 steps)...", flush=True)
    try:
        for wi in range(3):
            t_w = run_step(step_fn, loader)
            print(f"[worker B={B}] warmup {wi+1}: {t_w:.3f}s", flush=True)
    except Exception as e:
        print(f"[worker B={B}] OOM in warmup: {e}", flush=True)
        _write_result(OUT_JSON, {"oom": True, "error": str(e), "B": B, "K": K})
        return

    # Timed steps (wall clock)
    print(f"[worker B={B}] Timing (3 steps)...", flush=True)
    step_times = []
    try:
        for ti in range(3):
            t_s = run_step(step_fn, loader)
            step_times.append(t_s)
            print(f"[worker B={B}] timed step {ti+1}: {t_s:.3f}s", flush=True)
    except Exception as e:
        print(f"[worker B={B}] OOM in timing: {e}", flush=True)
        _write_result(OUT_JSON, {"oom": True, "error": str(e), "B": B, "K": K})
        return

    s_per_step = float(np.median(step_times))
    examples_per_sec = B / s_per_step
    print(f"[worker B={B}] s/step={s_per_step:.3f}  ex/s={examples_per_sec:.2f}", flush=True)

    # DEBUG=2 profiling step: print sentinels around it so we can slice the log
    # The log file path is passed in via LOG_FILE env so we can re-read it below.
    print(f"PERCEIVER_BSWEEP_MEASURE_START B={B}", flush=True)
    sys.stderr.flush()
    try:
        run_step(step_fn, loader)
    except Exception as e:
        print(f"PERCEIVER_BSWEEP_MEASURE_DONE B={B} error={e}", flush=True)
        _write_result(OUT_JSON, {
            "oom": False, "B": B, "K": K,
            "s_per_step": s_per_step, "step_times": step_times,
            "examples_per_sec": examples_per_sec,
            "eater_occ": None, "eater_gflops": None,
            "debug2_error": str(e),
        })
        return
    sys.stdout.flush()
    sys.stderr.flush()
    print(f"PERCEIVER_BSWEEP_MEASURE_DONE B={B}", flush=True)
    sys.stdout.flush()

    # Parse the log file we're writing to
    # The kernel output comes from DEBUG=2 which goes to stderr; the orchestrator
    # redirects both stdout and stderr to the log file (stderr=STDOUT).
    # We read the log file and extract kernel lines between our sentinels.
    eater_occ = None
    eater_gflops = None
    eater_ms = None
    eater_pct = None
    eater_name = None
    eater_knum = None
    total_kernels = 0
    total_ms_debug = 0.0

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            log_text = f.read()

        # Find the LAST occurrence of the start sentinel
        start_marker = f"PERCEIVER_BSWEEP_MEASURE_START B={B}"
        done_marker = f"PERCEIVER_BSWEEP_MEASURE_DONE B={B}"
        start_idx = log_text.rfind(start_marker)
        done_idx = log_text.rfind(done_marker)

        if start_idx >= 0:
            end_idx = done_idx if done_idx > start_idx else len(log_text)
            measure_section = log_text[start_idx:end_idx]
            kernel_rows = parse_debug2_text(measure_section)
            total_kernels = len(kernel_rows)
            print(f"[worker B={B}] Parsed {total_kernels} kernel rows from measure section "
                  f"({len(measure_section)} chars)", flush=True)

            if kernel_rows:
                total_ms_debug = sum(r[0] for r in kernel_rows)
                eater = find_eater(kernel_rows)
                if eater:
                    e_ms, e_knum, e_name, e_gf, e_occ, e_bw = eater
                    e_pct = 100.0 * e_ms / total_ms_debug if total_ms_debug > 0 else 0.0
                    eater_occ = e_occ
                    eater_gflops = e_gf
                    eater_ms = e_ms
                    eater_pct = e_pct
                    eater_name = e_name[:80]
                    eater_knum = e_knum
                    print(f"[worker B={B}] EATER: knum={e_knum} occ={e_occ} "
                          f"GFLOP/s={e_gf} ms={e_ms:.2f} ({e_pct:.1f}%) "
                          f"name={e_name[:50]}", flush=True)

                # Top 10
                top10 = sorted(kernel_rows, reverse=True)[:10]
                print(f"[worker B={B}] Top-10 kernels:")
                for row in top10:
                    ms_r, knum_r, name_r, gf_r, occ_r, bw_r = row
                    pct_r = 100.0 * ms_r / total_ms_debug if total_ms_debug else 0
                    print(f"  knum={knum_r:5d} occ={occ_r:2d} "
                          f"GFLOP/s={gf_r:6d} ms={ms_r:8.2f} "
                          f"({pct_r:5.1f}%) {name_r[:60]}", flush=True)

                occ0_rows = [r for r in kernel_rows if r[4] == 0]
                all_occ0_ms = sum(r[0] for r in occ0_rows)
                print(f"[worker B={B}] occ=0 kernels: {len(occ0_rows)} "
                      f"total_ms={all_occ0_ms:.2f} "
                      f"({100*all_occ0_ms/total_ms_debug:.1f}% of step)", flush=True)
            else:
                print(f"[worker B={B}] WARNING: no kernel rows in measure section", flush=True)
                # Print the measure section for diagnosis
                print(f"[worker B={B}] Measure section first 500 chars: "
                      f"{measure_section[:500]!r}", flush=True)
        else:
            print(f"[worker B={B}] WARNING: sentinel not found in log file", flush=True)
    else:
        print(f"[worker B={B}] WARNING: log file {LOG_FILE} not found", flush=True)

    result = {
        "oom": False,
        "B": B, "K": K, "L_MAX": L_MAX, "ball_path": BALL_PATH,
        "s_per_step": s_per_step,
        "step_times": step_times,
        "examples_per_sec": examples_per_sec,
        "eater_occ": eater_occ,
        "eater_gflops": eater_gflops,
        "eater_ms": eater_ms,
        "eater_pct": eater_pct,
        "eater_name": eater_name,
        "eater_knum": eater_knum,
        "total_kernels": total_kernels,
        "total_ms_debug": total_ms_debug,
        "n_params_M": n_params / 1e6,
        "debug_level": debug_level,
    }
    _write_result(OUT_JSON, result)
    print(f"[worker B={B}] Done. Results in {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
