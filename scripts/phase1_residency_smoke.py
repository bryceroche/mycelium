"""phase1_residency_smoke.py — the Phase-1 build-order step 1 gate (spec §6.1).

THREE QUESTIONS, one GPU session (load-and-forward, NO training):
  (1) RESIDENCY: do the frozen Llama-3.2-1B L0-L3 trunk (2048d, GQA 32q/8kv) and the
      validated KenKen deducer (fg_kenken_k16_reg) fit TOGETHER on the 7900 XTX — and,
      the number that matters for planning, HOW MUCH HEADROOM REMAINS for the delta
      head + waist + capture hooks + notebook conditioning + Brick-A's batch dim?
      Deliverable: the memory-budget line for CLAUDE.md §5.
  (2) SUBSTRATE HAZARD: Llama-3.2-1B L0-L3 is a NEW JIT graph shape on the AM driver
      (the K=28 hang was discovered empirically, not predicted). One TinyJit'd repeated
      forward with the assign-in-place fixed-buffer replay pattern (compile once,
      replay) — confirm the new trunk doesn't trip anything BEFORE the delta head is
      built on top of it.
  (3) SANITY: real tokenized KenKen-in-words text (the actual Phase-1 input
      distribution, via the on-disk Llama tokenizer) through the frozen trunk produces
      FINITE activations with sane norms; the co-resident deducer still reproduces its
      known eval numbers after the Llama allocations.

MEASUREMENT: tinygrad GlobalCounters.mem_used (allocated buffer bytes) sampled after
each stage + peak; sysfs VRAM (mem_info_vram_used) when readable. Stages:
  S0 baseline -> S1 llama loaded -> S2 llama fwd (B=8,T=512) -> S3 JIT replay x3
  -> S4 deducer loaded -> S5 deducer eval batch.

USAGE:
  DEV=AMD .venv/bin/python3 scripts/phase1_residency_smoke.py
  (env: LLAMA_WEIGHTS to override the weights path; B/T for the llama batch shape)
"""
from __future__ import annotations

import json
import os
import sys
import time

_THIS_FILE = os.path.abspath(__file__)
_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(_THIS_FILE))

import numpy as np

LLAMA_WEIGHTS = os.environ.get(
    "LLAMA_WEIGHTS", os.path.join(_ROOT, ".cache/llama-3.2-1b-weights/model.safetensors"))
TOKENIZER_JSON = os.path.join(_ROOT, ".cache/llama-3.2-1b-weights/tokenizer.json")
FG_CKPT = os.environ.get(
    "FG_CKPT", ".cache/fg_ckpts/fg_kenken_k16_reg/fg_kenken_k16_reg_final.safetensors")
NL_SAMPLE = os.environ.get("NL_SAMPLE", ".cache/kenken_nl_test.jsonl")
B = int(os.environ.get("B", "8"))
T = int(os.environ.get("T", "512"))


def _mem(tag: str, table: list) -> None:
    from tinygrad.helpers import GlobalCounters
    mb = GlobalCounters.mem_used / 1e6
    vram = ""
    for p in ("/sys/class/drm/card0/device/mem_info_vram_used",
              "/sys/class/drm/card1/device/mem_info_vram_used"):
        try:
            vram = f"  sysfs_vram={int(open(p).read()) / 1e6:.0f}MB"
            break
        except Exception:
            pass
    print(f"[mem] {tag:28s} tinygrad_alloc={mb:8.1f}MB{vram}", flush=True)
    table.append((tag, mb))


def main() -> None:
    from tinygrad import Tensor, Device, dtypes
    from tinygrad.engine.jit import TinyJit

    table: list = []
    _mem("S0 baseline", table)

    # ---- S1: frozen Llama-3.2-1B L0-L3 ----
    from mycelium.llama_loader import (
        attach_llama_layers, load_llama_weights, LLAMA_3_2_1B_CFG, _rms_norm,
    )

    class _Host:  # bare attribute host (mirrors the model objects the loaders expect)
        pass

    host = _Host()
    sd = load_llama_weights(LLAMA_WEIGHTS)
    attach_llama_layers(host, n_layers=4, sd=sd, cfg=LLAMA_3_2_1B_CFG)
    del sd
    cfg = host.llama_cfg
    Device[Device.DEFAULT].synchronize()
    _mem("S1 llama L0-L3 + embed", table)

    # ---- S3 prep: real KenKen-in-words tokens (the actual Phase-1 input) ----
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOKENIZER_JSON)
    texts = [json.loads(l)["text"] for l in open(NL_SAMPLE)][:B]
    ids = np.zeros((B, T), dtype=np.int32)
    lens = []
    for i, t in enumerate(texts):
        e = tok.encode(t).ids[:T]
        ids[i, :len(e)] = e
        lens.append(len(e))
    print(f"[tok] {B} real NL samples, token lens min={min(lens)} max={max(lens)} "
          f"(T={T} window)", flush=True)

    def llama_forward(tok_ids: Tensor) -> Tensor:
        x = host.llama_embed[tok_ids]                      # (B, T, H)
        for layer in host.llama_layers:
            x = layer(x, host.llama_rope_cos, host.llama_rope_sin)
        return _rms_norm(x, host.llama_layers[-1].ffn_norm, cfg.rms_norm_eps)

    # ---- S2: eager forward on real text ----
    tok_t = Tensor(ids, dtype=dtypes.int).contiguous().realize()
    t0 = time.time()
    out = llama_forward(tok_t).realize()
    Device[Device.DEFAULT].synchronize()
    o = out.cast(dtypes.float).numpy()
    assert np.isfinite(o).all(), "NON-FINITE activations out of the frozen trunk"
    norms = np.linalg.norm(o, axis=-1)
    print(f"[fwd] eager L0-L3 on (B={B},T={T}) in {time.time()-t0:.1f}s; "
          f"per-token norm mean={norms.mean():.1f} p99={np.percentile(norms,99):.1f} "
          f"finite=True", flush=True)
    _mem("S2 llama eager fwd", table)

    # ---- S3: the AM-driver hazard check — TinyJit + assign-in-place replay ----
    fixed_in = Tensor(ids, dtype=dtypes.int).contiguous().realize()

    @TinyJit
    def jit_step() -> Tensor:
        return llama_forward(fixed_in).realize()

    times = []
    for rep in range(4):                    # rep0 compiles; 1-3 replay
        if rep > 0:
            fixed_in.assign(Tensor(np.roll(ids, rep, axis=0),
                                   dtype=dtypes.int).contiguous()).realize()
        t0 = time.time()
        r = jit_step()
        Device[Device.DEFAULT].synchronize()
        times.append(time.time() - t0)
        assert np.isfinite(r.cast(dtypes.float).numpy()).all()
    print(f"[jit] compile={times[0]:.1f}s replay={['%.3fs' % t for t in times[1:]]} "
          f"(assign-in-place fixed buffer; all finite)", flush=True)
    _mem("S3 llama JIT replay x3", table)

    # ---- S4/S5: co-resident deducer, known eval numbers ----
    from diag_kenken_granularity_probe import build_kenken_spec, build_kenken_deducer_model
    from capture_silhouette_trajectories import run_capture_forward
    from mycelium.kenken_data import KenKenLoader

    spec = build_kenken_spec(16)
    fg_model = build_kenken_deducer_model(spec, FG_CKPT, seed=0)
    Device[Device.DEFAULT].synchronize()
    _mem("S4 deducer loaded", table)

    loader = KenKenLoader(".cache/kenken_test_curriculum.jsonl", batch_size=8)
    kb = next(loader.iter_eval(8))
    reps, pred = run_capture_forward(fg_model, kb, spec, K=16, keep_rowcol=True)
    gold = kb.gold.numpy()
    valid = kb.cell_valid.numpy() > 0.5
    acc = float(((pred == gold) & valid).sum() / valid.sum())
    print(f"[deducer] co-resident eval batch cell_acc={acc:.3f} "
          f"(known-good ckpt range ~0.75-0.85)", flush=True)
    assert acc > 0.6, "deducer degraded while co-resident — investigate before building"
    _mem("S5 deducer eval fwd", table)

    # ---- the budget line ----
    total = table[-1][1]
    print("\n=== MEMORY BUDGET (tinygrad allocated) ===")
    prev = 0.0
    for tag, mb in table:
        print(f"  {tag:28s} {mb:8.1f}MB  (+{mb - prev:7.1f})")
        prev = mb
    headroom = 24_000 - total
    print(f"  {'HEADROOM vs 24GB':28s} {headroom:8.1f}MB  "
          f"(delta head + waist + hooks + notebook + Brick-A batch live here)")
    print("\n[PASS] residency smoke complete", flush=True)


if __name__ == "__main__":
    main()
