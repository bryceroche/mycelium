# Perceiver-Poincaré vs v98 KenKen: Per-Breath ~20x Gap — Root-Cause Memo

**Status:** STATIC ANALYSIS ONLY — GPU is busy (re-smoke + 5.5 h deduction run).
All confirmations and fixes are DEFERRED. Nothing in this document may be applied
to live training files until the GPU is free and the deduction verdict is banked.

---

## 0. The gap in one sentence

The perceiver trains at ~24.6 s/step (K=8) vs v98's ~2.7 s/step (K~20, 16–20
breaths). Normalised per breath: **perceiver ~3.1 s/breath, v98 ~0.14 s/breath
(~22x)**. Per-breath FLOPs are near-identical (2.33e9 vs 2.49e9; 46 latents vs
49 cells). This is NOT a FLOP gap — it is kernel-dispatch and memory-throughput
overhead. The four easy JIT wins (full step under one @TinyJit, deferred sync,
no cache-key thrash, constants hoisted, d_hyp once outside the loop) were
confirmed already present. What remains is a structural kernel-fusion pathology
with one dominant cause and one compounding secondary cause.

---

## 1. Ranked suspect table

| Rank | Suspect | Mechanism | Estimated speedup (this fix alone) | Confidence |
|------|---------|-----------|-----------------------------------|------------|
| 1 | **`_latent_self_attn_bias` called K=8 times inside the breath loop** (`perceiver_poincare.py:451`) with `Tensor(np.eye(...))` CPU→GPU upload and `.contiguous()` fusion-barrier at each breath — for a value that is **loop-invariant** (`latent_valid` never changes across breaths) | K forced JIT-graph materialisation barriers break cross-breath kernel fusion; K redundant np.eye host→GPU DMA uploads; `np.eye` Tensor construction adds a new constant node each call, preventing tinygrad from caching the subgraph; v98 pays this cost **0 times** (hoisted before the loop at `kenken.py:1121`) | **4–10x** of the per-step overhead (high confidence on mechanism; moderate on magnitude) | HIGH |
| 2 | **Full fp32 THINK** (`perceiver_poincare.py:452`: `h = latent_in.cast(dtypes.float)`) making every GEMM in the 4-layer THINK fp32×fp32, vs **v98's fp16 activation path** (`kenken.py:1141`: `x = x.cast(dtypes.half)`) which runs fp16×fp32 GEMMs | AMD 7900 XTX delivers ~1.5–2x higher throughput for fp16-input matmuls vs fp32-scalar FMAs on the AM driver (no ROCm rocBLAS packed-math offload). GEMM shapes are nearly identical: (B×46, 1024)@(1024,1024) vs v98's (B×49, 1024)@(1024,1024). fp32 doubles the weight-read bandwidth at the same arithmetic intensity. | **1.5–2x** (medium confidence; depends on AM driver matmul dispatch path for this shape) | MEDIUM |
| 3 | **Interaction: `.contiguous()` barriers inside the JIT prevent fusion of per-layer elementwise ops** (layernorm, residual-add, gate-mul) across the 4 layers per breath | tinygrad fuses elementwise ops into the adjacent matmul's epilogue; a `.contiguous()` fence materialises an intermediate buffer, forcing a separate dispatch for every following elementwise op. With K=8 barriers, each THINK's ~20 elementwise ops per layer emit as separate kernels rather than being fused. This is not independent of Rank-1 — it is the mechanism by which the Rank-1 barrier hurts beyond the DMA cost. | Subsumed in Rank-1 estimate (counted together) | HIGH (mechanism); merged with Rank-1 |
| 4 | **Structural READ+WRITE cross-attend overhead** (2 extra softmax+matmul passes over (B, L, 49) and (B, 49, L) per breath, absent in v98) | ~0.2% extra FLOPs per breath per the audit; on the AM driver at small B this adds ~2 extra kernel dispatches per breath for the matmul+softmax. | ~1.2x residual gap vs v98 **after** Ranks 1–2 are fixed (irreducible — architectural) | HIGH (structural; cannot be eliminated) |

---

## 2. Single most-likely explanation of the ~22x per-breath gap

**The dominant cause is the K=8 forced materialisation barriers from
`_latent_self_attn_bias` being called inside the breath loop.**

Confirmed by static analysis: `perceiver_poincare.py:451` calls
`_latent_self_attn_bias(latent_valid, cfg.n_heads)` on every iteration of
`for k in range(K)` (line 436). The function at lines 482–496 does two things
that are toxic on the AM driver inside a TinyJit-unrolled graph:

1. `Tensor(np.eye(L, dtype=np.float32), dtype=dtypes.float)` at line 488:
   allocates a fresh CPU-side numpy array and introduces a **new constant Tensor
   node** into the JIT graph every call. This makes the graph appear structurally
   different each breath to tinygrad's JIT cache, preventing subgraph reuse, and
   performs a host→GPU DMA upload of an (L, L) = (46×46) float32 matrix (~8.5 KB)
   K=8 times per step (68 KB of redundant DMA total, not counting VRAM
   allocation overhead).

2. `return bias.contiguous()` at line 496: forces immediate materialisation of
   the final (B=8, 16, 46, 46) = ~6.8 MB fp32 tensor inside the JIT-unrolled
   graph. On the AMD AM driver, `.contiguous()` inside a loop is the documented
   substrate-law anti-pattern (`memory/reference_tinygrad_am_quirks.md`). Each
   call breaks the kernel-fusion window for ALL ops before and after it in that
   breath's subgraph: the 4-layer THINK chain's ~20 elementwise ops per layer
   (layernorm scale/shift/eps, residual adds, gate multiplies, GELU activations)
   are emitted as separate kernels instead of being fused into the matmul
   epilogues. With K=8 breaths, this creates 8 hard schedule barriers in the
   combined forward+backward graph, fragmenting what should be one fused K-breath
   dispatch chain.

**The crucial asymmetry with v98:** `kenken.py:1121` calls
`build_kenken_attn_bias(...)` ONCE, before `for k in range(K)` at line 1171.
That function also ends in `.contiguous()` (line 622) — but it fires exactly
once, before the loop, so the JIT sees a single constant graph node shared
across all K breath subgraphs and can fuse the entire K-step THINK chain into
one dispatch group. The perceiver pays K barriers; v98 pays 1.

The secondary cause — full fp32 THINK vs v98's fp16 activation path — adds a
further ~1.5–2x factor on top. The two causes are partially multiplicative
(the fusion breakage means even the fp32 GEMMs are dispatched as smaller,
less-occupancy kernels). Together they explain the ~22x gap cleanly, with the
~1.2x irreducible architectural overhead (READ+WRITE cross-attends) accounting
for the residual after both fixes.

---

## 3. Ordered fix plan (root-cause first)

**GPU confirmation and all fixes are DEFERRED until the deduction run completes
and the GPU is free. The ordering below is the sequence in which to validate and
apply when available.**

### Fix A — Hoist `_latent_self_attn_bias` out of the K-breath loop (Rank 1, zero correctness risk)

**What:** Move `attn_bias = _latent_self_attn_bias(latent_valid, cfg.n_heads)`
from inside `for k in range(K):` (line 451) to just before the loop starts
(after line 429, after the `assert K <= K_max` guard). Use the pre-computed
tensor read-only inside the loop body.

**Why first:** `latent_valid` is assigned once at line 376 from `batch.latent_valid`
and is never written inside the loop body (verified: lines 436–477 write only to
`latent_hidden` and append to history lists). The hoisted result is byte-identical
to the per-breath computation every breath. This is the same pattern v98 already
uses.

**Additional hardening:** Replace `Tensor(np.eye(L, dtype=np.float32), ...)` at
line 488 with a module-level cache (compute once per unique `L`, store in a dict
keyed on `L`; or use `Tensor.eye(L, dtype=dtypes.float)` if tinygrad provides it
natively) to eliminate the numpy round-trip entirely. The `.contiguous()` at
line 496 is then paid once per step (not K times), and the JIT graph has a clean
constant node rather than K aliased-but-distinct nodes.

**Validation (when GPU is free):** Run `perceiver_train.py` for 20 steps before
and after, compare per-step wall-clock and assert `cell_logits_history` outputs
are bit-exact (same math, different scheduling). Expected speedup: 4–10x of the
per-step time.

**Correctness risk: ZERO.** The only failure mode is a future caller that
modifies `latent_valid` inside the breath loop; a comment at the hoisted call
site documenting this assumption is the guard.

### Fix B — Selective fp16 for THINK activations (Rank 2, low-medium risk, separate commit)

**What:** Remove `h = latent_in.cast(dtypes.float)` at line 452 (or replace it
with `h = latent_in` if `latent_in` is already the right dtype). Cast `latent_hidden`
and `cell_embed` to fp16 at init (mirroring `kenken.py:1141`: `x = x.cast(dtypes.half)`).
Keep `latent_hidden` accumulated in fp32 across breaths (the persistent state that
can grow over K steps); cast to fp16 only for the THINK forward pass, then cast
the output back to fp32 before the delta-gate update at line 456.

**The overflow issue (from the comment at lines 447–450):** The comment claims
fp16 overflow appeared in the residual at breath 11. The overflow is in
`latent_hidden` accumulation across K breaths (the persistent state), NOT in the
QKV projections (which are bounded by LayerNorm). The fix is therefore: keep
`latent_hidden` in fp32 (persistent state, can be large), cast only for the
THINK activations. Substrate-legal clamp at the cast site:
`h = latent_in.clip(-60000.0, 60000.0).cast(dtypes.half)` provides a hard
fp16-range guard with negligible forward-math effect.

**Validation:** After Fix A is confirmed working, apply Fix B separately. Run 40
steps and log `latent_hidden.abs().max()` per breath outside the JIT step. If
magnitude stays bounded (< 50000) across all breaths at K=8, the clip guard is
sufficient. Loss curve should track within noise vs the fp32 path. Parity is NOT
bit-exact (fp16 rounding differs), so validate by loss trajectory, not
bit-identity.

**Correctness risk: LOW-MEDIUM.** The clip-cast guard may be insufficient in
long trained runs if the residual norm grows unexpectedly. Apply only after Fix A
is validated. Keep the fp32 fallback path available (e.g. a flag
`PERCEIVER_FP16_THINK`) for easy revert.

### Fix A-sub — Cache or replace `np.eye` construction in `_latent_self_attn_bias`

This is a hardening sub-step of Fix A, not a separate fix. If `Tensor.eye(L,
dtype=dtypes.float)` is available in the current tinygrad version, use it to
avoid the numpy round-trip entirely. If not, a module-level `_EYE_CACHE: dict[int, Tensor] = {}` with `_EYE_CACHE.setdefault(L, Tensor(np.eye(L, ...), ...).realize())` gives the same result. The `.realize()` at cache-population time ensures the constant is committed to VRAM once and reused as a stable graph node.

---

## 4. GPU confirmation is deferred

**Every profiling run, smoke test, and fix validation listed above requires GPU
time.** The GPU is currently occupied: a re-smoke is running now, and a 5.5-hour
KenKen deduction run follows. No GPU job may be started for this investigation
until that deduction run completes and the Property-2 verdict is banked.

The fixes identified here are the result of static code analysis only. The
estimated speedups are analytical projections based on:
- Known tinygrad AM driver behaviour (`.contiguous()` inside a JIT loop breaks
  kernel fusion — documented in `memory/reference_tinygrad_am_quirks.md`).
- Direct structural comparison of the perceiver loop body vs the v98 kenken loop
  body (line-level code audit of `perceiver_poincare.py:436–496` and
  `kenken.py:1115–1194`).
- AMD 7900 XTX fp16 vs fp32 throughput characteristics under the AM driver.

The magnitude of speedup (4–10x for Fix A, 1.5–2x for Fix B) has not been
empirically confirmed. Confirmation is step 1 when the GPU is available.

---

## 5. Cumulative speedup estimate

| After fix | Projected step time | Projected per-breath time | Speedup vs baseline |
|-----------|--------------------|--------------------------|--------------------|
| Baseline (current) | ~24.6 s/step | ~3.1 s/breath (K=8) | 1x |
| After Fix A (hoist attn_bias) | ~3–7 s/step | ~0.4–0.9 s/breath | **3.5–8x** |
| After Fix A + Fix B (fp16 THINK) | ~2–4 s/step | ~0.25–0.5 s/breath | **6–12x** |
| Irreducible floor (READ+WRITE cross-attends, structural) | ~1.2x above v98 | ~0.17 s/breath | ~14x total max |
| v98 KenKen baseline | ~2.7 s/step | ~0.14 s/breath | — |

**Best-case projection:** Fix A + Fix B bring the perceiver from ~24.6 s/step to
roughly **2–4 s/step**, closing the 22x gap to **1.5–3x above v98**, where the
remaining overhead is structurally justified by the two extra cross-attend passes
(READ and WRITE) per breath that v98 does not have.

**Honest uncertainty:** The Fix A multiplier is the most uncertain quantity. The
`.contiguous()` fusion-breakage penalty on the AM driver has been observed to
range from 2x to 10x depending on the mix of fused vs unfused ops in the
surrounding graph. The 4–10x range is wide for this reason. Fix A must be
empirically confirmed before deciding whether Fix B is necessary.

**Overlap between suspects:** Rank-1 (hoist) and Rank-3 (fusion breakage from
`.contiguous()`) are NOT independent — Rank-3 IS the mechanism by which Rank-1
hurts beyond the DMA cost. They are listed separately in the table for clarity
but counted together in the speedup estimate. Rank-2 (fp32 THINK) is independent
of Rank-1 and stacks multiplicatively.

---

## 6. Code locations (verified by static read)

| Finding | File | Line(s) |
|---------|------|---------|
| `_latent_self_attn_bias` called inside K-loop | `mycelium/perceiver_poincare.py` | 451 (inside loop at 436) |
| `Tensor(np.eye(...))` — CPU alloc + DMA each call | `mycelium/perceiver_poincare.py` | 488 |
| `.contiguous()` fusion barrier inside JIT | `mycelium/perceiver_poincare.py` | 496 |
| `latent_valid` assigned, never modified in loop | `mycelium/perceiver_poincare.py` | 376, 436–477 |
| Full fp32 cast for THINK activations | `mycelium/perceiver_poincare.py` | 452 |
| fp16 overflow justification comment | `mycelium/perceiver_poincare.py` | 447–450 |
| v98 attn_bias hoisted BEFORE loop | `mycelium/kenken.py` | 1121 (loop starts 1171) |
| v98 fp16 activation cast | `mycelium/kenken.py` | 1141 |
| v98 `build_kenken_attn_bias` `.contiguous()` (paid once) | `mycelium/kenken.py` | 622 |
