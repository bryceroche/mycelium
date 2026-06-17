# Perceiver FP16-THINK Patch — Design Document

**Status:** PROPOSED — APPLY ONLY AFTER the powered Property-2 verdict is banked
and the GPU is free. Do NOT apply to live training files now.

**Prerequisite:** Fix A (hoist `_latent_self_attn_bias` before the K loop, which
accounts for ~4–10x of the ~22x gap) should be applied and smoke-validated
FIRST (see `docs/perceiver_perf_rootcause.md §3 Fix A`). Fix B in this document
(fp16 THINK) is the second, independent change — applied after Fix A is confirmed.

---

## 1. Root cause of the late-breath fp16 overflow

### Which op overflows, and why magnitudes grow

The overflow is in the **latent residual accumulation**, not in the QKV
projections. Each breath's THINK runs 4 transformer layers whose residual stream
is:

```
layer output:   x = x + attn_out + ffn_out
```

LayerNorm normalises the *input* to each attention block and FFN, but the
**residual bypass** is unconstrained: it carries the running sum of all
previous `attn_out + ffn_out` contributions. After 4 layers per breath and K
breaths of delta-gate blending, the persistent `latent_hidden` state
accumulates across the outer loop:

```python
latent_hidden = latent_hidden + gate_k * (h - latent_hidden)   # line 456
```

With `gate_k` approaching 1 early in training (the gate initialises to 1.0 at
`attach_perceiver_params`), each breath drives `latent_hidden` toward `h`,
which is the full output of 4 layers. The per-layer LayerNorm keeps the *pre-norm
residual* bounded at each layer input, but the residual stream itself has no
global magnitude bound. At K=11 (the comment at lines 447–450 gives the
observed failure point), 44 residual additions in fp16 push `max|latent_hidden|`
above the fp16 ceiling of 65504, producing NaN on the next softmax (`exp(large_score)`
in `_latent_layer_forward:L520`).

The current workaround is the full-fp32 cast at line 452:

```python
h = latent_in.cast(dtypes.float)    # fp32 to avoid overflow — the performance cost
```

This eliminates overflow at the cost of running all 4 Pythia layers (wq, wk,
wv, wo, w_in, w_out — 6 GEMMs per layer, 24 per breath) in fp32. AMD packed
fp16 GEMM throughput is ~2x that of fp32 on the 7900 XTX under the AM driver,
so each breath that should take 0.14 s instead takes ~0.28–0.56 s from the
dtype change alone (before accounting for the cross-breath fusion breakage from
`_latent_self_attn_bias` called K times inside the loop — see Fix A).

### Why a per-breath clip is the correct and sufficient fix

The overflow is a *tail event*: in normal early training `max|latent_hidden|` is
O(10)–O(100), well within fp16 range. It only exceeds fp16 max in:
(a) late training when norms have grown, or
(b) at large K if learning rate is high.

A **per-breath `latent_hidden.clip(−CLIP_VAL, CLIP_VAL)` before the THINK
cast** handles both cases cleanly. With `CLIP_VAL = 1000` (25x below fp16 max,
100x above typical early-training magnitudes), the clip fires only when overflow
is imminent. When it fires, it caps the residual before the fp16 cast, so the
softmax inputs are bounded. When it does not fire (the common case), the
computation is **bit-identical to the unclipped path** (clip of a value within
bounds is a no-op). Gradient flows through unclipped (≡ identity) positions and
is zero only where the residual was being clipped — which means the gradient was
already in an unstable region.

No new learnable parameters are needed. The clip values are Python float scalars,
which are legal inside the JIT step (they bake into the graph as compile-time
constants, not `dtypes.float32` Tensor literals — the substrate law is
specifically about `Tensor(value, dtype=dtypes.float32)` inside JIT, not Python
scalars passed to `.clip()`).

---

## 2. Proposed fix — exact unified diff

The patch touches two regions of `mycelium/perceiver_poincare.py`:

**Region A:** the env-gate block (add `PERCEIVER_FP16_THINK` flag, lines 67–96).

**Region B:** the breath loop body (the THINK section, lines 443–457 and the
new hoisted `attn_bias` line before the loop at line 436).

```diff
--- a/mycelium/perceiver_poincare.py
+++ b/mycelium/perceiver_poincare.py
@@ -91,6 +91,12 @@ PERCEIVER_LATENT_INIT = float(os.environ.get("PERCEIVER_LATENT_INIT", "0.02"))
 # Substrate clamp constants (REUSE the exact Tier-2 values).
 PERCEIVER_NORM_CLAMP = KENKEN_HYP_NORM_CLAMP
 PERCEIVER_ARG_MIN = KENKEN_HYP_ARG_MIN
 PERCEIVER_BLOCK = 1e4
+# FP16-THINK gate. ON (default OFF): run the 4-layer Pythia THINK in fp16
+# activations (matching the v98 kenken fp16 path) with a per-breath residual
+# clip guard to prevent late-breath overflow. Set PERCEIVER_FP16_THINK=1 after
+# Fix A (attn_bias hoist) is smoke-validated. fp32 stays the fallback (=0).
+PERCEIVER_FP16_THINK = int(os.environ.get("PERCEIVER_FP16_THINK", "0")) > 0
+# Clip threshold for the fp16 THINK residual guard (Python float scalar — legal
+# inside JIT). Must be well below fp16 max (65504) and well above normal training
+# range (~O(10–100)). 1000.0 gives 65x headroom below fp16 max.
+PERCEIVER_THINK_CLIP = float(os.environ.get("PERCEIVER_THINK_CLIP", "1000.0"))


@@ -425,7 +431,10 @@ def perceiver_breathing_forward(model: Any, batch: Any, K: int,
     K_max = int(breath_embed.shape[0])
     assert K <= K_max, f"K={K} exceeds K_max={K_max}"

+    # Hoist attn_bias BEFORE the K-breath loop — latent_valid is loop-invariant
+    # (never modified inside the loop body). This is the same pattern v98 uses
+    # (build_kenken_attn_bias hoisted before the loop at kenken.py:1121).
+    # Paying this .contiguous() barrier K=1 times vs K=8 times eliminates
+    # K-1 forced materialisation barriers that break cross-breath kernel fusion.
+    attn_bias = _latent_self_attn_bias(latent_valid, cfg.n_heads)  # (B,n_heads,L,L)
+
     cell_logits_history = []
     engagement_history = []

@@ -443,18 +454,28 @@ def perceiver_breathing_forward(model: Any, batch: Any, K: int,

         # === THINK: latent self-attention through the shared Pythia L0-L3. ===
         # Full self-attn over latents (a latent attends to all VALID latents). The
-        # mask is per-batch (latent validity). delta_gate-style residual: the THINK
-        # output BLENDS with the prior latent state (perceiver residual). The latent
-        # field is SMALL (L~59 vs 49 cells), so the THINK runs in FP32 — fp16 carry
-        # overflowed (max ~65504) on the LATE breaths (the residual stream grows
-        # across the 4-layer stack + K breaths -> breath-11 NaN in the fp16 path).
-        # fp32 here is cheap (small L) and eliminates the overflow at the source.
-        attn_bias = _latent_self_attn_bias(latent_valid, cfg.n_heads)  # (B,n_heads,L,L)
-        h = latent_in.cast(dtypes.float)
+        # mask is per-batch (latent validity; hoisted before loop — loop-invariant).
+        # delta_gate-style residual: the THINK output BLENDS with the prior latent
+        # state (perceiver residual). PERCEIVER_FP16_THINK=1 runs the THINK in fp16
+        # activations (2x AMD throughput) with a per-breath clip guard to prevent
+        # late-breath overflow. PERCEIVER_FP16_THINK=0 (default) keeps the fp32
+        # path that was the safe fallback during initial development.
+        if PERCEIVER_FP16_THINK:
+            # Per-breath residual renorm: clip latent_in to [-CLIP, CLIP] before the
+            # fp16 cast. This prevents fp16 overflow when the residual stream grows
+            # across K breaths. PERCEIVER_THINK_CLIP is a Python float scalar (legal
+            # inside JIT; bakes as a compile-time constant, NOT a float32 Tensor
+            # literal). The clip is a no-op in the common case (|latent_in| << 1000).
+            clip_val = PERCEIVER_THINK_CLIP   # Python float — substrate-legal in JIT
+            h = latent_in.clip(-clip_val, clip_val).cast(dtypes.half)
+        else:
+            h = latent_in.cast(dtypes.float)
         for layer in layers[:4]:
             h = _latent_layer_forward(layer, h, attn_bias)
-        gate_k = model.perc_delta_gate[k].cast(dtypes.float).reshape(1, 1, 1)
-        latent_hidden = latent_hidden + gate_k * (h - latent_hidden)  # (B,L,H)
+        gate_k = model.perc_delta_gate[k].cast(h.dtype).reshape(1, 1, 1)
+        latent_hidden = latent_hidden.cast(h.dtype) + gate_k * (h - latent_hidden.cast(h.dtype))
+        # Accumulate persistent state in fp32 to prevent cross-breath overflow.
+        # The fp16 path casts to half only for the THINK forward pass; latent_hidden
+        # is kept in fp32 across breaths (the outer residual). Cast back after gate.
+        latent_hidden = latent_hidden.cast(dtypes.float)
         latent_hidden = latent_hidden * latent_valid.reshape(Bn, L, 1)
```

### Substrate-legality checklist for the diff

| Rule | Status in this diff |
|------|---------------------|
| No `Tensor(value, dtype=dtypes.float32)` literal inside JIT | PASS — `PERCEIVER_THINK_CLIP` is a Python float scalar; `.clip(-clip_val, clip_val)` uses Python float args |
| `where()`-gated NaN guard (not multiply-gate) | UNAFFECTED — existing NaN guards not changed |
| `scores.clip(-1e4, 1e4)` before softmax | UNAFFECTED — `_latent_layer_forward:L520` already has this |
| No `.contiguous()` inside the breath loop on the hot path | ACHIEVED — `_latent_self_attn_bias` and its `.contiguous()` are now before the loop; the loop body has no new `.contiguous()` calls |
| Single-kernel `isfinite` for NaN skip (trainer-side) | UNAFFECTED — `perceiver_train.py` NaN guard is unchanged |
| `arccosh` arg floor + `|z|^2` clamp | UNAFFECTED — `_d_hyp_cross` unchanged |

### Note on `_latent_layer_forward` with fp16 input

`_latent_layer_forward` at lines 499–526 already handles dtype-agnostic input via:
```python
attn_in_dt = attn_in.cast(x.dtype) if attn_in.dtype != x.dtype else attn_in
mlp_in_dt = mlp_in.cast(x.dtype) if mlp_in.dtype != x.dtype else mlp_in
```
With `x.dtype = dtypes.half`, `attn_in_dt` and `mlp_in_dt` will be fp16,
so the 6 GEMMs per layer (wq, wk, wv, wo, w_in, w_out) will be fp16 input.
The weight tensors are fp32 (from `cast_layers_fp32`), so tinygrad dispatches
fp16×fp32 GEMMs which accumulate in fp32 internally (the standard AMD WMMA
mixed-precision path) — giving the throughput benefit of fp16 input addressing
without sacrificing accumulation precision.

The `attn_bias.cast(scores.dtype)` at line 519 will cast the fp32 attn_bias to
fp16 for the score addition. This is safe because the bias values are in `{0, -1e4}`,
both representable exactly in fp16.

---

## 3. Parity check plan

Run this plan ONLY after Fix A is landed and confirmed. All smokes are short
(≤50 steps) to fit in the GPU gap between other runs.

### Smoke A — Fix A alone (attn_bias hoist, no fp16 change)

Verify Fix A before adding Fix B.

```bash
PERCEIVER_TASK=1 PERCEIVER_K_MAX=8 PERCEIVER_FP16_THINK=0 BATCH=8 STEPS=20 \
  python scripts/perceiver_train.py
```

**Pass criteria:**
- Per-step wall-clock is ≤7 s (vs 24.6 s baseline; expect 4–10x speedup).
- `cell_loss` at step 5 and step 20 matches the fp32 baseline within 1e-4
  (Fix A is bit-exact; only scheduling changes).
- No NaN/Inf in `total` or `grad_norm`.
- Engagement metrics (`read_max`, `write_max`) are non-zero and stable.

### Smoke B1 — fp16 THINK at K=8 (short run)

```bash
PERCEIVER_TASK=1 PERCEIVER_K_MAX=8 PERCEIVER_FP16_THINK=1 \
  PERCEIVER_THINK_CLIP=1000.0 BATCH=8 STEPS=50 \
  python scripts/perceiver_train.py
```

**Pass criteria (parity with fp32 path):**
1. **Loss trajectory**: `cell_loss` at steps 1, 10, 25, 50 matches the fp32
   baseline within ±5% relative. fp16 rounding will cause small deviations; a
   diverging loss curve is a failure signal, not a ±5% deviation at step 50.
2. **Per-breath CE ladder**: The per-breath CE values (`per_breath_ce_0` through
   `per_breath_ce_7`) should show a descending ladder (later breaths lower loss),
   matching the shape of the fp32 ladder within ±10%. If the ladder inverts
   (later breaths worse), fp16 rounding is disrupting deduction.
3. **max_z probe**: `max_z` (max ball norm of cell coordinates) should remain
   below 0.95 at all steps (rim-guard headroom). If `max_z` spikes, the Poincaré
   ops are encountering fp16 precision loss in the distance computation.
4. **select_norm (engagement)**: `read_select_norm` and `write_select_norm`
   should track the fp32 run within ±15%. A large divergence means the
   routing geometry is disrupted by fp16 rounding in the cross-attend weights.
5. **NaN-free at the deepest breath (breath K-1=7)**: `total.isfinite()` must
   be True at every step. The clip guard is specifically validated by the absence
   of NaN at K-1 — if overflow reaches the softmax, it appears here first.
6. **Per-step wall-clock**: ≤5 s/step (vs 24.6 s baseline; Fix A + Fix B
   together target ~2–4 s/step).

### Smoke B2 — fp16 THINK at K=20 (the eventual operating point)

Run at K=20 to confirm the clip guard holds at the full depth (K=11 was the
observed overflow point in the fp32-less path, so K=20 is the stress test):

```bash
PERCEIVER_TASK=1 PERCEIVER_K_MAX=20 PERCEIVER_FP16_THINK=1 \
  PERCEIVER_THINK_CLIP=1000.0 BATCH=8 STEPS=30 \
  python scripts/perceiver_train.py
```

**Additional pass criterion beyond Smoke B1:**
- `total.isfinite()` True at every step for K=20.
- `cell_loss` at step 30 does not diverge vs the fp32 K=20 baseline.
- If NaN appears at any breath, lower `PERCEIVER_THINK_CLIP` to 500.0 and rerun
  (the clip threshold may need tightening for long-K runs).

### Smoke B3 — clip guard sensitivity sweep (optional, if B2 raises concern)

```bash
for CLIP in 500.0 1000.0 2000.0; do
  PERCEIVER_TASK=1 PERCEIVER_K_MAX=20 PERCEIVER_FP16_THINK=1 \
    PERCEIVER_THINK_CLIP=$CLIP BATCH=8 STEPS=20 \
    python scripts/perceiver_train.py 2>&1 | grep "step=20"
done
```

Compare `cell_loss` at step 20 across thresholds. A flat curve confirms the clip
is not firing in normal training (all three should match). A rising loss at lower
`CLIP` values indicates the residual is genuinely large, requiring fp32.

### What to log for each smoke

Add a temporary per-step `latent_hidden.abs().max().item()` call **outside** the
JIT step (after the `.numpy()` call on the returned scalars, not inside `_step`)
to monitor the pre-clip residual magnitude. This does NOT go inside the JIT. It
is added to the smoke runner only:

```python
# In the smoke loop (OUTSIDE _step), after _step returns:
# log max |latent_hidden| to confirm clip is not firing in normal ops.
# (This call is outside JIT and therefore costs one CPU-GPU sync per step --
#  acceptable for a 50-step smoke, NOT acceptable for production training.)
```

---

## 4. Expected speedup and rollback

### Expected speedup

| Configuration | Projected step time | vs baseline |
|---------------|--------------------|-----------:|
| Baseline (current, K=8) | ~24.6 s/step | 1x |
| Fix A only (attn_bias hoist) | ~3–7 s/step | 3.5–8x |
| Fix A + Fix B (fp16 THINK, this patch) | ~2–4 s/step | 6–12x |
| v98 KenKen reference (K~20) | ~2.7 s/step | — |

The projected **6–12x total speedup** brings the perceiver from ~24.6 s/step to
roughly **2–4 s/step**, close to the v98 KenKen reference. The remaining 1.5–3x
overhead above v98 is structurally justified by the two extra cross-attend passes
(READ and WRITE over (B,L,49) and (B,49,L)) that v98 does not have. These are
architectural, not performance bugs.

**Uncertainty:** the Fix A multiplier is the least certain. If the `.contiguous()`
barrier on the AM driver is less damaging than estimated (e.g. the tinygrad JIT
version in this repo defers materialisation differently than assumed), Fix A alone
may yield only 2–4x. In that case Fix B (this patch) becomes relatively more
important. The smokes will calibrate this.

### Rollback

The `PERCEIVER_FP16_THINK` env gate (default `0`) is the rollback. If any smoke
fails the parity check:
1. Set `PERCEIVER_FP16_THINK=0` (or leave unset) — the fp32 path is selected and
   the code is byte-identical to the pre-patch state.
2. The `PERCEIVER_THINK_CLIP` constant and the `if PERCEIVER_FP16_THINK:` branch
   add negligible overhead to the fp32 path (one Python-level branch at trace
   time; the `if` is evaluated at JIT compilation, not per step).

The fp32 THINK path is the permanent fallback — it is never removed.

---

## 5. Application sequencing (locked)

1. **Bank the powered Property-2 verdict** (live K=16 KenKen retrain, Tier 3).
   This is the gate for ALL perceiver work (§8 item 1 in CLAUDE.md).
2. **Apply Fix A** (hoist `_latent_self_attn_bias`; `perceiver_perf_rootcause.md §3
   Fix A`). Smoke-validate: per-step time drops by ≥3x, loss matches fp32 baseline.
3. **Apply this patch** (fp16 THINK + residual clip). Smoke-validate Smokes B1 and
   B2 above. Only proceed to production training if B2 passes at K=20.
4. **If Smoke B2 fails**: set `PERCEIVER_THINK_CLIP=500.0` and retry (Smoke B3).
   If loss diverges at any clip value, keep `PERCEIVER_FP16_THINK=0` and accept
   the ~3.5–8x speedup from Fix A alone.

This patch is **additive and gated**. Fix A is independent (zero correctness
risk, same logic as the v98 attn_bias hoist). Fix B requires validation but is
fully reversible.
