# Performance audit — tinygrad usage (2026-09-11, static; CPU-only measurements)

Measured on CPU (DEV=CPU, the family env, B=8): the fused `step()` graph is 2,465 kernels
(fwd+bwd+AdamW; 70 in-graph host->device copies); the masked 7-breath read pass 653 kernels
(~100/breath); the open pass 40. At 0.39-0.51 s/step that is ~180 us per kernel; analytic
FLOPs ~0.2-0.3 TFLOP/step (~0.5% of f32 peak): the step is latency/small-kernel bound.
Suspected recomputes (waist k/v per breath; the ladder's shared loss terms) are NON-findings:
tinygrad hash-conses identical UOps.

## Ranked findings (expected wall-clock gain first)

1. **Fused trainer at BATCH=32 (~2x per row).** step() phase1_algebra_head.py:4857-4955, buffers
   :4726-4744. Per-kernel fixed cost dominates; the step trainer does 0.024 s/item at B=32 with
   MORE kernels and syncs, so <=0.024 s/item is a floor-ish prediction for the fused graph. Memory
   ~4-5 GB activations at B=32 + 67 MB trunk input + ~280 MB params/Adam: fits. VERIFY: STEPS=5 at
   BATCH=16 then 32 with perf_counter around step(). GATE: a training-REGIME change (rows per
   update) — a registered twin under the dose law, not an eq gate. Fix #4/#5 first or the host
   feed pays ~150 ms/step.
2. **The 30-40 min capture-family compile = cross-process compile-cache MISSES from unstable
   kernel names.** tinygrad codegen/opt/postrange.py:50-60 appends a per-process counter `n{k}`
   to kernel names; device.py:264-268 keys the disk cache on the source, which embeds the name.
   Cache census: 214,049 AMD entries (1.5 GB); 45% carry a suffix; 23% (49,211) are byte-identical
   to another entry except the suffix. cache.db is 13 GB holding 1.6 GB (never VACUUMed). Warm
   codegen ~35 ms/kernel vs cold LLVM compile; the walker family ~5k kernels -> ~3-5 min warm.
   VERIFY: count(*) on the compile table before/after one walker start = cold compiles; patch the
   suffix to an AST hash (or strip the name from the key) and re-measure. GATE: eq_check A/B/C
   (renamed kernels, same code) — zero numeric risk.
3. **The facts/mask-prep pass: pool the solver, pread the memmap, overlap (42 -> ~10 min).**
   :4606-4680; _alt2_fact_buf_v1 :1120-1160 -> alternator_bridge.ping -> csp_core.gac_propagate.
   Rows independent (~9 ms/row python). The "forward half" is largely host I/O:
   states[sl_p].astype(np.float32) costs 68 ms/batch memmap read + 76 ms f16->f32 at B=32
   (pread: 38 ms). VERIFY: perf_counter on alt2_fact_buf and the read/astype per batch. GATE:
   FACTS/MASKS asserted bitwise vs the cache (_maskprep_finish); pooling keeps row order.
4. **Feed trunk states as fp16, cast on device (bit-identical).** :5259 (and :4638, :3990,
   :4102, loop_val.py:107, jit_read._feed). numpy f16->f32 ~2 ms/row; the PCIe copy halves.
   ~4% at B=8, ~15% at B=32. GATE: exact cast; eq_check covers.
5. **~30 separate .realize() dispatches per step in the feed** (:5259-5293; the gold loop
   :5327-5329 one per key). ~10-30 ms/step; 4x the bytes at B=32. GATE: data movement only.
6. **BEAM/JITBEAM never used** (no chain sets BEAM; beam table 856 rows). The 08-22 audit found
   kernels at 5-8% of peak. VERIFY: PROFILE=1 STEPS=3 per-kernel timings; if a few dominate,
   JITBEAM=2 IGNORE_JIT_FIRST_BEAM=1 (one-time hours, disk-cached). GATE: reduction order
   changes -> a tolerance ruling (the ST_EQ_TOL precedent), not bitwise.
7. **Hoist per-breath host constants out of the JIT'd graph** (:3152-3155, :2723-2726,
   :2952-2956, _clock_frame :109-117, NB_STAMPS :2431, :2505-2507): 60-70 NPY->device copies per
   step as graph nodes (no sync; <2%). GATE: bit-identical (the _polar_sink/_SGC idiom).
8. **T_ALG=256 padding tax** (:45; bank :2170-2224; mask broadcast :2185). form12 mean 94.5
   tokens (p50 72, 28% > 128); wild max 127; test23 max 226. Wild reads at T=128 ~1.5x; training
   would need length buckets (a regime change) — defer. GATE: tolerance (reduction trees change).
9. **_quick_val bypasses jit_read** (:4963-5023 eager; ~50 s x 3 per 12k run ~3%). The two-pass
   read is +6%, not 2x (the open pass never runs the loop). GATE: jit_read is byte-identical.
10. **Memmap pattern**: random rows are 1 MB contiguous; at B=8 ~2 ms; the convert (#4)
    dominates; only the sequential prep benefits from pread (#3).

Non-findings: Tensor.training=True has no effect (no dropout/BN); Tensor.stack/.contiguous()
sites are not graph splitters; .realize().numpy() inside forward are all behind _CENSUS.
