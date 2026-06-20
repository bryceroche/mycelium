# Deep-circuit scaling — reproduced results (amortized constraint propagation, 4 hops/breath)

**Reproduced 2026-06-20** on a fresh run (the integrity capstone for the
amortized-constraint-propagation headline of `docs/general_engine_results.md` §4).

- **Checkpoint:** `.cache/fg_ckpts/fg_circuit_deep_k16/fg_circuit_deep_k16_step1200.safetensors`
  (deep-mix D4–D16 training, stopped at step 1200 on eval plateau). *Checkpoints are not
  committed to git; this file records the reproduced numbers + the exact command so the
  claim is reproducible from the on-disk checkpoint.*
- **Command:**
  ```
  DEV=AMD FG_CKPT=.cache/fg_ckpts/fg_circuit_deep_k16/fg_circuit_deep_k16_step1200.safetensors \
  FG_TASK=circuit FG_N_INSTANCES=8000 \
    .venv/bin/python3 scripts/eval_circuit_scaling.py \
    --bands D6,D8,D10,D12,D14,D16 --k-sweep 4,8,12,16 --n-eval 400
  ```

## (A) Per-D accuracy (K = K_train = 16), n=400/band

| band | cell_acc | puzzle_acc |
|---|---|---|
| D6 | 0.959 | 0.682 |
| D8 | 0.964 | 0.675 |
| D10 | 0.948 | 0.547 |
| D12 | 0.951 | 0.547 |
| D14 | 0.939 | 0.448 |
| **D16** | **0.923** | 0.282 |
| OVERALL | 0.944 | 0.530 |

Flat cell_acc to D16 (gentle ~3.6 pt decline over a 10-level range) → **no depth ceiling**.
(puzzle_acc falls with depth — more gates that must *all* be right.)

## (B) K-sweep — cell_acc[D][K′] (the decisive breath-budget diagnostic: are the breath's 4 layers fully utilized?)

| band | K′=4 | K′=8 | K′=12 | K′=16 | ratio K′=4/K′=16 |
|---|---|---|---|---|---|
| D6 | 0.972 | 0.964 | 0.961 | 0.959 | 1.01 · 4 HOPS/BREATH |
| D8 | 0.974 | 0.969 | 0.966 | 0.964 | 1.01 · 4 HOPS/BREATH |
| D10 | 0.942 | 0.949 | 0.949 | 0.948 | 0.99 · 4 HOPS/BREATH |
| D12 | 0.931 | 0.949 | 0.953 | 0.951 | 0.98 · 4 HOPS/BREATH |
| D14 | 0.908 | 0.931 | 0.935 | 0.939 | 0.97 · 4 HOPS/BREATH |
| **D16** | **0.876** | 0.903 | 0.916 | **0.923** | **0.95 · 4 HOPS/BREATH** |

**Read:** `acc(K′=4, D=16) = 0.876 ≈ acc(K′=16, D=16) = 0.923` → **four breaths recover 95%
of full-K performance at depth 16**. All bands ratio ≥ 0.95 → the breath's 4 layers are
FULLY UTILIZED for propagation (the deducer resolves ~1 deduction level per transformer
layer, so a breath = 4 layers ≈ ~4 attention hops advances the front ~4 levels/breath ON
AVERAGE), so K_min ≈ D/4 — **LINEAR in depth** (a 4× constant-factor amortization, since
each breath's 4 layers all do useful propagation), NOT sub-linear. The depth axis is
SEQUENTIAL (~1 level/layer); the parallelism is BREADTH (all nodes at a given level
resolve together in one attention pass). The engine does NOT beat the fundamental
sequential-depth bound (depth-D still needs ≥ D propagation steps). The residual ~5% from
extra breaths is iterative refinement (the loopy structure), so it is "~4 levels/breath on
average", not a clean in-order wavefront.

## Reproduction note

These numbers match the original recorded run (D16: K′=4 0.877→0.876, K′=16 0.923→0.923,
ratio 0.95; per-D within noise) — a fresh re-run confirms the headline rather than relying
on a memory-note record. This is a property of the **learned** engine relative to a
naive 1-level-per-breath schedule (the breath's 4 layers all do useful propagation, a 4×
constant-factor amortization); it is **not** a beat of the fundamental sequential-depth
bound, and **not** a speed crown over a bespoke symbolic circuit evaluator (which is
trivially fast and exact). The owned claim is *general + amortized constraint propagation
(4 hops/breath, breadth-parallel, layer-sequential at ~1 level/layer) + learned*, honestly
bounded.
