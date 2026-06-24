# Session memo — 2026-06-23: KenKen frontier mapped → pivot to soft constraints

**Authors:** Bryce + Claude · **Branch:** `gen-weights` (commits `cfe7ed1`, `c8cc8de`)

## TL;DR
We thoroughly mapped the KenKen frontier for the v98-lineage deducer. **Fixed** the
generalization gap, **found** the one working lever (a *completed* soft-violation energy
that raises base per-instance validity), and **cleanly refuted** three appealing
directions (generate-and-verify volume, an expand-collapse multi-resolution waist, and a
spectral regional decomposition as a discovery mechanism). The coherent conclusion — which
matches the project's own deepest finding — is that **KenKen is a clean verifiable CSP where
symbolic search dominates**, so further KenKen polishing has bounded upside. Next: pivot to
the **soft / probabilistic / learned-constraint frontier** (CLAUDE.md §8.1) where symbolic
methods are unavailable and the deducer's "learned BP" nature can actually *beat* symbolic.

## What we FIXED — generalization (the unlock)
The first FG-KenKen deducer overfit hard: train cell-acc ~0.98 vs **test ~0.50** on a
1,020-puzzle set. Diagnosis (workflow): **pure overfit**, not distribution shift (train/test
matched field-for-field, leak-free, test not harder). Fix = the 39,996-puzzle **curriculum
corpus** (already emitted, unused) + the **v45 reg stack** (`WEIGHT_DECAY=0.05`,
`LABEL_SMOOTHING=0.1`, `STOCH_DEPTH_P=0.10`), one clean intervention vs the overfit base.
Result: test cell **0.50→0.80**, puzzle **0.013→0.35**, gap gone. Per-band: easy bands
near-saturated (g40 0.65 puzzle), hard band the frontier (g10 0.06). Best ckpt:
`.cache/fg_ckpts/fg_kenken_k16_reg/`. (memory: `project_kenken_generalization_fixed`)

## What WORKS — the energy is the base-p lever (modest, general)
Generate-and-verify **volume collapses** on KenKen (g10 best-of-64 = 0.0625 vs argmax
0.0563; 93.8% hard-core; M_eff_ratio = inf). KenKen is rigid/unique-solution → cell-perm
darts are solved-by-all-views or none. So **base-p, not volume, is the lever.**
(memory: `project_kenken_volume_collapse`)

The **soft-violation energy**, *completed* with a differentiable **cage-arithmetic** term
(the prior row/col-only energy was the easy half) and normalized to a bounded-relative O(1)
scale, **transfers** on the generalizing base: g10 puzzle **0.059→0.107 (~1.8×)** at tight
n≈2,140. A **per-breath wave** modestly beats final-breath-only (~1.7 SE), but the wave
**shape is irrelevant — monotonic == oscillating (0.107=0.107)**. The earlier
energy-transfer *null* flipped to positive *because* generalizing base + completed energy.
The general principle stands: **minimize a differentiable relaxation of the verifier in the
loss** to raise validity. (memory: `project_kenken_energy_wave_result`)

## What we REFUTED / characterized (the honest negatives)
- **Expand-collapse / V-cycle waist — REFUTED** by the granularity probe. Per-breath latent
  is **coarse-always + fine-late**, not coarse-early-then-collapse and not oscillating:
  COARSE flat-high (and a trivial confound — N readable from valid-cell count), REGIONAL
  inlet-confounded (untestable), LOCAL value rises 0.84→0.93 (real, but already exploited by
  the per-breath readout). Radial control ρ=0.006 (orthogonal — re-confirms the refuted
  radial-depth thesis). Bryce's expand-collapse intuition is now refuted on **both** sides
  (loss-side schedule tied a ramp; representation-side has no stratification to exploit).
  (memory: `project_kenken_granularity_probe`)
- **Spectral "ring to segment regional motifs" — characterized.** Full-graph spectral
  recovers ROW/COL (ARI 0.46), not cages (0.18 ≈ null); KenKen is a dense rook's graph with
  cages a weak perturbation, recoverable only by ~3× cage-edge upweighting (ARI 0.98). Cages
  are a **given semantic channel**, not emergent topology → feed `membership` directly;
  spectral discovery is the **generality** mechanism for domains where regions are *hidden
  and dominant*. (memory: `project_kenken_spectral_regional`)

## The discipline that made it productive
**Measure-before-machinery + adversarial-verify** killed each dead end *cheaply*: the
volume harness (parity-gated) settled volume in one eval; the granularity probe (cheap,
leak-free, verified) gated a multi-hour waist build we then did **not** run. Same pattern
that earlier killed the cathedral, perm-aug, and waist-classify. Every claim was read
against a null and independently spot-checked.

## Strategic conclusion → the pivot
KenKen is a **clean verifiable CSP**, and the project already knows symbolic search dominates
those (CLAUDE.md §4). Volume can't help (rigid), the representation-side multi-scale ideas
don't pan out, and the energy gives a real-but-bounded base-p lift. The one result with legs
beyond KenKen is **energy = differentiable-verifier-in-the-loss** (general base-p raiser).
The deducer's actual edge — **generality + parallel "learned BP"** — has headroom only where
**exact symbolic inference is intractable**: soft / probabilistic / learned constraints
(§8.1). That is the next frontier; the testbed must keep the factor-graph abstraction (a
soft factor's "predicate" returns a continuous potential, not SAT/VIOLATED).

## Artifacts
- Code (`gen-weights`): `scripts/factor_graph_train.py` (energy wave: `FG_ENERGY_CAGE`,
  `FG_ENERGY_WAVE`), `mycelium/factor_graph_engine.py` (`fg_resid_capture` hook, byte-identical
  off), `scripts/kenken_volume_eval.py`, `scripts/diag_kenken_granularity_probe.py`,
  `scripts/diag_kenken_spectral_validation.py`.
- Ckpts: `fg_kenken_k16_reg` (generalizing base), `fg_kenken_ew_{off,monotonic,oscillating}`.
- Memory: `project_kenken_{generalization_fixed,volume_collapse,energy_wave_result,spectral_regional,granularity_probe}`.
