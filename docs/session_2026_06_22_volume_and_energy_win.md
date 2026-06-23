# Session memo — 2026-06-22: amortized-speed direction → the soft-violation energy WIN

**Authors:** Bryce + Claude · **Branch:** gen-weights → merged to `origin/main` @ `2792f67`

## TL;DR
A day of disciplined exploration on the **amortized-speed / generate-and-verify** direction.
Killed three plausible ideas cheaply (cathedral, perm-aug, the waist's classify objective),
validated the **volume** thesis (symmetry-diverse generate-and-verify), and landed a real,
general, *deployed* win: a **soft-violation constraint energy** (a differentiable relaxation of
the verifier in the loss) that raises the base model's per-dart validity across the board.

**Headline (coloring, vs the `fg_coloring_k16` baseline, sweet spot `cw=0.1`):**
- best-of-64 (symmetry generate-and-verify): `0.80/0.76/0.53/0.44 → 0.945/0.915/0.735/0.69` (bands c=1.0/1.5/2.0/2.5; +57% at the hard band)
- single-shot p_argmax: up ~1.4–2.3×
- early-stop deployment forward-saving: `3.26×/2.64×/1.67×/1.50× → 6.50×/4.74×/2.36×/2.02×`
- cell-acc held (~0.62) → genuine valid colorings, not the flattening degeneracy
- **best ckpt:** `.cache/fg_ckpts/fg_coloring_k16_energy_cw01/`

## The arc (one line)
Killed the cathedral → picked amortized-speed → found the win is *volume* not raw speed →
chased it through three refuted hypotheses → adversarial review prescribed the aligned
objective → it worked.

## What we KILLED (cheaply — and that's the point)
- **Cathedral — refuted** by the read-at-settle control. The "forgetting tail" it was meant to
  fix is at the noise floor on all 3 domains *and shrinks with depth* (oracle interior-peak
  ceiling C−A: coloring +0.0032, circuit +0.0000, kenken +0.0004; noise band ±0.0030).
  (memory: `project_cathedral_forgetting_refuted`)
- **Standalone speed-vs-symbolic — dead at n≤49.** Symbolic search is both fast *and* exact on
  colorable instances at the engine's hard S=49 cap (GAC+DSATUR collapses the trees; blow-up
  lives on the UNSAT side + at n≈160–200). The recurring **S=49 wall** (quality + soft-MRF hit
  it too). Symbolic is a yardstick, not the opponent. (memory: `project_amortized_speed_direction`)
- **Permutation-augmented training — refuted** (both from-scratch [catastrophic] and gentle
  fine-tune [mildly negative]). Lesson: symmetry-TTA's diversity *comes from* the deducer's
  position-dependence; training for equivariance kills it. Don't fight the model.
  (memory: `project_perm_aug_refuted`)
- **The learned waist's `classify` objective — caught before firing.** Adversarial review flagged
  it as a discriminative trap (makes the rep separable, never moves the output); `attract` had the
  wrong geometry. The perm-aug guardrail stopped a wasted GPU run.
  (memory: `project_waist_build_objective_finding`)

## What we WON
- **Generate-and-verify VOLUME — validated.** Throw M cheap darts; a free exact verifier keeps any
  valid one. **Symmetry-diverse darts** (solution-preserving vertex permutation = test-time
  augmentation — the fast.ai / George-Hotz / Anna-Karenina connection) turn a 3–32% single-shot
  deducer into 44–80% best-of-64. (temp = incoherent/mean-field; multistart = collapses;
  symmetry = the winner.)
- **THE WIN — soft-violation energy.** `make_coloring_constraint_energy` reuses the generic
  all-different form `relu(membership @ softmax − 1)` on the **softmax** (differentiable), wired as
  the coloring `constraint_energy_fn` (was `None`), minimized alongside CE. Directly steers the
  deduction toward collision-free (valid) outputs. The first base-model p-raiser that worked;
  **general** (transfers to KenKen / any all-different domain). cw sweep: monotonic gains
  baseline→0.05→0.1, plateaus by 0.2; **cw=0.1 is the sweet spot.**
  (memory: `project_soft_violation_energy_win`)
- **Common-mode instinct — confirmed real.** The de-biased cluster probe: weak at raw 1024-d
  (transferable AUC 0.582) but *sharpens under compression* (PCA 0.658 @ d=256); the learned-waist
  gate proved it decisively — a learned nonlinear projection separates valid/invalid at **0.85** vs
  PCA's 0.58. The in-deducer waist infra is **built and banked** (off by default, byte-identical,
  oracle-clean) for when we want it. (memory: `project_common_mode_centroid_probe`)

## Perf engineering (all measured, not assumed)
- **JIT forward: 9.4×** over eager; byte-identical parity; multistart noise verified to vary per
  replay (assign-in-place into the captured buffer, not attribute rebind = the frozen-noise trap).
- **Batching is a no-op** — GPU is compute-bound at B=8 (per-instance cost flat 8→128).
- **K reduction is a wash** for volume (cheaper darts need more darts); keep K=16.
- **Adaptive early-stop** (stop at first verified dart): 1.5–3.3× fewer forwards at *identical*
  solve-rate (parity-verified: early-stop solve-rate == full-M); compounds with higher p.

## Tooling shipped (all additive, oracle-clean, byte-identical-off)
- `scripts/amortized_frontier_measure.py` — speed + volume harness (best-of-N vs independent ideal,
  fragility buckets, throughput, JIT forward, `--early-stop`, `--capture-darts`).
- `scripts/dart_cluster_probe.py` — de-biased cluster-separability probe (n-fair spread, symmetric
  anchor, by-instance CV, PCA dim sweep).
- `scripts/learned_waist_gate.py` — learned-vs-PCA gate (by-instance CV, leak-free).
- `mycelium/factor_graph_engine.py` — banked off-by-default learned waist (zero-init-gated convex blend).
- `scripts/factor_graph_train.py` — `FG_PERM_AUG` (refuted, kept off), `FG_WAIST*` (banked),
  `make_coloring_constraint_energy` (**the win**, `FG_CONSTRAINT_WEIGHT`).

## The durable lesson
**Measure-before-machinery + offer-critique + adversarial review** is what made the day productive:
it killed the cathedral, perm-aug, and the waist-classify dud *cheaply*, and the same review that
killed the dud *prescribed* the aligned objective that won. To raise p: **minimize a differentiable
relaxation of the verifier in the loss** — do NOT retrain for invariance or bolt on a post-hoc
classifier (those fight the model / buy nothing next to a free verifier).

## Where it stands / open next moves (none firing without the word)
- Carry the energy recipe to **KenKen / a fresh domain** — the generality test (it's generic).
- **Per-breath energy** (the ladder) vs final-breath only — possible further squeeze.
- **Stack the banked waist** on the now-stronger base (real p-raiser to build on).
- Eventually: scale **N > 49** + dims **1k → 4k (Llama)** — where the poly-vs-exp math finally
  favors the deducer (load-bearing unknown: does per-sample p hold as N grows? — measure p-vs-N first).
