# Mycelium: A General Factor-Graph Reasoning Engine — Consolidated Results

**Status:** CONSOLIDATION (2026-06-20). Two halves BUILT + VALIDATED (the general
deducer; the general symbolic search tier, Phases 0 + 2). Three honest negatives
banked. The geometric (Poincaré / hyperbolic) front-end is SPEC-STAGE; the
radial-depth "deep prize" is REFUTED; the soft-MRF frontier is BURIED. Author:
Bryce + Claude. Mirrors `docs/general_factor_graph_search.md`, `README.md`,
`CLAUDE.md`. Every quantitative claim is traced to a committed artifact (§8).

---

## 0. Abstract

Mycelium is a **general factor-graph reasoning engine**: ONE engine for ANY factor
graph (variable nodes + typed factor nodes + membership). It has two validated
halves — a general **deducer** (the v98-lineage breathing transformer, Pythia-410M
L0–L3 shared across K=16 breaths, framed as learned loopy belief propagation on a
factor graph) and a general symbolic **search tier** ("the deducer proposes, complete
search disposes"), whose only domain code is a per-factor-type predicate plus a thin
bridge. **The contribution is generality, not solution quality.** The same engine
solves graph coloring, Boolean SAT, KenKen, and hierarchical Boolean circuits with
zero general-core edits each; the deducer's distributed deduction is *parallel* and
scales sub-linearly in depth (~4 deduction levels per breath; depth-16 circuits in
~4 breaths). But on inference QUALITY the engine beats nothing: on clean verifiable
CSPs symbolic search dominates and learned probabilistic propagation is net-negative;
on soft MRFs cheap baselines (loopy BP for marginals, simulated annealing for MAP)
already close the gap on both flat and DAG topologies; and the radial-depth thesis
(deduction-depth ↔ radial position ↔ breath-count) is refuted (ρ ≈ 0.13). The
characterized edge is **generality + amortized-fast parallel inference (a fixed
forward pass), NOT better answers than a bespoke solver.**

**Thesis sentence.** One engine over any typed factor graph, whose proven edge is
cross-domain generality plus amortized-fast parallel deduction — *not* superior
inference quality, which on every regime we measured belongs to a cheaper bespoke
method.

---

## 1. The thesis — one engine, any factor graph

A problem's topology should not be hardwired. A factor graph — variable nodes, typed
factor nodes, and a membership relation (who shares which factor) — is a substrate
general enough to express graph coloring, SAT, arithmetic CSPs (KenKen), and layered
Boolean circuits in the same vocabulary. Mycelium commits to that substrate and asks
a single question: *can ONE engine, with no per-domain core surgery, reason over all
of them?* The answer is **yes for generality** (§3) and **no for quality** (§5).

The honest framing matters because the negatives are as much the science as the
generality win. This write-up is built to survive an over-claim audit: it never
implies the retired perceiver works, never implies the spec-stage geometry is built,
never implies the refuted radial-depth prize holds, and never claims the engine beats
a bespoke solver on quality.

---

## 2. Architecture

### 2.1 The two channels — the conceptual frame

A factor graph decomposes into **two orthogonal channels**:

- **TOPOLOGY** — *who connects to whom* (membership). Drives the per-head attention
  masks in the deducer; in the search tier it is the neighborhood / GAC incidence.
- **SEMANTICS** — *what relation must hold* (the predicate / the verification content).
  In the deducer it is the per-factor verification inlet; in the search tier it is the
  predicate registry.

The channels are independent: graph coloring and a hypothetical "adjacent vertices
must be EQUAL" problem share the *same* topology but have *opposite* predicates, so
topology alone cannot distinguish them. Keeping the two separate is a hard-won
discipline — v100's "C2 death" tried to fold constraint semantics (op-type) into the
mask channel and failed, which is why the project's rule reads *arithmetic as
VERIFICATION, never an op-type mask channel*
(`memory/project_factor_graph_two_channels.md`).

### 2.2 The general deducer (the breathing transformer)

The deducer is the v98-lineage breathing transformer and is **byte-identical to the
validated v98 KenKen executor when driven with KenKen inputs**; the same architecture,
same shared weights, same forward generalizes to coloring and to Boolean circuits.
(It is the v98-lineage deducer, **not a perceiver** — the perceiver is retired, §7.)

- **Iterative shared-weight prefill.** K passes through Pythia-410M L0–L3 (4 layers,
  h=1024, 16 heads × 64, FFN 4096), the SAME weights every breath; the 1024-d residual
  is the persistent state. ~35.7M shared transformer params + ~51.5M token embeddings
  = **~87M total** (`CLAUDE.md` §4 / `README.md`).
- **Per-head masks BUILT AT RUNTIME from membership** (topology channel). For each
  factor type, adjacency `A_t = membership_tᵀ @ membership_t > 0`; heads are allocated
  by `cell_mp_head_allocation` (deterministic, pure numpy); a `{0, −1e4}` additive bias
  is applied with validity masking. Masks vary per batch / per problem — they are never
  hardwired. KenKen example: T=3 types, H=16, G=1 → **5 row / 5 col / 5 cage / 1 global**
  (the v98 5/5/5/1 layout) (`mycelium/factor_masks.py`).
- **Per-breath additive marker.** Orthogonal embedding (QR, scale 0.5), allocated once;
  separates per-breath gradients without forcing specialization.
- **Verification inlet** (semantics channel). Per-factor relation content (KenKen:
  op-type 4-way + log-bucketed target + cage-size). Added to the residual EVERY breath
  (`x_in = x + breath_marker_k + inlet_h`) — **never gated** (funded-vs-starved).
- **Per-breath `delta_gate`.** Learnable convex residual blend
  `x = x_pre + gate_k·(h − x_pre)` (init ones = full update), the K breaths JIT-unrolled
  into one graph.
- **Value-codebook readout.** Orthonormal codebook (QR, scale 0.1), aligned to the
  state embedding at init; per-breath `cell_logits = layernorm(x) @ codebookᵀ + bias`.
  **Per-cell, not per-query** (`factor_graph_engine.py:386`). This per-cell-independent
  softmax is the **mean-field ceiling** that bounds marginals (§5.2).
- **Per-breath calibration head.** Reads mean-pooled valid cells → a scalar confidence,
  trained by MSE against a gold-free convergence instrument.
- **Loss: the per-breath weighted-CE ladder**, `weight_k = 1 + k/(K−1)` — the reason the
  K axis is meaningful — plus calibration MSE and an optional codebook-orthogonality
  penalty (OFF by default; byte-identical to v98 when off).

The architectural identity (three vocabularies, one object): K breaths = **loopy-BP
rounds**; the masks = factor topology; the codebook softmax = beliefs; equally, K
breaths = an **iterative ODE integrator** stepping energy descent; equally, an
**iterative prefill** with the residual as persistent state.

### 2.3 The general symbolic search tier ("deducer proposes, search disposes")

A predicate-driven, sound, complete CSP solver. The **only** domain code is one
per-factor-type predicate plus a thin bridge; everything else derives generically:

| derived operation | how it falls out (no domain code) |
|---|---|
| **verifier** (`verify_complete`) — the sole `solved` arbiter | every factor predicate `== SAT` on the complete assignment |
| **partial-soundness gate** (`is_consistent_partial`) | no factor predicate `== VIOLATED` on its partial tuple |
| **GAC propagation** (`gac_propagate`) | prune unsupported values; **commit only the SOLE survivor of a domain — logically forced, never a guess** |
| **MRV var-ordering** | fewest legal values, tie-break factor-degree then index |
| **LCV val-ordering** | value removing fewest options from co-factor members |

The predicate is three-valued (`VIOLATED / UNVIOLATED / SAT`); a hole-monotonicity
contract (**L-MONO**: assigning a hole never turns VIOLATED into not-VIOLATED) is
enforced by a randomized check at registration. **DSATUR-is-MRV** and **AC-3-is-GAC**
are proven *by construction*, not asserted — the core never knows it is doing DSATUR or
AC-3 (`mycelium/csp_core.py`, `csp_registry.py`, `csp_domains.py`). A cost guard
(`arity_cap = 20000`, mirroring `build_kenken_data.propagate:217`) routes oversized
factors to an optional `specialized_propagator` (e.g. `l_alldiff_propagator` for the
7-ary all-different), keeping the tier sound *and* tractable. The neural signal, when
present, enters **only as a prior to the orderers — never into `gac_propagate`**.

---

## 3. The generality result — one engine, four factor graphs, zero core edits

The same engine spans four structurally different factor graphs. The search tier's
core (`csp_core.py` + `csp_registry.py`) has **zero domain identifiers** — every grep
hit is docstring/comment prose; the executable code is domain-free — and the *only*
file that changes between domains is the bridge `csp_domains.py` (proven by `git diff`).

| domain | factor structure | search-core edits | result (source) |
|---|---|---|---|
| **Graph coloring** | binary `not-equal` cliques (flat, non-partition) | zero | symbolic B3 (AC-3 ceiling) **0.95**, B1 (no-prop) **0.85** @b100; B0 pure deduction **0.025** |
| **Boolean SAT** | n-ary clauses | zero (predicate + bridge only) | solved + UNSAT-certified by the leak-hunter add, touching neither core nor registry |
| **KenKen** | param-carrying n-ary arithmetic cages + 7-ary all-different (flat partition) | zero (`git diff` = only `csp_domains.py`) | symbolic B1 + B3 solve **100%** across all givens bands g40/g30/g20/g10 |
| **Boolean circuits** | AND/OR/NOT layered DAG (hierarchical) | zero | deducer solves + scales (§4) |

**Phase 0 (Jun 19, commit `6af771c`) — faithful + general.** The refactored core
reproduces coloring byte-identically: `solve_symbolic` new==old on **22+ fixtures**
(status + decisions + backtracks), GAC==AC-3 on live branches, MRV==DSATUR at every
node; **105 tests PASS + 15 fixtures with real backtracking**. GPU anchors reproduce
exactly (B0 0.025 / B1 0.85 / B2b 0.825 / B3 0.95), **`VERIFY_PARITY max|Δlogit| = 0`,
n=40, ~14 min wall** (`memory/project_phase0_general_search_core.md`). The leak-hunter
added a full SAT domain through the public predicate + bridge API alone.

**Phase 2 (Jun 19, commit `afc4a2f`) — real generality.** The *untouched* core solves
KenKen — structurally unlike coloring — with domain code only in `csp_domains.py`
(`cage_pred` + `problem_from_kenken` + the `l_alldiff_propagator`). Symbolic GAC
collapses the tree exactly as the propagation-strength claim predicts:

| band | decisions B1 → B3 | backtracks B1 → B3 |
|---|---|---|
| g40 | 28.6 → **0.0** | — |
| g10 (hardest) | 305.1 → **0.9** | 386.3 → **1.1** |

The all-different's Hall value-occurrence rule was scope-guarded to the permutation
regime (`len(scope) == |value-universe|`) after an adversarial review caught an
off-permutation unsoundness — sound as deployed (KenKen rows/cols are full
permutations) and now soundly general (`memory/project_phase2_kenken_generality_proven.md`).

**The one-trick-pony guarantee held:** three (four with circuits) domain-distant
problem classes, predicate + bridge per domain, zero search-core edits.

---

## 4. The deducer's characterized edge — generality + parallel deduction that scales

The deducer's distributed deduction is **parallel, not depth-sequential**, and it
scales sub-linearly in depth (probe machinery committed at `b8d73a1`,
`scripts/eval_circuit_scaling.py`; the numbers below are from the recorded run on the
trained `fg_circuit` checkpoint, logged in
`memory/project_distributed_deduction_scales_parallel.md` — they depend on that
checkpoint, which is not in-tree, so they are not reproducible from the committed script
alone).

- **K-sweep on depth-16 circuits.** cell_acc at K=4 is **0.877** vs K=16 **0.923** —
  ratio **0.95**. Four breaths recover 95% of full-K performance. All depth bands D6–D16
  show K=4/K=16 ratios 0.95–1.01 → PARALLEL.
- **Per-depth accuracy is FLAT** (no depth ceiling): D6 0.958 / D8 0.966 / D10 0.944 /
  D12 0.953 / D14 0.935 / **D16 0.923** — a gentle ~3.5 pt decline over a 10-level
  depth range.
- **The mechanism.** Each breath is a 4-layer transformer ≈ ~4 attention hops of
  constraint propagation, so **~4 deduction levels resolve per breath** → `K_min ≈ D/4`,
  sub-linear in depth. This is the loopy-BP wavefront signature: a breath is a 4-deep
  propagation front, not one sequential step.

This is the engine's real superpower, and it is amortized: a *fixed* forward pass
(within the AMD 7900 XTX JIT capacity of ~20 breaths) resolves depth that a sequential
solver would pay for step by step. It is a claim about depth-parallelism on the
circuit DAG — **not** a meta-learning, transfer, or train-to-larger-problems claim.

---

## 5. The honest boundaries (the heart of the rigor)

The engine beats nothing on inference QUALITY. The negatives below are independent,
each instrumented, each gated before any build.

### 5.1 On clean verifiable CSPs, symbolic search DOMINATES — and learned propagation is NET-NEGATIVE

On hard 3-coloring (depth ≥ 3, n=40), symbolic methods win for free with smaller trees,
and the learned probabilistic propagator costs forwards and *hurts* (commit `649855a`,
`memory/project_pathb_search_coloring_result_jun19.md`):

| config @b100 | solve_rate | decisions | backtracks | forwards/solve |
|---|---|---|---|---|
| B3 symbolic AC-3 (ceiling) | **0.95** | 16.1 | 15.6 | 0 |
| B1 symbolic, no propagation | **0.85** | 46.9 | 29.5 | 0 |
| B2b neural-prop + DSATUR + LCV | 0.825 | 44.2 | 31.6 | 48.3 |
| B2 neural-prop + entropy + policy | 0.475 | 45.2 | 83.3 | 55.8 |
| B2c neural-prop + entropy + LCV | 0.350 | 52.9 | 101.6 | 67.4 |

B2b (0.825) ≤ B1 (0.85, the *identical* config minus the learned propagator): adding
the neural propagator marginally HURTS, costs ~48 forwards, and does not collapse the
tree.

**The mechanism, nailed by a CONF_THRESH sweep — "commit only when forced."** B2b
solve_rate @b100 vs the propagation-commit threshold: at 0.90 it is **0.825** (below
no-prop, i.e. the commits are net-losing bets); at 0.95/0.99/1.0 it is **0.850**
(== B1). At threshold = 1.0 (never commit) **B2b is byte-identical to B1**
(decisions 46.9 == 46.9, backtracks 29.5 == 29.5). **No threshold rises above B1 →
neural probabilistic propagation has ZERO positive operating point on coloring.** The
design principle, now proven not argued: *a propagation commit must be a logically
FORCED move; the neural signal's role is ORDERING, not committing.* (Caveat: this is a
finding about the deducer's signal on *clean* CSPs; it does not generalize to
constraints where symbolic propagation is unavailable.)

The search-tier viability gate (commit `b973197`) localizes why: G1 (policy ranks gold)
**FAILS** — mean gold-prob **0.261 < uniform 1/k = 0.333** at wrong vertices
(confidently wrong); G2 (entropy flags ambiguity) **PASSES**, AUC **0.69**; G4 (clamp
gold, re-deduce) **PASSES** monotonically, **+0.77 / +1.08 / +2.21 pt** for 1/2/4 clamps,
holding off-distribution; calibration-vs-frac-correct correlation is a weak
**~0.27–0.31**. The search *substrate* works (localize + propagate); the learned
*prior + value* do not, on clean CSPs.

### 5.2 The soft-MRF frontier is BURIED — marginals and MAP, flat and DAG

Soft / probabilistic factor graphs (Ising/Potts) are the one regime where the
clean-CSP net-negative might invert: a soft factor `exp(J·sᵢsⱼ + h·sᵢ)` is never
VIOLATED, so symbolic GAC has no admissible move; the task becomes #P-hard marginals +
NP-hard MAP, whose tractable SOTA is loopy BP — which the deducer *is*. Two cheap
kill-gates ran before any training.

**Gate-1 — marginals (commit `4c8f6eb`, `scripts/frontier_bp_gap_gate.py`): WEAK PASS
= effectively a SOFT-KILL.** On toroidal 2D Ising (brute-force-exact vs damped loopy
BP), the BP-vs-exact gap and its localizability are **anti-correlated**: at β 0.3–0.5
error is localized but the gap is tiny (MAE ≤ 0.01, BP ≈ exact); at β ≥ 0.9 the gap is
large (MAE 0.10–0.35) but per-spin error is *more uniform than random noise* AND BP
stops converging (BP convergence collapses to as low as **17%** — i.e. up to ~**83%**
non-convergent; mean convergence on the large-gap cells is ~**45%**, i.e. ~55%
non-convergent on the mean). **Only 1/10 gap-cells is both localized-above-null and on
a stable BP fixed point** — no clean PASS. Ferro control verifies the BP implementation
(MAE ≤ 2e-4 in the disordered band).

**Gate-2 — MAP + DAG salvage (commit `688f1b5`, `scripts/frontier_map_dag_probe.py`):
KILL, by both the build's verdict and an independent adversarial re-derivation.**
On flat 2D Ising MAP the gap is real (Hamming to 0.60) but **cheap simulated annealing
closes ~87%**, and a **fair stronger SA (2000 sweeps × 5 restarts) closes 100% on every
cell**; where any residual survives, BP is 25–38% non-convergent and the wrong nodes
are unlocalizable by BP confidence (AUC 0.33–0.53 = chance). On the DAG noisy-circuit,
**max-product BP finds the exact MPE in every band (Hamming 0.000, 100% convergence,
robust to ε ≤ 0.30)** — the DAG is BP-trivial, no gap for a learned corrector.

**The mean-field ceiling (structural, not a bug).** The readout is a per-cell-independent
softmax (`cell_logits = layernorm(x) @ codebookᵀ`, `factor_graph_engine.py:386`). A
product-form readout can only emit mean-field (independent) marginals — it structurally
projects away the pairwise correlations that *are* the marginal. A true marginals win
would need a Bethe / pairwise readout, i.e. a real core change. This is load-bearing
(the small codebook bootstraps from task gradient; a joint posterior is intractable),
not "fixable with better training."

Verdict: across the full matrix — clean CSPs (symbolic wins), soft-MRF marginals
(gate-1), soft-MRF MAP (gate-2), on **both** flat and DAG — *deducer beats BP/symbolic
on inference QUALITY* is refuted. (Caveat: the soft-MRF probes are measured on
tractable-regime testbeds — flat 2D toroidal Ising for marginals, n≤10 trees and
consistent-clamp DAGs for MAP — chosen so an exact reference exists; the one narrow
unexplored escape is an adversarial inconsistent-evidence high-treewidth DAG, which
would still need cheap baselines to also fail.)

### 5.3 The radial-depth "deep prize" is REFUTED

The hypothesis — deduction-depth ↔ radial position ↔ breath-count, the breath cycle as
a geodesic engine where radial position = abstraction level — does NOT hold on the
Boolean-circuit DAG testbed (commit `98d1bea`,
`memory/project_radial_depth_thesis_refuted.md`). Direct test:
**ρ(per-node settle-breath, topological depth) = 0.134, lower-CI 0.115, against a bar
of 0.30** → NULL. This is a real signal, not a degenerate one (15 unique settle-breaths,
std 4.4, range [2, 16], p_perm 0.0001, n=10214; depth-shuffle null collapsed to
ρ ≈ 0.02) — but it is well below the bar. The engine solves hierarchy (~0.97 cell, no
depth ceiling) via **distributed constraint-satisfaction, not depth-ordered radial
traversal**. This is *consistent* with §4: the engine is depth-PARALLEL (~4 levels per
breath), which is precisely why settle-breath does not track depth. The two findings
support each other.

---

## 6. Honest positioning — what the novelty actually is

The contribution is **(a) cross-domain generality** — one engine, any factor graph,
zero general-core edits, demonstrated on four structurally different domains — **(b) the
rigorous negatives** — three independent, instrumented refutations that map exactly
where a learned solver does and does not pay off — and **(c) the amortized framing** —
the deducer's value is a fixed, parallel forward pass (~4 deduction levels per breath),
*not* better answers than a bespoke solver.

The reframe is the durable conclusion: **the deducer's proven edge is GENERALITY +
AMORTIZED-FAST PARALLEL inference, NOT inference quality.** On clean CSPs symbolic
search dominates (0.95 vs 0.025 deduce-to-search gap; neural propagation net-negative);
on soft MRFs cheap baselines win (loopy BP on marginals, SA on MAP); on the DAG the
radial prize is refuted. The novelty is engineering + generality, stated as such, not
extrapolated to a quality headline. The "learned loopy BP" framing is a *framing* of an
approximate iterative solver trained on CSPs — not a soft-factor BP variant that
inherits BP's convergence guarantees.

---

## 7. Limitations + what we do NOT claim

- **We do NOT claim the engine beats SOTA or any bespoke solver on inference quality.**
  It does not. Symbolic search dominates clean CSPs; loopy BP wins soft-MRF marginals;
  SA wins MAP. The edge is generality + amortized parallel inference.
- **We do NOT claim learned probabilistic propagation helps on clean CSPs.** It is
  net-negative; the CONF_THRESH → 1.0 sweep shows it has zero positive operating point
  (byte-identical to no-propagation at its best).
- **We do NOT claim the soft-MRF frontier is promising or that the deducer generalizes
  to soft constraints.** It is BURIED on both marginals and MAP, flat and DAG. (The
  probes are tractable-regime testbeds with exact references; the soft regime is
  research-stage, not a banked result.)
- **We do NOT claim the radial-depth "deep prize" holds.** It is REFUTED (ρ ≈ 0.13).
  The engine is depth-parallel, not depth-sequential; this is not a future direction or
  an extension point.
- **We do NOT claim the Poincaré (hyperbolic) embedding or hyperbolic mask generator
  are built, working, or validated.** They are **SPEC-STAGE**. The geometry can reproduce
  the hard masks frozen (byte-exact at t=0), but **relaxation is gradient-dead for
  non-partition factor graphs** (clique-union co-location has no per-node radial DOF to
  learn from), and the radial-depth payoff it was meant to deliver is refuted.
  Geometric transfer / interpolation is UNPROVEN, gated on the blocked relaxation.
- **We do NOT claim the perceiver works.** The perceiver is RETIRED — 5× refuted
  (v118–v121 negatives; v300 perceiver-core failed flat at chance). The engine is the
  v98-lineage breathing deducer, not a perceiver, and is not framed as "perceiver
  beats X."
- **We do NOT claim a neural deducer + symbolic search beats symbolic alone.** On KenKen
  symbolic GAC + all-different trivializes the tree (305 → 0.9 decisions on g10), leaving
  no residual for neural ordering to reduce.
- **"Byte-identical to v98 KenKen" is exact only on KenKen inputs** with matching
  hyperparameters. The correct framing for coloring/circuits is *same architecture,
  proven general*, not "byte-identical generalization."
- **The neural-ordering arm verdict on KenKen is UNVERIFIED.** The coloring
  neural-ordering numbers (B2 0.925 / B2c 0.900, redesigned per spec §3.2) still use the
  refuted auto-commit propagation + complete fallback at n=40 — *not* a clean
  ordering-only test, and coloring's true ceiling is symbolic AC-3 0.95. The decisive
  test (neural ordering on top of symbolic GAC, on KenKen where GAC is weaker) is gated
  on a trained KenKen deducer checkpoint **that does not yet exist**. The symbolic
  generality is proven; the neural-KenKen verdict is not.

---

## 8. Sources (every number traces here)

**Commits.** `6af771c` Phase 0 (predicate-driven general core) · `afc4a2f` Phase 2
(KenKen generality) · `b973197` search-tier gate (G1/G2/G4) · `649855a` Path B
(verifier-driven branch-and-propagate) · `b8d73a1` B3 scaling probe (deep-skinny
circuits) · `98d1bea` B3 settle-breath vs depth analyzer (radial refutation) ·
`4c8f6eb` frontier gate-1 (BP-gap kill-gate) · `688f1b5` frontier salvage probe
(MAP + DAG KILL).

**Code.** `mycelium/factor_graph_engine.py` (deducer; per-cell readout `:386`) ·
`mycelium/factor_masks.py` (runtime masks, `cell_mp_head_allocation`) ·
`mycelium/csp_core.py` / `csp_registry.py` / `csp_domains.py` (search tier) ·
`mycelium/kenken.py` (v98 regression oracle) · `mycelium/circuit_data.py` /
`graph_coloring_data.py` (testbeds) · `scripts/eval_circuit_scaling.py` ·
`scripts/analyze_circuit_rho.py` · `scripts/search_coloring.py` (CONF_THRESH sweep) ·
`scripts/search_kenken.py` · `scripts/frontier_bp_gap_gate.py` ·
`scripts/frontier_map_dag_probe.py`.

**Memory / docs.** *(`memory/` is the agent memory store outside the repo tree, not a
committed directory; the in-tree, clone-reproducible artifacts are the commit hashes,
the scripts, and `docs/` above. The memory notes are supplementary records of runs —
in particular §4's circuit-scaling numbers and §5.3's ρ=0.134 are recorded-run outputs
on a trained `fg_circuit` checkpoint that is not in-tree, so they are reproducible only
from that checkpoint, not from the committed scripts alone.)*
`memory/project_phase0_general_search_core.md` ·
`project_phase2_kenken_generality_proven.md` ·
`project_pathb_search_coloring_result_jun19.md` ·
`project_search_tier_gate_jun19.md` ·
`project_distributed_deduction_scales_parallel.md` ·
`project_radial_depth_thesis_refuted.md` ·
`project_frontier_soft_factor_graphs.md` ·
`project_factor_graph_two_channels.md` · `docs/general_factor_graph_search.md` ·
`docs/hyperbolic_mask_generator_spec.md` (SPEC-STAGE) · `CLAUDE.md` · `README.md`.
