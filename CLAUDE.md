# Mycelium: A General Factor-Graph Reasoning Engine — Agent Brief

**Author:** Bryce + Claude · **Deadline:** Dec 25, 2026 · **Target:** MATH-500
**Platform:** Shadow Glass (AMD 7900 XTX, 24GB) · tinygrad + AM driver · no ROCm

**What Mycelium is (2026-06-20): ONE general engine for ANY factor graph.** A problem
becomes (variables + typed factor nodes + membership); the engine reasons over it. Two
halves are BUILT + VALIDATED: a general **DEDUCER** (the v98-lineage breathing
transformer, now domain-general) and a general symbolic **SEARCH tier** ("the deducer
proposes, complete search disposes"). **Generality is the holy grail — resist all
domain-specific code in the engine/core.** A would-be geometric front-end (the Poincaré
tiers) remains **spec-stage**. Read §0 first. Conceptual writeup: `README.md`; search-tier
design: `docs/general_factor_graph_search.md`; pre-v98 vision/empirics: `docs/archive/`.

---

## 0. The frame (read first)

Mycelium decomposes any problem into a **factor graph** and reasons over it with two
validated, general components, plus a spec-stage geometric front-end:

- **THE DEDUCER (built, validated, general).** A small iterative transformer (Pythia-410M
  L0–L3 SHARED across K=16 breaths) performs factor-graph inference by K passes through
  the same weights, with per-head attention masks built at runtime from `membership`. It
  is **byte-identical to the validated v98 KenKen executor** when driven with KenKen
  inputs, and **generalizes to graph coloring and hierarchical Boolean circuits** — one
  engine, three structurally different factor graphs (`mycelium/factor_graph_engine.py`).
  Its real superpower: **distributed/PARALLEL deduction that SCALES** (§4).
- **THE SEARCH TIER (built, validated, general).** A predicate-driven systematic search
  that wraps the deducer: "deducer proposes the ordering, complete symbolic search
  disposes." The **only** domain-specific code is a per-factor-type predicate + a thin
  bridge; the verifier / GAC propagation / variable-ordering (MRV) / value-ordering (LCV)
  / backtracking are ALL derived generically. Proven on coloring, SAT, and KenKen with
  **zero general-core edits each** (`mycelium/csp_core.py`, `docs/general_factor_graph_search.md`).
- **THE GEOMETRIC FRONT-END (spec-stage, NOT built).** A learned Poincaré (hyperbolic)
  ball that would *generate* the attention masks (topology) instead of hardwiring them.
  Reproduces hard masks frozen (byte-exact) but its relaxation is **blocked** for
  non-partition graphs, and its headline "radial-depth" payoff is **REFUTED** (§4, §7).

**THE TWO CHANNELS (the conceptual spine — keeps each piece honest).** A factor graph
splits into two ORTHOGONAL channels: **TOPOLOGY** (who connects to whom — `membership` →
attention masks) and **SEMANTICS** (what relation must hold — the predicate). The
Poincaré ball is the topology channel (it *generates masks*, carries no relation content).
The search tier's **predicate registry** is the semantics channel — and it is the
symbolic twin of the deducer's **verification inlet** (op+target features). Folding
semantics into the topology/mask channel is the **refuted move** (v100's "C2 death":
arithmetic as VERIFICATION, never an op-type mask channel).

**Discipline (the over-claim guard — getting this wrong is the main failure mode):**
- **Built/validated = the DEDUCER + the SEARCH TIER.** The Poincaré embedding + hyperbolic
  mask generator are **spec-stage** — NEVER state or imply they are built/working.
- **The engine is the v98-lineage deducer, NOT a perceiver.** The PERCEIVER IS RETIRED (5×
  refuted v118–v121; v300 failed flat at chance).
- **On clean verifiable CSPs, SYMBOLIC search dominates** (free, smaller trees); learned
  *probabilistic* propagation was net-negative. The deducer's demonstrated value is
  **generality + parallel deduction**, NOT search value/ordering (§4).
- **The radial-depth "deep prize" is REFUTED.** Do not present it as the goal.

---

## 1. The deducer in one paragraph (the validated executor)

A small iterative transformer (4 Pythia-410M L0–L3 layers SHARED across all K breaths,
h=1024, 16 heads, ~32M trainable + ~52M token-embeddings) performs factor-graph inference
by K passes through the same weights. Each breath: add a per-breath additive marker → 4-layer
transformer with a structured per-head attention mask encoding the factor topology → a
learnable per-breath `delta_gate` convex residual blend → per-breath layernorm +
value-codebook readout → per-breath calibration head. K breaths are JIT-unrolled into one
graph. Training: per-breath weighted CE (`loss = Σ_k (1 + k/(K−1))·CE(logits_k, target)`),
the "ladder" that makes K matter. The general engine (`mycelium/factor_graph_engine.py`,
`factor_breathing_forward` + `FactorGraphSpec`) is byte-identical to the v98 KenKen path
(`mycelium/kenken.py`, the regression oracle, never touched) and parameterizes it for any
typed factor graph: variable nodes, typed factor nodes, `membership`, an optional
verification inlet (`has_factor_inlet` — KenKen feeds op+target; coloring/circuits leave it off).

---

## 2. Deducer components, as-built (the breathing recipe)

Entry: `factor_breathing_forward` (`mycelium/factor_graph_engine.py`); KenKen oracle:
`kenken_breathing_forward` (`mycelium/kenken.py`).

| Component | What it does | Where |
|---|---|---|
| **Iterative shared-weight prefill** | K passes through Pythia L0–L3, SAME weights every breath; 1024d residual is the persistent state | `factor_graph_engine.py` |
| **Per-breath additive marker** | Orthogonal per-breath embedding added to the residual | `breath_embed` |
| **Per-head masks from membership** | `adj = membershipᵀ@membership > 0` → per-relation per-head masks (v98's 5/5/5/1 falls out generically via `_cell_mp_head_allocation`). Hard `{0,−1e4}` bias. | `mycelium/factor_masks.py`, `factor_graph_engine.py` |
| **Verification inlet (semantics channel)** | Per-factor op-type + log-bucketed target features (arithmetic as VERIFICATION, never an op-type mask channel — v100's C2 death). The neural twin of the search predicate. | `build_verification_inlet`, `kenken.py` |
| **Per-breath `delta_gate`** | Learnable convex residual blend `x = x_pre + gate_k·(h − x_pre)` | `model.*_delta_gate` |
| **Per-breath calibration head** | Scalar confidence per breath (Dopri5-style error-estimator hook) | `*_calib_head_*` |
| **Value codebook readout** | N-value codebook; aligned to `state_embed` at init | `value_codebook` |
| **Convergence instrument** | Min-based gold-free `breath_count_min` (argmin consecutive-belief JSD) + settled=correct-at-settle | `convergence_instrument` |
| **Codebook-orthogonality penalty** | Penalizes off-diagonal cos of row-normalized codebook gram; *rotates* collinear rows apart | trainer |
| **Per-breath weighted CE (the ladder)** | `1 + k/(K−1)` weighting; the reason K matters | trainer |

Trainer: `scripts/factor_graph_train.py` (`FG_TASK={kenken,coloring,circuit}`, kenken default).

---

## 3. The general search tier (Path B — `docs/general_factor_graph_search.md`)

Two layers over the factor-graph abstraction. **The only domain code is a per-factor-type
predicate + a thin bridge; everything else is general.**

- **Layer 1 — the general CSP interface** (`mycelium/csp_core.py`, ZERO domain identifiers):
  a three-valued predicate `predicate(ftype, params, member_values) → {SAT|UNVIOLATED|VIOLATED}`
  is the sole seam. The verifier (all factors SAT), GAC propagation (prune values with no
  supporting tuple; **commit only FORCED singletons**), MRV variable-ordering, and LCV
  value-ordering all derive from it. **DSATUR = MRV-on-coloring; AC-3 = GAC-on-not-equal**,
  proven by construction + parity-pinned. An `arity_cap` guards generic GAC; large-arity
  factors get an opt-in `specialized_propagator`.
- **Layer 2 — pluggable search strategies** (`backtrack_search` + `solve_symbolic`): systematic
  DFS now; MCTS / local-search later slot in with zero domain code. The neural deducer enters
  ONLY as **ordering priors** (bias MRV/LCV), never as commits.
- **The registry + per-domain bridges** live in `mycelium/csp_domains.py` (the ONLY file with
  domain knowledge). Adding a domain = a few `register()` calls + one bridge.

Drivers: `scripts/search_coloring.py` (+ the neural ordering arms), `scripts/search_kenken.py`;
parity gates: `scripts/test_csp_parity.py`, `scripts/test_kenken_parity.py`; legacy reference:
`mycelium/csp_coloring_legacy.py`.

---

## 4. Empirical status (the findings that fix the architecture)

**Generality PROVEN (Phase 0 + Phase 2 done, branch `mycelium-factor-graph`).**
- Phase 0 (commit `6af771c`): the predicate-driven core reproduces coloring **byte-identical**
  (symbolic `solve_symbolic` new==old on 22+ fixtures; GPU anchors B0 0.025 / B1 0.85 / B2b
  0.825 / B3 0.95 exact, VERIFY_PARITY max|Δlogit|=0). Zero domain identifiers in the core; a
  full **SAT** domain was added via predicate+bridge alone (solved + certified UNSAT), touching
  neither core nor registry.
- Phase 2 (commit `afc4a2f`): the SAME core solves **KenKen** (param-carrying arithmetic cages +
  7-ary all-different) with `git diff` showing only `csp_domains.py`. Symbolic search solves
  **100% across all givens bands**; B3 (GAC + all-different) collapses the tree (g10: 305
  decisions → 0.9). Three structurally different domains, one search tier, zero core edits each.

**Distributed deduction is PARALLEL and SCALES (the real superpower).** The deducer resolves
~4 deduction LEVELS per breath (a 4-layer transformer ≈ 4 attention hops), so it solves
depth-16 Boolean circuits in ~4 breaths (K_min ≈ D/4, sub-linear in depth; per-D flat to D16).
It is depth-PARALLEL, not depth-sequential.

**Symbolic dominates clean CSPs (the honest negative).** Neural-propagation-inside-backtracking
on hard 3-coloring LOST to symbolic search for free (AC-3 ceiling 0.95; neural probabilistic
propagation net-negative). A CONF_THRESH sweep confirmed the mechanism: at threshold→1.0 the
neural arm becomes byte-identical to no-propagation — sub-100% commits are pure losing bets.
**A propagation commit must be logically FORCED, not a confident guess.** The neural signal's
role is ordering, not committing. (Neural-ordering-as-PRIOR showed an UNVERIFIED hint on
coloring — gated, low headroom; the real test is the non-symbolic frontier.)

**Radial-depth deep-prize REFUTED.** On the circuit DAG testbed, ρ(per-node settle-breath,
topological depth) = 0.13 (bar 0.30, real spread, clean shuffle-null) — the breath allocation
is NOT depth-ordered. The engine solves hierarchy (~0.97) via distributed constraint-satisfaction,
not a depth-ordered geodesic traversal. Consistent with depth-PARALLEL above.

Detailed empirics live in `memory/` + git history.

---

## 5. Specifications

- **Init:** Pythia-410M L0–3 (attn + FFN + token embeddings 50304×1024), all 4 layers SHARED
  across K breaths.
- **Dimensions:** h=1024, 16 heads × 64, FFN 4096, vocab 50304. KenKen: 49-cell grid (N_max=7),
  7-value codebook, K=16, BATCH=8. Coloring/circuit set `n_values`/`s_max` per task.
- **Params:** ~32M trainable + ~52M token embeddings. AM K-graph limit: K=28 hangs; K=16 known-good.
- **Checkpoints:** `.cache/fg_ckpts/` — coloring (`fg_coloring_k16`) + circuit (`fg_circuit_*`).
  **No KenKen factor-graph deducer checkpoint exists yet** (`FG_TASK=kenken` on the existing
  trainer would produce one — no new code).
- **Platform:** AMD 7900 XTX, tinygrad, AM driver (Secure Boot off + `vm.compact_unevictable_allowed=0`).
  Ubuntu 24.04. No ROCm/CUDA/PyTorch.

---

## 6. Editing rules (durable, hard-won)

- **No mid-breath token generation.** Reasoning stays in the 1024d residual; tokens (if any)
  generated once at the end. ("had had had" if violated.)
- **Diversity must be structural, not learned.** Masks are geometric/structural. Every learned
  diversity mechanism (scales, soft tokens, fingerprints) collapsed to a constant within one epoch.
- **Digit/value-spaced for arithmetic.** Single-cell values; whole-number BPE tokens force memorization.
- **Factor per-NODE, not per-EDGE (v112b).** Prefer per-position gating over pairwise structures
  (learned attention biases, edge tensors). Edges are already captured by the binary masks.
- **Attention bootstrap.** New attention/pointer pathways (~30+ positions) don't bootstrap from
  task gradient on diverse data — they need an anchor or direct supervision. Codebook selection
  (≤32-way) bootstraps from task gradient alone. *Why the perceiver failed 5×.*
- **A propagation commit must be logically FORCED, not a confident guess** (the search-tier law).
  The neural deducer enters search ONLY as ordering priors, never as commits.
- **Soundness tests must cover the GENERAL regime, not just the deployed one.** Phase 2's
  all-different propagator was sound as deployed but unsound off the permutation regime, and the
  test only covered the deployed regime → false confidence. Test the general case to protect generality.
- **Substrate laws (tinygrad + AM driver):** no `dtypes.float32` literal inside the JIT step;
  `scores.clip(-1e4,1e4)`; where()-gated NaN guard (NOT multiply — NaN×0=NaN); single-kernel
  `isfinite`; knobs in the JIT cache key; assign-in-place fixed buffers for repeated JIT'd forwards
  (compile once, replay). Hyperbolic-specific: clamp `|z|²≤1−1e-5`, arccosh arg `≥1+1e-7`. See
  `memory/reference_tinygrad_am_quirks.md`.
- **Bryce wants root-cause perf fixes**, not workarounds, when perf is the bottleneck.
- **Process discipline:** commit/push ONLY when asked; **hold for the word before firing training
  runs**; **offer engineering critique before rubber-stamping** (esp. enthusiastic/gut-feel relays).

---

## 7. The geometric front-end (Tiers 1–2 — spec-stage, NOT built)

The would-be Poincaré embedding (Tier 1) + hyperbolic mask generator (Tier 2):
`docs/hyperbolic_mask_generator_spec.md`. **Status: spec-stage.**
- **Reproduces frozen, byte-exact** (the geometry can regenerate any factor graph's hard masks at
  `t=0`, machine-precision, partition + non-partition).
- **Relaxation BLOCKED for non-partition graphs.** The clique-union + both-members gate is
  gradient-dead (anchor grad = 0 under real CE); partition relations (KenKen rows/cols) relax but
  are flat. A relaxable non-partition construction is unbuilt research.
- **The radial-depth deep-prize is REFUTED** (§4) — radial position ≠ abstraction depth in the
  executor's dynamics. The geometry's transfer/interpolation payoff is UNPROVEN (gated on the
  blocked relaxation). Strictly additive, the hard mask is the permanent fallback if ever built.

---

## 8. Current direction (2026-06-20)

Consolidating the proven general engine, then aiming at the real frontier. Two threads:

1. **The frontier — a problem where symbolic propagation isn't enough.** Clean verifiable CSPs are
   won by symbolic search for free; the neural deducer earns its keep only where symbolic propagation
   is unavailable: **soft / probabilistic / learned / NL-specified constraints**. The deducer is
   literally "learned BP on a factor graph" — its natural frontier is approximate inference where
   exact symbolic methods are intractable. Pick a testbed that KEEPS the factor-graph abstraction
   (so generality holds — a soft factor's "predicate" returns a continuous potential, not SAT/VIOLATED).
2. **Minimize retraining when switching tasks (weight-side generality).** Today the engine code is
   general but the *weights* are per-domain (`fg_coloring`, `fg_circuit`, …). The holy grail is the
   weight-side mirror of the code-side win: a single domain-agnostic backbone (multi-task co-training),
   constraint semantics fed as INPUT (the neural predicate-registry / verification inlet — the
   two-channel framing), a universal masked codebook readout, optionally tiny per-domain adapters. Goal:
   switch tasks with zero/near-zero retraining.

**Resist ALL domain-specific code in the engine/core.** New domains enter through the predicate +
bridge (search) and the membership + inlet (deducer) — never the core.

**Key memory notes:**
- `memory/project_phase2_kenken_generality_proven.md` — generality proven on KenKen, zero core edits.
- `memory/project_phase0_general_search_core.md` — the predicate-driven core + the SAT-via-bridge proof.
- `memory/project_pathb_search_coloring_result_jun19.md` — symbolic dominates clean CSPs (the honest negative).
- `memory/project_factor_graph_two_channels.md` — topology vs semantics; registry ≠ Poincaré ball.
- `memory/project_distributed_deduction_scales_parallel.md` — the parallel-deduction superpower.
- `memory/project_radial_depth_thesis_refuted.md` — the refuted deep-prize.
- `memory/feedback_offer_engineering_critique.md` — push back before rubber-stamping.
- `memory/reference_tinygrad_am_quirks.md` — substrate laws.
