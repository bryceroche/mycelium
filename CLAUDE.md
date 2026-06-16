# Mycelium: The Breathing Transformer — Agent Brief

**Author:** Bryce + Claude · **Deadline:** Dec 25, 2026 · **Target:** MATH-500
**Platform:** Shadow Glass (AMD 7900 XTX, 24GB) · tinygrad + AM driver · no ROCm

**The architecture we are building toward (2026-06-16): a THREE-TIER system built
around a learned Poincaré (hyperbolic) ball.** Tier 3 (the executor) is DONE,
VALIDATED, and LIVE. Tiers 1–2 (the Poincaré embedding + the hyperbolic mask
generator) are the ACTIVE RESEARCH PROGRAM — **spec-stage, not built, not tested**
(`docs/hyperbolic_mask_generator_spec.md`, the foothold is not yet fired). Read §0
first; it frames everything below. For the conceptual writeup see `README.md`; for
the paper `paper/outline.md`; for pre-v98 vision/empirics `docs/archive/`.

---

## 0. The three-tier architecture (the frame)

A problem's topology should not be hardwired. The target architecture compiles a
continuous topology *signature* into an executable attention mask, then runs pure
iterative deduction on it. The Poincaré ball is the substrate because problem
topologies are **hierarchical** (cell ∈ cage ∈ board; nested DAG sub-computations),
and hyperbolic space embeds hierarchy with low distortion — **radial position =
abstraction level**.

- **TIER 1 — Structural Mapping (continuous topology embedding).** A problem's
  geometry + dependency-logic is mapped to continuous coordinates in a learned
  Poincaré ball. Replaces rigid one-hot problem IDs with a continuous signature →
  structural interpolation/transfer across problem classes. **SPEC-STAGE, NOT BUILT.**
- **TIER 2 — The Compiler / "virtual factor graph" (the HYPERBOLIC MASK GENERATOR).**
  Generates the per-head attention masks from the Tier-1 coordinates instead of
  hardwiring them. ONE coordinate field **per relation** (row/col/cage — the triangle
  inequality forbids one field for all three), closed-form **ANCHORED at `t=0` to
  reproduce the v98 hard mask exactly (~1e-3)**, then RELAXED;
  `bias = −softplus(α·(d_hyp − r))`. A differentiable virtual machine that compiles a
  continuous topology "program" into an executable mask. Design doc:
  `docs/hyperbolic_mask_generator_spec.md`. **SPEC-STAGE, NOT BUILT, FOOTHOLD NOT FIRED.**
- **TIER 3 — The Core Executor (the VALIDATED v98 KenKen breathing transformer).**
  Pure iterative deduction on whatever masks Tier 2 provides. This is the breathing
  recipe: shared Pythia-410M L0–L3, K=16 breaths, per-breath `delta_gate` +
  calibration head, value-codebook readout, per-breath weighted-CE ladder, gold-free
  convergence instrument. **THIS TIER IS VALIDATED AND LIVE** — the Property-2 K=16
  curriculum run is training now (§3). Today it consumes hardwired masks; it already
  takes topology as a runtime input (`build_kenken_attn_bias`), so it is mask-flexible.

**Discipline (the over-claim guard — getting this wrong is the main failure mode):**
- **Only Tier 3 is validated/built/live.** Tiers 1–2 are the next, spec-stage work.
  NEVER state or imply the Poincaré embedding or the hyperbolic generator are
  built/working/validated.
- **Tier 3 is the v98 executor, NOT a perceiver.** The PERCEIVER IS RETIRED (5×
  refuted v118–v121; v300 perceiver-core failed flat at chance). The earlier
  "Mycelium blueprint" put a perceiver in the executor slot — we REPLACED that with
  the validated v98 executor. The three-tier is NOT the perceiver-core.
- **The ANCHOR discipline is WHY Tier 2 is buildable where the perceiver wasn't.** The
  generator initializes to reproduce the validated hard mask, then RELAXES — *learning
  relaxes a known geometry, never discovers one from random.* This neutralizes the
  attention-bootstrap wall that killed the perceiver. (See §5, bootstrap rule.)

**THE DEEP PRIZE:** deduction-depth ↔ radial traversal ↔ breath-count. The breath
cycle becomes a **geodesic engine** — the waist "exhale" drives the representation
inward (abstraction), which auto-widens the attention horizon; the "inhale" descends
to project onto local nodes. Phased `r` roadmap (do NOT bundle): **static global `r`**
(foothold: does ONE field generalize across N=5/6/7) → **monotonic `r_k` per breath**
(the "climb"; continuous form of v100 topological staging) → **`r=f(|z|)`** (horizon a
function of radial position — the climb IS the expansion). KenKen is FLAT (lateral
cliques): the radial-depth bloom (the Tier-1/2 payoff) needs a HIERARCHICAL (DAG)
testbed; a muted KenKen radial signal is the geometry faithfully reflecting a flat
problem, NOT a failure (§5 caveat).

---

## 1. Tier 3 in one paragraph (the validated executor)

A small iterative transformer (4 Pythia-410M L0–L3 layers SHARED across all K
breaths, h=1024, 16 heads, ~32M trainable + ~52M token-embeddings) performs
factor-graph inference by K passes through the same weights. Each breath: add a
per-breath additive marker → 4-layer transformer with a structured per-head attention
mask encoding the factor topology → a learnable per-breath `delta_gate` convex
residual blend → per-breath layernorm + value-codebook readout → per-breath
calibration head. K breaths are JIT-unrolled into one graph. Training: per-breath
weighted CE (`loss = Σ_k (1 + k/(K−1))·CE(logits_k, target)`), the "ladder" that
makes K matter. The current instantiation is the v98 KenKen executor: variable-N
(N∈{5,6,7}, laid on a fixed 7×7 = 49-cell grid) Latin-square+cage CSP, **hard
structured row/col/cage attention masks** (the validated engine — this is the `t=0`
slice Tier 2 anchors to), a per-cage verification inlet, a 7-value codebook readout,
a gold-free convergence instrument (the Property-2 telegraph), `K=16`, plus a
codebook-orthogonality penalty and masked-given self-supervision validated this arc.

---

## 2. Tier 3 components, as-built (the breathing recipe)

KenKen is a direct mirror of the v98 Sudoku paradigm (box→cage). Entry point:
`kenken_breathing_forward` (`mycelium/kenken.py`).

| Component | What it does | Where |
|---|---|---|
| **Iterative shared-weight prefill** | K passes through Pythia L0–L3, SAME weights every breath; 1024d residual is the persistent state | `mycelium/kenken.py` |
| **Per-breath additive marker** | Orthogonal per-breath embedding added to the residual | `breath_embed` in the forward |
| **Structured per-head masks (the engine)** | KenKen: 5 row + 5 col + 5 cage + 1 global head (`_build_kenken_fixed_masks` + per-instance cage clique in `build_kenken_attn_bias`). Hard `{0,−1e4}` bias. This is what works — and the `t=0` anchor for Tier 2. | `mycelium/kenken.py:198,455` |
| **Verification inlet** | Per-cage op-type + log-bucketed target + cage-size features (arithmetic as VERIFICATION, never an op-type mask channel — v100's C2 death) | `build_verification_inlet`, `kenken.py:309` |
| **Per-breath `delta_gate`** | Learnable convex residual blend `x = x_pre + gate_k·(h − x_pre)` | `model.*_delta_gate` |
| **Per-breath calibration head** | Scalar confidence per breath (Dopri5-style error-estimator hook) | `*_calib_head_*` |
| **Value codebook readout** | 7-value codebook; aligned to `state_embed` at init | `value_codebook`, `kenken.py:962` |
| **Convergence instrument (Property 2)** | Min-based gold-free `breath_count_min` (argmin consecutive-belief JSD) + `status_min` (settled=correct-at-settle-breath) + JSD-floor secondary | `convergence_instrument`, `kenken.py:658` |
| **Codebook-orthogonality penalty** | `KENKEN_CODEBOOK_ORTHO` — penalizes off-diagonal cos of row-normalized codebook gram; *rotates* collinear 6↔7 rows apart (a bias/reweight can't) | `kenken.py:851` |
| **Masked-given self-sup** | `KENKEN_MASK_GIVENS_P` — per-step Bernoulli hides givens → forces deeper deduction; eval-off | `scripts/kenken_train.py` |
| **Per-breath weighted CE (the ladder)** | `1 + k/(K−1)` weighting; the reason K matters | trainer |

---

## 3. Empirical status (current)

**The KenKen reframe (Jun 15) — the load-bearing finding.** The famous v98 Sudoku
"97.65% cell / 79% puzzle" is **EASY-only at 43% givens**; on its 33%-givens band
Sudoku collapses to 0.82/0.05 — the SAME cell-high/puzzle-near-zero collapse KenKen
showed. KenKen was measured only at 10–12% givens, on 8× less data + ⅓ the steps. So
the "0 puzzle-acc ceiling" was an **eval-regime artifact, not an architecture wall**.
Fix = a **givens-density curriculum corpus** (bands g40≈0.44 … g10≈0.10, 39,996 train
/ 8,004 test, leak-free by D4-canonical structural signature, depth labels intact;
`scripts/build_kenken_data.py`).

**Property-2 first read (Jun 15) — UNTESTABLE-by-restriction, not NULL.** On the
leak-free settled set (kenken_k8, hard-only K=8): the competence-gated settled set is
depth-narrow {2,3,4} (the model only *solves* shallow puzzles), so the binding read is
UNTESTABLE (restriction-of-range). The full-range companion ρ≈0.5 is a **ceiling
artifact** (`rho_no_ceiling` flips negative; ~46% pin breath=8) — caught by its
control. Analyzer (`scripts/analyze_kenken_property2.py`) patched so restriction-of-
range forces UNTESTABLE (can't fake a null on a compressed axis). `rho_no_ceiling` is
now a **required companion control** for every K-budgeted read.

**Live run (Tier 3).** Cold curriculum + **K=16** + ortho 0.05 + masked-given 0.15 +
v45 reg (`kenken_curric_k16_cont`, warm-resumed from step 2000). Early peek (step
2000): settled set growing (0→65/240) and **deepening** (now depth 5), settle breath
moved to 7–13 (vs the U-curve min ~3–4 at K=8), N=5 settled ρ=0.72 with
`rho_no_ceiling` 0.67 (NOT a ceiling artifact — underpowered but the right
trajectory). Watch: N=6/N=7 settled growth (N=7 was 0), `frac_strict`>0.80 (else K=16
still truncates → K=20), depth-span 4→8.

Detailed v98–v300 empirics (v98 Sudoku 97.65%/79%, v100–v107 number-level, v109pi
K-sweep, v110-step3 easy 0.610/med 0.509/hard 0.399, v112b NEW PROJECT HIGH hard
0.3945, the v118–v121 + v200/v300 perceiver refutations) live in `memory/` + git
history.

---

## 4. Specifications (Tier 3)

- **Init:** Pythia-410M L0–3 (attn + FFN + token embeddings 50304×1024), all 4 layers
  SHARED across K breaths.
- **Dimensions:** h=1024, 16 heads × 64, FFN 4096, vocab 50304. KenKen: 49-cell grid
  (N_max=7), 7-value codebook, K=16, BATCH=8.
- **Params:** ~32M trainable + ~52M token embeddings. The Llama-2048 backbone
  (`KENKEN_BACKBONE=llama`, SmolLM2-1.7B, 512 waist) is import-reachable but the live
  runs use Pythia.
- **Platform:** AMD 7900 XTX, tinygrad, AM driver (Secure Boot off +
  `vm.compact_unevictable_allowed=0`). Ubuntu 24.04. No ROCm/CUDA/PyTorch.

---

## 5. Editing rules (durable, hard-won)

- **No mid-breath token generation.** Reasoning stays in the 1024d residual; tokens
  (if any) generated once at the end. ("had had had" if violated.)
- **Diversity must be structural, not learned.** Row/col/cage masks are
  geometric/structural. Every learned diversity mechanism (scales, soft tokens,
  fingerprints) collapsed to a constant within one epoch.
- **Digit/value-spaced for arithmetic.** Single-cell values; whole-number BPE tokens
  force memorization.
- **Factor per-NODE, not per-EDGE (v112b).** Prefer per-position gating (each position
  gets its own activation in the shared backbone) over pairwise structures (learned
  attention biases, edge tensors). v112b's attention-bias channel REFUSED to engage
  (`bias_scale`≈0); its per-position residual gate became load-bearing. Edges are
  already captured by the binary masks. *(Also why the Tier-2 generator anchors at the
  hard mask and relaxes — §0.)*
- **Attention bootstrap.** New attention/pointer pathways (~30+ positions) don't
  bootstrap from task gradient on diverse data — they need an anchor or direct
  supervision. Codebook selection (≤32-way) bootstraps from task gradient alone. This
  is *why the perceiver failed 5× and v300 failed*, and why the Tier-2 hyperbolic
  generator MUST initialize to the validated hard mask, not random — the anchor IS the
  bootstrap.
- **Property-2 reads:** min-based instrument (not JSD-floor); settled = converged-AND-
  correct; check the settled set's depth spread FIRST (depth-narrow → UNTESTABLE, not
  NULL); `rho_no_ceiling` is a required control; bar = lower-CI ρ>0.30 + perm p<0.01
  (point-ρ≥0.50 = STRONG), NOT raw ρ>0.5.
- **KenKen is flat — reserve the radial-depth verdict for a DAG.** KenKen's lateral
  cliques mean the Tier-1/2 radial bloom can't fully express; a muted KenKen radial
  signal is the geometry reflecting a flat problem, not a manifold failure. The N=5/6/7
  foothold cleanly proves the static-`r` claims; the `r=f(|z|)` prize needs a
  hierarchical (DAG) testbed.
- **Substrate laws (tinygrad + AM driver):** no `dtypes.float32` literal inside the JIT
  step; `scores.clip(-1e4,1e4)`; where()-gated NaN guard (NOT multiply — NaN×0=NaN);
  single-kernel `isfinite`; knobs in the JIT cache key; perf fix #1 (deferred per-step
  sync logging). Hyperbolic-specific: clamp `|z|² ≤ 1−1e-5` and arccosh arg `≥ 1+1e-7`;
  watch boundary gradients (`1/(1−|z|²)` explodes near the boundary). See
  `memory/reference_tinygrad_am_quirks.md`.
- **Bryce wants root-cause perf fixes**, not workarounds, when perf is the bottleneck.

---

## 6. Current work in progress (2026-06-16)

**Two-phase split (locked, and the practical face of the three tiers).** Phase 1 =
structure-finder (for CSPs a FREE deterministic spec-reader → the cage mask; for GSM8K
a learned NL→graph parser, a separate later project) — this is where Tiers 1–2 land.
Phase 2 = the mask-flexible v98 executor (Tier 3; takes topology as a runtime input,
`build_kenken_attn_bias` already does). The perceiver (fused discover-and-execute) is
RETIRED (5× refuted v118–v121; v300 failed flat at chance).

1. **Powering the Property-2 flag (live, Tier 3).** Finish the cold curriculum + K=16
   retrain → bank a *powered* verdict on the leak-free settled set. The K=16-vs-K=20
   question rides on `frac_strict` (the ceiling-pin fraction). Entry points:
   `scripts/kenken_train.py`, `mycelium/kenken.py`, `mycelium/kenken_data.py`,
   `scripts/build_kenken_data.py`, `scripts/analyze_kenken_property2.py`.
2. **Tier 2 — hyperbolic mask generator (`docs/hyperbolic_mask_generator_spec.md`).**
   Replace the hardwired masks with masks generated from continuous Poincaré
   coordinates (one field per relation — row/col/cage; the triangle inequality forbids
   one field for all three), anchored at `t=0` to reproduce the hard mask, then
   relaxed. **SPEC-STAGE — foothold NOT yet built or tested.** Phased `r` (do NOT
   bundle): static `r` foothold (does ONE field generalize across N=5/6/7) → monotonic
   `r_k` (the geodesic "climb"; continuous form of v100 topological staging) →
   `r=f(|z|)` (the waist's inward climb auto-widens the horizon — breath cycle as
   literal radial traversal). The deep prize: deduction-depth ↔ radial traversal ↔
   breath-count. Train-with-the-ball via tangent-space params (standard Adam, no
   Riemannian optimizer) + boundary-gradient guards; **strictly additive, gated behind
   the powered verdict, the v98 hard mask is the permanent fallback** (frozen-off is
   byte-identical). Radial bloom needs a hierarchical (DAG) testbed — KenKen is flat.

---

## 7. What we carry forward / left behind

**Forward (Tier 3):** Pythia-410M L0–3 init · full weight sharing across breaths ·
iterative prefill (residual as persistent state) · per-breath `delta_gate` +
calibration head · per-breath weighted CE ladder · structured per-head masks (the
engine — the `t=0` anchor for Tier 2) · verification inlet · value-codebook readout ·
the min-based convergence instrument · codebook-orthogonality + masked-given mechanisms
· the two-phase split.

**Kept in-tree for reference (not on the live path):** v98 Sudoku core+trainer
(`mycelium/sudoku.py`, `scripts/sudoku_train.py`, `scripts/eval_v98_sudoku.py`,
`scripts/v98_*.sh`) — the paradigm parent + paper's central claim; the named
factor-graph milestones `mycelium/factor_graph_v112b.py` (NEW PROJECT HIGH),
`factor_graph_v110_step3.py` (project all-times), `factor_graph_v106_step3.py` (PUCT,
for when BP improves), `factor_graph_v121.py` (perceiver refutation) + their trainers;
`scripts/setup_am_driver.sh`.

**Left behind (preserved in git history, removed from tree Jun-16):** all v100–v121
residual-stream factor-graph variants beyond the kept milestones; v200/v300
perceiver-core; v105 digit family; the Phase-1 DistilBERT classifier; GSM8K/IB/sudoku-
data tooling; one-off `diag_*` scripts; the deleted briefs (v200/v300/kenken-v300/
phase1-parser). Recover via `git show <snapshot-commit>:<path>` / `git checkout`.

**Long-abandoned:** Controller/Notebook/LookupTable closed loop, π-cycled within-breath
RoPE, sine-modulated temperature, WaistController AR-decode (still import-clean in
`mycelium/breathing.py` for the v98 core, never called by the KenKen path).

---

## 8. Active research threads (sequenced)

1. **Bank the powered Property-2 verdict** (live K=16 retrain, Tier 3) — then decide
   the flag ALIVE / WEAK / NULL / (still) UNTESTABLE, and K=16-vs-K=20.
2. **Tier 2 foothold** (`docs/hyperbolic_mask_generator_spec.md`, §6 here):
   static-`r` replication sanity + N=5/6/7 single-field generalization (frozen) →
   relaxation drift → train-with-the-ball → `r_k` → `r=f(|z|)`. Each gated on the
   prior; the foothold is NOT yet fired.
3. **The radial-depth prize (Tier 1/2 payoff)** — `r=f(|z|)` makes the breath cycle a
   radial-traversal engine; its real verdict needs a hierarchical DAG testbed (KenKen
   is flat).

**Key memory notes:**
- `memory/project_kenken_property2_first_read_untestable.md` — the Jun-15 reframe,
  Property-2 UNTESTABLE-by-restriction, `rho_no_ceiling` control, the retrain spec.
- `memory/project_csp_target_survey_jun14.md` — the canonical KenKen log.
- `memory/feedback_offer_engineering_critique.md` — push back before rubber-stamping.
- `memory/reference_tinygrad_am_quirks.md` — substrate laws.
- `memory/project_v121_perceiver_5x_refuted.md` — why the perceiver is retired (Tier 3
  is the v98 executor, NOT a perceiver).
