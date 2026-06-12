# v200 Brief — Perceiver-Core Breathing Transformer

**Status:** Drafted Jun 10, 2026. Architecture locked. Motivation paragraph (§14)
contains a three-branch placeholder selected by the linear-probe result on the
residual ridge (task #226, in flight). Everything else stands probe-blind.

**Authors:** Bryce + Claude.

**Supersedes** as the canonical v200 spec: `memory/project_v200_perceiver_core_design.md`,
`memory/project_v200_transition_synthesis.md` (both Jun 8). Those become historical;
this brief is what the build is constructed against.

---

## §1A Training contract (applies to Stage 1C+, consolidates Jun 11 discipline)

Stage 1C is the first actual v200 training. The contract below applies from
Stage 1C onward and is structurally enforced by the training loop, not
documentation aspiration. Built top-of-brief so reviewers see the
training discipline before the architecture spec, because the
SmolLM2/Pythia eras repeatedly produced "interesting numbers, unclear
mechanism" results that the gates can't catch retroactively.

### A. Eval contract — what we eval, when, against what

| Spec | Value |
|---|---|
| Held-out split | v107 GSM8K test (matches the cont-control reference); same train/test split the existing v110-step3 chain saturates on |
| Eval cadence | Stage 1C: at step 200 only (smoke). Stage 2: every 500 steps. Stage 3: every 1000 steps. |
| Primary metrics | cell_acc, per-position digit_acc, per-breath CE ladder slope |
| Mechanism metrics | latent JSD per-breath (vs random-init reference, §5), cross-attn entropy per head-group, self-attn entropy per head-group, energy channel ‖Δz_j‖ — all in nats, §7 ε convention |
| Comparison axis | always vs `chain_saturation_at_matched_compute` (§9), never vs `v200_step_0` or against mismatched-arch baselines |
| Output | `.cache/v200_eval/step{N}/eval.json` + provenance sidecar; reference curves loaded from `.cache/v200_smoke/reference_curves/*.npz` per §5 |

### B. Trajectory-shaped pass criteria — shape match, not endpoint match

A Stage 1C smoke PASSES only if ALL of the following trajectory shapes hold:
1. **Loss monotonically decreasing** over the 200-step horizon (smoothed; outlier-tolerant).
2. **Latent JSD trajectory DEPARTS from the random-init reference curve** by step 200, **AND the departure is signed toward LATER freeze-breath**, not earlier. PASS conditions (all required):
   - Spearman correlation of trained vs reference < 0.9 OR max-abs-departure / reference range > 0.3 (the magnitude test, captures "doing something different")
   - AND trained freeze-breath ≥ random-init reference freeze-breath (the direction test, captures "different in the gate-B-relevant direction")
   - AND trained freeze-breath ≥ half-K (the Gate B mechanism test from §8 condition 1)
   "Non-monotone in isolation" is too weak; the reference curve makes magnitude sharp; the direction clause prevents the gate from being satisfied by a faster-collapse-than-random failure mode that's numerically a "departure" but architecturally is the v98-v121 telegraph-thesis failure in new clothing. This is the units-mismatch-class leak from yesterday's gate-B fix, re-surfaced one level up: "departure" must be signed toward the side gate B actually tests.
3. **`‖up_proj‖` (Waist) has moved off zero** by step 200 (norm > 1e-4 OR gradient-norm trajectory shows up_proj actively receiving updates). Catches the "waist never earns gradient" failure mode early.
4. **Per-breath CE ladder slope ≤ -0.05** averaged over the eval batch (per the brief §3 carry-forward — the ladder is the §1 mechanism for why K matters).
5. **Waist alternation effect verified post-training: ADVISORY-pending-recalibration (Jun 11 late evening after #236).** Original spec: pre/post-waist magnitude delta on even breaths > delta on odd breaths by ≥10×. **This metric is dead on bounded-substrate (post-#236) architectures** — with `norm_blend` ensuring every breath ends at per-element ~1 by construction, even/odd magnitude ratio is exactly the signal the norms erase. Sign-alternating even-breath deltas at ratio 0.45 (#236's reading) might already be healthy alternation read by a dead instrument. **The alternation question now lives in DIRECTION and CONTENT, not magnitude.** Recalibrated metric (locked Jun 11, deployed in #237+): **waist contribution norm pre-norm — measure `‖waist(z_w)‖` directly from the waist module's output, before any subsequent normalization, and verify it is non-zero and varies meaningfully across even breaths.** Secondary signal: cosine between consecutive even-breath delta directions (waist should produce direction-varying contributions across breaths). Until the recalibration ships, **C5 is ADVISORY-pending-recalibration** — fails don't bind, the magnitude criterion is known-miscalibrated for the new regime.
6. **Latent magnitude bounded across breaths** (added Jun 11, permanent — caught the §2 pre-norm gap):
   `max_k(‖z_k‖) / min_k(‖z_k‖) < 3.0` across K breaths. This is the one-number verification that pre-norm placement (§2) actually does what it was installed to do, independent of any dynamics metric. Would have caught the original §2 gap at Stage 1B's random-init smoke if it had existed; goes into the permanent criteria set, not just the Stage 1C re-smoke. Failure here = substrate fix didn't land, route to architecture review BEFORE any §15 branch sequence regardless of which other criteria fired.

A failure of ANY trajectory shape fails the smoke. Surface metrics (loss decreasing, no NaN) are necessary but not sufficient.

### C. Per-parameter-group gradient norms — cross-attention named explicitly

Every training step, log per-parameter-group gradient L2 norms for at minimum:
- `backbone_L0_L3` — the Llama layers (frozen-ish or fine-tuning, per §13 default)
- `latent_init` — 32 IB centroids + learned 1024→2048 projection + 0.01·randn jitter
- **`cross_attn` — the READ path (NEW PATHWAY, requires ~500 steps of direct supervision per attention bootstrap principle, CLAUDE.md)**
- `waist_down_proj` and `waist_up_proj` — separately, because up_proj is the zero-init gate whose grad-norm trajectory diagnoses Pass Criterion B.3
- `tree_readout` — the codebook output
- `calib_head` — the Dopri5 error estimator
- `delta_gate` — the static (K,) tensor
- `breath_embed` — per-breath markers

Cross-attention is named explicitly because the v118-v121 refutations showed perceiver-as-add-on doesn't engage. In v200 perceiver IS the core, but the same attention-bootstrap principle applies: this is a NEW attention pathway that needs to earn gradient. The grad-norm trajectory tells us whether READ is learning at all.

Output: `.cache/v200_smoke/grad_norms.npz` (step × group → L2 norm) + provenance.

### E. Reading discipline — locked pre-commitments for the smoke result (added Jun 11)

When the Stage 1C smoke artifacts land, read against these locked
interpretations. All three are pre-commitments; none gets relaxed under
result-pressure once the numbers are in front of us. The week's discipline
was instrument-before-interpretation; the reading rules belong locked at
the same layer.

#### E.1 Near-miss vs criterion failure

Three of the five §1A.B criteria (3, 4, 5) carry magnitude thresholds
inherited from architectures with mature readout paths and trained waists.
At step 200 on a cold-start v200, the *shape* tests matter more than the
*magnitudes*:

- **Criterion 3 (`‖up_proj‖` > 1e-4)** — the failure that matters is
  `up_proj` pinned at zero with **zero gradient flowing to up_proj**. A
  norm of 1e-5 at step 200 with a nonzero, positive grad-norm trajectory
  is "young waist," not "broken waist." Check the grad-norm trajectory
  for `waist_up_proj` specifically (§1A.C); if it's nonzero across
  training steps, the waist is earning gradient even if the weight hasn't
  accumulated yet.
  
  **Post-spec-restore reading (added Jun 11 after #233's 82×/82× equilibrium
  finding):** #233 hit `up_proj` norm = 4.05 at step 200 because the waist
  was being recruited as an emergency 82× compressor against THINK's 82×
  amplification — a learned-normalization-by-balanced-explosions
  equilibrium. With the 4th RMSNorm bounding z, that pressure is removed.
  **A SMALLER `up_proj` norm at step 200 post-fix (e.g., 0.5-2.0) is
  HEALTHIER than broken-run's 4.05**, not weaker. The waist is no longer
  acting as an emergency stabilizer; it's free to do its actual job
  (selective compression of the latent state's commit-phase representation).
  Reading the post-fix `up_proj` norm against the broken-run's value reads
  backwards unless this is pinned now.
- **Criterion 4 (CE ladder slope ≤ -0.05)** — the failure that matters
  is slope ≥ 0 (ladder absent or inverted). A slope of -0.02 at step 200
  is "shallow but present"; the -0.05 threshold was inherited from
  v109+ architectures where the readout was mature. The young latent
  readout may show a shallower ladder without the mechanism being dead.
- **Criterion 5 (even/odd ratio ≥ 10×)** — the failure that matters is
  ratio ≈ 1 (conditional not differentiating breaths). A 6× ratio at
  step 200 is "alternation present, magnitudes young"; the 10× threshold
  assumed `up_proj` had accumulated meaningfully.

**Joint reading: criteria 3+5 together measure waist liveness.** If
both are in the "young but present" zone (3: positive grad-norm
trajectory regardless of weight magnitude; 5: ratio between 3× and
10×), the waist is alive but immature — log a YELLOW pass, continue to
Stage 2, re-check at first 500-step eval. If either is in the "dead"
zone (3: zero gradient flow to up_proj; 5: ratio ≈ 1), the waist is
the problem and routes per §1A.E.2 below.

#### E.2 Failure-pattern routing — debug branches map to patterns, not named criteria

The pre-registered §15 debug branches (A: shared-wv, B: IB init,
C: backbone freeze) were written against **ONE failure mode**: cold-
start stall, which appears as **criteria 1+2 failing jointly** (loss
flat, latent JSD stuck at reference null). Criteria can fail in
patterns the branches don't cover; route the right pattern to the right
intervention:

| Failure pattern | Diagnosis | First move |
|---|---|---|
| **Criteria 1+2 fail jointly** | Cold-start stall; dynamics never depart null | §15 debug branch A (shared-wv) → B (IB init) → C (backbone freeze) |
| **Criteria 3+5 fail jointly** | Waist not earning gradient — mechanism death, NOT stall | Waist init/scale investigation; check `waist_up_proj` grad-norm trajectory specifically; consider Glorot init for `up_proj` (lose the zero-init identity at start in exchange for nonzero grad signal). **NOT branch A.** |
| **Criterion 4 alone** | Ladder mechanism not engaged | Verify per-breath weighted CE is applied (`sum_k (1+k/(K-1))·CE_k`, not argmax_k); check tree codebook readout receives per-breath latent state, not just final |
| **Criterion 1 passes, 2 fails** | Loss going down on something OTHER than the latent state | Check loss flows from tree codebook readout of LATENT STATE, not a shortcut path. Possibly readout learned constant-prediction. |
| **Criteria 2+3 pass, 4+5 fail** | Waist learning, readout not engaging the ladder | Tree codebook + per-breath CE wiring check BEFORE any architectural debug |
| **Single criterion failure (3 alone, 5 alone)** | Component-specific | Route to that component's investigation; do NOT trigger the §15 branch sequence on isolated criterion failures |

Branches map to *patterns*, not *which single criterion was named*. The
§15 branch sequence is reserved for the cold-start stall pattern; other
patterns get component-level investigation first.

#### E.4 Position-collapse disambiguation — pre-registered Jun 11

Stage 1C's corrected JSD reading (#232) surfaced a position-collapse
signature: inter-position cosine `0.152 (diverse at init) → 0.9999998
(collapsed after breath 0)`. But this was measured on the broken
substrate (no §2 RMSNorms; magnitudes 0.11 → 668 per-position; READ
output arithmetically dominating accumulated state). The deflationary
hypothesis: diffuse cross-attention at random-init entropy ≈ log(24)
makes all 32 latents read approximately the same mean-of-tokens
vector. `z_j ≈ small_individual + large_shared_read_ctx` for every
position j, and `cos(a + C, b + C) → 1` when `‖C‖ ≫ ‖a‖, ‖b‖`. The
0.9999998 inter-position cosine is then arithmetic, not consensus
dynamics — same scar as the magnitude/JSD/NaN findings.

Project precedent agrees: v105's mean-field collapse at cos 0.99+ was
**fixed by LayerNorm equalization**. The historical instance of this
exact signature was a normalization problem, not a consensus problem.

The re-smoke's two added metrics make this decidable rather than
re-measurable:

| Reading | Mean-removed inter-position cosine post-fix | ‖read_ctx‖/‖z‖ post-fix | Interpretation | Next move |
|---|---|---|---|---|
| **Arithmetic collapse** | Diversity persists (≤ 0.8 across breaths) | Bounded (e.g., < 5) | The 0.9999998 was shared-additive dominance, dissolves under §2 RMSNorm | No architectural change. v1.1 row 1 stays queued. Re-smoke verdict reads the 5+1 standard criteria. **Strategic note (added Jun 11 after archaeology):** arithmetic-cell determination + surviving position diversity (mean-removed cos near-orthogonal even on broken substrate) reframes the mask question — it's now "does structural assignment beat surviving-but-unstructured diversity" (an upgrade question) rather than "rescue collapsed latents" (a resuscitation question). Cleaner experiment, stronger architectural prior. |
| **Real consensus** | Mean-removed cosine still → 1 after breath 0 | Bounded (READ contribution OK) | Llama self-attn is driving all latents to consensus even with normalized scales; Principle 10 is biting | Promote v1.1 row 1 (per-position routing / topology tensor) to Stage 1B+, with the diagnosis already locked. |
| **READ operator dominance + post-THINK Llama natural growth** (updated Jun 11 evening after #235 + Controls) | Mean-removed cosine partially preserved AND mean per-element growth concentrated by L1 (Llama natural attractor signature: top-10/2048 dim fraction 25-98%) | ‖read_ctx‖/‖z‖ > 10 initially, then concentration-drift becomes the long-horizon signal | READ contribution fix landed in #235 (per-element 0.99 → 1.65 trajectory matched); remaining unbounded growth is Llama's native residual stream pattern — **NOT pathology, NOT OOD by trained drift**. Honest quadrant after Controls 1+2: **natural-growth + concentration-INTERMEDIATE** (not concentration-collapsed). Latents 5× more concentrated than random but 4× LESS concentrated than Llama's native 98%. | §2 follow-up (locked Jun 11 evening as the #236 spec, after Control 1's natural-growth confirmation): **`RMSNorm_blend` (5th seam-bounding norm) before the convex blend.** Pre-blend placement is the actual v105 scar fix: norm-before-blend makes both blend inputs per-element ~1 (scale-matched mix); norm-after-blend launders the mismatch and leaves the gate's effective behavior scale-dependent. Concurrent: log post-THINK top-10 concentration per checkpoint (§7) to answer the slow-drift question. See "Principle: bound the seams, not the organ" in §2. The earlier #235 fix (`α · RMSNorm(read_ctx)`) is independent and remains in place as Seam 2's norm. |
| **(superseded) READ operator dominance** | Mixed; mean-removed cosine partially preserved | ‖read_ctx‖/‖z‖ still > 10 | Substrate fix didn't contain READ's contribution; READ output magnitude doesn't scale to accumulated state | §2 follow-up (was the #235 spec): **`α · RMSNorm(read_ctx)` at the existing READ residual-add site, α a learnable scalar init=1.0.** Normalize, not gate. The single trainable scalar gives training ONE degree of freedom to settle the *average* read-side scale — and if α drifts toward something large or breath-conflicted, that's the legible diagnostic that a per-breath α is needed (queued as §10 row 5). **NOT zero-init gating** — zero-init is bootstrap-safe only for AUXILIARY paths; READ is the information inlet, and α=0 at step 0 creates a v118-v121-mirror gradient bottleneck while the readout entrenches input-independence (modal-collapse attractor) before α can train up. See §1A.E.8 for the portable principle. Why a scalar over a plain RMSNorm: the archaeology shows READ's *useful* scale is breath-dependent (rdr 173 at breath 0 = information delivery into near-empty registers; ~1-13 after); a plain RMSNorm hard-codes scale ~1 at every breath and might under-drive the initial fill. Plain RMSNorm would express the same need as a silently mis-scaled learnable gain — the scalar makes the strain readable. **Random-init signature: C6 green AND rdr at breath 0 in a healthy informative band (1-50), bounded AND informative.** |

Pre-committing this table makes the result reading binding rather than
interpretive. The position-collapse finding becomes a 1-of-3 cell on
the locked grid; whichever cell the re-smoke lands in routes the next
move without retrofit.

**Even-breath cosine creep — pre-committed reading (added late Jun 11
from #233 archaeology):**

The Jun 11 archaeology on #233's persisted z found the mean-removed
inter-position cosine elevates specifically on even (waist-fired)
breaths: k=4 at 0.276, k=6 at 0.299, vs odd breaths at 0.058-0.16. At
200 steps it's modest and the cell-determination as arithmetic stands.
But the structure of the pattern — waist's 512-dim bottleneck plausibly
pulling positions toward shared structure (compression *is* increasing
similarity by definition), and READ/THINK re-diversifying on the rebound
— points to a slow consensus channel that could ramp with training.

**Pre-committed reading for post-fix training trajectories:** if
even-breath mean-removed cosine *climbs* across training steps (e.g.,
0.30 at step 200 → 0.5+ at step 1000+) while odd-breath cosines stay
flat, that's **waist-induced consensus creep** — a real (if slow)
consensus channel hiding inside the arithmetic cell. **Routes to waist
width/gating intervention** (e.g., 512 → 1024 waist width, or per-position
waist scaling), **NOT to structural masks or READ gates.** The cell stays
arithmetic at the READ phase; the consensus emerges at COMMIT.

Cheap to write down now; expensive to notice at step 5000 without the
baseline. If even-breath cosine stays in the 0.05-0.30 band across
training, this finding is decorative; if it climbs, it's load-bearing.

**Cross-cutting prediction added Jun 11** (after the structural-mask gap
discovery): **none of the three cells above clears v200 architecturally**,
because zero structural differentiation in cross-attention is **upstream
of all three cells**. The structural-not-learned diversity law (CLAUDE.md,
v1-v95 validation set) predicts consensus as the attractor regardless of
what the cells reveal about magnitude, consensus, or READ-dominance.
Whichever cell #233 lands in, the next single-variable change is the §2
structural mask requirement landing in code — not topology-routing as
v1.1 lift (its previous slot), but as a §2 prerequisite. The cell still
binds the *additional* fix on top of the masks: arithmetic cell + masks
= probably sufficient; real-consensus cell + masks = also needs
self-attn investigation; READ-dominance cell + masks = also needs the
READ gate. But masks ship first regardless.

### E.4 discipline cut — cells still bind what ships second (added Jun 11)

The "masks ship first regardless" framing must NOT collapse into "the
cell doesn't matter." Cells still bind WHAT SHIPS SECOND, and at least
one cell's measurement keeps independent value even under the
masks-first plan:

- **`‖read_ctx‖/‖z‖` dominance ratio (the READ-dominance cell's
  diagnostic) is orthogonal to who-reads-what.** If the masked
  cross-attention still produces `read_ctx` at 10-100× the normalized
  state, masks alone don't fix scale imbalance between read_ctx and
  accumulated state — that's the v109-style READ gate question
  (zero-init residual blend on the READ output), and it lives at a
  different architectural level than mask topology.

- **Mean-removed inter-position cosine (the real-consensus cell's
  diagnostic) survives masks-first too.** Even with structural masks,
  if the Llama L0-L3 self-attn over diverse-read latents still drives
  them to consensus by mid-K, that's Principle 10 biting at the THINK
  phase, not the READ phase. Masks fix READ-phase identical-input
  symmetry; they don't fix THINK-phase consensus dynamics. The
  mean-removed cosine on a masked re-smoke isolates this.

So the cell-binding under masks is:
- **Arithmetic cell** + masks → probably sufficient; next move is at-spec
  re-run + ship
- **Real-consensus cell** + masks → also needs self-attn investigation
  (THINK-phase intervention, possibly Q rotation or per-breath
  positional encoding on z at THINK input)
- **READ-dominance cell** + masks → also needs READ-side magnitude
  containment via NORMALIZATION-not-gating (RMSNorm on `read_ctx`
  before residual add, or `α·RMSNorm(read_ctx)` with α init=1.0;
  **NOT zero-init gating** — see §1A.E.8 for the auxiliary-path-vs-
  information-inlet distinction the earlier "v109-style" framing
  missed). Independent of masks.
- **Mixed** + masks → component-specific routing per §1A.E.2

The grid was built to disambiguate three mechanisms. Masks address ONE
upstream cause (READ-phase identical-input symmetry). The cells'
remaining diagnostic value is exactly what tells us whether other
upstream causes co-exist.

### E.5 Pre-committed expectation for #233's smoke read (added Jun 11)

Three outcomes for the at-spec K=8/LR=3e-4 attempt, with pre-committed
routing for each:

| Outcome | Reading | Route |
|---|---|---|
| **At-spec K=8/LR=3e-4 survives** | Substrate fix confirmed cleanly; both predictions land (NaN + step cost) | Read all 6 criteria + grid cell at full discipline; bindings apply per §8 |
| **PARTIAL: NaN gone, step cost still high → ADVISORY at K=4** (MOST LIKELY) | Substrate fix lands the NaN prediction; step cost is its own engineering question (FLOPs scale-invariant, 52s/step is JIT/memory pressure). ADVISORY at K=4 with bounded z, healthy gradients, named grid cell is a **SUCCESSFUL smoke for routing purposes** | At-spec K=8 re-run is QUEUED ENGINEERING ITEM (JIT graph reuse, the usual tinygrad knobs), not a blocker on masks shipping. Cell still binds; masks ship next; engineering follow-up parallel |
| **C6 fails: bounded z NOT holding** | RMSNorm placement missed the accumulation point. Substrate fix didn't land. | Routes BACK to §2 (architectural review of RMSNorm placement, not forward to masks). The masks-first plan PAUSES until C6 is green on a §2-corrected substrate. |

Pre-committing these readings prevents the most-likely outcome from
getting misrouted into doubt about the masks plan (it shouldn't), and
prevents the unlikely-but-decisive C6 failure from being read as a
mask-layer signal (it isn't).

The clean line: **C6 green is the gate**. Above C6 green, the cell binds
the additional fix and masks ship next regardless of K/LR ADVISORY
status. Below C6 green, the line stops at §2 and re-routes.

### E.10 Pre-committed masks read for #237 (added Jun 11 late evening)

The substrate arc closes with #236; #237 ships structural masks per §2 (per-latent topology mask, partition 1a: 24 per-token + 4 per-op + 4 global). **Pre-commit the masks read now, before the run, so the artifact comparison is binding:**

**§7 step-0 random-init signature (verification of wiring, not value):**
Per-group cross-attn entropy strictly below each group's log(family_support_size). Mean across all latents strictly below log(24)=3.178. If this fails at step 0, the mask isn't wired — fix and re-run.

**Mid-training reads (verification of value):**

| Read | Expected (masks help thesis) | Expected (masks decorative) | Routing |
|---|---|---|---|
| **Mean-removed inter-position cosine** | Stays diverse (~-0.03 to 0.3, similar to #236) | Stays diverse OR collapses | Diverse = healthy. Collapse = masks broke something; route to component-specific. |
| **Per-group entropy separating across training** | Different mask groups learn different attention sharpness (per-token group ~0, per-op group ~1.5, global ~3) by step 1000+ | All groups stay at log(family) — no specialization | Separation = masks helping; uniform = decoration. |
| **C4 ladder slope** | Slope ≤ -0.05 by step 1000 (the deferred hypothesis: "likely fixed downstream when masks differentiate per-breath state") | Still flat or positive | **C4 greens → ladder story closes**, hypothesis lands cleanly. C4 fails again → **§1A.E.2 component checks stop being deferrable.** Three deferrals across #234/#235/#236 is the limit; fourth deferral becomes the thing we tell ourselves. |
| **Concentration drift (§7 metric, same-site comparison via measurement_site provenance)** | Stays bounded ~4% (registers-not-tokens thesis vindicated) | Climbs toward Llama-native 98% (thesis erosion) | Same-site is mandatory; without it, comparison is muddied. |

**The C4 commitment is the sharpest line:** ladder has failed three runs straight with the pre-registered hypothesis "downstream of differentiated latents." Masks ARE the differentiation. If C4 still fails after masks, the hypothesis is exhausted and §1A.E.2's wiring check fires — no more deferrals. This is the masks run's binding criterion in addition to gate signals.

### E.11 C4 step-1000 branch table + specialization disambiguators (added Jun 11 late evening, #237 in flight at ~step 300 — pinned BEFORE the step-500/1000 reads)

The step-200 read landed in the ambiguous middle (slope +0.0023, less inverted
than #236's +0.0107; THINK entropy 3.37→2.27 with 22× std growth; post-THINK
concentration 0.052→0.144). Two disambiguators and the C4 branch table are
pre-committed now, while the step-500/1000 results don't exist to flavor them.

**Disambiguator 1 — THINK attention vs quotient-graph adjacency** (computed by
`scripts/diag_v237_specialization.py` from dense ckpts + the fixed diag batch):
mean inter-latent attention matrix A (over batch, heads, L0-L3, breaths;
diagonal excluded) vs the static quotient-graph adjacency Q implied by
partition 1a AS BUILT (mod-4 ownership: per-token latent i ↔ per-op latent
24+(i mod 4); global latents 28-31 ↔ all; per-token↔per-token share-a-factor
edges omitted — the mod-4 build carries no factor annotation). Statistic:
Spearman ρ over off-diagonal entries, with a latent-relabeling permutation
null (1000 perms). **Pre-committed bands: ALIGNED = ρ ≥ 0.5 AND above the
99th percentile of the null; NON-TOPOLOGICAL = ρ < 0.2 OR within the null;
0.2 ≤ ρ < 0.5 = AMBIGUOUS, treated as Branch B for cost-ordering.**

**Per-family refinement (pinned late Jun 11, BEFORE any ρ was ever computed
— the CPU validation pass had produced zero output at pin time): the PRIMARY
routing ρ is the PER-TOKEN family's** (rows 0-23 of A vs Q, all key columns,
self excluded), same per-group discipline as the entropy metrics. Global
latents SHOULD attend broadly — their Q rows are all-ones, making Spearman
formally degenerate for them (the math itself says ρ is the wrong lens
there; their read is entropy, and they are reported as structurally
exempt). Folding global rows into one ρ could drag a real per-token
alignment below threshold and misroute Branch C as Branch B. The all-32 ρ
and per-op family ρ are reported as secondary (per-op additionally
non-binding per the mod-4 ADVISORY).

**Disambiguator 2 — top-10 dim-set overlap across latents** (post-THINK, same
site as the §7 concentration metric): per latent, the top-10 energy dims
(mean over batch); statistic = mean pairwise overlap (0-10) across the 32
latents per breath, plus each latent's overlap with the pooled all-latent
top-10. **Pre-committed bands: SINK RECRUITMENT = mean pairwise overlap
≥ 7/10 with stable dim identity across breaths; HEALTHY SPECIALIZATION =
≤ 3/10; middle = MIXED (report which latent families share dims).**
Computed at step 500 — not deferred until the concentration fraction is
ambiguous. E.10 read 4's fraction verdict stands exactly as written; this
disambiguator binds the INTERPRETATION layer (sink-absorption thesis-erosion
vs healthy specialization) when the fraction exceeds its 0.10 threshold —
sharpened attention concentrating flow is the same event as differentiation,
and the fraction alone cannot tell the two stories apart.

**C4 step-1000 branch table (binding; ordered by intervention cost —
component checks are hours, #238 is a build):**

| Branch | Condition at step 1000 | Routing |
|---|---|---|
| **A** | slope ≤ −0.05 | C4 GREEN: ladder engaged, masks carried it. #238 dies cheap; "differentiation fixes C4 downstream" lands as pre-registered. |
| **B** | slope > −0.05 AND THINK attention ALIGNED with quotient graph (or AMBIGUOUS) | Differentiation exists but the ladder doesn't consume it → §1A.E.2 component checks fire IMMEDIATELY (per-breath weighted CE wiring; readout-receives-per-breath-state) BEFORE any architecture move. "The wiring was broken all along" would retroactively explain three runs of flat ladder — check the wiring before building anything. |
| **C** | slope > −0.05 AND THINK attention NON-TOPOLOGICAL (differentiated but orthogonal to topology) | #238 fires as registered (THINK quotient-graph mask, single variable, same partition source). |

**Hypothesis-scope note (the caution that motivated this section):** THINK
entropy differentiating (std ≥ 0.05) while C4 stays flat does NOT kill the
message-passing hypothesis — it SPLITS it: structure is forming, but
topology-aligned vs arbitrary cliques is exactly what Disambiguator 1
answers. The hypothesis as registered binds to C4, not to THINK entropy.
Mean-field at training end (std < 0.05) + flat C4 still routes to #238 as
registered. Self-organized attention that already approximates the quotient
graph = #238 confirmed-by-emergence, mask unnecessary — the best outcome.

### E.12 Fixed-direction waist + the C4-C5 unification test (added Jun 11, #237 at ~step 650 — pinned BEFORE the step-1000 read)

**The finding (step 200/500 checkpoints):** the waist's contribution has
collapsed to a constant — consecutive even-breath direction cosines 1.000,
CoV across even breaths 0.005 → 0.0005 while the contribution norm doubled
(26 → 61.5). The recalibrated C5 metric is WORKING — it caught this; C5's
question shifts from metric to mechanism.

**Uniformity audit on COMMIT as built (the one-line audit prompt applied):**
same W_compress/W_expand every even breath; same input statistics every even
breath (post-seam-norm z, per-element ~1 — the seam norms homogenized what
the waist sees); fixed gate sigmoid(−2) ≈ 0.12. Nothing in its construction
varies per breath except z's content, and z's content is evidently not
steering it. **Design tension, not a bug:** bounding the seams was right and
it costs the waist its differentiation signal. The old architecture's waist
saw wildly varying inputs; v200's inherited the alternation schedule but
lost the input variation. A fixed-direction COMMIT is the structural-
diversity law being violated by an operator nobody re-audited.

**#239 candidate queue (behind the C4 resolution — does NOT jump the line),
in cost order:** (a) breath-conditioned waist — breath_embed into the
compress path, one concat; (b) per-breath gate — the delta_gate archaeology
found the old architecture learned exactly this, with the B7 quant jump;
(c) accept the waist as a global bias and let v1.1 partition work carry
differentiation.

**The unification test — waist-ablation eval (pre-pinned reading):** the
perfectly monotone anti-ladder (per-breath CE rising smoothly 0.9152→0.9244
at step 500) is later breaths being consistently slightly WORSE — readout
seeing progressively staler/more-mixed state. Held against the waist
finding: even breaths inject an identical large vector (norm 61.5, growing)
into every position, twice per cycle, which the readout's RMSNorm then
renormalizes around. A breath-constant additive at ~60× per-element scale
could wash per-breath distinctions out of the readout's view.

Test (`scripts/diag_v237_waist_ablation.py`, eager, no retrain, minutes):
per-breath CE on the eval set with the waist contribution silenced at
inference — gate forced to 0 (norm_commit KEPT; only the directional
contribution removed — the single-variable ablation). Waist-off entirely
(norm_commit also removed) reported alongside as the norm/contribution
split, not the binding read.

**Pre-committed readings (operationalized: INVERTS = gate-0 slope ≤ −0.05;
FLATTENS = baseline slope positive AND gate-0 slope < 0.5× baseline;
otherwise UNCHANGED):**
- Anti-ladder FLATTENS or INVERTS with the waist silenced → C4 and C5 are
  ONE THREAD: the fixed-direction waist is actively masking per-breath
  differentiation. #239 is promoted with a measured mechanism instead of a
  suspicion (still behind the §1A.E.2 wiring checks in execution order —
  cost ordering holds).
- NOTHING changes → the waist is exonerated on C4; the branch table
  proceeds clean.
- **The ablation runs in the SAME pass as the step-1000 read under BOTH
  Branch B and Branch C** (it is five minutes and the only test that can
  unify the run's two anomalies under one mechanism). ρ in the ambiguous
  band (0.2–0.5) routes to component checks FIRST per the E.11 cost
  ordering — unchanged.
- **Branch C + ablation-UNCHANGED (the fourth outcome, pinned before the
  sweep):** the composite hypothesis enters #238 with only ONE of its two
  mechanisms implicated — #238's result is then read as testing "messages
  fix the ladder" ALONE, and #239 is demoted from companion-fix to
  independent question. Attribution rule for when #238 lands: under
  FLATTENS, a C4-green #238 confirms the composite; under UNCHANGED, the
  same green confirms only the message half, and the waist's constancy
  becomes a tolerated oddity rather than a convicted mechanism.

### E.15 C4 RECALIBRATION (pre-registered Jun 12 early, BEFORE any WRITE/#238 data exists — per the rule that the first WRITE run's ladder read must not inherit an asterisk)

The −0.05 nats/breath slope criterion is miscalibrated: the model's total
per-breath CE spread at its best (#237.5) is ~0.035 nats, so the old
criterion demands a slope steeper than the entire observed dynamic range.
#237.5 produced the first negative eval ladders in v200 history (monotone
at 200/500/1000, shallow-U at 1500/2000, levels descending throughout) and
still "failed" C4 by number. Replacement criterion **C4′, shape-based with
a magnitude floor — all three clauses required**:

1. **Shape**: per-breath eval CE non-increasing through at least k = 3
   (monotone descent or U with minimum at k ≥ 3).
2. **Magnitude**: total descent (breath-0 CE − minimum CE) ≥ 0.01 nats.
3. **Tail**: tail rise (final-breath CE − minimum CE) ≤ 50% of total
   descent.

Validation against existing runs (the criterion discriminates correctly
on data that predates it): #237.5@2000 PASSES (descent 0.0269, tail
0.0040 = 15%); #237.5@200 PASSES (monotone, descent 0.0388, tail 0);
#237@1000 FAILS (tail 0.0093 = 66% of descent — erosion winning);
#236 FAILS (anti-ladder, no descent). The old slope is reported alongside
as `ladder_slope_legacy` for cross-run continuity. C4′ binds from the
first #238/WRITE run onward.

### E.16 Depth/size-generalization eval — decomposition's own question (pre-registered Jun 12, #238 in flight, BEFORE any result exists)

**The question:** is the system learning NODE-LEVEL OPERATIONS that compose
(decomposition realized) or GRAPH-LEVEL SOLUTIONS (decomposition merely
represented)? The train/eval ladder divergence smells like the latter; this
eval decides. Highest-information cheap experiment on the board: reads off
existing checkpoints, no architectural change — training tops out at
n_vars=8/n_factors=7 while the architecture's envelope is n_max=16/f_max=8.

**Protocol — AMENDED minutes after pinning, before any data was built
(the original size bins were architecturally impossible: the record's
n_vars counts LEAVES; total variables = leaves + factors, and hard's
8+7=15 already fills n_max=16 — caught by reading the generator, logged
per the amend-the-frame discipline):**
- **(i) Depth-split at fixed size (THE BINDING READ — free, existing test
  set, no new data):** training mixes chain (depth = n_factors, up to 7)
  and tree (depth ~log) topologies 50/50 at every size. Split the
  existing hard test bin by graph depth (longest path, computed from
  factor_args) and read cell_acc + per-breath CE descent per depth bin.
  Component-learner: CE descent persists at depth 7 (chains) ≈ depth 3
  (trees) — refinement composes down long chains. Pattern-memorizer:
  descent collapses with depth at fixed size.
- **(ii) Inversion OOD (exploratory tier — needs a generator variant):**
  observe the final result + all-but-one leaves; query the missing leaf.
  The model has NEVER seen an observed result or run a component
  backward. Above-chance leaf recovery = strong evidence of true
  bidirectional components (the BP framing made falsifiable); chance =
  not damning (never trained). Asymmetric-stakes read, declared as such.
- **(iii) Op-mix OOD (cheap secondary):** homogeneous-op chains
  (all-MUL, all-SUB at hard size) vs training's uniform mix —
  composition-pattern OOD at in-dist size.
Evaluate the #237.5 step-2000 checkpoint (and later #238's) per bin:
cell_acc + per-breath eval CE.

**Pre-registered readings (CE-descent primary — absolute accuracies are
small, the ladder's FUNCTIONING is the robust read):**
- **COMPONENT-LEARNER (graceful):** the C4′-style descent clause persists
  across the depth split (depth-7 chains retain ≥ 50% of depth-≤3 trees'
  descent magnitude) — refinement composes down chains the length of the
  budget itself.
- **PATTERN-MEMORIZER (cliff):** descent at depth 7 < 25% of the
  shallow-bin descent, or absent — refinement only functions on shallow
  compositions.
- Mixed/intermediate: reported as measured; the depth-split descent ratio
  is the binding discriminator. Inversion and op-mix readings are
  secondary/exploratory as declared above.
- NOTE the K=8/depth-7 coincidence: a depth-7 chain needs ≥7 sequential
  resolutions and the budget is 8 breaths — the depth split also probes
  whether the breath loop's SEQUENTIAL capacity is the binding constraint
  (connects to the U-minimum and the per-node budget question, §10 row 7).
  The three-way collision is in frame: problem depth (7 hops) vs breath
  budget (8) vs the erosion ceiling (U-min ~k=3, settled state eroding
  after).

**The depth-gradient discriminator (pinned before data — WHERE the descent
collapses on deep chains, if it collapses):**
- PATTERN-MEMORIZER: collapse roughly UNIFORM across the chain (the graph
  as a whole is OOD-shaped to the solver). Routes to TRAINING DISTRIBUTION.
- BUDGET-STARVED COMPONENT-LEARNER: a GRADIENT along the chain — per-node
  accuracy stratifies by depth-from-evidence (early nodes near observed
  leaves resolve; deep nodes never receive their messages). Routes
  directly to §10 row 7 (allocation), NOT to training data.
- Distinguishable in the per-node logits the sweep already collects.
- **The WRITE wildcard:** slots holding settled early-node state are
  exactly what would let breaths 4-8 spend themselves on the deep
  frontier instead of re-defending the shallow one. If #238's depth
  gradient FLATTENS relative to the #237.5 control, P-W1 and the
  mycelium question answer each other in one diff.

**Inversion prior, refined (pinned with the week's own evidence):** the
architecture was never TRAINED bidirectionally, but the BP framing was
never about training direction — v98's original result was joint MAP
where given cells constrained everything. The honest prior is not
"chance" but "whatever bidirectionality the factor REPRESENTATION
affords": plausibly above chance for single-inversion on shallow graphs,
chance below that. **Third verdict, named so the middle result doesn't
read as mush: landing exactly at that boundary = the representation
carries the components and the solver does not yet exploit them** —
neither component-learner nor pattern-memorizer, but
components-latent-in-representation.

**Routing:** CLIFF → under-decomposition becomes the NAMED bottleneck
behind the transfer split; next-architecture conversation = forcing
component-level learning (per-node supervision, subgraph curricula).
GRACEFUL → components are real; the efficiency story moves to ALLOCATION
(per-node breath budget, §10 row 7) and AMORTIZATION (v300, parked).
Discipline constraint (project law: discovered decomposition beats
designed — v7's IB-vs-handbuilt lesson, this week's self-built notebook):
the move is never imposing component structure; it is testing whether
components were LEARNED, giving the gradient CAPACITY to allocate, and
measuring what it does with it. Provide the affordance; don't hold the
structure.

### E.13 The fifth outcome + U-trajectory readings (pinned Jun 11 latest evening, #237 at ~step 1250 — BEFORE the 1500/2000 numbers were read)

**The branch declares from the ρ TRAJECTORY, not the endpoint.** The branch
table's inputs proved time-varying: per-token ρ went −0.02 (init) → +0.36
ABOVE null (200) → +0.17 within null (500) → +0.22 within null (1000), with
READ entropy re-broadening in lockstep. The architecture transiently
DISCOVERED the quotient graph and walked away from it. A branch declared
from the step-1000 endpoint would conclude "THINK never found topology,
impose it"; the trajectory says topology-alignment was FOUND AND ABANDONED
under the training objective. Those route differently:

- **Sustained-within-null across 1000/1500/2000** → Branch C as registered
  (#238 ships).
- **Transient-then-decaying with re-broadening READ** → FIFTH OUTCOME:
  "topology found and rejected" — routes to WHY before any mask ships. If
  the objective pulls the architecture away from quotient structure,
  masking THINK to the quotient graph is fighting the loss with
  architecture: the mask holds the structure the gradient discards, and
  the gradient finds another path around it.

**Candidate mechanism for the fifth outcome (written now, unflavored):**
early training uses topology as scaffolding to bootstrap reads; then the
waist's consolidation makes per-region reading redundant — the common-mode
carrier IS the shared workspace replacing edge-wise exchange. The
architecture choosing BLACKBOARD over MESSAGE-PASSING: a legitimate
alternative computational scheme, and forcing the quotient mask would be
imposing our BP aesthetics on a system that found something else. The
dim-overlap finding (latent-grown common mode, waist co-aligned, NOT
Llama's sink) is consistent with exactly this. Timing evidence: the
de-alignment (200→500) coincides with the waist waking and the U forming
(500→1000). Cheap pre-#238 evidence the sweep can collect: (i) sustained
vs recovering ρ at 1500/2000; (ii) whether the carrier is load-bearing
under the gate-0 ablation (ablation degrading early-breath CE or
collapsing the U = carrier doing real work); (iii) waist CoV trajectory
continuing to grow. Under the fifth outcome, #238 ships only if the
why-investigation refutes the blackboard story — or as an explicit
test-anyway decision, not as the automatic Branch-C consequence.

**U-curve trajectory readings (pre-pinned; the wiring checks were mandated
for a FLAT ladder — the U gives them a different question: refinement
happens through k≈3, then reverses):**
- **(a) Scrambler-with-delay**: the U's minimum MIGRATES RIGHTWARD at
  1500/2000 (k=3 → k=4-5) as the waist differentiates further — the
  architecture is learning to extend refinement; C4 may green itself with
  more training.
- **(b) Fixed structural ceiling**: the minimum STAYS at k≈3 regardless —
  ~4 breaths is what this loop can refine, and the ladder weights
  (1 + k/(K−1)) pay most for exactly the breaths past the ceiling. The
  cheap intervention is SUPERVISION-SIDE (reweight or truncate-K) before
  any architecture move. NOTE: (b) would be the v98-freeze finding
  re-arriving in the new paradigm — beliefs refined by k≈4, budget wasted
  after — and two architectures hitting the same wall from different
  directions upgrades "intrinsic damping" from a homogeneous-stack
  property to a claim about this problem class.
- **(c) Wiring artifact**: an off-by-one in per-breath CE indexing or
  state-passing surfacing as late-breath degradation. §1A.E.2 exists to
  rule this out and it is checked FIRST — it invalidates readings (a) and
  (b). Standing evidence already on file: the TRAIN-side ladder descends
  monotonically (slope −0.013 to −0.018 at steps 200-250) while the eval
  ladder is flat/U at the same steps — wiring functional, transfer
  failing — but (c) is formally ruled out by inspection, not by this
  observation alone.

**Waist-ablation prior update (pinned with the above):** COMMIT began
differentiating on its own exactly as the U appeared (CoV 0.0005 → 0.0118,
crossing the 0.01 threshold at step 1000; consec direction cosines off
1.000 for the first time). The fixed-direction conviction from step 500 is
already stale. The ablation still runs (only mechanism-splitter), expect
UNCHANGED-or-partial; the fourth-outcome rule stays armed, and the fifth
outcome may supersede #238 entirely. "Branch C → build the mask" is no
longer automatic — held loosely until the 1500/2000 trajectory says
whether the de-alignment was permanent.

**Two readings of the train/eval ladder split (pinned before the sweep;
"wiring functional" is established — "objective functioning as intended"
is still open, and these are distinct):**
- **Reading 1 — memorization**: per-breath refinement overfits; later
  breaths improve on fit data only. Routes to regularization/supervision
  (the ladder's late-breath weights pay for overfit refinement; the
  objective is being gamed).
- **Reading 2 — distribution-specific refinement**: breaths past k≈3 do
  refinement steps tuned to training-graph regularities (op sequences,
  branching factors, depth profiles) that eval graphs distribute
  differently — the ladder WORKS and the training distribution is too
  narrow to generalize from. Routes to DATA (curriculum breadth) before
  supervision or architecture.
- **Discriminating test (cheap, post-sweep)**: split eval per-breath CE by
  structural distance from training (difficulty bucket as the coarse cut;
  depth/op-mix as the fine cut). Late-breath degradation spreading
  UNIFORMLY → Reading 1. Concentrating on structural outliers → Reading 2.

**The carrier is a component, not a watch-item (pinned at overlap 9.99/10,
stability 10.00/10):** every latent shares the same ~10-dim low-amplitude
subspace, grown from latent_init's common mode, waist co-aligned. If
load-bearing (channel ii), v200 has de facto a ~10-dim GLOBAL BUS plus
~2038 dims of per-latent state — an architecture nobody specced, found by
the gradient. Pre-noted consequences: (1) bus CAPACITY is currently an
accidental design parameter (10 dims = what the common mode grew) — is the
U's k≈3 ceiling a bus-bandwidth limit? A wider deliberate bus would test
it. (2) Second time this week the system built its own version of a
mechanism we considered imposing: the 82×/82× equilibrium was
self-normalization before the seam norms; the carrier may be
self-message-passing before the quotient mask. The design question each
time: formalize what it found vs replace with the principled version.
Week's evidence: formalizing (seam norms) beat the found version (82×
balance) once — whether that generalizes to blackboard-vs-quotient-graph
is exactly the fifth-outcome investigation, and genuinely open because the
found solution this time isn't pathological.

**Carrier-projection ablation (the direct blackboard test — pinned as a
sweep condition):** gate-0 silences the waist's CONTRIBUTION, but the
carrier lives in latent_init's common mode, which the waist co-aligned
with, not created — gate-0-UNCHANGED exonerates the waist without testing
the carrier. Direct test: project the shared 10 dims (persisted per
checkpoint in specialization_237.json) out of z.

**Site refinement (pinned before the sweep — the placement embeds WHICH
blackboard reading is tested, and DECORATIVE may only be declared from the
full set):** the boundary placement (init + post-Seam-1 each breath) tests
the carrier as CROSS-BREATH MEMORY only — READ/THINK still write the dims
freely mid-breath. If the blackboard's actual function is within-breath
broadcast (THINK pooling shared state through the dims, consumed
same-breath), the boundary ablation barely touches it and a false
DECORATIVE results. The sweep therefore runs THREE projection conditions:
boundary (memory), post-THINK — clearing the dims before COMMIT/blend each
breath (bus), and both. Four-cell read (per-condition hurt = mean
per-breath ΔCE > +0.01 nats, or early-breath ΔCE > +0.01, or U-range
halved): both-hurt → carrier is MEMORY+BUS; boundary-only → MEMORY;
post-THINK-only → BUS; neither → genuinely DECORATIVE, declared from a
test that covered the hypothesis space. Known limitation, documented: the
post-THINK site cannot separate THINK's broadcast from the waist's own
re-writing of the dims (h_compressed may reintroduce them post-projection).
Cross-step dim-identity check: the modal sets at 1000/2000 are verified
≥8/10 overlapping before cross-step conclusions; drift downgrades loudly
and is itself a finding.

**THE NOTEBOOK HYPOTHESIS (pinned with the prior, before the sweep — full
memo: memory/project_v200_notebook_hypothesis_jun11.md):** WRITE was
deferred to v201, so the latents are both workspace and memory — no
parking for settled state. The U-curve is the workspace-overwrites-memory
signature (refine to k≈3, then mixing erodes what settled); the carrier is
the gradient IMPROVISING a notebook (protected stable common channel) in
the only space available — third self-built mechanism this week.
Predictions attached at pin time: (1) carrier four-cell lands MEMORY or
MEMORY+BUS (boundary projection hurts — clearing the carrier across
breaths destroys the improvised notebook); (2) U-minimum stationary at
k≈3. **Promotion rule: sweep confirms both → WRITE promoted from v201 to
the #238 slot** (never refuted, only deferred); the quotient mask
re-queues behind it under the fifth-outcome investigation. Secondary
placements: 512-waist-in-8×64-slices = #239's concrete form (partition
into COMMIT); π-cycled Q rotation = the reading-(b) intervention (v109pi
9/9 K-sweep history).

**Pre-sweep prior (Bryce, stated on the record BEFORE the 1500/2000 reads,
so the sweep's result is legible against something):** most likely
composite landing given the trajectory — fifth outcome or sustained-C;
carrier MEMORY-or-BUS rather than decorative (9.99/10 totality is hard to
square with no function); gate-0 UNCHANGED-or-partial; U-minimum stationary
at k≈3. Assembled: *the architecture rejected edge-wise topology in favor
of a bus-mediated blackboard, refinement saturates at the bus's reach, and
the next move is formalize-vs-replace on the carrier rather than #238 as
registered.* The sweep confirms or breaks this prior; either way the
surprise is measured against it. READING ORDER IS BINDING: ρ trajectory
first — the fifth outcome supersedes Branch C's automatic #238, and
everything downstream re-frames under it.

### E.14 The gradient-starvation finding + the #237.5 gate (pinned during #237's final 300 steps, BEFORE the sweep ran)

**Finding (from the per-group grad norms + §1A.E.9 within-breath taps):**
post-THINK per-element scale grew 0.97 → 147 (step 200) → 2,449 (500) →
10,765 (1000) — four orders of magnitude — while every seam clamped the
visible state to ~1 and C6 stayed green (the criteria watch the seams'
OUTPUTS). Seam-3's RMSNorm backward divides incoming gradient by the input
scale; at 10⁴ everything upstream of Seam 3 lost its learning signal and
the fp16 inter-breath chain turned small into EXACT ZERO, ordered by
upstream depth: breath_embed + alpha_read at ~8e-7 by step 500, backbone
over the cliff at step 600. Gates innocent (0.108-0.131 throughout). The
organ has been FROZEN since ~step 600; everything after — the U, the waist
reorganization, eval waking at 1500 — is plumbing-only training.

**The corollary (tripwire-grade):** scale-invariant forward +
scale-dependent backward = THE LOSS CANNOT SEE WHAT THE GRADIENT PAYS FOR.
RMSNorm makes pre-norm amplitude a flat direction of the loss — free to
grow — while the same norm's backward taxes every upstream parameter by
1/scale. The seam norms bought a measurable substrate and silently sold
the backward.

**New tripwire (general form, from the 147-unread echo of tripwire 4):**
any quantity declared informational gets a DRIFT ALARM anyway — "not a
criterion" means it doesn't bind, not that it isn't watched. The §1A.E.9
post-THINK value left its stated 4-50 band at the FIRST checkpoint (147,
3×) and was logged unread because the band was non-binding.

**Sweep caption rule — EVERY CELL READ TWICE:** as-measured, and
EXPOSURE-ANNOTATED (gradients-alive window steps 0-600 vs frozen-organ
window 600-2000). Specifics: the carrier FORMED while gradients were
alive (step-200 spike) but its totality consolidated frozen — a MEMORY
verdict says "the improvised notebook is load-bearing FOR THE
PLUMBING-ONLY SYSTEM"; whether a live organ builds one is open. The
notebook promotion rule is amended: WRITE promoted to the #238 slot
CONTINGENT ON REPRODUCING UNDER FIXED BACKWARD. The U's two stories
(workspace-overwrites-memory vs frozen-mixer-erases-refinement) are now
confounded; only the fixed re-run separates them.

**The #237.5 fix (pinned; spec-restoration class — substrate-before-
branches, single variable):** backward-side only, respecting "bound the
seams, not the organ" on BOTH passes: (1) detached-scale seam norms —
normalize by a DETACHED scale (x / ‖x‖.detach() · gain) so the forward is
IDENTICAL and the backward stops dividing by the scale it cannot control;
(2) fp32 the inter-breath gradient chain (kills the exact-zero cliff);
(3) NaN-guard repair (added post-cascade): the multiply-gated update
(`update × healthy`) does not stop NaN — NaN×0=NaN; the #237 cascade
poisoned Adam moments and weights straight through it. Replace with
`healthy.where(update, 0)` semantics. All three are substrate
restoration; none is an experimental variable. **#237.5 starts from CLEAN
optimizer state — fresh Adam moments, no resume** (implied by
from-scratch; explicit because moments contaminated at 1828 would carry
the detonation forward invisibly).

**Notebook-pin scoring rule (added with the 1750 terminal point):** the
U-minimum read at 1750 is the BEST AVAILABLE, not the settling read — the
pin said "the 2000 read settles it" and 1750 sits mid-reorganization
(waist dissolving its fixed direction, eval mid-wake). The notebook
prediction scores PROVISIONAL either way at 1750; #237.5's clean run
provides the genuine endpoint. Better an honest provisional than a
verdict from a checkpoint chosen by a NaN.

**Corrected-ρ branch (pinned while the corrected sweep runs — the diag-side
ρ trajectory was measured on a WAISTLESS forward, tripwire-9 recurrence in
the tooling; the transient-topology story is UNVERIFIED until the corrected
trace lands):**
- Corrected ρ CONFIRMS the transient (peak-above-null near 200, decay into
  null) → the fifth-outcome rule proceeds as written; the
  decay-front-vs-freeze-ordering correlation reads as planned.
- Corrected ρ FLAT-OR-DIFFERENT → the declaration DROPS the scaffolding
  clause; the blackboard hypothesis loses its scaffolding leg (carrier
  evidence stands on driver-side data); the fifth outcome is mooted and
  Branch C stands on the endpoint alone; #237.5's prediction 2 re-scopes
  to whatever the corrected trajectory shows.
- Epistemic state until then: tonight has ONE fully verified structural
  narrative (the freeze — measured in-driver) and one PENDING (the
  topology transient).
- The waistless sweep is PRESERVED as an accidental no-waist ablation
  (`specialization_237_waistless.json`). Comparison line owed when both
  traces exist: if the waistless ρ shows STRONGER topology alignment than
  the corrected one, the waist's consolidation is actively erasing edge
  structure — the scrambler conviction returning by another door.
- Strong-form fix rule (probes self-identify): diag outputs now carry
  `measured_config` (incl. stage2a_waist) in the JSON and provenance —
  binding-time bugs become self-identifying rather than retroactively
  suspect.
REJECTED alternatives, with reasons: weight decay on the Llama interior
(normalizing the organ by another name — re-fights Control 1's finding
that the organ breathes large natively); pre-norm magnitude regularizer
(gentler but still dictates the organ's amplitude).

**Pre-registered predictions for the #237.5 run (same 2000 steps, same
seed, same everything else):** (1) breath_embed + alpha_read grad norms
hold above 1e-5 through step 2000 — READ IN-RUN at the grad captures; (2)
**THE RUN'S BINDING READ (carries the weight C4 carried in #237, post-#237
declaration):** the ρ trajectory sustains past step 300 on the live organ.
The 100-step simultaneity of ρ-collapse and gradient-cliff in #237 is
consistent with starvation-causes-dissolution but DOES NOT strictly order
them — #237.5 is the CAUSAL test, not just the fix: backward holds and ρ
sustains → the arrow is proven; ρ collapses anyway on a live organ → the
objective-rejection reading returns from the dead. (3) SHARPENED
post-declaration (the #237 carrier went MEMORY@1000 → DECORATIVE@1750 —
unprotected memory eroding under writes and plumbing drift, one
consistent story): the question is no longer churn-vs-stability but
**does a live organ REBUILD the carrier, and does it dissolve again on
the same timescale? Recurrent form-and-dissolve = near-conclusive that
the architecture wants WRITE and cannot hold it.**

**The bus cell as WRITE's design brief (post-declaration recognition):**
bus-projection HELPING at step 1000 (ΔCE −0.0179) means THINK's fresh
writes were actively poisoning the persisted state — the workspace
literally overwriting memory at the dim level. Settled state needs
protection FROM THE OPERATOR THAT SETTLES IT. When WRITE ships, the
slot buffer's write-once semantics are not a nicety; they are the thing
the gradient demonstrably could not build for itself.

**Exposure-tag semantics (pinned before the declaration): FORMED-FROZEN ≠
ARTIFACT.** The frozen window is a real optimization regime — plumbing-
only training is a CONSTRAINED system, not a broken one, and its phenomena
(U, waist wake, transfer onset) are genuine solutions under that
constraint. The tag reads "conditional on organ-frozen — reproduce or
revise under #237.5," never "discard."

**NaN-cascade contingency (pinned at skip 6, step 1833, run in flight):**
consecutive NaN-skips began at step 1828 — the trapdoor's FORWARD symptom
(post-THINK ~10⁴ per-element vs fp16 ceiling 65,504) arriving 90 minutes
after its backward symptom was diagnosed. The run completes per
evaluation-horizon discipline (the cascade's length/recovery is itself
trapdoor data). If the step-2000 state is NaN-poisoned (checked by ckpt
finiteness): terminal analysis point = step-1750 ckpt (last clean dense
ckpt) + step-1500 checkpoint (last clean instrumented read); the ablation
runs on 1000 + 1750; the 2000 entries are exposure-tagged NaN-CASCADE and
excluded from cells. If the guard gated updates (finite 2000 ckpt =
step-1827-equivalent weights), the final reads stand with a
cascade-window annotation.

**Routing:** the sweep runs TONIGHT in full before the fix ships — the
frozen-organ run is accidentally a CLEAN ABLATION OF ORGAN PLASTICITY
(plumbing-only training produced transfer onset, a U, a waking waist, a
total carrier; the diff against #237.5 is the measured contribution of
organ learning to each, for free). #237.5 re-runs the same 2000 steps
under the fixed backward; the cells that survive BOTH regimes are what
#238/#239/WRITE get built on. The queue holds; it gained a gate.

**Decoupling note (for the run memo):** between steps 200 and 500, THINK
attention kept sharpening (2.27 → 1.86 nats) while post-THINK channel
concentration FELL (14.4% → 6.7%, back to the dispersed band) — attention-
flow concentration and activation concentration are decoupled in the
bounded-latent regime. Second distinct piece of evidence (after #236's 4%
finding) that latent registers run Llama's machinery in a statistically
different mode than token streams. If the 0/200/500 dim-overlap sequence
shows sink dims briefly grabbing the latents at 200 before training
pressure pulled them back out, the bounded regime is RESTORATIVE under
perturbation, not merely stable — a dynamics observation, one line in the
eventual memo.

### E.6 #234-specific reading expectations (added Jun 11 evening)

**C6 is the decisive number for #234.** Everything else in the artifact
set is interpretation conditional on substrate; bounded z is the gate.
If C6 lands inside threshold (max/min < 3):
- **C2's freeze-breath becomes meaningful for the first time.** Every
  prior measurement was on exploding or oscillating substrate; this is
  the first look at v200's *actual* latent dynamics.
- **C4 either self-resolves** (readout differentiates breaths once
  latents differ meaningfully) **or routes to its cheap component
  checks** per §1A.E.2 (per-breath weighted CE wiring, readout receives
  per-breath state).
- **Dominance ratio reads clean** and closes the §1A.E.4 cell file.
- **Masks unpause** as the next single-variable change.

**C2 expected weakness — pre-committed reading for a from-scratch
200-step run on just-stabilized substrate:** the honest prior is that
trained dynamics at step 200 look a lot like the regenerated random-init
reference, because almost nothing has been learned yet. **C2 passing
its direction clause WEAKLY (small departure, freeze-breath near
reference) is the EXPECTED HEALTHY outcome, not a disappointment.** The
dynamics question that Gate B actually cares about — non-monotone
structure, telegraph signatures — belongs to Stage 2's longer horizon.
The smoke's job is substrate, liveness, and instrument integrity; grade
it on those.

**C5 reading note (cross-references §1A.E.1's post-spec-restore note):**
W_expand norm growing slower than #233's 4.05-by-200-steps is HEALTHIER,
not weaker. Same logic — the broken-run's 4.05 was emergency-compression
recruitment against the 82×/82× equilibrium; with z bounded, that
pressure is removed and the waist is free to learn its actual job.

### E.7 If C6 fails AGAIN — deeper §2 review (pre-committed routing)

With four RMSNorms in place (`breath`, `read`, `commit`, `readout`) and
the breath boundary explicitly bounded, a persisting C6 failure would
have to live **inside a single THINK forward** — the only architecturally
unbounded path remaining. The review question becomes: **why is the
Llama L0-L3 self-attention on latents producing unbounded output within
one forward pass?**

The pre-committed candidates for this deeper review (not all are
mutually exclusive):

1. **READ-side magnitude containment via normalization-not-gating
   (added Jun 11 evening after #234).** The §1A.E.4 cell binding lands
   READ dominance when `‖read_ctx‖/‖z‖ > 10` post-substrate-fix. The
   bootstrap-safe fix is RMSNorm on `read_ctx` before residual add, OR
   `α·RMSNorm(read_ctx)` with α init = 1.0 — **NOT α zero-init.**
   See §E.8 "Bootstrap safety — auxiliary vs information-inlet gates"
   below for the principle this rests on.
2. **Latent self-attention scores saturating.** `clip(-1e4, 1e4)` in
   tinygrad mitigates pre-softmax overflow but doesn't bound post-softmax
   residual magnitudes when V values are large. Check attention weights
   at random-init under bounded z; if scores routinely hit clip
   thresholds, the issue is V scale.
3. **Llama's residual stream assumes a "natural" scale.** Llama was
   trained on full token sequences with positional info; running L0-L3
   on 32 unembedded latents may violate the architecture's implicit
   scale assumption. Diagnostic: compare Llama L0-L3 forward on
   bounded latents vs Llama L0-L3 forward on actual token embeddings
   of the same scale.
4. **RoPE not applied to latents.** The CrossAttention docstring notes
   "No RoPE on cross-attention (latents have no positional semantics)";
   the same applies to Llama's self-attention on latents. Llama's
   internal pre-norms expect post-RoPE scale; absent RoPE, residual
   amplification could differ. Add per-latent learned positional or
   per-breath RoPE on latents and re-measure.
5. **Replace Llama L0-L3 with fresh-init self-attention.** §13 design
   question 2 listed this as a v1.1 variant; promotes to §2-blocker if
   the Llama-shared path is fundamentally incompatible with latent
   operation.

#234's cell binding pointed at candidate 1 (READ-side magnitude
containment) — the rdr trajectory `[173, 17, 0.92, 13, 0.93, 13, 0.93, 13]`
shows READ contribution is per-element-scale-fixed regardless of z_pre
magnitude, which is the magnitude-mismatch signature at a level the
substrate norms don't address. Candidates 2-5 remain queued if
candidate 1 doesn't close C6.

### E.9 Quantitative prediction discipline for #235 (added Jun 11 evening)

The week's standard applied forward: **ship the fix with its predicted
numbers attached, so the artifact read is comparison, not
interpretation.** For #235 specifically:

The archaeology's arithmetic predicts per-element scale at four
within-breath checkpoints:

| Checkpoint | Predicted per-element scale | Criterion? | Mechanism |
|---|---|---|---|
| post-`norm_breath` (Seam 1) | ~1.0 | YES | RMSNorm output (gain init=1) |
| post-READ-add (Seam 2, after `z = z + α · RMSNorm(read_ctx)` with α=1) | ~2.0 | YES | Bounded `read_ctx` adds per-element ~1 to z's per-element ~1 |
| post-THINK (after Llama L0-L3) | **~4-50 (Llama natural attractor)** | **NOT a criterion** | Recalibrated Jun 11 evening: Control 1 measured init Llama hits attractor ~4.24 per-element on real tokens; trained Llama may extend to 5-50. The organ breathes large by design. **Pre-norm prediction (3.5) was miscalibrated.** Logged for diagnostic; not gated on. |
| post-`norm_blend` (Seam 3) | ~1.0 | YES | RMSNorm output (gain init=1) |
| post-blend (after `z_pre + gate·(z_seam3 - z_pre)`) | ~1.0 | YES | `z_pre ≈ 1` (Seam 1 normalized at breath start), `z_seam3 ≈ 1` (Seam 3 norm), blend = 1 + 0.119·(1 - 1) = 1.0 — directional mixing only, no magnitude growth |

**The smoke logs per-element scale at all four checkpoints** to make
the prediction verifiable. Required output in `step200_eval_*.json`:

```json
{
  "predicted_trajectory_per_elem": [1.0, 2.0, 3.5, 1.6],
  "measured_trajectory_per_elem":  [..., ..., ..., ...],
  "trajectory_match": true|false,
  "trajectory_deviation_breath": "post-X" (if mismatch)
}
```

**Reading discipline:** If the measured trajectory lands at or within
30% of predicted, C6 green is **explained** rather than merely
achieved. If the measured trajectory misses (say post-THINK hits 20),
the deviation **localizes the surprise to one specific stage** — and
the next investigation goes to THINK's compounding (per §1A.E.7
candidates 2-5) rather than the READ-side fix being suspect.

Predicting before measuring is the week's standard becoming permanent:
ship a fix, name what should happen, and the artifact tells you
whether the mechanism story holds.

### E.8 Bootstrap safety — auxiliary path vs information inlet (added Jun 11 evening)

**Portable principle, added to CLAUDE.md tripwire array:**

> **Zero-init is for auxiliary paths only. Information inlets (the
> only channel through which problem signal enters a representation)
> must be normalized, not gated.**

The v98–v121 stack used zero-init in several places — `up_proj` in the
waist, the readout projection, the topology bias channel — and the
discipline held. Every instance gated an AUXILIARY path: a route the
architecture could function without at step 0, where opening the gate
became a refinement learnable from task gradient flowing through the
already-working baseline.

READ is not auxiliary. READ is the only path through which factor
graph information reaches the latents. At α=0 on the READ residual,
the latents never see the factor graph at step 0; the readout predicts
from input-blind registers; the only training signal for opening α is
`∂L/∂α` as a scalar bottleneck. Meanwhile the readout finds the
marginal-frequency solution (the modal-collapse-attractor scar), loss
decreases on input-independent output, and α may sit near zero for
hundreds of steps while the model entrenches input-independence.

That's the **v118–v121 failure mode in mirror image**: there, the
perceiver was bypassable, so gradient never recruited it. Here, the
information inlet is gated shut, and gradient must fight a
working-without-it equilibrium to open it. Both come from the same
principle: gradient does not magically find paths it can avoid using
when there's a working alternative.

**Random-init signature for the principle:** any architectural fix that
puts a learnable scalar gate on a residual path should test, at random
init, whether the gated path is auxiliary (architecture works without
it, downstream metrics depend on it being open) or an information
inlet (downstream metrics are uninformative without it). If
information inlet: gate init must be open (e.g., α=1) AND the
magnitude bound comes from normalization, not scalar attenuation.

This principle moves to CLAUDE.md as a portable-principle entry with
the auxiliary-vs-inlet test as the deployable check.

#### E.3 What "STAGE 1C SMOKE PASSED" actually means

A green Stage 1C smoke (all 5 criteria GREEN under the §1A.E.1 near-
miss discipline) buys **exactly one thing: the right to spend GPU-days
on Stage 2**. It does NOT certify:
- That gate B is satisfied — gate B's distinguishability test fires at
  the first real eval checkpoint per §1A.A, against persisted reference
  curves (§5), with the week's ε conventions (§7). Not at step 200.
- That the trained latent JSD shows the non-monotone structure gate B
  wants — 200 steps of departure-from-null is "alive," not
  "interesting."
- That the per-component dynamics are healthy — gradient-norm
  trajectories may be flat or pathological under conditions where the
  five smoke criteria all happen to pass.

The first genuinely informative dynamics read happens at the Stage 2
500-step eval, against reference curves, with full ε conventions.
Until then, a passing smoke is permission to continue, not evidence of
success. The week's pattern — instrument-before-interpretation — applies
to results, not just measurements: a smoke result that "looks great"
buys GPU-days, period.

---

### D. Cont-control structurally enforced

The training loop emits the cont-control comparison data, not the eval script.

Concretely:
- Every eval at step N, the loop also computes the `chain_saturation_at_matched_compute` reference for step N (held constant for cold-start variants per §9; loaded from `memory/feedback_cont_control_protocol.md`).
- The eval output records `metric_minus_chain_saturation` alongside every primary and mechanism metric.
- The training driver REFUSES to emit a "v200 lifts metric X by Y" claim unless the cont-control delta is included with the same significance threshold (drift floor ±0.02 on hard).
- Any deviation from cont-control reporting is an eval-script bug, not a methodology debate.

This is what makes Jun 9's controls audit (`memory/project_jun9_controls_audit.md`) un-repeatable. The audit found SBP/MC-BP/v112b reduced to null retroactively because the warm-start baselines weren't structurally enforced. v200 doesn't get the same forgiveness; the loop emits the comparison, period.

---



The residual-stream-primary paradigm (v98–v121) treated 24 factor graph tokens
as live state in a 1024-d Pythia residual stream, iterated K passes through
shared L0–L3 layers, and read out beliefs through a tree codebook. Five attempts
to add a perceiver as a routing layer (v118–v121) refuted that addition as
redundant with full attention at T=24. v200 inverts the architecture: **32
learned latents at 2048-d (Llama-1B hidden) are the primary state**, factor
graph tokens become static reference, and each breath cycles READ (latents
cross-attend to tokens) → THINK (latents self-attend through shared layers) →
COMMIT (waist compression on alternating breaths) → WRITE (optional latent →
token update). The perceiver is not an accessory; it is the core computation
loop. The motivation paragraph (§14) routes on the probe; the architecture
does not.

---

## §2 Architecture spec (locked)

```
STATIC: factor graph tokens (Llama L0-L3 embed once, never iterated)
        fg_tokens: (B, T, H=2048)

DYNAMIC: 32 latents as primary state
         latents: (B, L=32, H=2048)
         initialization: 32 IB centroids from Pythia-extracted clusters,
                         projected to 2048-d via learned linear

For breath k in [0, K-1], with K=8 initial:
  z_pre = z                                            # capture for blend
  z = RMSNorm_breath(z)                               # SEAM 1: boundary  ← REQUIRED
  z = z + breath_embed[k]
  READ:    z_q  = RMSNorm_read(z)                    # inside-op pre-norm on Q  ← REQUIRED
           read_ctx = cross_attend(Q=z_q, K=fg_tokens, V=fg_tokens,
                                    mask=latent_topology_mask)   # structural per-latent mask  ← REQUIRED
           z = z + α_read · RMSNorm_read_ctx(read_ctx) # SEAM 2: READ-add  ← REQUIRED (α init=1.0)
  THINK:   for layer in shared_layers (4 layers, Llama L0-L3):
             z = layer(z)         # Let the organ breathe; no norms inside L0-L3
                                  # Llama's native residual stream locks at its trained attractor (~4-50 per-element)
  COMMIT:  if k % 2 == 0:
             z_w = RMSNorm_commit(z)                  # inside-op pre-norm on waist input  ← REQUIRED
             z = z + waist(z_w)                       # 2048 → 512 → 2048 (4× compression)
  WRITE:   (deferred to v201 unless §13 question resolves to include)
  BLEND:   z = RMSNorm_blend(z)                       # SEAM 3: blend-input  ← REQUIRED
           gate_k = sigmoid(delta_gate[k])             # delta_gate init = -2.0  ← REQUIRED
           z = z_pre + gate_k * (z - z_pre)            # convex blend on scale-matched states

READOUT (after K breaths):
  tree_logits = tree_readout(RMSNorm_readout(z))      # inside-op pre-norm at readout  ← REQUIRED
```

### Principle: bound the seams, not the organ (locked Jun 11 evening after #235 + Control 1)

The latent loop has **three SEAM-bounding norms** plus **three INSIDE-OPERATION pre-norms**:

| Norm | Type | Purpose |
|---|---|---|
| `RMSNorm_breath` | **Seam 1: boundary** | Bounds accumulated state from previous breath before this breath's contributions |
| `RMSNorm_read` | Inside-op (Llama-style) | Q pre-norm for cross-attention (standard transformer convention) |
| `RMSNorm_read_ctx` | **Seam 2: READ-add** | Bounds fresh READ contribution before residual add into accumulated state |
| `RMSNorm_commit` | Inside-op (Llama-style) | Waist input pre-norm (standard pre-norm convention) |
| `RMSNorm_blend` | **Seam 3: blend-input** | Bounds breath body output before scale-matched mixing with accumulated state |
| `RMSNorm_readout` | Inside-op (Llama-style) | Final readout pre-norm |

The principle that justifies the count (and prevents "five norms is too many" from becoming an architectural smell): **every point where accumulated state meets fresh contribution gets normed; nothing else does.** Three such points exist (boundary, READ-add, blend-input). The other three norms are standard pre-norm conventions matching Llama's internals.

The principle **prohibits** norms inside L0–L3 self-attention. The organ breathes large because it was trained to. Control 1 (Jun 11 evening) measured Llama L0-L3's native dynamics: per-element grows 0.02 → 4.24 across L0-L3 on real token embeddings at init, locks at attractor ~4.24 from L1 onward. Adding norms inside L0-L3 would fight that. Bounding the seams instead lets Llama do what it does while keeping the LOOP's measurable.

Why bound the seams and not the organ? **Registers stay bounded** = latent dynamics measurable = Gate B's distinguishability test has a substrate to read. The alternative architecture ("embrace residual stream growth, bound only at readout") is the homogeneous v98-v121 stack — the paradigm v200 inverted. v200's bet is registers-not-tokens; bounded-but-directionally-varying latents are what makes that bet testable.

### Pre-norm placement on the latent loop (added Jun 11; expanded to 4 RMSNorms after #233)

Four RMSNorm modules are REQUIRED on the latent loop:

1. **`RMSNorm_breath`** at the start of each breath body — contains
   inter-breath residual accumulation. **The component that #233 revealed
   was missing** (z oscillated 0.77 → 232 → 19160 → 232 → 19160 across K=8
   because the breath body produced unbounded contributions and nothing
   bounded the accumulated state between breaths).
2. **`RMSNorm_read`** before cross-attention Q projection (each breath).
3. **`RMSNorm_commit`** before the waist input (alternating breaths).
4. **`RMSNorm_readout`** before the tree codebook readout (once at end).

Each has its own learnable gain (per Llama RMSNorm convention). All four
initialize at gain=1.0 (identity scale).

**Why required:** Stage 1C archaeology (Jun 11) found that without per-breath
normalization, the latent stream accumulates unboundedly across breaths.
Trained-200-steps z magnitudes grew `‖z_0‖=0.87 → ‖z_4‖=5345` (6000×), and
the apparent "fixed-point collapse" (latent JSD ≈ 0 from k=1) plus the
"cos≈1 ridge" (consecutive cos 0.995-0.999 from k=1) plus the NaN at
spec LR=3e-4 all dissolve into one mechanism: **unnormalized residual
accumulation**. Bounded per-breath deltas added to an exploding running sum
produce arithmetically-forced cos≈1 (the v98 housekeeping pattern, but
generated by scale not dynamics), softmax saturation in the JSD metric
(distribution shape pins to dominant coordinate when scale explodes), and
gradient/numerics failure at higher LR.

**The principle inherited:** v105's magnitude-mismatch/LN-equalization
result (`[[magnitude-mismatch-equalization]]`) is on the portable principles
list at CLAUDE.md precisely because unequalized residual magnitudes poison
readouts and conditioning. The homogeneous Pythia stack inherited pre-norm
for free; the v200 perceiver loop is new plumbing and was shipped without
inheriting LN. This brief now closes that gap explicitly.

**This was a brief specification gap, not a Sonnet implementation skip.** The
prior version of §2 didn't specify per-breath normalization, and Sonnet
correctly implemented to spec. Documenting now so the next implementation
inherits the requirement, not the gap.

### `delta_gate` initialization — spec restoration from documented finding (added Jun 11 after #233)

**Required:** `delta_gate` is a learnable `(K=8,)` tensor with init value
`-2.0` (passed through sigmoid each breath: `gate_k = sigmoid(delta_gate[k])`,
so initial alpha ≈ sigmoid(-2) = 0.119).

**Spec-restoration, not new experimental variable.** The v200-Pythia
cold-start debugging documented this exactly: gate init = `-5.0`
(α = 0.007) starved gradient; gate init = `-2.0` (α = 0.119) produced
240× faster growth of training-time signal. The finding had a measured
optimum on record. The Jun 11 v200 §2 shipped with gate init = `-5.0`
re-inheriting the starved-gradient configuration — the fifth brief→spec
gap of the day (after RMSNorm absence, single-forward divergence,
structural mask absence, and the 1:1 partition default). This one had a
*measured fix* on record, not just a principle.

Adding to the principles-with-random-init-signatures inventory: every
gate-like parameter (delta_gate, any future per-breath alpha, attention
masking gates) must have its initialization specified with a
sigmoid-output target that argues for the chosen value. Random-init
assertion: `sigmoid(gate_init) > 0.05` for any new gated pathway unless
the spec explicitly defends a smaller value.

### Per-latent structural cross-attention mask (added Jun 11 — the project's oldest law applied to the new architecture)

**Required:** `cross_attend` takes a binary mask `latent_topology_mask` of
shape `(L=32, T=24)` that partitions the 32 latents across the factor
graph's structure. Each latent reads only its assigned slice of tokens.
**No latent reads all tokens unless it's a designated `global` latent.**

This carries forward CLAUDE.md's oldest hard law — "diversity must be
structural, not learned" — into v200's READ phase. The Pythia-era
validation set is unambiguous: v98 row/col/box/global masks (5+5+5+1
per-head); v100 topological staging masks; v105 per-digit RoPE. Every
validated architecture in this project had **geometric** differentiation
built into who-attends-to-what. v1–v3 learned-diversity mechanisms
(scales, soft tokens, codebooks, fingerprints) all collapsed to constant
within one epoch.

v200's pre-Jun-11 §2 had: 32 latents with **identical cross-attention
scope** (everything), identical self-attention scope (each other),
identical breath_embed added to all of them (consensus-pusher), and 0.01·randn
init jitter as the sole source of differentiation. **Jitter is noise
diversity** — the kind v1–v3 proved collapses within an epoch. 32
structurally identical units given identical inputs and identical
operators converge to consensus by the architecture's own attractor.

**First partition to wire (32 latents, T=24 = n_max=16 nodes + f_max=8 factors):**

| Latent indices | Mask family | Reads |
|---|---|---|
| 0..23 | **per-token** | one latent per fg token; `mask[l, t] = (l == t)` |
| 24..27 | **per-op-type** | 4 latents one per op (ADD/SUB/MUL/DIV); reads tokens of that op |
| 28..31 | **global** | full L×T attention; reads all tokens |

**Critical caveat on the 1:1 per-token slice (added Jun 11 after partition discussion):**

24 per-token latents is a 1:1 latent-to-node assignment, which **quietly
re-creates token-resident state through the back door**. If each of 24
latents owns one node, the latent array IS the node array with extra
steps, and the architecture drifts back toward the homogeneous Pythia
stack's shape — *defeating the perceiver's whole bet* that computation
should live in a SMALLER state than the problem.

v98's masks were genuinely many-to-many: heads owned **constraint
families** (row/col/box), not cells. The factor-graph analogue is
latents assigned to op-types, topological levels, and subgraph
neighborhoods — each latent reading a structured **region**, with
overlap across latents and overlap between regions. That preserves the
compression property (32 registers smaller than the belief state) while
breaking symmetry geometrically.

The 1:1 partition is the easiest to wire and FINE AS THE FIRST
CONFIGURATION — but it's the partition that **least tests the perceiver
thesis**. The compression claim only gets tested by partitions where
latents own regions, not single nodes. **The partition-alternatives row
now in §10 row 1 is therefore not a tuning question — it's where the
compression claim actually gets falsified or vindicated.** Calling out
explicitly so the first partition doesn't become the unexamined default
the way full-L×T did at the prior level of this same gap pattern.

Other partitions are valid as long as they satisfy the random-init
entropy verification (§7) — but the partition must be STRUCTURAL, not
learned. Partition assignment is a config constant, not a trainable
parameter.

**Self-attention left unmasked** (Llama L0-L3 internal attention over
L=32 latents). The READ phase's structural diversity is sufficient: each
latent enters THINK with already-diverse state from its specialized read,
and the residual stream preserves the diversity even as self-attn pushes
toward consensus. v98's analogue: each head's read was specialized, but
their THINK paths shared structure; specialization happened at the read,
not at the inter-head mixing.

**Why this was hidden until Jun 11:** §13's design question 3 listed
"Default: full L×T at Stage-1; add topology routing at v1.1 row 1." The
"default" came from treating masks as an OPTIMIZATION over a working
baseline. Bryce's catch: **without masks, there is no working baseline** —
the architecture specifies consensus as its attractor, and any
trained-latent diagnostic is measuring noise on top of arithmetic
collapse. Topology routing isn't a v1.1 lift; it's a Stage 1
prerequisite. §13 question 3 is now closed (mask required); the v1.1
row 1 description below is recharacterized accordingly.

### WRITE operator — #238 spec (pinned Jun 12 early, promoted from v201 by the #237/#237.5 carrier cells; design brief = the measured numbers)

**The design brief, extracted from instrumentation:** the system needs
cross-breath persistence (carrier MEMORY in both regimes, early ΔCE
+0.0106/+0.0146/+0.0164), needs protection from THINK's writes
(bus-projection HELPING twice: −0.0036/−0.0066 — the workspace literally
overwrites memory), and cannot hold a stable address for what it builds
(carrier dims drifted 7/10 between 1000 and 2000 while function held).
**WRITE inverts all three: fixed address, write-once-per-slot, read-many.**

**The medium mapping (the wave analogy, made precise by the cells):** the
medium is the latent state, two components — the carrier (→ WRITE) is the
PERSISTENT medium, the 32 registers the PROPAGATING one; the masks are the
boundary conditions, not the medium. Belief refinement is a wave that
should leave a wake in the notebook. **WRITE makes the medium elastic —
able to hold a deformation after the wave passes.**

**Design fork (note 1, decided from the carrier's own choices):** the
improvised carrier was SHARED across all 32 latents (10-dim common
subspace, low amplitude) — the blackboard form, not per-latent slots. So:
**shared K-slot buffer**, accumulate-notebook lineage (v110 precedent,
5.2M params, validated):
- Notebook state `N`: (B, K, H) — zeros at forward start (activations,
  not parameters).
- **WRITE (once per breath, at breath end — after the gate blend, the
  settled state):** `N[k] = RMSNorm_w(pool(z_k)) @ W_write` — slot k
  written exactly once; subsequent breaths CANNOT overwrite it
  (write-once BY CONSTRUCTION — the fixed address is the breath index,
  which directly solves the churn). The written content passes a seam
  norm (the write is a seam: workspace meets memory).
- **READ-BACK (during READ phase, breaths k ≥ 1):** latents cross-attend
  to slots 0..k−1 (causal over breaths) via a small dedicated
  cross-attention; contribution enters as
  `z += g_nb · RMSNorm(nb_ctx)` with **g_nb zero-init**.

**The gate question (note 2, reasoned explicitly — neither precedent
inherited silently):** §1A.E.8 says information INLETS get normalized,
not gated — READ is the only channel through which the problem reaches
the latents, so α=0 there would bootstrap-shut the system. The notebook
READ-BACK is structurally different: at step 0 the architecture is
COMPLETE without it (#237.5 runs, learns, transfers — the workspace
already holds everything the notebook would return). The read-back is
auxiliary-at-init in exactly the sense the v98-v121 zero-inits were:
gradient flows through the working baseline, and opening g_nb is a
learnable refinement, not a bootstrap requirement. Zero-init is therefore
CORRECT here — and the WRITE side needs no gate at all (writing to memory
nobody reads yet is free; the gate lives only on the read-back).
Additionally the seam discipline applies on both sides: norm on the
written content, norm on the read-back contribution — both DETACHED-scale
per #237.5's substrate (no new flat directions).

**Attention-bootstrap check (CLAUDE.md principle):** the notebook
read-back is pointer-attention over ≤ K−1 = 7 keys — well inside the
≤32-way support where task gradient alone bootstraps (the principle's own
threshold; the ~500-step direct-supervision requirement applies to ~30+
position pointers). No auxiliary supervision needed; engagement is P-W3's
read instead.

**Step-0 signature (the gate for #238's launch — AMENDED at review to the
runtime-verifiable form; the original (a) compared against a forward that
no longer exists at HEAD):** (a′) `g_nb == 0.0` exactly, asserted at the
gate — the read-back contribution is an exact-0 multiplier, so identity to
#237.5 holds BY CONSTRUCTION (bitwise identity of the gated term verified
by the pre-launch review microtest, on record); (b) write-once is
by-construction (slots appended, never reassigned) with runtime asserts:
K slots written, K−1 read-back attentions fired, all slot contents
finite; (c) the §7 mask verification re-runs unchanged (READ masks
untouched).

**Pre-registered predictions for the first WRITE run (#238):**
- **P-W1 (binding, scored by C4′):** the U's tail flattens — erosion
  absorbed by protected slots. C4′ passes at step 1000+, or at minimum
  the tail-rise fraction drops below #237.5's 15%.
- **P-W2:** the improvised carrier DECOMMISSIONS — carrier-projection
  ablation ΔCE at 2000 falls below #237.5's +0.0164, and/or overlap
  totality decays (the gradient stops needing the improvisation once the
  real operator exists).
- **P-W3:** engagement — g_nb departs zero within ~500 steps and the
  slot-read attention develops breath structure (later breaths reading
  earlier slots above uniform). **[ENGAGED at the FIRST capture, step 100:
  g_nb=+0.0015, notebook grads 2.5e-2 — recorded in-flight.]**

**Three-cell engagement read (pinned at ~step 150, BEFORE the step-200
checkpoint — engagement ≠ adoption, and the middle cell needs an owner):**
- **REPLACED** — P-W2 clean (carrier ablation ΔCE collapses on #238's
  ckpts): WRITE absorbed the function; the workaround decommissioned.
- **COEXISTING** — g_nb risen AND the carrier still load-bearing at 1000
  (boundary projection still hurts): the notebook may have RELOCATED the
  workaround (read-back retrieving the same ~10-dim content through a new
  door) rather than replaced it. Routes to "WHY doesn't WRITE dominate" —
  capacity (the pinned slot-width fork), read-back bandwidth, slot timing
  — NOT to confirmation or refutation of the notebook arc.
- **IGNORED-AFTER-EXPLORING** — g_nb decays back toward zero across
  captures: the gradient visited and declined. Routes to gate/seam design
  (was the read-back the wrong shape?), with the v118-v121 mirror
  explicitly NOT applicable (this one was funded; it chose to leave).

**The erosion-clock read (pinned at step ~250, after two g_nb points
showed 0.00147 → 0.00112):** the trail's shape is read against the
EROSION CLOCK, not just against zero. At step 200 the slots hold
barely-settled early-training state — there is little worth reading yet;
cross-breath persistence first acquires VALUE when erosion starts
destroying it (the control's U formed at ~1000-1500). Two readings of a
sagging early trail: (a) PARKING — the gate relaxes now and RE-ENGAGES in
the window where the control's U formed; g_nb inflecting upward on the
erosion clock = engagement tracking need, the most mechanistically
satisfying P-W3 possible. (b) slow decay to IGNORED. The discriminator is
timing correlation across the full 20-capture trail. A gate that sleeps
until erosion starts isn't ignoring the operator — it's timing it.

**SLOT-DISTINCTNESS RESULT (steps 0 + 200, read minutes after the pin):
DEGENERATE AND CONVERGING — mean pairwise cos 0.9925 (init) → 0.9998
(200), adjacent slots 1.000.** The mechanism is in the write path's own
construction: `pool(z_k)` = mean over all 32 latents — which extracts the
COMMON MODE and averages away exactly the per-breath/per-latent
differentiation that would make slots distinct. The WRITE as built
computes, every breath, approximately the vector the improvised carrier
already carried — the COEXISTING cell's "relocation through a new door"
mechanism, pre-announced by the probe before the gate opened. Corollary:
read-back over eight identical keys is necessarily uniform — P-W3's
breath-structure clause CANNOT fire on this run regardless of training.

**Routing executed as pinned: the fork is slot KEYING → #238.1
(pre-registered now, ships after #238 completes as the photocopy
baseline):** single variable = key the write content by breath BEFORE the
common-mode extraction can erase it. Design: slot_k =
RMSNorm_detached(pool(z_k) + breath_embed[k]) @ W_write — the existing
per-breath marker (orthogonal by construction) keys each slot; one
addition, no new parameters.

**The subtractor-vs-recall TENSION (pinned at the eleventh capture, before
1500):** the two functions want OPPOSITE slot content — a subtractor wants
accurate common-mode estimates (photocopy content is exactly what you'd
subtract; pooling is the right estimator), a reference card wants DISTINCT
slots. The symmetry claim if the subtractor holds: the carrier arc
(gradient builds a common mode; bus ablations show removing its fresh
writes helps) and the notebook arc (given a common-mode repository, the
gate learns to subtract it) converge on ONE statement — this
architecture's workspace performs better with the shared component
REMOVED at read time, and WRITE-as-subtractor is the gradient finally
having an operator to do the cleaning it has wanted all along. #238.1
design implication: keying improves recall content while DEGRADING
subtraction content — the tension may appear directly in #238.1's curve
as gate magnitude SHRINKING as slot distinctness crosses 0.8: two
functions trading off in one number.

**#238.1's TWO-STAGE TRAIL PREDICTION (pinned with the g_nb trail at four
points, before #238's 500 read — the two explanations are LAYERED, not
exclusive):** photocopy slots make the gate directionless NOW, and even
with distinct slots the gate may still park until erosion makes read-back
valuable. So #238.1 predicts BOTH inflections on their own clocks:
(1) DIRECTIONAL COMMITMENT early — the sign-flipping trail straightens
once slots differ (the keying effect); (2) MAGNITUDE GROWTH in the
~1000-1500 window where the control's U formed (the timing effect,
engagement tracking need). Both inflections at their predicted clocks =
two mechanisms confirmed in one curve. One without the other is itself
diagnostic: commitment-without-late-growth = the content mattered but
erosion doesn't drive demand; late-growth-without-early-commitment = the
keying didn't differentiate anything the read-back can use. Alternative candidates if the marker proves
too weak (it is zero-init and may stay small): per-slot write biases, or
family-pooled writes (pool per mask-family → 3 sub-slots per breath —
which also answers the slot-width fork's "what does a slot hold").
#238-as-is REMAINS VALUABLE: it is the photocopy baseline #238.1 diffs
against, its g_nb/erosion-clock trail still reads (the gate's behavior
toward a degenerate memory is itself informative), and P-W1/P-W2 at the
sweep are read WITH the precondition failure on record — if C4′ improves
vs control even with photocopy slots, the mere existence of ANY protected
persistent signal helps; if not, #238.1 explains why before anyone
mourns.

**Slot-distinctness precondition (pinned before any slot content was
examined — the read that can fire 500 steps before C4′'s window):**
distinct slots are the PRECONDITION for everything P-W1/P-W2 hope to see;
a notebook storing eight near-copies makes read-back worthless regardless
of gating. Read: pairwise cosine between the K slot contents on the diag
batch (from the tapped forward on dense ckpts — slots are activations,
computable post-hoc). Pre-committed bands: DISTINCT = mean pairwise cos ≤
0.8; DEGENERATE ("eight photocopies") = mean pairwise cos ≥ 0.95; between
= PARTIAL. DEGENERATE at 500 → the design fork is slot KEYING (write
content needs per-breath differentiation — breath_embed into the write
path, or per-slot W_write), named BEFORE the 1000 binding read so P-W1
doesn't test an operator whose memory is photocopies. Honest prior:
post-blend pooled states are similar across breaths (the substrate's own
stability), so early DEGENERATE-or-PARTIAL is expected; the question is
whether training differentiates them as the gate opens.

**P-W1 ATTRIBUTION (pinned mid-sweep, BEFORE the ablation movement ran):**
P-W1-confirmed is a RUN-LEVEL fact; which operator owns it is an
ablation-level question. The rival to the notebook reading: #238's waist
ended doing enormous breath-differentiated work (norms 124-203, cosines
to 0.72, far beyond the control) — a violently differentiated COMMIT
could flatten the tail by making late breaths genuinely different rather
than by protecting settled state. The splitter is gate-0 on #238's final:
tail SURVIVES gate-0 intact → the waist carried more of P-W1 than the
notebook; WRITE's confirmed contribution narrows to the
gradient-shortcut/subtractor roles. Tail DEGRADES under gate-0 → the
read-back owns its share directly. waist_off provides the complementary
split.

**Subtractor-mechanism conditional (pinned with it):** if the slot probe's
final word is re-converged photocopies, the notebook's adopted function is
COMMON-MODE ESTIMATION — pooled slots as the cleanest available estimate
of the shared component, subtracted at read time. Law 7 then promotes
with mechanism fully specified: the architecture didn't just want the
shared component removed, it REPURPOSED THE MEMORY OPERATOR AS THE
ESTIMATOR. Consequence: #238.1's keying fix becomes officially
COUNTERPRODUCTIVE as specced (distinct slots degrade a common-mode
estimate), and v300 subsystem-1's fork flips — not "key the slots better"
but "give the subtractor a dedicated estimator, then ask whether a
SEPARATE reference memory earns its keep."

**Torsion×ablation cross-read (pinned BEFORE the sweep ran):** the torsion
diag's carrier-fraction decline on #238 (0.27→0.09 as the notebook
matures) is P-W2's decommission showing up in geometry before the carrier
ablation runs. Cross-read: carrier-fraction decline + carrier ablation
DECORATIVE on #238 = the same decommission measured at two levels
(geometric + functional — the clean confirmation). Decline + ablation
still-MEMORY = the trajectory left the carrier dims but the function
didn't — genuinely confusing, gets its own cell, no forced resolution.
P3-proper (slot-subspace overlap of twist directions) is the torsion
diag's v1.1: slot vectors persist from this run forward (slot probe saves
them).

**The 1500 clock caution (pre-stated BEFORE the 1500 read):** #238's
structural clocks run ~2× the control's (READ entropy at 1000 = control's
2000; waist at 1000 = control's 1500) — so the EROSION WINDOW may also
arrive early. A tail already past its minimum at 1500 is NOT
erosion-beat-the-notebook; it may be the clock shift moving the test
window. The per-breath CE trajectory across 1000/1500/2000 is read
against STRUCTURAL markers (waist wake, READ-entropy plateau), not raw
step count — the C4′ thresholds are step-indexed, and the honest read
holds them against the run's own developmental stage. Cheap to note now;
expensive to untangle after a surprising tail number anchors the room.

**Fifth engagement cell — WRITE-AS-GRADIENT-SHORTCUT (named at step 500,
pinned at the 1000 convergence, BEFORE the sweep):** the write path adds a
gradient route from every later breath's loss back through every earlier
breath's pooled settled state — direct pressure on early-breath quality
that #237.5 never had, operating through the BACKWARD alone, requiring no
open read-back. Evidence pattern at 1000: gate ≈ 0 ("closed"), yet READ
entropy at the control's step-2000 sharpness, waist differentiation at
the control's step-1500 state, slots differentiating unaided (min pair
cos 0.396), C4′ tail 0%. Read at the sweep: structural clocks shifted
earlier ACROSS THE BOARD vs control = shortcut; isolated accelerations =
variance. SHARPENED (localization prediction, checkable in the per-group
grad trails already on disk): the speedup should CONCENTRATE in
parameters UPSTREAM of the write taps — breath_embed, early-breath
attention, cross_attn — if the shortcut is the mechanism; uniform
speedup across all groups would instead suggest generic regularization.
Precedent: TEACHING BEATS TELLING (the C1-A auxiliary-head law — a
gradient path outperformed added information); the shortcut would be its
third instance: gradients routed where you want learning, rather than
features routed where you want function. If the shortcut is the operator's real function, WRITE's value
is as TRAINING SCAFFOLD (deep supervision through memory) rather than
inference memory — which #238.1's keyed run can separate: keyed slots +
still-closed gate + still-accelerated clocks = shortcut confirmed;
keyed slots + gate opening = recall was waiting on content all along.

**Fourth engagement cell — ADOPTED-AS-SUBTRACTOR (pinned at the seventh
g_nb point: two consecutive negatives, −0.00107 → −0.00340, growing):**
a NEGATIVE gate is still adoption — the read-back content SUBTRACTED from
the latents at READ time. Mechanistic precedent is already on record: the
#237/#237.5 bus ablations showed removing carrier-dim content HELPS
(−0.0179/−0.0066) — fresh common-mode content in the workspace hurts. A
negative g_nb would be the model learning exactly that operation: read
what persisted, CANCEL it from the workspace — the notebook as
common-mode subtractor (protect-by-removing, freeing workspace capacity)
rather than as recall. Read: g_nb persistently negative and growing while
C4′/eval hold or improve = SUBTRACTOR adoption; routes to the sweep
measuring slot-content vs carrier-dim overlap (is what's being subtracted
the carrier?). Distinct from IGNORED (magnitude grows, not decays) and
from REPLACED/COEXISTING (those assumed additive recall).

**The funded-vs-starved law (the week's design law in one sentence —
§1A.E.8's third data point):** the gradient funds what it must pass
through and starves what it can route around. v118-v121's five perceivers
were routable-around and starved across whole runs; WRITE sits on the
readout's path holding state nothing else preserves, and was funded in a
hundred steps. Build mechanisms un-routable-around or expect them
unvisited.

**Slot width — an explicit scale-factor decision, pinned BEFORE #238's
result locks the default (Jun 12, run in flight at ~step 100):** #238 v1
runs slots at full H=2048 (W_write is H×H) — an INHERITED dimension, the
unexamined-default pattern's favorite hiding place. The carrier's own
measurement argues otherwise: the improvised notebook ran at ~10 dims of
2048 (a ~200× implicit scale factor between what PERSISTS and what
COMPUTES). The design fork, queued as the FIRST WRITE ablation (one
variable, after #238 baselines): d_slot ∈ {16, 64, 256, 2048} via
low-rank W_write (H→d_slot) with slots read back through a d_slot→H
expansion — the carrier's 10-dim choice is the prior that persistent
state wants to be much smaller than workspace state. Uniformity audit
note: "slot width = H" was a default, not a choice; this paragraph makes
it a choice.

config_sig token: `_write8`. Reference curves regenerate (§2 change).
Same seed/steps/data; #237.5 is the control.

### Single-forward consolidation (added Jun 11)

**Required:** there is ONE forward function for v200, living in
`mycelium.factor_graph_v200.FactorGraphV200.forward`. Training, eval,
probe, and any diagnostic instrumentation all call this same function
(varying only `return_taps: bool`). **Parallel reimplementations in the
training script or scripts/ are forbidden.**

This came out of Stage 1C's #232 audit: the corrected JSD method had to
be patched at TWO sites (`mycelium/factor_graph_v200.py:1429` AND
`scripts/v200_perceiver_train.py:875`) because the training script
reimplemented the forward with its own conventions — including a
readout RMSNorm at training line 619 that did not exist in the class
API. The training path and the instrumentation path were measuring
DIFFERENT architectures, and every probe result would have carried a
silent asterisk about which architecture it measured.

This is precisely the **same-backbone-different-architecture failure mode
that killed the v200-Pythia port** — now living inside one file with
two forwards. The brief refuses it structurally.

Concretely: the training script (`scripts/v200_perceiver_train.py` or
its successor) calls `FactorGraphV200.forward(fg_tokens, K=K,
training=True, return_taps=False)` for the loss path and
`FactorGraphV200.forward(fg_tokens, K=K, training=False,
return_taps=True)` for diagnostics. The class IS the architecture; the
script is the runner. PR review must reject any new forward path that
isn't a thin wrapper over the class method.

| Spec | Value | Notes |
|---|---|---|
| Base model | **Llama-3.2-1B** | hidden=2048; ~4× waist ratio (vs Pythia 2×). HF auth set up Jun 11. Stage 1A smoke validated on both SmolLM2-1.7B (no GQA) and Llama-3.2-1B (GQA: 8 KV heads, 32 Q heads). SmolLM2 retained as regression-baseline ckpt. Loader must be GQA-aware (consolidation is Stage 1B's first task; see §15). |
| Hidden dim | 2048 | Latents and tokens both at 2048 |
| Shared layers | 4 (L0-L3) | Same self-attention path each breath |
| Latents (L) | 32 | IB-anchored init; one per semantic cluster |
| Tokens (T) | 24 | n_max=16 + f_max=8 from existing factor graph |
| K | 8 initial | Same as v110-step3; sweep planned (§10) |
| Waist | 2048 → 512 → 2048 | 4× compression; alternating breaths |
| Params (total) | ~295M | ~268M Llama frozen-ish + ~27M new |

---

## §3 Carry-forward components (validated v98–v121 → v200 adaptation)

| Component | Source | v200 adaptation |
|---|---|---|
| Tree codebook output | v108 | 5-level digit readout from pooled latents |
| Per-breath weighted CE ladder | v98+ | `loss = Σ_k (1 + k/(K-1)) · CE_k`; per-breath latent readout |
| Calibration head (Dopri5) | v110-step3 | Per-breath scalar from pooled latents; drives delta_gate |
| Waist + alternation | v109/v109a | Latents waist on even breaths; identity on odd |
| Per-breath Q rotation | v109pi | Rotate latent self-attention queries by k·π/K |
| Mirror staging | v114 | Late breaths (k ≥ K/2) reverse token presentation order in cross-attn |
| Topology tensor (per-node identity) | v112b | Routes cross-attention: per-latent embedding → which tokens it reads |
| SBP training noise | v110-step3-sbp | Gaussian σ=0.02 on latents at start of each breath, 50/50 |
| Per-breath delta_gate | v98+ | Convex blend on latent post-update; static (K,) tensor |
| IB-anchored codebook init | v105.1.2+ | Latent init from 32 IB centroids (Pythia-extracted) |
| Right-aligned RoPE on digit axis | v105.1.2 | In tree codebook readout (ones digit at RoPE pos 0) |
| Accumulate notebook (optional) | v110-acc | K-slot diary of latent commitments; test if redundant with persistent latents |
| Basin-landing monotonic waist | v115 | Heavy compression early, light late — sweep variant in §10 |

---

## §4 What does NOT carry forward

| Component | Why it doesn't apply |
|---|---|
| Residual-stream-primary processing | Latents replace the residual stream as primary |
| Per-position residual gate (v112b's load-bearing piece) | No residual stream; topology tensor reused for cross-attn routing |
| Multi-token per variable (v105 family) | Variables are tokens latents READ from; collapse problem doesn't arise |
| Feedback channel (v105.11 arc) | Latent state IS the belief; no explicit feedback. (Principle 12 still applies to SBP) |
| Perceiver as add-on (v118–v121) | The refutation pattern that motivated this pivot |
| Domain codebook (200-bin input) | Latents read from token embeddings directly |
| Per-digit input tokens (v113) | Factor graph tokens unchanged; latent representation is learned |
| Most warm-start ckpts | Cold-start from Llama base; no Pythia-era ckpt is compatible |

### Pythia-era pins re-tested on Llama family

These are NOT carried forward as settled. The pin existed and was validated in
the Pythia codebase; the evidence base does NOT cover SmolLM2/Llama attention.

| Pythia-era pin | Status on SmolLM2-1.7B |
|---|---|
| **wv-sharing across layers (L0-wv shared)** | **REFUTED on SmolLM2 (cos=0.5126, Jun 10). UNCLEAR-because-instrument-lost-power on Llama-3.2-1B (cos=0.9715, Jun 11).** Read cos=0.9715 carefully: GQA's narrower V (Llama wv=(2048,512) vs SmolLM2 (2048,2048), 4× fewer DoF) leaves less room for layer divergence to begin with, so the static-forward cos test compresses toward 1.0 *regardless of whether sharing helps or hurts training*. Not "evidence the pin transfers" — evidence the instrument lost discriminative power on this architecture. **Asymmetry worth naming explicitly:** on Pythia, SHARED wv was the v98-VALIDATED configuration and PER-LAYER was the one that stuck at chance during cold-start. v200 Stage 1B's conservative default (per-layer) is conservative relative to the SmolLM2 refutation but is the ANTI-validated choice relative to the only cold-start success the project has on this problem class. Probably still correct — the newer backbones differentiate V per-layer in ways Pythia didn't, and SmolLM2 says so directly — but the prior is mixed enough that wv-sharing earns a pre-registered debug-branch slot (see §15). A training-time test on Llama-3.2-1B at Stage 1C is what would actually settle the pin. See `[[wv-sharing-refuted-smollm2-jun10]]` + `.cache/v200_smoke/llama32_load.log`. |
| **Per-breath Q rotation by k·π/K (v109pi)** | UNVERIFIED on SmolLM2. Validated on Pythia 1024-d. Stage 1C smoke (training-time) should validate or refute. |
| **delta_gate as static (K,) tensor** | UNVERIFIED. Pythia 1024-d evidence base. Stage 1C smoke. |
| **Mirror staging on late breaths (v114)** | UNVERIFIED. Pythia 1024-d evidence base. Stage 1C smoke. |
| **IB-anchored codebook init** | **GEOMETRY PRESERVED, SEMANTIC RANK LOWER THAN 32** (Jun 11). Centroid structure check: Pythia 1024-d → Llama 2048-d via linear projection preserves pairwise cosine ordering at Spearman 0.99+ across Glorot/std=1/√out/semi-orthogonal/identity-padded variants (chance baseline 0.03). The learned linear has IB structure to refine. **BUT** ~5 centroid pairs at cos_sim > 0.97 in native Pythia space cross operation boundaries (`MUL.0.0.1 ≈ ADD.0.0.2 ≈ DIV.0.2.2`); effective semantic rank ≈ 20-25, not 32. Stage 1B uses project-from-Pythia with **small init jitter (0.01·randn) for symmetry-breaking** so near-duplicate latents don't track each other from step 0. v1.1 queue: de-duplicated init OR jittered init OR random-init-for-duplicate-slots as a row. Semantic alignment to Llama embedding space remains UNVERIFIED — re-extraction from Llama is queued as v1.1 contingent on Stage 1B's IB-engagement signal. |

**Documentation discipline (lesson from wv-sharing refutation, Jun 10):**
documentation of pinned decisions must carry its evidence base, not just the
conclusion. A pin without its evidence context can't tell future-you whether
it transfers. The brief now lists every Pythia-era pin with its evidence base
explicitly tagged, so Stage 1B/1C can decide per-pin whether to re-test or
inherit.

---

## §5 Persistence of intermediates (new, Jun 10)

Latents persist across all K breaths by design — that is the architectural
inversion. Intermediate artifacts of the build (residual traces, latent
trajectories, attention weights, calibration outputs) ALSO persist, by
discipline.

**Rule:** every per-breath quantity that could route a future decision gets
saved to disk during training, not just held in graph state for the current
forward. Default save targets, per smoke and per prod step (every N steps):
- Per-breath latent state `z_k` (B, L=32, H=2048), bf16, sampled subset of B
- Per-breath attention weights for cross-attn (Q=latent, K=token), per head-group
- Per-breath delta_gate value (scalar)
- Per-breath calibration head output (scalar per puzzle)
- Per-breath read_ctx norm and self-attn residual norm
- Per-breath tree codebook readout (logits, not just argmax)

**Why this matters.** The v98–v121 era repeatedly hit "the measurement we needed
wasn't saved." The Jun 9 controls audit found SBP/MC-BP/v112b reduced to null
under cont-control because the warm-start baselines couldn't be recovered
post-hoc — we had only the converged endpoints, not the trajectories. v200
starts with the discipline: **save what an honest controlled comparison would
need, every run, from step 0.**

This is not a documentation aspiration. It is a smoke-test requirement: a v200
training run that does not produce the persistence bundle fails the smoke check.

**Random-init reference curves (added Jun 11 after Stage 1B).** Stage 1B's
arch smoke produces the random-init trajectories for every per-breath
metric the §8 gates test (latent JSD, energy channel ‖Δz_j‖, cross-attn
entropy, self-attn entropy). These are the **null shapes** Gate B compares
against — saved permanently at:
- `.cache/v200_smoke/reference_curves/latent_jsd_random_init.npz` + provenance
- `.cache/v200_smoke/reference_curves/energy_channel_random_init.npz` + provenance
- `.cache/v200_smoke/reference_curves/xattn_entropy_random_init.npz` + provenance
- `.cache/v200_smoke/reference_curves/self_attn_entropy_random_init.npz` + provenance

Each `.npz` contains the per-breath metric values for K=8 over a B≥32
random-init forward pass. Provenance sidecar names `with_what.ckpt =
"random_init_seed_{seed}"` so the reference curve is reproducible.

**Why the reference curve matters.** A monotone-decay random-init latent
JSD is uninformative ("random weights settle to a fixed point" is the
baseline; that's not the §8 damping failure mode). The gate-B signal is
the *trained* model *departing* from the random-init null. Without a
persisted reference, "non-monotone in isolation" is too weak a test —
trained noise can satisfy it. With the reference, "trained model's latent
JSD trajectory differs significantly from the random-init null" is sharp.
Same logic for the entropy distinguishability test: random-init cross-attn
entropy ≈ log(T) (fully diffuse) is the reference, and the trained model's
cross-attn entropy departing from log(T) during READ is the actual signal.

---

## §6 Provenance metadata (new, Jun 10)

Every persisted artifact carries a four-axis provenance dict, written next to
the data file (`<artifact>.provenance.json`). The four axes:

```python
provenance = {
    "what": {
        "metric": "...",              # e.g., "latent_z_per_breath", "attn_jsd_mean"
        "units": "...",               # nats, raw, bf16, etc.
        "shape": [...],
        "head_group": "..." or None,
    },
    "where": {
        "file": "/abs/path/to/file",
        "key": "..." or None,          # inside JSON/npz/etc.
    },
    "when": {
        "timestamp_iso": "...",
        "git_sha": "...",
        "config_diff": "..." or None,   # diff from named base config
        "step": ... or None,
    },
    "with_what": {
        "ckpt": "...",
        "split": "...",                 # train / val / test, named
        "seed": ...,
        "arch_version": "...",          # see ARCH_VERSION discipline below
        "env": {
            "tinygrad_sha": "...",
            "device": "AM driver/AMD 7900 XTX",
            "env_vars": {...},          # K_MAX, FIXED_LEN, BATCH, etc.
        },
    },
}
```

### `with_what.arch_version` — required field added Jun 11

Every artifact's provenance MUST carry `with_what.arch_version`, a string
identifying the exact §2 architecture revision the artifact was produced
under. Concretely:
- Format: `"v200-{git_sha[:8]}-{config_signature}"` (e.g., `"v200-a3f8e2d4-K8_L32_prenorm3"`)
- Required on EVERY persisted artifact, including reference curves,
  smoke logs, checkpoints, eval JSONs, persistence bundles
- Comparison scripts MUST refuse to compare artifacts with mismatched
  `arch_version` (fail loudly, not silently)

This came out of Stage 1C: the original reference curves were generated
under pre-RMSNorm §2; the Jun 11 update added three RMSNorms; comparing
trained-with-norm against ref-without-norm is a category error of the
class this week keeps catching. `arch_version` makes the category check
structural rather than vigilance-based. Every time §2 changes, the
reference curves get regenerated; comparison machinery sees the version
mismatch automatically and demands the fresh null.

**Rule:** an artifact without a provenance dict is not a v200 artifact. Plotting,
comparison, and write-up scripts read provenance first, fail loudly on missing
fields, and never silently align across mismatched `with_what` axes (different
ckpt, different split, different seed).

This is the discipline the Jun 10 entropy-map salvage made load-bearing: a
measurement whose `what` axis was misidentified (entropy on input region, not
JSD on decode trajectory) silently propagated into a wrong "architectural K=2"
inference. The provenance check would have flagged the misalignment before the
inference fired. v200 bakes it in from step 0.

---

## §7 Instrumentation (matches this week's ε convention)

Per-breath latent JSD logging uses the **same ε convention and freeze-breath
reporting** as this week's diagnostics (`scripts/diag_interior_probe_v110_v98.py`,
`memory/project_interior_probe_results_jun10.md`), so v200's gate-B numbers are
directly comparable to the v110-step3 and v98 baselines just established.

**Conventions (locked):**
- ε = 5% of breath-0 metric value (per-metric scale, NOT global)
- Freeze breath = first k where metric ≤ ε
- "Moving" vs "frozen" by half-K threshold (K=8 → half_K=4)
- Report freeze-breath as integers per layer per metric, plus raw per-breath
  trace vectors below the table
- Split by outcome class (solved/failed by existing readout) and node type
  (given/non-given, observed/unobserved)
- Per-head-group attention metrics: group by mask family, not head index

**v200-specific instrumentation tracks:**
- Per-breath latent JSD (between breath k and k+1 latent distributions, via the corrected pairwise inter-position cosine fingerprint metric; method_sha recorded per §7's JSD identity discipline)
- Per-latent step-size ‖Δz_j‖ (the energy channel; see §10 v1.1 row 1)
- Per-cross-attention head-group entropy per breath
- Per-self-attention layer entropy per breath (NOT JSD; like-units per §8 Gate B clause)
- Per-breath calibration head output and effective delta_gate
- **Per-latent THINK attention entropy** (added Jun 11 late evening, after the message-passing gap finding). For each breath, each latent computes a self-attention distribution over the OTHER 31 latents (inside Llama L0-L3); compute the entropy of that distribution per latent, report mean and standard deviation across the 32 latents per breath. Random-init expected: ~log(31) ≈ 3.43 per latent (uniform attention over others). Diagnostic for whether THINK is doing structural message passing or mean-field consensus mixing. **Uniform low-std entropy at training end → mean-field consensus → routes to #238 (THINK quotient-graph mask).** Differentiated per-latent entropy → some inter-latent specialization is happening without explicit structure → THINK quotient-graph mask may not be needed. See `[[v200-message-passing-hypothesis-jun11]]` for the pre-registered #238 hypothesis derived from this diagnostic.
- **Concentration-drift metric: top-10/2048 dim energy fraction of post-THINK state per checkpoint** (added Jun 11 evening after Control 1+2; refined late-evening after #236). Init Llama on tokens locks at 98%; v235 at step 200 measured 25.8%; **v236 at step 200 measured 4.0% (random-level)**; random baseline 4.8%.

**Named finding (#236, Jun 11 late evening): Seam-3 norm de-recruits trained Llama L0-L3 from the massive-activation regime.** Same trained Llama, same weight structure, different input statistics → different attractor. Bounded latent registers (post-norm_breath ~1, post-READ-add ~1.3, post-norm_blend ~1) drive Llama L0-L3 to produce post-THINK states at per-element 67 (Llama natural attractor range) BUT with random-level dim concentration — large magnitudes distributed across many channels rather than concentrated into a sink. This is a regime distinct from anything the Llama-family literature describes for token-stream operation. Implication for the registers-not-tokens thesis: bounded latent registers produce Llama-mediated computation that has Llama's amplification character but NOT its sink-attractor character. Llama-as-organ does different work when given different input statistics, and the seam principle is what gates which work it does.

**Measurement-site provenance (added Jun 11 late evening after the 4%-vs-26% ambiguity caught attention):** the `concentration_drift` provenance dict's `with_what.measurement_site` field MUST record where in the forward the metric was captured: `"post-THINK-pre-norm_blend"` (the architecturally-anchored site) vs `"post-norm_blend"` (post-norm, would be artificially uniform) vs `"reconstructed-via-delta-gate-inversion"` (the v235 archaeology method). Cross-architecture comparisons require same-site numbers; without this field, a 4-vs-26 jump between runs could be either dynamics change or instrument change. Site-provenance pre-empts the ambiguity.

The slow-drift question this metric answers — refined post-#236 finding: is the v200 trained latent post-THINK concentration **STABLE at the seam-bounded ~4% regime across training** (= latents stay out of Llama's sink mechanism, registers-not-tokens thesis holds), or **drifting TOWARD Llama's native 98% across steps** (= the seam containment is being learned around, the latents are getting absorbed into token-stream mechanics on a slow timescale). Stable-low ≈ thesis vindication; rising → 98% ≈ thesis erosion. One number per eval checkpoint, persisted with explicit site provenance, tracked across the training horizon.
- **Inter-position cosine, mean-removed, per breath** (added Jun 11) — for each breath k, subtract the across-position mean from each latent position then compute the mean pairwise cosine of the residuals. Shared-additive dominance (e.g., diffuse cross-attention adding the same mean-of-tokens vector to all latents) drives raw inter-position cosine to 1 by arithmetic; mean-removed cosine is the component that arithmetic can't fake. Position-collapse disambiguation hinges on this metric — see §1A.E.4.
- **‖read_ctx‖ / ‖z_pre_breath‖ per breath** (added Jun 11) — the dominance ratio of READ's contribution to the pre-breath latent state. If the READ output is 100× the normalized state even after §2's RMSNorm placement lands, the READ operator needs its own gate or scale (v109-style zero-init residual blend), and that's a §2 follow-up item rather than a downstream symptom.

### Structural-mask verification at random init (added Jun 11)

The §2 per-latent structural mask requirement has a **construction-time
verification** that fires before any training. **Per-group, not just
mean** (refined Jun 11 after the partition discussion): each mask
family's mean entropy at random init must be ≤ log(family_support_size)
with the inequality strict whenever family_support_size > 1. Specifically,
for the §2 first partition:

| Mask family | Family support | Random-init entropy assertion |
|---|---|---|
| Per-token (24 latents) | 1 token each | mean entropy = 0 (by construction; one-hot attention) |
| Per-op-type (4 latents) | ~6 tokens of one op | mean entropy ≤ log(6) ≈ 1.79 |
| Global (4 latents) | all 24 tokens | mean entropy ≤ log(24) ≈ 3.18 |

**The assertion is per-group, NOT mean-over-all-latents.** Cheap to
specify now: a heterogeneous mean can sit below log(T)=3.178 while
individual groups are silently unmasked. The week's other per-head-group
metrics ($1A.B, §7) follow the same discipline — comparison is at the
group level, never at the cross-group aggregate.

Group-wise check at random init:
```
for family, latent_indices, expected_max_H in partition_table:
    H_family = mean(entropy(attn_weights[batch, head, latent_indices, :]))
    assert H_family < log(family_support_size[family]) + epsilon,
        f"mask family {family} entropy {H_family} exceeds support log({...})"
```

Smoke-level assertion, not manual log inspection. **Adding this check
to the smoke would have caught the §2 gap in seconds two days ago.**
Once the §2 mask requirement lands, this same check at random-init
confirms the mask is structurally present without needing training-time
signal.

Generalizes to a portable rule: **every group-aware metric in the brief
asserts per-group, never cross-group mean**. The portable-principles list
[[principles-as-tripwires-jun11]] now carries this as a meta-principle
about how principles' verifications get specified.

**Comparability:** smoke and prod runs print the freeze-breath table in the
same format as this week's diagnostics. v200's gate-B test (§8) reads off the
same shape of table; numbers slot in directly next to v98 / v110-step3 rows.

### JSD method identity (added Jun 11)

The latent JSD metric used for Criterion C2 (§1A.B.2) and Gate B (§8) MUST
be the SAME function applied to BOTH the persisted random-init reference
curves AND the trained-model trajectory. "Same function" is enforced at
the code path level, not narrative:
- The provenance sidecar's `what.metric_sha` records the git SHA of the
  exact function implementation used. Reference curves and measurements
  carry matching `metric_sha`.
- Comparison scripts refuse to compare across mismatched `metric_sha`
  alongside the §6 `arch_version` check.
- If the JSD method is revised (Stage 1C+ likely requires it because the
  current implementation reports ~0 on a scale-exploded substrate), the
  revised function REGENERATES reference curves before becoming the
  binding measurement.

**Concrete pick for Stage 1C re-smoke:** the JSD method should compute
divergence between consecutive-breath latent distributions in a manner
that is meaningful on bounded-scale latents. Softmax-pooled per-latent-
position distributions become meaningful once `‖z‖` is bounded by the
pre-norm placement; this is the implementation default. If a different
projection is needed (e.g., tree-codebook-projected JSD), the choice
gets recorded in `metric_sha` and applied identically to both null and
measurement.

This discipline came out of Stage 1C: the original JSD method reported
`~0` on a scale-exploded substrate, contributing to the misread. With
substrate fixed and method-identity enforced across null and
measurement, the next reading is structurally airtight.

---

## §8 Kill criterion — gates A and B

v200 must clear both gates within its evaluation horizon to earn the next
stage. Each gate has a pre-registered threshold and a regression cap.

### Gate A — Accuracy

**Threshold.** Cell_acc on hard ≥ `chain_saturation - 0.02` at the
configured evaluation step.
- `chain_saturation` is the converged cell_acc of the cont-control chain at
  comparable compute (currently 0.376 for v110-step3 on hard, K=8).
- The -0.02 tolerance is the empirically-established drift floor
  (`memory/feedback_cont_control_protocol.md`).

**Regression cap.** Cell_acc on easy and med must each be ≥ the cont-control
chain by at least -0.05 (no easy/med catastrophic loss in exchange for hard
gains).

### Gate B — Mechanism

**What Gate B is testing (updated Jun 10 after probe).** The Jun 10 linear
probe retired the readout-bottleneck framing — there is no discarded signal
in the homogeneous loop's interior for v200's latents to recover. Gate B
therefore tests the cleaner, harder claim: **v200's latents must generate
dynamics the homogeneous loop never had.** Not recover, not surface — generate.
A pass means the latent stream sustains non-trivial information flow across
breaths under cross-attention READ + self-attention THINK alternation, where
the v98–v121 residual stream went dead at k≈4.

**Threshold.** Two distinguishable conditions, both required:
1. Per-breath latent JSD is **non-monotone** across breaths (i.e., latents
   are not collapsing to a fixed point by mid-K) AND the freeze breath under
   the §7 ε=5% convention is **≥ half-K** on at least 2 of 4 self-attention
   layers. **The "non-monotone" test compares against the persisted random-
   init reference curve from Stage 1B arch smoke (see §5)** — not against
   "non-monotone in isolation." Random weights settle to a monotone decay
   trajectory at any depth; the gate-B signal is whether the trained model
   *departs* from that null shape, not whether the trained model's latent
   JSD happens to be non-monotone (which can be noise).
2. Cross-attention and self-attention head-group entropies are **distinguishable**:
   cross-attn entropy is higher at READ phase, self-attn entropy is higher at
   THINK phase, and the difference is statistically robust across the eval
   batch (per-batch sign agreement ≥ 80%).

**Like-units clause (added Jun 11, after Stage 1B revealed the trip-hazard).**
Both metrics in condition 2 must be **attention-weight entropies** (units:
nats), per head-group, computed under the §7 ε=5% convention. **JSD and
entropy are NOT interchangeable; the gate cannot be satisfied by a units
mismatch.** Stage 1B's instrumentation reported a "distinguishable 3.18 vs
0.06" comparison that was actually entropy (3.18 nats, log(24) = uniform
attention) against between-breath JSD (0.06) — a category error that
could have passed gate-B inadvertently three weeks from now if the metric
mismatch hadn't been caught at random init. The instrumentation must
expose `compute_xattn_entropy_per_breath` AND `compute_self_attn_entropy_per_breath`,
both returning attention-weight entropy in nats per head-group, with the
same ε convention. Comparing entropy-vs-JSD is a method error, not a gate
result.

**Regression cap.** No latent layer shows attn-JSD freeze at k < half-K with
calibration confidence > 0.8 (high-confidence freeze = the dead-architecture
failure mode v200 is supposed to escape).

### Both gates active

Gate A protects against v200-on-mechanism-without-accuracy (the "interesting
dynamics, no signal" failure). Gate B protects against v200-on-accuracy-
without-mechanism (the "got numbers by accident, no architectural story"
failure). Either gate failing kills the variant at the evaluation horizon.

### Evaluation horizon

- **Stage-1 smoke:** 200 steps. Gate-A waived; Gate-B is "any non-trivial signal."
- **Stage-2 short prod:** 2000 steps. Gate-A: ≥ 0.05 cell_acc on hard; Gate-B: full.
- **Stage-3 full prod:** 5000–10000 steps. Gate-A: full chain-saturation – 0.02;
  Gate-B: full plus per-breath ladder slope ≤ –0.05 in CE.

If a stage fails gates, the next stage does not run on the same variant; v1.1
queue (§10) gets the next attempt.

---

## §9 Cont-control protocol (inherits from Jun 9 methodology)

**Rule.** Any claim that v200 lifts a metric over a prior baseline compares to
the **continuation chain** `cont_{i+1}_stepN` (a same-length training run from
the same warm-start anchor under the same config), NOT to the frozen warm-start
anchor itself. The drift floor is ±0.02 on hard.

**Application to v200.** Cold-start variants don't have a warm-start anchor;
they compare to the **chain-saturation reference** from §8 (v110-step3
saturation on hard). When v200 reaches a milestone (Stage 2 or Stage 3
evaluation), the comparison is:

```
v200_stage_N  vs  chain_saturation_at_matched_compute
```

NOT:

```
v200_stage_N  vs  v200_step_0  (trivial; v200 hadn't started)
NOT
v200_stage_N  vs  v110_step3_cont8_step1000  (mismatched compute, mismatched arch)
```

When v200 has its own warm-start chain (post Stage 3, multiple variants), the
cont-control protocol applies straight from `memory/feedback_cont_control_protocol.md`.

### v200 drift floor is UNMEASURED — borrowed reporting discipline (added Jun 11)

The cont-control protocol's ±0.02 drift floor on hard was empirically
measured on the v110-step3 chain (Jun 9 controls audit,
`memory/project_jun9_controls_audit.md`). **v200 is a different paradigm
(perceiver-core vs residual-stream-primary).** Until two same-architecture
v200 continuation segments exist from a shared warm-start anchor, v200's
own drift floor is UNMEASURED, and the v110-step3 ±0.02 is being borrowed
as a placeholder.

**Reporting rule until v200's drift floor is measured:**
- Every v200 metric delta is reported with `±0.02 (borrowed from v110-step3)`
  explicitly annotated. Not the value alone, not "drift floor ±0.02" — the
  full borrowed annotation.
- Cont-control comparison structures (per §1A.D) emit the borrowed-floor
  marker in the eval JSON: `"drift_floor_status": "borrowed_from_v110-step3"`.
- Any "v200 lifts X by Y > 0.02" claim must carry the borrowing in its
  first sentence, not in a footnote.

**When v200's drift floor becomes measurable:**

Stage 2's first real training run is also the control-generator. Concretely:
- The first Stage 2 prod run trains v200 to step N with dense early
  checkpoints (every 100 steps for the first 1K, every 250 steps after).
- After Stage 2 lands a converged result, run TWO additional continuation
  segments from the same intermediate warm-start anchor (e.g., from
  `v200_stage2_step500.safetensors`) with different seeds.
- The variance between the two continuation segments at matched
  step-from-anchor IS v200's drift floor on hard.
- This replaces the v110-step3 borrowed floor in the brief permanently.

**Persisting dense checkpoints at the start is cheap; doing it retroactively
is impossible.** The v200 training driver must save checkpoints at the
dense cadence specified above from step 0 of Stage 2. Sub-agents
implementing Stage 2 are required to verify the checkpoint cadence in the
smoke before any production training fires.

**Why this matters:** the Jun 9 controls audit found SBP/MC-BP/v112b all
reduced to null under proper cont-control. That audit was retroactive — the
warm-start anchors for those claims weren't preserved, so the drift floor
had to be reconstructed post-hoc. v200 cannot inherit this debt; the
drift-floor measurement is a Stage 2 deliverable, not a Stage 3 audit.

---

## §10 v1.1 ablation queue (updated Jun 10)

These are the planned first-round variants after Stage-3 cold-start lands a
baseline number. Each is one variable changed from the baseline (§2), one
smoke + one short prod.

| # | Variant | Description | Why it matters |
|---|---|---|---|
| 1 | **Energy channel (‖Δz_j‖ priority)** | Log and analyze per-latent step-size every breath; ablate by clamping per-latent step-size to uniform across latents | Tests whether v200 develops latent specialization (some latents working harder than others). The energy channel is the §7 instrumentation that surfaces this; the ablation tests whether the specialization is load-bearing. |
| 2 | **Dual-mode cross-attention (within-read K=2)** | Within each READ phase, two sub-cross-attentions (focused-mode and broad-mode) with separate learned gates | Tests the Jun 10 entropy-map finding (HMM K=2 BIC-modal on input region; sharp/diffuse dwells 10.2/18.0 tokens). The within-read K=2 finding was a measurement at a different level than architectural K=2; this variant tests whether it transfers as a within-read design. |
| 3 | **N sweep** | N ∈ {1, 2, 3, inverted 2-cross:1-self}. N = cross-attentions per self-attention block within a breath | N=1 is baseline. N=2/3 test whether more reads per think helps. The **inverted 2-cross:1-self** variant survived the entropy reframing's death only as a sweep point — state-assignment never got settled, just mooted at the measurement level — but it stays in the sweep because the question is still empirically open. |
| 5 | **Per-breath α on `α·RMSNorm(read_ctx)`** (added Jun 11 evening, contingent v1.1) | Replace the scalar `alpha_read` with `alpha_read_per_breath` of shape `(K,)`, init = `[1.0] * K`. Only promote from queued to active **if #235's scalar α shows strain** — drifts large (> 5), breath-conflicted gradient signal in `grad_norms`, or per-breath rdr trajectory wants different scales than a single scalar can express. | Tests whether the archaeology's "READ's useful scale is breath-dependent" observation (rdr 173 at breath 0 = info delivery; ~1-13 after) lands as a per-breath need, or whether one scalar settles the average usefully enough. Diagnostic for whether breath structure is load-bearing in READ. |
| 4 | **IB init variants** (added Jun 11; framing sharpened by C2 inheritance) | Three sub-variants: (a) de-duplicated IB centroids (drop near-duplicate pairs at cos_sim > 0.95, ~20-25 centroids); (b) random init for duplicate-cluster slots only; (c) re-extracted IB centroids from Llama-3.2-1B embeddings | The default Stage 1B init is project-from-Pythia + 0.01·randn jitter. **C2 prior inherited (see [[ib-centroid-structure-check-jun11]] §"Cross-op collapse is C2 re-emerging"):** op identity is not separable in any natural feature space without supervision, so unsupervised clustering can't yield op-distinct anchors. The right question for variant (c) is NOT "find op-distinct anchors" (probably impossible) but "is Llama's embedding manifold a different axis of variance than Pythia's, and does that matter for READ phase usefulness?" Variants (a) and (b) test whether the IB axis (depth/position/magnitude context, per C2's prior search) is useful at the rank it actually supports (~20-25). All three queued at v1.1; row promotes to 1B-blocking only if Stage 1B's IB engagement signal is null. |
| 9 | **Perf wins, series-boundary class** (added Jun 12 — audit: JIT fundamentals already in place: fused train step incl. Adam, warm kernel cache, fixed eval shapes) | (a) BEAM kernel search (env-only; 1.5-3× typical on AMD matmul graphs; ULP-level kernel differences → enters at a SERIES BOUNDARY with provenance, never mid-series; 50-step timing smoke + same-seed trajectory check first); (b) JIT the eval path in resmoke drivers (compile_jit_eval_v200 exists, unused — ~3 min/run); (c) thin ALL EAGER INSTRUMENTATION to every 200 after step 1000 — written generically, not grad-norms-specifically: the slot read, notebook taps, and any future WRITE instrumentation inherit the same eager cost profile as they multiply (keep early density; the early dynamics are where the catches live); (d) BATCH headroom (6GB used of 24 — ~2× throughput; config-bound, series boundary). Not-easy tier: organ internals at fp16/bf16 with the fp32 chain kept (the starvation was inter-breath, not intra-organ) — own smoke, pre-registered trajectory match. BEAM smoke additionally records KERNEL-CACHE SIZE and COMPILE TIME as their own lines — the disk cache is shared state across runs, and a corrupted or version-bumped cache is a silent confound; tinygrad version + BEAM level belong in provenance `with_what` the same way arch_version does. | Runs cost ~3h; (a)+(d) together could halve that. Same-seed control comparability is active currency — perf config changes only between series, noted in provenance. |
| 8 | **Waist compression-ratio sweep** (added Jun 12 — the scale factor's natural home) | Sweep waist_dim ∈ {128, 256, 512, 1024} (current 512 = 4× was INHERITED, never swept). Companion question: compute-state and persist-state may want DIFFERENT scales — the carrier ran persistence at ~10 dims while the waist compresses compute-state at 512. | The week made the question measurable: the carrier's ~200× persist-vs-compute scale factor is evidence the inherited 4× is far from any optimum. JPEG framing: quantization is lossy scaling-down where what survives is what mattered — the ratio IS the psychoacoustic model's aggressiveness. v1.1 row; one variable per run. |
| 7 | **Per-node breath budget (JSD-driven scheduler)** (added Jun 12 — decomposition's allocation gain, unexploited) | Allocate breaths per NODE rather than uniformly: per-cell JSD is computable in-loop (the validated discriminator: 1.8-9.3× early-motion separation by difficulty); cells whose beliefs froze stop consuming budget, cells still moving keep breathing. The principled per-node form of Goldilocks (global step-size → per-node effort). | The dead-breaths finding showed ~half the budget spent on frozen beliefs; a decomposed problem's whole point is per-component effort. Gated on a working scheduler design + E.16's depth-split result (if sequential capacity binds, allocation is the lever). Discovered-beats-designed law applies: provide the allocation AFFORDANCE, measure what the gradient does with it. |
| 6 | **Per-parameter-group LR** (added Jun 11 late, during #237 — pre-registered, FIRST ELIGIBLE at the first real Stage-2 run, NOT a mid-experiment patch) | Lower LR on the pretrained Llama L0-L3 interior; spec LR (3e-4) on the from-scratch plumbing (cross-attn projections, waist, latent_init, norms, gates). Optional companion: warmup + decay schedule, promoted only if the #237 2000-read shows reorganization phases still churning at run end. | The principled form of "lower LR to stabilize." #237's oscillations (ρ up-down, concentration up-down, READ entropy down-up-plateau, waist norm up-down with directions diversifying) are REORGANIZATION, not optimizer instability — substrate green, loss descending, eval waking at 1500; a global LR cut would stretch the phases past the horizon and read as "masks failed" when the truth was "still cooking." Project scar on file: the one LR deviation (3e-4→1e-4, first 1C smoke) masked missing normalization — the architecture fix was the real answer. LICENSING EVIDENCE for any LR move: a specific parameter group showing oscillating gradients in the per-group norms the driver already logs — that names WHICH LR. Single-variable rule applies to the optimizer: changing LR for #238 would put an optimizer asterisk on every cross-run comparison. |

**Row 1 reclassification (Jun 11).** What was previously v1.1 row 1
("topology-guided cross-attention masking") has been **promoted to a
Stage 1 §2 requirement** — see §2's "Per-latent structural cross-attention
mask" section. v1.1's row 1 slot is now **partition-alternatives** —
specifically the variants that test whether the compression claim
survives. Stage 1's default partition (24 per-token + 4 per-op + 4
global) is the easiest to wire but the LEAST informative on the
perceiver thesis (per §2's "Critical caveat on the 1:1 per-token slice").

| Variant | Partition | Tests |
|---|---|---|
| 1a (default) | 24 per-token + 4 per-op + 4 global | First-wire baseline. Cross-attn mask works at all. |
| 1b **(the perceiver thesis test)** | 8 per-op-region + 8 per-topo-level + 8 per-neighborhood + 8 global | Region-owning latents, no 1:1 token assignment. Compression claim genuinely tested. |
| 1c | All 32 global (no mask, ablation control) | Falsification check — if this matches 1a's accuracy, masks aren't load-bearing on this task class. |

Row 1 is now the perceiver-thesis ablation. Promotion to Stage 1B+
happens if 1a passes the §1A.E.4 grid; otherwise 1b/1c run alongside
1a as the partition-vs-mechanism disambiguation.

After v1.1 lands a winner, v1.2 sweep is contingent on results: SBP composition,
Mirror staging on cross-attn token ordering, Topology tensor routing density.

---

## §11 Delegation workflow (lesson Jun 10)

**Rule:** "Sonnet returned" ≠ "work done" when the delegation pattern is
launch-script-and-monitor. Completion is defined by the artifact landing (results
JSON, checkpoint, memo), not the agent's turn ending.

**Applied to v200:**
- Sub-agents that run training, eval, or extraction return their AGENT turn when
  they finish writing & launching the script + arming a monitor.
- The parent (Claude or the user) waits for the *artifact* (results JSON,
  ckpt, traces). The monitor fires progress events, then a "done" event.
- Task completion in the task ledger is gated on the artifact, not the agent
  status field.
- When delegating, the prompt must specify the expected output ARTIFACT path
  and contents — not just the work to be done.

This lesson came out of task #226's `Linear probes on residual ridge` launch:
Sonnet wrote and started the script, armed a monitor, and returned (which
correctly marked its agent turn complete). The user-facing task #226 was
incorrectly auto-marked completed, but the script was still extracting residuals.
Surfacing this discipline up front prevents the next 40-minute false-completion.

### Spec-deviation discipline (added Jun 11 after Stage 1C)

Sub-agents may make unilateral engineering calls (reduce K, lower LR, swap
optimizer) when the spec is infeasible under the GPU budget. These are
ALLOWED — they're usually defensible — but they **promote the run to
ADVISORY status**, with three structural consequences:

1. **Pass/fail does not bind.** ADVISORY runs report metrics but their
   PASS/FAIL verdict against the trajectory criteria (§1A.B) is for
   diagnostic value only. The criteria thresholds were calibrated against
   the spec configuration; a deviation invalidates the calibration. No
   ADVISORY pass certifies any gate; no ADVISORY fail triggers any debug
   branch sequence (§15).
2. **Deviation must be surfaced in the first line of the report**, not
   discovered by audit. The smoke log's `STAGE 1C SMOKE PASSED/FAILED`
   line must read `ADVISORY: <deviation summary>` if any spec value was
   changed from the brief. The next-level reader sees the deviation
   before reading the numbers, not after.
3. **The next pass must run at spec.** Either the deviation is justified
   permanently (then the spec gets updated in the brief and the next run
   is at-spec by the new spec) or the deviation is provisional (then the
   next run resolves the gap — optimizing the forward pass, finding a
   stable LR, etc.) and the actual at-spec result lands before any
   binding gate decision.

This came out of Stage 1C: Sonnet correctly reduced K=8→4 (52s/step
budget) and LR=3e-4→1e-4 (NaN at spec), each defensible — but the
criteria thresholds (half-K from 4→2, slope window from 7 points → 3
points, "young zone" boundaries from §1A.E.1) were calibrated for K=8.
Silent invalidation of threshold calibration is how a smoke result
becomes argued-over rather than read. ADVISORY promotion prevents this:
the run is informative, the verdict is for diagnostic only, and the
binding result waits for the at-spec re-run.

---

## §12 Long-haul commitment

User decision Jun 8: "we might fail 100 times (hopefully not lol). lets be
patient and with the truely novel approach."

**Cadence change.** v98–v121 ran 1–3 day cycles per architectural variant. v200
runs **weeks per variant**. Each variant is a substantial architectural
variation — energy channel implementation, dual-mode cross-attention,
N-sweep position — not a config flag.

**Evaluation horizon discipline.** No variant is killed before its evaluation
horizon (§8 stages). No variant gets MORE than its evaluation horizon without
explicit user sign-off ("this is interesting; let's run it longer"). The
kill-criterion gates prevent "interesting but dead" variants from eating the
budget.

**Failure documentation.** If v200 fails its Stage 3 gates after 5K–10K steps,
the residual-stream paradigm had a known queued Phase-2 path (per-position
delta_gate from topology, v112b extension; see CLAUDE.md §6) that was not built
when the pivot was committed. That path remains the documented fallback.

---

## §13 Open design questions (deferred to v1.1)

1. **Latent count L:** 32 (matches IB cluster count). Alternative: L=24 to match
   n_max+f_max. L=64 to give per-token routing slack. **Default L=32 unless
   smoke argues otherwise.**
2. **Latent self-attention layers:** Llama L0-L3 (warm-start) vs fresh init.
   **Default: Llama L0-L3 warm.** Fresh-init is a §10 variant if warm fails.
3. **Read attention masking:** ~~topology-guided (v112b adaptation) vs full L×T.
   Default: full L×T at Stage-1; add topology routing at v1.1 row 1.~~
   **CLOSED Jun 11.** Per-latent structural mask is a Stage 1 REQUIREMENT,
   not a v1.1 lift. See §2 "Per-latent structural cross-attention mask"
   section. The full-L×T default specified consensus as the attractor;
   the structural-not-learned diversity law (CLAUDE.md) predicts the
   collapse Stage 1C's #232 surfaced.
4. **Write step:** include latent → token write per breath? **Default: NO at
   Stage 1–3; included as v201 variant if Stage 3 hits Gate A but misses Gate B.**
5. **Cross-attention dimensionality:** latents stay at 2048d. **Locked.**
6. **Waist placement:** on latents, alternating breaths. **Locked.**
7. **Waist schedule:** uniform (constant 4× ratio) vs basin-landing monotonic
   (heavy early, light late, per v115). **Default uniform at Stage 1; basin-
   landing is v1.2 contingent on v1.1.**

---

## §14 Motivation paragraph — LOCKED (Branch C, Jun 10)

The linear-probe task (#226) ran against the three-branch grid pre-registered
in `memory/project_interior_probe_pre_registration_jun10.md`. Results
(`memory/project_linear_probes_residual_ridge_jun10.md`):

| Outcome | Threshold | v98 | v110 | Status |
|---|---|---|---|---|
| A — ridge accumulates info | late > early by ≥ 5% | +2.94% | +0.67% | REJECTED |
| B — codebook bottleneck | late ≈ early > existing by ≥ 5% | probes below | probes below | REJECTED |
| C — ridge empty / housekeeping | late ≈ early ≈ existing within ±3% | -3.75% (just outside) | -3.08% (just outside) | SELECTED with deviation |

### Motivation paragraph (locked)

> Linear probes find no information gradient across breaths and no discarded
> signal on failures; the interior dynamics observed in the probe are settling
> and uncertainty churn, not unread computation. The cheap readout-side
> alternatives are closed on evidence. v200 proceeds on paradigm grounds —
> telegraph thesis, must-be-core, gate B — **with the readout-bottleneck
> framing retired, not deferred**.

### Deviation note (instrument floor, not finding)

The strict Branch C threshold required probes ≈ existing within ±3%. Actual:
probes 3-4% below existing on both ckpts. **This is an instrument-floor fact,
not a finding about the residual stream.** The existing readout is a trained,
nonlinear, structured tree codebook learned over the full run; a fresh linear
probe losing to it by 3-4% is the expected result whenever the readout is
competent. The strengthening claim must not be sourced from this gap.

### What actually carries the structural-change conclusion

Two narrower findings, both cross-cutting v98 and v110:

1. **Late ≈ Early ≈ Δ** on both ckpts. The ridge direction (Δx alone) extracts
   the same information as the residual at any breath. The ridge isn't a
   separate information channel — it carries the same signal as the residual
   it's drifting through. That's the empirical signature of **housekeeping**:
   norm equilibration / fixed-point settling, directionally consistent
   (cos≈0.999, per the Jun 10 interior probe), informationally same-as-residual.
   No information accumulates after the freeze; there is nothing in late breaths
   for any reader, linear or otherwise, to tap.

2. **v110 thrashing refutation.** The Jun 10 interior probe found "interior
   more alive on failures." That was either discarded signal or thrashing.
   Probe-LATE on failed beats baseline by +0.022, well below the +0.05
   threshold. **Thrashing / uncertainty churn**, not knowledge. Failure-side
   churn carries no extractable signal.

Together these close the cheap-fix routes — not because linear readers are
weak, but because the information isn't there to extract.

### v98 probe limitation (carry honestly)

n=300 medium puzzles were all solved by the existing readout, so the
discarded-signal split was degenerate on v98. **The failure-side test only
actually ran on v110.** The conclusion above holds because v110 was the
decision-relevant ckpt for the Cell-4 interpretation, but the brief does not
claim both ckpts passed the failure-side test. One did, one couldn't.

### Yesterday's Cell-4 reading: partially overturned by its own follow-up

The Jun 10 interior probe assigned v110 to Cell 4 ("interior moving, beliefs
frozen, readout bottleneck"). The motion was real and that observation stands.
The **bottleneck interpretation** died today. Gate B is therefore testing a
cleaner, harder claim than the brief originally implied (see §8 below for the
updated framing): v200's latents are not being asked to *recover* signal the
old readout discarded — **there was none** — they are being asked to *generate*
dynamics the homogeneous loop never had. This is the provenance discipline
applied to interpretations, not just artifacts.

### Framing guard (unchanged)

The probe routed one paragraph and retired the readout-bottleneck pillar.
v200's three remaining pillars stand on independent evidence:

1. **Telegraph thesis** — homogeneous loops have no rhythm to break;
   perceiver-core's read/think alternation is the architectural rhythm.
2. **Must-be-core finding (v118–v121)** — five attempts to add perceiver as
   accessory all refused; the perceiver must be the core or it doesn't engage.
3. **Gate B re-establishment stakes** — without an architecture that routes
   information through latents by construction, the next freeze reappears.
   (Updated framing per §8 — v200 generates dynamics, doesn't recover them.)

The kill criterion (§8) gates v200, not the probe.

---

## §15 Build plan (stages from §8)

**Stage 1A — Llama-1B base loader + verification smoke (~2-4 hrs):**

Foundation; built before any v200 architecture exists. Deliverables, each at
its specified path with a sibling `.provenance.json` per §6:

- `mycelium/llama_base.py` — Llama-3.2-1B (or SmolLM2-1.7B fallback) tinygrad
  loader. Exposes: `.embed`, `.layers[0..3]`, `.ln_f`. **Hook points exist for
  pre-LN residual taps from day one** (the v1.1 energy channel ‖Δz_j‖ in §10
  is a feature add, not a refactor). Eager-only at this stage (no JIT).
- `scripts/v200_llama_smoke.py` — verification smoke. Loads weights,
  forwards a small batch (B=2, T=64, random tokens), prints per-layer
  activation norms, shape checks, no-NaN check, peak GPU memory, timing.
  **Emits forward statistics WITH AND WITHOUT wv-sharing across L0-L3** so
  the Pythia-era pin's portability gets measured on this backbone, not
  inherited as settled.
- `.cache/v200_smoke/llama_load.log` — full smoke log; final line is
  `SMOKE PASSED` with key metrics OR `SMOKE FAILED <reason>`.
- `.cache/v200_smoke/llama_weights.sha256` — hash of loaded weights for
  reproducibility.
- `.cache/v200_smoke/llama_load.provenance.json` — four-axis sidecar per §6.

**Stage-1A completion criterion (per §11 — artifact, not turn).** The
Stage 1A delegation returns successful ONLY when all five artifacts above
land at the specified paths and the smoke log's final line is `SMOKE PASSED`.
Sonnet's turn ending without these artifacts is not completion.

**Stage 1A → 1B interface contract (pinned now, while one mind holds both halves):**
- `llama_base.py` exposes named attributes `.embed`, `.layers[k]` for k in 0..3,
  `.ln_f` (Llama RMSNorm name).
- Each `.layers[k]` exposes `.attn` (with `.wq, .wk, .wv, .wo`) and `.mlp`
  (with `.gate_proj, .up_proj, .down_proj` — SwiGLU pattern).
- Each `.layers[k]` exposes a **pre-LN tap hook**: calling
  `layer.forward_with_taps(x)` returns `(x_post, {"pre_ln_resid": ..., "post_attn_resid": ..., "post_mlp_resid": ...})`.
- The latent state tensor signature for Stage 1B is `(B, L=32, H=2048)`.
- Provenance sidecars are placed at `<artifact_stem>.provenance.json` (same
  directory, same stem, `.provenance.json` extension).

**Stage 1B — perceiver-core architecture (after Stage 1A lands AND brief
re-read by user):**

Sub-tasks, in order:
1. **Loader consolidation** — promote the inline `LlamaBase32` class (from
   `scripts/v200_llama32_smoke.py`) to `mycelium/llama_base.py` as a
   unified GQA-aware loader. Auto-detects GQA from HF config; same
   `forward_with_taps()` contract; handles both SmolLM2 (no GQA) and
   Llama-3.2-1B (GQA 8 KV / 32 Q) via internal branching, not separate
   classes. SmolLM2 smoke must still pass under the unified loader.
2. **`mycelium/factor_graph_v200.py`** — perceiver-core architecture:
   - Latent init from 32 IB centroids (`.cache/ib_centroids_gsm8k_partial.npz`)
     projected 1024→2048 via learned linear + **0.01·randn jitter** for
     symmetry-breaking (per `[[ib-centroid-structure-check-jun11]]` —
     effective semantic rank is ~20-25 not 32; near-duplicates at
     cos_sim > 0.97 need symmetry break)
   - READ: cross-attention `Q=latents, K=fg_tokens, V=fg_tokens`. Default
     full attention; topology routing is v1.1 row 1.
   - THINK: latent self-attention through Llama L0-L3 (use the unified
     loader). Each layer uses its OWN wv (wv-sharing pin UNCLEAR on
     Llama-3.2-1B, conservative default).
   - COMMIT: waist `2048 → 512 → 2048` on alternating breaths (`k % 2 == 0`).
   - Per-breath delta_gate (static `(K=8,)` tensor) on latent post-update.
   - Tree codebook readout from pooled latents (5-level digit
     decomposition).
   - Calibration head (scalar per breath; Dopri5 error estimator).
3. **Persistence bundle hooks (§5)** — per-breath save targets exposed
   from the model (latent state `z_k`, cross-attn weights, delta_gate,
   calib output, read_ctx norm, self-attn residual norm, tree codebook
   readout). NOT serialized in this stage; just exposed.
4. **Provenance metadata (§6)** — every save path gets `<stem>.provenance.json`
   sidecar; module exposes helpers to generate the four-axis dict from
   the runtime context.
5. **Instrumentation hooks (§7)** — latent JSD, ‖Δz_j‖ energy channel,
   cross-attn head-group entropy, self-attn layer JSD computed
   per-breath, exposed for §8 gate-B testing.

Stage 1B does NOT run training (that's Stage 1C). Stage 1B's smoke:
random-init forward through K=8 breaths on dummy factor graph input,
verify shape contracts, no NaN, persistence + provenance + instrumentation
hooks fire correctly.

**Stage 1B completion criterion (per §11):**
- `mycelium/llama_base.py` consolidated and handles both backbones
- `mycelium/factor_graph_v200.py` exists and is importable
- `scripts/v200_arch_smoke.py` exists and runs to completion
- `.cache/v200_smoke/arch_smoke.log` final line is `SMOKE PASSED`
- `.cache/v200_smoke/arch_smoke.provenance.json` exists per §6
- Stage 1B does NOT need to clear gate A or gate B; those gates apply at
  Stage 1C+.

**Stage 1C — training driver + 200-step smoke:**
- `scripts/v200_train.py` + `scripts/v200_smoke.sh`
- Goal: cold-start to step 200 without crash, loss decreases, JIT compiles,
  persistence bundle (§5) lands, provenance metadata (§6) attached.
- Gate-B partial: any non-trivial signal in latent JSD.

### Pre-registered debug branches — cold-start failure modes (added Jun 11)

**Routing precondition (per §1A.E.2):** these branches are written for the
**cold-start stall pattern only** — criteria 1+2 failing jointly (loss flat
+ latent JSD stuck at reference null + per-position digit_acc ≈ 0.10).
Other failure patterns route to component-specific investigation per the
failure-pattern map in §1A.E.2 — DO NOT enter the branch sequence below on
isolated criterion failures (3 alone, 5 alone, or 3+5 jointly = waist
problem, not stall).

When the cold-start stall pattern fires:

**Debug branch A — shared-L0 wv** (the v98 cold-start success configuration).
Set wv-sharing ON: broadcast L0's wv to L1-L3 in the backbone's THINK phase.
This is the *anti-validated* choice under SmolLM2 evidence but the
*validated* choice under Pythia's only cold-start success on this problem
class. The Llama-3.2-1B static-forward test (cos=0.9715) was instrument-
power-limited and does not adjudicate the training-time outcome. If shared
wv lifts cold-start cell_acc above chance, the pin transfers on Llama
despite the SmolLM2 refutation — and the structural mechanism is different
enough between full-V (SmolLM2) and narrow-V (Llama GQA) that the prior
should be treated as backbone-class-dependent, not architecture-class-
dependent. ~1-2 day ablation; runs before any architectural soul-searching.

**Debug branch B — IB init variants** (per §10 row 4): if shared-wv doesn't
lift, the next first-touch is the IB init story (de-dup, random-for-dup,
re-extract from Llama). Same horizon.

**Debug branch C — backbone freeze toggle**: if A and B both fail, test
whether the backbone L0-L3 is the right scaffold at all. Freeze backbone vs
fine-tune backbone. If both produce the same cold-start failure, the
problem is at the architectural level (READ/THINK/COMMIT design) not the
training-time level.

These three branches consume the first ~5-7 days of debug effort if Stage
1C fails. Architectural changes (latent count, K, masking) come AFTER these
three, not before. Pre-registration prevents misattributed debugging.

**Stage 2 — Short prod (~2000 steps, ~1 week):**
- Per-breath ladder forms (monotonic CE decrease)
- Tree codebook digit_acc > 15% on any position
- Gate A: hard cell_acc ≥ 0.05; Gate B: full (§8)
- If pass: Stage 3. If fail: try v1.1 row 1 (energy channel).

**Stage 3 — Full prod (~5K–10K steps, ~2 weeks):**
- Gate A: hard cell_acc ≥ chain_saturation – 0.02 (= 0.356 at current
  v110-step3 saturation)
- Gate B: full (§8) plus per-breath ladder slope ≤ –0.05 CE
- If pass: v1.1 ablation queue (§10).
- If fail: documented failure, try v1.1 row 2 OR pivot to residual-stream
  Phase-2 path per CLAUDE.md §6.

**v1.1 — Single-variable ablation sweep (~3-4 variants × ~2 weeks each):**
- Energy channel, dual-mode cross-attn, N-sweep
- Each clears its own evaluation horizon under §8 / §9

---

## §16 Related memory pointers

- `[[interior-probe-pre-registration-jun10]]` — locked grid the probe reads against
- `[[interior-probe-results-jun10]]` — interior dynamics measurements (Jun 10)
- `[[v121-perceiver-5x-refuted]]` — must-be-core empirical signal
- `[[v112b-phase1-validates-factorization]]` — per-node gating principle adapted
- `[[ode-integrator-framing]]` — K breaths as ODE integration in latent space
- `[[musical-keys-topology]]` — rhythm-topology principle in latent space
- `[[cont-control-protocol]]` — §9 inherited methodology
- `[[pre-register-kill-criterion]]` — §8 inherited methodology
- `[[jun9-controls-audit]]` — the empirical floor §8 thresholds derive from
- `[[parked-photon-hypotheses-jun9]]` — three parked variants not folded into v1.1
- `[[big-paper-strategy]]` — paper holds for v200 outcome

---

## §17 Stamp

Drafted Jun 10, 2026, before linear probe (task #226) reports. Architecture
§2, instrumentation §7, gates §8, cont-control §9, ablation queue §10,
delegation §11, branches §14 all written probe-blind. Motivation paragraph
will select one of branches A/B/C when the probe lands.

Brief is locked at this draft; updates appear as appended dated sections.
