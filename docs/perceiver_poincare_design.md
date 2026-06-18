# Perceiver-Poincaré — Generalizable Reasoning Engine (design memo)

**Branch:** `perceiver-poincare` (off `main` 2b7566b). **Status:** design — nothing fired.
The PERCEIVER revival, with the one ingredient it never had: the **Poincaré ball + the
Tier-2 `g_φ` anchor machinery** — the documented fix for the bootstrap wall that killed it.

This branch is the speculative bet. **`main` (the v98 KenKen executor) is the permanent
safety net** and stays validated/working regardless of what happens here.

---

## 0. Thesis, and why this is not refutation #6

The perceiver (latents-as-primary-state, Perceiver-IO style) is the right *shape* for a
generalizable engine: a fixed pool of latent observer/executor nodes that route into **any**
problem's variables — "a different region of the manifold per problem." v98 (per-cell
residual) is welded to the grid; the perceiver is problem-agnostic.

It was refuted **5×** (v118–v121) and v300 failed flat at chance, all for ONE cause: the
**bootstrap wall on the latent→token routing** — random-init attention to ~N positions never
learns content-dependent routing on diverse data → garbage context → chance. The fix we
validated this whole arc: **anchor the routing to a known structure, relax from there.** The
perceiver never had an anchor. The Poincaré ball gives it one. *That* is the new ingredient.

---

## 1. The factor-graph anchor (NOT hardcoded roles)

The universal language of every logic problem is the **factor graph** (variables +
constraints + topology). Latents anchor to the problem's **constraints**, never to
KenKen-specific row/col/cage:

```
z_latent = closed_form_base(constraint) + g_φ(cells_in_constraint)     # g_φ zero-init
```

`g_φ` is the **Stage-2 DeepSets encoder** (permutation-invariant over a constraint's
cell-set) — already built + validated. For KenKen the constraints *are* row/col/cage → this
**organically recreates the partitioned brain at t=0**. For a GSM8K DAG, latents land at
sub-computation nodes. For an alien puzzle, at its factors. **Zero grid-logic in the code —
the factor graph is the input.** The DAG pivot is a *parameter* (feed a different graph), not
a rewrite. (This is the paradigm shift: anchor to constraints, not roles.)

---

## 2. Co-embedded geodesic engine

Cells AND latents are points in **one** Poincaré ball:
- **Cells** = the raw problem nodes (placed by their factor-graph role).
- **Latents** = floating observer/executor nodes (placed by their constraint, §1).

`READ = d_hyp(z_latent, z_cell)` — **spatial attention and hierarchical depth become one
mechanism.** Global latents at the origin (widest horizon, see everything); constraint
latents out near the rim (local). Climb inward → abstract; descend → project. The geodesic
engine is only possible co-embedded.

---

## 3. KEY CLARIFICATION — the anchor is NOT a bit-exact v98 reproduction

The Tier-2 foothold matched v98 **bit-for-bit** because both were *cell→cell* masks. The
perceiver is a **different mechanism** (latent-mediated: latents read cells → think → write
back). Therefore:
- The anchor gives **sensible latent→cell routing at init** (each latent reads its
  constraint's cells) → coherent context from step 0 → the **bootstrap wall is cured**.
- But **matching v98's accuracy requires TRAINING** — the latents must *learn to deduce*,
  even with perfect routing. Brick-1 is a real training run, **not** a frozen replication.
- **Kill criterion (brick-1):** trains *off chance* + latents stay **engaged** (the prior
  perceivers' tell was `select_norm`/engagement stuck ~0 → anchored, they start engaged →
  verify they *stay*) + approaches v98 cell_acc. NOT "t=0 == v98."

The anchor removes the *routing-bootstrap* excuse. If it still flatlines, the perceiver has a
**deeper** problem and we'll have learned exactly what (an honest 6th refutation, not a
mystery). **Anchor is necessary; maybe not sufficient.**

---

## 4. The breath cycle

`READ` (d_hyp latent←cell, anchored) → `THINK` (latent self-attn) → `COMMIT` (multi-res
waist) → `WRITE` (notebook — deferred, §6).

---

## 5. Multi-resolution waist (the v99 receipt)

A single 512-d squeeze every breath tanked GSM8K (`v99_optimal_prefill`: "decode read from a
compressed 512-d waist with **no expansion**"). So:
- **Dancer (1024d):** bypasses the waist, stays in the `THINK` loop — keeps the high-freq
  arithmetic (exact digits) for the next deductive step.
- **Silhouette (512d, RMSNorm seam):** the compressed common-mode → the notebook.

Both, every breath. Never just the squeeze. (RMSNorm-before-the-waist is the substrate law
already coded in `kenken_llama.py` — reuse `_rms_norm_detached`.)

---

## 6. Notebook = hyperbolic branch-tree (DEFERRED — brick 4)

MCTS search is a **tree** → hyperbolic's native object. Slots as a hyperbolic branch-tree →
`d_hyp(branch_i, branch_j)` = branch similarity → "does branch B match a trap I mapped in A"
becomes a **geodesic query** (strictly better than Euclidean + π-RoPE, a phase-locking patch
for a problem the tree geometry doesn't have). **Build LAST** — don't build the roof before
the foundation. Note the target design; defer the build.

---

## 7. Build ladder (each anchored, each a kill criterion, each earns the next)

1. **ANCHOR TEST** — co-embed latents+cells, constraint-anchored routing (zero-init `g_φ`),
   **train**. Kill: off chance + latents engaged + approaches v98 cell_acc. *Make-or-break:
   does an anchored perceiver hold the engine at all.*
2. **RELAXATION** — unfreeze `g_φ` (dynamic constraint placement). Kill: holds/improves vs (1).
3. **MULTI-RES WAIST** — dancer + silhouette. Kill: holds vs (2) (the waist doesn't lose the
   arithmetic).
4. **HYPERBOLIC NOTEBOOK + MCTS** — last.

KenKen is the **proving substrate** (match v98 → the architecture works on a flat CSP), NOT
the goal. Then DAGs, where pooling + hierarchy actually pay.

---

## 8. Discipline (inherited, hard-won)

- **Anchor everything** (routing, waist init, notebook) to a known structure; never bootstrap
  from random. That single lesson, applied everywhere.
- **Fidelity↔trainability:** relaxation lives in the soft-mask regime (`RELAX_BLOCK_ARG≈6`) —
  a faithful mask gives vanishing gradient. Anchor sharp, relax soft.
- **Substrate laws (Tier-2):** `d_hyp` boundary clamps (|z|²≤1−1e-5, arccosh-arg≥1+1e-7),
  where()-gated NaN guard, tangent-space params (standard Adam, no Riemannian optimizer),
  coord-grad clip + tangent-norm bound (now covering `g_φ` outputs too), no `dtypes.float32`
  literal in the JIT step, finite −1e4 block.
- **Pre-registered kill criteria + baselines** (vs v98) at every brick. No instrument/metric
  shopping after the fact (the Property-2 lesson).
- **HONEST framing:** the perceiver is the project's most-refuted bet. The anchor is the new,
  evidence-based fix, but it may fail for *other* reasons (latent capacity; the waist
  info-loss — mitigated by multi-res; alt-fixed-point sharpness). This is a long-haul research
  bet *with* kill criteria, not a sure thing. A clean failure at brick-1 is a real finding,
  not a setback.

---

## Reuse map (what's already built, on `main`, that this inherits)

- `mycelium/kenken.py`: the hyperbolic mask generator (`_d_hyp_pairwise`, `_exp0_map`,
  `_relation_bias_from_z`, the simplex anchors, the **Stage-2 `g_φ` DeepSets encoder** +
  segment-mean perm-invariance + zero-init), the relaxation guards, the convergence instrument.
- `scripts/kenken_train.py`: the coord-only-optimizer + freeze + warmup + grad-clip +
  tangent-clamp + grad-norm logging harness (the validated relaxation rig).
- `mycelium/kenken_llama.py`: the 512-waist + `_rms_norm_detached` seam (for §5).
- `docs/hyperbolic_mask_generator_spec.md`: the Tier-2 spec + §8 relaxation harness + findings.

---

## 9. RESULTS — Brick-1 (2026-06-16): the perceiver BREATHED

After 5 flatlines, the first non-refutation. The routing-bootstrap wall was a **geometric
init problem**, not an inherent flaw — anchored, it woke up.

- **Engagement GENUINE (the kill switch is alive):** read `select_norm` ~0.45 / write ~0.59
  vs the **uniform floor 1/√S ≈ 0.14**; read entropy ~2.1 nats vs ~3.9 uniform → attention is
  *peaked*, not flat; stable across 50 steps, grads finite. The 5-perceiver death signature
  (routing → uniform) did NOT happen. Review independently confirmed genuine.
- **The anchor is load-bearing:** independent-simplex base gave membership 0.008; the
  **segment-mean of the constraint's cell tangents** lifted it to 0.785/0.883.
- **Triangle-inequality, confirmed + handled:** single unified ball rejected at t=0
  (membership 0.785 < 0.95 — a cell in row∩col∩cage → overlapping centroids, Tier-2 §0
  verbatim); **data-driven fallback to per-constraint** (0.883; row/col recall 1.000 exact,
  cage 0.847 = the per-instance floor brick-2 relaxes).
- **Off-chance but modest, under FROZEN routing:** cell_acc 0.187→0.189 vs 0.143 chance — it
  *breathes* (off-chance start, no chance-plateau = cured wall), it does not yet *deduce*.

**Brick-1 caveats (carry into brick-2):**
1. **Kill-metric mis-calibrated:** `select_norm` floors at 1/√S (can't reach 0), so the
   trainer's "ALIVE if >1e-3" flag is non-discriminating (would false-positive a dead-flat
   run). Genuine *this* time (human read vs floor + entropy), but FIX before relying on it:
   DEAD if within ~10% of the floor / `read_max` near 1/S; ALIVE only clearly above.
2. **Generalizability debt:** the path that trains (per-constraint) is KenKen-shaped (3
   relation-type fields). The role-agnostic single ball — the clean "anchor to constraints,
   not roles" story — was rejected at t=0. Brick-2 must report whether unfreezing g_φ closes
   the 0.785→0.95 single-path gap (restoring it).
3. **Perf/substrate ceilings:** K=8 is the AM-driver limit (K=12 HUNG the device — large-JIT
   quirk); fp32 THINK required (fp16 overflowed at the late breath); **~23 s/step (≈9× v98)**
   → training to v98-level (~8000 steps ≈ 50 h) is infeasible without a perf root-cause fix.

## 10. Brick-2 plan (the deduction test) — per-constraint FIRST

- **Goal = the TREND, not the match:** does cell_acc *clearly climb* once g_φ unfreezes, in a
  feasible ~500–1000 steps (~3–6 h at 23 s/step). Climb → the engine deduces → then invest in
  the perf root-cause for a full train-up. Flat (with clean routing) → can't deduce → stop.
- **Per-constraint FIRST** (clean 0.883 routing isolates deduction), **then single-path**
  (deduction confirmed → isolates whether g_φ closes the geometry gap). NOT single-only
  (confounds deduction with muddy routing).
- **Bake in:** the kill-metric recalibration (vs uniform floor) + per-step trajectory
  persistence (JSONL in run_dir, auditable) + K=8 + fp32 THINK. `main`/v98 stays the fallback.

## 11. Generalizability: paying off the per-constraint debt (factor-graph spine, derived not hardcoded)

Brick-1's per-constraint fallback ({row,col,cage} 3-field) is KenKen-shaped — generalizability
debt. The generalist resolution, in order (a *generalizability* thread, gated behind brick-2
deduction):
1. **Relaxed single ball first.** Does unfreezing g_φ warp the *single* unified ball enough to
   reproduce membership (close brick-1's 0.785→0.95 gap)? If yes → χ=1, the pristine unified
   geodesic engine (one radial axis), no fragmentation. The cleanest outcome.
2. **Derived greedy graph-coloring (the fallback, replacing hardcoded {row,col,cage}).** If the
   single ball is *irreducibly* triangle-bound: build the **constraint-conflict graph** (two
   constraints adjacent iff they share a cell), **greedily color** it (NOT NP-optimal — just
   disjoint-per-color), and give each color its own metric subspace. Within a color no two
   constraints share a cell → each cell in ≤1 → triangle-safe by construction. For KenKen this
   *derives* {row,col,cage} from the adjacency; for any factor graph it discovers the right
   partition. **No grid logic, ever.**
   - **Density-adaptive:** χ = chromatic number scales with overlap density — sparse graph → 1
     color → the unified ball; dense (KenKen) → ~3. The unified engine is the sparse/hierarchical
     case, which is also where hyperbolic curvature pays — the cases align.
   - **Fragments routing, NOT the engine:** colors are separate READ/WRITE geometries, but the
     THINK (latent self-attn) mixes across all latents of all colors → deduction stays unified;
     only the radial-hierarchy becomes per-color (minor).
3. **The generalist spine:** factor graph → bipartite (variables + constraints, the engine never
   sees "row") → g_φ DeepSets placement → relaxed single ball, else derived greedy-coloring →
   latents anchor to the right routing hubs, fully agnostic to rows/cols/grid. The DAG pivot is
   then a *parameter* (feed its factor graph), not a rewrite.

## 12. DEFERRED — the temporal axis: π-cycled aperiodic wave (SHELVED, data-gated)

If §1–§11 are the **spatial** axis (the Poincaré ball + g_φ route latents to the right
constraints), the **temporal** axis is the complementary question: how does the engine keep
breath *k* distinct from breath *k′* so a deep recurrent chain through shared weights does not
**phase-lock / resonate** (activations at a late breath accidentally aligning with an early one,
trapping the deduction in a loop)? Proposed mechanism: a **π-cycled aperiodic wave** (π-RoPE) —
an irrational-frequency phase embedded in the state so the phase signature *never* repeats,
giving every breath a unique continuous chronological timestamp.

**The architectural ruling — "discrete-marker-now, continuous-wave-when-K≫8":**
- **Discrete temporal anchor = today.** The per-breath additive orthogonal marker `breath_embed`
  (`perceiver_poincare.py:706-729/:365-369`; the **validated V11 champion** mechanism) already
  gives K *discrete* orthogonal timestamps. At **K=8** (the AM-driver ceiling) eight markers
  trivially separate eight breaths — there is no "step 50" to collide with. Phase-lock is a
  **deep-time (large-K) pathology**; using an irrational π-wave to separate 8 steps is an atomic
  clock for a boiled egg.
- **Continuous temporal anchor = the unbounded-K future.** The aperiodic wave's real
  justification is that it scales to **K ≫ 8 without enumerating K embeddings** — exactly the
  **geodesic-engine deep-deduction regime** (`r=f(|z|)`, breath-as-radial-traversal; the deep
  prize in CLAUDE.md §0). That is its home: where deduction depth physically demands many breaths
  and the discrete marker set would have to grow unboundedly.

**History receipts (why this is shelved, not adopted):**
- Resonance is **real and already met**: V22 found literal head-collision resonance dips at
  multiples of π/8; the fix that *worked* was **V23a frozen per-head pitch with max-decorrelation
  init** — a STRUCTURAL fix, consistent with the durable rule "diversity must be structural, not
  learned." π-cycled within-breath RoPE specifically is on the **long-abandoned** list
  (CLAUDE.md §7). So the wave does NOT override the structural decorrelation that already solved
  resonance — it is a *complementary continuous temporal anchor for large K*, nothing more.

**Placement ruling (for when it IS motivated — do NOT wire until then):**
- For the **phase-lock purpose** → the **THINK loop (the "Dancer", 1024-d active recurrent
  state)**. That is where the recurrence through shared weights lives, so that is where
  resonance would form.
- Timestamping the **"Silhouette" (512-d) as it is written to the notebook** is a *separate*
  function — memory ordering / branch disambiguation, a **Brick-4 (hyperbolic-MCTS notebook)**
  concern — NOT the anti-resonance fix. Two distinct jobs; only the Dancer one is about
  phase-lock. Neither the Dancer/Silhouette split nor the notebook exists yet (deferred bricks).

**Data-driven trigger (the gate — do NOT pull this forward speculatively):** add the aperiodic
wave ONLY when BOTH (a) K physically scales ≫ 8 (deduction depth demands it), AND (b) the data
shows a phase-lock signature — **periodicity in the convergence-instrument JSD** or a
**plateauing-with-cycling cell_acc** in the trajectory JSONL. Near-term hook: the brick-2
deduction read should **scan the trajectory for periodicity/phase-lock signatures** so the
decision to bring the wave forward is evidence-driven. Until then it stays shelved; introducing
it would also violate one-variable-at-a-time (it would confound the g_φ deduction gate).

## 13. THE CATHEDRAL — integrated dancer / silhouette / notebook / π-memory-wave (spec; build-ready, FIRE-GATED)

The integrated cross-breath memory system. Each breath: the renorm-stabilized **Dancer**
(1024-d THINK state) is projected to a compressed **Silhouette** (512-d common-mode) via a
SIDE-CHANNEL waist; the silhouette is **π-stamped** (aperiodic chronological phase) and written
to a **Notebook** tensor slot; later breaths read the π-stamped slots back to disambiguate and
build on prior sub-deductions. **The bet: ORGANIZE deductions across breaths — not add raw
capacity.**

**GATING (read first):** FIRE only if the foundation-capacity gate (renorm-alone @ 5000 steps)
shows the foundation has capacity (cell_acc climbs toward v98). If the foundation plateaus at
~0.37, the cathedral is **futile** — memory cannot add capacity the foundation lacks (you'd be
storing weak deductions). Built default-off byte-identical so it never disturbs the validated
renorm baseline, and not fired until the gate verdict + an explicit go.

**Components (composable toggles, default-off byte-identical, in the JIT cache key):**
1. **Dancer (`PERCEIVER_THINK_RENORM`, have it):** the 1024-d THINK loop, renorm-stabilized.
   UNCHANGED. Full-fidelity — NO compression in the reasoning path (the v80 lesson: an in-path
   waist deletes the arithmetic KenKen needs).
2. **Silhouette (`PERCEIVER_SILHOUETTE`, side-channel waist):** a projection `W_sil: 1024→512` of
   the THINK OUTPUT each breath, computed AFTER the THINK, OFF the reasoning path. The compressed
   common-mode, used ONLY as the notebook write source — NEVER fed back into the THINK (that is
   the in-path v80 bottleneck). Its only effect is via the (zero-init) notebook write → bootstrap-
   safe by construction.
3. **π-stamp (`PERCEIVER_NB_PIROPE`, the memory wave — on the MEMORY, NEVER the compute):** as the
   silhouette is written to slot at breath k, imprint phase `k·π/K_max` (RoPE-style on the write +
   the slot-read query, so reads can match/disambiguate by phase). NOTEBOOK write/read ONLY —
   NEVER the THINK query (3 variants proved compute-π → rim escape + flat). K-caveat: value is
   large-K (K≫8); at K=8 `breath_embed` suffices → expect ~no K=8 benefit; built for the deep-
   recurrence future, off until K can scale.
4. **Notebook (`PERCEIVER_NOTEBOOK`, extend the existing K-slot memory):** stores the π-stamped
   silhouettes. WRITE: π-stamped silhouette → slot (accumulate). READ: π-aware cross-attn over
   slots → injected into the latent state. **THE READ INJECTION IS THE RIM-RISK** (it feeds the
   compute → perturbs g_φ → latents flee the rim; proven — the full-1024 notebook drove
   max_latent_z → 0.90). MANDATORY mitigations: zero-init read-out proj (t=0 contribution exactly
   0, the verified pattern); the latent rim-clamp (committed); a small/gated read. The cathedral
   notebook stores the **512 silhouette** (compressed) vs the current full-1024 store — the open
   test is whether compressed memory behaves better than the full-state notebook (which failed:
   flat + rim).

**Lessons baked in (the build MUST honor — these are WHY prior screens failed):** π on MEMORY not
the THINK query; silhouette/waist SIDE-CHANNEL not in-path; ALL injecting projections (write +
read) ZERO-INIT → bootstrap-safe + byte-identical-off; the latent rim-clamp covers everything;
default-off byte-identical (master + sub toggles) + JIT cache key + substrate-legal (no float32
literal in JIT, single-kernel, no .contiguous() in the breath loop) + FAST_GRADNORM-compatible.

**Ablation plan (one-variable-at-a-time — the bundle lesson):** built together, but EVALUATED
incrementally on separate gates — (i) renorm + silhouette-notebook (no π): does *compressed*
memory beat the failed full-state notebook AND stay rim-clean? (ii) +π-stamp: only meaningful at
K≫8. NEVER fire the 4-component big-bang and read one number.

**Honest expected outcome:** given the fragility pattern (every compute-feeding addition → rim),
the read injection likely still pressures the rim (contained by the clamp). The live hope: the
*compressed* silhouette memory is cleaner than the full-state notebook → better retrieval →
enables refinement (fixes the flat ladder). Unproven; the gate + ablation (i) decide it.
