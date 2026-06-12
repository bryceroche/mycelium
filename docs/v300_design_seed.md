# v300 Design Seed — living document, not a build plan

**Created:** Jun 12, 2026, with #238 at step ~1150 (mid-run).
**Last reconciled:** Jun 12, 2026, post-#238-sweep + composition-6+7
fold-in (all sections agree with the trigger block as of this stamp).
**Reconciliation rule (tripwire 15 — the living-document twin of the pin
manifest):** landing a cell OBLIGATES updating every section that cites
it, and the stamp above moves. A living document whose early sections
present stale state as current is the constant-reference problem
(tripwire 10) wearing prose; the record must be trustable without
archaeology.
**Status:** SEED, TRIGGER ARMED. Resolved cells are marked
`[RESOLVED Jun 12: …]` in place. Rule inherited from the week: one
operator per run, predictions before data, controls always — v300 will be
built as a sequence of single variables on the spine the evidence chose
(see trigger block), not as a bundle.

---

## The spine (what v300 is, in one statement)

The composite statement, four clauses (origin: message-passing memo +
notebook arc, Jun 11-12):

> **Partition the state** (per-latent regions — READ masks, validated #237).
> **Pass messages along the partition's edges** (operator-side structure —
> form still OPEN; quotient masks demoted, scaffold compensatory ×3 —
> see "topology" below).
> **Let no operator average across the partition** (COMMIT/pooling
> discipline — the uniformity audit applied to the loop; NOTE the slot
> pooling deliberately violates this and the violation turned out to be
> the estimator function, law 7).
> **Give settled state a write-once home** (WRITE — [RESOLVED Jun 12:
> shipped as memory, functioned as TEACHER + common-mode ESTIMATOR; the
> gradient route was the value, the read-back functionally silent at
> inference — see trigger block]).

**The medium mapping** (the wave analogy made precise): the latent state is
the medium — the carrier/WRITE is the *persistent* component, the registers
the *propagating* one; masks are boundary conditions, not medium. WRITE
makes the medium elastic — able to hold a deformation after the wave
passes. v300 is the architecture that takes this mapping as its design
language rather than discovering it by debugging.

---

## Subsystems and their evidence dependencies

### 1. WRITE / the notebook family
- **Function identity** — [RESOLVED Jun 12 (#238 sweep): **TEACHER +
  ESTIMATOR, not inference memory.** The SCAFFOLD branch confirmed by
  ablation (gate-0 left the tail intact; localization read: redistribution
  not amplification — breath_embed +69%, cross_attn −41%); the SUBTRACTOR
  branch confirmed in content + sign (slots re-converged to a common-mode
  estimate, terminal gate −0.0053) with a small inference effect; the
  MEMORY branch did NOT confirm for the notebook — but the improvised
  CARRIER still reads MEMORY at both anchors (the confusion cell:
  function without trajectory, carried unresolved). Three-branch fork
  collapsed to the trigger block's three questions.]
- **Slot keying** — [RESOLVED Jun 12: #238.1-as-specced DOES NOT SHIP —
  keying is counterproductive against the estimator function the operator
  actually adopted (distinct slots degrade a common-mode estimate).
  Superseded by the dedicated-estimator fork, trigger question 2.]
- **Slot width** (d_slot fork) — [MOOT Jun 12: superseded by the
  estimator-design fork; the carrier's ~10-dim persistence prior carries
  forward into the estimator's design instead.]
- **Subgraph-keyed slots, persistent ACROSS problems** — cross-problem
  amortization, the mycelium metaphor's literal content: motifs
  (unit-price×quantity, total−part) made identical by the factor
  representation, solved once, reused [waits: the confusion cell's
  resolution + the dedicated-estimator fork (a reference memory must
  first earn its keep — trigger question 2); E.16's component verdict is
  IN (composes — favorable); v300-proper].
- **MC-BP composition test** — premise REWRITTEN Jun 12: WRITE resolved
  NOT-memory, but erosion IS tamed (tail 4%) and the CARRIER still reads
  MEMORY — so the anti-freeze-redundancy question lives under a new
  premise: does MC-BP's gain shrink on #238-line checkpoints where
  erosion is already absorbed, regardless of which operator owns the
  absorption? Falsifiable, cheap, still the bridge from the old stack's
  one real result to the new stack [resolves: MC-BP eval on #238 ckpts].

### 2. Topology / message passing
- **Quotient-graph THINK masks: DEMOTED** — topology-alignment is
  transient scaffolding the objective discards, replicated in both
  regimes (#237 frozen, #237.5 live). Do not hold the structure.
- **Provide-vs-grow the scaffold** [open, NOW DISFAVORED — updated Jun 12
  with the third run]: #237 (starved) peaked at 200; #237.5 (live) used
  topology at 500-600 then discarded it; **#238 — the healthiest, all-green
  run — used NONE at all** (no above-null ρ anywhere; early trace briefly
  anti-topological). Three points: scaffold usage SHRINKS as functional
  machinery grows. This weighs against providing the scaffold at all —
  it may be unnecessary or regressive on a healthy substrate. Kept only
  as a falsifiable low-priority probe; the burden of proof flipped.
- **Partition 1b (region-owning latents)** — the perceiver compression
  thesis's real test, still queued (§10 row 1) [resolves: v1.1 row].
- **True op_type routing** (the mod-4 ADVISORY's at-spec form) [decision
  pending: brief blessing vs data-dependent masks].
- **The rhythm question — duty cycle / N (never set by evidence):** the
  architecture alternates by schedule (validated: peaked-dwell finding),
  but the read/compute RATIO was inherited, not measured. On record:
  dwell-ratio prior 1.76 within-read (entropy-map salvage), N=1-with-flag,
  the sweep N ∈ {1, 2, 3, inverted 2-cross:1-self} (§10 row 3), and
  critically the **decode-region entropy backfill** — the queued
  measurement that recovers the teacher's real read/compute ratio, and
  the only item on the board that re-grounds the architectural telegraph
  teacher-side if gate B ever needs reinforcement [resolves: backfill
  fires on genuine GPU idle; sweep is a v1.1 row].
- **π-cycled Q rotation** — [RETIRED Jun 12: the re-entry condition
  evaluated and not met — #238's tail flattened (4%) AND endogenous
  torsion is healthy and self-organizing (the 1500 coherence collapse).
  Full closure argument in the measurement tier's torsion entry: the
  schedule cannot know when to stop zigzagging; this architecture can.]

### 3. Decomposition / compositionality
- **E.16 depth-split** — [RESOLVED Jun 12: **COMPONENT-LEARNER
  (graceful)** in both runs — monotone descent with tiny tails at every
  depth incl. depth-7 chains on a K=8 budget; no cliff anywhere.
  Adapted-anchor caveat declared (d≤3 bin had n=18, skipped; d4-anchored
  ratio 0.82 control / 0.74 #238, both ≫ 0.5). #238 lifted cell_acc at
  EVERY depth (+0.02-0.03). Consequence: v300 needs NO forcing machinery;
  the efficiency story is allocation + amortization. The depth-flattening
  wildcard did not fire. **Mycelium is a measured property.**]
- **Inversion OOD** (run components backward; three verdicts incl.
  components-latent-in-representation) [resolves: generator variant +
  eval, post-sweep].
- **Per-node breath budget** (JSD-driven scheduler; validated per-cell
  discriminator 1.8-9.3×) [§10 row 7; feeds v300's allocation story].
- **The energy channel — the budget's native in-loop carrier** (fully
  specced v1.1 row, §10 row 1-as-renumbered): ‖Δz_j‖ as the per-latent
  priority scalar; split-channel design — content normalized, ENERGY as
  explicit features; the internal analogue of the JSD discriminator. The
  allocation story above cites the discriminator; this is the channel
  that carries it inside the forward [resolves: v1.1 row + row 7 design].
- **Transfer-split diagnostic** — E.16's companion read (the two-readings
  pin): late-breath degradation UNIFORM across eval structure =
  memorization → regularization; CONCENTRATED on structural outliers =
  distribution-specific refinement → curriculum breadth before
  supervision or architecture [resolves: post-sweep diagnostic, queued].
- **Fractal self-similarity** — the same breathing rule at node /
  subgraph / problem scales; mycelium literally (same growth rule at
  every scale) [v300-proper; waits on E.16 + amortization groundwork].

### 4. The three-act structure (phases)
- **Act 1 — basin landing**: build the factor graph from NL (segment +
  classify; 4 ops; coarse rep suffices — settled law), then RING it:
  per-node resonant frequency as the basin-landing confidence signal
  [waits: Phase-1 classifier build (spec exists); per-node resonance
  instrumentation maturing from the success-contrast JSD line].
- **Act 2 — expand**: iterative prefill (the current v200 line IS this
  act's prototype).
- **Act 3 — collapse**: the JPEG codec (transform → quantize → encode,
  psychoacoustic model = what survives is what mattered). The waist
  family is this act's seed; compression ratio NEVER swept (§10 row 8)
  [resolves: row 8 + readout-side design work].
- **Phase-1/2 TCP handshake**: acknowledgment signal — can phase 2 reject
  a malformed graph and request re-parse? Confidence bit from
  basin-landing into the iterative phase [waits: Act 1 build + a
  calibration head that works (Goldilocks dependency)].
- **The calibration-head repair — dependency-of-dependencies** (Act 1's
  confidence bit AND the per-node budget both wait on it): the queued fix
  is to FEED EARLY-BREATH JSD INTO THE HEAD AS INPUT rather than asking
  it to infer confidence it provably doesn't have (measured: coin-flip on
  hard). Mechanism + motivating result both on record [resolves: one
  instrumented run with the JSD-fed head; unblocks Goldilocks, the
  handshake, and row 7's scheduler].
- **Phase-boundary hand-off**: smoothing REFUTED (discrete beat continuous
  ×3); a learned representation-conversion at the boundary remains
  v1.1-class [open].

### 5. Substrate (carried forward as law, not as questions)
- Detached-scale seam norms; fp32 inter-breath chain; where-gated guards.
- Bound the seams, not the organ — ON BOTH PASSES (the corollary: the
  loss cannot see what the gradient pays for; audit every protection at
  forward/backward/optimizer-state levels — tripwire 13).
- Funded-vs-starved law: build mechanisms un-routable-around or expect
  them unvisited.
- Zero-init for auxiliary paths only; inlets normalized, never gated;
  outlets (workspace→memory) may zero-init — reasoned per case (§2 WRITE
  gate paragraph as the template).
- Every quantity declared informational gets a drift alarm (tripwire 12);
  probes write measured_config; method identity applies to the null.

### 6. Perf substrate (series-boundary class, §10 row 9)
- BEAM kernel search (timing smoke first; cache + version into
  provenance), JIT'd eval path, generic eager-instrumentation thinning,
  BATCH headroom, organ-internals fp16 (not-easy tier).

---

## Measurement tier (diagnostics, not operators — runnable on existing artifacts)

- **Torsion diagnostic (pinned Jun 12, predictions before data):** the
  week measured speed (Δ norms), direction persistence (cos(Δₖ,Δₖ₊₁) —
  the ridge), and position diversity (ipc) — never whether the path
  TWISTS. Discrete torsion: consecutive deltas define osculating planes
  sharing Δₖ₊₁; the dihedral angle about that shared line is τₖ. The
  memory-plane/computation-plane hypothesis: WRITE creates a functional
  memory subspace → the latent trajectory twists out of the computation
  plane toward slot-aligned directions at write/read moments, then back.
  **Three pinned predictions:** (1) τ > 0 above the estimator floor —
  refuting pure planar descent (the null the cos≈1 ridge finding implied);
  (2) PERIODICITY at breath parity — the alternation visible in
  third-order geometry; (3) #238's twist-direction overlap with the slot
  subspace exceeds the control's overlap with anything (WRITE bending the
  trajectory toward its operator). **Estimator discipline (pinned with
  the predictions):** discrete torsion is noise-sensitive exactly in the
  ridge regime (near-parallel deltas) — report τ alongside local
  curvature, and MASK breaths where the orthogonal component fraction
  sin(θ) of either flanking delta vs the shared delta falls below
  **floor = 0.05** (~3° of parallel); otherwise the twist angle is
  numerically meaningless precisely where the question is most
  interesting. Runs on persisted z bundles (pure numpy — no model, no
  GPU). Q-ROTATION RE-ENTRY SHARPENED: re-enter not just if the tail
  needs later-breaths-different, but specifically if measured torsion is
  LOW while the tail erodes (trajectory stuck in-plane = the failure mode
  rotation was built to break); healthy endogenous torsion + clean tail
  RETIRES rotation permanently on the discovered-beats-designed law.
  π-rotation imposed exogenous torsion by schedule (9/9 K-sweep said
  twisting helped when injected); this measures whether the alternating
  operators now generate it endogenously.
  **RESULTS (Jun 12, both runs, all bundles):** P1 confirmed with shape —
  trained trajectories ZIGZAG (τ 110-178°, oscillatory not helical;
  init-state ridge correctly masked at 6-12% validity). P2 confirmed
  modest — even breaths twist 4-11° more than odd, both runs, every
  checkpoint. P3 REVERSED informatively — twist directions sit in carrier
  dims at 10-60× chance in both runs, but #238's carrier-fraction
  DECLINES as the notebook matures (0.27→0.09): the operator FREES the
  trajectory from the improvised subspace rather than attracting it into
  slot planes. UNORDERED FINDING: #238's late-breath τ COLLAPSED 140°→54°
  at step 1500 (the four-instrument reorganization checkpoint) WITH
  curvature RISING 2-3.5× (0.13→0.28-0.50) — true directional coherence,
  not the ridge returning; the control never resolves (τ 120-145°, curv
  ~0.15 throughout). The zigzag = competing pressures (refine vs erode,
  write vs overwrite); the collapse = those pressures resolving into
  coordinated motion. First measurement showing WRITE changing HOW the
  architecture computes, not just what it scores.
  **π-ROTATION: RETIRED (the week's cleanest discovered-beats-designed
  closure):** the mechanism that imposed twist by schedule retires because
  the architecture, given the right operator, generates twist endogenously
  AND THEN ORGANIZES IT — something no schedule could do, because the
  schedule cannot know when to stop zigzagging. Rotation's 9/9 K-sweep was
  real; it was compensating for an architecture that couldn't twist on its
  own. This one can.
- **Transfer-split diagnostic** — (also listed in subsystem 3; it lives
  in both tiers.)
- **Slot-norm amplitude watch (carried from the #238 declaration):** slot
  norms climbed 30 → 48 → 67 across 1500/1750/2000 — the loss-invisible
  direction (W_write output enters storage unnormed; read-back output is
  detached-normed, so slot scale reaches the loss only via softmax
  sharpness). The only open substrate-adjacent watch item; exactly the
  slow-drift class tripwire 12 exists for. One number per checkpoint on
  any future notebook-bearing run.

## Owed documents (deliverables the seed's findings cite but the record lacks)

- **The frozen-vs-live full diff** (#237 vs #237.5, same seed — the
  organ-plasticity comparison nobody ordered): informs the topology
  provide-vs-grow question, the carrier story, and the clock-speed
  observations. Cited throughout; not yet written as its own document.

## Design laws collected (the week's compressed output)

1. The gradient funds what it must pass through and starves what it can
   route around.
2. The loss cannot see what the gradient pays for — audit protections on
   both passes and in optimizer state. MIRROR (Jun 12, the week's close):
   the loss also cannot see what the gradient LEARNS THROUGH — WRITE's
   value was its backward pass; the backward giveth and taketh, and both
   directions are now measured.
3. Discovered decomposition beats designed; provide affordances, don't
   hold structures.
4. Teaching beats telling — gradients routed where you want learning beat
   features routed where you want function.
5. Discrete phase transitions beat continuous interpolation at the
   read/compute boundary.
6. Uniformity is the null hypothesis; every "everything treated the same"
   is a choice or a silent default.
7. The workspace performs better with the shared component removed at
   read time — PROMOTED (Jun 12): the architecture repurposed the memory
   operator as the common-mode ESTIMATOR (slots re-converged to
   photocopies; terminal gate negative). Corollary: WRITE shipped as
   memory, functioned as a TEACHER — its causal channel was the gradient
   route, not the read-back (gate-0 left the tail intact).
   COMPOSITION 6+7 (Jun 12, stated at reconciliation acceptance):
   "let no operator average across the partition" is a law about the
   COMPUTATION PATH, not about averaging itself. The slot pooling
   averaged across everything — deliberately violating the clause — and
   that average became the one place where uniformity is exactly what you
   want: a reference. The same operation that is poison as content is the
   signal as a baseline-to-subtract. Composed form: uniformity is the
   null hypothesis AND the null is useful — estimate it explicitly,
   remove it at read time, never let it masquerade as computation.
8. Instruments catch what's there and wrong; the gut catches what isn't
   there at all. Pre-register both.

---

## Build trigger

v300's build begins when these cells have values, not before:
**STATUS Jun 12: cells 1-3 FILLED** (WRITE = teacher+estimator, not
inference memory; #238.1-as-specced does not ship — the fork flipped to
dedicated-estimator + does-separate-reference-memory-earn-its-keep; E.16 =
component-learner, premise holds, no forcing machinery needed). Cell 4
(d_slot) is MOOT under the flip — superseded by the estimator-design fork.
THE TRIGGER IS ARMED: spine selection is now a design conversation on
filled cells.

**SPINE (stated by Bryce Jun 12 morning, confirmed with amendments):**
v300 = the three-act structure with WRITE-class GRADIENT SCAFFOLDING as
its training discipline, not its inference architecture. First three
single-variable questions, amended per the localization read:
1. Scaffold-removal — MOSTLY ANSWERED ALREADY: removability proven
   (gate-0 left the tail and accuracy intact = the deployment-simplified
   model exists), value measured (#238-vs-#237.5 diff: +0.02-0.03 cell
   acc at every depth). Residue: one cheap structurally-absent
   verification eval + the G-BOTTLENECK mechanism question (the gradient
   route's bandwidth was scaled by g_nb ≈ 0.003 yet taught breath_embed
   +69% — how a near-closed gate teaches is the open mechanism, feeding
   question 2's design).
2. Dedicated-estimator fork — under composition 6+7 this is now formally:
   should the NULL-ESTIMATOR be a purpose-built running mean (costs
   nothing), or does the notebook — having invented the job — keep it?
   Secondary read: with the estimator role taken, does the notebook find
   a DIFFERENT job or go idle?
3. Depth-lift × budget composition — #238 lifted every depth at uniform
   K=8; the per-node JSD-driven budget (§10 row 7) is the cheapest test
   of whether allocation compounds the component-learner's gains.
LOCALIZATION READ (scored Jun 12, was pinned-unscored): MIXED-STRUCTURED —
breath_embed +69% (the write path's most direct target: teacher confirmed
in its sharpest prediction), backbone +37%, waist +32-66%, delta_gate
+13%, but cross_attn DOWN 41%, readout flat. The scaffold REDIRECTED
learning (from reading harder to differentiating state better) rather
than amplifying it — consonant with the torsion coherence (less gradient
fighting itself). Blanket-upstream version refuted; redistribution is the
teacher's measured signature.
CARRIED MYSTERY (resist narrative resolution): the confusion cell —
carrier function without carrier-dim trajectory — earns its own
diagnostic when the tools mature.

(Original four-cell trigger list, for the record — all dispositioned
above: 1. #238 sweep → FILLED; 2. #238.1 keying → RESOLVED-doesn't-ship;
3. E.16 → FILLED; 4. d_slot → MOOT under the flip.)

Then: sequence the v300 builds one variable at a time — trigger questions
1-3 above, in cost order — with #238-line checkpoints as controls. The
seed updates as cells land AND every section citing a landed cell updates
with it (reconciliation rule, header); it is part of the record, not a
vision document.
