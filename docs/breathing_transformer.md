# The Breathing Transformer — canonical definition (2026-10-28)

**Author:** Bryce (definition) + Claude (evidence column). **Status:** the design standard for the reasoning side of Mycelium.
Bars govern claims; this document governs what gets built (the compatibility ruling). Every row cites the ledger
(`docs/phase1_skeleton_spec.md`). Regime tag on every toy result: CPU toys, 10-22..10-26, MHA breathing loop.

## Where it sits

English → **reader** (vanilla transformer, sees all surface variation) → **the bottleneck** (a relation DSL: anonymous
entity/value symbols, relations as typed factors; no word or number crosses) → **the breathing transformer** (reasons over
the relations) → **the executor / solver** (fixed algebra: values, exact arithmetic, the answer key). Neural proposes,
symbolic disposes. The reader's job (NL → relations) is measured on its own; the breathing transformer is measured behind
the gold DSL (`scripts/rel_dsl_*.py`).

## The definition

| # | Organ | What it is | Evidence (toy unless stated) |
|---|---|---|---|
| 1 | **Phase binding** | Symbols and roles are phase vectors on complex planes; a composite is the phasor sum of bound pairs, exp(i(θ_symbol + ρ_role)); reading is **conjugate unbind** z ⊙ conj(e^{iθ}) e^{-iρ}. The composition is fixed algebra, never a learned nonlinear combiner. | T2x: bind 1.000 on unseen pairs vs learned binder 0.181; the SAME bound state read by an MLP 0.003 — generalization lives in the unbind operation (the composition law, T2u/T2v/T2x/T2z). |
| 2 | **Breaths + the snap** | One shared block applied NB times (breaths never specialize; heads do). Each breath commits a discrete choice (straight-through snap) that re-enters the state; the snap target is a tree centroid / a symbol, so drift cannot accumulate across many hops. | T3: depth-3 unseen chains at 12.5 % coverage, binding 1.000 vs flat 0.41. T3d: at NB 24 the snap is load-bearing for learning the loop at all (0/5 without). T2u U2: snap holds long chains. |
| 3 | **The tree (SQLite)** | Storage is separate from reasoning: the permanent tree (Welford/BIRCH centroids, `mycelium/tree_db.py`) replaces the MLP as memory; it grows only on evidence connected to correct chains; a step notebook records what each breath derived. | T2p tree grown on hard evidence 0.996; T2r answer-filtered volume clean, hunches collapse; T2w cross-process handshake ALL PASS; tree_db v2 + compaction PASS. Not yet inside a real-data model. |
| 4 | **U-Net macro/micro** | Each cycle expands and collapses: macro comprehension (types, relations) and micro (instances). **Micro carries ID tags only** — any richer micro channel gives the optimizer a bypass around the bottleneck. | T2v: the tree's ADDRESS STRUCTURE generalizes (FLAT_FREE 0.094 vs composed 1.000); the U-Net shape itself tied a factored codebook with one entity per vector (V2 FAIL) — its case is several entities sharing one vector, untested. |
| 5 | **The Nazaré rejoin** | When macro (rule) and micro (ID tag) phases align, the composite rejoins by constructive interference — the read fires only on alignment. | T2x: the phase rejoin perfect on unseen pairs, +0.819 over learned binding. |
| 6 | **No MLP** | The block is MHA without the feed-forward. The MLP memorizes the cross join (the literature's picture: keys stored in combinations of near-orthogonal directions — not tested by us). | Composition law: every learned nonlinear combiner memorizes (0.003 / 0.094 / 0.113 / 0.181). Real data: eq_slot NOMLP still recalled (N1 PERSISTS) — removing the MLP is necessary, not sufficient. |
| 7 | **Breath rotation** | Each breath rotates the ATTENTION-space planes (queries by the breath, stored keys by their write time), never the T6 clock dims: relative phase = how many breaths ago. | T2u: rotation transfers to unseen hop counts 0.985 vs labels 0.398. It interpolates, it does not extrapolate (offset never trained: every arm fails). |
| 8 | **The T6 clock** | Six gears, 180/60/30/15/7.5/3.75° per tick (periods 2..96) on the torus T^6; every gear turns every tick; ≥ 2 gears alias-free over the span (`mycelium/rotor_clock.py`). Tracks order (time); a component of the composite key (slot, T6 stamp, tree level). | T3b: 10/10 landed vs a four-wave base rate ~1/3; the T6 ruling 10-26. |

**The superpower, stated exactly:** binding is exact for one bound pair; bundled pairs are superposition with crosstalk
~1/√P (measured 0.067 at P = 256), cleaned by the snap; binding is the only composition that survives superposition (T2x:
additive at chance 0.507; T3e2: a 64-plane bus fails, 256 lands). The claims to prove on real data: far less training data
for the same accuracy, and chains far longer than training.

## The build (organs into the real-data arm, one at a time, each against the previous arm with pinned bars)

| organ | arm | status |
|---|---|---|
| 6 no MLP + 2 breaths/snap + 8 T6 stamp | NOMLP | rel-DSL cov 0.125: 0.990/0.991 vs VANILLA 0.941/0.951 (R1 +0.045 vs +0.05 bar) |
| 1 phase bind / conjugate unbind | PHASE | Q1 EQUIVALENT to linear binding (+0.009); Q2 beats VANILLA at 3 % data (+0.072); enters the stack |
| 3 the notebook: executor in the loop | NOTEBOOK | K1 PASS: trained depth ≤ 3, depth 5–8 at 0.992 (PHASE 0.909, VANILLA 0.731) |
| 7 breath rotation by derivation time | ROTATE | next (needs the notebook's write times) |
| 3 snap to tree centroids in SQLite | TREE | after ROTATE |
| 4 + 5 U-Net macro/micro + Nazaré | UNET | last |
