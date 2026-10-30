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

## The two rulings (2026-10-29)

- **No MLP, ever.** Nothing we build or train carries a feed-forward MLP: memory is the SQLite tree, binding is phase binding,
  composition is attention. Exceptions: the frozen Llama trunk (a teacher until a distilled MLP-free student replaces it) and
  MLP / vanilla arms as measurement controls only.
- **Alternation is the core rhythm.** The breathing transformer and the CSP solver take turns: a breath proposes / derives, the
  solver executes, certifies and writes back what is now known (the notebook — never raw values into the reasoning state), the next
  breath reads it. The NOTEBOOK arm is this rhythm's first evidence (0.994 vs 0.724 at depth 5–9 from depth ≤ 2 training).

```
English -> reader (frozen Llama now; distilled MLP-free student next) -> relation DSL
        -> [ breathing transformer  <->  CSP solver ]  (alternation, breath by breath)  -> answer (the key grades)
```

## The definition

| # | Organ | What it is | Evidence (toy unless stated) |
|---|---|---|---|
| 1 | **Phase binding** | Symbols and roles are phase vectors on complex planes; a composite is the phasor sum of bound pairs, exp(i(θ_symbol + ρ_role)); reading is **conjugate unbind** z ⊙ conj(e^{iθ}) e^{-iρ}. The composition is fixed algebra, never a learned nonlinear combiner. | T2x: bind 1.000 on unseen pairs vs learned binder 0.181; the SAME bound state read by an MLP 0.003 — generalization lives in the unbind operation (the composition law, T2u/T2v/T2x/T2z). |
| 2 | **Breaths + the snap** | One shared block applied NB times (breaths never specialize; heads do). Each breath commits a discrete choice (straight-through snap) that re-enters the state; the snap target is a tree centroid / a symbol, so drift cannot accumulate across many hops. | T3: depth-3 unseen chains at 12.5 % coverage, binding 1.000 vs flat 0.41. T3d: at NB 24 the snap is load-bearing for learning the loop at all (0/5 without). T2u U2: snap holds long chains. |
| 3 | **The tree (SQLite)** | Storage is separate from reasoning: the permanent tree (Welford/BIRCH centroids, `mycelium/tree_db.py`) replaces the MLP as memory; it grows only on evidence connected to correct chains; a step notebook records what each breath derived. | T2p tree grown on hard evidence 0.996; T2r answer-filtered volume clean, hunches collapse; T2w cross-process handshake ALL PASS; tree_db v2 + compaction PASS. Not yet inside a real-data model. |
| 4 | **U-Net macro/micro** | Each cycle expands and collapses: macro comprehension (types, relations) and micro (instances). **Micro carries ID tags only** — any richer micro channel gives the optimizer a bypass around the bottleneck. | T2v: the tree's ADDRESS STRUCTURE generalizes (FLAT_FREE 0.094 vs composed 1.000); the U-Net shape itself tied a factored codebook with one entity per vector (V2 FAIL) — its case is several entities sharing one vector, untested. |
| 5 | **The Nazaré rejoin** | When macro (rule) and micro (ID tag) phases align, the composite rejoins by constructive interference — the read fires only on alignment. | T2x: the phase rejoin perfect on unseen pairs, +0.819 over learned binding. |
| 6 | **No MLP** (RULING 10-29) | The block is MHA without the feed-forward; no MLP anywhere we build. The MLP memorizes the cross join (the literature's picture: keys stored in combinations of near-orthogonal directions — not tested by us). | Composition law: every learned nonlinear combiner memorizes (0.003 / 0.094 / 0.113 / 0.181). Real data: eq_slot NOMLP still recalled (N1 PERSISTS) — removing the MLP is necessary, not sufficient. |
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
| 3 the notebook: executor in the loop = the ALTERNATION (ruling 10-29) | NOTEBOOK | K1 PASS: trained depth ≤ 3, depth 5–8 at 0.992 (PHASE 0.909, VANILLA 0.731) |
| 7 breath rotation by derivation time | ROTATE | T1 EQUIVALENT, leaning harmful (−0.018 on DEEP, both seeds, ~3.5 SE) on the hard split; NOT in the stack (NOTEBOOK 0.994 vs VANILLA 0.724 at depth 5–9 from depth ≤ 2 training); ROTQ (queries only) the same: V1 HURTS −0.021 at 6 % data, ≈ ROTATE (+0.004) |
| 3 snap to tree centroids in SQLite | TREE | U1/U2 HURTS (−0.029 / −0.023 vs NOTEBOOK); step-type centroid + ID loses the free snap's context; not in the stack. The task has no cross-problem facts for a tree to hold |
| 8 write-time stamp in the notebook | NOSTAMP | S1 CEILING (−0.006): the known bit carries the notebook; the coupling is not needed here |
| 4 + 5 U-Net macro/micro + Nazaré | UNET | last |

## The reader side (NL binding bench, 10-29; templated stories over gold DSL structures; `scripts/nl_reader.py`)

| organ | arm | status |
|---|---|---|
| 4 micro ID tags only (fresh random phase per name per story) | ID | VOID (C0): alone, ID strips gender from every name and fails in-distribution (IN 0.558) |
| 3 + 4 tree macro channel (SQLite name-gender, test-time provisional → promoted) | TREE | IN 0.985, unseen names 0.996 (VANILLA 0.260); partly by construction (the DB holds what weights cannot) |
| 7 + 8 T6 sentence-clock rotation, no absolute positions | CLOCK | K1 PASS: long stories 0.949 vs TREE 0.394 (+0.555) |
| 6 no MLP | NOMLP | N1 THE MLP IS NEEDED (−0.142 on LONG at −61 % params); S PASS: stack beats VANILLA on both axes (+0.734 / +0.441) |
| 6 no MLP, parameter-matched | NOMLPW | N2 EQUIVALENT to CLOCK (−0.007): the N1 deficit was capacity; the MLP is not needed at matched size |

## The reader side on GSM8K text (step 0, 10-29; frozen Llama-3.2-1B L0–L3 states → pointer programs; `scripts/gsm_trunk_reader.py`)

The ladder rebuilt under the no-MLP ruling, every arm ~1.2M params, two seeds; primary read = GSM8K-test per-step accuracy (step_all), a floor-near read (VANILLA 0.111).

| organ | arm | status |
|---|---|---|
| — MLP control (benchmark only) | VANILLA | TEST step 0.111, answer 0.026; CFNAME answer 0.443 |
| 6 attention only, parameter-matched | BASE | B1 EQUIVALENT (−0.005): the MLP buys nothing here either |
| 4 micro ID tags | ID | I1 EQUIVALENT (−0.016 on CFNAME): Llama's context words still carry the name |
| 3 tree (name-gender) | TREE | T1 EQUIVALENT (+0.004): gender does not decide GSM8K operands |
| 7 + 8 T6 clock | CLOCK | K1 EQUIVALENT on TEST (+0.007); CF reads rise (CFNUM 0.591 vs 0.497) |
| ALTERNATION (reader ↔ executor) | NOTEBOOK | A1 EQUIVALENT (+0.006); S NOT ON BOTH: TEST step +0.009, but CFNAME answer **+0.068** and CFNUM **+0.066** over VANILLA on both seeds, TEST answer 0.041 vs 0.026 |

Reading: on a floor-near TEST read no organ separates; the MLP-free + alternation stack is the most ROBUST to numeral and name swaps (unbarred for S, which needed both). Tree follow-ups (`gsm_type_forest.py` B-CEIL FAIL: errors are the right KIND; `gsm_graph_ceiling.py` G-CEIL PASS: an (owner, kind, event) key identifies the gold numeral in 0.745 of keyed operands — the walk's target is a per-problem GRAPH, the query is the open problem). GRAPH arm (ledger 21:29): VOID by its own fit gate (C0 read the LEARNED walk on ORIG, 0.438); reads only — the ORACLE walk beats NOTEBOOK +0.052 TEST step on both seeds, the LEARNED walk loses −0.047 (owner choice 0.59; hard switches cut gold off). Target real, query still the wall; rescue = soft switches (needs the word). Rescue (soft switches, ledger 10-30 08:04): C0 not void; G1 +0.022 (under +0.03: the 10-29 oracle read was mostly the net leaning on gold switches); G2 EQUIVALENT (+0.008, the soft switch is inert — SOFT = OPEN). The learned walk CLOSES on this bench; the query stays the wall (perfect query worth +0.022 TEST step). S-CEIL (the slot as a node, ledger 10-30): FAIL — inherited (owner, event) keys address a gold result uniquely only 0.342; the question's owner ties its placebo (0.898 vs 0.894). GSM8K problems are one or two people doing a few verbs: who/verb is too coarse an address for results. TICK (a T6 tick per step, ledger 10-30 13:03): T1 EQUIVALENT on TEST step (+0.010, at the reproduction floor); lag-1 / numeral-for-slot reads unmoved; UNBARRED: counterfactual answers +0.05..0.08 on both seeds against both NOTEBOOK comparators — a twin is registered-pending.
