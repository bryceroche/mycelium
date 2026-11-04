# CORE THEORIES — the north star (carved 2026-11-04)

**Author:** Bryce. **Status: CARVED IN STONE.** Bryce, 2026-11-04: *"for perspective, everything should build towards our longer
term vision."* This file is the single canonical copy of the long-term vision. CLAUDE.md §8, the project memory
(`user_core_theories_2026_11_04.md`) and every top-level design doc point here.

**How it governs (unchanged since the compatibility ruling, 2026-08-29):**
- **The theories govern what gets BUILT. Bars govern what gets CLAIMED.** A failed bar closes a *form*, never a theory line (the patience
  charter). Name the next form.
- **Every pin names the theory line it builds toward.** A run that serves no line below needs a stated reason (a control, an instrument,
  a custody check).
- Where a line meets a banked ruling, the reconciliation table below says which built form we follow and why. A conflict is flagged to
  Bryce, never silently resolved in either direction.
- It supersedes the 2026-10-30 statement (memory `user_core_theories_2026_10_30.md`, kept as history). The deltas are listed below.

## The text (Bryce's words, verbatim)

```text
Core Theories 
Building the Bridge between continuous and discrete.  The core rhythm is alternation between continuous and discrete 
LLMs have a solver inside.  Our Sudoku result in July shows models naturally build internal representations of CSP solvers in their weights.  Our the core rhythm includes a call to a CSP solver
Traditional ML trains on the SQL cross join of concepts which is why the entire internet of data isn't enough training data.  Complex planes enable clean binding and unbinding which allows us to avoid training on the SQL cross join. Our phase binding superpower is not about the superposition (we can fit in ~20 items per complex plane before degradation).  It's about clean binding and unbinding via phase angle and conjugate multiplication.  
512 dim representation space compress down to 64 dim at bottleneck Silhouette of Dancer
Traditional transformer MLP stores unique keys of concepts like Michael Jordan via combinations of roughly perpendicular dimensions.  Our goal is to get world knowledge and memorization OUT of the transformer MLP and INTO our tree which is a literal database with coarse to fine grain granularity levels.  
Abstract reasoning is literally zooming out on the Tree DB and merging the macro level understanding with micro level composite keys(slot id, tree level, T9 clock stamp) that bypass unet compression and are merged into macro via skip connection
SNAP cleans up the representation state between multihop reasoning.  Much like autotune recognizes which key the song is in and snaps the note to codebook centroid.  
The T9 clock tracks address and order.  T9 clock timestamp is part of the volatile per problem composite key. 
Beam search across top choices
"Breathing transformer" is MHA without the MLP.  MHA reads text and compresses it down into composite key ( slot, tree level, T9 clock stamp)
Live Oak Tree – BIRCH decides when to add new children nodes on the tree and welford stats manages tree statistics and maintenance and the MHA searches the tree with KNN.  Overfit the facts to the tree.  Memorize aggressively as we’re building a lookup table.  Generalize the neural net reader to search and read
Welford statistics and Birth math maintain and grow the Tree database
Separate storage from reasoning where storage is the permanent tree structure in the database managed by welford stats and birch math. Every data point connected to a correct logic chain is saved in the database to update running welford stats. As the tree grows to maturity we expect the number of new branches to plateau. MHA reads the problem and creates composite keys for each token (slot, T9 clock stamp, tree level). We discard the MLP and use the Tree instead. Our theory is that traditional ML memorizes the cross join and our superpower is clean binding, unbinding, and snap-to-centroid mechanism which allows us to compute long chains of reasoning without degradation of representation space. 
The breathing transformer loops six times with each "breath cycle" rotating the attention space (not the T9 clock dims) by 60 degrees with six turns mapping to a full circle rotation. The T9 clock tracks time over these six "breath cycles" allowing the model to orient itself.  The clock gears are query rotations 30, 60, 180 degrees so the model can learn any possible pattern across the 6-breath cycle.
Unet is zooming out on the tree and reasoning at the category level.  We cannot send full facts through skip connection – only the ID tags
Nazare – the logic rule traveling through the bottleneck and the ID tags through the skip connection need to rejoin in phase.  if they align they stack into a massive signal otherwise cancel out
Unet expand and collapse the representation space each cycle with the silhouette of the dancer (segment and classify) at the bottleneck
Rotate the queries not the attention planes by 60 degrees each breath cycle
Complex planes are key to lossless binding and unbinding.  Decomposition is the namesake of the project: Mycelium
Notebook, T256 rotational bus
3 Rotor stack: Temporal T9 clock, Spatial ROPE, Rotational T256 bus
The conjugate is a mathematical mirror
Mobius - zoom function from Tree search
Diffusion inspired – breath cycles must not specialize to per-step tasks but attention heads can specialize to avoid gradient tug of war
```

## What changed since the 2026-10-30 statement

- **#2, THE SOLVER INSIDE (added by Bryce 2026-11-04, after the carving):** LLMs have a solver inside; models naturally build internal
  representations of CSP solvers in their weights; the core rhythm includes a CALL to a CSP solver.
- **The bridge is the headline.** "Building the Bridge between continuous and discrete" now opens the theories (CLAUDE.md §9 is its
  component-by-component form).
- **The waist is numbered: 512 → 64**, the silhouette of the dancer. The BT's built waist is exactly that (`BT_D=512 BT_DB=64`).
- **The clock is the T9 clock** everywhere the 10-30 text said T6. The built T9 = the T6 gears (180/60/30/15/7.5/3.75°) plus slow gears
  7–9 (`mycelium/rotor_clock.py`; `DR_T6=3 DR_T9NS=1`). One clock family; the breath stamps ride its gears.
- **The composite key is (slot id, tree level, T9 clock stamp).** Token id is no longer listed.
- **The Tree is maintained by Welford statistics and BIRCH math** ("Birth math" in the text = BIRCH), and storage is the permanent tree
  structure in the database.
- **The payoff is stated:** clean binding, unbinding and snap-to-centroid let long chains of reasoning run without degrading the
  representation space.

## Reconciliation with the ledger (where a line meets a banked result; the build follows the right-hand column)

| Theory line | Banked result / ruling | Built form today |
|---|---|---|
| LLMs have a solver inside ("our Sudoku result in July") | **Date: the Sudoku result is v98, 2026-05-29** (breathing transformer, cell acc 0.78 easy / 0.67 medium at step 200; puzzle-level correlation 7 orders above the independent baseline = learned joint MAP inference, loopy-BP-like); June: the breathing deducer byte-identical to the KenKen oracle, general across coloring / circuits / KenKen. Caveat: its per-head masks carried the factor topology (given, not discovered). Neural-guided clean-CSP SEARCH is CLOSED (the symbolic solver does ~all the work on tractable CSPs) | the internal solver = the BT's breaths (learned BP, continuous mirror); the CALL = the symbolic CSP solver each breath (discrete mirror; the alternation ruling); only a proof crosses hard |
| 3-rotor stack: "Spatial ROPE" | **THE RoPE RETIREMENT RULING (11-03):** RoPE retired; the T9-NS reader at two-seed parity (TW2 + CY) | Spatial position = the **T9 clock's offset gears** (a fixed-gear rotary position: RoPE's form, the clock's frequencies). **FLAGGED to Bryce 11-04**: RoPE returns only by his word. |
| "rotating the attention space by 60°" / gears 30, 60, 180 | T2s: one 60° rotation 0.918 vs breath label 1.000; T2t multi-speed 1.000; THE T6 CLOCK RULING (10-26) | The QUERIES rotate (never the attention planes) by the clock's gears 180/60/30/15/7.5/3.75°; six breaths; 60° is one of the gears |
| ~20 items per complex plane | measured ~20–30 keys per VECTOR / bus (crosstalk ~1/√P) | bind on demand; registers for capacity; superposition is not the superpower |
| The SQL cross join | BT8 (11-04): purely factored silhouette (3 independent factors, no per-combination atom) FAIL F1/F2; the diag: each factor decodes at parity, the JOINT falls (lost conditioning) | **BT8b (pinned 11-04):** chained phase binding, kind keys bound to op / op + kind a; 11 shared Welford atoms; no cross join |
| Composite keys bypass the U-Net, merge via the skip | composite-key audit (11-03): only the T6 tick bypassed; slot id + tree level went through the waist | **BT9 (pinned, open):** slot id + tree level + clock tick as phase tags after the waist (the ID skip); to rerun on the GPU |
| Abstract reasoning = zooming out on the Tree | RB8 (11-03): the reader's breaths descend the Tree one level per breath; breaths 2 and 3 do work only where a level enters (RB-D PASS); wild +0.019 over MBR (twin running) | the reading breaths walk the Tree coarse-to-fine; zoom p(op)·p(type given op) in the BT |
| MHA searches the Tree with KNN | G-TREE3: soft keys beat exact (+0.032); A1 TK1 (11-03): kNN over banked leaves FAIL on TEST (+0.0128 / +0.0015) | the reader reads the Tree's rows per breath; KNN-by-state is the named next form (Phase B) |
| BIRCH branches plateau | M-CEIL M3 FAIL as pinned; plateau visible at T 0.1 | shrinkage n/(n+κ) (κ* = 0.5, the data chose ~hard) |
| Every data point on a correct chain updates Welford | custody: the answer key gates every permanent Tree write | wild entities enter PROVISIONAL, promoted on a correct chain |
| MHA without the MLP | NO-MLP RULING (10-29); LLAMA RETIREMENT (10-31); RoPE RETIREMENT (11-03) | the reader: 8M breathing reader on the T9 clock (MBR-8M states = the bench default since P8, 11-04); the BT: attention + Tree + binding |
| Beam search across top choices | BEAM-1 PASS (11-03: +0.0104 TEST answers); BEAM-2 (re-breath) FAIL | parse-side beam, width 8, UNSAT prune (only a proof crosses hard) |
| Alternation / the bridge | ALTERNATION RULING (10-29): NOTEBOOK 0.994 vs VANILLA 0.724 at depth 5–9 | BT ↔ CSP executor each breath; write-back of what is known, never raw values |
| Breath cycles never specialize; heads may | ATOMS + DIFFUSION RULING (10-30) | weight-tied breaths; every breath advances every step |
| Overfit the Tree, generalize the reader | not a conflict | memorize in the Tree; regularize the reader |

## The platform the vision is built on (Bryce, 11-03/04)
- **8M is THE reader line** (iterate fast; ~2% wild cost). **tinygrad on the GPU** for every training run (the BT port, 11-04).
- **No external API calls in the system, ever.** GSM8K text is never committed. MATH-500 is measured, never trained on.
