# Mycelium — Agent Brief (2026-07-15)

**Author:** Bryce + Claude · **Deadline:** Dec 25, 2026 (SOFT — extendable
at any time, ruling 2026-08-29: patience over pressure) · **Target:** MATH-500
· **Roadmap:** `docs/rotational_bus.md` §5 (the three-rotor stack)
**Platform:** Shadow Glass (AMD 7900 XTX, 24GB) · tinygrad + AM driver · no ROCm
**No external API calls in the system, ever** (dataset downloads are fine).
MATH-500 is MEASURED, never trained on; the MATH *training* split is the harvest.

**What Mycelium is:** a two-jaws reasoning system for NL math. The **CONSTRUCTION
jaw** (built, July): a small trained head over a frozen Llama-3.2-1B L0–L3 trunk
parses algebra-in-words into typed factor graphs. The **SOLVING jaw** (built,
June): a general predicate-driven symbolic search core (GAC/MRV/LCV, forced-only
commits) solves them exactly; the answer key gates all training data. *Neural
proposes, symbolic disposes* — abstraction may live in annotation and
recognition, **never in verification**.

**Ground truth documents (read in this order):**
1. `docs/NEXT_SESSION.md` — cold-start entry point; current board state.
2. `docs/phase1_skeleton_spec.md` — THE LEDGER: every registration, bar,
   verdict, and law, chronological. If it isn't in the ledger it didn't happen.
3. `.cache/GENERATION.json` — the manifest: machine-readable truth about the
   deployed stack (parser/specialist/centroids/mouth + hashes + waivers).
   **The word PROMOTED and the manifest write are one atomic act.**

---

## 1. The deployed stack (the composed system)

| Component | What | Where |
|---|---|---|
| **Trunk** | frozen Llama-3.2-1B L0–L3 + embed; input-space only; NEVER trained | `mycelium/llama_loader.py`; host CACHED per process |
| **Parser head** | ~3.2M at gen-9b; **12.4M in the research lineage** (ds242: the loop's organs — see `docs/architecture.json`, drawn by `scripts/arch_diagram.py`): two slot banks (24 vars ↔ letters positionally, 24 factors) over a 512d waist; bilinear pointers; 8-way ftype (rel/given/mod/sel/pct/fdiv/macro/frac) + `ALG_DUP` arg-multiplicity bit; digits MSD-first ≤999 | `scripts/phase1_algebra_head.py` (envs `ALG2=1 ALG_FTYPES=8 ALG_DUP=1`; deployed envs = the manifest's) |
| **Gate ckpt** | see manifest; gen-9b era = `phase1_gen9b_head`; reader_v1 (val 0.8989, bigtest 1149) in battery | `.cache/*.safetensors` |
| **TTA vote** | 5 sentence-permutation views, majority ≥3; unanimity 5/5 = certification tier (measured 1.0000 on 866/1500) | `scripts/tta_views.py`, `tta_alg2_dials.py` |
| **Cert-v2** | cross-model×view lattice: gate + armB (lineage axis) + cap2x (width axis); unanimity across LANDSCAPES; wild-register dissent ~85-90% = second wall behind the mouth | `scripts/lattice_member_votes.py`, `lattice_join.py` |
| **Mouth** | input-space OOD recognition (pooled trunk kNN vs training-family bank); **length-corrected** (`.cache/mouth_length_correction.npz` — apply to ALL reads; thr 0.0072) | `scripts/recognition_mouth.py`, `mouth_recal_gen9b.py` |
| **Specialist/NACK** | repair head remined per entourage pass on the gate's organic failures | `scripts/phase1_algebra_nack.py` |
| **Monitor centroids** | kind centroids in the gate's fst space (re-anchor per generation: coordinates ROTATE across gens, aligned cos 0.988) | `.cache/monitor_centroids_gen9b.npz` |
| **Solver** | general CSP core; predicate registry; zero domain code in core; 5000-decision gate budget at mint (budget exhaustion REJECTS) | `mycelium/csp_core.py`, `csp_domains.py` |

Chain of custody: **mouth (register) → vote (diagram-invariance) → panel
(landscape-invariance) → key (truth).** Each link has a named specimen showing
why it can't be removed.

## 2. The generational protocol

- **Registered predictions**: bars/kill-criteria pinned BEFORE measurement;
  honest negatives banked. Mixed tables read by pre-pinned frames.
- **Battery → verdict script → manifest**: the battery checks every bar
  mechanically and either writes `GENERATION.json` + prints PROMOTED, or prints
  the kill and touches nothing (`scripts/gen11_verdict.py` / `reader_verdict.py`).
- **Training regime**: flat mix always (curriculum is DEAD at scale — the
  schedule probe: flat won +88/+116; ~2/3 of curriculum-era budgets were burned).
  Gentle continuation from the gate lineage (restarts jostle basins;
  continuation deepens). Warm-start = pad-warm (never discard a trained router).
- **Entourage duty per promotion**: remine specialist, rebuild centroids,
  recalibrate watchers; re-run the certificate pile (re-audition).
- **Diet**: kind/knot-rehearsal matrix (canonical WL digest = problem ID; mint
  dedups knots; bump gate asserts train/test disjointness up to isomorphism).
  Dose law: declare BOTH share-of-mix AND reps-per-unique.

## 3. The books campaign (the critical path)

Four instruments triangulated **n** (unique annotated rows) as the critical
path. **Book 2 (n=100) verdict: BOOKS SCALE** — odometer +31.1%
(length-controlled, straight-to-straight), disjoint-census slope +8 at the bar,
bigtest RECORD as a side effect (prose at 2.9%×10 reps REGULARIZES the dialect;
pure prose at 340 epochs is poison −243 — tune between).

- **Census pool** = the first 100 in-reach L1-3 harvest problems: a MEASUREMENT
  FIXTURE, never a substrate source. Disjoint reads exclude trained items.
- **Annotation rulebook** (written by refusals): consecutive letters ALWAYS;
  all values incl. answers ≤300; one fdiv per item; frame-strip flags; lexical
  explicitation of KNOWNS allowed, unknowns never gifted.
- **Three lanes**: L1 machine-banked (~1%) / L2 repair (~16%) / L3 surgery
  (~82%). Gate = 5-view vote ≥3 + answer key. v2-retry before fresh surgery.
- **Registry** (the organ's waiting room): certificates carry grammar-version
  stamps; re-auditioned every promotion (families SHRINK on contact — the isq
  door reclaimed distance-formula; consecutive letters dissolved "double-X").
  Standing residents: the **rate family** ([45],[7] — trunk frame-entanglement,
  z=−2.05; the binding-layer pathology) and **chained-fdiv** (autopsy staged:
  derived-value digit path). Kingdom-dissolved confirmed at n=94.
- **The recursion charter**: miner (`schema_miner.py`) finds recurring subgraph
  classes → rank-never-admit gate proposes macro-factors → book N+1 annotates
  one floor up. Rails: macros expand before the solver sees anything (the key
  grades in primitives, always); hand-quota constitutional.

## 4. The law family (violate at your peril; ledger has full forms)

- **THE COMPATIBILITY RULING (2026-08-29)**: select the most COMPATIBLE
  component family, never the highest per-run scorer. Target = the
  THREE-ROTOR STACK (token rotor: frozen trunk RoPE / breath rotor: the T6
  GEAR CLOCK (`mycelium/rotor_clock.py`; ruling 2026-10-26) + pi-cycled attention
  rotation in sync / bus rotor: T^256
  phasor binding as an ACTIVE RECURRENT LOOP). Full form:
  `docs/rotational_bus.md` §0. Bars/kills still govern claims; fit governs
  what gets built forward.
- **THE NO-MLP RULING (2026-10-29)**: nothing Mycelium builds or trains carries a
  feed-forward MLP. Memory = the SQLite tree (`mycelium/tree_db.py`); binding = phase
  binding / conjugate unbind; composition = attention. Exceptions ONLY: the frozen Llama
  trunk (teacher/instrument until a distilled MLP-free student passes its bars — it did, 10-31: RETIRED, see below) and
  MLP/vanilla arms as MEASUREMENT CONTROLS (never built forward, never promoted).
  Evidence: N2 (param-matched NOMLP = CLOCK), the composition law. Ledger 10-29.
- **THE ALTERNATION RULING (2026-10-29)**: the core rhythm is alternation between the
  BREATHING TRANSFORMER and the CSP SOLVER — each breath proposes/derives; the solver
  executes, certifies and writes back what is now known (never raw values into the
  reasoning state); the next breath reads it. Evidence: NOTEBOOK 0.994 vs VANILLA
  0.724 at depth 5-9 from depth <= 2 training (rel-DSL). Definition:
  `docs/breathing_transformer.md`.
- **THE LLAMA RETIREMENT RULING (2026-10-31)**: Llama is retired from the research line: no new Llama forward passes. The
  reader is the MLP-free student NMDB@30k (`.cache/distill_reader_nmdb768x8_s30000.safetensors`; 22.0M, attention + the Tree's
  DB row; frozen-head wild swap 0.3637 ≥ 0.3593 PASS); benches read ITS states (GSM8K: `RD_STATES=gsmb_student_states.npy`).
  Banked teacher artifacts stay as history/instruments; the embedding table is still a lookup loaded from Llama's file (fold+prune
  = how the student owns it); the deployed manifest moves only by a battery. All in on the Breathing Transformer. Ledger 10-31.
- **THE ATOMS + DIFFUSION RULING (2026-10-30)**: the tree stores ATOMS (step types, 2-4 op fragments),
  never per-problem graphs; radius = confidence, unbind = conjugate (never 1/z); the three rotors (T6 / RoPE / T256)
  own DISJOINT complex planes; the silhouette is PER STEP (bottleneck classifies the step against atom centroids, the
  micro keys bypass it); contrastive energy pulls bindings toward centroids; every breath advances EVERY step in
  parallel (discrete-diffusion form; the executor commits what is determined each breath), breaths are weight-tied
  (never specialize), heads MAY specialize. Ledger 10-30.
- **THE T6 CLOCK RULING (2026-10-26)**: the breath rotor is six continuous
  gears, 180/60/30/15/7.5/3.75 deg per tick (periods 2..96) = T^6, every gear
  turning every tick; single source `mycelium/rotor_clock.py` (T6_SPEEDS,
  check_range). It supersedes the SEXTET T^3 (breath hand period 6, parity,
  pass wheel; kept bit-identical as LEGACY for banked bodies). CLOCK-RANGE
  RULE: >= 2 gears whose period exceeds the relative-phase span (one free
  gear gets out-voted: T3a seed 1; T6 landed 10/10 in T3b; T2m FLAT6 0.906
  vs GEARS 1.000). The deployed gen carries NO clock; the research chassis
  (K3/CLK3/SRK3/CRK3) carries ALG_SIXWAVE (a period-6 slot<->sentence
  resonance, sw_g ~0.08-0.15): non-conforming but a PASSENGER (sever
  read 10-26: sw_g->0 and the sextet clock-band sever both < 0.010 paired
  on SRK3/CLK3), so NO replacement run; T6 enters the head with a job
  that needs range (loop > 6 breaths, multi-pass; T3d: the snap/clock's
  job grows with depth).
- **The binding theorem**: concepts are bindings — not classifiable from surface
  (C2), not recoverable from wiring (Brick-M). Schema recognition is parse-side;
  the graph is frame-free.
- **Prose promotions don't move machines**: state the system depends on updates
  in the SAME transaction that creates the dependency, or the check fails loudly.
- **Representability audit**: a metric flat across 3 trainings + a targeted data
  intervention is structurally excluded, not starved (args=[a,a] was
  unrepresentable; one bit fixed three generations of mystery). Two-terminal
  form: emission AND gold feed, or the grad is None.
- **Scope decay**: ablation verdicts expire with their regime (curriculum won
  2026-07-10, inverted at mixed-register scale). Regime tags on all verdicts;
  deliberate exclusions carry expiration tags.
- **Estimator variance masquerades as distance**: pooling variable-length
  evidence inherits a sample-size coordinate ("is this distance or is this n?").
  Mouth reads are length-controlled; vintages never mix (straight-to-straight).
  **THE NUMBER (2026-09-12):** the wild fac-exact read (n=2051 slots, p~0.25)
  has SE 0.0096 (a two-read difference 0.0135); mint (n=3668, p~0.98) SE
  0.0023. A +0.010 wild bar is ONE standard error: wild claims need >= 0.020
  or a seed twin on both sides; paired reads (open vs idle/sever) quote the
  paired SE (`LV_PER_SLOT` + `scripts/paired_read.py`). T1's "record" 0.2706
  fell inside the seed spread (the seed-241 control read 0.2652).
- **Never mix generations' head-space coordinates** (drift = pure rotation,
  aligned 0.988): Procrustes-align or re-anchor. Radius is a consolidation clock
  (two-channel reads: angle=identity, radius=consolidation).
- **Means-vs-overlaps**: aggregate equality masks item-level diversity (cap2x:
  same scores, 24.7% disagreement). Panels recruit by measured behavioral
  distance; diagnostic checkpoints are PANEL-ELIGIBLE, not archive fodder.
- **Audit before diet**: refusals get slot-level autopsies before any corpus
  line claims the fix (saved a gen-12 line twice).
- **Temperature ⊥ truth**: vote entropy reads basin depth, never correctness;
  correct-but-shallow items are self-identified rehearsal targets.
- **Circuit rehearsal, not file rehearsal**: erasure is charged per unshared
  circuit; rehearse KINDS/knots. Depth is not zero-sum under gentle continuation.
- **Percentages smuggle repetition at small n** (dose law); **the texture rule**
  (2 unexplained curve shapes = mechanism probe); **structural entry** (binding
  enters as structure: masks, spans, letters — the pointer law's four remedies).
- **THE MANDATORY-ROAD LAW (2026-09-08)**: every organ enters as a road with
  no bypass, never as an invitation through a gain — under gentle continuation
  the residual already solves the loss and gated organs are voted down from
  birth (mh_gain 0.02→0.0002). Proofs: the seal (channel 0→0.2), the clock (no
  gain, probe 1.0). **THE PRE/POST KNOB LAW**: census every organ's injection
  pre- and post-gain per breath before any claim (gain 0.008 × stamps 1e4 =
  0.4× of the state). **THE HEADROOM COROLLARY (2026-09-09)**: a cooker converts
  headroom = the bypass's removal cost, measured FIRST (sever, read the drop,
  then cook); the mask cooker woke its organ to 0.84× and changed nothing
  because the slot mask's removal cost was zero. Ledger 2026-09-08/09.
  **THE RESIDUAL-SEAL LAW (2026-09-10)**: a cooker converts headroom into
  the new organ only if the seal leaves the residual NO way to carry the
  sealed rows itself — the loss takes the cheapest road, and robustness to
  the seal beats carrying (token cooker: flat floors rose; balanced cooker:
  lane 1's removal cost fell, station 5 never woke; the pressure cooker
  worked because it severed the residual). Sever the residual's own path on
  sealed rows, or the seal teaches survival. Ledger 2026-09-10.
- **Poincaré-Euclidean marriage clause** (for when the ball's flag lifts):
  hyperbolic quantities never enter softmax without a log-map; cosine is wrong
  in the ball.

## 5. Engineering substrate (tinygrad + AM; hard-won)

- Training step: TinyJit + assign-in-place fixed buffers (0.06s/step). No
  `dtypes.float32` literal inside JIT; `scores.clip(-1e4,1e4)`; where()-gated
  NaN guards; single-kernel isfinite. See `memory/reference_tinygrad_am_quirks.md`.
- **Precompute states → disk-backed memmap** (`STATES_NPY`), never a giant RAM
  array (three OOM kills). Trainer reads batches off the memmap.
- **Long GPU work runs as `systemd-run --user` transient units** (own cgroup,
  own log) — background-task reapers kill watchers, not work. Use
  `--property=WorkingDirectory=` + absolute paths (a 4× repeated cd-omission
  taught this).
- **`DEV=PCI+AMD`, never bare `DEV=AMD`** (since kernel 7.0, 2026-09-06): KFD
  exposes the Raphael iGPU (gfx1036) and tinygrad's probe takes it first.
  AM driver patched locally (src/tinygrad f7a6c9b) to survive the post-unload
  PMFW state (Mode1Reset BAD_PREREQ -> full init). Ledger 2026-09-06.
- Trunk host cached per process (was reloading 2.4GB per call). Trunk JIT
  deferred (zero-arg capture recaptures with this layer code; the residency
  smoke's buffer pattern is the recipe).
- Eval-only ckpt loads HARD-ERROR on key mismatch; no silent fallbacks anywhere;
  chain scripts `set -eo pipefail`; mint gates log rejects and knot-dups.
- Generators: mint packs letters consecutively by construction; solution-first;
  uniqueness gate budget 5000 (exhaustion rejects — cannot certify uniqueness).

## 6. The June engine (validated, resting)

The v98-lineage breathing deducer (Pythia-410M L0–L3 shared across K=16 breaths,
per-head masks from `membership`) is byte-identical to the KenKen oracle and
generalizes across coloring/circuits/KenKen (`mycelium/factor_graph_engine.py`).
Depth-PARALLEL deduction scales (K_min ≈ D/4). On clean CSPs symbolic search
dominates (neural-guided clean-CSP search CLOSED — two-death-mode law). The
deducer's Alternator role (differentiable critic / format-definer / soft-graph
solver) and the Poincaré tiers remain spec-stage; the perceiver is retired as
core, sanctioned only as monitor/segmenter. Details: archived CLAUDE
(`docs/archive/CLAUDE_2026_07_04.md`) and `docs/state_of_mycelium.md`.

## 7. Process discipline

- **Commit freely (local, reversible); push is standing practice this campaign;
  sync gen-weights → main at chapter boundaries.**
- **Hold for the word before firing training runs** (GPU cost); zero-GPU reads
  on banked artifacts fire immediately.
- **Offer engineering critique before rubber-stamping** — especially
  enthusiastic relays; the member-selection catch and the dose re-phrasing were
  critique products.
- Bryce wants root-cause fixes, not workarounds. The gut is 16-for-16: when it
  fires, audit — the drawer usually has something in it.

## 8. Bryce's core theories + the Breathing Transformer (design constitution, 2026-10-30)

Bryce's statement of what Mycelium IS (full text: memory `user_core_theories_2026_10_30.md`). Like the twelve hills, it
governs what gets BUILT; bars govern what gets CLAIMED.

**Core theories.**
- Traditional ML trains on the **SQL cross join** of concepts (why the whole internet isn't enough data). Complex planes give
  **clean binding and unbinding** (phase angle + conjugate multiplication), so we never train on the cross join. The superpower is
  NOT superposition (~20 items per bus before degradation) — it is clean, on-demand bind/unbind.
- The transformer MLP stores concept keys (e.g. "Michael Jordan") in combinations of near-perpendicular dimensions. Move world
  knowledge and memorization OUT of the MLP and INTO **the Tree** — a literal database, coarse-to-fine granularity levels.
- **Abstract reasoning = zooming out on the Tree**, merging macro understanding with micro composite keys (slot id, token id,
  T6 stamp, tree level) that bypass the U-Net compression and rejoin via the skip connection.
- **SNAP** cleans the state between hops — autotune: recognize the key, snap the note to the codebook centroid.
- **The T6 clock tracks address and order**; its stamp is part of the volatile per-problem composite key.
- **The core rhythm is ALTERNATION** between continuous neural space and the discrete symbolic CSP solver. Beam search across top choices.
- **Decomposition is the namesake**: mycelium breaks matter into atoms; the atoms are Tree centroids (lego bricks, not whole models).

**The Breathing Transformer.**
- **MHA without the MLP.** MHA reads text and compresses it into composite keys; MHA for reading comprehension, binding for
  composition; keep the MHA reader and build structure at the joins.
- **The Live Oak Tree:** BIRCH decides when to add children; Welford stats maintain centroids at every level; MHA searches it by
  KNN (a filing cabinet with KNN message passing). **Overfit the facts to the Tree** (it is a lookup table; memorize aggressively);
  **generalize the reader.** Storage is separate from reasoning; every data point on a correct chain updates the Welford stats;
  branch growth should plateau as the tree matures. Unknown entities met in wild text enter the DB via a stored procedure.
- **Volatile per-problem composite keys** (slot id, token id, T6 stamp, tree level) are made by phase binding and unbinding and
  snapped on; **phase binding is a network switch** — attention masks open when phase frequencies align.
- **U-Net per cycle:** expand and collapse the representation each breath, with the **silhouette of the dancer** (segment +
  classify) at the bottleneck; only ID tags travel the skip, never full facts. **Nazaré:** the rule through the bottleneck and the
  tags through the skip rejoin in phase — aligned they stack into a massive signal, misaligned they cancel; wave energy funnels
  into the bottleneck, coupled with beam mass.
- **Six breath cycles; rotate the QUERIES, not the attention planes**; the T6 clock lets the model orient itself across them.
- **Three-rotor stack on disjoint planes:** temporal T6 clock, spatial RoPE, rotational T256 bus (+ the notebook). **The conjugate
  is a mathematical mirror** (never 1/z; radius = confidence). **Möbius** = the zoom function for Tree search.
- **Diffusion-inspired:** every breath advances every step in parallel; breaths never specialize; heads may.
- **Data:** GSM8K text, not home-grown problems — the goal is NL comprehension, graded on intermediate gold answers. Build
  something too small to memorize GSM8K that instead pulls from the entity DB by binding; train and test on the same set (the
  counterfactual fixture is how that is graded honestly). Regularization / dropout on the READER.

**Reconciliation with the ledger (where a line above meets a banked result; the build follows the ruling column):**
| Constitution line | Banked result / ruling | Built form |
|---|---|---|
| 60° per breath; gears 30/60/180 | T2s: single 60° rotation 0.918 vs breath label 1.000; T2t multi-speed 1.000; THE T6 RULING (10-26) | queries rotated by the six T6 gears 180/60/30/15/7.5/3.75°; rel-DSL ROTATE/ROTQ cost −0.018/−0.021 (not in that stack) |
| "overfit is the point" + "regularization/dropout" | not a conflict | overfit the TREE; regularize the READER |
| T6 tracks address + order (storage dropped) | clock capacity 10-16: rotation adds no storage; registers do (theta-gamma 200 @ 1.00) | stamp = address/order; capacity = clock-selected registers |
| ~20 items per complex plane | measured ~20–30 keys per VECTOR / bus (crosstalk ~1/√P) | bind on demand; registers for capacity |
| MLP stores "Michael Jordan" keys | literature (superposition), not tested by us; ours: every learned nonlinear combiner memorizes (composition law); attention also recalls (eq_slot N1) | NO-MLP ruling + Tree |
| BIRCH branches plateau | M-CEIL M3 FAIL as pinned; plateau visible at T 0.1 (0.82 of tokens stop above the deepest level, −0.012 R²) | shrinkage N/(N+κ) is the next form |
| unknown entities via stored procedure | custody: the answer key gates permanent tree writes | wild entities enter PROVISIONAL, promoted on a correct chain (the NLB TREE form) |
