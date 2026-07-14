# Mycelium — Agent Brief (2026-07-15)

**Author:** Bryce + Claude · **Deadline:** Dec 25, 2026 · **Target:** MATH-500
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
| **Parser head** | ~3.2M: two slot banks (24 vars ↔ letters positionally, 24 factors) over a 512d waist; bilinear pointers; 6-way ftype (rel/given/mod/sel/pct/fdiv) + `ALG_DUP` arg-multiplicity bit; digits MSD-first ≤999 | `scripts/phase1_algebra_head.py` (envs `ALG2=1 ALG_FTYPES=6 ALG_DUP=1`) |
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
