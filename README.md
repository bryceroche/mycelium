# Mycelium

**A self-certifying natural-language math reasoning system, built on one rule:
neural proposes, symbolic disposes.**

**Author:** Bryce + Claude · **Date:** 2026-07-15 · **Deadline:** Dec 25, 2026
**Target:** MATH-500 (measured, never trained on)
**Platform:** one AMD 7900 XTX (24GB) · tinygrad + AM driver · no ROCm, no CUDA,
no PyTorch, **no external API calls** — everything below runs on one desktop.

> The authoritative agent brief is [`CLAUDE.md`](CLAUDE.md). The chronological
> ledger of every registered prediction, bar, verdict, and law is
> [`docs/phase1_skeleton_spec.md`](docs/phase1_skeleton_spec.md). The
> machine-readable deployed-stack manifest is `.cache/GENERATION.json`.
> The June-era engine docs are archived at [`docs/archive/`](docs/archive/).

---

## The idea

A word problem becomes a **typed factor graph** — variables, and factors like
`add(a,b)=c`, `given(c)=45`, `mod`, `sel`, `pct`, `fdiv` — and splits into two
jaws:

- **CONSTRUCTION (neural, small, supervised):** a ~3.2M-parameter head over a
  *frozen* Llama-3.2-1B L0–L3 trunk reads the text and emits the graph — slot
  banks for variables and factors, bilinear pointers, digit heads. The trunk
  never trains; all learning lives in the head.
- **SOLVING (symbolic, complete, auditable):** a general CSP core (GAC
  propagation, MRV/LCV ordering, forced-only commits) solves the graph exactly.
  Domain knowledge enters only through a predicate registry — the core contains
  zero domain code.

Between them sits a **certification chain** measured link by link:
**mouth → vote → panel → key.** The mouth (input-space OOD recognition on
frozen-trunk geometry) refuses foreign register before parsing. The vote (5
sentence-permutation views, majority) certifies invariance across *diagrams* —
5/5 unanimity measured at **1.0000 precision on 866/1500**. The panel
(cross-model consensus across training lineages and widths) certifies
invariance across *landscapes*, and dissents on ~90% of wild-register parses
that slip the mouth. The answer key grades everything that enters training.
Abstraction may live in annotation and recognition — **never in verification**.

## How it's built: the method is the product

Every capability was grown the same way, and the ledger records all of it:

1. **Registered predictions.** Bars and kill-criteria are pinned *before* any
   measurement. Batteries end with a verdict script that either writes the
   manifest and prints PROMOTED, or prints the kill and touches nothing — the
   word and the JSON are one atomic act.
2. **Instruments before campaigns.** The odometer was straightened (a length
   bias in pooled distance, native control r = −0.825 → −0.024 after
   correction) *two reads before* it would have graded its first book.
   Train/test disjointness is verified **up to graph isomorphism** (canonical
   WL digests; 42 invisible isomorphs found and excluded before any tables froze).
3. **Refusal → mechanism → recipe.** Every failure gets a slot-level autopsy
   before any fix is claimed. Three generations of flat performance on squares
   turned out to be an *unrepresentable* target (the decoder literally could not
   emit `args=[a,a]`); one bit fixed it, 0.32 → 0.75. A "wall family" dissolved
   into an annotation convention (consecutive letters). The schedule probe found
   curriculum training was burning ~2/3 of every budget at scale.
4. **The books campaign.** Real MATH-train prose is annotated by hand into the
   parser's dialect, gated by vote + answer key, and mixed into training as a
   *regime* (2.9% share × 10 reps — measured: regularizes; pure prose fine-tune:
   poison). **Book 2 (n=100) verdict: books scale** — register distance −31.1%,
   census recovery +8 at the pre-pinned bar, and a new in-register record as a
   side effect. December's budget is now arithmetic, not hope.

## Current state (2026-07-15)

- **Parser (gate lineage):** one-shot ANSWER 1149/1500 on the widest fixture;
  factor-exact validation 0.899; 8/8 acceptance probes unanimous at the last
  promoted gate; certification tier at 1.0000/57.7% coverage.
- **The frontier**, counted honestly after every standing door was tried: one
  frame-entanglement family (trunk-level, z = −2.05 — the pathology lives in
  the language↔graph *binding*, which is also the project's central theorem:
  concepts are not recoverable from surface alone or wiring alone) and one
  suspected encoding gap (chained division), plus counted solver-side families.
- **The recursion:** a schema miner proposes recurring subgraph classes as
  macro-factors through a rank-never-admit gate; future books annotate one
  floor up while the key grades every layer in primitives. Books teach the
  system to read books that couldn't be written yet.
- **The June engine** (validated, resting): a breathing-transformer deducer that
  runs coloring, Boolean circuits, and KenKen with one weight set, and a search
  tier proven across coloring/SAT/KenKen/Sudoku with zero core edits. It waits
  as the Alternator's soft-graph backend. See `docs/archive/`.

## Repository map

| Path | What |
|---|---|
| `CLAUDE.md` | agent brief: stack, protocol, laws, substrate |
| `docs/phase1_skeleton_spec.md` | THE LEDGER (chronological, complete) |
| `docs/NEXT_SESSION.md` | cold-start board state |
| `scripts/phase1_algebra_head.py` | the parser head (train/eval/precompute) |
| `mycelium/csp_core.py`, `csp_domains.py` | the symbolic jaw |
| `scripts/book2_*.py`, `harvest_*.py` | the books campaign |
| `scripts/lattice_*.py`, `recognition_mouth.py` | certification instruments |
| `scripts/schema_miner.py`, `knot_matrix.py` | the recursion's instruments |
| `mycelium/factor_graph_engine.py` | the June deducer (validated) |
| `.cache/GENERATION.json` | the deployed-stack manifest |

*The mycelium holds: many threads, one organism, every growth ring recorded.*
