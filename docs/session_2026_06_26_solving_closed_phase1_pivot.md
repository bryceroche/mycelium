# Session memo — 2026-06-26: the solving jaw is done; pivot to construction (Phase 1)

**Authors:** Bryce + Claude · **Branch:** `gen-weights`
**Companion:** `docs/session_2026_06_25_capacity_and_dual_view.md` (the capacity + dual-view
first half; this memo is the capstone covering Sudoku → QCP → the strategic pivot).

## TL;DR
We set out to find a domain where the deducer **beats symbolic at solving**, and instead we
**definitively closed that thesis** — across five clean negatives — and in doing so found the
*right* frame for the project. The honest landing:

- **The SOLVING jaw is done.** On clean, hard-constraint factor graphs, the symbolic search
  tier solves exactly + fast (Sudoku 5000/5000 at *median 0 decisions*), and the neural
  deducer adds nothing it can win on. We validated the deducer-as-ordering-prior mechanism
  (net-positive) but proved it's **marginal-to-useless** on clean CSPs.
- **Neural-guided clean-CSP search is CLOSED** — two death modes mapped (below).
- **The two jaws are best drawn at the SYSTEM level: (1) CONSTRUCTION, (2) SOLVING** — i.e.
  the project's own Phase-1 (NL→factor graph) + Phase-2 (graph→answer). The within-solving
  two-jaws ("deducer orders, symbolic disposes") *collapsed* into one (symbolic) on clean CSPs.
- **The deducer's role narrowed to a coherent, unrefuted niche:** a **differentiable, general
  approximate-inference backend** — a *critic* + *format-definer* + *soft-graph solver*, NOT a
  better solver and NOT the NL parser.
- **The next frontier is CONSTRUCTION (Phase 1).** See `docs/phase1_construction_brief.md`.

## The arc (one line)
Sudoku search-tier 100% → neural ordering net-positive but Sudoku too shallow → prey hunt →
QCP (the "least-dead" candidate) → QCP killed (value symmetry) → clean-CSP solving closed →
two-jaws reframe (construction + solving) → deducer = differentiable critic/backend → pivot to Phase 1.

## Findings

**1. The search tier solves Sudoku 100% (candidate-sets + recursive branch).** A 5th domain
via one bridge (`problem_from_sudoku` + `sudoku_registry`), reusing KenKen's `all_diff_pred`
+ `l_alldiff_propagator`, zero core edits. 5000/5000 solved+correct (easy/med/hard),
**median 0 decisions** — GAC candidate-set propagation alone cracks most; backtracking mops the
tail. Bryce's "candidate sets + recursive MCTS → fully solve" is *exactly* the search tier; it
was already built, in its right home (symbolic). (memory: `project_sudoku_search_tier_solve`)

**2. Neural value-ordering: the TWO DEATH MODES (the session's central law).** We wired the
deducer's policy into the search as a value-ordering prior (`csp_core.policy_valorder`, the
AlphaZero-style child prior; orders, never commits — GAC+verify keep soundness). Then measured
it against an **oracle policy** (one-hot at the solution = the *upper bound on any policy*, no
deducer needed — the cheap kill-gate):

| domain | deep tree? | value-ordering matters? | no symbolic incumbent? |
|---|---|---|---|
| **Sudoku** | ✗ shallow (median 0; hardest ≤199) | ✓ oracle 33× | ✓ |
| **QCP @ transition** | ✓ heavy-tailed | ✗ **oracle ≈ symbolic (value symmetry)** | ✓ |
| SAT / TSP / coloring | ✓ | ✓-ish | ✗ CDCL / LKH / DSATUR |

Real-policy check (v98 Sudoku deducer, 0.64 on branchers): net-positive (win 27 / tie 31 /
lose 2), captured 29% of the symbolic→oracle gap — a *methodology* win, not wall-clock (Sudoku
too shallow). **The law:** neural value-ordering needs **deep tree AND value-sensitivity AND no
symbolic incumbent simultaneously** — no clean exact-propagatable CSP has all three. The regime
is **scarce-to-empty**. (memory: `project_neural_guided_search_clean_csp_closed`)

**3. The prey hunt + QCP kill.** A workflow generated/filtered candidate prey against 7
kill-criteria; the least-dead was **balanced QCP/QWH** (Latin-square completion at the phase
transition — Sudoku's *hard cousin*, with only *generic* CSP heuristics as the incumbent, the
one seam). The cheap oracle-upper-bound kill-gate **killed it for ~free**: across n=12/18/25 and
every hole-density, oracle/symbolic ≈ 1.0–1.2 — *perfect* value-ordering gives ~nothing, because
Latin-square **value symmetry** makes the hardness combinatorial/structural, orthogonal to value
choice. (QCP is also *expensive-but-shallow*: costly GAC, not deep search.) This closed the last
clean-CSP escape — without training a deducer.

## The reframe: two jaws = construction + solving
The grouper's oral jaws (seize + orient) = **CONSTRUCTION** (NL → factor graph); the pharyngeal
jaws (swallow) = **SOLVING** (graph → answer). This is the project's documented two-phase
architecture. The session's whole story is the *within-solving* two-jaws ("deducer orders,
symbolic disposes") **collapsing** — on clean CSPs symbolic swallowed both roles. So the
load-bearing two-jaws is at the *system* level, and the status is asymmetric:
- **Solving jaw: BUILT + VALIDATED** (symbolic search tier + deducer as general/amortized backend).
- **Construction jaw: NOT BUILT — the unrefuted frontier.**

## The deducer's narrowed (but coherent) role
After five negatives, the breathing transformer's value proposition shrank from "a better
solver" to a **differentiable, general approximate-inference engine** with three jobs symbolic
*structurally cannot do*:
1. **Differentiable critic** — solve the proposed graph → an end-to-end training signal whose
   gradient flows back into the constructor (you cannot backprop through a symbolic solver).
2. **Format-definer** — its membership + inlet vocabulary *is* the target the constructor emits.
3. **Soft-graph backend** — solves the *uncertain* graphs NL construction naturally produces
   (symbolic needs hard, certain constraints).
It is **not** the NL parser (that needs language comprehension — the bare breathing transformer
hit a reading-comprehension wall on GSM8K, which is *why* v98 pivoted to Sudoku).

## The cathedral, rehomed (hypothesis)
The unbuilt "cathedral" (512-d waist + notebook + coarse-to-fine + per-breath state) was aimed
at the *solving* jaw — where its core justification was **refuted** (cross-breath memory at the
noise floor; the residual already carried the state). But those pieces are *exactly* the kit for
**incremental construction** from language: a **notebook** to accumulate the graph as you read, a
**waist** to compress language→latent, **coarse-to-fine** to scaffold-then-refine. So the cathedral
may have been built for the wrong jaw — its plausible home is **Phase-1 construction**, bolted onto
a comprehension-capable base (NOT the bare breathing transformer). *Hypothesis, not result.*

## Durable laws + reusable methods (the real yield)
- **The two-death-mode law** (above) — when neural value-ordering can/can't help.
- **The oracle-upper-bound kill-gate** — a one-hot-at-solution policy via `policy_valorder`
  bounds *any* policy's benefit, pure CPU, no deducer; settles "is X the prey?" for ~free.
- **Symbolic dominates clean exact-propagatable CSPs, full stop** (3 domains now).
- **"Faster climb, same ceiling"** — capacity (width), transfer, multi-view all accelerate but
  don't raise the ceiling (it's per-round fidelity, not reach/channels). (companion memo)

## Artifacts
- Code (`gen-weights`): `csp_core.policy_valorder`; `csp_domains.{problem_from_sudoku,
  sudoku_registry, problem_from_qcp, qcp_registry}`; `factor_graph_engine` S-agnostic layer +
  dual-view (`channel_messages`, `_channeling_messages`, `primal_s_max`); `kenken_dual_data.py`;
  `scripts/{search_sudoku, search_sudoku_ordering, sudoku_neural_ordering, search_qcp,
  test_general_layer_parity, smoke_dual_kenken_encoding}.py`.
- Memory: `project_{sudoku_search_tier_solve, neural_guided_search_clean_csp_closed,
  dual_view_channeling_result, kenken_width_capacity_plateau}`.

## Next
Open the **Phase-1 (construction) chapter** — see `docs/phase1_construction_brief.md`. The first
question that decides everything: **is the NL→answer signal trained end-to-end (then the
differentiable deducer is essential) or stage-wise (parse → harden → symbolic)?**
