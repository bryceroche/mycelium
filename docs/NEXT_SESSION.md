# NEXT SESSION — start here (handoff, 2026-06-26)

Cold-start entry point. Read this first; it points to everything else.

## Where we are (one paragraph)
The **SOLVING jaw is done** — on clean factor graphs the symbolic search tier solves
exactly + fast, and neural-guided clean-CSP search is **closed** (two-death-mode law:
Sudoku too shallow, QCP value-symmetric; symbolic dominates). The two jaws are
**CONSTRUCTION (NL→factor graph) + SOLVING (graph→answer)**; construction is the unbuilt
frontier and the next chapter. The deducer's role narrowed to a **differentiable critic +
format-definer + soft-graph backend** (its *unique, unclaimed* lane: differentiable solving
through a **per-instance variable-topology** graph — SATNet/FourierCSP can't). And in prep we
found **we already ran a Phase-1 attempt** (`.cache/gsm8k_factor_graphs_*.jsonl`): an
LLM-parser → execution-verified arithmetic-DAGs, ~88% of GSM8K.

## START HERE next session: the FRESH schema-design step
This is the one judgment step we deliberately kept for ourselves (a wrong schema cascades).
Three sharp, grounded questions (detail in `docs/phase1_prep_grounding.md`):
1. **Schema: refine the prior arithmetic-DAG vs adopt MathWorld's container-relation graph?**
   The DAG is execution-proven on GSM8K but *procedural* (forward eval); MathWorld's is
   *declarative* (closer to a real CSP) and may generalize to algebra. Choose against the
   **search regime**, not GSM8K.
2. **Pin the band where solving stops being `eval()` and starts needing the engine** —
   algebra / multiple unknowns / MATH-500. *That* is the real Phase-1 target. **GSM8K is
   forward-arithmetic → a calculator suffices → the engine is overkill there.**
3. **VALUE-PROBE FIRST** (the discipline that paid off all session): does
   `parse → exact-solve` beat **LLM-direct** *anywhere* (e.g. removing the LLM's arithmetic
   slips)? Runnable on the 186 verified test graphs. If not, the two-phase bet is in question
   for ~free — surface it before building a parser.

The other gate to settle early (decides the architecture): **end-to-end** (differentiable
deducer = centerpiece) **vs stage-wise** (parse → harden → symbolic). Hinges on supervision:
gold structure exists only for *easy* problems; *hard* ones (GSM8K/MATH-500) are answer-only.

## Assets in hand (don't rebuild these)
- **Solving jaw:** symbolic search tier (`csp_core` GAC + MRV/LCV + backtrack; `policy_valorder`
  for neural ordering) + the breathing **deducer** (differentiable approximate backend). Domains
  via one bridge each in `csp_domains.py` (coloring/SAT/KenKen/Sudoku/QCP), zero core edits.
- **Prior Phase-1 data:** `.cache/gsm8k_factor_graphs_{train,val,200test_v2}.jsonl` (4432/899/186
  execution-verified) + `gsm8k_phase1_classifier_*.jsonl` (per-variable parse annotations) +
  `*_rejected` (the ~12% + reasons). The 4 ops map onto the existing `cage` predicate.
- **The deducer ckpts / registry / inlet vocabulary** (the format-definer target).
- Network works via `curl` (the `datasets` lib is NOT installed); a fresh GSM8K slice is in
  `.cache/phase1/`.

## The breadcrumb trail (read in this order)
1. `docs/phase1_prep_grounding.md` — the prior schema + the forward-DAG reframe + worked example + the schema-step questions. **The most important read.**
2. `docs/phase1_construction_brief.md` — the construction/solving frame, who-does-what, the design questions.
3. `docs/session_2026_06_26_solving_closed_phase1_pivot.md` — why solving is done + the two-death-mode law.
4. Memory: `project_phase1_prior_attempt_found`, `project_phase1_construction_scouting`,
   `project_neural_guided_search_clean_csp_closed`. CLAUDE.md §8 has the current direction.

## Do NOT redo (closed this session — don't re-litigate)
- Neural-guided search beating symbolic on **clean CSPs** — closed (Sudoku/QCP/coloring/SAT).
- **Width capacity** as a ceiling lever — speed not ceiling.
- **Dual-view / multi-channel** on small grids — speed not ceiling (value symmetry / no new info).
  *(Multigrid/multichannel may revive ONLY in the large/deep-graph reach-limited regime — gated,
  with the reach-vs-capacity probe to confirm.)*

## The honest open risk (keep it in the crosshairs)
The two-phase value proposition is pinched: clean structure exists where problems are *too easy*
(LLM-direct wins), and the factor-graph framing is *strained* where it'd matter (MATH-500 ≠ finite
CSP). The existential question is **"is there a problem band where the engine beats LLM-direct?"**
The value-probe (step 3 above) confronts it directly. Don't build around it — test it.

— end handoff. 🐟→🏛️
