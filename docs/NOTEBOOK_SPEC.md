# THE NOTEBOOK SPEC — CAMBIUM's rings, made concrete (2026-08-03)

**STATUS 2026-08-04: BEGUN — §1 is ALIVE (rings v1 fired, graduated
0.941 in-register, pulsing 0.650/dial-bends at the frontier). THE
ORGAN ORDER, set by the frontier's own compression (a finding, not a
preference): (1) ~~loc anchors~~ ALIVE 08-04 (AUC 0.928, trust 0.908; misbinding
IS mislocated attention — wrong slots 0.063 in-span; artifact banked);
(2) THE REVERSE GEAR — trigger LICENSED 08-04 IN-REGISTER ONLY
(frontier: FP 68.9%, AUC inverted — does NOT travel; recall-vs-read
hypothesis registered)
(0.000% FP / 99.9% agreement held-out at blind thr 0.3648;
**DYNAMICS BUILT 08-05: ALG_XOUT=1, three arms, revoke as input port
(solver-side transport by construction), pinned rates 0.5/0.15,
xrel ledger emitted; smoke PASS ×5 incl. #150's acceptance —
graded 0.933 vs rate-matched 0.493, the fork is three real arms;
init-closed bit-exact; W_cmt live through the release path. The
TRAINING fire (revoke gold from wrong bindings + re-bind
improve-vs-degrade under each arm) awaits its word**); (3) the clock. Each its own fire (#126).**

*(Rung 3's candidate design, drafted so the word lands on a design
rather than a direction. Spec-first; nothing here authorizes a build
or a fire. Ancestors: v24c dual notebook (May, validated, resting) ·
R2 reversible commitments (BINDING_BENCH_BRIEF, three arrivals) · the
ratchet law (#94: engagement MEASURED, never emergent) · #124 (the
diffusion compiler carrying the notebook) · #125's active-inference
clause (an objective for WHEN to revise) · #127 (barriers: flatten
while moving, cherish once arrived).)*

## 1. The two notebooks (per problem, per breath)

- **REPLACE (the cambium):** the working state — per-breath 512d
  waist/fst refinement at all altitudes simultaneously (signature C's
  measured dynamics, June engine; the parser inherits BY MEASUREMENT
  only). Overwritten every breath. This is what exists today,
  implicitly and destructively — why bindings erode.
- **ACCUMULATE (the rings):** the committed record. A ring =
  one committed binding: (slot j, ftype, args/var pointers, value) +
  **its evidence anchor** (the source span it was read from — #69's
  loc attribute, rung 3's second customer) + **its engagement point
  STATED** (the breath index at which it committed — the ratchet
  law's designed-engagement requirement, checkable live by the
  engagement-detection instrument).

## 2. Commit and cross-out

- **Commit:** a binding moves replace→rings when its commit signal
  fires (see §3 — NOT settle). Committed rings stop receiving
  replace-side updates; the pawl clicks at a DESIGNED point.
- **THE RELEASE FORK (#136, 2026-08-04):** cross-out is NOT a switch —
  the transition has dynamics. THREE arms at build (#137): DUMP-TO-ZERO vs
  GRADED RELEASE vs ELASTIC DECAY (leak toward rest; self-resetting;
  most machinery — the mass has no restoring force by construction) (reduce mass, re-settle against the anchor); the
  re-bind improve-vs-degrade rate measured UNDER EACH (it is
  plausibly a function of the release profile). #127's risk on the
  soft arm stated: a release without a barrier may never re-settle —
  AND IT IS INSTRUMENTED: settle reads re-settling directly
  (audit-side, never-tier). Release RATES are PINNED PARAMETERS
  (2-3, scaled to the clock's ~3.5 breaths/layer — slower than a
  layer's resolution is unactionable), never chosen by feel.
  **THE GRADED-ARM PREREQUISITE (#150, 2026-08-05):** the graded arm
  requires a RESISTING TERM (something that opposes the release —
  e.g. the anchor's pull, a re-commit pressure, a floor) or it
  collapses into dump-to-zero at a slower rate: a frictionless
  release goes to zero regardless of rate. At build, the graded arm
  must NAME its resisting term before the fire; the acceptance read
  checks the arm's trajectory is distinguishable from rate-matched
  dumping, else the fork is really two arms, not three.
- **Cross-out (the reverse gear):** a ring is revocable by a LATER
  constraint: solver-side contradiction (unsat on the partial graph),
  trace-layer literal mismatch (the value isn't in the text), or
  cross-view disagreement. Revocation re-opens the slot WITH ITS
  ANCHOR PRESERVED — the re-read targets the span, not the whole
  text (the copy-not-generate affinity: revision is re-reading).
  **THE TRANSPORT REQUIREMENT (#152, 2026-08-05):** per-slot rings
  are a PARTITION, and cross-out is TRANSPORT — a later constraint
  objecting to an earlier commitment must have a path between
  partitions. Fully slot-independent rings give cross-out nothing
  to propagate through; full sharing degrades settle as a per-slot
  signal. The design states WHERE on that axis it sits and WHY,
  before the fire. **LEADING CANDIDATE (word 2026-08-05): the
  solver as transport** — the symbolic side was never partitioned;
  unsat on the partial graph is cross-slot by construction, so the
  objection arrives through the jaw that sees everything while
  rings stay independent and settle keeps its signal. Zero new
  channel, zero diversity cost. THE HONEST CAVEAT: leading on
  ARCHITECTURE, not evidence — whether contradictions arrive often
  enough and specifically enough to drive cross-out is UNMEASURED,
  and the solver is blind exactly where the graph is well-formed
  and WRONG ([1293]'s species) — trace mismatch and view
  disagreement stay in the design for the solver-blind cases.
- **The WHEN (#125's clause):** revise while residual prediction
  error exceeds the cost of revising — operationally: cross-out
  fires only on a named contradiction, never on low confidence
  (temperature⊥truth: entropy never triggers revision).

## 3. THE REGISTER CLAUSE (the spec's hardest constraint, new)

The obvious commit trigger is settle — and **settle is never-tier in
the diagnostic register.** If the loop commits on settle and the loop
is ever trained end-to-end, gradient flows through the commit gate
and the campaign's best mechanism-grade signal becomes a training
target — the Goodhart fence's exact scenario. THE RULE: **the commit
trigger is its own trained head with its own gold** (commit-correct
supervision from solver-verified bindings: gold = "this binding, at
this breath, was final"), OR the loop is never trained through the
trigger (frozen-gate training only). The spec REFUSES settle as a
component; settle remains the instrument that MEASURES whether
commits are landing in basins (audit-side, register-protected).

## 4. What this is NOT

- Not the tower: breaths are fixed-point iteration at constant
  altitude (#69 interp 5, twice-tested); the rings do not lower.
- Not the mouth's business: admission stays trunk-space, pre-parse.
- Not deployed until barred: the key grades primitives; TTA/quorum
  semantics untouched until a vote-semantics window opens
  (permuted_view landmine rides that same window).

## 5. The four customers, mapped

1. **Redirect (#113):** load leaves the beam via rings — committed
   bindings stop competing for waist capacity.
2. **Evidence anchor:** rings carry loc; every commitment is
   auditable to a span (the trace layer gains a per-binding check).
3. **Frontier gate:** anchored-settle at the frontier reads ring
   stability (gated-on-anchor, the rung-2b finding) — measurement,
   not mechanism.
4. **Surface ceiling:** re-reading anchored spans under paraphrase
   is the revision path the surface-band law says one-pass lacks.

## 6. Measurement plan (bars pinned at fire registration, per the
   B1-margin lesson — numbers at smoke-pass, shapes now)

- **Fixtures:** dup-misbinding-under-load (53% population, the bench's
  standing pathology) · pct part/base species (args_wrong 57% floor) ·
  [33]-shape query-binding slips · [42]-shape in-dialect garbles
  (the census's poster) · engagement-point sweeps (does the pawl now
  click AT the operand's arrival instead of before it?).
- **Bar shapes:** (i) misbind-under-load improves at matched dose vs
  a no-notebook control (cont-control protocol: gentle-continuation
  baseline, never restart-vs-notebook); (ii) engagement points land
  at designed positions (engagement-earlier-than-designed = the
  failure the topology exists to prevent, caught live); (iii) B2-
  style no-regression on the standing fixtures; (iv) NO conversion
  bar (ceiling ≠ competence, standing).
- **Instruments owed at build:** the revisability meter (#93's rent)
  and engagement detection (#94's rent) — both pre-registered, both
  audit-side, both in the diagnostic register at birth (never-tier).

## 7. Costs, honestly

K breaths multiply parse-side inference cost ×K (K=2..4 initially);
JIT recapture on loop structure (the zero-arg capture lesson);
the commit head is a NEW TERMINAL — the two-terminal law applies at
build (its own smoke, per the sign-terminal precedent), and the
buffer-spec door should exist BEFORE this build adds a sixth den.
#126 stamp: the notebook is a MACHINERY change — it takes its own
generation; it does not ride E2, E4, E5, or any diet.

## 8. THE ANCHOR-GATED RE-READ (the named successor, 2026-08-05)

Justified from both sides by the XOUT fire: the revolving door
(same-wrong 0.89–0.95, all arms + untrained ctl — release without
changed evidence returns to where it was) and the filler inversion
(new-right collapses exactly where mis-anchoring lives: the re-read
lands at the same shifted address). DESIGN SENTENCE: don't release
and re-read — release and re-read FROM THE ANCHOR'S SPAN.
**THE PERSISTENCE CLAUSE (#153):** anchor pressure is TRANSVERSE
(steering, not braking — mass ops alone are longitudinal and proved
insufficient) and must apply EVERY breath after revocation, not
once: the index stays shifted for the whole row (the filler
finding), so a one-shot re-target drifts back. Bars at its own
registration; the ctl-forced baseline (0.911 same-wrong) and the
filler-row failure site are its inherited fixture and bar.
**THE UPSTREAM FORK (#155), pre-written before the stage-1 read:**
PASS = the arm can be steered (attention is where the decision is
made, at least at re-read). FAIL in the revolving-door shape
(filler new-right still collapsed despite persistent anchor
pressure) = the diagnosis moves UPSTREAM — the shifted index lives
in the position machinery (sent_emb's consumption of shifted
indices), not in attention; steering the grip cannot fix a stroke
generated wrong, and the next instrument points at the waist/head
state, not a stronger anchor term. A second revolving door is a
LOCALIZATION finding, never "anchors don't work."
