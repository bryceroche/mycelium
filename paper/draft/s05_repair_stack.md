## 5. The Repair Stack and Its Boundary

*(§5 and §6 of the original skeleton, merged: the stack's measured
capabilities, ending on the boundary they stop at. The survivor arc's
narrative belongs to §9.2 and is not retold here.)*

The thesis of this section: **the repair stack is measured to its
boundary, and the boundary is a population, not a mystery.**

**The anomaly portfolio.** Two signals with opposite grammars watch
every accepted parse: cross-view *agreement* (a dense ranker — it
orders the whole population, best single AUC 0.840) and latent
*centroid distance* (a rare flag — nearly silent, but precise where it
fires). Their correlation is a modest 0.464, and combining them loses
whole-ranking AUC while winning at every abstention operating point
actually used (top-10% flags: kept-precision 0.862 vs 0.846) — the
metric-must-match-decision-structure law's fourth sighting, measured
inside our own registration (Table 9-1).

**Withhold-and-solve.** When a parse is suspect, withholding the
lowest-confidence factor and letting constraint propagation re-derive
it recovers 26% of would-be errors *for free* — zero training, and
zero silent-wrong introduced at any withhold depth, because a
re-derived value that contradicts the graph refuses rather than
guesses. This is the solver's exactness used as a repair instrument:
the deduction is only as available as the graph is redundant, and the
mechanism prices exactly that.

**Selective retransmission.** The two-checkpoint design (§3.2's parse
head and repair specialist, one frozen trunk) lets flagged *fields* be
re-asked rather than whole problems re-parsed. The clean result:
field-level structural flags beat gold text-localization as repair
conditioning — knowing *which slot* is wrong outperforms knowing which
*words* are wrong, the structural-entry law's cleanest demonstration,
with the gold-leakage bound measured at zero. At its best measured
operating point the specialist recovered 148 of 627 survivors of the
one-shot pass.

**The stack's half-life.** Repair rounds compose, and their yield
decays hard: 19.6% → 7.7% → 1.1% → 0% across four rounds — a
front-loaded shape observed four independent times (Table 9-1). The
stack is therefore run shallow by design; composed end-to-end it lifted
the then-current system to 47% on one domain and 32% on the other at
convergence. Nothing in the stack pretends to iterate its way through
a wall.

**The boundary.** What remains after the stack is a specific,
counted population, and its character is the finding: the survivors'
internal states are 99.6% correctly decodable — the failure is a
mis-aimed pointer, not a missing fact — and *no downstream mechanism
meaningfully re-aims a pointer* (a perfect oracle ceiling of 13.9%;
trained and input-side repairs in single digits; the arc's full
narrative and its nine registered kills are §9.2). The stack's verdict
on this population is detect-and-abstain, and its inheritance is the
prevention constraint that now governs construction: every new
relation's pointer is born candidate-restricted and span-supervised,
because the place to fix a pointer is before it is trained. The
boundary did not end the repair story; it relocated it upstream.
