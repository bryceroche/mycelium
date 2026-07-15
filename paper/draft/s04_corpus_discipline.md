## 4. Corpus Discipline (how the training data cannot lie)

### 4.1 Solution-first, gate-refused

Every generated problem in this system is built backwards: a solution
is constructed first, then a graph consistent with it, then text. The
inversion is load-bearing because it makes correctness properties
*checkable at mint time* rather than hoped for downstream. Two gates
run on every candidate: a uniqueness gate (ban-and-resolve search under
a fixed decision budget, where budget exhaustion *rejects* — an item
whose uniqueness cannot be certified is not emitted) and a round-trip
gate (the emitted text must parse back to a graph whose solution
matches the one it was built from).

The design principle the gates enforce is that edge cases are made
**unrepresentable rather than handled**. Three specimens, one
principle: quadratic-family items are minted with
perfect-square-by-construction discriminants, which *dissolved* the
no-real-roots policy instead of implementing it; ill-defined selectors
(a "largest" with no unique largest) self-gate as constraint violations
and never reach text; and repeated-argument circuits carry an explicit
no-give mechanism so the givens gate cannot leak the queried value.
Three edge policies, zero new mechanisms — the generator's grammar
simply cannot say the broken thing.

### 4.2 Difficulty as measured axes, and the curriculum tombstone

Corpus difficulty is controlled by named, measured axes rather than
vibes: *teeth* (rendering wildness — how far surface forms stray from
canonical templates) and *bands* (structural depth — variables,
chain length, relation mix). Both axes exist so that claims like
"harder" have coordinates, and so that evaluation fixtures can be
stratified by construction.

One honest sentence owed here: progressive difficulty ordering — the
curriculum — is dead in this system at scale. It won its early
ablation honestly (regime-tagged as such in the ledger), then inverted
when the register mix widened: the schedule probe ran flat-mix against
staged orderings from the same warm start and flat won outright. All
training in the frozen stack is flat-mix; verdicts about orderings
expire with their regime, and this one did.

### 4.3 The grading policy, measured against itself

Before any headline number was quoted, the grading policy was made
uniform and then *audited as an instrument*. The uniform metric is
forced-answer at the query variable (solution-set equivalence, not
string match). Re-grading the then-current fixture under it: a raw 802
one-shot correct decomposed into 5 lucky-unforced answers (bounding the
old metric's luck inflation at **0.6%**) and 797 forced-correct. The
audit's real finding: **132 of 797 (16.6%) of correct answers come from
graphs that differ from gold factor-wise** — right where asked, wrong
somewhere unqueried. That ~17% equivalence class is stable across
independent corpus draws (16.6/17.2) and is treated as a design
parameter of the domain, not noise: it is why the certification chain
(§7) grades answers through the solver rather than trusting graph
match, and why "parse accuracy" and "answer accuracy" are never used
interchangeably in this paper.

### 4.4 Disjointness up to isomorphism

The audit-swarm's first question — does the test set leak into
training? — is answered here at a stricter standard than string
deduplication. Every problem is identified by a canonical
Weisfeiler-Leman digest of its factor graph (Weisfeiler & Leman, 1968;
Shervashidze et al., 2011): two problems with different letters,
different surface text, and different generators that share a knot
share a digest. WL-equivalence is coarser than exact isomorphism, so
treating digest-equal items as identical is *conservative* for this
purpose — the exclusion removes at least every true isomorph. Sweeping every fixture against every
training corpus at this standard found **42 cross-boundary isomorphs**
— items sharing a graph with something across a train/test boundary
despite sharing no text (Fig. 3-a's bottom block shows one such pair).
They were excluded, and the check is now a *generation-bump gate*:
every promotion asserts train/test disjointness up to isomorphism
before any battery number is read. We believe this standard should be
ordinary; it is checkable in any system whose problems have canonical
structure, and it is the difference between "we deduplicated" and "we
know no knot is on both sides of the wall."
