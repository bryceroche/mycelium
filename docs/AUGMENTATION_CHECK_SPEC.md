# THE AUGMENTATION CHECK — DESIGN SPEC (gating word 1 of 3)
*(2026-07-31, the word given in order: check design → source → dose.
Governed by the coverage-boundary law: the check licenses the paraphrase
classes it can verify; unverifiable classes mint into a separately-
marked pool that never enters training gold — narrow-verified beats
broad-unverified, because the poisoned-gold path is the one failure
custody cannot catch downstream.)*

## The principle: verification by construction, never by inspection

Free-form paraphrase of a minted row cannot be text-side verified at
mint time (the drift classes worth having are exactly the ones no
marker catches). So the check inverts the flow: **nothing free-form
ever touches a row.** All surface variety enters through a LICENSED
TABLE of template-level renderings, and every table entry is
solver-verified equivalent at TABLE-BUILD time — instantiated N=50
times with random values, both renderings parsed to graphs by
construction (they are templates over the same factor schema), all 50
solving equal via solve2 under the corrected uniqueness guard. A table
entry that passes 50/50 is licensed for unlimited use; one that fails
once is refused. Row-level verification then reduces to EXACT
MEMBERSHIP: a rendered sentence either instantiates a licensed
template or the row does not mint.

## Licensed classes v1 (each with its verifier)

| Class | Transform | Verifier |
|---|---|---|
| C1 sentence order | permute non-anchor sentences (roster first, query last) | by construction (permuted_view's own class — content identical) |
| C2 phrasing substitution | template pairs from the licensed table ("a times b equals c" ↔ "The product of a and b is c" ↔ "Multiplying a by b gives c"; "When c is divided by K, the quotient is d" ↔ "Dividing c by K gives d"; per-construction families for add/mul/fdiv/mod/pct/sel/given) | 50/50 solver-equivalence at table build; exact membership at mint |
| C3 number rendering | digits ↔ number-words (≤ twenty + tens) | WORDNUM bijection + value-count equality both sides (the trace layer's own counts) |
| C4 roster relabeling | permute letter↔value assignment, consecutive-from-a preserved | the same permutation applied to factors; answer invariance by construction; regex-complete letter remap verified (no residual old letters) |

## The pen's lawful role (feeds gating word 2)

The pen does NOT paraphrase rows. The pen proposes **table entries** —
new template pairs per construction — which enter the 50/50
verification like any candidate. Pen creativity extends the lexicon;
the lexicon renders the rows. This is the two-axis source built into
the check: mechanical transforms (C1/C3/C4, deterministic axis) +
pen-authored templates (stochastic axis), both passing one verifier,
decorrelated by construction per the budgeting rule.

## The unlicensed pool

Anything outside the classes (free paraphrase, wild-style compression,
notation shifts toward LaTeX) may be minted for STUDY into
`.cache/aug_unlicensed_pool.jsonl` with `"gold_eligible": false` —
never enters a training mix's gold path (load_alg-style guard rides
the mix builder). The boundary is the law's own: narrow-verified in
gold, broad-unverified quarantined.

## The bars (already banked, restated)

Flip rate on the wild ledger's transformed set: primary < 40% (from
53.1%), strong < 27%; mouth-native share on the 1404 rises above 1.6%;
in-register fixtures unmoved (bigtest in band, cert-v2 ≥ 0.998). PLUS
the two fixtures tonight's law created: **dup held-out-config 24% and
fdiv varied-surface 0/8 — the surface-band law's own before-numbers;
an augmented retrain must move them or the augmentation taught its own
table** (the recursion of the surface-band law onto its cure is the
first thing the after-read checks: held-out-of-table templates are the
eval, licensed-table templates are the diet — the eval templates NEVER
enter the table).

## What waits on the remaining words

Gating word 2 (SOURCE): the initial table's authorship split
(mechanical seed set + pen tranche sizes) and the diversity pricing.
Gating word 3 (DOSE): share-of-mix + reps, with the dose-ordering
blind call (books-then-augment vs augment-first) already pinned.

## The unique-reading audit (2026-08-02, the punctuation registration's rent)

The solver's 50/50 verifies the template's GRAPH, never its ENGLISH —
a template whose punctuation admits a second parse would license
cleanly and teach a wrong reading with perfect gold behind it (the
poisoned-gold path through the authoring door). The audit: every
entry's rendering is read for a second parse under a reasonable
reading (attachment + punctuation), by eyes, at licensing time.
v4's 41 audited 2026-08-02: ZERO ambiguous (ledger entry has the
scrutinized cases). The audit rides the licensing of every future
tranche — including the 120-entry slope probe's pen tranche.
Ambiguity is a licensing question no solver check can answer.

**The alphabet-exclusion clause (2026-08-02, Bryce's countersign —
the mechanical form):** a template is safe when the ambiguous variant
REQUIRES A CHARACTER THE TEMPLATE CANNOT EMIT. Punctuation-forced
ambiguity is then checkable by asking what the template's alphabet
excludes, not by reading for parses — the same move the tokenizer pin
made: turn a judgment into an assertion. (`less-than` passes because
"c is 5 less than a" has no degree-modifier reading of a bare numeral
in that position — the flip requires a comma the template never
emits; `when-div` passes because its comma is fixed and load-bearing;
notation admits one reading.) The interpretive read stays for
attachment cases the alphabet can't decide (rem-of's semantic
incompleteness); the clause converts the punctuation class to
mechanical.
