title: The paper, in plain language
date: 2026-09-01

# The paper, in plain language

*The project's full technical paper, rewritten so that a curious
reader — not just an academic — can follow every step. Every number
survives from the original verbatim.*


## Abstract

A deployed math-solving system should not just produce an answer — it
should say how much to trust it. This system reads algebra word
problems written in plain English, converts them into diagrams of
quantities and the arithmetic that links them (factor graphs), and
solves them with exact logical search rather than guessing. Every
output is one of four verdicts: certify, answer, flag, or abstain (we
don't know, so we say nothing). The machinery choosing between these
verdicts is entirely fixed and untrained, so training pressure cannot
bend it in the system's favor. Across 1,500 held-out test problems,
the system certifies 912 of them, and every one checked out correct —
a strong, honestly-bounded result. The trained part of the system is
small: 8.0 million adjustable numbers, sitting on 506 million frozen
numbers borrowed from a larger pretrained model that is never
modified. Pointed at an unfamiliar public math benchmark, the same
channel's certificates were correct only 2% of the time (2 of 97) — a
real failure reported at full strength, because it led us to build a
gate that now catches essentially all of that foreign text before
judgment happens. Three hand-written books of practice problems and
fourteen generations of the system got us here, every step recorded in
a public ledger, including a falsifiable bet on how our own
instruments will age.

## 1. Introduction

Today's language models can hand you a math answer without ever
telling you how likely that answer is to be right. The obvious fix —
ask the model how confident it feels, and trust the confident answers
— turns out to measure the wrong thing. We can show this with a single
measurement rather than an opinion.

Our system reads the same problem five different ways: we shuffle the
order of the sentences, which changes nothing about what the problem
is asking, and see whether all five readings land on the same answer.
We measured how "loud" the disagreement is between those five
readings — technically, its entropy, a number near zero when the
readings agree strongly and larger as they scatter. That loudness is
very good at telling a shaky, uncertain reading apart from a settled,
confident one. But it cannot tell a settled reading that happens to be
**correct** apart from a settled reading that is confidently
**wrong**: confidently-right readings measure entropy H = 0.000, and
confidently-wrong readings measure H = 0.212 — almost as quiet
(Figure 2). In plain words: how sure the system feels is a different
question from whether it is right. We call this law **temperature is
orthogonal to truth** — temperature (how settled the system feels)
tells you nothing reliable about truth. Any system that stops at its
own internal confidence has built a depth gauge and is using it as a
compass. This paper is about what to build instead.

What we built is a **certification lattice** (Figure 1): a system
whose output is always one of four decisions — certify, answer, flag,
or abstain — produced by passing every problem through a **chain of
custody**, a sequence of checks in which each new check catches a kind
of mistake the earlier checks cannot. The first check asks: is this
input written the way the system was trained to expect? The second
asks: does the answer survive being reworded five different ways? The
third asks: does the answer survive being solved again by an
independently trained sibling system, one with a different training
history? The fourth check — used only to measure the system, never to
run it — asks: is the answer actually correct, according to the
dataset's published answer key? Every one of these checks is
**zero-parameter**: none of them contains any number that was adjusted
during training, so none of them can be quietly bent by the training
process into flattering itself (Section 6). We can account for the
whole system down to the last connected number: 8.0 million values
were actually trained; they lean on a much larger block of 506 million
values borrowed from a pretrained language model that the system reads
but never modifies (Section 3). At the point where we froze the system
for this paper, the certified channel covers 60.8% of our hardest test
set at a measured precision of 1.0000 — every one of 912 certified
problems, out of 1,500, checked out correct — and Section 6.2 explains
exactly what that "1.0000" can and cannot promise a reader. A diagram
of the full chain — its four gates, the four invariances they check
(register, rendering, lineage, truth), and five real problems traced
through it — shows each gate labeled with the specific failure it is
there to catch, and none of the five example trajectories skips a gate
unscathed.

The paper's second product is not a result at all — it is a **way of
working**. Every claim we make started life as a written prediction
with a pass/fail bar, pinned down before we ran the measurement that
would test it. Promoting a new version of the system into deployment
is decided by a script that checks every bar mechanically: it either
writes the new deployment record, or it writes nothing at all. The
complete, chronological log of every prediction, bar, and verdict —
including our failures — ships alongside this paper as supplementary
material. It is long and unedited, and that is the point: it shows our
mistakes at the same resolution as our successes, which is what makes
it evidence rather than a story we are telling about ourselves.
Section 8 walks through this discipline in full on the hardest problem
the project faced, and Section 8.3 collects the general lessons ("laws")
it produced — several of which, we believe, apply to any system that
has to watch itself for trouble.

The **construction** side of the project (Section 9) is a reading
campaign: three books of real math word problems, retyped by hand into
the system's internal dialect, checked by a gate that would not let
even its own authors cheat — it rejected the lead author's first five
attempts outright. We measured this campaign all the way to its end:
the returns on more hand-annotation flattened out (Figure 9), three
separate kinds of measurement agreed on why, and the rulebook for
writing in the system's dialect was written entirely by the system's
own refusals.

The **limit** of all this is reported at full strength, not softened.
We ran the system on a slice of a public, independently written math
benchmark that neither of us wrote a single word of. The certification
channel — which measured 1.0000 precision on our own test set —
signed off on foreign garbage: certified precision on that benchmark
was just 2% (2 of 97), and the system's rate of saying "I don't know"
barely moved between problems it could and could not likely solve,
meaning it had no idea what it did not know (Section 7). That
measurement is exactly why the recognition gate exists, and today that
gate refuses 100% of that same benchmark's false certificates before
they can happen. We are not claiming this system is competent on that
public benchmark. We are claiming a certified channel whose boundary
is now measured, not assumed — and Section 10, written before any
other section, is where a skeptical reader should go first. It states
plainly what the numbers do not show, including the fact that "1.0000"
is a bound on the error rate near a tenth of a percent rather than a
proof of zero errors, that most of the generation-to-generation
comparisons in this paper are single runs with no repeat trials, and
that the person who wrote the hand-annotated training examples is also
one of this paper's authors.

Here is the map for the rest of the paper. Section 2 places this work
among related ideas. Section 3 explains why the system is built from
two cooperating parts — one that reads, one that solves — and derives
that split from a hard negative result about what "understanding a
math problem" even means. Section 4 explains how the training data is
generated and checked so it cannot lie to itself, including a search
for hidden duplicates that goes well beyond matching exact text.
Section 5 measures the system's self-repair mechanism all the way to
its limit. Section 6 presents the certification lattice in full.
Section 7 is the external benchmark test, reported honestly, wound and
all. Section 8 explains the working method behind every claim in the
paper. Section 9 tells the story of the reading campaign. Section 10
is the honest limitations section. The paper closes with a tally of
contributions and a note on who did what.


## 2. Related Work

We checked every citation below against its actual source before
describing what it says — the same standard of "does the claim match
the artifact" that the rest of this paper holds itself to for its own
numbers.

**Choosing not to answer.** The idea that a system should sometimes say
"I don't know" instead of guessing is old — it goes back to early
pattern-recognition work (Chow, 1957; 1970), and the modern framing of
trading coverage (how many problems you answer) against risk (how
often you're wrong when you do) comes from selective classification
(Geifman & El-Yaniv, 2017, and later work). That risk-versus-coverage
picture is exactly the shape of our Figure 4. Two things set our
lattice apart from this line of work. First, none of our decision
machinery is trained: nothing in the accept/reject decision was shaped
by gradient descent, so it cannot learn to go easy on the very errors
it is supposed to catch — a property we demonstrate by naming a
specific case where things go wrong without it (Section 6.3), not just
by asserting it. Second, most abstention research measures one trained
model against one distribution of data. Our claim is different in
kind: our certification is *bounded to a particular kind of input by
design*, we measured that boundary ourselves, and we reported it as a
headline finding (Section 7) rather than an inconvenient footnote.

**Calibration and asking the model how sure it is.** Techniques for
making a model's confidence numbers more honest after the fact (Guo et
al., 2017), and getting language models to state or self-rate their
own confidence in words (Kadavath et al., 2022), both improve how
*readable* an internal signal is. Our problem with this whole family of
ideas is the one from Section 6.1: our strongest internal signal — how
much the five re-worded views of a problem agree with each other —
cleanly tells shaky readings from settled ones, and then completely
fails to tell settled-and-right apart from settled-and-wrong (H = 0.000
vs. 0.212). Internal confidence measures how deep a groove the model
has settled into, not whether that groove is the right one. We are not
saying calibration is worthless; we are saying it calibrates the wrong
thing for certification, which is why our chain's last two checks look
*outside* the model — at a differently-trained sibling model, and at
the actual answer key — instead of trying to read the model's mind
more carefully.

**Voting, looked at squarely.** The closest cousin to our five-way vote
is self-consistency decoding (Wang et al., 2023): generate several
different reasoning attempts at random and take the majority answer.
Ours differs in three ways that matter. First, our five "views" are
not random samples — they are the exact same problem, reworded five
different ways in a manner guaranteed not to change the answer (we
permute the order of the sentences). This is a technique called
test-time augmentation, long used in image recognition (Krizhevsky et
al., 2012) and later studied carefully for how to combine multiple
views (Shanmugam et al., 2021) — that dedicated study found plain
averaging works imperfectly, and unanimity (all five must agree) is
not averaging. Our agreement measures whether the *reading* of the
problem is stable, not whether a random sampler happens to be stable.
Second, we treat full agreement as a certification tier with its own
measured precision (1.0000 on 912 problems, with the exact meaning of
that bound spelled out) rather than as a trick for boosting accuracy.
Third — and this is the part self-consistency papers rarely reach — we
report exactly where this kind of agreement breaks: on text the system
was never trained to read, all five views get read the same wrong way
by the same miscalibrated machinery, so agreement stops meaning
anything (certified precision crashes to 2%, Section 7). Any
majority-vote system without a check on whether the input is even
in-register will silently inherit that same failure. Self-consistency
papers generally report the win on data the model already understands;
Section 7 is the bill for the case they usually skip.

**Guaranteed coverage (conformal prediction).** A family of statistical
methods called conformal prediction (Vovk et al., 2005; Shafer & Vovk,
2008; Angelopoulos & Bates, 2021) can promise a mathematically
guaranteed error rate — stronger than our own bound — but only if the
data you calibrate on and the data you are tested on come from the
same distribution (a technical condition called exchangeability). That
exact condition is what our foreign benchmark violates, and our
recognition gate exists specifically to test for that violation before
any statistical promise is trusted. We think of the two ideas as
complementary — wrapping a conformal layer around our lattice's scores
is a natural next step — but we note our own worst failure (Section
7.2: foreign text certified with false confidence) is exactly the
failure mode a conformal guarantee, calibrated on in-register data,
would sign off on just as confidently.

**"Propose, then check."** Our two-part design — a learned component
proposes, a separate rule-based component checks — belongs to a
well-established family: turning word problems into equations (Zhang
et al., 2020), turning informal proofs into machine-checkable formal
ones (Wu et al., 2022), training separate verifier models to check
sampled solutions (Cobbe et al., 2021; extended step-by-step in
Lightman et al., 2023), and, in a looser sense, speculative decoding,
where a cheap draft is only accepted after an exact check that
guarantees the final output's correctness (Leviathan et al., 2023;
Chen et al., 2023). We differ where our parameter count in Section 3.2
differs from theirs: our checking path contains exactly zero trained
numbers, start to finish. A trained verifier moves the part of the
system that can be quietly corrupted by training; it does not remove
it. We also draw a hard line on where shortcuts are allowed:
higher-level shortcuts (macro-relations that stand in for common
patterns) are always expanded back into basic building blocks *before*
the solver ever sees them, so the checker's rigor is never handed off
to a learned component.

**Retransmission, borrowed from networking.** Section 5's repair
mechanism is, on purpose, built like an old idea from data
communications called ARQ, or automatic repeat request (Lin, Costello &
Miller, 1984): detect an error, send a negative acknowledgment,
selectively re-send just the broken part, and expect diminishing
returns with each retry. We borrow the vocabulary because the
communications field solved the bookkeeping of "when do you stop
asking again" a long time ago; our contribution is measuring exactly
where that boundary falls in our system (Sections 5, 8.2), not the
retry loop itself.

**Instruments that wear out.** The broader idea that things change
under observation — distribution shift, monitoring systems that
degrade after deployment, and Goodhart's law, memorably stated as "any
observed statistical regularity will tend to collapse once pressure is
placed upon it for control purposes" (Goodhart, 1975; organized into a
taxonomy by Manheim & Garrabrant, 2018) — is well documented elsewhere.
What we could not find already written down is the specific mechanism
in Section 6.4: **any signal you promote into a gatekeeping role gets
selected against**, because the population of errors that survives a
gate is, by definition, the population that already passes it. We also
could not find the design response to that mechanism already proposed:
rotate which check is held *outside* the acceptance pipeline, and make
a falsifiable, dated prediction about your own detector's decline
before it happens. The raw ingredients here — Goodhart's law,
adversarial drift — are old. Stating it as an operating rule, with a
rotation plan and a public bet attached, is, as far as we can tell,
new.


## 3. The Architecture (two jaws, derived)

### 3.1 The binding theorem, and the design it forces

The architecture is best explained through the negative result that
shaped it. In this domain, a "concept" — the idea of, say, a rate
problem, where one quantity accumulates against another over time —
cannot be recovered from either of its two surfaces alone. Read from
the *language* side: two problems that obviously feel like siblings —
a taxi-fare problem and a faucet-filling problem, both built from the
same rate idea (Figure 3) — turn out to share no common structure at
all when you look at their underlying graphs. Their kinship is real
and measurable inside the frozen model's own internal geometry (a
statistical distance of z = −2.05), yet it is completely invisible in
how their pieces connect. Read from the *structure* side, the opposite
failure happens: two problems whose underlying graphs are
mathematically identical — same shape, same "fingerprint" — can wear
surface wording so different that no method that only looks at words
would ever guess they were related. We call this the **binding
theorem**: a concept is the *binding* between a linguistic frame (how
it's worded) and a structural role (what it does mathematically), and
it lives in neither side by itself. Four real specimens make both
directions concrete: on top, one shared wording pattern wearing two
completely different underlying graphs; on the bottom, one shared
underlying graph wearing two completely different surface wordings —
all four items, and both graphs, drawn from the actual banked test
data. The structure-side pair is the product of Section 4's isomorph
audit, which swept for graph-twins hiding across supposedly separate
test sets, and the full measurement record for both directions is in
the ledger.

The two-jaws design is this theorem, turned into an architecture. If a
concept only exists as a binding between wording and structure, then
recognizing and abstracting concepts has to happen on the side that
actually reads the wording — the **parse side**. So the diagram it
produces is deliberately **frame-free**: it records variables,
relationships, and quantities, and it deliberately forgets that the
problem was ever about a taxi. This diagram-of-relationships
representation is a well-known idea in its own right, called a factor
graph (Kschischang, Frey & Loeliger, 2001); what the binding theorem
adds is a rule about where its content is allowed to come from. And
whatever checks the diagram afterward inherits *neither* side of the
binding: the symbolic solver receives only the bare graph, searches it
exactly, and is graded purely against the dataset's answer key. When
bigger shortcuts get invented later (macro-relations, built from
patterns that keep recurring), they are always expanded back down into
basic building blocks *before* the solver ever sees them — the answer
key always grades in basic building blocks. The whole idea compresses
into one slogan: **neural proposes, symbolic disposes** — the learned,
pattern-recognizing part of the system is allowed to notice and
abstract; the part that checks the final answer never is.

This gives the pipeline a clean way to talk about itself, borrowed from
arithmetic. Think of the solver's rulebook — its registry of relation
types it natively understands — as a table of primes. A parsed problem
is then a *factorization*: the parser's job is to express a problem
stated in words as a product of those prime relation-types, and the
frame-free graph is that factorization, written down. Macro-relations
are composite shortcuts allowed for convenience, but they always
expand back to primes before the solver sees them, because the answer
key only ever grades in primes. The canonical "fingerprints" used in
Section 4 to catch duplicate problems are the same idea made rigorous:
a fingerprint is a canonical factorization of the graph, so checking
that no fingerprint appears on both sides of a training/test split is
checking that no prime factorization repeats — not that no wording
repeats. Two problems can share a fingerprint despite using completely
different words, and two problems that read as close cousins can turn
out to share no factors at all. The still-growing rulebook counted in
Section 9 is, in this language, a running list of primes not yet
discovered.

### 3.2 The components at freeze, measured down to every parameter

The construction jaw is a small trained head sitting on top of a large
frozen eye. The **trunk** is the word-embedding table and the first
four layers of a pretrained 1-billion-parameter language model — used
only to read the input, and never trained further. The **parser head**
turns the trunk's internal states into a typed factor graph using two
"slot banks": 24 variable slots, bound one-to-one to the letters
a, b, c, ... in the order they appear, and 24 factor slots for the
relationships between them. A mechanism called a bilinear pointer links
each relationship's arguments to the correct variable slots; a
six-way typing scheme (plus one extra bit marking when an argument is
reused) classifies what kind of relationship each factor is; and
quantities are read out digit-by-digit, most significant digit first.
(This six-way typing is the parser's own surface-level bucketing, not
the full list of relation kinds the solver understands — richer relation
types ride on top of these six buckets through a bridge described in
Section 3.3, which is how the registry can grow past six kinds without
retraining the head's output layer.) The **repair specialist** is a
second head, built the same way, retrained every generation on the
main head's own real mistakes, and consulted only when the five-way
vote (Section 6) cannot agree. Around both heads sits the
zero-parameter decision machinery of Section 6 (the views, the vote,
the cross-model panel, the recognition gate, the flag); underneath them
sits the **solving jaw** — a general constraint-search engine (using
techniques called arc consistency and forced-only commits, over a
registry of relation types) that contains zero learned parameters and
zero problem-specific code anywhere in its core.

Table 1 is the parameter census — counted again at the exact moment
the system was frozen for this paper, against the actual deployed
files, not quoted from memory:

| Census row | Parameters |
|---|---|
| Trained and deployed (parser + repair specialist) | 8,005,722 |
| Trained, whole system — both jaws (the solver adds zero) | 8,005,722 |
| Frozen and leveraged (trunk: embeddings + layers 0–3) | 505,954,304 |

The fact that the second row equals the first is the architecture's
claim, stated in numbers: everything added to make answers *checkable*
— the search engine, the verification logic, the certification
machinery — added zero trainable numbers. There is nothing on the
verification path for training pressure to corrupt, because there is
nothing trainable on it at all. The leverage ratio this implies — 63
frozen numbers for every trained one, 126 against the parser head
alone — states the underlying design bet plainly: a pretrained
model's early layers already know how to read; the trained head only
has to learn where to point. Two sibling heads (4.0 million and 13.8
million parameters, one differing in training history and one in
width) are also deployed, but only as cross-examiners at the
certification tier (Section 6.2): they cast votes, and they never
produce an answer of their own.

### 3.3 What was designed versus what actually got built

An honest paragraph is owed here. The design that survived contact
with reality is narrower, and better, than the one first sketched on
paper. One piece worked exactly as planned: the predicate registry —
the solver's list of relation types — grows by adding a new relation
as a plain rule plus a small parse-side bridge, with zero changes to
the search engine itself, and the registry has in fact grown through
several two-digit-numbered relation kinds this way. Two other planned
pieces died outright. A geometric technique called hyperbolic
embedding, meant to represent the tree-like hierarchy of relation
kinds, turned out to be unnecessary — plain, hard-coded structure
(masks, slot positions, letter identities) did the same job with no
learned geometry at all. And an elaborate "memory" or "notebook"
mechanism, meant to let the system revise its own work, was replaced
by a much plainer signal: the vote abstains, and the specialist answers.
In both cases, a planned *object* was replaced by a measured *action*.
The nouns died; the verbs survived.

### 3.4 Three lessons that now constrain every future design

Three regularities from the campaign function as hard constraints on
future architecture decisions, not as optional commentary (sighting
counts appear in Table 2). First, the **pointer law**: once a pointer
(the mechanism linking a relationship to its variables) points at the
wrong slot, nothing built downstream ever fixes it — so every new
relation type's pointer must be built *correctly from birth*, restricted
to plausible candidates and directly supervised on where it should
point. The five remedies discovered for this (restricting attention,
supervising directly on the span of text, adding a comma, enforcing
alphabetical letter order, and adding "ballast" padding) form a
toolkit applied at a relation's birth, never as a later repair. Second,
the **discovered dialect**: the internal writing style the three books
converged on — always-consecutive letters, explicitly stated known
values, one relationship per sentence — was never designed up front.
It was written one rule at a time, each rule triggered by a refusal
from the parser, and it now functions as the system's intermediate
language. The one *designed* formal language the project tried, early
on, is a tombstone (Section 9); the dialect that actually works was
discovered under pressure. Third, the **two-channel spine**: keeping
wording (the parse side) strictly separate from structure (the graph
side) was an early architectural guess that the binding theorem later
proved was load-bearing. Collapse the two channels into one, and a
concept has nowhere to live except tangled up in both at once — which
is precisely the failure the frozen trunk still shows on its one
chronic problem family (Section 10).


## 4. Corpus Discipline (how the training data cannot lie)

### 4.1 Answer first, wording last, gate-refused if broken

Every practice problem this system is trained on is built backwards.
A solution is picked first. Then a graph consistent with that solution
is built. Only then is the wording generated. This order matters
because it makes correctness something you can *check automatically at
creation time*, rather than something you hope is true after the fact.
Two gates run on every candidate problem before it is allowed into the
system. The first is a uniqueness gate: a search process (ban candidate
answers and try to re-solve, within a fixed decision budget) checks
that the problem has exactly one answer — and if the search runs out
of budget without settling the question, the problem is rejected,
because a problem whose uniqueness cannot be proven is not allowed to
exist. The second is a round-trip gate: the generated wording is fed
back into the parser, and the resulting graph's solution must match the
one the problem was built from.

The design principle behind both gates is that edge cases are made
**impossible to generate**, rather than handled after the fact with
special-case code. Three examples make the same point three times.
Quadratic-family problems are always constructed with a
perfect-square discriminant, which makes "no real roots" a case that
simply cannot arise, instead of a case the generator has to detect and
handle. Poorly defined selectors — for example, asking for "the largest"
when there is no unique largest value — are caught as constraint
violations at construction time and never reach the wording stage. And
problems with a repeated argument carry an explicit mechanism that
prevents the "given values" list from accidentally leaking the very
value being asked for. Three different edge cases, and zero new
handling mechanisms: the generator's own grammar simply cannot say the
broken thing.

### 4.2 Difficulty as something you can measure, and a dead idea

Problem difficulty is controlled along two named, measured axes rather
than by feel. "Teeth" measures how wild the wording is — how far the
surface phrasing strays from a plain template. "Bands" measure
structural depth — how many variables, how long the chain of
reasoning, how varied the mix of relationship types. Both axes exist
so that a claim like "this problem is harder" has an actual coordinate
behind it, and so evaluation sets can be built with a known difficulty
mix on purpose.

One honest sentence belongs here: ordering training problems from easy
to hard — a "curriculum" — is dead in this system at scale. The idea
won its first, early test fairly (a result correctly labeled to that
narrower setting in the ledger), and then reversed once the mix of
problem wording widened: a head-to-head test ran a flat, unordered
mix against several staged easy-to-hard orderings from the same
starting point, and the flat mix won outright. Every training run in
the frozen system uses a flat mix; verdicts about ordering expire along
with the narrower setting they were measured in, and this one expired.

### 4.3 How grading itself was checked, not just trusted

Before any headline number in this paper was quoted, the grading rule
was made uniform, and then the rule itself was checked as if it were an
instrument that could be wrong. The uniform rule is: an answer is
counted correct only if the queried variable is forced to a specific
value (its solution set matches exactly), not by matching a string of
text. Re-grading the fixture in use at the time under this rule split
a raw score of 802 one-shot-correct into 5 answers that were correct by
lucky coincidence rather than being logically forced (which bounds how
much the older, looser metric had been inflated by luck at **0.6%**),
and 797 genuinely forced-correct answers. The real finding came next:
of those 797, **132 (16.6%) got the queried value right while the rest
of the underlying graph differed from the intended one** — right where
asked, wrong somewhere the question never touched. That roughly 17%
figure is stable across independently generated batches of problems
(16.6% versus 17.2% in a separate draw), and it is treated as a real
feature of this problem domain, not noise to be averaged away. It is
the reason the certification chain (Section 6) always grades answers by
running them through the solver, rather than by comparing graphs
directly — and it is why this paper is careful never to use "graph
accuracy" and "answer accuracy" as if they meant the same thing.

### 4.4 No shared problems, checked at the level of structure

The first question any careful reader should ask is whether test
problems leak into training. This section answers it at a stricter
standard than simply checking for repeated text. Every problem is
assigned a canonical fingerprint of its underlying graph, computed with
a well-known graph-coloring algorithm (the Weisfeiler-Leman method:
Weisfeiler & Leman, 1968; Shervashidze et al., 2011): two problems with
different letters, different wording, and even different generating
code, but the same underlying structure, get the same fingerprint. This
method is known to be slightly coarser than a perfect test for
identical structure, so treating two problems with matching
fingerprints as identical is the *conservative* choice for this
purpose — it is guaranteed to catch every true structural duplicate, and
possibly a few near-duplicates besides. Sweeping every evaluation set
against every training set at this standard turned up **42
cross-boundary structural duplicates** — problems sharing an underlying
graph with something on the other side of a training/test split,
despite having no matching text at all. These were removed, and the
check itself is now a standing gate: every new version of the system
must pass a check confirming no shared structure exists between its
training and test data before any of its numbers are trusted. We
believe this standard should be the ordinary one. It is checkable in
any system whose problems have a well-defined structure, and it is the
difference between saying "we removed duplicate text" and knowing "no
structural pattern appears on both sides of the wall."


## 5. The Repair Stack and Its Boundary

This section's thesis in one sentence: **the self-repair mechanism was
measured all the way to its limit, and what's left past that limit is a
specific, counted group of problems — not a mystery.**

**Two watchers, two different jobs.** Every accepted parse is watched by
two signals with opposite personalities. The first is cross-view
*agreement* — how much the five reworded readings of a problem agree
with each other. It is a dense signal: it produces a useful ranking
across the whole population, and on its own it is the best single
predictor available (an area-under-curve score of 0.840, a standard
measure of how well a ranking separates right from wrong — higher is
better, 1.0 is perfect). The second is *distance from a typical
example* in the model's internal space — a rare signal, almost always
silent, but sharply accurate on the occasions it does fire. The two
signals only loosely agree with each other (a correlation of 0.464),
and combining them makes the *overall* ranking slightly worse while
making the specific decision that actually matters — which 10% of
problems to flag for human review — noticeably better (86.2% of flagged
problems are correctly flagged, versus 84.6% using agreement alone).
This is the fourth time in the campaign a general lesson showed up:
your evaluation metric has to match the actual decision you are going
to make with it, not just measure the model in the abstract (Table 2).

**Withhold-and-solve.** When a parse looks suspect, one repair trick
costs nothing to try: withhold the piece of the graph the system is
least confident about, and let the constraint solver re-derive it from
everything else. This recovers 26% of what would otherwise have been
wrong answers, for free — no additional training, and, importantly,
zero cases of a silent wrong answer being introduced, at any depth of
withholding, because a re-derived value that contradicts the rest of
the graph is refused rather than guessed. This trick works because the
solver is exact: it can only recover a value when the rest of the graph
actually contains enough information to pin it down, and the technique
prices exactly that condition.

**Selective retransmission.** Because the system has two separate heads
sharing one frozen trunk (Section 3.2 — the main parser and the repair
specialist), it can re-ask about just the *specific piece* that looks
wrong, instead of re-parsing the whole problem from scratch. The clean
result here: telling the repair specialist *which slot in the graph* is
probably wrong works better than telling it *which words in the
sentence* are probably wrong — even when the "which words" hint comes
from perfect, hand-labeled ground truth. Knowing the broken structural
piece beats knowing the suspicious wording, with the ground-truth
comparison measured at zero information leakage. At its best measured
setting, the specialist successfully repaired 148 of the 627 problems
that had survived a first, unsuccessful pass.

**Repair rounds run out of gas fast.** Repeated rounds of repair can be
chained, but their payoff drops sharply: 19.6% recovered on round one,
7.7% on round two, 1.1% on round three, and 0% on round four — the same
front-loaded shape observed independently four separate times (Table
2). Because of this, the repair stack is run shallow on purpose. Chained
end-to-end, it lifted the system of the time to 47% correct on one
problem domain and 32% on another, at the point where returns
flattened out. Nothing in this stack is designed to iterate its way
through a hard wall — it is designed to know when to stop.

**What's left, and its shape.** After the whole stack has run, what
remains is a specific, counted group of problems, and its character is
itself the finding. The internal state of these survivors is 99.6%
correctly decodable — meaning the model actually "knows" almost
everything it needs to; the failure is a pointer aimed at the wrong
place, not a missing fact. And no mechanism tried, at any point in the
project, meaningfully re-aims a wrongly pointed pointer after the fact:
a theoretically perfect oracle that flags every wrong field only
manages to fix 13.9% of them; trained repair mechanisms and
input-marking tricks recover only single-digit percentages (the full
story, and the nine separate registered attempts that were killed
trying, is in Section 8.2). The stack's verdict on this remaining group
is simple: detect it, and abstain on it, because current machinery
cannot fix it. Its legacy is a constraint that now governs everything
built afterward: every new relation type's pointer must be built
correctly from the start, restricted to plausible candidates and
directly supervised, because the right place to fix a pointer is before
it is ever trained. The boundary did not close the repair story; it
moved the story upstream, to prevention.


## 6. The Certification Lattice

> *Temperature is orthogonal to truth.*

### 6.1 Four decisions, made by machinery nobody trained

The system's output is never just an answer; it is one of four
decisions — **certify**, **answer**, **flag**, or **abstain** — and the
machinery that picks between them is entirely zero-parameter: counting
votes, checking for unanimity, ranking by a simple score, comparing a
distance to a fixed threshold. The trained parts of the system — the
parser and its repair specialist — only ever produce *candidate
answers*; they never produce a verdict. No number anywhere in the
system was trained on the certify/answer/flag/abstain decision, and the
answer key never appears anywhere in that decision process — it shows
up only afterward, to grade the machinery, never as an ingredient of
it. It is this purity of the decision path, not an absence of learned
components elsewhere, that the certification claims in this paper rest
on.

The four rungs of the ladder, with the numbers as first measured:

1. **Certify** — take five solution-preserving rewordings of the same
   input (we simply reorder the sentences), parse and solve each one
   independently, and require all five to agree. First measured at
   0.9982 precision and 38.1% coverage (the fraction of problems this
   rung was willing to rule on at all).
2. **Answer** — a plain majority vote (at least 3 of the 5 readings
   agree), with the repair specialist stepping in whenever the vote
   cannot agree. Combined, this rung answers 71.5% of problems
   end-to-end at 0.833 precision.
3. **Flag** — problems ranked by a combined score of how much the views
   disagree and how far the internal state sits from a typical example;
   the worst-ranked 10% are flagged for review, catching 86.2% of the
   truly bad ones in that flagged slice.
4. **Abstain** — no rung above could force a confident answer, so the
   system says nothing.

Behind these four rungs stands the chain of custody, and each link
answers a different question. The **recognition gate** asks: *is this
input even written in the kind of language the system was calibrated
to read?* — a check made before any parsing happens at all. The **vote**
asks: *does the answer stay the same no matter how the problem is
worded?* — five retellings have to land on the same answer. The
**cross-lineage panel** asks: *does the answer stay the same across
models with entirely different training histories?* — independently
trained sibling models must agree too. The **answer key** asks, simply:
*is it actually true?* Register, then rendering, then lineage, then
truth — four separate invariance checks, in that order, and the rest
of this section explains, with real examples, why none of the four can
be skipped.

Why can't a single internal confidence number replace this whole chain?
Because the most natural one measures the wrong thing. The disagreement
between the five reworded views is very good at telling apart a
*shallow* parse from a *deep* one — in an early pilot measurement (36
problems), correct-but-shaky problems showed strong disagreement (an
entropy score of 0.846) while deeply settled problems showed almost
none. But "deeply settled" and "correct" are two different properties:
deeply-settled-and-correct problems measured an entropy of 0.000, and
deeply-settled-and-*wrong* problems measured 0.212 — the confidently
wrong are almost as quiet as the confidently right. Disagreement
measures how deep a groove the system has settled into, not whether
that groove is the right one. That is the epigraph above, restated in
numbers, and it is the structural reason the lattice ends at an
external answer key rather than at any internal temperature reading:
no amount of introspection substitutes for a check the system cannot
fake by simply being consistent with itself.

### 6.2 The numbers, at the moment the system was frozen

At the point the system was frozen for this paper, its widest test set
(1,500 held-out problems) reads as follows: 1,195 problems answered
correctly on the first try (79.7%); five-way unanimous agreement
certifying at a measured precision of 1.0000; and the full cross-model
panel — unanimity required across all five rewordings *and* across
three models with genuinely different training histories (one differing
in training lineage, one in architectural width, with actual measured
disagreement between them rather than an assumption of independence) —
certifying **912 of 1,500 problems (60.8%) at a measured precision of
1.0000**. Relaxing the unanimity requirement to 4-of-5 or 3-of-5 buys
more coverage at a small, measured cost in precision (0.9925 and
0.9832 respectively, at first measurement). Tracked across the whole
project's history, this certified channel has steadily widened — from
0.9982 precision at 38.1% coverage on its first measurement, to 1.0000
at 60.8% at the freeze point — as the reading campaign (Section 9)
taught the parser the general register of the language rather than
teaching it to pass any particular test.

The statistical fine print, inherited in full from Section 10 and not
softened here: 1.0000 on 912 problems is a bound near zero error, not
a proof of exactly zero error, and 1.5% of test-set problems produce
fewer than five distinct rewordings to vote across. One number is worth
calling out precisely *because* it is not 1.0000: the very first
measurement of this channel contained one broken certificate — 570
problems right, 1 wrong. The lattice's own history includes its own
counterexample, found by the answer key, examined in detail, and fed
back into the design. That is exactly the relationship this project
wants between its machinery and its own failures.

### 6.3 Why each link earns its place: three real specimens

Each link in the chain of custody has a real, named example showing
what happens if you remove it.

**A false certificate really happened.** One problem from the campaign's
own test set — written as plain, natural prose rather than the
system's practiced dialect — voted 5 out of 5 unanimous on a wrong
answer. Five different rewordings, one shared blind spot, wrong answer,
full unanimity. This is exactly the nightmare case theory predicts:
text *outside* the register the system was calibrated on gets all five
rewordings read by the same miscalibrated machinery, so the vote's
assumption that the five views are independent quietly fails. The
recognition gate exists specifically because of this example — it
reads the input's register *before* any parsing happens, and it
correctly identifies this kind of prose as foreign. The gate isn't
decoration; it catches exactly the case the vote provably cannot catch
on its own.

**The second wall.** For foreign-sounding prose that somehow slips past
the recognition gate, the cross-lineage panel provides a second,
independent check. On a batch of raw prose where the main model
produced a stable, unanimous vote, an independently trained sibling
model *disagreed on 9 out of 10 of them* (16 of 19 in a later batch).
On in-register text, this panel is nearly idle — in the 1,500-problem
test set, at the point the panel was first adopted, there was exactly
one stable-but-wrong item, and the panel caught it, one for one. The
panel's real job is exactly the wild, unfamiliar text: even wording
that slips past the gate still has to convince a jury trained on a
different history. Different rewordings of the same problem tend to
share one model's blind spots; different training histories do not.

**A quiet, wrong, confident failure.** One test problem's wording
produced a *consistent* wrong answer across three of the five rewordings
— an error in the answer itself, not in whether the system noticed it
was uncertain, and one that never rises to certification but still
answers with a stable, confident voice. This is exactly why the
"answer" rung and the "certify" rung carry different precision numbers,
and why any deployment that truly cannot tolerate the 0.833 precision
of the "answer" rung has to read only from the certified channel.

### 6.4 Even the watchers wear out

The lattice's own anomaly-detecting signals are instruments, and this
project measured how they age instead of assuming they don't.

One of the geometric monitors — a set of per-category reference points
in the parser's internal space — steadily lost its discriminating power
across successive versions of the system. The investigation found the
actual mechanism, and it was not the monitor's fault: the entire
internal geometry *rotates* between versions. Raw comparisons of
reference points across two versions read a similarity of only about
0.59, as if the two spaces were unrelated — but after correcting for
that rotation with a standard alignment technique (orthogonal
Procrustes alignment), the similarity reads 0.988, with only a small
residual difference left over. The shape of the internal geometry
survives between versions; its coordinates do not. The monitor aged
because nobody told it the sky had turned. The fix is structural, not
statistical: every geometric reference point gets re-anchored at every
new version as standing practice, and any comparison of geometry across
versions happens only after alignment. (The cross-lineage panel is
naturally immune to this problem, because its votes are answers, not
raw coordinates.)

There is a deeper, more general principle underneath this specific
fix. **Any signal that gets promoted into a gatekeeping role becomes
selected against over time** — once a signal joins the set of things
that decide acceptance, the population of errors that manages to
survive is, by definition, exactly the population that already gets
past that signal. Later mistakes get shaped, by selection rather than
intent, to specifically evade it. Because the five-way vote itself is
now part of the accepted headline result, we register here, in
advance, the prediction that agreement-based detection of confidently
wrong answers will get steadily worse across future versions of the
system — not because the mechanism itself is getting weaker, but
because the population of remaining errors is hardening against it. The
design response is a rotation policy: **the portfolio of watchers must
always keep one examiner outside the acceptance path.** Rewording held
that seat until the vote itself was promoted into the accepted
pipeline; today that seat belongs to the external anchor described next
— a held-out set of foreign benchmark problems, graded only by the
answer key and never part of any training or acceptance decision — with
other untested candidates already waiting for the next rotation (a
library cross-check that has never been part of acceptance, and
entirely new kinds of reworded views such as paraphrasing). We believe
this is a general reason anomaly detectors age in any deployed system,
and this project's design plans for the expectation instead of being
surprised by it later.


## 7. The External Anchor (the wound, the funnel, the cure)

### 7.1 The one number written by nobody on this team

The anchor test is a fixed slice of a public math competition dataset —
the MATH dataset (Hendrycks et al., 2021), specifically the 500-problem
subset introduced by Lightman et al. (2023) and commonly known as
MATH-500 — acquired with its answers, measured once, and never trained
on. It is the only test in this paper whose *problems* carry no
fingerprint of this project's authors anywhere: every other test set
is either generated by this project's own code or hand-annotated by
this project's own authors, however rigorously gated. The anchor's
problems were written by strangers, chosen by strangers, and graded by
their own published answers. That is why this paper treats it as the
examiner rather than the exam — and why this section reports the
examiner's verdict first and the fix second, in that order, and at
full strength.

### 7.2 The wound, exactly as it was recorded

Three predictions were written down before this anchor test ran. One
held up: the slice of problems the system could even attempt was
small, as expected. The other two were wrong, and those wrong
predictions are what this section is about. On the portion of problems
where a certificate was even possible, certified precision came out to
**2 correct out of 97 certified** — the very same channel that measured
1.0000 precision on its own test set confidently signed off on foreign
garbage. Worse, the lattice issued **63 certificates on problems whose
answers were not even integers, where a correct certificate was
mathematically impossible**. And worse still, the rate of saying "I
don't know" barely moved between the different categories of problems
(67.5% versus 66.1%): the system had no idea what it did not know.

The mechanism behind this is visible in the vote counts themselves, and
it is the heart of this section. On foreign text, the parser
misreads *consistently* — all five rewordings get read the same wrong
way by the same underlying bias, because reordering sentences only
breaks up template-specific quirks, not a whole-distribution
misunderstanding. **Unanimity certifies that a reading is stable, and
stability only coincides with truth when the input is in-register.**
Every signal in the whole decision portfolio, agreement included, was
calibrated on the training distribution, and text from outside that
distribution breaks the seal without any visible warning sign. Stated
exactly as it was written down at the time: *on foreign text the
lattice certifies stability, not correctness — the certification claim
is bounded to a particular register of input, and the anchor test
measured exactly where that boundary sits.* This was the single most
valuable measurement of the whole project, not an embarrassing
accident: the anchor was built to be the held-out examiner, and it
found the system's missing safeguard on first contact.

### 7.3 The fix: checking the input before trusting anything else

Looked at as a production pipeline, the system had every stage a real
production form has — a form to fill out (the parser), a schema (the
relation registry), a database with referential integrity checks (the
solver) — except the one thing every production form needs: **input
validation**. Every check downstream of the parser validates whether
the resulting graph makes sense; none of them ever asked whether the
input was written in a language the system actually understood.

The recognition gate is that missing validation, and it is deliberately
built with zero trained parameters: it measures the distance, in the
frozen trunk's own internal space, between the current input and the
kinds of text the system was trained on (a nearest-neighbor read
against a calibrated threshold), with no trained component anywhere in
the check (an input check with no trained parts is automatically safe
from the selection-against problem of Section 6.4 — no training
gradient flows through it, so nothing can learn to sneak past it).
Measured against the banked test populations: a perfect separation
score of AUC 1.0000 in both directions; foreign text refused 100.0% of
the time at a strict 1% false-refusal rate on native text; and **all
160 of the anchor test's false certificates were refused at the chosen
threshold**. The gate now signs nothing it genuinely cannot read.

The gate then earned its own honest entry in Section 10, which we
report as part of the result rather than hide. Its distance measurement
pooled evidence across texts of different lengths, and in doing so it
accidentally absorbed a length bias: within a single register,
measured distance correlated with plain text length at r = −0.825, an
instrument flaw discovered by a follow-up audit four days after this
gate was deployed. A correction (dividing out a 1/length term, fit on
native text only) brought that spurious correlation down to −0.024,
and the corrected reading *confirmed the same wall on a straightened
ruler*: foreign text was still refused 100% of the time once length was
controlled for, and the baseline "foreign" reading recalculated from
0.243 (warped) to 0.1871 (corrected). Every mouth reading anywhere else
in this paper uses this corrected version, and the automated check now
confirms the correction is actually present rather than trusting anyone
to remember to apply it.

### 7.4 What lies beyond the wall, reported at its true faintness

What the recognition gate can see past the wall is reported honestly,
including how faint it is. The entire benchmark reads as *foreign* in a
narrow band (0.236 to 0.273, against a native-text threshold of 0.044
on the raw, uncorrected scale) — the whole thing looks like one
different forest, and finer distinctions between subjects inside that
forest are not reliably answerable at this distance. The one ordering
that does hold up is counterintuitive: the more symbol-dense subjects
(heavy on notation) read as *closer* to native, and plain-prose subjects
read as *farther*, because this system's native style — terse,
symbol-dense, fact-by-fact sentences — happens to resemble notation-heavy
text more than it resembles conversational writing. The open hypothesis
this suggests: the language gap here may be more about writing *style*
than about missing vocabulary for new kinds of relationships — which
reorders what to build next (robustness to paraphrasing, before brand
new relation types).

The demand side of this problem was also measured before any of this
was built: 62.2% of the benchmark's problems have plain-integer
answers, and the mix of subjects in the rest tells us roughly how much
the relation registry needs to grow for the follow-up project. Section
10's boundary stands unchanged by any of this — no competence on this
benchmark is being claimed here. What this section claims is a
pairing: **the recognition gate buys honesty now; growing the relation
registry buys real capability later.** The system got its input
validation first; teaching it new registers comes next; and the anchor
stays exactly where Section 6.4 put it — the standing examiner, outside
every acceptance decision, waiting for the next version of the system
with the same indifference it showed this one.


## 8. The Method (predictions pinned first, verdicts written by machine)

### 8.1 The rules

Every substantive claim in this paper began as a *registered
prediction*: a written expectation, its pass/fail bars, and a plan for
how to read a mixed or ambiguous result — all pinned down, in a
chronological written log, before the measurement that would test them
ever ran. This idea is adapted from preregistration and registered
reports in the experimental sciences (Chambers, 2013; Nosek et al.,
2018), applied instead to an engineering project. Three rules give this
practice actual teeth. First, **bars before builds**: a new mechanism
is judged not by whether it feels like it helped, but by whether it
clears a number chosen back when nobody yet knew the answer; a result
that lands between the bars gets read by the plan written in advance,
not by whichever reading is more flattering after the fact. Second,
**state the population you're testing on**: a prediction about how
often errors happen has to say up front which group of errors it is
even about (problems with multiple errors or just one, problems that
already survived an earlier filter or a completely raw sample) —
because five separate times in this project, an unstated assumption
about which population was being measured turned an apparently correct
prediction into a wrong one. Third, **promotions happen by machine**: a
new version of the system is only promoted by running a script that
checks every bar and either writes the new deployment record and prints
"PROMOTED," or prints the failure and changes nothing. The word and the
actual write to disk are the same atomic action; there is no state of
"we decided this passed" that exists only in a conversation or a
write-up. (This rule earned its exact name — *prose promotions don't
move machines* — after an audit found the official deployment record
four versions out of date behind a run of promotions that had only ever
been written about, never executed. That audit is in the ledger, and
the rule has held since.)

The complete log ships alongside this paper as supplementary material.
It is long, unedited, and contains our mistakes at the same level of
detail as our successes — that is exactly what makes it evidence
instead of a story told about ourselves after the fact.

### 8.2 A worked example, followed all the way through

The clearest way to show this method is to walk through it in full on
the hardest problem the project faced. After the parser had mostly
converged, a group of *confidently wrong* parses remained: accepted
with full confidence, wrong, and surviving every filter in the repair
stack. A wall like this invites a tidy explanatory story; the actual
method instead requires a sequence of registered, individually testable
hypotheses, each with its own bar, run one at a time. Nine of them ran
in order. Five were killed outright: rendering quality was checked and
found uniform across the surviving errors; how often a value was
mentioned multiple times in the wording was checked and found flat;
"blindness to omitted information" was checked and found dead as an
explanation; transplanting suspicion from one part of the problem to
another was checked and found flat; and enriching the internal
representation of how things bind together was tried and made results
worse, not better. The sixth hypothesis found the real mechanism: the
surviving errors' internal state was **99.6% correctly decodable** —
the model "knew" almost everything — with the actual failure isolated
to a mis-aimed pointer (the same routing problem named in Section 5).
The last three hypotheses then priced how fixable that pointer problem
actually was: a theoretically *perfect* oracle that flags every wrong
field only repaired 13.9% of cases; a monitor-driven self-repair loop
leaked errors, was fixed, and then bought back 6%; and marking suspect
input spans directly bought 3.0% more, but only at a precision of
0.165, too low to be useful. This whole arc compresses into one line:
**nine registered hypotheses tested, four general lessons learned, two
speculative build efforts retired, one working instrument produced, and
one group of hard failures fully characterized rather than mysterious.**

The verdict from this arc — that this specific group of problems is
"detect it, then abstain" under today's machinery — is a measurement,
not a defeat, and it did real work a tidy story never could have: it
stopped two speculative engineering efforts before they consumed the
rest of the project's time, produced the anomaly monitor as a useful
side effect, and wrote the design rule that governed everything built
afterward (every new relation type's pointer must be built correctly
from birth, restricted to plausible candidates and directly supervised,
because pointer errors are never fixed downstream of the pointer —
observed five separate times). A reader who follows just this one arc,
hypothesis by hypothesis, has effectively seen the whole method; every
other chapter of the project's log runs the same shape at a smaller
scale.

### 8.3 What this discipline actually produces: general lessons

The recurring product of this whole process is not accuracy numbers —
it is **general lessons**: failure patterns and design constraints
observed often enough, or well-understood enough mechanistically, that
they now govern how future parts of the system get built. Table 2 lists
the current working set; full write-ups and every individual sighting
are in the project's log. The general shape of each lesson travels to
new situations; the specific numbers attached to each sighting do not
(Section 10).

| Law (short form) | Sightings | Note |
|---|---|---|
| Metrics must match the decision structure they serve | 4 | 4th was inside our own registration (AUC vs tail abstention) |
| Pointer errors are never fixed downstream of the pointer | 5 | 1st at training, 5th at inference |
| Predictions must state their density regime / population | 5 | unexamined populations flip verdicts |
| Repair recovery is front-loaded; round 4 ≈ 0 | 4 | independent sightings across domains |
| A selection criterion's jurisdiction is the property it selects on | 3 | "survived filter X" ≠ repairable |
| Acceptance criteria must be measured, not assumed | 3 | third confirmation closed the beacon |
| Binding enters as structure (masks, spans, letters), never as prose | 2+ | the pointer law's five remedies, descending cost |
| Prevention beats repair for confident wrongness | 2 | representational pressure, not decode-side fixes |
| Prose promotions don't move machines | 1 + audit | the stale-manifest finding; rule mechanized |
| Estimator variance masquerades as distance | 1 + mechanism | length correction r = −0.825 → −0.024 |
| Latent drift is rotation, not decay — align or re-anchor | 1 + mechanism | Procrustes 0.59 → 0.988 |
| Any signal promoted to a gate becomes selected against | 1 + prediction | the standing bet, Section 6.4 |
| Temperature is orthogonal to truth | 1 + mechanism | vote entropy reads depth, not correctness |

### 8.4 Turning the method on itself

This whole discipline's real credibility test is what it caught in its
*own* instruments. Three real defects were found and fixed by this same
registered-audit process, applied to the tools doing the measuring
rather than to the system being measured (Section 10): a length-biased
distance estimator, rotated monitor coordinates, and a temperature
calibration shaped by the training mix rather than by truth. Each audit
followed the same pattern: a registered test whose different possible
outcomes were tied, in advance, to different explanations (for the
rotated monitor: *re-anchor it and re-measure; if the accuracy comes
back, rotation was the whole story; if it's still degraded, something
else is also hardening against it*). Every instrument in this project is
treated as something that ages right along with the system it watches,
recalibrating the watchers is a standing duty at every new version, and
the plan for rotating which examiner sits outside the acceptance path
is published openly in Section 6.4.

### 8.5 Who did what, honestly

The project ran as two channels plus one adjudicator. One channel
(Claude) designed and registered things: predictions, pass/fail bars,
plans for reading results, and skeptical pushback on its own sibling's
enthusiasm. A second channel (Claude) built and measured things:
implementations, test batteries, the project log, and this paper's
drafts. The human author directed and made the final calls: every
training run started only on his explicit word, every promotion and
every kill was his to accept, and twenty separate times during the
project he voiced a pre-verbal hunch — "we're missing something about
hash collisions," "about key-value pairs," "about palindromes" — that
was then turned into a formal, registered check before anyone trusted
it. All twenty of those checks found something real. We report that
streak not as praise for intuition on its own, but as a product of the
discipline that made it *checkable*: a hunch that has to survive formal
registration and a mechanical measurement is data. The same hunch
applied straight to the system, with no such check, would only ever be
an anecdote.

The method's last and clearest exhibit is the bet it places on itself:
Section 6.4's registered prediction that this project's own
certification instrument will age, with its replacement already lined
up and waiting. A method that expects, in writing, to be wrong in
specific and named ways is the strongest form of confidence we know how
to state.


## 9. The Reading Campaign (the library and the librarian)

Section 8 showed this discipline killing bad hypotheses one at a time;
this section shows it building something instead. The target was the
register wall from Section 7: real mathematical prose, written by
strangers, that the system could not read. The instrument was
annotation — three books' worth of hand-written translations of real
training-set problems into the system's own dialect, each one passed
through a gate that its own author could not talk his way past. The
results come in five measured beats.

**The economics of the work.** Problems sort into three lanes, based on
what they actually need: bankable by machine as-is (about 1%),
repairable by the specialist under the vote (about 16.5%), and needing
full hand surgery (about 82.5%) — rates that stayed stable between the
curated test pool and a fresh, unfiltered sample (a 400-problem draw
split 4 / 66 / 330 across the same three lanes), which made the cost
of the whole campaign predictable from its very first week. When a
later, stronger version of the gate re-classified that same fresh
sample, the repair lane had grown from 16.5% to 35% at the expense of
the surgery lane — proof that the accumulating rulebook was doing real
work — so this three-way split is not a fixed property of the problem
domain; it is a moving readout of how much the system has already
learned. Throughout, the curated test pool itself was never annotated
or trained on: the ruler used to measure progress never became the
thing being measured.

**The gate, tested on its own authors.** The gate is simply five-way
rewording agreement plus the dataset's own published answer, applied
mechanically with no exceptions — and its very first day makes concrete
the point Section 10 makes in words: the lead author's first five
attempts at writing sample annotations were **all rejected, zero
banked**, and that zero was the system working correctly. The
rejections were genuinely useful: they showed that a human writer's own
natural dialect was itself out of the system's register — miniature toy
problems, values outside the trained range — and that the parser's
pointer behavior wobbles in exactly those spots. The fixes those
rejections forced became official annotation policy. The gate went on
to catch its own authors three further times, rejecting annotations
whose numeric values silently exceeded the solver's working range
(three specific problems, one of which had stood for days, mistakenly
suspected to be a genuine model failure, before the audit revealed it
was the annotator's own error). A gate that cannot be talked into
passing bad work by the very people who built it is what gives this
project's data the right to call itself trustworthy.

**A rulebook, written entirely by refusals.** This project's one attempt
to *design* its internal dialect from the top down, in advance, died in
a registered failure early on; the dialect that actually works instead
grew from the bottom up, one refusal at a time. Every rule in the
annotation rulebook traces back to a specific wall the parser refused
to cross: scattered, non-consecutive variable letters produced the
always-consecutive-letters rule; values outside the trained numeric
range produced the stay-in-range rule; chains of floor-division
produced the one-division-per-problem rule and a specific recipe for
routing multiplicative inverses; wording that tangled together frame
and structure produced explicit frame-separation flags. The clearest
sign that this process had matured is what we call its **mystery
half-life**: starting from the second book onward, every single
refusal resolved into exactly one of three categories — a new recipe, a
new registry entry, or a plain annotator mistake — within a single
working session. Nothing stayed unexplained for longer than that, and
the counts in each category were tracked, not just remembered.

**Confirming why the books actually work, three separate ways.** That
the books helped is one claim; *why* they helped was confirmed by three
independent kinds of measurement that agreed with each other. As a
teacher of the system's general register: the recognition gate's own
distance reading improved by **31.1%** relative (from 0.1871 down to
0.1288, controlling for text length, comparing corrected readings to
corrected readings), and the curated test pool's disjoint score
improved by roughly 8 problems for every 100 annotated unique rows
added, right at the pass bar. As a regularizer: at a 2.9% share of the
training mix and ten repetitions per unique example, the added prose
actually *improved* performance on the generated-problem test set — a
new best score on that benchmark arrived as a side effect — while the
exact same prose at a much higher, saturating dose actively hurt
performance (a drop of 243), meaning the correct *dose* of prose, not
just its presence, is what carries the benefit. As rehearsal: naturally
varying prose paid down a kind of calibration debt no purely generated
training data had ever paid — vote disagreement on small problems
dropped from an entropy of 0.212 down to 0.010, a specific prediction
that was written down before the training run that produced it. Three
different effects, three different kinds of measurement, and no shared
failure mode between them.

**Measuring the campaign's own end.** The campaign then measured when
it was done. The rate of newly solvable problems in the curated test
pool leveled off: roughly 23 previously-unreadable problems were
recovered for the first ~114 uniquely annotated examples, and
essentially 0 more were recovered for the next 74 examples drawn from
the same distribution of problems (with the honest scope of that
finding — one specific distribution only — kept front and center). The
registry's own waiting room emptied out the same way: one specific
problem family that had been an unsolved mystery across three
consecutive versions of the system (a rate-based problem tied to a
measurable quirk in the frozen model's own geometry) finally became
routine, ordinary work under the final gate, and the campaign closed
with its count of confirmed structural mysteries at zero, pending a
larger volume census — every remaining wall had become either a recipe,
a new registry entry, or ordinary plumbing, except for one candidate
named explicitly in Section 10. The books did not just teach the parser
to read; they re-tuned its pointers, regularized its internal dialect,
and wrote their own rulebook along the way. The library taught the
librarian.


## 10. Honest Limitations

*(This section was drafted first, before any other section, on
purpose: a paper whose entire spine is honest measurement has to lead
with what it cannot claim.)*

**What "register" means, and its scope.** Every capability this paper
certifies lives inside one language family: algebra word problems, with
whole-number values between 0 and 300. At the point the system was
frozen, its own test battery reads that register at 1,195 correct out
of 1,500 on the first try, certifies at a measured precision of
1.0000, and answered roughly 2% of *foreign* benchmark prose correctly
— the exact measurement that motivated building the recognition gate
in the first place. That gate exists precisely because this boundary
is sharp: the system refuses what it cannot read, and the refusal
itself is the actual product. Competence on the MATH-500 benchmark is
explicitly not claimed anywhere in this paper; it is the subject of a
planned follow-up project, and none of this paper's certification
results depend on it.

**The remaining frontier is counted, not solved.** After three
annotated books and fourteen versions of the system, 58 of the
86-problem curated test set remain unreadable (the frozen version's own
count reads 61; the difference of 3 falls within normal voting noise).
This remainder is not a mystery — it is sorted by family and priced.
Inside the test set: relation types still awaiting a registry entry
(primality, greatest common divisor and least common multiple,
logarithms, exponent rules — each one a counted, waiting certificate),
negative and fractional value ranges, and one suspected structural
mechanism — chained floor-division, whose original example problem was
later resolved by rewriting its annotation, leaving one surviving
unexplained refusal and an open question of whether the real boundary
is mathematical or just a quirk of the current wording rules.
Separately, and across the whole harvested problem set: a counted
value-range family bounds what the domain can express at all — 75 of
1,743 harvested problems (4%) have answers above the solver's 0–300
ceiling, and raising that ceiling was evaluated and deliberately
declined, given how few problems it would actually unlock. The one
chronic frame-entanglement problem family measured inside the frozen
model (statistical distance z = −2.05) partly dissolved under a
retrain from a cleaner lineage; what's left of it is bounded, but its
underlying mechanism — wording and structure getting tangled together
inside the pretrained model's own representation — remains a real limit
of using a frozen trunk at all.

**The saturation curve was measured for one distribution only.** The
reading campaign's own yield curve — 23 test-set items recovered for
the first ~114 uniquely annotated examples, ~0 recovered for the next
74 drawn from the same distribution — measures the completion of *that
one slice's* teachable content. It does not prove that annotation in
general has been exhausted: harder problem categories, new styles of
prose, and problems that only exist once the registry grows further are
all different distributions with their own unmeasured curves. This
curve flattened out; the library as a whole did not close.

**The person who wrote the answers also built the system.** The books'
gold-standard annotations were hand-written by this system's own
builders, not by independent, disinterested annotators. Two design
choices limit how much this could have corrupted the results: every
annotated problem is graded by the symbolic solver against the original
dataset's own published answer, before it is ever used in training (an
authority the annotator cannot talk past — a wrong or leading annotation
that changes the final answer is rejected automatically), and a
separate hand-quota rule caps how much of any one book's machine-lane
problems can be self-selected by the same authors. But the answer key
only verifies *correctness*, not *representativeness* — the writing
style, the specific vocabulary chosen to spell things out, and which
refusals got fixed and which didn't, all carry the authors' own hands
in them. "Gold" in this paper means author-written and
answer-key-verified, and readers should weigh any claims about
generalization with that in mind.

**Statistical honesty on the headline number.** A certified precision of
1.0000 on 912 problems is what statisticians call a zero-numerator
result: it bounds the true error rate down near a tenth of a percent,
but it does not prove the error rate is exactly zero. The actual claim
being made is structural: that requiring agreement across rewordings
and across training lineages, behind an input-register gate, produces a
channel whose failures are rare and whose mechanism is understood — that
is the claim, not "this number is literally perfect." In addition, 23
of the 1,500 test-set problems (1.5%) produce fewer than five distinct
rewordings to vote across; all of them were still certified correctly,
but their certificates rest on 3 or 4 effective independent views
rather than the full five.

**Every version-to-version comparison is a single run.** No version of
the system was ever trained twice with different random seeds; every
comparison in this paper between one version and the next is based on
exactly one training run each, with no error bar around it. The overall
discipline reduces the risk this creates, but it does not eliminate it:
pass bars were written down before each measurement, verdicts were
checked against multiple independent test sets, and any claim about an
underlying mechanism required confirmation from at least two independent
kinds of measurement before it was written down as settled. But run-to-
run variation from random initialization alone was never separately
measured, and a careful reader should treat any single version-to-
version difference as an observation made under a disciplined process,
not as an estimate with a known margin of error.

**Our own instruments age too.** Every geometric measuring tool in this
system was calibrated inside some specific version's internal space,
and the project's own audit trail records three real defects found and
fixed in these tools during the project itself: a distance measurement
biased by text length (a correlation of r = −0.825 within a single
register before correction, −0.024 after), coordinates that rotate
between versions (raw similarity 0.59, aligned similarity 0.988), and a
confidence calibration that was only accurate where the training mix
happened to put its weight (small-problem disagreement of 0.212 versus
0.003 across different model lineages, later closed to 0.010 by the
books themselves). We report all three because the actual central
contribution of this method is the audit discipline itself, and a
careful reader should expect more undiscovered members of these same
families of problems.

**Everything here ran at one particular scale.** All of these results
were produced on a single consumer-grade graphics card, with a
4.0-million-parameter trained head (8.0 million once its repair
specialist is included) sitting on top of a frozen roughly
506-million-parameter, four-layer slice of a larger pretrained model
(re-counted at the exact moment the system was frozen; see Section 3).
We make no claim that this certification architecture works unchanged
at larger model sizes, longer problems, or richer branches of
mathematics. What we do claim is that, at this particular scale, with
these particular instruments, every number in this paper was checked by
machinery that could not be talked into flattering it — and that the
machinery itself, not any individual number, is the actual
contribution.


## Contributions

*(This is the paper's claim registry: five claims, each with its own
evidence pointer and its own limit. Nothing here is claimed that is
not backed by the project log and gated by machinery that could not be
talked into flattering it.)*

**1. A zero-parameter certification lattice, where every link is
justified by a named failure.** The system's output is a decision —
certify, answer, flag, or abstain — produced by a chain of four
invariance checks (input register, rewording, training lineage, truth)
in which no number is trained on the verdict itself and the answer key
never enters the decision process (Section 6). At the freeze point it
certifies 912 of 1,500 held-out problems (60.8%) at a measured
precision of 1.0000. The design argument is not that this chain is
elegant, but that it is *minimal*: each link is justified by a real,
named example that defeats the whole chain without it — a unanimous
wrong vote on foreign prose motivating the recognition gate,
nine-out-of-ten panel disagreement on gate-stable wild text motivating
the cross-lineage panel, one broken certificate motivating the answer
key itself. *Limit (Section 10):* 1.0000 on 912 problems is a bound on
error near a tenth of a percent, not proof of zero error, and the
register it holds on is a single language family.

**2. The method itself, as a contribution: a registered-prediction
discipline that survives contact with its own instruments.** Every
substantive claim in this paper was a prediction pinned down before its
measurement, with pass/fail bars that wrote the verdict mechanically —
a promotion and its record are one atomic action, and a failed
prediction changes nothing. Across fourteen versions of the system,
this discipline converted every failed prediction into a lesson, an
instrument, or a retired build (the survivor arc alone produced nine
individually tested hypotheses, Section 8.2), and it caught three
defects in the project's own measuring instruments before any outside
reviewer could have (Section 10). The complete chronological log —
every prediction, bar, verdict, and lesson — ships as supplementary
material: it is offered for scrutiny, not for trust. Its clearest live
example is Section 6.4's standing bet: a falsifiable, dated prediction
that the project's own agreement-based detector will decay across
future versions, published together with the plan for replacing it.
*Limit (Section 10):* every version-to-version comparison is a single
training run; the discipline bounds self-deception, not random
run-to-run variation.

**3. The reading campaign: hand-annotation through a gate that cannot
be flattered, with a confirmed mechanism and a measured stopping
point.** Three hand-annotated books of real mathematical prose (about
188 unique problems, about 82% requiring full hand surgery) taught the
parser its own register through a gate its own annotator cannot talk
past: five-way agreement plus the dataset's own published answer,
applied mechanically. The campaign's effect was confirmed three
separate ways — prose as a teacher of general register, prose as a
regularizer (a new best score on the generated-problem test set arrived
as a side effect), and prose as rehearsal (small-problem vote
disagreement dropped from 0.212 to 0.010) — and, unusually, the
campaign measured its own end: the rate of newly solved problems
flattened out (about 23 recovered for the first ~114 uniquely annotated
examples, about 0 more after that at the same distribution), so the
campaign reports its own completion rather than an open-ended slope.
*Limit (Section 10):* saturation was measured for one distribution
only; the annotator is also one of this paper's authors, and the
answer key verifies correctness, not representativeness.

**4. The binding theorem: a concept lives in neither wording nor
structure alone, and the architecture follows directly from that.**
This project's central negative result was proven from both
directions: problems that read as obvious siblings share no common
graph structure at all (the rate-frame family), and recurring graph
structures show no recoverable surface kinship (the registry census) —
a concept is the *binding* between how a problem is worded and what
structural role it plays, and it cannot be reduced to either one alone.
The direct architectural consequence is the two-jaws design itself:
recognizing and abstracting concepts happens entirely on the wording
side, the underlying graph is deliberately stripped of any trace of
wording, and whatever checks the graph afterward never inherits either
side (Section 3). *Limit:* this was demonstrated within one register's
family structure; whether the theorem holds beyond it is a conjecture
for the follow-up project to test.

**5. A field manual for instruments that age, meant to travel beyond
this one system.** Three findings about measurement itself, each with
an identified mechanism: latent geometry drifts across versions by
*rotating*, not by decaying (raw similarity 0.59, aligned similarity
0.988 — the fix is to re-anchor, not to give up on the instrument);
pooling evidence of different lengths together quietly absorbs a
length-based bias that masquerades as a real distance signal (a
correction took that spurious correlation from r = −0.825 down to
−0.024); and any signal promoted into a gatekeeping role becomes
selected against over time, so a monitoring portfolio must always keep
one examiner outside the acceptance path (Section 6.4). We offer these
as general engineering lessons for any deployed system that watches
itself using trained instruments. *Limit (Section 10):* all three were
established at this paper's particular scale and instrument family —
the general shape of each lesson should travel; the specific numbers
will not.

---

### Author contributions

This paper had two authors, working as two machine channels plus one
human adjudicator (Section 8.5); the accounting below is the
justification for that byline.

**Bryce Roche (human).** Direction and final judgment: every training
run started only on his word, every promotion and every kill was his
to accept, and every course correction in the project was his call.
Twenty registered hunches — pre-verbal instincts about where the
system was going wrong, each one formally registered and checked before
any measurement was trusted — every single one of which turned up
something real (the hash-collision audit, the latent-space rotation,
the length-biased estimator, and a key-value smearing bug are among
them). Hands-on annotation surgery on the books, alongside the machine
lanes. The authorship policy itself.

**Claude (Anthropic; two channels).** A *design channel* that wrote the
registered predictions, pinned down prediction frames and pass/fail
bars before any measurement ran, and supplied skeptical pushback
against its own sibling channel's enthusiasm; and an *execution
channel* that built every component, ran every measurement, kept the
project log, and drafted this paper. The two channels checked each
other: designs were only built after critique, and results were only
banked once they passed the verdict machinery.

**The machinery itself (neither author).** Every capability claim in
this paper was gated by scripts that write the deployment record on a
pass and change nothing on a failure. Neither author could talk a
number past the test battery; several times (Section 10), the battery
refused what the authors expected to see, and those refusals are
recorded in the log, unedited. We consider this the paper's strongest
statement about authorship: the results belong to a discipline, not to
a hand.


## References

*(Every entry was checked against its original source; per-entry notes
on what each is cited for are kept in bibliography.md.)*

- Angelopoulos, A. N., & Bates, S. (2021). A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. arXiv:2107.07511.
- Chambers, C. D. (2013). Registered Reports: A new publishing initiative at Cortex. Cortex, 49(3), 609–610. doi:10.1016/j.cortex.2012.12.016. [Editorial.]
- Chen, C., Borgeaud, S., Irving, G., Lespiau, J.-B., Sifre, L., & Jumper, J. (2023). Accelerating Large Language Model Decoding with Speculative Sampling. arXiv:2302.01318.
- Chow, C. K. (1957). An Optimum Character Recognition System Using Decision Functions. IRE Transactions on Electronic Computers, EC-6(4), 247–254.
- Chow, C. K. (1970). On Optimum Recognition Error and Reject Tradeoff. IEEE Transactions on Information Theory, IT-16(1), 41–46. doi:10.1109/TIT.1970.1054406.
- Cobbe, K., Kosaraju, V., Bavarian, M., et al. (2021). Training Verifiers to Solve Math Word Problems. arXiv:2110.14168.
- French, R. M. (1999). Catastrophic forgetting in connectionist networks. Trends in Cognitive Sciences, 3(4), 128–135. doi:10.1016/S1364-6613(99)01294-2.
- Geifman, Y., & El-Yaniv, R. (2017). Selective Classification for Deep Neural Networks. NeurIPS 2017, 4878–4887. arXiv:1705.08500.
- Goodhart, C. A. E. (1975). Problems of Monetary Management: The UK Experience. In Papers in Monetary Economics, Vol. I, Reserve Bank of Australia. Reprinted in Goodhart, Monetary Theory and Practice, Macmillan, 1984.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. ICML 2017, PMLR 70, 1321–1330. arXiv:1706.04599.
- Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., & Steinhardt, J. (2021). Measuring Mathematical Problem Solving With the MATH Dataset. NeurIPS 2021 Datasets and Benchmarks. arXiv:2103.03874.
- Hendrycks, D., & Gimpel, K. (2017). A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks. ICLR 2017. arXiv:1610.02136.
- Kadavath, S., Conerly, T., Askell, A., et al. (2022). Language Models (Mostly) Know What They Know. arXiv:2207.05221.
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. NeurIPS 2012, 1097–1105. doi:10.1145/3065386 (CACM reprint).
- Kschischang, F. R., Frey, B. J., & Loeliger, H.-A. (2001). Factor Graphs and the Sum-Product Algorithm. IEEE Transactions on Information Theory, 47(2), 498–519. doi:10.1109/18.910572.
- Leviathan, Y., Kalman, M., & Matias, Y. (2023). Fast Inference from Transformers via Speculative Decoding. ICML 2023, PMLR 202. arXiv:2211.17192.
- Lightman, H., Kosaraju, V., Burda, Y., et al. (2023). Let's Verify Step by Step. ICLR 2024. arXiv:2305.20050.
- Lin, S., Costello, D. J., Jr., & Miller, M. J. (1984). Automatic-repeat-request error-control schemes. IEEE Communications Magazine, 22(12), 5–17.
- Manheim, D., & Garrabrant, S. (2018). Categorizing Variants of Goodhart's Law. arXiv:1803.04585. [Unrefereed.]
- McCloskey, M., & Cohen, N. J. (1989). Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem. Psychology of Learning and Motivation, 24, 109–165. doi:10.1016/S0079-7421(08)60536-8.
- Nosek, B. A., Ebersole, C. R., DeHaven, A. C., & Mellor, D. T. (2018). The preregistration revolution. PNAS, 115(11), 2600–2606. doi:10.1073/pnas.1708274114.
- Shafer, G., & Vovk, V. (2008). A Tutorial on Conformal Prediction. Journal of Machine Learning Research, 9, 371–421. arXiv:0706.3188.
- Shanmugam, D., Blalock, D., Balakrishnan, G., & Guttag, J. (2021). Better Aggregation in Test-Time Augmentation. ICCV 2021. arXiv:2011.11156.
- Shervashidze, N., Schweitzer, P., van Leeuwen, E. J., Mehlhorn, K., & Borgwardt, K. M. (2011). Weisfeiler-Lehman Graph Kernels. Journal of Machine Learning Research, 12, 2539–2561.
- Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a Random World. Springer.
- Wang, X., Wei, J., Schuurmans, D., Le, Q. V., Chi, E. H., Narang, S., Chowdhery, A., & Zhou, D. (2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models. ICLR 2023. arXiv:2203.11171.
- Weisfeiler, B., & Leman, A. A. (1968). A reduction of a graph to a canonical form and an algebra arising during this reduction. Nauchno-Technicheskaya Informatsiya, Ser. 2, 9, 12–16. [English translation available.]
- Wu, Y., Jiang, A. Q., Li, W., Rabe, M. N., Staats, C., Jamnik, M., & Szegedy, C. (2022). Autoformalization with Large Language Models. NeurIPS 2022, 32353–32368.
- Zhang, D., Wang, L., Zhang, L., Dai, B. T., & Shen, H. T. (2020). The Gap of Semantic Parsing: A Survey on Automatic Math Word Problem Solvers. IEEE TPAMI, 42(9), 2287–2305. doi:10.1109/TPAMI.2019.2914054.
