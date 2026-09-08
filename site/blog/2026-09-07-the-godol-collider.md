title: The godol collider
date: 2026-09-07

# The godol collider: disambiguation by collision

The phrase arrived in a dream, and it is a pun before it is anything
else. A *gadol* — often spelled *godol* — is Hebrew for "great," and in
yeshiva usage it names a revered sage, an authority whose reading of
a text settles arguments. The Large Hadron Collider is the machine at
CERN that accelerates particles to nearly the speed of light and
smashes them together to see what flies out. A *godol collider* is the
old yeshiva joke: a machine that smashes two great authorities into
each other at relativistic speed and observes what fundamental
disputes, sparks, and books come out of the wreckage.

Our machine has two authorities that disagree all the time. They are
the rival readings of a sentence.

## Rival readings

The neural jaw reads a word problem into a diagram — quantities,
roles, the arithmetic that connects them — and at every slot of that
diagram it holds a distribution of beliefs, not a single answer. For
most slots the belief is sharp and we commit it. But on wild prose
some slots are torn: this number is the *total*, or it is the *rate*;
this relation links *a* and *b*, or *a* and *c*. The head picks the
likelier one, and when it picks wrong the failure has a name in our
ledger, the mis-aimed pointer. We have measured that these failures
are not ignorance: the internal state of the survivors that beat
every repair we have built is 99.6% decodable. The machine knows
almost everything. It has aimed one pointer at the wrong slot.

The ordinary fix is to ask the head to be more confident. That is the
wrong instrument, and this project has a law about why: temperature
is orthogonal to truth. Confidence tells you how settled a reading
is, not whether it is right.

## The collider

So do not ask the reader. Smash the readings together.

Take the two rival parses of a torn slot — the two gedolim — and
hand each one, together with everything else the head has committed,
to the exact solver. The solver does one thing, deterministically: it
propagates. Given these facts and these relations, what is forced?
What flies out of the collision is one of three things. A reading may
yield facts and a single forced answer: it survives. It may leave the
question underdetermined: it neither survives nor dies. Or it may
produce a contradiction — a variable with no possible value at all —
and that reading is dead, killed not by a guess about it but by the
rest of the problem it could not live with.

That is disambiguation without an oracle. Neural proposes both
readings; symbolic disposes of the one that cannot be true. It is the
two-jaws law applied inside a single slot, mid-deliberation, and it
needs no answer key: consistency is checkable by the solver alone,
which means the collider is something the machine can *deploy*, not
only something we can measure. The key grades the collider afterward;
it never enters it.

There is a second reading of the pun, and it is the same machine seen
from the failure side. Spell it *Gödel* collider and you get the
logic-nerd joke: smash axioms together until the paradox appears.
Smash a parse against *itself* — drop one committed factor at a time
and re-propagate — and the contradiction names its own participants,
the smallest set of commitments that cannot coexist. Those slots are
exactly the ones to re-read. The godol collider tells you which
reading survives; the Gödel collider tells you where to collide.

## What we already own

Almost every part of this exists. The bridge between the jaws has a
one-call propagator that returns forced facts, or a contradiction.
The problem generator has a uniqueness gate that can say whether a
diagram forces exactly one answer. And the ledger holds the
collider's ancestor: *withhold-and-solve*, where the machine withholds
the piece of the diagram it trusts least and lets the solver re-derive
it from everything else. That trick recovered 26% of what would have
been wrong answers, for free, and introduced not one silent wrong
answer at any depth, because a re-derived value that contradicts the
rest of the graph is refused rather than guessed. The collider is the
same idea with the last step generalized: instead of withholding one
piece, enumerate its alternatives and let the solver grade them.

One honest correction belongs here. An earlier post on this site said
the solver "can hand back a minimal unsatisfiable core." The core's
search *can* be made to do that, but no organ in the deployed solver
is built and named for it. The Gödel side of the collider needs one,
and the cheap version — delete one factor, re-propagate, repeat — is
a few lines over the propagator we have. It gets built as part of
this, and the earlier sentence gets to become true.

## What a collider cannot do

Consistency is not truth. An exactly-determined problem can carry a
consistent *wrong* reading: swap two quantities in a system with no
redundancy and you may get a different problem that solves perfectly
well. So the collider does not return an answer. It returns a
*survivor set*, and the rule of the fingerpost governs what happens
next: one survivor, and the slot is disambiguated; several survivors,
and the machine stays silent, because silence earned by counting is
worth more than a guess.

That limit is also the first thing to measure, and it is a number
nobody has yet read: of the wrong readings our machine produces on
wild text, what fraction are self-contradictory at all? If most wrong
readings are consistent, the collider is blind to them and
disambiguation has to come from somewhere else. We call that number
collider visibility, and we pinned it before looking: at least 30% of
wrong parses must be contradictory for the collider to be worth
building into training, and below 10% we call it dead.

## The bars

The read is queued behind the run currently on the card, and it costs
no training at all: take the champion's first-pass readings of two
thousand wild problems, enumerate the second-best alternative at every
torn slot, collide the original and every variant, and count. Flip a
reading only when the original is contradictory and exactly one
variant survives with a unique answer. Then, and only then, grade the
flips against the key. Pinned: visibility of at least 30%; flip
precision of at least 0.60; net recovery of at least 5% of the wild
failures, with new silent-wrong answers held under 1% — the ancestor
managed zero.

If the bars hold, the build is a *veto channel*: the collider's
verdicts ride into the next cycle of deliberation as conditioning, on
the same road the solver's facts already travel, never as a training
signal on the head's confidence — a monitored signal in the loss
teaches concealment, not cure. Trained under this week's pressure
cooker, the machine's committed channel would learn from collisions.
If the visibility bar fails, the Gödel half still earns its keep as
an autopsy instrument: the paradox that names which slot was wrong.

Two sages walk into an accelerator. What comes out is a smaller set
of things that could possibly be true. That is what reasoning was
always supposed to be.

---

## Postscript, September 7: the collider is dead, and here is the shape of the body

The read ran the same evening, on the champion, both fixtures,
bars exactly as pinned above.

Collider visibility, the fraction of wrong readings that are
self-contradictory, came in at **5.1% on wild problems and 2.1% on
synthetic ones**, against a bar of 30% and a kill at 10%. Killed at
the first bar, on both fixtures. The reason is not that wrong
readings are consistent. It is that they are *underdetermined*:
87% of the wrong wild readings committed too few slots for the
propagator to force the query at all, so there was nothing for a
rival reading to collide with. Zero flips fired. Zero silent-wrong
answers were introduced, which is the one bar that passed, and it
passed vacuously.

We then tested the natural second rule offline, unpinned and
labelled as such: when the original is underdetermined and exactly
one rival makes the query unique, trust that rival. It is worse
than nothing. The rival's slot was correct 7% of the time on wild
and 1.5% on synthetic, and on synthetic the *original* slot had
already been right in 57 of 65 cases. The easiest way to force a
unique answer is to break the reading. Uniqueness is not truth, and
on an exactly-determined grammar it is closer to anti-truth.

What survives: the deletion core, the Gödel half, stays as an
autopsy instrument, because a paradox that names its own
participants is still the most legible refusal we have. And the
principle the post opened with survives unharmed: the solver's "no"
still has to reach the organ that steers attention. It will not
arrive through rival enumeration. It will arrive, if it arrives,
through the solver's silence made legible: which slots were
committed when the row went quiet. That is registered, with no bar
yet.

Two sages walked into the accelerator. Most of the time nothing
came out, because most of the time neither sage had said enough to
contradict the other. That is also a result.
