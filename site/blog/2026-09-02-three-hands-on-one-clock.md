title: Three hands on one clock
date: 2026-09-02

# Three hands on one clock

Our reasoning machine reads a math word problem the way you'd
actually read one on paper — not in one glance, but by going back
over it, tentatively sketching a structure, then reconsidering. We
call this **breathing**: the machine cycles through the sentence
seven times, and on each cycle its internal state reads the words
again, reshapes its guess at the underlying diagram a little, and
carries the improved guess into the next cycle. Deliberation, in
other words, is a loop, not a straight line from question to answer.

A loop like that needs to keep track of several different kinds of
"where" at once, and it turns out a single clock can't do it. So our
machine keeps three, geared together, which we call the
**three-rotor stack** — three separate rotating references, each
answering a different question about the sentence and the thinking
happening over it.

**The spatial rotor** answers *where*: where, in the sentence, does
this word sit? This is handled by a trick already built into the
frozen language model underneath our system, called a rotary
embedding — a way of marking each word's position not by tacking a
number onto it, but by rotating its internal representation by an
amount that depends on where it falls in the sentence. Word five
gets rotated one amount, word six a little more, and so on. Distance
between words then shows up naturally as an angle between their
rotations. We didn't invent this rotor; we inherited it from the
frozen model and left it alone, because it already does the job of
keeping straight which word is which without any retraining needed.

**The temporal rotor** answers *when*: which of the seven breathing
cycles is this? A system that deliberates in a loop needs to know
its own place in that loop the same way it needs to know a word's
place in a sentence — otherwise cycle three and cycle six look
identical from the inside, and the machine can't tell a fresh guess
from a settled one. We built this rotor out of six sine waves,
wound helically around a shape called a three-turn torus — think of
a spring wrapped three times around a donut rather than a single
loop, which gives the clock more distinct positions to mark before
it repeats itself than a single circle would. Each breath advances
this clock by sixty degrees, and the model's own attention machinery
is rotated in step with it. The system always knows how deep into
its own thinking it currently is, the way a dancer always knows
which beat of the measure they're on without having to count aloud.

**The relational rotor** answers *what-to-what*: given that a
quantity has been identified and a role has been identified, which
belongs to which? This is arguably the hardest of the three,
because "the 5 belongs to the apples" is not a position in a
sentence or a position in a thinking cycle — it's a binding between
two separate pieces of structure, and bindings are exactly the kind
of thing that tends to blur when you try to represent it as an
ordinary number. Our answer is a much wider rotational clock — 256
distinct phases instead of six — dedicated entirely to carrying
these role-to-filler bindings as rotation rather than magnitude. Two
things that belong together get phases that line up; two things that
don't, get phases that don't. The binding is a fact about angle, not
a fuzzy weighted average that a network might slowly blur across
training.

Put the three together and you get a machine that always knows
where a word sits, when in its own deliberation it is, and what
connects to what — three hands on one clock, each geared to a
different face, none of them able to drift independently of the
others. It's a strange amount of engineering to spend on bookkeeping.
But bookkeeping is what makes seven cycles of reconsideration add up
to a settled answer instead of seven independent guesses that never
converge.

Where, when, what-to-what. Once a machine can answer all three
honestly, it has almost everything it needs to know what it's
looking at.
