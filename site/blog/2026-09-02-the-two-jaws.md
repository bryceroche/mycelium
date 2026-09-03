title: The two jaws
date: 2026-09-02

# The two jaws

The grouper is a fish with two sets of jaws. The front jaws sit
where you'd expect a mouth to be, and they grip: they lunge, they
seize, they deal with the chaos of open water — a fleeing meal that
darts left when you expect right. Behind them, deeper in the throat,
sits a second set of jaws that never sees open water at all. Its job
is narrower and absolute: crush whatever the front jaws deliver, no
improvisation required.

Our reasoning machine reads math word problems written in plain
English and solves them, and it is built the same way, on the same
principle: use the flexible thing where you need flexibility, and
the exact thing where you need exactness, and never let the two
swap jobs.

The **front jaw** is neural — a small trained head, a few million
adjustable numbers, sitting on top of a much larger frozen language
model. Its job is to grip a sentence. Word problems arrive in every
possible costume: different names, different objects, different
orders of telling, sentences that bury the important number in a
subordinate clause or state it twice with different words. None of
that variation is arithmetic. All of it has to be handled before any
arithmetic can happen, and handling it requires something that can
generalize across costumes it has never seen in exactly that
combination before. That is what neural networks are good at, and
it's the only place in our machine we let one operate. The front jaw
reads the sentence and grips it into a typed structure: a diagram
naming the quantities involved and the relationships between them.

The **throat jaw** is symbolic — an exact constraint solver, the
kind of program that does deterministic logical search rather than
statistical pattern-matching. It never touches English. It takes the
diagram the front jaw produced and does the decisive work: it
crushes the diagram down to a number, by search that either succeeds
because the arithmetic checks out, or fails and says so. There is no
guessing in the throat jaw. There is no probability anywhere in it.
Given the same diagram twice, it produces the same answer twice, and
if the diagram is contradictory, it does not produce an answer at
all — it says why.

The law of the house, and it is a law we hold to without exception,
is: *neural proposes, symbolic disposes.* The front jaw is allowed
to be creative, statistical, occasionally uncertain — that
uncertainty is the honest cost of gripping something as slippery as
natural language. The throat jaw is never allowed to be any of
those things. It is not permitted to guess, hedge, or approximate.
Its entire value to the system comes from being the one part of the
pipeline that is not doing pattern-matching at all.

Why does the split matter so much that we'd build an entire essay's
worth of machinery — a narrow information "waist," a rotating
attention clock, a wall of independent witnesses — around
protecting it? Because the alternative is a system that is fuzzy all
the way down, and a fuzzy system cannot tell you when it's wrong. If
the same network that reads the prose also does the arithmetic, a
confident-sounding wrong answer looks identical, from the outside,
to a confident-sounding right one. Splitting the jaws means the
system's confidence and its correctness are produced by two
different processes that can be checked against each other. The
neural jaw can be wrong about the diagram; the symbolic jaw will
still crush whatever it's handed with total honesty, and if the
diagram was wrong, the crushing will fail in a legible way — a
contradiction, a refusal, a shape you can inspect.

Abstraction, in other words, is allowed to live in *recognition* —
in the front jaw's reading of the sentence. It is never allowed to
live in *verification* — in the throat jaw's judgment of whether the
arithmetic holds. That boundary, kept without exception, is what
lets the rest of the machine trust its own answers enough to
sometimes refuse to give one.

Grip, then crush. Two jaws, two different kinds of honesty, and the
whole animal survives on the difference between them.

## The ping-pong: the jaws take turns

The division of labor is not a handoff done once. During deliberation itself, the jaws alternate:

Our reasoning machine reads math word problems with two different
parts working together: a neural front jaw that grips messy English
prose into structure, and a symbolic throat jaw that crushes that
structure into an exact answer, by deterministic logical search
rather than guessing. Ordinarily you'd picture these two parts
working in sequence — read the whole problem, hand the whole
diagram to the solver, get an answer back. That's how the system
behaves at the end.

But during the machine's seven cycles of deliberation, before it has
committed to anything final, we built the two jaws to talk to each
other mid-thought, not just at the finish line. We call this the
**ping-pong**.

Here's how a cycle goes. The neural jaw looks at the sentence and
its current guess at the structure, and commits to whatever piece of
it it's genuinely confident about — maybe it's now sure that a
particular number is a total rather than a rate, even though it
hasn't yet worked out where every other number goes. That committed
fragment gets handed to the symbolic jaw immediately, mid-cycle,
rather than waiting for the whole diagram to be finished. The
symbolic jaw then does something cheap and exact with it: it
propagates the consequences. This is the same move a sudoku player
makes when they fill in one square and then look around the board
for any other square that is now *forced* — a square where, given
what was just committed, only one number could possibly go. The
symbolic jaw does this for the emerging math diagram: given what the
neural jaw just committed to, what else is now logically forced?

Whatever gets derived this way — the forced consequences, not
guesses — is handed back to the neural jaw as established fact for
the next cycle. And it does something more than just add information
to a list. The structure that's now been committed reshapes the
neural jaw's own attention: which parts of the sentence it's allowed
to look at next, and how closely, shifts based on what's already
settled. We call this **dynamic attention masking** — the parse's
own developing skeleton deciding what the next breath is permitted
to focus on. A number that's already been pinned down doesn't need
the same scrutiny it needed before it was pinned down; the machine's
attention moves on to what's still unresolved.

Neural proposes, symbolic disposes, neural re-attends. Ping, pong,
ping again — a real conversation between the two jaws happening
*during* the read, not just a relay handoff at the end of it.

We want to report where this actually stands, honestly, because it
would be easy to describe a mechanism this elaborate and let the
elaborateness imply a result. The machinery is built. It is
verified — we can watch it commit fragments, propagate consequences,
and reshape attention exactly as designed. And on the synthetic
training data we built the system on, it changes essentially
nothing. The reason turned out to be simple once we found it:
synthetic problems, generated the way ours were, tend to state
everything needed up front, in a form that doesn't require much
mid-read deduction to unlock. There's no ladder of consequence for
the symbolic jaw to climb — nothing gets forced, because nothing was
withheld. The sudoku board arrives mostly filled in.

Real textbook prose is not like that. It states things indirectly,
across sentences, in an order that often withholds the easy
inference until later — exactly the kind of writing that *does*
build a ladder of consequence a solver can climb rung by rung as the
read proceeds. Whether the ping-pong earns its keep on that kind of
prose — whether mid-thought deduction actually helps once the text
stops being so obliging — is not yet a settled question. It's the
campaign underway as this is written: teaching the system to read
wild, real prose, and measuring, honestly, whether the machinery we
built for exactly this moment finally has something to do.

Built, verified, and waiting for the kind of sentence that needs it.
That's not a failure to report. It's a bet we're still watching
resolve.
