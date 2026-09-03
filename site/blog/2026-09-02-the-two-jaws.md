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
