title: The Lean horizon
date: 2026-09-04

# The Lean horizon: a reasoning engine that could write real proofs

Everything on this site describes a reasoning engine with one shape:
a neural side that reads messy language and *proposes* structure, an
exact symbolic side that *disposes* — verifies or refuses — and a
port between them through which verdicts flow back and change what
the neural side looks at next. Today that engine reads algebra word
problems. But the shape was chosen for where it can go.

## The port is the whole game

Here is the thing we keep re-learning, most recently with hard
measurements: a reasoning system is only as good as its **port** —
the channel through which exact feedback re-enters the thinking
loop. We spent weeks discovering that our solver could derive over a
thousand forced conclusions per batch that the neural side simply
threw away, because no port existed to receive them. We built the
port; the numbers moved. The lesson generalizes: proposal without a
verdict channel is just eloquence. The port is criticality
infrastructure — and there is a second sense of "critical" here too.
Our measurements show the deliberation loop operates *near
criticality* in the physicist's sense: poised at the edge where
signals neither die out nor explode, the only place memory and
sensitivity coexist. A reasoning engine wants to live on that edge,
and the port's verdicts are what keep it honest there.

## Compression is what makes it reasoning

The other pillar is **compression**. Our engine forces every problem
through a narrow waist that destroys phrasing and keeps structure —
because understanding *is* selective destruction. And note what a
mathematical proof actually is: the ultimate compression — a short
certificate that stands in for infinitely many cases. "The sum of
two even numbers is even" compresses an infinite table of facts into
three lines. An engine built around compression at every layer —
silhouettes for reading, factor graphs for structure, certificates
for truth — is an engine already speaking proof's native language.

## The loop we can see from here

Now put the pieces together and look one horizon out. There exists a
language called **Lean** — a proof assistant in which mathematical
proofs are written as code, and a small, ferociously audited kernel
*compiles* them. If a proof compiles, it is correct. Not probably
correct, not persuasively correct: correct, with the same finality
as our answer key, but for all of mathematics rather than one
arithmetic answer.

Our architecture maps onto that world with almost nothing changed:

- The **neural jaw** reads a theorem statement — messy human
  mathematics — and proposes proof steps, exactly as it now proposes
  factor graphs.
- The **Lean kernel** replaces our CSP solver as the symbolic jaw:
  the universal verifier, the answer key for everything.
- The **port** carries the compiler's verdicts back — and Lean's
  errors are not just "no": they say *where* the proof breaks and
  *what* was expected, precisely the graded, structured feedback our
  dynamic masking is being built to steer by.
- The **ping-pong** becomes the proof loop: propose, compile, read
  the failure, re-attend, propose again — each iteration certified
  or refused by machinery that cannot be argued with.

That last property is everything. A system that iterates against a
compiler cannot fool itself, and it cannot fool you. Every accepted
proof is checkable by anyone, forever, independent of the neural
network that found it. The certified-or-silent principle we built
for word problems, scaled to mathematics itself.

We are not there. Today the engine is learning to read wild algebra
prose, and the honest ledger says so. But we did not choose this
architecture for algebra. We chose it because the shape — propose,
verify, feed back through the port, compress at every layer, live
near the critical edge — is the shape that scales from "Maria has
three apples" to theorems no one has proved yet. The engine we are
building is the small, auditable version of the engine mathematics
will eventually want.

The horizon is far. The road points at it.
