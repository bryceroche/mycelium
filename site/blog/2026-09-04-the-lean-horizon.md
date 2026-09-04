title: The Lean horizon
date: 2026-09-04

# The Lean horizon: two loops, one engine

This project runs on a two-line theory of thinking:

> **Compression that preserves the right ports is what makes
> abstraction. Alternation between neural attention and an exact
> solver is what makes reasoning.**

This post unpacks both lines, and then describes where they lead: an
outer loop, wrapped around our whole engine, whose judge is the Lean
proof compiler.

## Abstraction: compress, but keep the ports

Every abstraction is a compression — you throw away detail and keep
a summary. But compression alone isn't abstraction; a blurry photo
is compressed too. What makes an abstraction *work* is that the
compression **preserves the ports**: the connection points through
which the summarized thing still interacts with everything else.
"A function that sorts a list" throws away the algorithm but keeps
the interface — and the interface is what everything downstream
plugs into. Compress away a port you needed, and the abstraction
doesn't simplify your problem; it severs it.

Our engine is built as a stack of exactly such compressions. The
512-dimension waist destroys phrasing but preserves quantities,
roles, relations — the ports the solver plugs into. The factor graph
throws away the story but keeps every constraint. We learned the
port lesson the hard way, twice, with measurements: an organ fed a
boolean where graded signal existed (a crushed port) sat inert for
weeks; a solver deriving a thousand conclusions per batch with no
port to deliver them through watched its work discarded. Port
criticality is not a slogan here. It is the difference between
abstraction and amputation.

## Reasoning: the inner loop

If abstraction is the statics, reasoning is the dynamics — and it is
an **alternation**. Inside each deliberation, every breath, the
neural heads commit what they're confident of; an exact constraint
solver floods the consequences; and what returns through the port
reshapes what attention looks at next. Neural proposes, symbolic
disposes, neural re-attends — several times per problem, in
milliseconds. That inner ping-pong is, as of this week, no longer a
design: our new trainer runs it live at every breath, and the engine
is learning under the rhythm for the first time.

The most underrated player in the inner loop is the solver's way of
saying no. When constraints cannot all hold, it can return a
**minimal unsatisfiable core** — the *smallest* subset of
commitments that contradict each other. Not "something's wrong," but
"these three things cannot all be true; everything else is fine."
An MUC is itself a compression with perfect ports: it discards every
innocent commitment and keeps precisely the handles the neural side
needs to grab in order to reconsider. Refusal, made steerable.

## The outer loop: Lean as the court of final appeal

Now wrap the whole engine in a second, slower loop. There is a
language called **Lean**, in which mathematical proofs are code and
a small, ferociously audited kernel *compiles* them. If a proof
compiles, it is correct — with the same finality as our answer key,
but for all of mathematics.

The future we're pointing at keeps both loops, nested:

- **Inner loop (milliseconds, every breath):** neural heads ↔ our
  CSP solver — drafting steps, flooding consequences, pruning with
  MUCs. Fast, local, exact-but-narrow. The scout.
- **Outer loop (seconds, every draft):** the assembled proof goes to
  the Lean compiler. Compile = done, certified forever, checkable by
  anyone, independent of the network that found it. Fail = Lean
  returns its own version of an MUC — an error that says *where* the
  proof breaks and *what* was expected — which flows back through
  the port, re-shapes attention, and launches the next inner-loop
  drafting session. The judge.

Two verifiers at two timescales, each speaking "no" in the most
compressed, most portable form available — the MUC inside, the
compiler error outside — and neural attention alternating with both.
The inner loop makes drafts good enough to be worth compiling; the
outer loop makes the whole system impossible to fool, including by
itself.

We are not there; today the engine is learning to read wild algebra
prose, and the ledger says so plainly. But the two-line theory is
already load-bearing at the small scale — measured, this week, in
our own machine. Scaling it means adding an outer judge, not
changing the shape. Abstraction by compression-with-ports; reasoning
by alternation; certification by a judge that cannot be argued with.

The horizon is far. The road points at it.
