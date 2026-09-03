title: Crossing the bridge
date: 2026-09-02

# Crossing the bridge: from neural to symbolic

Our reasoning machine lives on both sides of the deepest divide in
computing, and its whole job is to carry meaning across the bridge
between them. This post is that bridge, drawn slowly, with every
term defined.

## The two sides

**The neural side** is *continuous* and *non-deterministic*.
- **Neural**: computation done by a network of learned weights —
  millions of numbers tuned by training, no one of which means
  anything alone.
- **Continuous**: everything is a shade. A neuron's activation is
  0.73, a similarity is 0.988; nothing is simply yes or no, and
  every representation lives in a smooth space where "between" is
  always defined.
- **Non-deterministic** (in practice): outputs are graded beliefs,
  sensitive to training randomness and input phrasing. Ask twice, in
  two costumes, and you may get two answers — which is exactly why
  our certification re-asks in five costumes and demands agreement.

**The symbolic side** is *discrete* and *deterministic*.
- **Symbolic**: computation over explicit tokens with exact rules —
  variables, operations, constraints. Everything is legible: you can
  print the state and read it.
- **Discrete**: no shades. A variable equals 7 or it doesn't. A
  constraint is satisfied or violated. There is no 0.73 of a fact.
- **Deterministic**: the same input always yields the same output.
  Our solver run twice gives byte-identical answers, forever. Its
  refusals are as reproducible as its solutions.

Neither side can do the other's job. Wild language is a continuous
phenomenon — costumes, shades, ambiguity — and only the neural side
can grip it. Truth is a discrete phenomenon — right or wrong — and
only the symbolic side can guarantee it. The law of the house:
*neural proposes, symbolic disposes.* The bridge carries proposals
one way, and verdicts back.

## The bridge itself

The crossing happens in stages, each destroying a little more
continuity. A sentence enters as 2048 continuous numbers per word;
the **waist** squeezes each to 512, destroying phrasing and keeping
structure; cycles of deliberation sharpen graded beliefs into
committed choices; and at the far end the parse **snaps** to
discrete form — a typed factor graph, all integers and named
operations, which the deterministic solver either solves exactly or
refuses legibly. Continuous in, discrete out, and the moment of
snapping is the single most important event in the machine.

## The two atlases

Recognition on the neural side is guided by **atlases** — maps of
the operation-kinds the machine knows, one for each side of the
bridge:

- **The language atlas** maps how English *expresses* operations:
  the many surface silhouettes of "multiplication" or "a total."
  Today it chiefly guards the door — an out-of-distribution check
  asking "is this sentence from territory we know?" before the
  machine is allowed to answer.
- **The math-operation atlas** maps what operations *are*: each
  known kind summarized as a **centroid** — the running average
  location of every example of that kind in the machine's internal
  space. Centroids are maintained with **Welford's algorithm**, the
  numerically careful way to update a mean and variance one example
  at a time without storing the history. We are on the **seventh
  generation** of these centroids — one per era of the trained head,
  rebuilt every era because new training *rotates* the internal
  coordinate system (nearly purely: aligned generations agree at
  cosine 0.988), and an old map must never be trusted in new
  coordinates.

## An open question, honestly stated

The two atlases are not treated equally. The math-operation atlas
is consulted, re-anchored, audited every generation. The language
atlas mostly stands at the door. We suspect an imbalance there —
that the reading itself could lean on the language atlas the way
navigation leans on a chart, not just as a border checkpoint. It is
on our books as a registered question, which in this project means:
it gets a pinned prediction and a measurement, and the ledger
records the answer either way.

Two sides, one bridge, two maps — and every crossing audited.
