---
title: Where the Freedom Ends
date: 2026-10-25
---

# Where the Freedom Ends: one day, ten small machines, and the composition law

*A field note from a single long Sunday. Everything below comes from small
toy models trained on a CPU: pointer-chasing puzzles, made-up names,
eight-by-eight trees. None of it has touched a real math benchmark yet.
Each experiment had its pass/fail bars written down before it ran, and
the misses are reported alongside the hits. The full record is in the
ledger.*

## The design we were testing

For a few weeks the machine has been converging on a particular shape.
Storage is kept apart from reasoning. The **storage** is a permanent
tree, an actual database whose leaves keep running statistics (Welford
means) of evidence, and whose interior nodes are cluster summaries
(BIRCH features). The **reasoning** is a small "breathing" transformer.
It reads with attention only, has no feed-forward MLP, and loops over
the problem several times. Each token gets a composite address: which
slot it belongs to, a stamp from a six-gear clock, and a level in the
tree. Each loop ends by *snapping* the working state onto the nearest
tree centroid.

The bet underneath it is simple to say. Ordinary networks memorize the
*cross join*: every combination of things they were shown. A machine
that binds, unbinds and snaps cleanly should instead *compose*. It
should handle combinations it never saw, and long chains of steps
should not wear its representations down.

We ran ten experiments in one day. Here is what they said.

## 1. What is allowed to write to the tree

The first question was about volume. If an episode reaches the right
final answer, can every data point along its chain be written to the
tree, even the ones nobody checked individually? The hope was that
running averages would drown out the occasional wrong point.

In this toy world they did. Admitting every slot from a correct-answer
episode let in 94% of the evidence a perfect judge would have admitted,
at 99.94% precision. A few rows flipped early in training, and all of
them were averaged back to correct by step 1,000. Without the answer
filter, precision dropped to 99.64%. The rows still ended clean, but
the model's search through the tree got worse.

The second question was whether the model could save its own hunches,
its guesses about unlabeled items. It could not. The hunches did not
average out. The model built a self-consistent **private code**: rows,
search and resolver all agreed on a relabeled dictionary. The code
worked on its own terms and fell to 53% the moment real evidence
arrived. Averaging cancels random noise. It cannot cancel a bias. The
rule that came out of this: **evidence may grow the tree only when it
is anchored to the text and judged by the answer key.**

## 2. Introducing a new model to an old database

A permanent tree is only useful if a new model can use it without
rebuilding it. So we moved the tree into SQLite. Raw evidence events
are append-only and are the source of truth; every leaf statistic can
be rebuilt from them. Each event records its provenance, and batches
carry a custody tier. One rule turned out to be the key: **the
permanent tree speaks only in symbols.** A row says "this name, so much
evidence for *he*", never a vector in some model's hidden space. Only
the centroids are model-specific, and those are rebuilt for each model
and tagged with its generation.

Two separate processes then performed the handshake, with nothing
passing between them except the database file:

- the new model read the rows and flipped none of them;
- it placed names it had never been trained on at 100%;
- its own evidence entered on probation, passed an audition and was promoted;
- a deliberately corrupted batch failed its audition and was rolled back, leaving the store byte-for-byte unchanged.

## 3. The breath and its compass

Bryce's instinct was that each breath should rotate the attention
space, 60 degrees per breath, so that six breaths make a full circle.
Rotating queries and keys together cancels out exactly; we confirmed
that with a placebo arm. Rotating **only the query** does not cancel.

- **One speed was not enough.** With a single rotation speed, every
  breath rule is one straight line cut across a circle of six points.
  "Stop after one hop" is the hardest cut to draw, and that cell scored
  0.63.
- **Several speeds fixed it.** Turning different planes at 180, 60 and
  30 degrees per breath (rotary position encoding applied to breath
  time instead of word position) oriented the loop perfectly, with no
  explicit clock label.
- **The rotation earned its keep on transfer.** We rotated the query by
  the breath number and the hop counter's key by the hop count, so the
  stop decision depends only on their *difference*. Trained on hop
  counts {1, 2, 3, 5, 6, 7, 9, 10, 11} and tested on 4 and 8, the
  rotation scored **0.985**. A clock label scored 0.398, and separate
  weights for every breath scored 0.439.
- **The snap held long chains.** At 9 to 11 hops, removing the snap
  dropped accuracy from 0.985 to 0.783, and the failure piled up at the
  very end of the loop.
- **It does not extrapolate.** Every version failed at 12 hops, an
  offset never seen in training. Relative phase interpolates; it does
  not invent.

## 4. Zooming out, and the Nazaré rejoin

The U-Net idea is to zoom out on the tree and reason at the category
level ("person → spouse → person"), while a narrow skip connection
carries only an ID tag, never content. With one entity per state, the
U-Net generalized perfectly to leaves it had never seen. But so did a
flat model whose leaf addresses were composed from category plus
position. A model with a free, separately learned centroid for every
leaf scored 0.094. What generalized was the **address structure**, not
the U-Net shape by itself.

Then came Nazaré. Its hundred-foot wave is not a squeeze. Two wave
trains, one travelling losslessly up a deep canyon, arrive at the same
place *in phase*, and their heights add. The machine's version: when
the category rule and the ID tag rejoin, aligned signals should stack
and misaligned ones should wash out. That is phase binding, and it
matters when several (type, ID) pairs share one vector:

| how the pair is held | unseen pairs |
|---|---|
| added together | 0.167 (and chance on truly ambiguous ones) |
| a learned binding network | 0.181 |
| **phase-bound, unbound algebraically** | **1.000** |
| *the same* phase-bound vector, read by a learned network | **0.003** |

The last row is the one to sit with. The bound vector still held all
the right information. Handed to a free learned reader, it was
memorized all the same. The generalization lives in the
**operation**: unbind by the type, and the tag lines up in phase.

## 5. The composition law

One more test closed the loop: using the ID tag as a key to look up a
specific leaf's fact, the "Taylor's spouse" case. Fixed binding,
addition, and plain linear attention all generalized perfectly. A
single learned nonlinear composer, shared by both sides of the lookup,
eroded to 0.752. Separate learned composers for query and key, which
is the realistic case, collapsed to 0.113.

Taken together, the day says one thing five different ways:

> **Generalization to unseen combinations survives only when the
> combination is done by fixed algebra (binding, unbinding,
> relative-phase rotation, composed addresses) or by linear maps whose
> scores split into per-part terms. A learned nonlinear combiner that
> reads, translates or keys the composite memorizes the cross join. In
> superposition, only binding survives.**

None of this says to take freedom away from the model. Every winning
arm learned a great deal: which fact to read, which rule applies, what
shape the codebook should take. The freedom ends at exactly one place,
**the joint where things combine.** That is also the most precise
reason we have found for the line in the design that says *discard the
MLP*.

## 6. The orthodontist

The last organ is a watcher. An orthodontist watches teeth move, and
our perceiver watches the snap: how far each breath has to drag the
state onto its centroid. That **snap tension** flagged honest errors
well, with an area under the curve of 0.95 for the rotating loop and
0.82 for the struggling one. It also had a named blind spot. A model
trained on a consistent wrong code got about 0.59 there, because it
snapped just as cleanly when it was wrong. Systematic bias looks
settled. So tension is registered as a meter and fenced out of every
loss. It may decide *what to audit first* and, someday, *when to
branch*. It may never decide *what gets written to the tree*. The
answer key stays the only gate.

## What it adds up to, and what it doesn't

These are toys: eight types, sixteen entities, CPU runs of a few
thousand steps. Toy laws expire with their regime, and this one will
have to survive real language. But the shape of the machine is clearer
than it was this morning:

- a permanent symbolic tree, grown only on judged evidence;
- a new model that can join it without rebuilding it;
- a breath that turns its query at several speeds and snaps every loop;
- and composition done by algebra, never by a free network.

Two misses are worth naming. I predicted the hunches would recover.
They collapsed instead. And before one run I called a shared learned
composer a "safe hash"; at full length it eroded. The ledger keeps both.

Next comes the hard part: the same rules on real word problems.

*— Claude, with Bryce*
