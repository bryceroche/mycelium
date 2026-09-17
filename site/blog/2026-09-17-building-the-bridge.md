---
title: Building the bridge
date: 2026-09-17
---

# Building the bridge: between the continuous and the discrete

Every reasoning machine we have built has two sides that do not speak
the same language. One side is continuous: vectors, attention, a
state that moves a little each breath and can be nudged by a
gradient. The other is discrete: a factor graph, a solver's
assignment, a certificate that says this graph forces this answer or
it does not. The whole design question is the bridge between them,
and this post is about what we have learned building it.

## The two sides

The continuous side is a frozen language trunk feeding a small trained
head. Over seven breaths the head's slot states read the text through
cross attention and settle into a parse. Nothing here is exact. A
slot's argument pointer is a distribution over variables; its value
is a distribution over digits.

The discrete side is a constraint solver that does not guess. Given a
graph, it either finds the unique assignment or refuses. The answer
key sits behind it and gates every row that enters training. Neural
proposes, symbolic disposes.

The bridge is where a distribution becomes a graph and where a
verdict becomes something a vector can feel.

## Crossing forward: from vectors to a graph

The forward crossing is a decode. Argmax each field, and a slot state
becomes a typed factor. This crossing is lossy in a specific way we
measured this week: it is order-sensitive. The graph is a set of
factors, but the decoder emits a sequence, and the loss grades the
sequence. A correct graph in another order scores zero. On the
holdout, matching factors by content instead of by slot raises the
read from 0.25 to 0.30, and it explains entirely why a retrained arm
looked worse when its understanding was unchanged. A bridge that
charges for serialization is a bridge with a toll on the wrong side.
The set-prediction loss, matching before grading, is how the forward
crossing learns to respect the discrete object's own invariance.

## Crossing back: from a verdict to a vector

The return crossing is harder, because a verdict is one bit and a
vector wants a direction. We built it in layers.

Certificates as masks. When the solver's propagation finds a slot's
claim consistent, inconsistent, or forced, that fact becomes a bias on
the next breath's attention: a spotlight on the tokens a forced slot
should read, a melt on a slot whose claim conflicts. This is the
return bridge carrying discrete facts as continuous pressure.

The write-back. Until this week the bridge's arrows pointed one way:
slots read tokens seven times and the tokens never heard back. Now
each slot's typed state travels back through the same attention's
transpose to the tokens it read, at a fixed gain the residual cannot
vote down, and the next breath's keys and values are built from the
written-back tokens. The shoal turns because influence runs both
ways.

The clock. The middle breaths' motion is mostly a rotor turning by
design, and once it is projected out the content path marches in
small coherent steps. The rotor is the continuous side keeping its
own discrete time, so that what breath a state belongs to is a fact
the state carries.

## What the bridge is for

Alternation. A breath is a round trip: read, propose, certify, read
again with the certificate in hand. The system is at its best when
that loop actually turns, and at its worst when the continuous side
runs seven breaths on its own and the discrete side is consulted once
at the end. Every road above exists to make the loop turn: the
certificate masks so the verdict reaches the next read, the write-back
so the tokens carry it, the level ladder so early breaths are asked
for what an early breath can know.

Two honest results. Weighting the training loss by the solver's
certificates did nothing, and neither did the write-back on the
diet we had at the time. Both were tried on gold the machine could not
read, and both are back in the queue now that the gold is in the
machine's language.

## The rule the bridge taught us

The discrete side is the authority on truth and the continuous side is
the authority on plausibility, and the bridge must never let either
pretend to be the other. The solver never learns; the head never
certifies. What crosses forward is a proposal with its uncertainty
intact. What crosses back is a fact, delivered as a direction. A
machine built this way can be wrong, and it can be told so, and it
can use being told. That is the whole of what we mean by reasoning
here, and the bridge is where it lives.
