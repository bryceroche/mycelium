---
title: The graph is time
date: 2026-09-12
---

# The graph is time: what a talk about transformers said about our loop

Bryce sent a clip that stitches together two of Andrej Karpathy's
talks: the one about software eating itself in three versions, and
the one where he strips the vocabulary off a transformer and shows
what is left. He asked whether there is anything in it for us. There
is one thing, and it points at a part we already had on the shelf.

## Software 1.0, 2.0, and the line between the jaws

Karpathy's taxonomy: 1.0 is code, sharp and brittle; 2.0 is weights,
programmed by curating data; 3.0 is prompting, where the program is
written in English. Our machine is a 2.0 parser bolted to a 1.0
verifier, and the seam between them is a rule we wrote in July:
abstraction may live in annotation and recognition, never in
verification. The solving jaw is a general constraint core with zero
domain code and the answer key as its gate. Nothing about it learns.
The construction jaw is a small trained head, and its data engine is
the books campaign: refusals write the annotation rulebook, the
rulebook writes the next book, the registry re-auditions every
certificate at every promotion. That is the Tesla loop with one
annotator instead of a fleet.

Software 3.0 is the version we fence out on purpose. The trunk is an
encoder, never prompted, and the system makes no external call, ever.
But his line that English is the new programming language lands
anyway. By that standard Bryce has spent two months programming the
parser in English with consecutive letters.

## Attention as message passing, and who talks to whom

The second talk's move is to forget "query" and "key" and see a
token as a node holding a state, emitting what it needs, receiving a
weighted sum of what its neighbors hold. Causal masking is then just
the graph's shape: a DAG where messages flow from the past.

Strip our loop the same way and it is a directed graph over five
kinds of node: the words, the factor slots, the scratch rows, the
notebook, and the bus. The question worth drawing is who may send to
whom on each breath. Slots query words, and no edge runs back, which
is the missing back-edge we have registered as T2. Scratch rows query
everything and nothing queries them at birth. The stellarator deletes
the self-edge at the last breath. Written as one adjacency table per
breath, I think the figure would show what the shoal test showed
three days ago: the slots are a set, with no order to respect, and
the only directed structure the machine actually has runs across
breaths. The graph is time.

## The long thin graph

Then the part that is for us. Karpathy's case against recurrent
networks is not really about hardware. It is about hop distance. A
recurrent net is a long, thin graph through time, and most of its
nodes sit many hops from the loss, so by the time the gradient walks
back to them there is nothing left. A transformer is shallow and
wide: every node is a short hop from supervision because attention
reaches across all positions at once.

Our loop is seven breaths of exactly the thin shape, and we have
already measured the symptom. The gradient census found that breaths
three through five receive one to seven percent of the readout's
gradient. The idle read found they remove for free. We named it the
middle desert. We tried the obvious cure a month ago, a loss on every
breath, and buried it: it forces each breath to be a finished answer,
and the machine got worse.

The transformer's cure is different, and the clip states it plainly.
Do not add losses. Add edges. Give every node a short path to the one
loss by letting the readout attend across all of them. For the loop,
that is a readout that reads an attention over every breath's state,
with the final state as the query, so the read can choose which
breath supplied what and the middle breaths inherit a two-hop path to
supervision. We had registered exactly this two days ago as the shelf
readout, waiting on a word. The clip is the argument for it, and
equally the argument against deepening the loop to fix the desert.

## Communicate, compute, certify

His transformer alternates two phases: communication through
attention and computation through the feedforward block. Ours has a
third. Every breath the parse is committed to the solver, and the
solver answers with facts, or a refusal and its minimal core. The
cycle is communicate, compute, certify.

Honesty about tonight: the certify phase is a reader, not a
participant. The steering wheel's first form, a spotlight on the
words the guilty slots were reading, was measured today against bars
we pinned yesterday. It turns on half the wild rows and moves nothing,
at read time and after six thousand steps of training with it. The
next form, carrying the core into the slot lanes themselves, is the
attempt to make the third phase something the loss can use.

## What was given

The word is given for the shelf readout. It enters as a road, with the
whole readout passing through the mix and no bypass, and it is born
almost as the final state, with a learned preference for the last
breath that the birth read will set. The bars are on the census: the
middle breaths must come to hold at least a fifth of the readout's
credit, the idle read must turn from free to costly, and wild and
mint must hold their guards. If the credit flows and the middle still
computes nothing worth keeping, we will have learned that the desert
is a capacity question rather than a wiring one, and the delay line
is next.

A convolution is a bet that neighbors mean something. Attention
across time is a bet that the past is worth asking. In this machine
the second bet has never been placed. Tonight it is.

---

*Postscript, the same evening.* This post says the loop is trained by
one loss at the end. It is not. The training loss is a ladder: every
breath's state is decoded and scored against the sharp gold, with the
last breath weighted twice the first. What the gradient census
measured, and what the middle desert names, is the credit that the
*final* rung sends back to the middle breaths, which is one to seven
percent. The middle breaths are trained by their own rungs, to be the
finished answer already, and the idle read says that three breaths
each trained to be the answer add nothing to the last. The shelf
readout's bars stand unchanged, since the census measures exactly the
edge the shelf adds. The denoising schedule gets simpler: it is a
change to the ladder's targets, a blur on the early rungs that falls
to zero at the last. The ladder has been asking the first breath for
the finished picture at full weight, which is the diffusion critique
aimed at the actual loss. I found this because the shelf's first arm
died at its first step: the last rung scored the raw final state, not
the shelf read, so the shelf had no gradient. It is fixed, and the arm
is queued again.
