---
title: Two loops, one shape
date: 2026-09-14
---

# Two loops, one shape: the language side gets its own atlas, its own certifier, and its own masks

The machine scores about 0.25 on wild prose and about 0.98 on
synthetic problems of the same algebra. That gap is the whole
project right now, and this week we learned where it lives to the
field. Decoding every breath's state and scoring each field, wild
gets presence, type and result pointers at 0.88, 0.84 and 0.78, and
fails on two things: argument pointers at 0.39 and digits at 0.32.
The machine knows a relation is there and roughly what kind. It
cannot say which quantities the relation binds, and it cannot read a
value off ordinary sentences. The math is not the blocker. Reading is.

## What we tried this week, and what it said

Four organs touched the token roads, each with bars pinned before
the read, and each left wild where it was, inside one standard error
of the measurement. A token convolution learned to be the identity.
A steering wheel that melted contradictory slots and forbade their
refuted bindings walked parses to consistency, and consistency is not
correctness: a wrong parse that satisfies itself draws no
contradiction. A structural claim mask, tokens marked as taken once a
slot binds them, gave half a point and stayed flat across its whole
sweep. And an aligner that anchored each given's value to its digits
in the text leaned negative on both seeds, for a reason worth
stating: the variable is the quantity, "the base coat's drying time,"
and the numeral is its value. Anchoring the variable to the numeral
taught the wrong thing.

The common lesson is that the wild register lacks something no organ
manufactured: knowledge of what ordinary phrases mean as quantities
and operations. The books supply it by hand, one annotated row at a
time, and the books scale. Nothing automatic reproduced it.

## Where abstraction is allowed to live

The project's founding rule is that abstraction may live in
annotation and recognition, never in verification. The solving jaw
is a general constraint core with zero domain code; the answer key
gates everything; nothing on that side learns. The construction jaw
proposes, and that is where knowledge about the world is permitted.

Bryce's proposal, which I now think is the right one, follows that
rule to its conclusion. The language side should have a curated table
of what prose means: "a dozen" is 12, "twice as many as" is multiply
by 2, "half of" is a division, "each" and "per" are a multiplier over
a count, "more than" is add, "fewer than" is subtract. Recognition
knowledge, symbolic, written once, zero parameters. It is the
language side's macro library: a floor of the ladder that is expanded
before the neural organs see the sentence, exactly as the math side's
macros are expanded before the solver sees the graph.

And it lands on the two fields that fail. Values arrive in prose as
words and multipliers, not numerals, and the head has only ever seen
numerals; a table turns "twelve" into a value mention with a span.
Operators hide inside "twice as many as the boys"; a table turns that
into a typed anchor with its operand's position. Those are the
footholds the grounding roads have been missing, and no trunk layer
gives them.

## The symmetry

Look at what the math side has, and the shape is three organs. An
atlas: kinds, centroids, macros, the knowledge of what problems look
like. A certifier: the constraint solver, which disposes of what the
neural organs propose. And dynamic masks: the structure the machine
routes messages along, rebuilt every breath from what it has
committed.

The language side can have the same three, in its own medium. Its
atlas is the lexicon. Its certifier is the fingerpost: the machine
already certifies parses by unanimity across five permuted views of
the sentence, on the principle that a correct reading is the same
from every angle and wrong readings each differ in their own way, and
a reading that holds across views is the language-side analog of a
parse that satisfies the solver. Its masks are claims: a token taken
by one slot is marked so other slots read around it, and a slot the
certifier melts releases its tokens. Two loops, one shape, each
proposing neurally and disposing symbolically in its own medium, with
the waist between them as the handoff.

This also settles a question we spent a day on. The frozen trunk stays
four layers of Llama. Looping frozen layers is a refuted cell in our
own ledger: a layer trained to consume its predecessor blurs when it
consumes itself. And deeper layers, which were the next candidate,
are the most expensive parameters in the system. If the language
loop's knowledge comes from a table and its certificate from the
fingerpost, the trunk does not need to grow.

## The cautions, so the eureka survives contact

A lexicon must be a thin anchor layer that feeds the neural roads,
never a grammar. The moment it tries to parse sentences we have
rebuilt the 1990s, and the trunk exists so we don't have to.

It covers the head of the distribution, and prose has a tail. So the
first step is a census, not curation: how many of the pen rows'
givens are numerals, how many are number words, how many are
multipliers, and how relations are phrased. That sizes the table
before anyone writes it.

And coverage is not correctness. Each entry is a claim about meaning.
The fingerpost should certify entries the way it certifies parses: an
entry that produces view-invariant readings on the rows it fires on
stays, and one that doesn't is struck.

## The order

Census first, on the CPU, no training. Then the aligner redone into
the value channel, with numerals and number words, two seeds against
the existing controls, bars in standard-error units. Then operators
and multipliers as the lexicon's second rung. Then the claims as the
language loop's masks, protecting anchors that finally exist. The
depth probe waits behind all of it, because if the lexicon moves
wild, depth was never the question.

A convolution bets that neighbors mean something. A blur in the loss
bets the middle can be graded on its own work. A lexicon bets that
the machine has been asked to rediscover, from 0.25 accuracy, what
every reader already knows about a dozen. That bet is the one on the
board this week.
