---
title: One map, two charts
date: 2026-09-14
---

# One map, two charts: how two loops share an atlas and a bridge without sharing code

Yesterday we settled that the language side of the machine gets the
same three organs the math side has: an atlas, a certifier, and a
set of masks. Today's question was the obvious next one. The two
atlases are different and related. The two mask systems are
different and related. How do you write that down so the code cannot
quietly drift into two copies that agree by luck?

## The map

A topographical map and a physical map of the same valley are not
two maps. They are one map with two charts. The grid is shared. A
place you find on one chart sits at the same coordinate on the
other, and that is the entire content of the word "same." What
differs is what each chart draws at that coordinate: elevation on
one, rivers and roads on the other.

Our valley is the space of problem kinds. A row's kind is one fact
about it, and it is seen from two sides. The math side sees it as the
trajectory the slot states walk through seven breaths. The language
side sees it as the trajectory the reading walks, the token states
the slots are looking at as they commit. The atlas file already held
both, a slot chart and a token chart under one class index and one
era stamp, and the code treated the second as an optional extra bank
that a few readers knew to ask for. Now it is one object. An atlas
has charts, one per medium, and every chart is indexed by the same
classes. You locate a row on whichever chart you are standing on.
You read the other chart at that coordinate. That read has a name,
transport, and it is the only way across. The cross-atlas prior we
built last week was a special case of it: locate on the token chart
at breath zero, read the slot chart's trajectory. The happy-family
radius, how tightly a row sits inside its kind, is now a question you
can ask of either chart with the same call.

## The bridge

Masks are where the two loops touch, so they are where ad hoc code
grows fastest. We had a spotlight built in one place for training
and rebuilt in another for reading, a claim mask built from the
attention matrix, a melt built from the solver's core, a locus melt
built from view disagreement, and each one knew privately how to get
from slots to tokens or back. Five functions, one idea.

The idea is a bridge with two lanes. The forward lane is the waist.
The trunk's reading lives in two thousand dimensions and the slots
live in five hundred and twelve, and the waist carries the text down
into the space the slots can read. The return lane is the cross
attention matrix itself. Every breath, each slot spreads its
attention over the tokens, and that matrix of weights is the only
correspondence between the two mediums that exists. So it is what
carries certificates back.

A certificate is born in one medium and is projected into the other
by the bridge. The solver's contradiction is born on slots: these
slots cannot all be true. Projected to tokens through the matrix, it
becomes a spotlight on the sentences those slots read from. Kept on
slots, it becomes the melt. The fingerpost's instability is born on
slots too, a slot whose reading changes when the sentences are
reordered, and it projects the same two ways. The claim mask is born
on the matrix itself: a token a slot has taken is taken. One object
holds the matrix, the sentence ids, and the real-token mask, and
answers two questions, to tokens and to slots. Nothing else in the
machine builds a cross-medium mask by hand anymore. The trainer's
copy of the spotlight and the reader's copy are the same call now,
and were shown to produce the same numbers before the old code was
removed.

## Why the shape matters

The two loops already share the breath clock and the rotation bus.
With the atlas and the bridge as single objects, the list of things
they share is exact: the clock, the bus, the map, the bridge. The
list of things they own is exact too: a medium, a certifier, and the
masks in their own medium. A loop is a medium plus its organs. What
crosses between loops is the waist going forward and a certificate
carried back. Both loops breathe the same air and run the same
logic. One speaks in words and the other in equations.

None of this is a score. The language loop's first version is
training now, two seeds against controls, with its bars pinned in
standard-error units before the read. The atlas and the bridge are a
refactor by code motion, bit-identical on the fixture before and
after, and the reason to do it before the results land is the reason
to do any refactor early: the next organ, the locus-driven token
mask, would otherwise have been a sixth private function. Now it is
a certificate and a call.

## Postscript, the same afternoon

The language loop's first version finished its two seeds. Against the
matched controls, wild moved by a tenth of a point on one seed and
minus seven tenths on the other, with paired tests at z of minus 0.16
and plus 1.12. Synthetic moved up by a fifth of a point on both. That is
a null on both bars, and it is banked as one. The census says the organ
was alive: its injection ran at about five percent of the token state
at every breath, and its output matrix grew from zero. A sentence-local
token attention, once per breath, does not move binding on ordinary
prose. The second version, with the shared clock on the tokens and the
locus-driven token mask, is now a certificate and a call away, which is
what the refactor was for.
