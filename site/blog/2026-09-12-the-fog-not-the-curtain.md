---
title: The fog, not the curtain
date: 2026-09-12
---

# The fog, not the curtain: why the uniform kernel beats the masking kernel for a parse that sharpens

When we decided to borrow diffusion's training trick for the loop's
middle breaths, the first question was which kind of corruption to
undo. Discrete diffusion, the kind that works on symbols rather than
pixels, offers two. They look similar on paper and they are opposite
in what they ask of the machine. Bryce's gut picked one before I had
named either, with a picture: an image coming into focus from total
noise to a sharp photograph, everything at once. This post is about
why that picture is the right one for us, and what the other one
would have done.

## Two ways to ruin a picture

The masking kernel hides symbols. At the noisiest level every slot is
blank; as the noise falls, symbols are revealed in groups, and a
revealed symbol is exactly right. Undoing it means filling blanks.
This is the kernel behind the text-generation diffusion models that
paint in words a few at a time, and behind MaskGIT for images.

The uniform kernel corrupts symbols. At the noisiest level every slot
holds a random symbol; as the noise falls, each slot is right with
rising probability, and all of them are wrong with the same
probability at any given level. Undoing it means correcting every
slot a little. Nothing is blank and nothing is certain until the end.

Both reach the same clean picture at zero noise. They differ in the
shape of the road.

## A curtain is serial

Under the masking kernel, at every level some symbols are finished
and the rest have not started. The picture is revealed like a curtain
drawn back: a finished region, a frontier, and darkness. That is a
serial process wearing parallel clothes. Each step's job is to decide
what to reveal next, and once revealed, a symbol is never revisited.
Errors are permanent, because the kernel has no notion of a symbol
being partly wrong.

For a factor graph that is the wrong shape of job. The slots of a
parse are not independent pixels. A relation slot's operator, its two
argument pointers, and its result pointer only make sense together,
and a given's value only matters through the relations that read it.
Finishing one slot while its neighbors are blank asks the machine to
commit to a binding before the things it binds exist. It is the
pointer law's nightmare: structure entered one piece at a time, in an
order the machine has to invent.

## A fog is parallel

Under the uniform kernel, at every level every slot is a noisy
version of the truth. Nothing is finished and nothing is missing. The
picture is the whole picture at every step, blurred, and each step
sharpens all of it a little. There is no frontier and no order. A slot
that was wrong at one level can be right at the next, because being
wrong is the normal state of a slot in the fog, not a mistake to be
lived with.

That is the shape of the job the loop already does. The spectral read
this week showed the state's effective rank funneling from about
thirty at the first breath to a crest of ten to twelve at the last,
and it funnels for every slot at once. The machine sharpens in
parallel whether we ask it to or not. The gut's picture was not a
metaphor. It was a description of the measurement.

## What the kernel decides about the loss

We are not corrupting the input, at least not yet. We have no road
from parse space back into the state, and diffusion corrupts the
input. What we can corrupt is the target, and here the two kernels
make different losses.

A masking schedule turns into a curriculum: score a growing subset of
slots at each breath and ignore the rest. Early breaths would be
graded on a few slots, late breaths on all. That is serial supervision
with an ordering we would have to choose, and any ordering we chose
would be a claim about which parts of a parse come first, which the
binding theorem says is not knowable from the surface.

A uniform schedule turns into a blur: score every slot at every
breath, against a gold softened toward uniform by an amount that
falls breath by breath and reaches zero at the last. Early breaths are
asked to be uncertain in the right direction. Late breaths are asked
to be sharp. Every slot, every breath, in parallel, no ordering. And
because cross-entropy is linear in the target, the blurred loss is
exactly a mix of the loss against gold and the loss against uniform,
which means the whole thing lives inside the two helper functions
that every term of the loss already calls. No new organ, no ordering,
no new road.

## Why it matters that the machine is already confident

There is a detail that only shows up when you build it. The loop has
been trained by a ladder of per-breath losses, all sharp, since the
v98 era. That ladder has taught the first breath to decode the gold
with margins of about nine nats, the same as the last. Three of the
middle breaths, each trained to already be the finished answer, add
nothing to the final one, which is the desert we measured. Under the
masking kernel that confidence is fine, even wanted: a revealed
symbol should be certain. Under the uniform kernel it is the thing to
undo. The first breath should be sure in the direction of the truth
and unsure about the rest, and the sharpening should be the breaths'
work rather than something finished before they begin.

That is why the blur is not a nudge. Asking the first breath for
two-nat margins where nine exist is a real change to what the early
states carry, and a real risk to a checkpoint that reached its record
by being confident everywhere. Two arms are queued: the full fog, and
a gentle one, with bars pinned on the idle read and the guards.

## The dish line, and the tug of war

There is a third shape, and Bryce found it while the first arm was
training. When you wash dishes you do not take one plate through the
whole pipeline and then start the next. You scrape every plate, then
rinse every plate, then scrub every plate. The stages are serial and
the items are parallel. The curtain is serial in items. The fog is
parallel in everything. The dish line is serial in stages and
parallel in items, and for a parse the stages are not slots but
fields: which slots exist, what type each is, the digits read off the
text, then the argument pointers, then the results. Every slot at
every stage, in the order the dependencies dictate.

It matters because a single fog fights the line. The full blur asks
the first breath to be uncertain about everything, including presence
and digits that the grounding read makes obvious immediately. The
rung pulls those fields toward uniform while the sentence pulls them
sharp. That is a gradient tug of war on fields whose turn has already
come, and it is wasted force. A field-wise blur, each field's blur
falling to zero at its own stage, keeps the fog and drops the fight.

Before choosing the order by hand, we measured the one the machine
runs. Decoding every breath's state on the current checkpoint and
scoring each field per breath gave a flat table on wild text:
presence, type, operator, pointers and results at the same accuracy
from the first breath to the last, to the second decimal. Only the
digits climb, and only for two breaths. On synthetic, pointers and
digits sharpen across the first two breaths and then stop. The machine
was a two-breath parser with a decorative tail, and under the sharp
ladder it staged nothing. The same read on the first blurred arm shows
the picture coming into focus over three breaths instead of one, with
the middle breaths' share of the final gradient up four to seven
times, and idling them now costs a point of wild where it used to cost
nothing. The fog did what the fog should. The line is the next thing
to teach it.

## The line

A curtain reveals; a fog lifts. A parse is bindings, and bindings do
not come one at a time. The machine sharpens everything at once, so
the corruption it should learn to undo is the one that blurs
everything at once.
