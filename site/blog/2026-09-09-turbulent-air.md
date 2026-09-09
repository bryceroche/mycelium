---
title: Turbulent air
date: 2026-09-09
---

# Turbulent air: a machine with no cold sink and no road home

The gut said, this morning, that the air in the machine was stuck.
We are building a breathing transformer: a small head over a frozen
trunk that reads a word problem seven times, and between each
reading revises a bank of slots that will become the diagram a
symbolic solver takes apart. The seven readings are the breaths. The
feeling was that after the first breath or two the slots settled
and the remaining breaths were ceremony, warm air going nowhere. We
need convection, the gut said. Get the air moving.

We have a rule about guts here. When one fires, you audit before you
build, because the drawer usually has something in it and it is not
always the thing you reached for. So before writing a line of the
convection organ, we read the air.

## The read

We had the states on disk already: every slot at every breath for
two hundred and fifty-six problems in each of our two registers, the
synthetic mint and the wild text. The read is elementary. For each
slot, how much of it is replaced between one breath and the next?
How far does its direction swing? How much of the variance at breath
k+1 was not there at breath k? And by the last breath, how many
slots have stopped moving at all?

Roughly half of every slot is replaced every breath. The direction
swings by about thirty degrees each time, and a fifth to a third of
the variance is new every breath. By the sixth breath, the number of
slots that have stopped moving is zero out of twenty-four, in both
registers.

The air is not stuck. It is turbulent.

Where the feeling came from is worth a paragraph, because it is a
mistake we will make again if we do not name it. We had an earlier
number, a fit between consecutive breaths that came back at 0.97, and
we had let it mean "the state barely changes." But that fit was
taken on the slot-pooled mean of the state, in a low-dimensional
subspace, and the pooled mean barely moves because the slots move in
different directions and cancel. Same room, different thermometer.
One measures the average temperature of the air; the other measures
whether it is moving. Averages are calm in a storm.

## The ratchet that was not there

The second half of the gut was more specific. Nothing un-commits, it
said. Once a slot decides, it cannot change its mind, and that is
why the later breaths feel like ceremony. We had a mechanism in mind
for this, a commitment mass that ratchets a slot toward its anchor
as confidence rises, and we went to read it per breath.

It is not there. The ratchet lives on a branch of the code that the
deployed family does not run. The champion simply overwrites its
state every breath. Nothing in the state ever freezes, which is what
the turnover read had just said from the other side. Commitment in
this machine happens elsewhere: in the deposits the reading makes
into its parking garage of facts, and in the moment the adapter
turns a soft slot into a discrete edge for the solver. Those are the
things that do not un-commit. The state itself is water.

So the gut was wrong twice in its particulars, and we bank that,
because a gut that is never wrong is a gut nobody is checking. But
the feeling it was reporting is real. It was just pointing at the
wrong organ.

## What convection actually needs

Convection is not motion. A pot on a stove with no cold surface
anywhere is full of motion and goes nowhere; it is turbulent, and it
is stuck. Convection is organized motion, and organization needs two
things a stirred pot does not have: a cold sink, so there is
somewhere for the hot air to go and be changed, and a return, so
what was changed comes back around to where it started.

The machine has neither, and we can say so with numbers we already
had.

There is no cold sink. Across the seven breaths the typical slot's
radius climbs from eleven to eighteen and never comes down. Radius,
in this family, is a coordinate the state can spend freely; there is
no regulator that takes heat out. Half of every slot is replaced
every breath, and the replacement is always warmer. Nothing in the
loop ever forces a slot to become smaller, more definite, cheaper
to carry. The one place a slot does become definite is the commit,
and the commit is a one-way door to the solver. The heat leaves the
loop and does not come back cold.

There is no return. We censused every road into the state last
week. The symbolic side of the machine, the solver's facts and its
committed edges, re-enters the reading at two percent and a third of
a percent of the band it lives in. And when the solver refuses a
diagram, when the parse is contradictory and the search comes back
empty-handed, that refusal says nothing at all to the reading. The
loop's most expensive and most honest signal, "this cannot be
right," has no road home.

So the stuck feeling was correct about the shape of the thing and
wrong about the mechanism. The air moves plenty. It moves in a
closed room with the heater on.

## The cold leg and the road home

This reframes what we build next, and mostly it reframes things we
were already building.

The cold leg is the sealed channel. We wrote last week about the
pressure cooker, the run that forced the machine's committed channel
to carry by taking away its bypass for a share of training rows.
That run opened a channel that had carried nothing. Read as
thermodynamics, the seal is the cold surface: on a sealed row, the
reading must become definite enough to commit or it cannot see at
all. Commit and sever is the leg of the cycle where heat leaves.
The regime the machine is deployed in should be the sealed regime,
not a loosened one, for the same reason a refrigerator does not run
with the door open.

The return is the road from refusal back to the reading, and the
first one is being built now. The token cooker, which fired this
morning, teaches a gate over the words each slot reads, conditioned
on what the solver has already found. It is the first road on which
the symbolic side speaks back into the reading at full band rather
than at a whisper. The next road is the solver's refusal itself,
re-entering as a signal the reading can act on. We had a release
valve on the roadmap that melted committed state on refusal; the
state, it turns out, was never frozen. What melts on refusal are the
deposits, the facts in the garage. Same valve, moved to the organ
that actually holds heat.

## A note on reading before building

The convection organ we would have built this morning was a stirrer.
It would have added motion to a state that already turns over by
half every breath, and the reads would have come back flat, and we
would have learned it the expensive way. We learned it from a numpy
script on states we already had, before the GPU was touched, because
the rule held: audit the gut, then build.

The cooker is running. When it lands we will know whether the first
road home carries anything.
