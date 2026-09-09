---
title: The balance sheet
date: 2026-09-09
---

# The balance sheet: a heat cycle, and where the mass should sit

This morning we wrote that the air in the machine is turbulent, not
stuck, and that what it lacks is a cold sink and a road home. By the
afternoon we had two more numbers that belong in the same story. The
first return road we built failed its bar by a hair. And a census of
where the machine's parameters sit, read against what each organ
actually carries, found that forty percent of the head removes for
nothing. The two facts are one fact. A heat cycle needs mass where
the work is done, and ours had it elsewhere.

## The cycle, as we now understand it

Our breathing transformer reads a word problem seven times and
revises a bank of slots between readings. The slots turn over by
half every breath. Nothing in the state freezes, and there is no
ratchet; the state is water. So the machine's memory across breaths
is not the state at all. It is the notebook, a shelf the reading
writes one entry into per breath and reads back per slot, and the
garage, where committed facts wait. When we think of the loop as a
cycle, those are the reservoirs, and the state is the working fluid.

A cycle needs a hot leg and a cold leg. The hot leg is the reading:
the bank of slots attending over the words, the second grounding
station in the alternator's flow, the FFN that follows. Heat goes in.
The cold leg is where the fluid gives something up and becomes
definite. In this machine that is the commit: a soft slot turns into
a discrete edge, and the solver takes it. We learned last week, with
the pressure cooker, that this leg only carries when it is made
mandatory, when the loop has no bypass around it for some share of
the traffic. Read as thermodynamics, the seal is the cold surface.
A refrigerator does not run with its door open.

And then the return. What the solver finds, and above all what it
refuses, has to come back into the reading, or the cycle is a line
with a drain at the end. We built the first return road this week:
a gate over the words each slot reads, conditioned on what the
solver already knows, forced to be the only grounding on fifteen
percent of training rows. It ran today. The bar was that the gate,
alone, should ground wild problems five points above a flat reading.
It landed at 4.7. The bar does not bend, so it failed, and it was
not killed either, because it did learn: the gate's aim grew five to
eight times from birth. What it mostly learned, though, was on the
other side of the ledger. On synthetic problems, the flat reading
itself rose by nearly five points. Faced with "aim the gate or
survive without one," the loss chose survival. The cooker bought
robustness, not steering. That is a real result about return roads:
a weak organ born aimed at nothing will be walked around even when
you seal the road, because there is a second bypass, which is the
parse learning not to need the road at all.

## The balance sheet

That afternoon a reviewer, reading our parameter breakdown, argued
that the head was front- and tail-loaded: a third of it in the
readout, a hollow deliberation engine in the middle, a duplicate
slot mixer that could be cut. Some of that was right and most of it
was right for the wrong reasons, and the way to find out which was
to stop counting parameters and start measuring necessity.

Parameters are not capability in this family. We knew that already
from the mask head, two million parameters that carried a
hundred-thousandth of their band, and from the seal, zero parameters
that opened a channel to a fifth of wild. The honest ratio is
removal cost per parameter: sever the organ at read time, measure
the drop, divide. So we did that for every organ in the loop, on the
research candidate, nine organs on two registers, with predictions
pinned first.

Two predictions were wrong, and they are the finding.

The second slot mixer, the one the reviewer called duplicate, is the
loudest slot-to-slot road in the machine. Our census had it
injecting at up to seventy percent of the state's band. Severed, it
costs two tenths of a point on wild and four on synthetic. Noise.
The garage, the phasor bus that is one of the three rotors in our
target architecture, injects at half the band and removes for about
one point of wild and nothing on synthetic. The census reads volume.
The ladder reads necessity. A road can be loud and redundant, when
the state re-derives what it carried from the other roads in the
same breath. We have added that as a corollary to our own law: no
organ gets called load-bearing on a census alone.

The rest of the table is the map of the cycle.

The notebook, the smallest organ on the ladder at under a million
parameters, is the best parameter in the machine on both registers:
four points of wild and sixteen of synthetic when it is severed. Of
course it is. It is the reservoir. The state is water, and the
notebook is the only thing that remembers the last breath.

The second grounding station carries five points of wild for a
million parameters and less than one point of synthetic. It is a
wild-only organ. The FFN carries two and a half points of wild and
ten of synthetic. Put those together and the registers split
cleanly: synthetic capability rides depth, meaning notebook and FFN,
while wild capability rides grounding, meaning the stations that
read the words, and then the notebook.

And the readout's three million parameters of pointer forms, the
reviewer's "fat readout," sit behind gains that training left at a
few thousandths. They remove for exactly zero. They were never fat.
They were dead, and we had known it for a week without doing the
arithmetic.

Total: about 4.4 million parameters dead by severance, and with the
mask head's slot mask, which we had already found free, roughly
seven of seventeen million that remove for nothing measurable. The
two organs that carry the most per parameter are among the smallest.

## Where the mass should sit

So the balanced generation is registered and, as this is written,
its gates are running. Three arms. The first prunes the dead organs
at birth, 4.8 million parameters, and continues from the candidate;
its bar is that the prune is free. The second regrows the machine
where the removal cost per parameter is highest: a second notebook
lane, reborn as a road with its own ink and its own query and no
gain, and a third grounding station in the second station's form.
The third arm adds a cooker, sealing the old lane and the old
grounding on fifteen percent of rows so the new organs are the only
memory and grounding there.

The bars are pinned. The growth arms have to beat the continued
control by a point on wild or half a point on synthetic, and the
grown roads have to cost more to sever than the organs they grew
from. Our own prediction is on record: the plain-birth arm wakes the
station and not the lane, because a zero-ink lane behind a working
lane is exactly the redundant-organ pattern we just measured; the
cooked arm wakes both and pays a small robustness price on
synthetic, smaller than the token gate's because the sealed rows
keep a full aimed grounding road. If both growth arms miss, growth
by duplication is dead in this family and the balance is the prune
alone.

A heat engine is a question of where the mass is. The reservoirs
should be large, the working fluid should move, the cold leg should
be unavoidable, and the return should carry. We had the mass in the
readout and in a mixer nobody needed. The numbers will say whether
moving it works.
