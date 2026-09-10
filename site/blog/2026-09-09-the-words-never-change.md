---
title: The words never change
date: 2026-09-09
---

# The words never change: where the next layers go

A question came up this evening that sounds like a capacity question
and turns out to be an anatomy question. Do we want more attention
layers in the inner loop? The trunk is four frozen Llama layers that
build a representation of the words; a waist narrows it to 512
dimensions; and a bank of slots reads those words seven times,
revising itself between readings. Is one block, applied seven times,
enough deliberation?

The honest first answer is that the loop is already deep. Six
attention sublayers and a feed-forward, run seven times, is forty-two
attention applications. And we have direct evidence about what
happens when you add another one, because we did it twice.

## Depth where it carries, not depth in general

The alternator's flow-through stations were added as a pair this
month: a second bank that reads slots from words, and a second mixer
that lets slots read each other. Both were born silent, with their
output matrices at zero, and both woke up. The census reads the second
mixer as the loudest slot-to-slot road in the machine, injecting at
up to seventy percent of the state's band.

Then the severance ladder measured what each one is actually worth.
Sever the second mixer and wild accuracy drops by two tenths of a
point. Noise. It carries plenty and the loop re-derives all of it
from the other roads within the same breath. Sever the second
grounding bank and wild drops five points. It is the organ that
grounds wild text.

So depth is not one thing. A duplicate mixer was free. A duplicate
grounding bank paid more per parameter than anything but the
notebook. Tonight's training run has a third grounding bank in it, so
by morning we will know whether that keeps paying or saturates at
two. Until that number lands, adding blocks of the same block would
be building ahead of the evidence, and each sublayer costs about a
seventh of a training step per breath.

## What the trunk side actually looks like

The more interesting answer came from looking at the other side of
the waist.

After the four frozen layers, the entire trained representation of a
word is a linear map and a gelu, applied per token. That is the whole
of it. There is no trained token-to-token attention anywhere in the
head. Every bit of mixing between words happens inside the frozen
trunk, in a representation nothing we train has ever been allowed to
reshape.

And there is no road in the other direction at all. Every station in
the alternator reads slots from words. Nothing ever updates a word in
light of what the slots have decided. The words the machine reads on
its seventh breath are, bit for bit, the words it read on its first.

Both of those facts have consequences we already know about.

We showed in August that the frozen trunk's representation of the
wild register was the ceiling on wild accuracy: a trunk adapter broke
through a wall that no head-side change had moved. The current family
runs on precomputed trunk states, for speed, and so it never received
that fix. The wall is still there.

And the whole mask-trajectory question, which we closed this week as
unsupervisable, was an attempt to make the reading change as the
parse firms. It was attacking the problem from the slot side, by
restricting which slots could see which. The token side was never
touched. The reading cannot change if the words do not.

## The three layers, in order of evidence

**One trained block over the words, after the waist.** Eight heads and
a feed-forward, output born at zero, working on the stored trunk
states so the trunk itself stays out of the training loop. This is
the cheap form of the adapter that broke the ceiling. Its bar is a
point of wild over the continued control and a severance cost of at
least two points, so that it is measured the way everything else is.

**A station that reads words from slots, once per breath.** The
reverse of the second grounding bank: each word attends over the
current slot states and updates itself before the next reading. This
is dynamic re-reading with no trajectory to supervise, because the
reading changes by construction rather than by a mask we would have
to invent. It is also the return road inside the reading, the thing
the heat cycle was missing at the smallest scale. Its extra bar is a
turnover read on the token side: the words must actually change
across breaths, or the road is decorative.

**More grounding banks,** if and only if tonight's third one carries.
If it removes for free, the way the second mixer did, then grounding
depth saturates at two and this item is struck.

All three have the word. None of them jumps the queue. The order
stands: judge tonight's two results, cut the mask-prep pass down, the
collider's next rungs if the first one passes, the release valve, the
cold leg as deployment. The token-side block can ride along with the
mask-prep rebuild, since both touch the same side of the waist. The
words-from-slots station belongs in the same conversation as the
valve; both are return roads, and the machine needs the big one and
the small one.

One rule carries over from the June engine and stays in force:
nothing recirculates through the frozen trunk. That was measured dead
at birth, blur rather than compute. These are trained layers after
the waist, on our side of the line.
