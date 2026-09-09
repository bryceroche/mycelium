title: The player and the game
date: 2026-09-09

# The player and the game: where dynamic attention actually lives

Two days ago we published a census of every road into our machine's
state, and the loudest line in it was a silence. The mask head, two
million parameters we had written about as the player who redraws
the board each cycle, injected at about a hundred-thousandth of the
band it lives in. Its gain had been born at 0.02 by law and training
had closed it to 0.0002. In the newest generation the organ's own
output was growing while the gain kept falling. It was trying to
speak, and the loss kept turning it down.

We had a law for that by then. Every organ must be a mandatory road:
if it can be bypassed, assume it will be, and take the bypass away
for some share of the traffic. The pressure cooker had just proved
the law on the machine's committed channel, from nothing to a fifth
of wild problems in one run. So we built a second cooker for
attention. On a share of training rows the baseline mask among the
diagram's slots was severed, and the mask head's own logits became
the only lanes. If the player wanted to see anything, the player had
to draw the board.

## The organ woke, and nothing happened

The law held. On the sealed rows the mask head's gate now carries at
up to 0.84 of its band, from 0.00001. Forced to carry, it carried.
It even learned something specific: a gate that undoes the bias the
solver's committed edges add to attention, anti-aligned at −0.70.

And no number moved. Wild accuracy on the cooked arm, 0.2491; on the
control, 0.2506. Synthetic, 0.9626 against 0.9610. Every guard
passed, every read the same.

The explanation was in a number we read before firing and should
have read before building. With the slot mask severed *entirely*,
attention among the slots all-to-all, the champion candidate scores
0.2521 on wild against 0.2511 with its mask. The control reads
slightly better unmasked. The slot mask was never load-bearing. The
seal removed nothing the parse needed, so the organ, forced to
carry, had nothing to carry. A cooker had cooked an empty pot.

## Headroom first

The lesson became a corollary to the law, and it is the kind that
saves weeks. A cooker converts headroom, and headroom is the
bypass's removal cost, measured first. The pressure cooker had a
forensic behind it: sever the residual, wild drops from 0.26 to
0.0000, so there was 0.26 to convert, and the cooker converted a
fifth. The mask cooker had no such read. Its headroom was zero.
Sever, measure the drop, then cook. If the drop is nothing, the road
is not where capability lives, however loud you can make the organ
that owns it.

We had also, earlier in the week, measured how open the slot masks
were on wild text and found that more open went with more correct.
We read that as a confound at the time. It was the same fact seen
sideways: closer to all-to-all was closer to what the machine
already wanted.

## Why the slots didn't need a mask

Look at what a slot is. There are twenty-four of them, plus a few
scratch. They already share three bulletin boards: a notebook that
carries state across the cycle, a shelf on the phase bus where any
slot can park a relation and any slot can read it, and, it turns out,
dense attention among themselves. Twenty-four things talking is not
congestion. Letting slot three see slot seven costs nothing and buys
global consistency, and the machine prefers it. There was never a
selection problem to solve at that level, so an organ built to solve
it had no work.

## Where the game is

The token side is a different country. A word problem is a hundred
to three hundred tokens, most of them narrative scaffolding and
distractor quantities. Every slot has to ground itself in a specific
span of that text: this slot is the "5 apples", that one the "3
baskets", that one the "times". And we already know how this machine
fails on wild prose. It rarely fails at the shape of the diagram.
The paper measured the survivors of every repair we built and found
their internal state 99.6% decodable; what was wrong was a pointer
aimed at the wrong place. The machine binds the wrong number to the
right role. That is a grounding failure, and grounding is a
slot-to-token act, not a slot-to-slot one.

So the hypothesis, stated plainly and not yet measured: **dynamic
attention has no job among the slots, and its whole job is between
the slots and the words**, keeping each variable's read of the text
from bleeding numbers out of the rest of the paragraph while it
refines its assignment.

## The test before the build

The corollary says we do not build a token cooker on a hypothesis.
We sever first. The registered read: take the champion candidate,
force each slot's attention over the prompt tokens to be flat, every
token weighted the same, and read wild accuracy. Two outcomes,
pinned before the card is touched:

- **Headroom exists.** Wild falls hard, from 0.25 toward 0.05 or
  below. The machine depends on specific token grounding, the gap is
  real headroom, and a cooker on that road is worth building.
- **No headroom.** Wild barely moves. Then token attention isn't the
  bottleneck either, and the finding is larger than either cooker:
  dynamic attention masking, at any level, is not where this
  machine's wild capability lives, and the effort goes entirely to
  the Tolstoy needle and the certified solver interface.

Either answer is a result. The player is awake now. Whether it has
a game is a number, and the number is next.

---

## Postscript, the same morning: the number

The severance probe ran before dawn. Flatten every slot's attention
over the tokens during the six loop breaths, keeping the first
reading intact, and wild accuracy falls from 0.2511 to **0.1536**,
synthetic from 0.9556 to **0.7704**. Flatten the first reading too
and both collapse, 0.05 and 0.005, the floor. So the re-reading is
load-bearing: two fifths of the machine's wild capability and a fifth
of its synthetic capability ride on which words each slot reads as
the parse firms. The pot is not empty. Headroom of ten points on
wild and nineteen on synthetic, measured before a line of cooker is
written.

The texture inside the number is the useful part. The machine has
two roads from slots to words, the main reading bank and a second
one inside the alternator's flow-through stations, and on wild
problems the second road carries two thirds of the headroom. On
synthetic problems the main bank carries nearly all of it. Wild text
is being grounded, mostly, by the organ built for the ping-pong.

Outcome A. The player has a game, and it is between the slots and
the words. The token cooker is registered with its bars: on a share
of rows the mask head's own gate over the tokens becomes the
grounding, starting from the flat field this probe measured and
climbing, or not. That run needs the word.
