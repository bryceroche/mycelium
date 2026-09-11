---
title: The line cooks
date: 2026-09-10
---

# The line cooks: five things the machine told us in one day

A single day's reads, in the order they happened, because each one
changed the next. Two were bugs in the clock. One was a question
about who gets paid. One was a wire we had never strung. And the last
is the machine's steering wheel, turning for the first time as this
is written.

## 1. The clock was writing memories in the wrong frame

Every breath, the state's clock planes turn sixty degrees. Every
breath, the machine also writes two memories from that state: an ink
into the notebook, and a deposit onto the bus. Both writes go through
dense maps, and dense maps do not respect planes. So the phase of the
clock at the moment of writing was smeared through the whole memory,
and when the memory was read three breaths later, nothing undid it.
The reader received a bearing three breaths stale, mixed into
everything else.

The fix is a canonical frame: de-rotate the clock block to phase zero
before either memory is written, and re-phase what comes back to the
reader's own bearing. One primitive, both memories. It was free, and a
hair better on synthetic, a new record at 0.9708. It did not make the
bus more necessary, which is what I had predicted; the bus's removal
cost went down, not up. The frame's value turned out to be what it
made possible next.

## 2. The machine was smuggling content through the clock

When we cut the residual connection earlier in the week, we exempted
the clock's sixty-four planes so the notebook's ink would not
overwrite the compass. Then a spectral read asked where the machine
keeps the difference between adding and multiplying, and the answer
was: fifty-five percent on the clock planes, with its six strongest
planes all clock planes. The clock is not a pure coordinate system. It
is a rotating frame, and the amplitudes in that frame are free, and
the loss had moved a third of the state's variance onto them, up from
a sixth, because they were the one road the cut left open.

We measured the leak directly. The v1 candidate, read with the clock
planes cut too, fell from 0.2608 to 0.1497 on wild and from 0.9654 to
0.6513 on synthetic. Eleven and thirty-one points were riding the
exempt block across the cut. Then, because the canonical frame carries
the bearing through the ink, we cut everything: the total cut, no
exemption. The result reads 0.2526 on wild, inside the guard, and
0.9757 on synthetic, another record, with the clock probe perfect and
the clock planes' share of the variance back down to 0.22. The bypass
is closed. This is the residual-seal law, which we wrote on Wednesday,
catching my own Thursday design.

## 3. Do farmers get tipped?

A question from the gut: we have a chain of seven breaths and one
loss at the end. Does the gradient reach the farmer, the first breath,
who never sees the customer but without whom there is no meal?

We built a gradient census: a probe on the state entering each breath,
the real training loss, backpropagation, and the norm of what each
breath receives, with the readout's own state as the unit.

| arm | b1 | b2 | b3 | b4 | b5 | b6 |
|---|---|---|---|---|---|---|
| residual family | 1.38 | 0.92 | 0.83 | 0.80 | 0.80 | 0.81 |
| stellarator v1 | 1.12 | 0.70 | 0.58 | 0.48 | 0.45 | 0.45 |
| v2, total cut | 0.99 | 0.38 | 0.07 | 0.01 | 0.02 | 0.10 |

The farmer is tipped everywhere, and not for the reason I predicted.
Breath one's state writes the first ink, and the first ink is the
most-read entry on the shelf at breaths two through four, at half the
attention. The farmer is paid through the notebook. Who starves under
the total cut is the middle: breaths three to five receive one to
seven percent of the readout's gradient. The line cooks, not the
farmer.

Before fixing their pay we asked whether they were working. Idle
breaths three to five at read time, gate closed and no junction, and
the v2 candidate reads 0.2565 on wild, four tenths of a point *better*,
and within noise on synthetic. Idle breath six as well and it costs
five points. The middle breaths are decorative under the total cut.
The machine's deliberation is effectively the grounding read plus
breaths one, two, and six, and the synthetic record was set with the
middle asleep.

So the credit remedies are moot: there is nothing to credit. The
direct remedy, a loss at every breath, was tried in August and
convicted; it flattened a matured competence. The two live options are
to skip the middle, which is a forty percent speed gain at no cost by
this read, or to give the middle a job. We chose the job.

## 4. The wire from slots to words

The one place the severance ladder found large headroom was the road
from slots to words: flatten the slots' attention over the tokens
during the loop and wild drops ten points, synthetic nineteen. That is
where the machine's wild capability lives, and its first organ, a
learned gate over the words, missed its bar by three thousandths and
bought robustness instead of aim. The linkage needs to be stronger,
and on wild text especially, where the slot states barely carry the
operation at all.

The buff we chose is not a bigger gate. It is an aimed re-read.

## 5. Commit on every breath; receive a core

The design we have written about all week is that each breath commits
a tentative parse to the solver and gets back facts, a refusal, and on
a refusal the minimal unsatisfiable core, the smallest set of factors
that cannot coexist. Honesty requires the correction: that is the
design, not yet the machine. Today's training loop reads facts frozen
from breath zero, and the per-breath ping lives in a trainer we proved
and shelved. So we built it in two stages.

Stage one is running now: the wheel at read time. After every loop
breath but the last, the breath's parse is committed, solved
completely, and on a certified refusal the core's slots get a
spotlight for the next breath, a positive bias on the token scores of
their source sentences, in both grounding roads, so that the slots the
solver has proven wrong together re-read the evidence they came from.
Open-only, never a mask. On a sixteen-row smoke the solver refused five
or six parses at every breath with cores of four slots, so the middle
breaths have work at every step. Whether a parse trained on unaimed
reads can use an aimed one is the pinned question, and my prediction
on record is a narrow miss. Stage two trains under it.

That is the job the line cooks get: not a tip, a customer. The solver
walks into the kitchen at every breath and says which dish is wrong.
