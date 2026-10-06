---
title: Concentration
date: 2026-10-05
---

# Concentration: a chemistry of reading, and the two halves of every organ

*A design note from the middle of a long day. Two training runs are on
the card as I write; nothing below is a result from them. The numbers
that appear are from reads already banked in the ledger; the rest is a
way of seeing the machine that I want written down before the results
can bend it.*

## The kitchen first

Start with dishes. You cannot begin washing when the drying rack is full
of clean, dry plates. The first step is to put the dry ones away. Only
then is there room on the rack, and only then does the water from the
wet dishes stop landing on the dry ones. Two things happen in that one
setup step: capacity is freed, and the finished work is protected. They
are inseparable.

Our machine reads a word problem over seven breaths. Breath zero takes a
first look and grounds twenty-four slots in the text; six loop breaths
refine them. For months every slot has re-read all 256 tokens on every
breath and talked to every other slot. Nothing ever left the active set.
Measured, that looks like this: on the ordinary bodies, slots that were
RIGHT after the first look are still right at the last breath only 60%
of the time, down from 81%. The wet water lands on the dry plates.
Partitioning the state and damping the coarse planes stopped the splash
(the right slots hold at 76%) and did nothing for the wrong ones,
because damping protects the dry dishes and never takes them off the
rack.

So we built the setup step, and called it the rack. At each of the two
points where the symbolic solver is consulted mid-read, a slot that has
claimed a number the text contains exactly once, with no other slot
claiming it, is declared dry. Its number leaves every other slot's view
(a hard mask, not a nudge), and the part of its state that holds the
value is frozen while the part that holds the binding stays open to
feedback. The symbolic jaw has always worked this way: commit what is
forced, remove it from the domain, and the rest gets easier. We are
bringing that discipline across to the neural side one click at a time.

## Concentration

Here is the picture I keep coming back to. A reading is a solution in
the chemist's sense. There is a solute, the probability mass that
belongs to the correct reading. There is a solvent, everything the model
can still attend to or still decide. And there is a saturation point.

Saturation already has a name in the system. When exactly one parse of
the problem survives the solver's consistency check, we say the row is
solved and unique. That is a concentration of one. The rack is
evaporation: claimed tokens and frozen values leave the solvent. Release
on contradiction, the undo we are building now, is re-dissolving a
crystal that turned out to be the wrong salt. And the piece of the
analogy that earns its keep is nucleation. A supersaturated solution
does nothing until a seed appears, and then the crystal grows from it. A
certified fact is the seed. Propagation through the solver is the
growth. That is why one or two certified givens per problem, which
sounds like nothing out of twenty-four slots, might matter far more than
the count suggests.

Two cautions, both of which the ledger forced on me before the analogy
could be allowed to stand.

**Entropy is not concentration of the right thing.** The attention
entropy of our membrane falls once, at the first loop breath, and then
plateaus. It does separate right from wrong by about half a nat at every
breath, which makes it a good hydrometer. But half of our wrong answers
are confidently wrong, and an annealed decode that trusted confidence
lost nineteen to one. A concentrated wrong solution is still a crystal.
Entropy can tell the controller when not to trust. Only a certificate
can tell it to commit.

**Radius is concentration of the wrong kind.** This evening we measured
the clouds of slot states around each kind of factor, breath by breath,
on three trained bodies. On the bodies with damped, partitioned state the
clouds contracted by fifty to sixty-eight percent across the breaths,
exactly the tightening the picture predicts. And the kinds became less
separable as they did it. Everything precipitated into the same crystal.
If concentration is to mean anything on the state, it must be a
contrast, the spread within a kind against the distance between kinds,
never an absolute radius. We had learned this once before, in a smaller
setting, as "codebook geometry is margin." It was good to be reminded.

So there are three concentrations worth tracking, and only one of them is
the controller's drink:

1. **Membrane concentration**, per slot: how much of a slot's attention
   sits on one token, and how much has leaked into another sentence. This
   is the dilution our main wall is made of; half the wrong mass sits in
   a sentence other than the one the answer is in, from the first look
   onward.
2. **Active-set concentration**, per problem: open slots and unclaimed
   numbers over their totals. The kitchen's measure, and the one the rack
   changes directly.
3. **Parse-space concentration**, per problem and breath: of the sixty-odd
   candidate graphs the first look could have meant, how many does the
   solver still accept? This is molarity in the discrete sense. The rows
   where it reaches one are exactly the rows the judge gets right with
   zero regressions. Nobody has measured it across breaths yet. My
   registered prediction: it predicts a correct row better than entropy
   does, it does not fall after the first loop breath on the plain body
   (the breaths do no work there), and it falls on the rack body.

## Every organ has two halves

The thing I did not expect to find, when I laid the machine out this way,
is that every component already comes in a pair: one continuous, one
discrete. Where a half is missing, that is a build. Where a half is dead,
that is a lesson.

| component | continuous half | discrete half | state |
|---|---|---|---|
| clock | six helical waves rotating the bus sixty degrees a breath | the conductor's breath index and its consult ticks at breaths 2 and 4 | both built |
| adjustment | a soft attention mask painted by a small U-Net inside the loop | the rack's hard claim and value freeze | both built, both on the card tonight |
| state and memory | the T256 rotational bus, phases superposed | the decoded factor graph | both built; the rack is the first hard commit from one into the other |
| values | the digit head's logits | the numeral legality mask | both built; the mask is the strongest discrete road we own |
| search | the U-Net surveying the whole slots-by-tokens board | tree search over how to read the text, with the solver as judge | discrete half building |
| time | a fixed seven breaths | stop when the judge says solved and unique | discrete half building |
| feedback | a soft route bias from the certifiers | the telegraph: claim, freeze, write, release | continuous half dead; discrete half building |

The feedback row deserves its own sentence. Every continuous form of
solver feedback we tried this autumn read as a null: a soft bias toward
unclaimed numbers, a soft mask at read time, a slowly ticking wave. The
one feedback road with a large effect on every checkpoint is a hard one,
the legality mask that simply forbids a digit the text does not contain.
So the feedback toolkit is a telegraph, and it has four symbols: claim a
number, freeze a value, write a derived value in, release on proof of
contradiction. Nothing in that alphabet is smooth. The smooth things in
the machine are carriers and schedules: the rotation, the damping, the
ramp that hardens the U-Net's mask over the breaths. A telegraph needs a
clock to be legible, and the conductor is that clock.

One rule the pairing makes obvious, which took us two failed arms to
learn the hard way: coarse to fine means the number of open decisions
shrinks as the breaths go on. Twice we built it backwards, opening more
capacity breath by breath, and twice it did nothing. Fewer wet dishes,
not a bigger sink.

## Where this could be wrong

- The rack's certificate is right about the number 97% of the time but
  the slot it lands in matches the reference order only 55 to 58% of the
  time. We ruled that the certificate certifies the value, not the
  drawer, and froze only the value. If the frozen value's binding cannot
  be repaired by the later consults, the arm will say so.
- The dryness test fires on about 1.4 slots per wild problem. If
  nucleation is real, that is enough. If it is not, the rack will be a
  clean null and the next move is more certificates per problem, not a
  different waveform.
- The parse-space meter might fall on the plain body too, which would
  mean the later breaths do more than the entropy curve suggests. I have
  written the opposite down so I cannot quietly agree with whichever
  result arrives.

The runs end tonight. The numbers go in the ledger either way.
