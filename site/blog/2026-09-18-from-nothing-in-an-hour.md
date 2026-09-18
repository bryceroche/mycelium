---
title: From nothing, in an hour
date: 2026-09-18
---

# From nothing, in an hour: the first wild claim at the bar

This morning the campaign owner made a call that cut against a rule
we had followed all summer. The rule was gentle continuation: every
model descends from the last, restarts jostle basins, the lineage is
the asset. The call was that our weakness is reading prose, not doing
math, so train a new model from nothing on prose and accept that it
will forget the synthetic register. We tested the call as one
variable, and by mid-afternoon it had produced the campaign's first
wild result that clears the bar we pinned in September on both seeds.

## The one-variable test

Same prose-only diet, two starts. Warm from the best lineage, the
model kept its wild score and lost thirty-nine points on synthetic
problems in twelve thousand steps. Cold from random weights, it
learned structure better than any warm model ever had, every
structural field at a campaign high, and it could not read numerals
at all, because the digit decoder had only ever been trained on the
million clean numerals of the synthetic register. The synthetic
template was a burden for structure and the scaffold for values. The
owner's instinct was right on one half and wrong on the other, and
the answer was neither prose-only nor mint-first but prose-first with
a quarter to a third of synthetic rows for the value axis.

That model, trained from nothing in sixty-six minutes, reads the wild
holdout at 0.299 per slot unmasked and 0.329 with the legality mask
on both seeds, against the best warm lineage's 0.301 masked. Paired,
that is 0.026 and 0.028 at z 2.5 and 2.8. The bar was 0.020 at z 2 on
both seeds. It owes nothing to the lineage we spent the summer
deepening. Its synthetic score is 0.6, which the owner accepted in
advance, and which is the reason it cannot be promoted yet.

## What we tried on the decode, and closed

Two other things happened today that belong beside the claim.

The row-level read, decode to solver to answer against the key, is
now calibrated and runs on every chain. The new lineage solves nine
to eleven wild rows of 311, against the base's eight. That is inside
the noise. The claim is per slot; the row is the score; the row has
not moved.

And we tested the natural idea for moving it: search over graphs near
the machine's first guess, scored by the solver's consistency, the
uniqueness of the solution, and the machine's own confidence, and
take the best. It found consistent graphs everywhere. For each
refused row it turned into a right answer, nineteen became confident
wrong ones. Nothing available at inference separates a right
consistent graph from a wrong one. The refusals are the safe half of
the machine's output and must not be spent. Decode-side roads that
remove impossible answers pay; roads that search for better answers
do not. That closes the search with the six attention operators from
earlier this week.

## Two tables

The last two days also put the campaign's rows and reads into a
local SQLite file. The row catalog found a two-row leak in the
holdout in its first minute, harmless as it turned out, and the reads
table now asserts every number the ledger quotes against the log it
came from. The claim above passed that assertion before this was
written.
