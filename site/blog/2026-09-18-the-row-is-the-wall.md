---
title: The row is the wall
date: 2026-09-18
---

# The row is the wall: what a calibrated end-to-end read says

For a week we have been reporting wild accuracy per slot: what
fraction of the gold factors the machine decodes correctly on a
holdout of 311 GSM8K problems it has never seen. That number went
from 0.253 to 0.316 this week, through a legality mask on the decode,
a prose-heavy diet, and finally a model trained from nothing in
thirty-five minutes that matched the best warm lineage and beat it on
structure. Every one of those numbers was measured honestly and every
one of them holds.

Today we built the read that should have come first: hand the
machine's decoded graph to the solver, take the answer, compare it to
the key. That is the read MATH-500 will get, it does not care about
factor order, and it uses the same mask. We calibrated it on the
synthetic register, where the machine solves 278 of 300 problems, the
number we have known for months. Then we ran it on the wild holdout
for six checkpoints spanning the whole week.

Every checkpoint solves between two and eight of the 311 rows. The
base solves seven. The best per-slot reader of the week solves five.
At two percent, the standard error is about two and a half rows.
Every entry is the same number.

## Why six points of slot gain never reached the row

A row needs every factor right. A wild row has about six and a half
gold factors, and the per-row census says the median row gets four of
them wrong. One row in twenty is a single fix away. Eighty-three
percent are three or more fixes away.

That shape explains the week. The mask fixed a class of digit errors,
and the fixes landed on rows that were also wrong somewhere else. The
diets fixed a share of pointer errors, and those landed on other rows
that were also wrong somewhere else. Each road moved the slot count
and left the row count alone, because the errors are spread over most
rows rather than concentrated in a few.

The other half of the wall is the more uncomfortable one. Of the rows
the machine gets wrong, roughly half produce a graph the solver
accepts and solves cleanly to the wrong answer. Consistent is not
correct, at the row level, for half of all wild problems. That is
exactly the failure the answer key exists to catch during training,
and exactly the failure a certifier has to catch at inference, because
nothing inside the machine flags it.

## What we keep and what changes

The per-slot read stays as the diagnostic. It told us where the
errors are, and the census tells us how they are distributed. But the
score is the row, and the campaign's wild number is two percent.

What moves rows is whole-row reading, which at six factors a row means
per-slot accuracy far above 0.3 on the same problems, or an abstain
that keeps the machine to the problems it can do. The one-away rows,
four or five percent of the holdout, are where a certifier earns its
keep first. The from-scratch lineage is the one to push, because it
learns structure better than anything warm did and it costs
thirty-five minutes a run. And the row-level read now runs on every
chain, so the next six points of slot gain will be reported next to
the number that matters.
