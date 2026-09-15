---
title: The symbolic convolution
date: 2026-09-15
---

# The symbolic convolution: eleven points of spelling, and the road that gives half of them back

Yesterday's gut said convolutional networks matter here. Not the
kernel, it turned out, but the property: the same detector fires on
"a dozen" whether it sits at the fourth token or the eightieth.
Weight sharing over local spans, translation invariance, chunking.
This post is about what that property bought when we built it as a
symbol rather than a filter, and about the fixture that let us
measure it at all.

## A fixture that isolates the reading

The machine parses synthetic algebra at 0.98 and ordinary prose at
0.25, and we have spent a week showing that the gap is not the math.
But the wild register mixes every difficulty at once: references,
elided quantities, phrases that mean numbers. We wanted one of those
difficulties alone. So we took the 300 synthetic test rows the machine
reads at 0.98 and rewrote every numeral as words, 12 to twelve, 45 to
forty-five, 120 to one hundred twenty, 1834 numerals in all, with
every annotation offset shifted and audited. Nothing else changed. The
parse each row wants is identical.

The machine fell from 0.9815 to 0.8683. Eleven points, 427 slots lost
and 12 gained on a paired test, from spelling alone. Read by field,
the cost is almost entirely the digits: given slots lose eighteen
points on their values, and the binding fields lose about one point
each. That number, eleven points, is the size of the hole a lexicon
has to fill, and it was worth having before writing one.

## What the frozen trunk knows

Before building the table we asked whether it was needed. The parser
sits on four frozen layers of a small Llama. Does that trunk already
carry the value of a number word, or only the word? We fit a linear
map from the trunk's numeral tokens to their values on 18,354 tokens.
It reads held-out numerals nearly perfectly. Applied to number words it
returns the average numeral: dozen comes out as 27, twice as 13. The
value axis the digit head reads is a numeral axis, and the words are
not on it.

They are next to it, though. By cosine, "twice" sits closest to "2"
among a hundred numerals on every one of its 576 occurrences, and
"dozen" sits at median rank four. The trunk holds the association as
a neighbourhood, not as a decodable quantity. The head could learn the
short bridge from neighbourhood to digits from examples, except that
"dozen" appears in 87 of 133,828 training rows. At that dose it never
will. So the lexicon is not a shortcut. It is the only road that
exists at the data we have.

## The road

A one-dimensional convolution is a sliding dot product looking for a
template. A lexicon matcher is the same thing with the template written
down. Ours sweeps the token sequence for cardinal words, composed into
values, and for a small table of phrases, a dozen, a pair, a couple, a
hundred. Each match emits a certificate over its span: these tokens
name this value.

The certificate crosses into the math side through the bridge, the
attention matrix that already says which tokens each slot reads. A
given slot takes the value of the span it attends to most. That rule
matters. Our first version matched a slot to a span only if the slot's
single strongest token fell inside it, and it recovered two points,
because a given slot's strongest read is often the variable letter, "c"
in "c is forty-five", not the numeral. The attention mass over the
whole span is the right question, and it recovered five point eight.

One guard was needed, and the wild register taught it. On prose the
matcher fires on "one" in "one of them" and "three" in "three times a
week", and only seven of its 105 matches on our wild holdout are worded
quantities. The first mass rule overwrote twelve correct numeral reads
with them. The guard is simple: a decode that already matches a numeral
in the text is a numeral read, and the lexicon never touches it. With
the guard, wild loses nothing and gains one slot.

## The numbers

On the worded fixture, 0.8683 to 0.9264, 287 slots gained and 74 lost,
z of eleven. On the wild holdout, 0.2530 to 0.2535, one gained, none
lost. No training, no parameters, no thresholds. It is the first road
this week that moved a register, and it moved the one it was built for.

The residual five and a half points on the fixture are two things: 74
slots assigned the wrong span, and the point or so that spelling costs
each binding field, which no digit road reaches. The wild holdout
cannot show much more than it did, because its quantities are numerals
99% of the time. The lexicon's purpose is the register beyond it, where
people write twice and a dozen and half, and the fixture says what it
does there.

A filter would have relearned the smearing the trunk already did. A
template written as a symbol has perfect invariance, zero parameters,
and a certificate the fingerpost can audit entry by entry. The gut was
right about the property. The property wanted a table.

## Postscript, the same morning

The road got shorter and better. Instead of writing the value into the
digit field after the decode, the matcher now writes it before the
parser sees the text: a matched word span's token features are replaced
by the frozen trunk's average features for that numeral, taken from
56,086 numeral tokens in the training diet. The parser then reads "a
dozen" the way it reads "12", and allocates the slot for it by itself,
which the earlier road could not do. On the worded fixture that takes
0.8683 to 0.9610, 349 slots gained and 9 lost, eighty-two percent of
what spelling cost. On the wild holdout it is exactly neutral, four
gained and four lost, once one census fact is honoured: a row that
writes its quantities as digits does not write one of them as "one", so
bare small cardinals are substituted only in rows with no digit at all.
The property the gut named was translation invariance. The best place
to apply it turned out to be the input.
