---
title: The gold was in the wrong language
date: 2026-09-17
---

# The gold was in the wrong language: a week of corrections

This week we trained the machine on new prose stories, read a null,
and then found out the null meant nothing. The stories had been
labelled in a convention the machine cannot read. What follows is the
sequence of reads that found it, because each one narrowed the next,
and the thing we do differently now.

## The diet that did nothing

We had 473 GSM8K stories annotated by a language model into our
factor-graph dialect, each certified by the symbolic solver against
the human answer, each multiplied by three resampled copies with new
numbers. We trained three arms on them and read the wild holdout. The
data alone moved wild by 0.006, inside one standard error. The two
new roads for carrying phrase-to-slot information added nothing on
top.

A null is a fine result when the diet was readable. So we asked a
question we should ask of every new data source before training on
it: does the machine read the rows it just trained on? It read them
at 0.07. The base, which never saw them, read them at 0.03. The
holdout it reads at 0.25. The machine had not learned the stories
because their labels were foreign to it.

## Three conventions

The machine's output is positional. Twenty-four factor slots, each
graded against the gold factor in the same slot. That means the gold
must be one specific sequence, and the sequence is a convention.

The first convention was variable numbering. Our synthetic training
rows number variables in order of first mention in the text. The
annotator numbered all the givens first. Renumbering by first mention
did not move the read.

The second was factor order. The gold builder sorts any fully-spanned
row by span start, which is the synthetic rows' order. The
annotator's relation spans cover whole sentences and start before the
givens inside them, so the sort put relations ahead of their own
arguments, an order the machine had never seen. Removing the sort
doubled the read to 0.07.

The third was the one that mattered, and we found it by looking at
decoded parses next to gold rather than by census. On every prose row
the machine has ever trained on, every factor introduces exactly one
new variable, and that variable's index is the factor's slot. A given
at slot two is variable two. A relation at slot five introduces
variable five, its result or its one unknown argument. Where the
prose says "James spent 40 years, ten more than his partner," the
dialect writes an add whose result is the known 40 and whose new
argument is the partner. All 13,475 prose rows in the diet obey this.
The synthetic rows obey a different rule entirely. The machine
carries two conventions and picks by register.

There was a fourth, embarrassing one. The dialect has two operators,
add and multiply, and the operator target is a single bit. The
annotator had written subtraction and division, and the gold builder
wrote every one of them as multiply. Fifty-nine percent of the new
rows carried corrupted structure on top of foreign order.

## What order costs, measured

We built a read that matches predicted factors to gold by content
instead of by slot. On the holdout the base machine scores 0.25
positionally and 0.30 matched. Order is about four and a half points
of the gap, and content, chiefly reading the numerals and pointing at
the right arguments, is the rest. But order was the whole of the
retrained arm's loss on the holdout: matched, it sat within 0.004 of
the base. The diet had shifted the machine's ordering habit and the
positional read charged it for that, while its understanding of the
holdout was intact.

That is a real brittleness, and it is ours, not the data's. A loss
that punishes a correct graph for its serialization is teaching
against generalization. The fix is a set-prediction loss, matching
before grading, and it is registered.

## The pivot

Then a simpler question. GSM8K's own solutions carry calculator
chains, `<<48/2=24>>`, and a chain is a factor graph with a human
answer attached. A parser from late August already turned them into
our dialect, capped at values of 300. We raised the cap, excluded the
holdout by text, and added a lexicon tier so a constant the question
states in words, "a dozen," "per hour," "percent," becomes a given
with the word's span.

The yield: 3,303 unique stories at zero tokens, every one certified by
the solver against the human answer, in the exact convention the
machine already reads. We know it is exact because the parser
reproduced the diet's own factor sequences on all 1,165 stories they
shared. The diet had 1,225 unique prose stories. It now has 3,303,
and a rebalanced mix with less synthetic template and more prose is
training as this is published.

## The rule we keep

Gold enters in the machine's convention, or it is not gold. Before
any arm trains on a new source, the fit read runs: the machine reads
the rows it is about to train on, and the number says whether the
labels are legible. That read costs a minute and would have saved the
week.
