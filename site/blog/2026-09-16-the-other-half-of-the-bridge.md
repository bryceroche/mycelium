---
title: The other half of the bridge
date: 2026-09-16
---

# The other half of the bridge: what a shoal knows that our loop didn't

A herring in a shoal watches a few neighbours and adjusts. The
neighbours adjust back. Nobody sees the whole shoal, and the whole
shoal turns as one, because influence runs both ways and keeps
running. This week the question was whether that picture, and the
convolutional networks that formalise the same locality, had anything
left to give our machine. Most of it, it turned out, we had already
tested and spent. One piece we had never built.

## What was already spent

Locality on the token side paid exactly once, as a symbol rather than
a filter: a sliding matcher that writes a numeral under a number word
recovered most of what spelling costs. A learned local kernel over the
tokens learned to be the identity, because the frozen trunk had done
the local mixing before we arrived. On the slot side, the neighbourhood
rule, each factor slot attending only to slots in its sentence or
sharing its variables, was measured and found weightless: all-to-all
read 0.2521, masked read 0.2511. And when we asked whether argument
pointers favour recent facts, the way a shoal favours near neighbours,
the gold said no: on prose a third of the pointers reach forward to
something the text has not said yet, and half of them do on the
synthetic register. Recency is a coin flip. The mixer's all-to-all
shape is right for a graph whose references run both ways.

## What was never built

Then the gut said the shoal still had something, and the honest way to
check was to draw the bridge and look at the arrows.

The machine has two mediums. Tokens carry the text, slots carry the
factor graph, and the bridge between them is an attention matrix: each
breath, every slot spreads its attention over the tokens and reads.
Seven breaths, seven reads. That is the whole bridge, and every arrow
on it points the same way. The slots read the tokens. The tokens are
computed once, before the first breath, and they never hear what the
slots did with them. The only thing that ever travels the other way is
a certificate at read time, a mask that says "these tokens are taken",
which is a shout across the water, not a message.

Put beside the shoal, the asymmetry is plain. Our fish adjust to the
current and the current never adjusts to the fish. Put beside our own
history, it is plainer still. The June engine, the one that matched the
exact solver on KenKen and generalised across three puzzle families,
was learned belief propagation on a factor graph, and belief
propagation has two halves: variables tell factors, factors tell
variables, alternating until they agree. At the language bridge we
built the first half and stopped.

## The write-back

The missing half is small to state. After breath k's slots have read,
take the same attention matrix, transposed, and carry each slot's typed
state back to the tokens it read. A phrase's tokens now say "claimed by
a given slot as twelve" or "still unclaimed". Breath k+1 re-reads a
text that knows what was committed to it. A second slot reaching for a
phrase that is already taken meets the claim as content rather than as
a prohibition, and competition for phrases becomes two-sided, which is
the shape of the failure we have been chasing on prose all week:
the machine knows a relation is there and cannot say which quantities
it binds.

It enters as a road, in the sense this project uses the word. The
message is scaled to a fixed fraction of the token's own norm, not a
learnable gain that the loop can vote down at birth, and the next
breath's keys and values are built from the written-back tokens, so
there is no path through the bridge that skips it. It costs about a
million parameters. It is the "expand, collapse, repeat seven times"
that was named as a gut two days ago, before the bridge had been drawn
clearly enough to see which half was missing.

## The risk, stated before the read

A learned message can learn to be nothing. The token kernel did, and
for a good reason: the loss could be solved without it on the register
it was trained on. On synthetic algebra the one-way bridge already reads
at 0.98, so the write-back has no work there and would learn none. Its
work is on prose, where the binding is the thing the loss cannot solve
otherwise, and that is why it runs beside the newly annotated rows
rather than alone. Bars are pinned: paired against the same diet
without it, on wild and on mint, with the census reading how much of
the token state the message actually carries at every breath.

The shoal's lesson was not locality. We had squeezed that fruit. It was
mutuality, and the arrow we had never drawn.
