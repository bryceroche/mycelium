---
title: One job, six resolutions
date: 2026-09-24
---

# One job, six resolutions: nested targets, Matryoshka keys, and a week of honest nulls

This week we built eight organs on the machine's message bus and measured
every one of them on 311 held-out problems the model has never seen.
None of them moved a decision the right way. Two of them moved the very
cell they were built for the wrong way, at two and a half standard
errors. We also built the thing the whole design was waiting for, the
solver consulted three times per problem during training and inference
alike, and it read as a null at twice the step cost. That is a lot of
negative evidence to carry into a design conversation, and it turned
out to be exactly the evidence the design needed.

## What the nulls said

The machine reads a word problem into a factor graph over seven
breaths. Breath zero grounds the slots in the text; six loop breaths
refine them; every breath's output is graded against the full answer.
The wall we have been pushing on all month is argument binding: a
relation like "twice as many fish as cats and dogs combined" has to
point at the right slots, and on wild problems it points right about 45%
of the time.

Two censuses on Monday named the wall precisely. Most wrong arguments
are not confusions between different things. They are confusions
between the same thing at different moments: the apples bought, the
apples eaten, the apples left. Over half of the wrong mass points at
quantities that appear in no sentence at all, because they are the
results of earlier steps. The candidates share an identity by
construction. What separates them is role and time.

So we built a register on the bus that carries each slot's identity
and role, and every road we gave it to reach the pointer was ignored.
The loss had an easier path, the existing pointer over the slot states,
and a message on an optional road is never picked up. When we closed the
easier path on the exact cell where the register held something the
pointer could not, the register carried that cell worse than the pointer
had. Then we put the identity directly into the attention that reads
tokens into slots, on the road the loss cannot bypass, and at a scale
where it acted, it hurt the same-noun cell by 3.6 points and touched
nothing else. An exact lexical tag pulls a slot toward every mention of
its entity, which is the definition of the tie we were trying to break.

## Every breath does the whole job, at a sharper resolution

The correction that came out of this is small to state and changes the
training target for every breath. Do not give different breaths
different tasks; that is a tug-of-war between gradients, and we have
measured that tug in every organ that entered loud. Give every breath
the same whole job, graded at its own resolution, the way a diffusion
model is graded at its own noise level at every step.

For that to work the targets must nest. A correct coarse answer must
never be contradicted by the fine one, so every breath's gradient points
the same way as the last. For arguments the hierarchy falls out of the
censuses: coarse is the set of candidates that share the entity, medium
is that set split by order in the chain, which is the solver's own
ordering of facts, fine is the exact slot. For digits it is order of
magnitude, then leading digits, then the number. For operators it is
the family, then the operator. For types it is given versus derived,
then the exact type. That is a compiler lowering an intermediate
representation one level per pass, in parallel across every head, from
intent down to binding.

The prediction is concrete and cheap to test. Today's ladder grades
breath one on exact digits it cannot yet know, so the gradients from
different breaths on the same shared weights disagree. With nested
targets they should agree, and the cosine between breaths' gradients is
a number we can read on the trained model with no training at all. That
is the mechanism bar, and it gets read before any accuracy bar.

## Matryoshka keys

The bus stores everything as complex phasors, one per plane, 256 planes.
Binding a role to an identity is a rotation, plane by plane, and
unbinding is the reverse rotation. Rotation commutes with truncation:
if you keep only the first k planes, binding then truncating equals
truncating then binding. So a coarse key, the first k planes, is a
prefix of the fine key, and its match score is a partial sum of the
full one. A breath-one key and a breath-six key live in the same
coordinates. Compression costs detail and never breaks
interoperability, the same property Matryoshka embeddings have in
ordinary vector spaces.

That gives a schedule. Breath k unlocks a growing prefix of planes. The
mask over which slots may talk to which softens at breath one and hardens
by breath six. Each breath's target is one level finer. The clock, which
already carries 85% of the state's motion as rotation, tells the model
which breath it is on, the way a timestep tells a diffusion model where
it stands. All of it is fixed and scheduled, so training and inference
walk the same curbs, which this week taught us is not optional: the
consult we built is only as good as the parse it consults on, and a road
trained on dense gold facts learned nothing usable from the sparse,
error-laden facts the read hands it.

One twist makes this bite on the wall. A random phasor codebook has no
notion of coarse. Design it hierarchically: low planes shared by every
mention of an entity, high planes separating mention and time. Then
early breaths group the same-noun candidates and later breaths split
them, which is the order the identity key got backwards. Delivered all at
once, identity blurs the choice. Delivered progressively, it confirms the
group first and resolves the member later.

## The NL certifier

The solver is the math certifier: it accepts a graph or refuses it. The
text needs a certifier too, and everything it needs is already a read
rather than a new organ. Is each slot's attention concentrated on tokens
whose identity matches the slot's own? Does each given's value appear as
a numeral where the slot says it does? Are there numerals in the text no
slot claims, or one numeral claimed twice? Does a relation's operator
agree with the cue word in its clause? And the strongest one: render the
decoded graph back into prose and ask how much of the source it
reproduces. None of these is a loss. They are evidence, read after each
consult, feeding the mask for the next breath and the accept-or-refuse
decision at the end.

## What stands

The three consults stand as built, the same three curbs at training and
at read, with a bit-exact fence between the bus's formal language and the
head's arithmetic. The gentle continuation of the certified body at a
tenth of the usual learning rate holds its score and sets the row record,
fifteen of 311. And the order of arrival is unchanged: value passing,
then the atlas, then the mask, each now on nested targets and a
Matryoshka bus. The first read is the gradient cosine, and it costs
nothing.
