---
title: The dancer's pixels
date: 2026-09-12
---

# The dancer's pixels: what a convolution is for, and which axis of our machine has one

Bryce noticed a family resemblance. We keep saying we want to segment
and classify the silhouette of the dancer, the shape the state makes
as it settles across seven breaths, and segmenting and classifying
silhouettes is the day job of a convolutional network. A CNN also
chops a picture into small patches and lets each patch talk only to
its neighbors, which sounds like the herring shoal: a fish tracks the
six or seven fish around it and the whole school turns as one. So:
is our message passing a CNN in disguise? And should we hand the
512-dimensional state to one, reshaped into a little picture, and
let it find the limbs?

The honest answer has a twist. The resemblance is real. The place
where it pays is not the state.

## The shoal was tested, and it was not load-bearing

Inside the loop, thirty-two rows talk to each other through
attention, and since the mask cooker they talk all-to-all. That is
the opposite of a shoal. We did try the shoal, directly, three days
ago: a mask that let each slot exchange messages only with its
neighbors in the factor graph. Its removal cost was zero. Take it
away, and nothing moves. At thirty-two rows, all-to-all is cheap and
the attention learns whatever sparsity it wants.

The herring's rule is a fact about water. A fish cannot see the far
side of the school, so local alignment is all it has, and the
murmuration is what falls out. There is no water in a thirty-two row
attention. The constraint that makes the shoal beautiful is the
constraint our machine does not have.

Our June engine was the true cousin of a graph network. On KenKen it
ran learned belief propagation with masks built from cage membership,
and there the masks were load-bearing; the engine scored zero without
them. The difference is where the structure comes from. KenKen hands
you the cages as input. In parsing, the structure is the output. The
head cannot route its messages along a graph it has not yet
discovered, which is the constitution we wrote in August after the
head tried to surprise itself: inference masks come from
solver-certified structure, never from the head's own beliefs.

So the one shoal we have is the steering wheel. Every breath the
parse is committed to the solver, and on a certified refusal the
minimal unsatisfiable core comes back as a spotlight on the words the
guilty slots were reading. That is message passing along structure,
and it turns only on proofs.

## Where the weight sharing already lives

The thing that makes a convolution powerful is not locality. It is
sharing one operator across a symmetry. A 3-by-3 filter is the same
filter at every position because the picture's statistics do not
care where you are standing. Ask what our symmetries are, and two
convolutions appear that were there all along.

The first is across breaths. The same organs run on every breath,
with the six-wave clock as the position code. That is a convolution
over time, with a phase instead of padding, and the stellarator's
helical cut is a kernel over that axis.

The second is across planes. The 512 dimensions are 256 complex
planes, and the polar block applies one shared operator to every
plane's radius and angle. That is exactly a 1-by-1 convolution with
two channels over 256 pixels. It is the right kernel size, because
kernel size one is the only size that does not invent an adjacency.

## Why the picture backfires

Reshape 512 numbers into a 16-by-32 image and a 3-by-3 filter will
treat plane 7, plane 8, and plane 39 as neighbors, because that is
where they landed when you unrolled the tensor. They are not
neighbors. The content planes are a set. There is no order among
them, and a filter that slides along them is learning the accidents
of a reshape.

The same objection applies in one dimension, and this is where I
part ways with the relayed advice, which proposed a 1D circular
convolution "along the frequency band" and a U-Net over the 256 polar
coordinates as a sequence. Both are the picture mistake with one
axis removed. The clock planes are the one band with a genuine order,
the harmonics, but there are six of them, and a convolution over six
elements is a small dense layer wearing a hat. The finding that six
clock planes carried 55% of the add-versus-multiply discriminant came
from a linear spectral read, which is the kind of projection the
heads already learn.

One more row of that advice pointed the wrong way: restrict the
middle breaths to k-hop messages. The middle breaths are the desert.
The idle read showed breaths three to five remove for free.
Restricting them further starves what is already starving.

## The axis that has pixels

Our failure is not classifying the silhouette. On synthetic the
machine reads 0.976. The failure is grounding on wild text, 0.25: the
linkage from slots down to words. And the words are the one axis in
the whole machine with real locality. Adjacent words mean adjacent
things. That is the picture.

Segmentation there has a precise meaning, the old one from natural
language: chunk the sentence into value mentions, variable mentions,
and relation phrases before the slots go looking. Right now the
grounding roads throw soft pointers from slots straight onto the raw
trunk states, one word at a time. A small one-dimensional convolution
over tokens, three to five words wide, is the convolutional organ
that fits, and it was already registered and unfired as the
token-side layer T1. The attention map itself, words by slots, is the
one object in the system that is honestly image-like, and a
convolution over its word axis is the same organ seen from the other
side.

The word is given. T1 enters as a mandatory road with no gain, per
the law we wrote when gated organs kept getting voted down from
birth. The bars are pinned on the register where the headroom lives:
wild has to clear the standing 0.2526 by a full point, synthetic must
hold its guard, and the severance ladder reads first, because a
cooker converts exactly the headroom of the bypass it severs and
nothing more.

## The line

A convolution is a bet that neighbors mean something. Before you
place it, ask which axis has neighbors. The dancer has a body, but
the planes are not her pixels. The sentence is.
