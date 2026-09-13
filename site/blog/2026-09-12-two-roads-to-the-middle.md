---
title: Two roads to the middle
date: 2026-09-12
---

# Two roads to the middle: paving the path, or removing the need for it

The loop breathes seven times. A gradient census this week measured
what each breath is taught by the one loss at the end, and found a
desert: breaths three, four and five receive between one and seven
percent of the readout's gradient. An idle read confirmed it from the
other side. Switch those three breaths off and the machine reads a
little better. The middle of the loop is decorative.

This post is about two ways to fix that, chosen so they can be judged
on the same instrument. One paves a road from the loss to every
breath. The other takes away the need for the road.

## The road: the shelf readout

The final answer is read from the state after the last breath. Every
earlier breath reaches the loss only by passing through the ones after
it, and under the stellarator's residual cut most of that passage is
closed on purpose. Karpathy's argument against recurrent networks is
exactly this: a long thin graph through time leaves most nodes many
hops from supervision, and the gradient starves on the way back.

The transformer's answer is not a loss at every step. It is an edge.
Attention lets every position reach the loss in one hop, so no node is
far from anything. Our version is the shelf readout: the answer is
read from an attention over all seven breath states, per slot, with
the final state as the query and a learned preference per breath. The
read chooses which breath supplied what, and every breath's state
sits two hops from the loss.

It enters as a road, not a gain. All readout traffic passes through
the mix and there is no bypass. It is born almost as the final state:
a logit preference on the last breath that the birth read sets. That
read happened this afternoon on the untrained organ, and it was a
surprise. The strongest blend we offered, with 23% of the read drawn
from history, cost one thousandth of a point on wild and two
ten-thousandths on synthetic. The final state is already a consensus
of its own past. The arm is training as this is written.

The bars are pinned on the census, not on accuracy. The middle
breaths must come to hold at least a fifth of the readout's credit.
The idle read must turn from free to costly. Wild and synthetic must
hold their guards. If the credit flows and idling the middle still
costs nothing, the desert was never a wiring problem.

## The other road: the denoising schedule

Bryce put the second idea as a question. If a subtle change at a
middle breath was crucial to the final answer but does not show up in
it, how would the gradient ever know?

It would not, and it should not. Gradient credit is first-order in the
final loss. A change the answer does not reflect earns nothing, and
that is the definition working correctly. The desert is the other
case: effects that would move the answer, attenuated on the way back.

But the question points at how diffusion models are trained, and that
is the real lesson. A diffusion model is not trained by sending
gradient back through a thousand denoising steps. Each step is
trained alone. Take a clean image, corrupt it to a known level by a
known process, and teach the network to undo one step from that
input. The intermediate images look nothing like the final one, and
it never matters, because no step is graded on the final image. Each
is graded on its own well-posed job. There is no long thin graph
because there is no graph.

We tried a loss on every breath in August and buried it. I now think
we asked the wrong per-breath question. We put the sharp final parse
as the target at breath two, when breath two had almost nothing to
work with. A hopeless target teaches noise. The diffusion form asks a
question each breath can answer: given this much of the picture,
recover it.

## Which fog

Discrete diffusion has two ways to corrupt a picture, and they differ
in exactly the way Bryce's gut flagged. The masking kernel hides
symbols and reveals them in groups: serial in slots, some finished,
the rest blank. The uniform kernel corrupts every symbol a little at
the same level, so every slot is a noisy version of the truth and they
all sharpen together. That is the image coming out of the fog, and it
is also what the machine does unprompted. The spectral read showed the
state's effective rank funneling from about thirty to a crest of ten
to twelve across the seven breaths, all slots at once. The gut chose
the uniform kernel, and the machine agrees.

Diffusion corrupts the input. We have no road from parse space back
into the state, so an input-side schedule would need a new organ
first. There is a cheaper form that keeps the parallel-sharpening
shape and needs nothing new. Corrupt the target. At each breath,
decode every slot as we already do for the solver's ping, and score
it against a blurred gold: a mixture of the true one-hot parse and
uniform, with a blur weight that falls breath by breath and reaches
zero at the last. Early breaths are asked to be uncertain in the right
direction, not to be right. Every breath's state gets a two-hop path
to a solvable target, in parallel across all slots. It is a change to
the loss and to nothing else.

The blur schedule has an obvious clock. The stellarator already closes
the residual on a cos² curve from one at the first breath to zero at
the last loop breath. Put the blur on the same curve and the seal and
the sharpening move together: as the bypass closes, the picture comes
into focus.

## One instrument

The two roads will be read on the same gauges: the census credit of
the middle breaths, the idle read, and the guards. If the shelf alone
wakes the middle, the harder change is not needed. If the credit
flows through the shelf and the middle still computes nothing worth
keeping, the desert is a capacity question rather than a wiring one,
and the schedule, which changes what the middle is asked to do, is the
next word.

A convolution is a bet that neighbors mean something. Attention over
time is a bet that the past is worth asking. A denoising schedule is a
bet that the middle can be graded on its own work. The first bet was
settled this week. The second is on the GPU tonight. The third is
written down, with its bars, waiting.
