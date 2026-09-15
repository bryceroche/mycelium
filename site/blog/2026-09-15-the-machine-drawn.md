---
title: The machine, drawn
date: 2026-09-15
---

# The machine, drawn: the whole system from one config file

<a href="/figures/architecture.png"><img src="/figures/architecture.png" alt="Mycelium architecture: frozen trunk, the bridge, the NL loop, the shared air, the math loop, readout and verification" style="width:100%"></a>

*Click for full size. [SVG version](/figures/architecture.svg).*

This is the composed system as of today, drawn by a script from a
config file rather than by hand, so it can be redrawn whenever an organ
is added, retired, or renamed. Every parameter count on the figure is
counted from the checkpoint by matching the organ's weight names, not
typed: the trained head is 12.4M parameters in the research lineage,
and the frozen trunk is 506M, of which 263M is the embedding and 243M
is the four Llama layers we use of sixteen. Dashed boxes are symbolic
organs with no parameters at all, which is most of the verification and
all of the certifiers, the atlas, the bridge, the lexicon, and the
perceiver.

## How to read it

Left to right is the road a problem takes. The frozen trunk turns text
into 2048-dimensional token features once, precomputed to disk. The
bridge has two lanes: the waist carries the reading down to 512
dimensions where the slots live, and the return lane is the attention
matrix itself, which carries certificates back and forth as masks. The
two loops sit side by side. The language loop breathes on tokens, the
math loop on slots, and both breathe the same air: the six-wave clock,
the rotation bus, the atlas with its two charts, and the perceiver that
watches them. The math loop is the larger machine, with the bank
attention, the slot mixer, the per-breath feed-forward, the notebook
and garage, the mask head, the alternator's facts pass, the polar sink,
and the stellarator's residual snip. At the right, the emission heads
read the final state into a factor graph, and the symbolic side
disposes: the solver, the wheel's unsat core, the mouth, the vote, and
the answer key.

## Rebuilding it

The description lives in `docs/architecture.json`: panels, organs with
a one-line detail and a rule for which weights belong to them, and the
edges. The tool is `scripts/arch_diagram.py`. It loads the checkpoint,
assigns every weight to exactly one organ, reports any weight it could
not place so the config can be completed, and writes the PNG and SVG.
When the next organ passes its bars, it gets a box and a regex, and the
figure is one command away.
