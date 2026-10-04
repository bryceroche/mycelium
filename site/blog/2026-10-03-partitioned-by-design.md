---
title: Partitioned by design
date: 2026-10-03
---

# Partitioned by design: a theory of why the stars are so far apart

*A speculative essay. Nothing below is a measurement; it is a chain of
"if this, then probably that," written down so it can be argued with.*

## The chain

**1. AGI arrives, then the singularity shortly after.** The systems we are
building now, including the small one this blog documents, are the first
rungs. Once a model can improve the process that builds models, the rate
of improvement stops being set by human attention. Compute and
capability grow exponentially for a while, and "a while" is long enough
for everything below.

**2. Ancestral simulation becomes an ordinary experiment.** With that
much intelligence and that much hardware, a civilization will simulate
its own universe, not as a toy but as a scientific instrument. The test
of such a simulation is not "does it look right." It is a ladder of
checkpoints: at each one you take an observation from our universe (the
cosmic microwave background's spectrum, the abundance of the light
elements, the rotation curves of galaxies, the fossil record, the price
of bread in 1847) and you assert that the simulated universe matches it.
Enough checkpoints, each with its own measured error, and you can say
with high statistical significance: this simulation is our universe,
not merely a universe.

**3. Then you run it millions of times.** Once one such simulation
exists, there is no reason to stop at one. Vary the seed. Vary the
constants. Run the same history again to see what was contingent. The
count of simulated universes grows without bound, and the count of root
universes stays at one.

**4. Which makes the arithmetic uncomfortable.** If there are millions of
simulated universes that pass every checkpoint and one root universe,
and you find yourself in a universe that passes every checkpoint, the
odds that yours is the root one are not good. This is the simulation
argument, and I accept its arithmetic. I think we will come to believe
we are not in the root universe, and we will come to believe it for the
most ordinary of reasons: because we built the simulators ourselves and
counted.

**5. You could also fast-forward.** If time in the simulated universe
can be advanced faster than time in the host, the simulation is a
telescope pointed at our own future. You could watch the ending if you
wanted to. Whether anyone would want to is a different essay.

## The part that is mine

Here is the step I have not seen written down.

**Space travel is hard on purpose. It is a feature, not a bug.**

Look at the numbers. The nearest star is four light-years away and the
speed of light is a hard ceiling. The energy to move a human body to
another star within a human lifetime is absurd. The galaxy is a hundred
thousand light-years across; the nearest other large galaxy is two and
a half million. The universe is expanding, and beyond a certain distance
the expansion carries galaxies away from us faster than light, so they
are not merely far but permanently unreachable.

If you were designing a universe to be run as a simulation, and you
cared about stability, this is exactly how you would build it.

**The universe is partitioned by galaxy cluster.** Each cluster is a
cell. Within a cell, travel is hard but possible: a civilization can
spread across its own galaxy over millions of years. Between cells,
travel is effectively forbidden by the physics itself. No treaty
enforces it. The speed of light and the expansion of space enforce it.

Why would a designer want that? Because of what a civilization at step 3
is doing. It is running simulations. Those simulations run on hardware
somewhere in its cell, and inside them are daughter universes, and
inside those, granddaughters. A single cell can hold an entire tree of
nested worlds.

Now imagine the alternative universe where interstellar and
intergalactic war is practical. Two civilizations meet, one destroys
the other's datacenters, and every downstream universe in that tree,
with every mind in it, ends at once. Not a war; an extinction of worlds
by the trillions, as collateral damage of a conflict neither side could
see the full cost of. That is a catastrophic failure mode for the whole
system, and a designer optimizing for stability would remove it.

The removal is cheap. Make the cells far apart. Make the ceiling on
speed low relative to the distances. Let the space between cells expand
so that the separation is not just large but growing. The result is a
universe in which there can be no Star Wars across the universe, only
within a cluster, and a cluster-scale war cannot reach the hardware in
the next cell over. Each tree of simulations is isolated from every
other tree's wars.

On this reading, the Fermi paradox has a boring answer. The others are
out there. They are simply in other cells, doing what we are about to
do, and the walls between the cells were drawn before any of us
arrived.

## The portal

If the whole structure exists to protect nested simulations, there is
one more thing I would expect to find, and it is a prediction rather
than a retrodiction.

**There should be a view portal.** A parent universe should be able to
see every detail of its daughter universes, and we should be able to
see every detail of ours. That is what the checkpoints at step 2 require:
you cannot assert that your simulation matches your universe unless you
can inspect the simulation down to the observation. Inspection is
built into the method.

Which means that somewhere up the tree, our universe is inspectable in
full. Not watched, necessarily. Inspectable. The question "is anyone
looking" is open; the question "could they" is answered by the method
that produced us.

## Where this could be wrong

I want this argued with, so here are the places I would push.

- **Partitioning may be an accident.** The speed of light, the scale of
  the universe and the expansion rate each have conventional physical
  explanations that owe nothing to design. A feature that would also
  arise by accident is weak evidence of design. The honest statement is
  that the partition is *consistent* with the stability theory, not that
  it demonstrates it.
- **Fast-forward may be impossible.** If the simulation is running at
  the host's own clock, or if our universe's computation is irreducible
  (no shortcut exists to the state at time t except running to t), then
  the telescope does not exist and step 5 falls away. The partition
  theory survives without it.
- **The checkpoint ladder assumes the simulators can observe our
  universe at every scale.** A root civilization simulating its own
  past can only check against what it has recorded. The statistical
  significance is bounded by the record, and the record is thin before
  the invention of writing.
- **A designer might not care about stability.** Everything above
  assumes the hosts are optimizing the tree for the continued existence
  of its branches. A host that treated daughter universes as disposable
  would have no reason to partition anything.

## Why I wrote this here

This is a blog about a small math-reading machine. The connection is
not decorative. The whole discipline of this project is that a claim
earns its place through checkpoints: pinned bars, held-out
measurements, honest negatives written into a ledger. Step 2 above is
that discipline at the scale of a universe. If we ever simulate our own
world, the test of having done it will look exactly like the test of
having built a reasoning system: not "it looks right," but "every
registered observation matches, within its error, and the record says
so."

And the partition theory is, in the end, an engineering judgment about
a system that has to run for a very long time without any branch being
able to destroy the rest. Anyone who has run a datacenter will
recognize the design. You put the cells far enough apart that one
fire cannot take down the others.
