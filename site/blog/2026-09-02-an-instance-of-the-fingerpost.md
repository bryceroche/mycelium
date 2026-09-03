title: An instance of the fingerpost
date: 2026-09-02

# An instance of the fingerpost

Iain Pears wrote a novel called *An Instance of the Fingerpost* that
tells a single story four separate times, through four narrators,
none of whom can be fully trusted. Each one has their own blind
spots, their own self-interest, their own honest confusion about
what actually happened. No single account settles the matter. The
title comes from Francis Bacon, who used the Latin phrase
*instantia crucis* — the crucial instance — for the one piece of
evidence that finally points a single direction when every other
signpost has been ambiguous. Bacon's idea wasn't that any one
witness becomes trustworthy. It's that the right *collection* of
witnesses, compared against each other, can settle what no one of
them could settle alone.

Our reasoning machine is built around exactly that idea, and it's
where an earlier fact about the system — that correct readings of a
problem all agree with each other, while wrong readings scatter each
in their own particular way — finally gets put to work. If agreement
is evidence of truth, then the way to get honest answers out of an
imperfect reader is not to trust the reader. It's to build several
imperfect readers whose mistakes don't overlap, and only speak when
they all land in the same place.

The wall has layers, each one a different kind of witness.

First, the same problem gets read **five times**, with the sentences
permuted — reordered, reshuffled — so that nothing about the
underlying arithmetic changes but the surface presentation does. If
the machine actually understood the problem, permuting harmless
detail shouldn't move the answer; if it was pattern-matching on
sentence order or position, the readings will disagree with each
other. All five readings have to land on the same diagram before this
stage passes.

Second, the problem is checked against **models trained
separately**, from different lineages and different widths —
different starting points, different training runs, not just
different random permutations of the same trained head. Two models
that came from genuinely different training histories are unlikely
to share the same blind spot, so if they land on the same graph
anyway, that agreement means something the first stage's agreement
alone couldn't guarantee. Many landscapes, one shape.

Third, behind both of those, an out-of-distribution check — we call
it the **mouth** — asks a different question entirely: not "did the
readers agree," but "does this problem even live in territory the
system has been trained to recognize at all." A problem can pass the
first two walls by accident if it happens to sit somewhere strange
enough that every reader makes the same unusual mistake; the mouth
exists to catch exactly that failure mode, by checking familiarity
rather than agreement.

Independent witnesses, independent failure modes. Wrong readings
scatter across all three checks in their own particular ways; only a
correct reading survives being interrogated from three unrelated
directions at once. Only unanimity crosses the wall — a fingerpost
built, the way Bacon meant it, out of testimony rather than
authority.

And when the wall doesn't pass — when the machine can't certify an
answer — it doesn't just fail silently. The symbolic solver at the
core of the system, the one doing the exact logical search, has a
particular kind of honesty built into its refusals. When a diagram
turns out to be contradictory — when the constraints it's been
handed genuinely cannot all be true at once — the solver doesn't
just report failure. It can hand back a **minimal unsatisfiable
core**: the smallest subset of those constraints that, on their own,
already cannot be satisfied together. Not "no." Not even "no,
because of these fifteen things." Just the smallest possible "no,
and here is exactly the contradiction" — the precise handful of
facts that don't fit, stripped of everything else that was
irrelevant to the failure.

That is the deeper point the fingerpost was always making. A single
unreliable witness can't be trusted on their own account, but a
collection of them, compared honestly, can point one direction with
real confidence — and even when the story falls apart entirely,
falling apart is not the same as going silent. Even failure, read
correctly, has a shape.
