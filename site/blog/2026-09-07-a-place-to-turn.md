title: A place to turn
date: 2026-09-07

# A place to turn: the T²⁵⁶ state and the attention space that rotates

We have written on this site that our reasoning machine breathes in
seven cycles, that a clock of six helical waves turns sixty degrees
with each breath, and that the machine therefore "always knows how
deep into its own thinking it is." This week we measured that
sentence, and it was not true.

The measurement is simple to describe. Take the machine's internal
state at each of its seven breaths, across a few hundred problems,
and ask a linear probe the plainest question there is: *which breath
is this?* If the clock were in the state, the answer would be
trivial. The probe scored 0.60 on our synthetic problems and 0.53 on
wild ones, against a bar of 0.95. The confusion matrix was more
eloquent than the score. Breath zero and breath one were identified
perfectly, every time. From breath two onward the machine lost
count: its state at breath three looked like breath two, or four,
or five. Then we fitted the best rotation between the state at one
breath and the next and read off the angles. A working clock would
put a peak at sixty degrees. Ours put ninety-eight percent of the
mass within twenty degrees of zero. The state does not turn. It
only grows: its norm climbs from about seven at the first breath to
about twelve at the last, monotone, on every problem.

## A rotation is invisible on a growing radius

That sentence is the whole finding, and it explains why the clock
we did build never registered. There was a rotor in the machine —
the last upgrade installed a sixty-degree turn per breath on the
queries of one attention organ. But it lived on eight of thirty-two
planes per head, on the query side only, behind gains that were born
at zero and woke to about two percent. A whisper, and a whisper
applied to attention rather than to the state. Meanwhile the state
itself was doing something that drowns any whisper: accumulating.
Each breath added to the residual and the vector got longer. A
rotation is a change of angle. On a vector whose length is the only
thing changing, angle is the one coordinate nobody is keeping.

One more number sharpened the picture. The loop's state is five
hundred and twelve numbers wide, but the top sixty-four principal
directions carry ninety-five percent of its variance. The room is
mostly empty. The machine has been thinking in a corner of a hall
it never learned to use, and growing in the one direction that
carries no information.

## The compass and the odometer

The fix is a change of coordinates, and we already had the law for
it. Months ago, while chasing why the machine's geometric monitors
aged, we wrote down a two-channel reading discipline: *angle is
identity, radius is consolidation.* It was a rule about how to read
the machine. The rebuild makes it the way the machine is built.

Each slot of the loop's state becomes two things instead of one. A
**direction** — a unit vector on a torus of two hundred and
fifty-six planes, the same T²⁵⁶ geometry our binding bus already
uses to carry role as phase. And a **radius** — an explicit,
non-negative number that is allowed to grow, and that means what
the growing norm always meant: how much has settled. The compass and
the odometer, separated. Every organ that used to read one vector
reads the product of the two and sees exactly what it saw before;
nothing downstream needs to know the coordinates changed. But the
clock finally has something to hold.

On that torus, the sextet is no longer a whisper behind a gain. The
master clock — a small module that has existed as a specification
since August and was never imported by anything — becomes the
single source of truth in fact. Its wheel table drives a frozen,
gainless rotation of designated clock planes of the direction, sixty
degrees per loop breath, with a parity wheel at twice the rate for
redundancy and a pass wheel held static until the machine deliberates
in more than one pass. Nothing about the rhythm is learned. The same
rotation, on the same planes, is applied to the queries of the
attention space, so that what the state carries and what attention
sees are turned by one clock. Breath zero stays outside time: it is
the raw reading, before the clock starts. The content planes stay
unclocked; they are for meaning.

And the deposit that commits facts into the machine's shelf — whose
amplitude we found this week to be three to four orders of magnitude
louder in the old machine than a trained amplitude wants to be, and
which under pressure collapsed to zero and detonated a training run —
gets a radius channel of its own, calibrated to the state's scale, so
it can neither shout nor vanish. Three fixes that looked separate in
the ledger turn out to be one geometry.

## What a rebuild has to promise

None of this is measured yet. It is registered, which in this house
means the bars are written before the run. This week's clock read
becomes the instrument: breath identity must decode at 0.95 or
better on both kinds of problem, and at least half of the rotation
mass between consecutive breaths must sit near sixty degrees. The
machine's accuracy on wild and synthetic problems must stay within
two standard deviations of a twin trained the old way from the same
starting point. The committed channel that opened this week — the
one that now carries a fifth of wild problems through the certified
path — must not close. And the kill is written too: if breath
identity still reads below 0.80 after the run, the coordinates are
wrong, and no amount of training will fix them.

With the flag off, the machine is byte-for-byte the machine it was;
the equivalence gate proves that before the first step. With the flag
on, it is a new generation, and gets measured as one.

## On deck

Once radius is a real coordinate, one more door is a single step
away. A space where distance from the center means depth — where
nested kinds sit inside each other the way a hierarchy wants and the
rim has infinite room for the specific — is the Poincaré ball, and we
wrote its rules of engagement in July before we had any use for
them: hyperbolic quantities never enter a softmax without a log-map,
and cosine is the wrong distance inside the ball. The contraction
this machine now does by machinery, the geometry would do by itself.
Not this generation. But the odometer is the door, and the door is
now on the hinge.

We named the site for a shape: a braid that tightens, winding fixed
by clocks, radius shrunk by evidence. This week we learned the
winding was never fixed, because the braid had no place to turn.
Now it does.

---

## Postscript, September 8: the machine keeps its clock

The twin ran. Two arms from the same starting point, twelve thousand
gentle steps each, judged against a plain continuation trained the
old way.

The clock survived training. After twelve thousand steps a linear
probe reads the breath off the machine's state at **1.0000** on both
synthetic and wild problems, where the old machine read 0.60 and
0.53. Three quarters of the breath hand's variance turns at sixty
degrees per breath, the parity wheel turns at one hundred twenty,
and the content planes turn at nothing. The learned write shaved the
rotation from 0.83 at birth to 0.72 and did not cancel it. The
machine knows its breath.

The first arm, the clock alone, paid for it: about five points of
accuracy on synthetic problems, and one on wild. The second arm added
the two ideas the compass-and-odometer picture had been waiting for,
a bottleneck on the content planes only, so the clock's planes are
never squeezed, seeded from the machine's own top directions, and a
small Maxwell-style exchange between neighbouring slots on the clock
planes, so phase travels along the lanes attention allows. That arm
keeps the clock at the same strength and gives the accuracy back
entirely: **0.2511 on wild and 0.9556 on synthetic**, level with the
plain continuation to within a fraction of a point on both. A
machine that knows its breath, at no measured cost against one that
does not.

One bar failed as written, and we say so. The control clause we
pinned for the content planes asked that at least ninety percent of
their motion sit near zero degrees, to guard against the plane
selection manufacturing a fake sixty. After training they sit near
eighty percent, on both arms. The artefact the clause guards against
is absent, content shows two to four percent at sixty against
seventy-two on the breath hand, but the clause as pinned failed, and
bars do not bend after the fact. It is recorded as a fail, and the
better wording is registered for the next twin, not applied to this
one.

Single seed. The second seed, and the question of which half of the
sink pays, are the next fires. The braid has a place to turn now,
and it turned.

*Second seed, September 8: the clock replicates, breath identity at
1.0 and the breath hand at 0.70 to 0.76 on both fixtures, and the
synthetic accuracy holds within a point of its control. Wild does not
fully replicate: 0.2409 against a control of 0.2686 on this seed,
where the first seed had matched its control. Across the pair the
machine costs about one point of wild on average and up to three on
one seed. "At no measured cost" above is therefore too strong; the
honest phrase is at no measured cost on synthetic problems, and a
small, seed-dependent cost on wild, now registered as the
generation's open debt.*
