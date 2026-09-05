title: The drafting problem
date: 2026-09-05

# The drafting problem: what a roguelike knows about attention

There is a video game called *Slay the Spire*. You climb a tower,
fight monsters, and after each fight you draft: pick one card from a
few offered, add it to your deck. Sometimes you gain a relic — a
permanent modifier that quietly changes how everything else plays.
The game has made millions of people fluent in a deep idea that most
of machine learning hasn't fully absorbed:

**There are no good cards. There are only good cards for this
deck.**

A card that wins one run is dead weight in another. Value is not a
property of the card; it is a property of the card *given*
everything you've already committed to. Every drafting decision must
consider the whole state — deck, relics, character, what's coming —
at once. Drafting is conditional valuation, and conditional
valuation is the skill.

## The mask head is the player

Our reasoning machine deliberates in cycles, and each cycle it must
decide **where attention may look next** — which parts of the
problem talk to which. For most of the project this was decided by a
reflex: a fixed rule, applied unconditionally. Same constraint? Open
the lane. Always.

That is drafting by card ranking — take the "best" card every time,
ignore your deck. Every Spire player knows how that run ends.

The organ we are building now — a dedicated, multi-headed attention
controller we call the mask head — exists to draft properly. Its job
is value-given-state: *this* attention lane, given *these*
commitments, *this* family of problem, *this* feedback from the
solver. It is the only component that sees the whole board at once,
which is why we resource it like a player and not like a subsystem:
its own attention heads, its own working memory, and the richest
inputs in the machine.

## The mapping, piece by piece

- **The deck** is the parse state: everything committed so far.
  Every commitment re-prices every future option.
- **Relics are the solver's facts.** A relic is a permanent modifier
  acquired mid-run that silently re-values every later choice.
  When our exact solver derives that some variable equals twelve,
  that fact sits on the run forever after, re-pricing every edge —
  a relic the neural side didn't choose but must play around.
- **Character classes are the atlas families.** The Silent drafts
  toward poison, shivs, or discard; the Ironclad toward strength,
  block, or exhaust. Our machine's problems draft toward families
  too — the operation-kinds in its atlas — and here is the subtle
  part: early in a run, all families are open and cards are valued
  flexibly; a few picks later the run has *committed* to an
  archetype, and every valuation tilts. Our atlas mirrors this
  exactly — it keeps seven versions of every family's centroid, one
  per deliberation cycle, because a family's meaning at breath one
  (everything still possible) is not its meaning at breath six
  (committed, converging). The archetype's power curve, floor by
  floor.
- **The death screen is the minimal unsatisfiable core.** A
  roguelike doesn't just kill you; it tells you which fight did it.
  When our solver refuses, it can return the smallest set of
  commitments that contradict each other — not "the run failed" but
  "these three picks could never coexist." Failure with a readable
  shape, so the next run drafts differently.

## The thinning problem

One more thing expert players know: removal is power. Thinning the
deck — paying to *delete* weak cards — often beats adding strong
ones. This is the one place our machine negotiates with the
metaphor, because our masks are open-only by constitutional law: the
mask head may open lanes of attention but can never close the
baseline ones. (The law exists because a learned mask that can
silence its own inputs learns to hide its mistakes.) The resolution
is quieter than deletion: attention renormalizes. Amplifying the
right lanes *is* muting the wrong ones. The mask head thins the deck
the only way the constitution allows — not by burning cards, but by
never drawing them.

## Why this is the most important seat in the machine

Two players with identical card pools finish wildly different runs.
The cards don't play the game; the *integration* does — the single
point where deck, relics, class, and the fight in front of you
become one decision. In our machine that seat belongs to the mask
head: it reads the parse, the facts, the family, and the frozen
model's interpretation of the problem, and it turns all of it into
the next cycle's geometry of attention. Steering, in the end, is
just drafting continuously.

We spent months treating that seat as a reflex. The game knew
better all along.
