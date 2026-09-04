title: The herring shoal
date: 2026-09-03

# The herring shoal: message passing without a messenger

A herring shoal turns as one. A predator lunges at the edge, and
within a moment the whole school — thousands of fish, spread across
water no single fish can see — has wheeled away. No fish is in
charge. No signal crosses the shoal. Each herring watches only its
nearest neighbors and reacts to them; the reaction propagates
neighbor to neighbor as a wave, and the wave moves through the
school faster than any individual fish can swim.

That is message passing, and it is the deepest computational pattern
in our reasoning machine.

A factor graph — our diagram of quantities and the arithmetic
linking them — is a shoal. Each variable knows only its immediate
neighbors: the constraints it appears in. When one variable's value
becomes known, the news propagates: every constraint touching it
tightens, which pins other variables, which tighten further
constraints. No global coordinator ever sees the whole problem. Just
local rules, applied everywhere at once, and a wave of implication
sweeping the graph.

Our early experiments measured the herring property directly. When
we ran deduction as breathing cycles — every part of the graph
updating in parallel each cycle — solving depth scaled the way a
shoal's wave does: a problem requiring D sequential steps of
reasoning resolved in roughly D/4 breaths, because the wave
propagates from every known value simultaneously, meeting in the
middle. Deduction is not a chain. It is a flood.

The same pattern lives at every scale of the machine. Inside a
breath, slots attend to slots — each committed fact is a fish that
flinched, and its neighbors react. Inside the solver, constraint
propagation is the wave run with exact arithmetic. And between the
two, the alternation: the neural side commits a few facts, the
symbolic side floods their consequences through the graph, and the
returning wave changes what the neural side looks at next.

## What a message actually needs

Watch the shoal closely enough and you can factor out what any
message passing system requires. Three things:

1. **A channel** — a medium the signal travels through. The herring
   have water and the lateral line; our slots have attention edges;
   the solver has the port.
2. **A language** — a form both ends understand. The herring's
   language is motion itself; ours is typed structure: factor roles,
   committed edges, phase-bound wires. A channel without a shared
   language carries noise, not news.
3. **A time interval for the handoff** — and this one is the
   sleeper. Sender and receiver almost never act at the same
   instant. A message must *survive* between being sent and being
   read.

That third requirement may be the quiet reason attention works as
well as it does. Attention is not a phone call — it is a **parking
garage**. A fact gets written into the residual stream or a notebook
slot and simply *parked there*, addressed by its content, until some
later cycle's query comes looking for exactly that shape of thing
and picks it up. Sender and receiver never need to meet. We learned
this as a design law the hard way: dropping a passenger at the
airport is easy, timing a pickup is hard — so build systems around
drop-offs. Every organ in our machine that works well works this
way: the notebook parks facts across breaths, the solver's port
parks derived values for the next cycle, the committed structure
parks itself in the masks. The wave through the shoal is fast
precisely because no fish waits for a reply.

There is a lesson in the herring for anyone building reasoning
systems: you do not need a general to move an army. You need good
local rules, honest neighbors, a medium the wave can travel through
— and a garage where messages wait, content-addressed, for whoever
comes needing them. Most of our architecture is just that medium and
that garage, kept clear.
