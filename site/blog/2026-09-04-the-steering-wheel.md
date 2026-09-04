title: The steering wheel
date: 2026-09-04

# The steering wheel: dynamic attention and the one-line language of looking

An attention mask answers the simplest question in a reasoning
machine: **who may look at whom?** Every cycle of thought, every cell
of a problem attends to some neighbors and ignores the rest — and
that geometry of permission decides, before any weight fires, what
the machine can possibly conclude. If deliberation is the engine,
the mask is the steering wheel.

For most of this project the steering wheel was welded straight.
The mask was set once, from a first reading of the problem, and held.
Later we added a reflex — committed conclusions could crack open a
few extra lanes of attention — but a reflex is not a driver: it had
no learned parameters, no dedicated heads, no ability to weigh what
it saw. The machine could think for seven cycles but could not
meaningfully *re-decide where to look* as its understanding changed.
An audit this week made it official: dynamic masking was the most
under-resourced organ in the machine, billed as central, built as an
afterthought. That is now being fixed — a dedicated, multi-head
**mask head** whose whole job is steering.

But the deeper discovery came from excavating our own history.

Two years of hand-built puzzle solvers — Sudoku, map coloring,
KenKen, logic circuits — each needed hand-crafted masks: for Sudoku,
attention along rows for some heads, columns for others, boxes for
others. We wrote those rules by hand, domain by domain. Digging
through all of them this week, we found that every one is the same
rule wearing different clothes:

> **Allow attention between two cells IF they share a constraint
> factor of type t.**

That's it. Same row? A shared row-constraint. Same cage? A shared
cage-constraint. Connected by a wire? A shared gate. One line,
five domains, every mask we ever built. The rules we thought we were
inventing were instances of a single sentence about structure.

When one sentence covers everything, it deserves to become a
language. So the mask head is getting a **DSL** — a tiny formal
language of looking, whose entire vocabulary is that sentence and a
few honest atoms: *shares-a-group*, *committed-by-the-solver*,
*self*, *all*, union, intersection. The neural mask head *proposes*
a program; deterministic machinery *executes* it into an exact mask.
Neural proposes, symbolic disposes — the deepest law of this
project, now applied to attention itself. And the language carries
its safety in its grammar: there is no negation atom, so a program
can open lanes of attention but can never close the baseline ones.
The forbidden act isn't checked for. It is unwritable.

The test we have pinned for this system is our favorite in the whole
campaign. We will hand the machine a Sudoku puzzle described in plain
prose, and ask its mask head to steer: parse the rules, propose the
program, execute the mask. The pass bar is exact equality — bit for
bit — with the mask we wrote by hand two years ago, before any of
this existed. The machine, reading language, must re-derive its own
history.

A steering wheel, a one-sentence language of looking, and a driving
test graded by our younger selves. That is where dynamic attention
is headed.
