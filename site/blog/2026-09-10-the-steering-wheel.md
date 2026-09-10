---
title: The steering wheel
date: 2026-09-10
---

# The steering wheel: true alternation, and the solver as the driver

In June we built a small breathing deducer out of four frozen Pythia
layers and a set of hand-written attention masks, and it solved
Sudoku. Take the masks away and it scored zero. Not worse. Zero. The
masks were not a regularizer or a hint. They were the rules of the
game written as "which cell may look at which," and the model with
the rules was a solver while the model without them was noise.

That fact has been sitting in our ledger for three months, and this
week it turned out to be the blueprint for the thing the machine has
been missing. Here is the assembly.

## Why the slot mask felt free

Our current machine reads a word problem seven times and revises a
bank of slots between readings. It has a slot mask, built from the
machine's own first parse by the same rule the Sudoku masks used:
allow attention where a constraint is shared. This week's severance
ladder found that mask costs nothing to remove. All-to-all among the
slots reads as well as masked.

On its face that contradicts June. It does not. A word problem has
five factors. Dense attention over five items loses nothing. A Sudoku
has eighty-one cells and twenty-seven constraints, and dense attention
over eighty-one items is total noise. The mask is load-bearing in
proportion to how sparse the constraint graph is relative to the
sequence. Word problems are the small end of that law, and we had
been testing a steering mechanism in the one domain where the road
is short enough not to need one.

## The parts we already had

The wheel assembles from three things built earlier and one thing
that was never built.

**True alternation is built.** In early September we factored the
breath loop so that each breath is a separately dispatched step, with
a solver ping between every one: commit the tentative parse, run the
symbolic propagation, hand the facts back as constants, take the next
breath. Forward and backward were proven against the fused version on
the real champion, every parameter's gradient in tolerance. We called
it the final boss and then set it aside when a different lineage took
the lead. It is not a design question anymore. It is a switch.

**But the feedback was wired to the wrong place.** The same build
recorded a discovery we underweighted at the time. The solver's facts
inject into the variable-slot state, and only the readout heads read
that state. The breath loop itself never sees them. So every solver
ping sharpened the answer and steered nothing. The census this week
found the return roads whispering at two percent; this is why. The
solver had been talking to the mouth, not the hands.

**The wheel has a language.** Also in September we wrote a tiny
deterministic language of masking rules, a handful of atoms like
"same group" and "committed," in which every hand-built mask we had
ever written, for Sudoku, KenKen, coloring, and circuits, is five
lines or fewer. The neural side selects a program; symbolic machinery
executes it. Tightening is unrepresentable by grammar, so the language
can only open lanes. And one test was pinned: present Sudoku in prose,
parse it, emit the program, execute it, and the mask must equal the
hand-built one exactly. The machine re-deriving from language the
rules we once typed by hand. That is the bar you asked for this
morning, and it has been pinned for a week.

**The driver was never built.** What the solver says back today is
facts, and silence when it refuses. What it can say is a proof: on a
refusal, the smallest set of factors that cannot coexist, the minimal
unsatisfiable core. Two or three slots, named. Last night's collider
read showed why this matters: disagreement between two readings marked
wrong slots at 78% and right slots at 72%, a front everywhere and
therefore nowhere. A core is not a front. It is a theorem about which
slots are wrong together.

## The assembly

On every breath, not on a chosen few:

1. Commit the tentative parse to the solver.
2. The solver returns facts, or a refusal with its core.
3. The core, the parse, and the commitments are inputs to a mask
   program. The executor produces next breath's mask: open lanes among
   the core's slots, re-focus their token reads on the sentences they
   came from, melt their deposits.
4. The mask enters the loop, in the lanes the loop actually reads, and
   the next breath happens under it.

The mask trajectory we could not supervise, and closed as
unsupervisable this week, becomes the trajectory the solver dictates.
Nobody writes a schedule. The rules write the first mask and the
refusals write the rest.

The two token-side layers we gave the word for last night are the
linkage. A trained block over the words feeds the wheel only when its
mask is derived from the parse and the core; free-learned attention
over the words is exactly the unmasked Pythia that scored zero. And a
station that updates words from slots is what lets a core change what
a word means on the next reading. Linkage and tires. The core is the
driver.

## Why KenKen first

On word problems the wheel turns only when the solver refuses, so its
ceiling is the refusal rate on wrong parses, and a wrong parse is
usually a coherent parse of some other problem. We will measure that
rate before building anything on it.

On KenKen the mask is the computation. There is no bypass, no
residual road for the loss to escape through, no cooker to design.
This week's law says a cooker only converts headroom when the seal
leaves the residual no way to survive; on a puzzle the domain enforces
that for free. The cages differ on every board, so the mask must be
generated from the prose. The oracle membership is in the data, forty
thousand rows. The parser that produces it has never been trained.

## The bars

Three, pinned before anything runs. On wild problems, the solver must
refuse at least fifteen percent of wrong parses with a median core of
three factors or fewer, or the wheel has nothing to steer with there
and lives on puzzles. The Sudoku recreation must be exact. And on
KenKen, parsed membership must match the oracle on nine boards in ten,
with the June engine solving at its oracle rate on the generated
masks. If the first and the third both fail, the wheel has no domain,
and that goes in the ledger like everything else.

The core is thirty lines in a solver that has no domain code in it.
That is where the build starts.
