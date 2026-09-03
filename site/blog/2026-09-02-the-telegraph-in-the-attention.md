title: The telegraph in the attention
date: 2026-09-02

# The telegraph in the attention

Early in this project we did something unglamorous: we recorded the
attention patterns of our models while they worked — over a hundred
traces, saved and archived — and then went looking for structure in
how attention *moved* over time.

We found a telegraph.

Fit a hidden Markov model — a statistical tool that asks "does this
signal secretly hop between a small number of states?" — to the
attention-entropy traces, and the answer came back with unusual
clarity: in 93 of 104 traces, the best-fitting model had exactly
**two** hidden states, with a switching pattern stable at 0.97 and
real dwell times in each state. Not a smooth drift. Not noise. A
square wave: the attention snapping between two regimes and holding
each one for a while, like a telegraph key — dot, dash, dot.

Our best reading of the two states: a **local** regime (attention
concentrated, working a small neighborhood of the problem) and a
**global** regime (attention spread wide, surveying). The model was
alternating between focusing and surveying on its own, with nobody
having asked it to. That observation became one of the seeds of this
project's central design: if alternation is what attention does
naturally when it reasons, build the machine around the alternation —
explicit cycles of reading, refining, and committing — rather than
leaving it buried in a trace. (One measurement honesty-note, logged
in our records: entropy measures how *spread* attention is, not where
it sits — a switch between two equally-spread-but-different-places
regimes would be invisible to it. A sharper instrument for switching
is on our books as a registered follow-up.)

## The solver that was already in there

A second observation from the same era, on a small open model —
Pythia-410M. We took a few of its early layers, froze them, attached
a small trained head, and gave it structured attention masks that
told it *which cells of a puzzle may talk to which*. Then we asked it
to solve Sudoku.

It did — and the same frozen slice, with the same trained head,
transferred to graph coloring and KenKen arithmetic puzzles,
propagating constraints step by step like a solver. The pretrained
weights, which had only ever read text, already contained machinery
that — given the right connectivity — *implements deduction*. We
didn't teach it to solve. We taught it where to look, and the solving
was already latent in there.

Both observations point the same direction, and it is the direction
this whole project walks: language models are not blank approximators.
They carry latent structure — alternating attention regimes, latent
constraint-propagation machinery — and the shortest path to reliable
reasoning may not be to train ever-bigger models to imitate it, but
to *expose* the structure that is already there, give it explicit
scaffolding, and let exact machinery do the part that must never be
approximate.

The telegraph was tapping out a message. We think it was a design
document.
