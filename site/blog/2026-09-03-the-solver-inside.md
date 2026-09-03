title: The solver hiding inside (simplified)
date: 2026-09-03

# The puzzle solver hiding inside every language model

*The plain-language version. A companion post tells this story with
full technical detail.*

Here is a strange experiment we ran, and what we think it means.

We took a small language model — the kind that predicts the next
word in a sentence — froze its first few layers so they could never
learn anything new, and gave it Sudoku puzzles. One change made it
work: we told its attention where it was *allowed to look*. Each
puzzle cell could only pay attention to the cells it shares a row,
column, or box with — the puzzle's own rules, drawn as a map.

Then we let it think in cycles: look, update, look again, sixteen
times.

It solved the puzzles. All of them, on the cases it was sure about.
And the same frozen layers, with the same trick, went on to solve
map-coloring puzzles and arithmetic grids — puzzles it had never
seen, in formats it had never seen.

Remember: those frozen layers had only ever read text. Nobody taught
them to solve anything. The solving machinery was *already in
there* — built as a side effect of learning language — waiting for
someone to hand it a map of where to look. We didn't teach it to
solve. We taught it where to look, and the solving was already
latent inside.

## The heartbeat

A while back we also recorded how a model's attention moves while it
works, over a hundred sessions. Buried in those recordings was
something like a heartbeat: attention snapping between two modes —
zoomed IN, working one small neighborhood, and zoomed OUT, surveying
the whole problem — holding each mode for a while, then switching.
Like a telegraph key: dot, dash, dot. Nobody asked it to do that. We
suspect bigger models do the same thing, though we haven't verified
that yet, and we say so.

## What we think it means

When a language model reasons, we believe it is *improvising a
solver* inside itself: zoom in, work out consequences, zoom out,
pick the next spot. That's what the heartbeat is. But it's all
improvised — rebuilt from scratch every time, with nothing exact
anywhere, because until now a language model had nowhere else to
put a solver except inside its own weights.

Our project builds the thing the models have been improvising. The
language part does what it's genuinely good at: reading messy human
sentences. A real puzzle-solver — exact, checkable, incapable of
guessing — does the deduction. And the two take turns, in cycles,
just like the heartbeat: the reader commits what it's sure of, the
solver works out what follows, and what the solver learns changes
where the reader looks next.

The models were humming a tune from memory the whole time. We're
building them the instrument.
