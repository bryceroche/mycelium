title: The ping-pong
date: 2026-09-02

# The ping-pong

Our reasoning machine reads math word problems with two different
parts working together: a neural front jaw that grips messy English
prose into structure, and a symbolic throat jaw that crushes that
structure into an exact answer, by deterministic logical search
rather than guessing. Ordinarily you'd picture these two parts
working in sequence — read the whole problem, hand the whole
diagram to the solver, get an answer back. That's how the system
behaves at the end.

But during the machine's seven cycles of deliberation, before it has
committed to anything final, we built the two jaws to talk to each
other mid-thought, not just at the finish line. We call this the
**ping-pong**.

Here's how a cycle goes. The neural jaw looks at the sentence and
its current guess at the structure, and commits to whatever piece of
it it's genuinely confident about — maybe it's now sure that a
particular number is a total rather than a rate, even though it
hasn't yet worked out where every other number goes. That committed
fragment gets handed to the symbolic jaw immediately, mid-cycle,
rather than waiting for the whole diagram to be finished. The
symbolic jaw then does something cheap and exact with it: it
propagates the consequences. This is the same move a sudoku player
makes when they fill in one square and then look around the board
for any other square that is now *forced* — a square where, given
what was just committed, only one number could possibly go. The
symbolic jaw does this for the emerging math diagram: given what the
neural jaw just committed to, what else is now logically forced?

Whatever gets derived this way — the forced consequences, not
guesses — is handed back to the neural jaw as established fact for
the next cycle. And it does something more than just add information
to a list. The structure that's now been committed reshapes the
neural jaw's own attention: which parts of the sentence it's allowed
to look at next, and how closely, shifts based on what's already
settled. We call this **dynamic attention masking** — the parse's
own developing skeleton deciding what the next breath is permitted
to focus on. A number that's already been pinned down doesn't need
the same scrutiny it needed before it was pinned down; the machine's
attention moves on to what's still unresolved.

Neural proposes, symbolic disposes, neural re-attends. Ping, pong,
ping again — a real conversation between the two jaws happening
*during* the read, not just a relay handoff at the end of it.

We want to report where this actually stands, honestly, because it
would be easy to describe a mechanism this elaborate and let the
elaborateness imply a result. The machinery is built. It is
verified — we can watch it commit fragments, propagate consequences,
and reshape attention exactly as designed. And on the synthetic
training data we built the system on, it changes essentially
nothing. The reason turned out to be simple once we found it:
synthetic problems, generated the way ours were, tend to state
everything needed up front, in a form that doesn't require much
mid-read deduction to unlock. There's no ladder of consequence for
the symbolic jaw to climb — nothing gets forced, because nothing was
withheld. The sudoku board arrives mostly filled in.

Real textbook prose is not like that. It states things indirectly,
across sentences, in an order that often withholds the easy
inference until later — exactly the kind of writing that *does*
build a ladder of consequence a solver can climb rung by rung as the
read proceeds. Whether the ping-pong earns its keep on that kind of
prose — whether mid-thought deduction actually helps once the text
stops being so obliging — is not yet a settled question. It's the
campaign underway as this is written: teaching the system to read
wild, real prose, and measuring, honestly, whether the machinery we
built for exactly this moment finally has something to do.

Built, verified, and waiting for the kind of sentence that needs it.
That's not a failure to report. It's a bet we're still watching
resolve.
