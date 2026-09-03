title: Lucy's notebook
date: 2026-09-02

# Lucy's notebook

In *50 First Dates*, Lucy wakes up every morning with no memory of
the day before. Nothing accumulates in her head overnight — every
sunrise starts her over. What lets her function at all is a
notebook, kept by her bedside, that someone re-reads to her each
morning: here is what has happened, here is who you are today, here
is the story so far. The memory doesn't live in Lucy. It lives in
the notebook.

Our reasoning machine has a Lucy at its core, and it is the same
character on purpose.

At the base of the system sits a large frozen language model — a
network with billions of internal numbers, trained once, long
before this project began, on a huge amount of ordinary text. We
never train it further. Not once. Every time our machine reads a
math word problem, it hands the sentence to this frozen model and
reads out its internal reaction to it — and then that reaction is
gone. The frozen model itself learns nothing from the encounter. The
next problem, it wakes with exactly the same weights it had before,
no memory of the last ten thousand problems it helped read, no
accumulated wisdom about this particular project at all. It is read
once per problem, the way Lucy is re-introduced once per day.

This sounds like it should cripple the system. If the part that
actually understands English never learns anything specific to the
task, where does the task-specific intelligence live?

It lives in a much smaller trained head sitting on top of the frozen
model — a few million adjustable numbers instead of billions, all of
them free to change as the system trains, none of them touching the
frozen giant underneath. All of our accumulated understanding of
what math word problems look like, everything the system has
learned across every problem it has ever been trained on, is stored
in this small head. It is the equivalent of Lucy's notebook: the one
part of the system permitted to remember anything from one problem
to the next, and it does the remembering explicitly, in an object
built for the purpose, not by quietly reshaping a billion frozen
numbers that were never meant to hold it.

There is a second, smaller instance of the same idea living inside a
*single* problem. Our machine doesn't read a sentence and answer
immediately — it deliberates, cycling through several passes of
attention before it commits to a diagram. Across those cycles, it
needs somewhere to put the things it has already figured out, so it
doesn't have to re-derive them on every pass. That somewhere is an
explicit **notebook**: a set of named slots the trained head writes
to and reads from as it works. When the head becomes confident that
a particular quantity plays a particular role, it writes that down
in a slot. On the next cycle, it reads the slot back rather than
re-deciding from scratch. Facts that get written to the notebook
persist; everything the head only half-considered evaporates when
the cycle ends, the way a thought you didn't bother to write down is
gone by lunchtime.

This is a deliberate stance against a very common alternative. Many
systems keep "memory" as a diffuse pattern smeared across a huge
number of weights or a long running context, with no single place
you could point to and say: this is the fact, this is where it
lives, this is what it means. Our notebook slots are named. They can
be printed out. If the system commits to the belief that a certain
number represents a rate rather than a total, that belief exists in
a specific slot, at a specific point in the deliberation, and you
can look at it. Memory that you can read is memory you can audit —
and audit, for a system whose entire second half is a wall of
witnesses checking each other's work, is not a nice-to-have. It is
the whole point.

Lucy never gets to keep her memories. But she gets to keep her
notebook, and it turns out that's enough to build a life on. Our
machine never gets to keep its frozen model's memories either. It
keeps a notebook instead, and it turns out that's enough to build a
mind on.
