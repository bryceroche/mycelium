title: The diffusion compiler
date: 2026-09-02

# The diffusion compiler

Chris Lattner — the engineer behind LLVM and MLIR, the infrastructure
most of the world's code compiles through — teaches a rule that
sounds almost moral: **"Premature lowering is the root of all evil."**

A compiler never jumps from source code to machine code in one leap.
It descends a ladder of intermediate representations, each one
preserving exactly the structure the next stage needs. Lower too
early — commit to registers while you still needed to reason about
loops — and you destroy information you cannot recover. Every
seasoned compiler engineer has the scars.

Our machine is a compiler for language. English prose lowers to a
silhouette of structure; the silhouette lowers to a typed factor
graph — a diagram of quantities and the arithmetic linking them;
recurring patterns lower to named abstractions; and at the moment of
verification everything lowers to primitives an exact solver can
check. Every rung is machine-checkable. No rung commits to detail the
next rung didn't ask for.

But there is one place we break with the compiler tradition — and it
is the most important design decision in the project.

A language model that writes out its reasoning as text is forced to
lower **left to right**, one token at a time, each word a permanent
commitment made before the sentence is finished. That is premature
lowering as a way of life. It is the reason chain-of-thought
reasoning is brittle: the first wrong token poisons everything
downstream, and there is no going back.

Our head lowers **in parallel**. The whole graph descends through the
representations together — every slot, every binding, every value
refined a little on each cycle of deliberation, the way an image
emerges from noise in a diffusion model: first the composition, then
the shapes, then the edges, all at once, everywhere. Nothing is
forced to commit first. Constraint flows in every direction while the
parts settle jointly, and the earliest commitments are the ones the
evidence forces, not the ones that happen to come first in reading
order.

A compiler's ladder, climbed the way an image is denoised. Lattner's
law, plus one amendment the compiler world never needed: when you do
lower, lower everything together.
