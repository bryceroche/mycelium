title: Happy families
date: 2026-09-02

# Happy families

Tolstoy opens *Anna Karenina* with a line so famous it has worn
smooth: happy families are all alike; every unhappy family is
unhappy in its own way. It sounds like a remark about marriages. For
us it turned out to be the deepest fact we know about machine
reasoning.

Our machine reads math word problems written in ordinary English and
turns them into a diagram — a small graph of quantities and the
arithmetic that connects them, the kind of thing you'd sketch on
scratch paper before actually solving anything. When it reads a
problem *correctly*, the diagram it produces is always the same
object, no matter how the sentence was worded. There is exactly one
correct diagram for "Maria has three times as many apples as Ben,"
and there is exactly one correct diagram for "the reservoir holds
triple what the tank does," and — because both sentences describe
the identical underlying arithmetic — those two diagrams are the
same diagram. Every correct reading is *alike*.

But when the machine misreads a problem, the ways it can go wrong
are not so cooperative. One misreading swaps which quantity is three
times which. Another invents a quantity that was never mentioned.
Another wires a number to the wrong role in the arithmetic — treats
a rate as a total, say. Each of these failures is broken in its own
particular way, and there is no small number of ways to be broken.
Wrongness has infinite variety. Rightness has exactly one shape.

This is not a cute observation to open an essay with. It is the
engine of the entire safety design.

Here is the move it enables. Suppose you don't trust any single
reading of a problem — you shouldn't, since any one reading might be
one of the infinite ways to be wrong. So instead of reading it once,
read it several times, independently, in ways that shouldn't matter
to the answer: reorder the sentences, shuffle which details come
first, change nothing about the arithmetic itself. If the readings
are genuinely independent — if they don't share a blind spot — then
their *wrong* answers will scatter. A misreading caused by getting
confused about sentence order in one pass has no reason to produce
the same wrong diagram as a misreading caused by mixing up two
quantities in another pass. Each broken reading is unhappy in its
own way, so different broken readings land in different places.

But the *right* answers don't scatter. They can't. There is only one
happy family — one correct diagram — for a given problem, so every
reading that happens to get it right lands in exactly the same spot.
Agreement across independent readings is therefore not just a vote
of confidence. It is close to a proof. If five readings of the same
problem, shuffled five different ways, all converge on the identical
diagram, the chance that five *different* mistakes all happened to
collide by accident is small — much smaller than the chance that
they converged because they were all correct.

This is why our machine doesn't answer a problem, on the strength of
a single pass, and hope. It re-reads. It re-reads with the sentences
permuted, and only certifies an answer when independent readings
land in the same place — a whole wall of witnesses built out of this
one asymmetry, layered deeper still with readers trained separately,
from different lineages, so that even a shared blind spot in one
lineage doesn't get mistaken for agreement. When the readings
scatter, the machine has learned something too: not the answer, but
the fact that this problem is not yet safe to answer. It stays
silent, and the silence itself is information, earned the same way
the confidence would have been — by counting who agrees with whom.

None of this works if wrongness and rightness are symmetric — if
mistakes cluster the way correct answers do. We built the entire
certification wall on a bet that they don't, that Tolstoy had
mathematics right by accident. So far, the bet holds: it is far
easier to be wrong in your own particular way than to be right in
someone else's.

Happy families are all alike. That is not a fact about families. It
turns out to be a fact about truth.
