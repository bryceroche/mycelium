title: A fancy lookup table
date: 2026-09-02

# A fancy lookup table

Strip away the vocabulary — the breathing cycles, the rotors, the
certification wall — and here is what our reasoning machine really
is: **a lookup table with fuzzy matching and grouping.**

That sounds deflationary. We mean it as the design's proudest
honesty.

The machine maintains **atlases**: maps of the operation-shapes it
knows, kept in two spaces at once — the language space (the many
ways English expresses an operation) and the math space (what the
operation actually does). Each entry is a **centroid**: the average
location of a known kind, the center of gravity of every example of
that kind the machine has seen. Reading a new problem means fuzzily
matching its silhouette against the atlas, grouping it with its
kind, and retrieving the exact machinery for that kind.

Keeping an atlas honest takes bookkeeping discipline. Centroids are
maintained with **Welford's algorithm** — the numerically careful way
to update a running mean and variance as examples stream in, one at
a time, without ever holding the whole history in memory. And the
atlas cannot simply be kept forever: we are now on the **seventh
generation** of centroids, one per era of the trained head, because
each new generation of weights *rotates* the internal coordinate
system. The rotation is nearly pure — aligned generations agree at
cosine 0.988 — but "nearly" is doing real work in that sentence, and
so every generation the atlas is re-anchored from scratch. Old maps
are never trusted in new coordinates. That rule has caught more
subtle bugs than almost any other in the project.

Why isn't this just memorization with extra steps? Because of what
the table's keys are. A memorizing system keys on surfaces —
sentences, phrasings, numbers — and shatters the moment a costume
changes. Our table keys on **silhouettes**: the structural shapes
that survive the 512-dimension waist after the costume is destroyed.
Ten thousand differently-dressed problems collapse onto a few
hundred keys. The table stays small; the coverage stays wide; and a
brand-new costume on a known dance looks up correctly on the first
try.

There is a respectable philosophical position that all cognition is
sophisticated retrieval — that expertise is less like derivation and
more like a chess master recognizing fifty thousand positions. We
take no side in that debate for humans. For machines we'll say it
plainly: retrieval with the right keys, over an honestly-maintained
atlas, backed by exact verification of whatever gets retrieved, is
not a lesser form of reasoning. It is the form of reasoning you can
*audit* — every lookup names the kind it matched, every kind names
the examples that built it, and every answer either survives the
solver or never leaves the building.

A fancy lookup table — with its keys chosen so well that looking up
becomes understanding.
