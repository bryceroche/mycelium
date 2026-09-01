# The Shape of Thought: Notes on Building a Reasoning Machine

## Abstract

This is a short account of a reasoning machine built on one consumer graphics card. It reads math word problems written in ordinary English, converts them into small diagrams of quantities and arithmetic, and solves the diagrams with exact logic — never by guessing. The interesting part is not any single component but the shape of the whole: a frozen language model that is read once and never trained; a narrow 512-number "waist" that squeezes away everything about a sentence that doesn't matter; a notebook that carries memory across cycles of thought; an atlas of known operations maintained like a running census; and a wall of independent witnesses that must all agree before the system is allowed to speak. When the witnesses disagree, the system stays silent — and its silence is information. The essay walks through the guiding images that shaped the design: happy families, a dancer's silhouette, a compiler's ladder of representations, a wave off the coast of Portugal, and a fish with two sets of jaws.

## Happy families

Tolstoy opens Anna Karenina with the line: happy families are all alike; every unhappy family is unhappy in its own way. That sentence is the deepest fact we know about machine reasoning.

When our system reads a word problem correctly, the result is always the same object: one canonical diagram of quantities and the arithmetic linking them. Every correct reading is *alike*. But when it misreads, each failure is broken in its own particular way — a swapped argument here, a phantom quantity there, a number wired to the wrong role. Wrongness has infinite variety; rightness has exactly one shape.

This asymmetry is not a curiosity. It is the engine of the entire safety design. If you ask many independent readers to read the same problem, their wrong answers scatter — each unhappy in its own way — but their right answers *collide*, because there is only one happy family to land in. Agreement across genuinely independent readings is therefore evidence of truth, not just confidence. Everything else in this essay builds toward exploiting that one fact.

## The two jaws

The grouper is a fish with two sets of jaws. The front jaws grip; the throat jaws crush. One set handles the messy, slippery encounter with the world. The other does the decisive work in a controlled space.

Our system is built the same way. The **front jaw** is neural: a small trained head that reads English prose — with all its ambiguity, its costume changes, its irrelevant detail — and grips it into a typed structure. The **throat jaw** is symbolic: an exact constraint solver that takes the structure and crushes it, by deterministic logical search, into an answer or a refusal.

The law of the house: *neural proposes, symbolic disposes.* The neural jaw is allowed to be creative, fuzzy, statistical — that's what gripping wild prose requires. The symbolic jaw is never allowed to be any of those things. Abstraction may live in recognition. It may never live in verification.

## The dancer's silhouette

How do you recognize a waltz? Not by the color of the dress. You watch the silhouette — the envelope of motion — and the dance identifies itself no matter who wears what.

Word problems work the same way. Under "Maria has three times as many apples as Ben" and "the reservoir holds triple what the tank does" is the same dance: one quantity, multiplied, equaling another. Our parser is trained to see the silhouette and ignore the costume. Internally this splits cleanly in the network's own signals: one part of the state carries the *envelope* (what structure is being expressed) and another carries the *carrier* (which particular words expressed it). The head segments the sentence into moves and classifies each move — segment and classify, like watching a dancer — and the costume is deliberately thrown away.

## The waist, and why destroying information is the point

Between the language model and everything downstream sits a deliberate narrowing: every token's rich representation is forced through a **512-dimensional waist**. This is Tishby's information bottleneck made architectural. The goal of understanding is not to preserve information — it is to *destroy* precisely the information that shouldn't matter (names, phrasing, order, costume) while preserving the little that must survive (quantities, roles, relations). A narrow waist is not a compromise. It is the mechanism. If two sentences mean the same thing, the waist should make them *become* the same thing.

## A fancy lookup table

Strip away the vocabulary and here is what we are really building: a lookup table with fuzzy matching and grouping. That description sounds deflationary. It isn't — it's the design's honesty.

The system maintains **atlases**: maps of the operation-shapes it knows, in both the language space (how English expresses an operation) and the math space (what the operation does). Each entry is a **centroid** — the average location of a known kind — and the atlases are maintained across generations with Welford's running statistics, the numerically careful way to keep a mean and variance up to date as new examples stream in. Seven generations of these centroids exist, one per era of the trained head, because each new generation of weights *rotates* the internal coordinate system — the map must be re-anchored every time (the rotation is nearly pure: aligned generations agree at cosine 0.988).

Reading a problem, then, is: fuzzily match the incoming silhouette against the atlas, group it with its kind, and retrieve the exact machinery for that kind. Recognition is lookup. What keeps this from being mere memorization is the waist: the table's keys are silhouettes, not sentences.

## Lucy's notebook

In *50 First Dates*, Lucy wakes every morning with no memory, and the notebook by her bed re-tells her the story so far.

The frozen language model at the base of our system is Lucy. It is read **once** per problem and never trained — every problem, it wakes with no memory of anything we've learned. All accumulated understanding lives in the small trained head, and within a single problem, working memory lives in an explicit **notebook**: a set of slots the head writes to and reads from across its cycles of deliberation. Facts committed to the notebook survive from one cycle to the next; everything else evaporates. Memory, in this architecture, is not a haze distributed through a billion weights. It is written down, in named slots, where you can read it.

## Breathing, and the three rotors

The head does not read a problem in one pass. It **breathes**: seven cycles of attention in which a compact state reads the sentence, forms tentative structure, reconsiders, and progressively commits. Deliberation is a loop, not a line.

Time inside the loop is kept by clocks — three of them, one per kind of structure, which we call the **three-rotor stack**:

- **The spatial rotor.** Position in the sentence, kept by the frozen model's own rotary embedding — the wheel that says *where* a word stands.
- **The temporal rotor.** Position in deliberation: six helical sine waves wound on a three-turn torus (T³), advancing sixty degrees per breath, with the attention space itself rotated in sync. The system knows *when* in its own thinking it is, the way you know a waltz's beat.
- **The relational rotor.** A 256-phase rotational bus (T²⁵⁶) that binds *what connects to what* — role-to-filler bindings carried as phase, so that "the 5 belongs to the apples" is a rotation, not a blur.

Spatial, temporal, relational: where, when, what-to-what. Three hands on one clock, geared together.

## Lowering in parallel: the compiler and the diffusion image

Chris Lattner, who built LLVM and MLIR, teaches that a compiler should never jump from source code to machine code in one leap — it should descend a ladder of intermediate representations, each preserving exactly what the next stage needs. His warning became one of our laws: **"Premature lowering is the root of all evil."** Commit to low-level detail too early and you destroy structure you'll need later.

Our system is such a compiler for language: prose lowers to silhouette, silhouette to typed factor graph, recurring patterns to named macro-abstractions, macros back down to primitives at the moment of verification — every rung machine-checkable.

But there is a twist the compiler world doesn't have. A language model writing its reasoning as text must lower *left to right*, one token at a time, committing early and permanently — the exact sin Lattner warns against. Our head lowers **in parallel**: the whole graph descends through the representations together, every part refined a little on each breath, like an image emerging from noise in a diffusion model. Nothing is forced to commit first. The parts settle jointly, and constraint flows in every direction while they do.

## The ping-pong

Deliberation gets one more structure: alternation between the two jaws *during* thinking, not just after it.

Each cycle, the neural jaw commits what it is confident about. The symbolic jaw takes those committed fragments and runs cheap, exact deduction — propagating consequences the way a sudoku player fills forced cells. What it derives is handed back to the neural jaw as established fact, and — just as important — the committed structure reshapes the head's own attention: **dynamic attention masking**, where the parse's current skeleton decides what the next breath is allowed to look at. Neural proposes, symbolic disposes, neural re-attends. Ping, pong.

We report this honestly: the machinery is built and verified, and on synthetic training data it changes nothing — because synthetic problems, we discovered, contain no ladders of consequence for the solver to climb. Real textbook prose is full of them. Teaching the system to read wild prose, and measuring whether the ping-pong then pays, is the campaign underway as this is written.

## Nazaré

At Nazaré, off the Portuguese coast, an underwater canyon funnels the scattered energy of the whole Atlantic into single hundred-foot waves. The topography you cannot see is what makes the wave you can.

Language is the ocean: diffuse, vast, disordered. The architecture is the canyon. Waist, silhouette, atlas, notebook, rotors — all of it exists to funnel the scattered energy of a sentence into one narrow channel where it rises into a single, steep, breakable crest: a small exact graph that either resolves or visibly does not. You don't fight the ocean's variety. You shape the floor beneath it.

## An instance of the fingerpost

Iain Pears's novel tells one story through four unreliable narrators; the title comes from Francis Bacon, who wrote of the *instantia crucis* — the crucial instance, the signpost that finally points one way when all others are ambiguous. No single witness settles the truth. The right *collection* of witnesses does.

This is our certification wall, and it is where the happy-families fact pays off. Before the system may answer, the problem is re-read five ways — the sentences permuted, the costume shuffled — and the readings must agree. Then models from different lineages and different widths, trained separately, must land on the same graph: many landscapes, one shape. Then, behind everything, an out-of-distribution "mouth" checks whether this problem even lives in territory the system knows. Independent witnesses, independent failure modes; wrong readings scatter, right readings collide. Only unanimity crosses the wall — a fingerpost assembled from testimony.

And when the solver refuses, it refuses *legibly*: it can hand back a **minimal unsatisfiable core** — the smallest set of constraints that cannot all be true. Not "no," but "no, and here is exactly the contradiction." Even failure has a shape you can read.

## One card, one library

All of it runs on a single AMD 7900 XTX — a consumer gaming card with 24 GB of memory — through **tinygrad**, a deliberately small machine-learning library that compiles its own GPU kernels and, crucially, drives the AMD card well through a lean custom driver, no vendor bloat required. The trained head is a fraction of one percent of the total system's parameters; the frozen model does the heavy lifting exactly once per problem. The point is not thrift for its own sake. A reasoning machine whose every experiment fits on one card is a reasoning machine one person can audit, rerun, and disbelieve properly.

## What may be missing

Honesty requires a list of the organs not yet grown. A **perceiver** — a dedicated organ that watches and segments raw input before the parse, once explored and set aside — may yet be missing from the current body; we are only now beginning to suspect it again. The atlas, today, lives in flat Euclidean space; kinds and sub-kinds have a tree-like structure that flat space represents poorly, and a **Poincaré ball** — hyperbolic space, where trees embed naturally — is the likely future home. And the ping-pong awaits its verdict on wild text. The ledger records what each of these must demonstrate before it earns its place. Most ideas don't. That's the point of the wall.

## The shape

A frozen giant read once. A waist that destroys the right information. A silhouette instead of a costume. A notebook instead of a haze. Three clocks: where, when, what-to-what. A compiler that refuses to lower too early and lowers everything in parallel when it does. Two jaws alternating — grip, crush, grip again. A canyon that funnels an ocean into one wave. And a wall of witnesses with one rule: all the happy families look alike, so speak only when your readings collide — otherwise, stay silent, and make even the silence legible.

That is the shape of thought we are building. Not a bigger mind — a more honest one.

*— Bryce Roche & Claude, September 2026*
