# MYCELIUM — an explainer for a technical friend (2026-08-09)

**Thesis in one line:** a neuro-symbolic math system where *neural
proposes and symbolic disposes* — a small trained head reads natural-
language algebra into typed factor graphs, an exact solver crushes
them, and an answer key certifies everything. The December claim:
**a system that never lies.** Concretely: it answers roughly 0.2%
of addressable MATH-500, has never certified a falsehood across 291
in-register rows and eleven vintage-labeled batteries, and abstains
40/40 at the wild frontier — where the mouth refuses 98.4% of
incoming problems. The never-lies property on wild text the mouth
*admits* is unmeasured, because it admits almost none. Never lies
where it speaks; speaks only where it was taught.

## The funnel (the core idea)
Think of it as a **fuzzy lookup table acting as a funnel**: an
information bottleneck that *destroys the surface variation of
English while preserving the math graph underneath*. "b plus b
equals c," "the sum of b and b," "twice b makes c" — one graph.
The **Anna Karenina principle is the certification story**: gold
parses of the same problem compile to ONE canonical knot (the
canonicalizer's false-merge rate: measured zero), while every
failing parse fails in its own way (a taxonomy of unhappy
families, each mapped by measurement). The parser reaches that
ceiling on roughly two-thirds of rows today — five independent
views collapsing to one knot — and the shortfall is itself a
diagnostic surface (18% of rows agree on *answers* while
disagreeing on graphs, which is exactly the kind of coincidence
the certification chain exists to catch). **The funnel's honesty
guard sits here too**: on memorized text the system scores 0.951;
on never-seen text, 0.023 — the lookup table *knows* it's a
lookup table, which is why it certifies rather than guesses.

## Two sets of grouper jaws
1. **The construction jaw** (neural): ~3.2M trained parameters over
   4 frozen Llama-3.2-1B layers (L0–L3, input-space only, never
   trained). The trunk feeds a **512-dim waist**; slot banks over it
   **segment and classify the dancer's silhouette** — attention maps
   claim each factor's text span (segmentation), typed heads
   classify it against the **predicate registry** (8 factor types,
   ops, argument pointers, digits).
2. **The solving jaw** (symbolic): a general CSP core — GAC/MRV,
   forced-only commits, zero domain code in the core. The interface
   between jaws is the typed graph itself; **the solver never sees
   attention, only the graph** (that separation is load-bearing:
   abstraction may live in recognition, never in verification).

## Breath cycles + the 50 First Dates notebooks
The parser currently runs **3 breath cycles** (its deployed config;
the earlier general engine ran K=8–16, and the ~3.5-breaths-per-
implication-layer constant from that era is conjecture-grade in its
own ledger entry) — iterative refinement at constant altitude,
slots attending to slots (evidence-sharing masks from the
model's own first parse). Two memories, like Lucy's notebook:
**REPLACE** (the working state, overwritten every breath — the pan)
and **ACCUMULATE** (the rings: committed bindings with evidence
anchors — the written record that survives the morning). We built
the commit pawl, a *beam exit* (committed slots leave the attention
mixer — finished dishes off the stove), and a reverse gear whose
transport is **message passing done honestly**: not loopy BP on the
solve path (refuted — arc consistency beats it; that door is welded)
but the solver's own unsat objection as a borrowed channel back to
the parse (measured: it carries 74% of the traffic at zero new
machinery).

## The Nazaré wave (the clock, organ 3)
Commitment timing. Our deepest measured pathology was **premature
commitment** — bindings decided at token 21 when the deciding
evidence arrives at token 23. The cure frame is not a periodic
clock (π-cycles failed three times on an earlier architecture) but
**Nazaré**: an underwater canyon that *focuses* ordinary swell into
one monster break. Graph structure is the bathymetry, evidence is
the swell, a commit is the wave breaking where the medium narrows.
V1 (commit gated on own-sentence completion) moved timing 13/15 →
1/15 premature — the WHEN tool works; the spatial grain needs its
own clock (constituency measured; its stationary reference is the
anchor).

## The tower (MLIR for math language)
The IR ambition: a **multi-level tower** — surface English lowers
through schema altitudes to primitive factors; macros expand before
the solver; **the key always grades primitives**. A census showed
83% of in-dialect failures are schema-altitude extraction misses,
so the tower's next floor (the "schema floor") is gated-in. One open
design question, pinned before build: is a floor a *committed
artifact* the parse consumes (certifiable) or a *refined altitude*
in the loop (dynamics — measured to be parallel, not staged)? Only
artifacts can pass the chain of custody.

## The week's governing discovery
**The band lives in the heads; the waist is band-general.** Linear
probes on the 512-dim waist read every "impossible" cell at 1.000 —
rare constructions, novel phrasings, crowded contexts — while the
trained output heads sit at chance or *anti-correlated* on the same
states. Surface-brittleness is not in the representation; it's in
the last linear maps, and it cures by **aim** (deliberate gold at a
cell) or **broad-gold folds**, never by volume. The full 2×2 is
measured: **mass without aim buys nothing** (42k balanced rows,
coin-flip head); **aim without the band buys a flicker** (a 2%
targeted sliver, near-zero transfer to held-out surfaces); **aim
with the band cures** (one deliberate fire holds a 2% minority at
0.95 permanently). "Cures by aim" is a measured claim with its
null.

## The graveyard that taught the laws (honest status)
- **Perceiver** (latent recirculation): refuted at architecture
  grade — frozen-trunk recirculation is dead at init (blur, not
  compute); the three-pole map (dead / violent / wrong-attractor).
- **HMM/telegraph signal** (alternating-layer carrier): lost to
  static layer polarization on the wrong substrate; dead on ours.
- **Neural diffusion compiler**: retired with its remainder — the
  loop is *parallel* refinement, depth-ordered, clocked (signature
  C); the sampling-path analogy didn't survive measurement.
- **Spectral decomposition**: era findings from the June engine
  (regional spectra; silhouette two-space: beliefs=envelope,
  residual=carrier) — banked, resting.
- **Poincaré ball**: FUTURE, parked at the interface for
  topological mapping over the predicate registry — with its law
  pre-carved (hyperbolic quantities never enter softmax without a
  log-map; cosine is wrong in the ball), so when its flag lifts it
  cannot arrive in refuted form.

## The method (why any of this holds)
Every mechanism above earned its status through a **pre-registration
door**: bars pinned before measurement, kill criteria named before
fires, verification at the *mechanism grain* (token-level timing
reads, per-station splits), and a registry of 186 gut instincts
each sorted against the record. Our recurring design question —
**"which mature field has already lived under this constraint?"** —
is where the frames come from (metallurgy, kitchens, surf breaks,
jet engines), and each frame must find its measured referent or be
refused with a citation. Three words have been *retired by
measurement* ("slope," "under load," "trade") for presuming shapes
the world didn't have.

**Where it stands:** zero certified lies across 291 rows and eleven
batteries; total abstention at the wild frontier; ~0.2% of
addressable MATH-500 answered and certified — and the hardest
internal wall
(duplicate-argument binding) disassembled to working machinery this
week. A certifier that grows, never a guesser that shrinks.
