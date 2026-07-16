## 3. The Architecture (two jaws, derived)

### 3.1 The binding theorem, and the design it forces

The architecture is best introduced by the negative result that shaped
it. A *concept*, in this domain, is not recoverable from either of its
two surfaces. From the language side: problems that read as obvious kin
— the taxi fare and the filling faucets of Fig. 3-a, both creatures of
the rate frame — share no graph class; their kinship is real, measured
in the frozen trunk's own geometry (z = −2.05), and entirely absent
from their wiring. From the structure side: problems with *identical*
canonical graphs — one digest, one knot — wear surfaces so different
that no lexical method recognizes them as related. We call this the
**binding theorem**: the concept is the binding between a linguistic
frame and a structural role, and it lives in neither side alone.
Fig. 3-a shows the four specimens; the structure-side sweep that found
graph-twins across fixture boundaries is §4's isomorph audit, and both
directions' full measurement records are in the ledger.

The two-jaws design is this theorem applied. If concepts are bindings,
then recognition and abstraction must live where the binding is made —
on the *parse side*, in the trained components that read language into
structure. The factor graph they emit is deliberately **frame-free**:
it records variables, relations, and quantities, and forgets that the
problem was about taxis. The factor-graph representation itself is standard (Kschischang,
Frey & Loeliger, 2001); what the theorem forces is where its content
may come from. And verification inherits *neither* side: the
symbolic solver receives only the graph, searches it exactly, and is
graded by the dataset's answer key. When higher-level abstractions
enter (macro-relations proposed from recurring subgraphs), they expand
to primitives *before* the solver sees anything — the key always grades
in primitives. The slogan form: **neural proposes, symbolic disposes** —
abstraction may live in annotation and recognition, never in
verification.

This gives the pipeline a cleaner algebra. The predicate registry is a
table of the solver's atoms — each relation kind an irreducible pattern
it understands natively, a *prime*. A parsed problem is then a
factorization: the parser's task is to express a composite stated in
words as a product of registry primes, and the frame-free graph is that
factorization written down. Macro-relations are composites admitted for
compression, but they expand to primes before the solver sees them,
because the answer key grades only in primes. The canonical digests of
§4 are the same idea made rigorous — a canonical factorization of the
graph — which is why disjointness *up to isomorphism* is disjointness of
prime structure, not of surface text. Two problems share a knot when
they share a factorization; strangers in words can be identical in
primes, and kin in words can be coprime. The registry's counted
frontier (§10) is, in this language, a list of primes not yet
discovered.

### 3.2 The components at freeze, census-verified

The construction jaw is a small trained head over a large frozen eye.
The **trunk** is the embedding table and first four layers of a
pretrained 1B-parameter language model, used input-side only and never
trained. The **parser head** reads the trunk's states into a typed
factor graph through two slot banks — 24 variable slots bound to
letters positionally, 24 factor slots — with bilinear pointers from
factor arguments to variables, a six-way factor typing plus an argument
multiplicity bit, and most-significant-digit-first quantity decoding.
(The six factor types are the parse-side surface, not the relation
inventory: registry relations enter as solver-side predicates bridged
onto these types — §3.3 — which is how double-digit relation kinds ride
on a six-way head output.)
The **repair specialist** is a second head of the same architecture,
retrained each generation on the gate model's organic failures and
consulted only when the vote abstains. Around them sits the
zero-parameter decision machinery of §7 (views, votes, panel join,
recognition gate, flag), and beneath them the **solving jaw**: a
general constraint-search core (arc consistency, forced-only commits,
a predicate registry) containing zero learned parameters and zero
domain-specific code in its core.

The parameter census, re-run at the freeze tag against the deployed
checkpoints rather than quoted from memory:

| Census row | Parameters |
|---|---|
| Trained and deployed (parser + repair specialist) | 8,005,722 |
| Trained, whole system — both jaws (the solver adds zero) | 8,005,722 |
| Frozen and leveraged (trunk: embeddings + layers 0–3) | 505,954,304 |

The second row equaling the first is the architecture's claim in
numbers: everything added to make answers *checkable* — search,
verification, certification — added nothing trainable, so there is
nothing on the verification path for training pressure to corrupt. The
leverage ratio (63× frozen per trained; 126× against the parser alone)
states the design bet: a pretrained model's early layers already carry
the reading; the trained head only learns where to point. Two sibling
heads (4.0M and 13.8M, one differing in lineage and one in width) are
additionally deployed at the certification tier as cross-examiners
(§7.2); they produce votes, never answers.

### 3.3 As-built versus as-designed

The honest paragraph. The design that survived contact is narrower and
better than the one first drawn. The predicate registry did what it was
designed to do: relations enter as a predicate plus a parse-side
bridge, with zero edits to the search core, and the registry has grown
through double-digit relation kinds that way. Two designed components
died: a hyperbolic embedding tier for the kind hierarchy was never
needed — hard membership structures (masks, slots, letter positions)
did its job with no learned geometry at all — and an elaborate
memory/notebook mechanism gave way to a plain repair signal: the vote
abstains, the specialist answers. In both cases a designed *object* was
replaced by a measured *action*. The nouns died; the verbs survived.

### 3.4 Design laws as constraints, not lore

Three regularities from the campaign function as standing constraints
on the architecture rather than commentary about it (sighting counts in
Table 9-1). First, the **pointer law**: pointer errors are never fixed
downstream of the pointer, so every new relation's pointer is *born*
candidate-restricted and span-supervised — the five remedies (masked
attention, span supervision, a comma, alphabetical discipline, ballast)
are a descending-cost toolkit applied at birth, not repairs applied
after. Second, the **discovered dialect**: the annotation language the
books converged on — consecutive letters, explicit knowns, one
declarative relation per sentence — was never designed; it was written
by the parser's refusals, one rule per refusal, and it now functions as
the system's intermediate representation. The one *designed* logical
form the project attempted is a tombstone (§10); the dialect that works
was discovered under selection. Third, the **two-channel spine**: the
strict separation of frame (parse-side) from structure (graph-side) was
an early architectural guess that the binding theorem later proved
load-bearing — collapse the channels and the concept has nowhere to
live except entangled in both, which is precisely the failure the
frozen trunk exhibits on its one chronic family (§11).

<!-- FIGURES riding §3:
  F-3a binding theorem specimens (banked, out/f3a_binding_theorem.*)
  Table 3-1: parameter census (inline above; re-run at freeze via
    safetensors headers — parser 4,000,813 / specialist 4,004,909 /
    trunk slice 505,954,304)
-->
