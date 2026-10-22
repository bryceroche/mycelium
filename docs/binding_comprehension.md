# Binding for comprehension — the branch's opening map (2026-10-22)

Branch `binding-comprehension`, cut from `complex-planes` at the end of 10-21. Nothing here is a claim. It is a map of
what the record already says, the theory as Bryce put it, and the first toys, which get bars pinned before any read.

## 0. The problem, in measured terms

Mycelium is a bridge from reading to a calculator: neural proposes, symbolic disposes. The calculator side (snap reads,
the CSP solver, the answer key) is sound. The READING side is the wall, and on 10-21 it got numbers:

- A wrong given pick is **blind to owner, type and event**: it hits each at exactly the row's base rate (filter_f).
- Behind a perfect (owner, type, event) filter, **79 % of wrong given picks would have one live value** (candidate census).
- Failures **cluster in hard rows** (whole rows right 7–11× the independent prediction). Hard rows collapse in their
  **given picks** (0.19 vs 0.83 on easy rows), and relations fail ~half the time even with every parent right.
- **Llama gives the vocabulary, not the grammar**: words and cues are sparse features in the trunk; OWNER is not, at
  layer 4 or 7 (the D4 / D7 dictionaries).

So comprehension's missing piece is not facts. It is a QUERY that knows what it is looking for, and a working memory in
which the problem's own mentions are bound to their roles, owners and events so that query can find them.

## 1. Bryce's theory, piece by piece, against the record

| Idea | What the record says |
|---|---|
| **LLMs make unique keys in the MLP from combinations of near-perpendicular dimensions** | The mainstream reading too: superposition (Elhage et al. 2022, *Toy Models of Superposition*) and the MLP as key–value memory (Geva et al. 2021, arXiv 2012.14913). Our D4 / D7 read agrees for THINGS (words, cues) and finds no key for BINDINGS (owner). |
| **Store knowledge in a database; the net searches instead of memorizing** | The literature's version: product-key memory (Lample et al. 2019), kNN-LM, Memorizing Transformers, RETRO. Our tree rung (10-20): path-sum codes route coarse queries to UNSEEN leaves (1.000) where per-node learned codes memorize prefixes (0.342); a learned orthogonal book found hierarchy-preserving codes (0.988). The decisive property is untested: answering about facts INSERTED AFTER TRAINING without retraining. |
| **The hierarchical tree: an infinite address book, coarse to fine** | Matches two of the twelve hills (the hierarchical codebook walked coarse-to-fine each breath; the atlas as tree centroids). The registry is already the owned world model; the tree is a natural ADDRESS BOOK for it (schemas, macro-factors, the lexicon). |
| **Binding concepts together into more complex thoughts** | The record is narrower than "we achieved compositional generalization with phases": phases beat orthogonal and free matrices inside binding machines (+0.065; K = 4 +0.068 / +0.207), but **a plain looped block with a snap after every hop beat every binding machine on unseen pairs** (snap control, 0.965 / 0.980 vs 0.755). The generalization came mostly from DISCRETIZING BETWEEN STEPS. |
| **Phase binding inside attention** (Gemini: "attention as wave interference") | Two failed precedents, both read on pinned bars: binding toy v1 (10-15: the phase-bound q/k arm NEVER led a plain transformer on held-out combinations) and the main-line carriers (10-15: phase carriers live and dominant in the router's KEY; wild −0.016; the QUERY could not use them). Binding in the KEY is not enough; the query side is where both walls sat. |

Corrections to the relay (Gemini): "strips out hallucinations entirely" is not a property any of this has shown;
"geometric precision rather than blurry probabilities" mixes two things (phases are continuous too; precision came from the
SNAP); and phase binding in attention is not untried, it is twice-refuted in the form proposed.

## 2. Where binding could plausibly enter the transformer (open thinking, ranked by evidence)

1. **A bound working memory written by reading, searched by structured queries** (strongest fit). Each mention is
   written as role ⊗ filler into per-event registers addressed by the clock (the 10-21 time rungs: registers store, bands
   address; gate-then-compare orders). A slot asks with a STRUCTURED query (owner ⊗ type ⊗ event expectation) and
   unbinds, then snaps. This is ARIS's entity-state table, learned; and it is exactly what the f-read says is missing.
2. **The MLP as an address generator into an external store** (Bryce's "search the tree"). The MLP's output is a
   coarse-to-fine path code; retrieval is exact; the store can grow without retraining. For Mycelium the store is the
   registry (lexicon, schemas), not world trivia.
3. **Cross-attention to the store, with binding as the join** — retrieved entries are bound to the problem's own
   mentions (the registry's unit "ounce = 16" bound to THIS problem's "8-ounce cup").
4. **Binding inside self-attention's q·k** — lowest: twice refuted as a key-side form; revisit only as a QUERY-side form.

## 3. The first two toys (bars pinned before any read; CPU; nothing fires without the word)

- **T1 — SEARCH, DON'T MEMORIZE.** Facts live in a tree store; arms: an MLP that memorizes them / a net that emits a
  coarse-to-fine path code and retrieves / path-sum codes / a learned orthogonal book. The decisive split: facts
  INSERTED AFTER TRAINING (and moved facts: an entry edited after training). A memorizer cannot answer them; a searcher
  should, at no retraining cost.
- **T2 — THE BOUND WORKING MEMORY.** Synthetic transfer stories (owners, objects, events; v2's lesson: the lookup needs
  EVENT membership, not just roles) with held-out (owner, verb, object) combinations and longer stories. Arms: plain
  transformer / + bound register memory with a structured query / + snap after each read / the same with an UNSTRUCTURED
  query (the f-read's failure in miniature). The question the main line asks: does a structured query fix the
  "blind wrong pick", and does binding beat a plain slot table at equal memory?

The main line's measured lever (the given-slot query gate, registered 10-21) is unchanged and parked, not dropped:
T2 is its toy-side twin.
