# Certify, Answer, Flag, Abstain: A Chain of Custody for Machine-Read Mathematics

*(Title per the lattice-is-the-brand decision; subtitle proposal —
"chain of custody" names the mechanism the four words decide through.
Final call at venue selection.)*

## Abstract

A deployed reasoning system's output should not be an answer; it
should be a decision. We present a system that reads algebra-in-words
into typed factor graphs, solves them exactly, and emits one of four
decisions — **certify, answer, flag, or abstain** — through decision
machinery that is entirely zero-parameter: no gradient flows through
any component that produces a verdict. Behind a four-link chain of
custody (input register, rendering invariance, lineage invariance,
truth), the system certifies 912 of 1,500 held-out problems at
measured 1.0000 precision — a zero-numerator bound, reported as such.
The trained footprint is 8.0M parameters against a 506M-parameter
frozen trunk slice; verification adds zero. On foreign benchmark prose
the same channel initially certified garbage at 2% precision. We
report this at full strength, because the zero-parameter recognition
gate it forced now refuses 100% of that text: certification is
distribution-bounded, and the bound is measured, gated, and published.
A three-book hand-annotation campaign moved the reading frontier until
its yield curve measurably saturated. Everything was built under
registered predictions with mechanized verdicts across fourteen model
generations; the complete ledger ships as supplementary material, and
the paper closes with a falsifiable prediction about its own
instrument's aging.

## 1. Introduction

Deployed language models produce answers without knowing when they are
right, and the discipline's natural instinct — read the model's own
confidence — measures the wrong thing. We can state this as a
measurement rather than a position. In our system, the entropy of a
five-way vote across re-renderings of the same problem separates
*fragile* parses from *settled* ones almost perfectly; it cannot
separate settled-and-correct from settled-and-wrong (H = 0.000 versus
0.212 — the confidently wrong are nearly as quiet as the confidently
right; Fig. 7-b). **Temperature is orthogonal to truth.** Any
architecture that ends at an internal confidence signal has built a
depth gauge and called it a compass. This paper is about what to build
instead.

The artifact is a **certification lattice** (Fig. 1): a system whose
output is one of four decisions — certify, answer, flag, abstain —
produced by a chain of custody in which each link checks one
invariance the previous link cannot. A recognition gate asks whether
the input is in the register the system was calibrated on; a five-view
vote asks whether the parse survives re-rendering; a cross-lineage
panel asks whether it survives a change of training history; the
answer key — present only in measurement, never in deployment — asks
whether it is true. Every link is zero-parameter, and every link is
justified in §7 by a named specimen that defeats the chain without it.
The accountability arithmetic is §3's census: 8.0M trained parameters
leveraged against a 506M frozen trunk slice, with every component on
the verification path adding exactly zero — there is nothing on that
path for training pressure to corrupt. At freeze, the certified
channel covers 60.8% of the widest fixture at measured 1.0000
precision, with the zero-numerator arithmetic stated where the number
is (§7.2, §11).

The paper's second product is the **method**. Every substantive claim
began as a registered prediction with kill bars pinned before
measurement; promotions are performed by battery scripts that either
write the deployment manifest or touch nothing; and the complete
chronological ledger — every registration, bar, verdict, and law —
ships as supplementary material. It is long, unedited, and contains
our mistakes at the same resolution as our results; that is what makes
it evidence rather than narrative. §9 shows the discipline at full
length on the campaign's hardest problem, and §9.3 tables the laws it
yielded, several of which (instrument rotation, estimator variance
masquerading as distance, selection against promoted gates) we believe
transfer to any measurement system that watches itself.

The **construction** is a reading campaign (§10): three books of real
mathematical prose, hand-annotated through a gate the annotators could
not flatter — it rejected its own author's first five attempts — and
measured to completion: the yield curve saturated (Fig. 9-a), the
mechanism was confirmed by three independent instruments, and the
annotation rulebook was written entirely by the parser's refusals.

The **limit** is reported at the same strength as the results. On a
public benchmark slice with no author fingerprint anywhere on its
input side, the certification channel initially signed foreign
garbage — 2% precision where it measured 1.0000 in-distribution — and
the flat abstention curve showed the system did not know what it did
not know (§8). The recognition gate exists because of that
measurement, now refuses 100% of the anchor's false certificates, and
the anchor sits permanently outside every acceptance path as the
standing examiner. We claim no benchmark competence; we claim a
certified channel whose boundary is measured, and §11 — drafted before
any other section — is the recommended first stop for a skeptical
reader: it states what the numbers do not show, including the
zero-numerator bound, the n=1-per-generation fact, and that the
annotator is the system's author.

The reader's map: §2 situates the work; §3 derives the two-jaws
architecture from the binding theorem; §4 gives the corpus discipline
(including train/test disjointness verified up to graph isomorphism);
§5 measures the repair stack to its boundary; §7 presents the lattice;
§8 the external anchor; §9 the method; §10 the reading campaign; §11
the honest limitations; contributions and author accounting close the
paper.

<!-- Assembly notes:
  - Section numbering: RENUMBER (decided 2026-07-15) — s05 merged the
    skeleton's 5+6, so s07->6, s08->7, s09->8, s10->9, s11->10; the
    reader's map, all cross-refs, and figure names update in the same
    assembly pass as the tag-check.
  - §7.2's "Figure N" placeholder resolves to the frontier figure at
    assembly (checklist item).
  - S2 (related work) drafts after this door per the relay's ordering:
    selective prediction + calibration mandatory; workspace/propose-
    dispose/neural-collapse/ARQ paragraphs as the framing demands.
  - Abstract word count ~200; every number verified present in a
    drafted section's own text.
-->
