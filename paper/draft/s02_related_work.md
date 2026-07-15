## 2. Related Work

Every citation below was verified against its actual source before it
was characterized, and the bibliography carries a per-entry note on
what each work is cited for — the same use-matches-source standard the
rest of the paper applies to its own numbers.

**Selective prediction and abstention.** The reject option is as old as
pattern recognition (Chow, 1957; 1970), and selective classification's
risk–coverage framing (Geifman & El-Yaniv, 2017, and successors) is the
natural coordinate system for our Figure 7-a. Two things distinguish
the lattice from this line. First, the decision machinery itself is
zero-parameter: nothing is trained on the accept/reject decision, so
the selective mechanism cannot be shaped by the errors it will later be
asked to catch — a property we argue for by named counterexample rather
than by taste (§7.3). Second, the abstention literature typically
evaluates a single trained model with a confidence head on one
distribution; our claim is *distribution-bounded by construction*, with
the boundary itself measured, gated, and reported as a headline result
(§8) rather than as a threat to validity.

**Calibration and internal confidence.** Post-hoc calibration
(Guo et al., 2017) and language models' verbalized or self-evaluated
confidence (Kadavath et al., 2022) all improve the *readability* of an internal
signal. Our measurement problem with this entire family is §7.1's:
vote entropy — the strongest internal signal we found — cleanly
separates fragile from settled parses and cannot separate
settled-correct from settled-wrong (H = 0.000 vs 0.212). Internal
signals read basin depth. We do not argue calibration is useless; we
argue it calibrates the wrong axis for certification, which is why the
lattice's last two links are *external* (a differently-trained sibling
and an answer key) rather than better-calibrated internals.

**Self-consistency, met head-on.** The closest relative of our vote is
self-consistency decoding (Wang et al., 2023): sample multiple
reasoning paths, take the majority. The mechanism differs in three
load-bearing ways. Our five views are deterministic, solution-preserving
*re-renderings of the input* (sentence permutations), not temperature
samples — test-time augmentation in lineage (Krizhevsky et al., 2012;
Shanmugam et al., 2021), repurposed from accuracy averaging to
certification; notably, the dedicated study of TTA aggregation found
naive averaging imperfect, and unanimity is not averaging. Agreement
here measures invariance of the *reading*, not stability of a
sampler. Unanimity is used as a certification tier with
measured precision (1.0000 on 912, zero-numerator bound stated), not as
an accuracy booster. And we report where the signal provably breaks:
on out-of-register input all views are read by the same miscalibrated
arm, agreement decouples from truth entirely (2% certified precision,
§8), and a majority-vote system without an input-register gate inherits
that failure silently. Self-consistency work generally reports the
in-distribution win; §8 is the out-of-distribution invoice.

**Conformal prediction.** Conformal methods (Vovk et al., 2005;
Shafer & Vovk, 2008; Angelopoulos & Bates, 2021) offer
distribution-free coverage guarantees, which is more than our
zero-numerator bound provides — *under exchangeability between
calibration and test data*. That precondition is precisely what our
external anchor violates and what the recognition gate exists to test:
the mouth is, functionally, an explicit exchangeability check run
before any statistical claim is trusted. We view the two approaches as
complementary — a conformal layer over the lattice's scores is natural
future work — but we note that our headline failure mode (§8.2, foreign
text certified confidently) is exactly the one a conformal guarantee
calibrated in-register would also have signed.

**Propose/dispose architectures.** The two-jaws design belongs to the
family in which a learned proposer is checked by a sound verifier:
semantic parsing of math word problems to equation forms (Zhang et
al., 2020), autoformalization into proof assistants whose downstream
proofs are machine-checkable (Wu et al., 2022), trained verifiers over
sampled solutions (Cobbe et al., 2021; process-supervised in Lightman
et al., 2023), and — in spirit — speculative decoding's
draft-then-verify asymmetry, where a cheap proposer's output is
accepted only by an exact check that preserves the target distribution
(Leviathan et al., 2023; Chen et al., 2023). The lattice differs
where §3.2's census row does: our verification path contains zero
trained parameters end to end (trained verifiers move the corruptible
component rather than removing it), and abstraction is confined to
the propose side by rule — macro-relations expand to primitives before
the solver sees anything, so the checker's soundness is never
delegated.

**Retransmission.** §5's repair stack is, deliberately, an ARQ system
(Lin, Costello & Miller, 1984): error detection (the
portfolio), negative acknowledgment (the flag), selective
retransmission of flagged fields (the specialist), and a measured
per-round recovery decay that motivates shallow retry budgets. We
import the vocabulary because the communications literature solved the
bookkeeping of *when to stop re-asking* long ago; our contribution
there is the boundary measurement (§5, §9.2), not the loop.

**Instrument aging.** Distribution shift, deployment monitoring, and
Goodhart-style dynamics — "any observed statistical regularity will
tend to collapse once pressure is placed upon it for control purposes"
(Goodhart, 1975; taxonomy in Manheim & Garrabrant, 2018) — all
describe measures and detectors degrading in deployment. What we have not found
articulated in the abstention or monitoring literatures is the
selection mechanism of §7.4 — *any signal promoted to a gate becomes
selected against, because the surviving error population is by
construction the population that passes it* — together with its design
consequence (a rotation portfolio that always holds one examiner out of
the acceptance path) and a falsifiable, pre-registered prediction of
the paper's own detector decaying. We state this novelty claim
carefully: the ingredients (Goodhart's law, adversarial drift) are
old; the articulation as a deployment law with a succession plan and a
standing bet is, to our knowledge, new.
