## 7. The Certification Lattice

> *Temperature is orthogonal to truth.*

### 7.1 The lattice as a decision structure

The system's output is not an answer; it is one of four decisions —
**certify**, **answer**, **flag**, or **abstain** — and every rung of
that lattice is zero-parameter and gold-free at inference time. Nothing
in the decision path is trained on the decision it makes, and nothing
consults the answer key: the key appears only afterward, as the grader
of the machinery, never as a component of it.

The four rungs, with their dials as first measured:

1. **Certify** — five solution-preserving re-renderings of the input
   (sentence permutations), each parsed and solved independently;
   unanimity of the five forced answers. First measured at 0.9982
   precision and 38.1% coverage.
2. **Answer** — majority vote (at least 3 of 5), with a specialist
   repair path on vote-abstain. Composite: 71.5% end-to-end at 0.833
   precision.
3. **Flag** — a rank-sum of view-disagreement and centroid distance,
   read at the tail: kept-precision 0.862 at 10% abstention.
4. **Abstain** — no forced answer anywhere.

Behind the rungs stands a chain of custody, and each link answers a
different question. The **recognition gate** asks *is this input the
kind of text the system was calibrated on?* — an input-space check,
before any parse. The **vote** asks *is the parse invariant to how the
problem is rendered?* — five retellings must produce the same answer.
The **cross-model panel** asks *is the answer invariant to which
training lineage produced the model?* — independently trained siblings
must agree. The **answer key** asks *is it true?* Register, rendering,
lineage, truth: four invariances, in that order, and the sections below
show by specimen why no link is redundant.

Why can't a confidence signal replace the chain? Because the most
natural one reads the wrong axis. Vote entropy across the five
renderings separates *shallow* parses from *deep* ones almost
perfectly — in the pilot measurement (n=36), correct-but-fragile items
read H=0.846 while deeply-settled items read near zero. But
deeply-settled and *correct* are different properties: deep-correct
items read H=0.000 and deep-wrong items read H=0.212 — the confidently
wrong are nearly as quiet as the confidently right. Entropy measures
basin depth, not truth. That is the epigraph in numbers, and it is the
structural reason the lattice ends at an external key rather than at
any internal temperature: no amount of introspective calibration
substitutes for an invariance the system cannot fake.

### 7.2 The dials at freeze

At the frozen generation, the widest fixture (1,500 held-out problems)
reads: 1195 answered correctly one-shot (79.7%); five-view unanimity
certifying at measured 1.0000 precision; and the full cross-lineage
panel — unanimity across five renderings *and* three independently
trained models — certifying **912 of 1,500 (60.8%) at measured 1.0000
precision**. Figure N (the precision–coverage frontier) plots every
rung: relaxing unanimity to 4-of-5 and 3-of-5 buys coverage at measured
precision cost (0.9925 and 0.9832 at first measurement), and the
frontier's history across generations shows the certification channel
widening — from 0.9982 at 38.1% coverage in its first measurement to
1.0000 at 60.8% at freeze — as the reading campaign (§9) taught the
parser the register rather than teaching it the test.

The statistical fine print is inherited from §11 and not repeated here:
1.0000 on 912 is a zero-numerator bound, not a demonstration of zero,
and 1.5% of fixture items yield fewer than five distinct renderings.
One number deserves emphasis precisely because it is *not* 1.0000: the
channel's first measurement contained a broken certificate (570 right,
1 wrong). The lattice's history includes its own counterexample, found
by the key, autopsied, and fed back into the design — which is the
intended relationship between the machinery and its failures.

### 7.3 The specimens are load-bearing

Each link in the chain of custody has a named specimen showing what
happens without it.

**The false certificate exists.** Problem [71] of the census fixture,
presented as raw prose, voted 5/5 unanimous on a wrong answer — five
renderings, one voice, wrong. This is the certification channel's
nightmare case, observed exactly where theory predicts it: on text
*outside the calibrated register*, where all five renderings are read
by the same miscalibrated arm and the vote's independence assumption
fails silently. The recognition gate exists because of this specimen —
it reads the input's register upstream of any parse, and it reads
[71]'s prose as foreign. The doorman is not decoration; it is the link
that intercepts the case the vote provably cannot.

**The second wall.** For prose that slips past the gate, the
cross-model panel provides an independent check: on the census
fixture's raw-prose items where the gate model produced stable
unanimous votes, the panel *dissented on 9 of 10* (a later generation:
16 of 19). In-register, the panel is nearly idle — the single
stable-wrong item in the 1,500-problem fixture at the generation the
panel was adopted was broken by cross-examination, 1 for 1. Its
load-bearing jurisdiction is exactly the wild register: even text that
fools the doorman meets a jury drawn from a different training history.
Renderings share a model's blind spots; lineages do not.

**The quiet failure shape.** Problem [78]'s dialect voted a consistent
wrong answer across three of five views — an answer-channel error that
certifies nothing but *answers* wrongly with a stable voice. It is the
specimen for why the answer rung and the certify rung carry different
dials, and why deployment modes that cannot tolerate 0.833 must read
from the certified channel only.

### 7.4 Instruments age, by mechanism

The lattice's anomaly signals are instruments, and the campaign
measured how they age rather than assuming they don't.

The geometric monitor — per-kind centroids in the head's latent space —
lost discriminative power across generations. The audit found the
mechanism, and it was not the monitor's fault: the latent constellation
*rotates* between generations. Raw cross-generation centroid cosines
read ~0.59, as if the spaces were unrelated; after orthogonal
Procrustes alignment they read 0.988 with small residual. The
constellation's shape survives; its coordinates do not. The monitor
aged because nobody told it the sky had turned. The repair is
structural, not statistical: geometric instruments are re-anchored
every generation as a standing duty, and cross-generation geometric
comparisons are made only after alignment. (The panel is immune by
construction — its votes are answers, not coordinates.)

The deeper principle is selective. **Any signal promoted to a gate
becomes selected against**: once a signal joins the acceptance path,
the population of surviving errors is precisely the population that
passes it, and subsequent generations of failures are shaped — by
selection, not by intent — to hold their story against that signal. The
vote joined the acceptance path at the composed headline, so we
register the prediction here, in advance, that agreement-based
detection of committed-wrong parses will decline monotonically across
future generations — not because the instrument weakens but because its
population hardens. The design consequence is a rotation discipline:
**the portfolio must always hold one examiner out of the acceptance
path.** Today the held-out examiner polices the geometry's blind spot;
when it is promoted, something unselected-against must replace it. We
believe this is why anomaly detectors age in deployed systems
generally, and the lattice is designed around the expectation rather
than surprised by it.

<!-- FIGURES riding §7 (all measurements banked):
  F-7a precision–coverage frontier: rungs 5/5, 4/5, 3/5 + cert-v2 panel,
       with the generational trajectory 0.9982@38.1% -> 1.0000@60.8%
  F-7b vote-entropy basin separation: H by class (deep-correct 0.000 /
       shallow 0.846 / deep-wrong 0.212 / refused 0.116), n=36 pilot
  F-7c chain-of-custody diagram (drawn, not plotted): mouth -> vote ->
       panel -> key, one specimen annotated per link — candidate Fig. 1
-->
