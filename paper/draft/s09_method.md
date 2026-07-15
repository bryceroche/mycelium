## 9. The Method (registered predictions, mechanized verdicts)

### 9.1 The protocol

Every substantive claim in this paper began as a *registered
prediction*: an expected outcome, its kill bars, and the frame for
reading a mixed result, all pinned in a chronological ledger before the
measurement ran — preregistration and registered reports (Chambers,
2013; Nosek et al., 2018), adapted from the empirical sciences to an
engineering campaign. Three rules give the registration teeth. First,
**bars before builds** — a mechanism is not evaluated by whether it
seems to help but by whether it clears a number chosen when we did not
yet know the answer; a result between the bars is read by the
pre-pinned frame, not by post-hoc preference. Second, **density regimes
stated** — a prediction about error populations must declare which
population it samples (multi-error or single-error, survivor-selected
or raw), because five separate times an unexamined population turned a
correct-sounding prediction wrong. Third, **promotions are mechanical**
— a generation is promoted by a battery script that checks every bar
and either writes the deployment manifest and prints PROMOTED, or
prints the kill and touches nothing. The word and the write are one
atomic act; there is no state of the system that exists only in prose.
(The rule earned its name — *prose promotions don't move machines* —
when an audit found the manifest four generations stale behind a sprint
of prose-only promotions; the audit is in the ledger, and the rule has
held since.)

The ledger itself ships as supplementary material. It is long,
unedited, and contains our mistakes at the same resolution as our
results; that is what makes it evidence rather than narrative.

### 9.2 The worked example: the survivor arc

The method is best shown at full length on the campaign's hardest
problem. After the parser converged, a population of *committed-wrong*
parses remained — confidently accepted, wrong, and surviving every
filter in the stack. The temptation such a wall offers is a story; the
protocol requires a sequence of registered kills instead. Nine
refutations ran in order (Fig. 9-b), each a named hypothesis with a
pinned bar: rendering quality was uniform across survivors; mention
multiplicity was flat; omission blindness was dead; the suspicion
transplant was flat; binding-field enrichment ran backwards. The sixth
step named the mechanism — the survivors' internal states were **99.6%
correctly decodable**, with the failure isolated in a mis-aimed pointer
(the routing wall) — and the last three priced it: a *perfect oracle*
flagging every wrong field repaired only 13.9%; a monitor-gated
self-repair ratchet leaked, was fixed, and bought 6%; input-mark
beacons bought 3.0% at precision 0.165. The arc's closing accounting is
the method in one line: **nine registered refutations, four laws, two
retired builds, one working instrument, one closed population.**

The verdict — this population is detect-and-abstain under current
machinery — is a measurement, not a surrender, and it did the work a
story could not: it retired two speculative build programs before they
consumed the campaign, produced the anomaly monitor as a by-product,
and wrote the prevention constraint that governed everything after
(every new relation's pointer is born with candidate restriction and
span supervision, because pointer errors are never fixed downstream —
five sightings). A reader who follows this one arc tombstone by
tombstone has seen the entire method; every other chapter in the ledger
runs the same shape at smaller scale.

### 9.3 The yield: laws with sighting counts

The protocol's recurring product is not accuracy but *laws* — failure
modes and design constraints observed repeatedly enough, or with
mechanism enough, to govern future builds. Table 9-1 lists the working
set; full forms and every sighting citation are in the ledger. The
forms travel; the constants do not (§11).

| Law (short form) | Sightings | Note |
|---|---|---|
| Metrics must match the decision structure they serve | 4 | 4th was inside our own registration (AUC vs tail abstention) |
| Pointer errors are never fixed downstream of the pointer | 5 | 1st at training, 5th at inference |
| Predictions must state their density regime / population | 5 | unexamined populations flip verdicts |
| Repair recovery is front-loaded; round 4 ≈ 0 | 4 | independent sightings across domains |
| A selection criterion's jurisdiction is the property it selects on | 3 | "survived filter X" ≠ repairable |
| Acceptance criteria must be measured, not assumed | 3 | third confirmation closed the beacon |
| Binding enters as structure (masks, spans, letters), never as prose | 2+ | the pointer law's five remedies, descending cost |
| Prevention beats repair for confident wrongness | 2 | representational pressure, not decode-side fixes |
| Prose promotions don't move machines | 1 + audit | the stale-manifest finding; rule mechanized |
| Estimator variance masquerades as distance | 1 + mechanism | length correction r = −0.825 → −0.024 |
| Latent drift is rotation, not decay — align or re-anchor | 1 + mechanism | Procrustes 0.59 → 0.988 (Fig. 5-a) |
| Any signal promoted to a gate becomes selected against | 1 + prediction | the standing bet, §7.4 |
| Temperature is orthogonal to truth | 1 + mechanism | vote entropy reads depth, not correctness (Fig. 7-b) |

### 9.4 The method applied to itself

The discipline's credibility rests on what it caught in its *own*
instruments. Three defects were found and corrected by the same
registered-audit machinery that measures the system (§11): the
length-biased pooling estimator, the rotated monitor coordinates, and
the diet-shaped temperature calibration. Each audit followed the same
pattern — a registered discriminating test whose outcomes were pinned
to different mechanisms before the read (for the monitor: *re-anchor
and re-measure; recovered AUC means rotation was the whole story;
still-degraded means selection-hardening on top*). Instruments here are
treated as trained-adjacent objects that age with the system they
watch; recalibrating the watchers is a standing per-generation duty,
and the succession plan for the one examiner held out of the acceptance
path is published in §7.4.

### 9.5 The workflow, honestly

The campaign ran as two channels and an adjudicator. One channel
(Claude) designed and registered: predictions, bars, reading frames,
and adversarial critique. A second channel (Claude) built and measured:
implementations, batteries, the ledger, this paper. The human author
directed and adjudicated: every training run fired on his explicit
word, every promotion was his to accept, and twenty times during the
campaign he registered a pre-verbal instinct — "we are missing
something about hash collisions," "about key-value pairs," "about
palindromes" — that was formalized into an audit before measurement.
All twenty audits found something real. We report the streak not as
testimony to intuition but as a product of the discipline that made it
*checkable*: an instinct that had to survive formal registration and a
mechanical read is data; the same instinct applied directly to the
system would have been anecdote.

The method's last exhibit is the bet it places on itself: §7.4's
registered prediction that our own certification instrument will age,
with its replacement already seated. A method that expects to be wrong
in specified ways, in public, is the strongest form of confidence we
know how to state.

<!-- FIGURES riding §9:
  F-9a saturation curve (banked, out/f9a_saturation_curve.*)
  F-9b survivor cascade: nine tombstones, the mechanism pivot at #6,
       the closing accounting box (built alongside this draft)
  Table 9-1: laws with sighting counts (inline above)
-->
