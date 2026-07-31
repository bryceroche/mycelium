# THE DUP-ARGS DIET — FIRE-READY DESIGN
*(2026-07-31, staged so the word ignites it. Nothing here fires without
the word — this is the gen-22 staging pattern: design cold, fire fresh.
Three blind calls ride one fire; all instruments banked before design.)*

## The target and its measured state

**Species:** dup-args binding competition (`args=[a,a]`) under
distractor load — the [655]/[1382]/[875] engage-slip family; the
re-engagement law's population ("the second pawl fights the first").
**Demand vs supply:** books demand 51.7%, mix carries 12.3% (coverage
sweep). **Live baseline:** 53% misbinding under distractor load
(48/91, bench_rung2b.json) — the sharpest before-number any diet has
had. **Representable since gen-9** (the args=[a,a] bit); starved, not
excluded.

## The three blind calls (all pinned in the ledger before this design)

1. **WETTING** (the exchange law's first call): a diet fired WITHOUT
   book-8's dup-carrying rows in the mix underperforms one fired with
   them (books wet, diets fill).
2. **DOSE NON-MONOTONICITY** (#112's amendment): the cure has an
   optimum; past it, more distractor-load rows REDUCE the cure (the
   corpus itself packs).
3. **AUGMENTATION ORDERING** rides separately (its own words); not
   this fire.

## The dose arms (per the straddle demand)

**Anchor, consulted not remembered:** 4.85% is the one measured
full-cure dose in the record (gen-22: 400×10 on 78,400; d1 basin
0.53→0.00). The constitution's working fraction sits in the 3–5% band
(prose regularization measured at 2.9%×10). **Predicted break point,
named per the demand:** the slack clause predicts the peak NEAR the
measured working band and degradation past roughly 2× it — the packing
prediction is that ~12%+ share (approaching the mix's existing dup
share plus the diet = heavy species concentration) reduces cure.
**Arms: 2% / 5% / 12%** — below, at, above. Reachability: mintable at
all three (the 2b generator produces solver-verified rows at will);
the mix's own distribution bounds nothing below ~20%.

## The wetting arms and the cross (the readout-structure demand)

- **WET** = mix includes book-8's certified rows (×10 reps, the book
  dose convention) — the surface contact that touched the dup axis.
- **DRY** = the gen-22 mix as-is (book-8 never entered it).

**Option A — full cross (6 arms: 2 wetting × 3 dose):** the clean
factorial; dose curve read WITHIN each wetting condition; ~6 training
runs (~1.25h each on the gen-22 recipe's shape). The design the
interaction deserves if GPU budget allows.
**Option B — economy (4 arms):** dose curve within DRY (3 arms) +
WET at the anchor dose (1 arm). Reads: dose non-monotonicity fully
(within-condition); wetting at the anchor only. The interaction is
partially read; a wetting effect at the anchor licenses the full
cross later. Half the GPU.
**The word chooses A or B; the readout structure is fixed either way:
no marginal reads across the cross.**

## The corpus (per arm)

Dup-args rows minted by the bench_rung2b generator's family, widened
per the factory constitution: both ops (add/mul), distractor load 2-4
spanned, letters consecutive, values ≤300, solution-first,
solver-verified under the corrected uniqueness guard (doors.certify_unique),
knot-dedup via canon. Uniques per arm sized so share × reps hits the
arm's fraction at 10 reps (dose law: BOTH numbers declared per arm).
Schema per the mix row contract (decisions = solver's measured count;
mentions dict; real solution vectors — the gen-22 schema lesson).

## The recipe

Gentle continuation FROM g22 (the promoted lineage), fire_gen22 shape:
SGDR 4×4k, LR 1e-4, states precompute + live-forward verification
(6 sentinel rows incl. diet rows), ALG_ALLOW_PEN_TRAIN=1 (mix carries
pen rows), seeds 9X per arm. Each arm = its own mix file + its own ckpt
(g23armN naming); NO arm touches the gate; g22 remains the gate until
a verdict prints PROMOTED (the winning arm, if any, goes to battery).

## The after-reads (per arm, all banked instruments)

1. **The 2b population re-run** (same seeds = same 120 rows): misbinding
   rate vs the 53% baseline — the cure number, cell grain.
2. **SETTLE at mechanism grain** (z-scored WITHIN this fixture per the
   scale law): did the diet teach settling (misbound-settle
   distribution collapses toward correct-bound) or memorize the
   template (rate improves, settle distribution unchanged)? The
   wetting call's mechanism-grade readout.
3. **The standing rehearsal, both tiers** (entourage duty if any arm
   promotes; DEPLOYED expected 40/40 refused — mouth-widening watch).
4. **Regression floor:** bigtest on each arm's final ckpt (the bar-noise
   band applies at verdict time only for a promotion candidate).

## The verdict structure (pinned)

- **Wetting call:** WET beats DRY at the anchor dose on cure rate
  (directional, cell grain) AND on settle-distribution shift (mechanism
  grain) → the clause earns predictive credit; no difference → the
  clause is decorative and the exchange law shrinks honestly.
- **Dose call:** cure(5%) > cure(2%) AND cure(5%) > cure(12%) within
  condition → the amendment earns its peak; monotone rise through 12%
  → the amendment dies as registered; flat → MIXED, band unclaimed.
- **The diet's own bar** (independent of both calls): any arm cutting
  misbinding below 25% (half the baseline) at unmoved regression = a
  promotion candidate for battery; below 40% = partial; no arm below
  40% = the starvation hypothesis takes a hit and the binding bench
  inherits the species (the rate-family precedent: binding pathologies
  that data cannot cure).

## Costs

Option A: ~7-8h GPU (6 fires + precomputes + after-reads).
Option B: ~5h. Corpus mint: CPU, minutes. All analysis: banked-instrument
re-runs.
