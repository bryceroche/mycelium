# rotor_clock.py band status (truth-maintenance, THE CLOCK AUDIT)

**Question:** is `BREATH_BAND_PAIRS` (legacy v0) or `ROT_BANDS`
(`.cache/rot_band_draft.json`) the live band allocation for the breath
rotor's attention-pair reservations?

**Answer as of this audit: NEITHER is live in the trained/deployed head.**

`mycelium/rotor_clock.py` itself is honest about the intended direction
(line 56 comment: `BREATH_BAND_PAIRS = tuple(range(24, 32))  # legacy
(v0); live = ROT_BANDS`; module docstring: the live bands are
"audit-drafted... and travel via the ROT_BANDS json (.cache/rot_band_draft
.json) consumed by the head"). But grepping the repo for importers of
`mycelium/rotor_clock.py` (`scripts/`, `mycelium/`) finds **zero callers
outside the module itself** — `breath_qk_angles`, `phase_of`,
`wheel_table`, `cycle_rates`, `cycle_phasor`, `cycle_cos_sin` are all
unused externally. `scripts/phase1_algebra_head.py` (the actual trained
head, door #62 / `ALG_SIXWAVE`) implements its own independent six-wave
phase logic inline and never imports `rotor_clock`.

`.cache/rot_band_draft.json` exists (`{"mand": [3,4,11,27], "elec": [0,6,
8,9,10,12,13,14,16,18,19,21,23,24,26,29]}`) but has no reader anywhere in
the codebase besides the comment in `rotor_clock.py` that names it.

**Conclusion:** `rotor_clock.py` is a specification/reference module ("the
single source of truth" per its own docstring) that is not yet wired into
the live head. Its docstring's claim that "S2/S3 import their schedule
from here... sync enforced by the import graph" does not hold against the
current import graph. Neither `BREATH_BAND_PAIRS` nor `ROT_BANDS` governs
anything the champion (`sharp_port242`) actually ran with. Do not cite
either as "the live bands" in a claim until one of them is actually
imported by `phase1_algebra_head.py` — this note exists so that gap isn't
rediscovered the hard way at the next clock-audit pass.

`mycelium/rotor_clock.py` was read-only for this audit (not edited).
