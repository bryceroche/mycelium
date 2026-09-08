# THE POLAR WAIST GENERATION — spec (2026-09-07, word given)

**Owner:** Bryce + Claude · **Status:** REGISTERED, unbuilt · **Ledger:** entries
of 2026-09-07 (the clock read, the amplitude read, the ladder verdicts, the
word). **Fences:** research lineage only; deployed stack/manifest untouched;
env-gated (`ALG_POLAR`), birth-equivalent when unset (eq A/B/C bit-identical);
no diagnostic ever supervised; rotor_clock.py becomes the single source of
truth by IMPORT, not by prose.

## 0. What was measured (the reasons)

| fact | number | source |
|---|---|---|
| loop state does not rotate | 0.000 Procrustes mass in [50,80) deg; 0.98 in [0,20) | clock_read.py, both fixtures |
| breath identity in the state | probe 0.60 mint / 0.53 wild (bar 0.95); breaths 0,1 perfect, then smeared | clock_read.py confusion |
| state grows radially | per-breath norm 7.0 -> 11.9 (mint), 8.3 -> 14.1 (wild), monotone | clock_read.py |
| loop state effective rank | 64 PCs keep 95% (mint) / 91% (wild) of variance | clock_read.py |
| the existing rotor is a whisper | fed item 7a: Q-side 60 deg on pairs 24..31 of 32 per head, behind mixer gains fed_mx_hg = 0.023 (champion), 0.013 (cooked) | rung 0a |
| six-wave door #62 | sw_g = 0.38 (open); acts as sin-phasing at head lines ~2187-2204; additive breath_emb (7x512, abs-mean 0.05) at line ~1514 | rung 0a |
| deposit stamps uncalibrated | champion stamp median 412, max 1.9e4; trained ~4; collapse to 0 under the live wire without a floor | stamp_amplitude_read.py |
| the atlas's road | correct readings: cosine-to-road AUC 0.85 mint at step 0; wild z-radius INVERTED (individuation) | needle_read.py |

A rotation is invisible on a growing radius. The sextet has nowhere to turn.

## 1. Invariants (the design in five sentences)

1. **The polar state.** Each loop-state vector (per slot, the `cur` rows of
   `breath_step`) is represented as `r * u`: `u` on the unit sphere of a
   T^256 torus (512 dims = 256 planes, the bus's own geometry), `r` an
   explicit, non-negative radius channel (the consolidation coordinate;
   allowed to grow monotone; DIAGNOSTIC REGISTER — never in a loss).
2. **The guaranteed sextet.** Every loop breath kb = 1..6 rotates the
   designated CLOCK BANDS of `u` by the rotor clock's wheel table
   (`rotor_clock.phase_of` / `wheel_table`: breath hand 60 deg, parity
   120 deg, pass wheel static this era), frozen frequencies, NO gains, NO
   learnable schedule; breath 0 is outside time (`is_clocked`). The same
   rotation is applied Q-side to the attention space on the same bands
   (relative-phase precedent; K unrotated). Content planes are unclocked.
3. **Writes are tangential, reads are polar.** The residual update writes
   into `u` (then re-normalized, where-gated) and into `r` separately; every
   organ that reads the state may read `(r, u)` or `r*u` — bit-for-bit the
   old vector when `ALG_POLAR` is unset.
4. **The deposit's radius.** The garage stamp becomes the deposit's radius
   channel, calibrated to the state's scale (fix B): amplitude can neither
   shout nor vanish; the ALG_PC_FLOOR safety stays.
5. **Birth equivalence.** With `ALG_POLAR` unset the graph is byte-identical
   (eq A/B/C). With it set at birth, `u = cur/||cur||`, `r = ||cur||`, bands
   rotated: the parse emissions may differ from the champion at birth (the
   rotation is unconditional) — this is a NEW GENERATION and is measured as
   a twin, not asserted equivalent.

## 2. Bands (from rotor_clock.py, made live)

- Planes 0..255 of the 512-d direction. `ROT_BANDS` (.cache/rot_band_draft.json:
  mand [3,4,11,27], elec [...]) was drafted per 64-d head (32 pairs); the
  polar state is 512-d (256 pairs). Builder proposes the allocation:
  suggested — breath hand on 32 planes, parity on 16 planes, pass wheel on 8
  (static), the rest content; the SAME plane indices for state and Q-side
  attention so the two clocks are one clock. Frozen at birth; recorded in
  the ledger and in `.cache/polar_bands.json` (read by the head — a reader
  must exist, the bands note's lesson).

## 3. Doors and defaults

`ALG_POLAR=1` (master), `ALG_POLAR_BANDS=path` (default .cache/polar_bands.json),
`ALG_POLAR_R_MODE=scalar|slotvec` (radius channel: one scalar per slot vs a
small vector; default scalar), `ALG_POLAR_QROT=1` (attention-space rotation;
default on under ALG_POLAR), `ALG_POLAR_STAMP=1` (the deposit-radius fix B;
default on). Every door prints its state once at build_params.

## 4. Anchors the builder must map (head = scripts/phase1_algebra_head.py)

- the per-breath state update: `cur = m_c*anchor + (1-m_c)*cur_new` (~line 2014)
  and `breaths.append(cur)` (~2024) — the polar re-parametrization lives here;
- `q_extra = cur + breath_emb[kb]` (~1514) — the additive time signal; keep
  (it is learned and small) but it must not be the only clock;
- the mixer QK (~1872-1892): fed item 7a's Q-side rotation on pairs 24..31 —
  REPLACE with the band rotation from rotor_clock (unconditional, no gains);
- the six-wave phasing (~2187-2204, sw_g) — keep as is (door #62; measured
  open) unless it fights the sextet; builder reports;
- the deposit stamp (~2071-2095: `_wn4`, the floor, the live wire) — fix B;
- the seal (~1587-1634) — the sealed rows' crossing must carry `(r, u)`;
- `breaths_all` (~2346) — the atlas tap must see `r*u` (the old coordinates)
  AND expose `u` (new; keyed `breaths_u`) so the clock read can probe both.

## 5. Proofs before any fire (CPU, zero training)

1. `apply_polar_waist.py --check` idempotent; anchors asserted; ast OK.
2. eq A/B/C bit-identical with `ALG_POLAR` unset (the contract).
3. pc_row_smoke ALL GATES with `ALG_POLAR` unset.
4. A birth smoke with `ALG_POLAR=1` on 8 rows (CPU): finite; `||u|| == 1`
   per slot at every breath (bitwise within tolerance); the clock bands of
   `u` at breath k+1 equal the bands of `u` at breath k rotated by 60 deg
   when the write is zeroed (a unit test of the rotation, not the model);
   `clock_read.py`'s probe on `breaths_u` at birth >= 0.95 (the rotation is
   unconditional, so breath identity must be decodable at birth — if it is
   not, the tap or the rotation is wrong).
5. Gradient contract: two-terminal check that `r` receives NO loss gradient
   from any diagnostic and that `u`'s renormalization is where-gated.

## 6. The twin (needs the word to fire)

Warm from fedon242 (polar at birth from its weights), 12k gentle, seed 242
then 241, vs the champion's plain continuation (pcctl242 / pcctl241 exist).
BARS: breath probe >= 0.95 on both fixtures at 12k (clock_read.py on
`breaths_u`); Procrustes mass in [50,80) >= 0.5; open-wild and mint within
-0.02 (two sigma) of the matching-seed control; sealed-wild reported and
must not read 0.0000 (the channel stays open under ALG_PC_MIX=0.15 +
floor if the cooker rides — decide at fire time, declare once). KILL:
probe < 0.80 at 12k = the parametrization is wrong.

## 7. On deck

THE POINCARÉ BALL: with `r` explicit, the hyperbolic reading (radius =
depth in the kind hierarchy) is one log-map away. The marriage clause
governs: hyperbolic quantities never enter a softmax without a log-map;
cosine is wrong in the ball. Not this generation.
