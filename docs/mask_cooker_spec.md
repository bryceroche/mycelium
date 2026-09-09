# THE MASK COOKER — spec (2026-09-08, registered by the organ census; run needs the word)

**Owner:** Bryce + Claude · **Status:** REGISTERED, unbuilt · **Law:** the mandatory-road
law (ledger 2026-09-08) · **Template:** the pressure cooker (apply_pressure_mix.py: a per-row
data buffer, stable index-hash assignment, one JIT graph, flat mix, dose declared).

## 0. Why (the census)

The mask head — 2.0M params, "the player", the dynamic-attention organ the two-jaws thesis
rests on — injects at ~1e-5x of the slot-score band on both machines and both fixtures
(post-gain 4e-5..1e-4 vs raw sc2 2.4-5.8); its PRE-gain organ is at 1-9% of the band; its
gain fell 0.02 -> 0.0011 -> 0.0002 across generations while in the polar arm its pre-gain
output GREW 2-5x. The organ is trying to speak; the loss turns it down. Three structural
reasons, all in the code: (1) two zero-born output doors (mh_wo, mh_headmix — the ResNet
law) that the loss never grew; (2) a gain the loss can and did close; (3) the OPEN-ONLY
constitution as implemented: `_mb = gain * (softplus(raw) - softplus(0)) * _sm_kb` — the
head can only re-weight lanes the baseline already opened, and the baseline opens 62-67%
of them (flag audit) — so the head has almost nothing to add, and the loss learns it has
nothing to add. The residual (notebook ~1.2x, garage ~0.4x) carries deliberation.

## 1. The design (the cooker's logic applied to attention)

On a SHARE of training rows (dose declared; opening dose 0.15 — the Goldilocks precedent —
with the floor idiom's stable Knuth-hash assignment; the _PCV pattern as a (B,1,1) buffer
`_MCV`), THE BASELINE MASK IS SEVERED: `_sm_kb` for sealed rows becomes the identity (self
lanes only). On those rows the mask head's lanes are THE ONLY ROAD: its raw logits enter as
a soft gate over ALL lanes, `sc2 + log(sigmoid(raw))`, ungated by `mh_gain`, unmultiplied
by the (now-identity) baseline — a lane opens where the head says open, closes where it
says closed. Open rows keep today's code path bit-for-bit (the additive, open-only,
gain-scaled `_mb`). Birth: raw ≈ 0 -> sigmoid 0.5 -> a uniform half-open field on sealed
rows (log 0.5 added everywhere = attention over all lanes at once): a cold but FUNCTIONAL
start, not a dead one; the loss sharpens the field. The constitution's worry (a learned
mask that silences its own inputs to hide mistakes) is answered by the pressure itself: a
sealed row with no lanes cannot be parsed, so closing lanes costs loss — the same argument
that made the pressure cooker lawful (a capability target, not diagnostic supervision).

## 2. Doors and mechanics

`ALG_MASK_COOK=<share>` (0 = off = byte-identical), `ALG_MASK_COOK_SKEL=self|sentence`
(the severed baseline's skeleton; default self), `ALG_MASK_COOK_LIVE=1` (the gate's raw
logits carry gradient into the mask head, its doors and its context encoder — the point).
The seal applies at every breath of the loop for sealed rows (the mask is per-breath). The
mixer's `_mx_sc + _mb` path: on sealed rows use the same log-gate (one mask, one road —
the polar B6 lesson: two consumers of one tensor, rotate/gate each exactly once). The
sealed rows' val/reads are excluded by the existing SC_EVAL="0"-style push (val compares
the OPEN regime), and a NEW meter is added: **mask-sealed-wild** — loop_val with the
baseline severed for ALL rows (env `ALG_MASK_SEAL=1` at read): parse accuracy carried by
the mask head's lanes alone. The champion reads ~0 there by construction (an identity
baseline + a 1e-5x additive head = no lanes); that 0 is the baseline the cooker opens.

## 3. Proofs before the fire (CPU)

--check + idempotence; env-inert bit-identity (staged vs applied, ALG_MASK_COOK unset, both
polar configs); per-row surgery bitwise (sealed rows differ, open rows bit-identical to the
all-open run — the pc_row_smoke GATE 2 idiom); the birth field: on a sealed row at raw=0
the attention equals the uniform-over-all-lanes attention (unit test); two-terminal: the
mask head's params (mh_wq/wk/wv, mh_wo, mh_headmix, the context encoder) receive gradient
from sealed rows and the OPEN rows' gradient path is unchanged; the seal of the pressure
cooker and the mask seal compose (both per-row buffers, independent hashes: a row may be
sealed by either, both, or neither — assert the four cells exist in a batch of 8).

## 4. The twin (needs the word)

Warm from polarsink242 (the research champion candidate: the clock kept, the sink on), 12k
gentle, seed 242, ALG_MASK_COOK=0.15, vs the polarsink242 plain continuation (a control
arm — polarsink242 continued 12k with the cooker off — must be fired alongside, seed 242).
BARS: (1) THE ROAD OPENS: mask-sealed-wild >= 0.05 (from ~0) and mask-sealed-mint reported;
(2) THE ORGAN CARRIES: the census on sealed rows shows the mask head's gate at >= 0.1x of
the score band (pre AND post); (3) GUARD: open-wild and mint within -0.02 of the control
(two sigma); (4) the clock stays: probe >= 0.95 on breaths_u. KILL: open-wild < control -
0.02 = the mask road costs the residual road; dose is then the first dial (0.05).
Follow-ups registered, not bundled: the FACT COOKER (altfact 0.02x: sever the var-slot
state's fact road? — no: the fact injection is additive into vst; the analogous pressure is
the existing pressure cooker's seal, which already forces facts through the shelf) and the
MIXER (2%): prune-or-cook decided by an ablation, not by this run.
