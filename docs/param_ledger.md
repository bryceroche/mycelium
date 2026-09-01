# The Parameter Ledger (2026-09-01 — genesis-era counts, verified)

Counted from `build_params()` under the genesis configs (H_W=512,
8 heads, BINDBUS=7, D=512), verified by full per-tensor enumeration
(66 tensors, zero unaccounted). Rounded to the nearest 100k; exact
totals at the foot. The stale "3.2M" folk number is struck. Trained
params only — the 1.24B trunk is frozen, constitutional, run once.

VERIFICATION CORRECTIONS (2nd pass): (1) the FFN is 2× expansion
(512→1024→512), not the 1× first claimed — still under the 4×
convention, but half the gap; (2) the macro value system (W_y + dig2)
was hiding in "other" — now named.

| organ | plain | stack | notes |
|---|---:|---:|---|
| bank attention + FFN | 2.1M | 2.1M | slot→token reads; FFN at 2× (convention 4× — audit flag, softened) |
| garage (shelf attn + read) | — | 1.3M | W_gq (0.3M) + W_busr (1.0M) |
| slot mixer | 1.1M | 1.1M | slot↔slot relations; SINGLE-headed (audit flag) |
| waist encoder | 1.0M | 1.0M | trunk 2048→512 via ONE linear map (audit flag) |
| pointer heads (the artery) | 0.8M | 0.8M | W_args/W_res/W_query — single bilinear forms each (audit flag) |
| notebook + breath embeds | 0.5M | 0.5M | the one full-voice organ (power ecology) |
| bind MLP (emission + garage writes) | 0.5M | 0.5M | W_bind1/2 — the bus's wire maker |
| macro value system | 0.3M | 0.3M | W_y + h_dig2 (macro/second-value machinery) |
| router | — | 0.1M | W_rs (snap-cond) + W_ra/W_rb (64-d) + r_gain |
| classifier + digit heads | 0.1M | 0.1M | ftype/pres/islit/dup/sgn/sel + h_dig |
| banks, gates, wave, misc | 0.0M (~30k) | 0.0M (~30k) | vq/fq/qq, sw_g, W_det (1.5k), gains |
| **TOTAL** | **6.4M** | **7.8M** | stack organs add 1.4M (+22%) |

Exact: plain 6,424,743 · stack 7,839,915 (66 tensors). Fractions of
the full system: 0.52% / 0.63% of the ~1.24B stack. Deliberation cost:
~2% of one trunk forward per 7-breath cycle.

## The capacity roadmap (registered)
- **Genesis** (next): these exact twins, 50k from noise, funhouse diet,
  seed 241 — skeleton frozen for baseline comparability.
- **The capacity curve** (on genesis's winner): base (6.4M) vs the fed
  mind (~8M: FFN 2×→4×, RESIDUAL 2-layer waist, multi-form pointers,
  fed ink, mixer heads free, +8 scratch slots) vs the 20M mind (+2nd
  bank layer per breath — additive, zero-init projection, identity
  path untouched — deep everything, +16 scratch slots). Slot-scaling
  doctrine (ledger 2026-08-31): var/factor slots FROZEN at 24 across
  the curve; scratch scales with tier (0/8/16). The slope is the
  deliverable; the register gap is the memorization tell.
