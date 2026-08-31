# The Parameter Ledger (2026-09-01 — genesis-era exact counts)

Counted directly from `build_params()` under the genesis configs
(H_W=512, 8 heads, BINDBUS=7, D=512). The stale "3.2M" folk number is
struck (conviction-index): the head grew through the eras to the
figures below. Trained params only — the 1.24B trunk (243M layers +
263M embeddings) is frozen, constitutional, run once per problem.

| organ | plain twin | stack twin | notes |
|---|---:|---:|---|
| bank attention + FFN | 2,113,024 | 2,113,024 | slot→token reads; FFN is 1× expansion (starvation-audit flag) |
| garage (shelf attn + read) | — | 1,310,721 | W_gq + W_busr + bus_g; the canonical shelf's reader |
| slot mixer | 1,050,624 | 1,050,624 | slot↔slot relations; SINGLE-headed (starvation-audit flag) |
| waist encoder | 1,049,088 | 1,049,088 | trunk 2048 → 512; one linear map (starvation-audit flag) |
| pointer heads (the artery) | 786,432 | 786,432 | W_args/W_res/W_query; single bilinear forms (flag) |
| notebook + breath embeds | 527,879 | 527,879 | the one full-voice organ (power ecology) |
| bind MLP (emission + garage writes) | 524,800 | 524,800 | W_bind1/2 — the bus's wire maker |
| other | 278,528 | 278,528 | letter/sephase machinery etc. |
| router | — | 102,913 | W_rs (snap-cond 73→512) + W_ra/W_rb (64-d) + r_gain |
| classifier/digit heads | 81,567 | 81,567 | ftype/pres/islit/dig/dup/sgn |
| var/query banks + gates | 12,801 | 12,801 | vq + sw_g etc. |
| determination wave | — | 1,537 | W_det (3→512) + det_g |
| alternator gain | — | 1 | alt_g |
| **TOTAL** | **6,424,743** | **7,839,915** | stack organs add 1,415,172 (+22.0%) |

Fractions of the full system: plain 0.52%, stack 0.63% of the ~1.24B
stack. Deliberation cost: ~2% of one trunk forward per 7-breath cycle.

## The capacity roadmap (registered)
- **Genesis** (burning next): these exact twins, 50k from noise, funhouse
  diet, seed 241 — skeleton frozen for baseline comparability.
- **The capacity curve** (the sequel, on genesis's winner): base (~6.4M)
  vs the fed mind (~8M package: FFN→4×, gated 2-layer waist, multi-form
  pointers, fed ink, mixer heads free) vs the 20M mind (+2nd bank layer
  per breath, deep everything). The slope is the deliverable; the
  register gap is the memorization tell.
