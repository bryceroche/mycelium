#!/usr/bin/bash
# THE TWO-ARM RUN's CPU STAGING (2026-10-17): arm A's split (realBP = form_mix_real rows of formpm35cBP) and the name views'
# SOURCE split (realPBsrcBP, the source rows of the 16,813 views, for precompute_masked's shape/tokmask contract). Gated by
# scripts/diet_stage.py (text, tokmask, sent, gold bit-identical); states sliced, never recomputed.
set -eo pipefail; cd /home/bryce/mycelium; exec > .cache/stage_real.log 2>&1
E="DEV=CPU ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_BREATH=7 ALG_NOTEBOOK=1 ALG_SIXWAVE=1 NB_PERSLOT=1 ALG_BINDBUS=7 ALG_BIND_D=512 BIND_CODES=.cache/bindbus_codes512r.npz ALG_SPAN_ALL=1 ALG_SPAN_ARGS=1 ALG_SPAN_OP=1 ALG_SPAN_RCUE=1 ALG_SPAN_ARCUE=1"
[ -f .cache/phase1_alg_states_realBP_states.npy ] || env $E .venv/bin/python3 scripts/diet_stage.py realBP .cache/form_mix_real.jsonl formpm35cBP@.cache/form_mix_pm35c.jsonl@.cache/form_mix_real.idx.npy
[ -f .cache/phase1_alg_states_realPBsrcBP_states.npy ] || env $E .venv/bin/python3 scripts/diet_stage.py realPBsrcBP .cache/form_real_PBsrc.jsonl formpm35cBP@.cache/form_mix_pm35c.jsonl@.cache/form_real_PBsrc.idx.npy
echo "STAGE REAL COMPLETE ($(date +%H:%M))"
