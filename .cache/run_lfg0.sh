#!/usr/bin/bash
# G0 THE ENGINE CONTROL (ledger 11-04 07:42 pin): BT4M8 config, seed 0, 18k trained on tinygrad / the GPU (gsm_breath_tg.py TG_MODE=train),
# scored by the torch scorer on copied-back weights; vs the torch BT4M8_s0 (TEST step 0.158, ORIG step 0.962; bars +-0.010 each).
set -eo pipefail; cd /home/bryce/mycelium; exec >> .cache/run_lfg0.log 2>&1
echo "== start $(date +%H:%M) =="; PY=.venv/bin/python3
export RD_STATES=gsmb_mbr8_states.npy BT_ROLEF=gsmb_role_feats_mbr8.npy; B="BT_D=512 BT_NH=8 BT_DB=64 BT_CYCLOCK=1 BT_ZOOM=1 BT_ROLE=1 BT_T6OFF=1"
flock -w 36000 .cache/gpu.lock env $B DEV=PCI+AMD TG_MODE=train BT_SEED=0 BT_STEPS=18000 BT_SAVE=1 BT_TAG=BT4M8TG_s0 TOY_THREADS=4 \
  $PY scripts/gsm_breath_tg.py 2>&1 | grep --line-buffered -v -i warn
$PY - <<'PY'
import json
g, t = (json.load(open(f".cache/gsmb_breath_{a}_s0.json")) for a in ("BT4M8TG", "BT4M8"))
dt, do = g["TEST_step_all"] - t["TEST_step_all"], g["ORIG_step_all"] - t["ORIG_step_all"]
ok = abs(dt) <= 0.010 and abs(do) <= 0.010
L = (f"G0: TEST step {g['TEST_step_all']:.4f} vs torch {t['TEST_step_all']:.4f} ({dt:+.4f}; +-0.010), ORIG step {g['ORIG_step_all']:.4f} "
     f"vs {t['ORIG_step_all']:.4f} ({do:+.4f}; +-0.010) -> {'PASS' if ok else 'FAIL'}\n"
     + "  beside: " + " | ".join(f"{m} tg {g[m]:.4f} torch {t[m]:.4f}" for m in ("TEST_type_ok", "CFNUM_step_all", "CFNAME_step_all", "TEST_ans")))
print(L); open(".cache/gsm_g0_verdict.txt", "w").write(L + "\n")
PY
echo "== done $(date +%H:%M) =="
