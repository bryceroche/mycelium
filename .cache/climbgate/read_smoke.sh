# THE CLIMB READ's CPU smoke: scripts/picker/climb_read.py, three bodies:
#   fresh  = balV242 + the sorting room at its seeded fresh init (CR_INIT_MISSING=1, SEED=242) = the untrained floor;
#   sort4  = the K3 sort4 body after 20 CPU steps WITHOUT the climb (the ALG_CLIMB=0 floor);
#   climb  = the K3 climb body after 20 CPU steps (gate.sh phase 5).
# (a) the WILD holdout (wildhold states; the DIAGNOSTIC-ONLY targets, opened by this read alone), rows ${CR_ROWS:-0:48};
# (b) the tiny64 TRAINING rows (all 64; .cache/climb_targets_tiny64.npz) -- the climb CE on FIXED rows (falls?).
source /home/bryce/mycelium/.claude/worktrees/agent-a6fbba480f6d464d0/.cache/climbgate/env.sh
exec > $G/read_smoke.log 2>&1
BODIES=fresh=.cache/sharp_balV242.safetensors,sort4=.cache/climbgate/ckpt_cost20_sort4k3.safetensors,climb=.cache/climbgate/ckpt_cost20_climbk3.safetensors
cd $WT
echo "== (a) WILD rows ${CR_ROWS:-0:48} =="
env PYTHONPATH=$WT DEV=CPU $COMMON $S8 ALG_SORT=4 ALG_LOOP_STOP=3 \
  ALG_TEST=.cache/wild_admitted_holdout.jsonl ALG_TEST_NAME=wildhold CR_ROWS=${CR_ROWS:-0:48} CR_INIT_MISSING=1 \
  CR_CKPTS=$BODIES $PY scripts/picker/climb_read.py
echo "READ SMOKE (a) rc=$?"
echo "== (b) tiny64 TRAINING rows (fixed-row climb CE) =="
env PYTHONPATH=$WT DEV=CPU $COMMON $S8 ALG_SORT=4 ALG_LOOP_STOP=3 \
  CR_SPLIT=train CR_TARGETS=.cache/climb_targets_tiny64.npz CR_INIT_MISSING=1 \
  CR_CKPTS=$BODIES $PY scripts/picker/climb_read.py
echo "READ SMOKE (b) rc=$?"
