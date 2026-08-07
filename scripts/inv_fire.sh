#!/bin/bash
# inv_fire.sh — THE INVARIANCE FIRE (2026-08-07; registered dd52e4eda739
# BEFORE mint/trainer existed; door customer #3). Two arms on the PAIR
# mix, cont-control: only delta = the agreement term. BENCH fire;
# nothing promotes; g22-lineage remains the gate.
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_FTYPES=8 ALG_DUP=1 ALG_ALLOW_PEN_TRAIN=1
PY=.venv/bin/python3
MIX=.cache/inv_pairs.jsonl
echo "=== INV: states for the pair mix (13,080 rows, full trunk pass) ==="
$PY - << 'PYEOF'
import sys, os, json
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
os.environ.setdefault("ALG2","1"); os.environ.setdefault("ALG_FTYPES","8")
os.environ.setdefault("ALG_HW","512"); os.environ.setdefault("ALG_DUP","1")
import numpy as np
from phase1_algebra_head import T_ALG, sent_indices, TOKENIZER_JSON, build_gold
import phase1_algebra_head as PH
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer
tok = Tokenizer.from_file(TOKENIZER_JSON)
mixp = ".cache/inv_pairs.jsonl"
rows = [json.loads(l) for l in open(mixp)]
n = len(rows)
npyp = ".cache/phase1_alg_states_invpairs_states.npy"
out = np.lib.format.open_memmap(npyp, mode="w+", dtype=np.float16, shape=(n, T_ALG, 2048))
for s0 in range(0, n, 8):
    ids = np.zeros((8, T_ALG), np.int32)
    ch = rows[s0:s0+8]
    for i, r in enumerate(ch):
        e = tok.encode(r["text"]); L = min(len(e.ids), T_ALG)
        ids[i, :L] = e.ids[:L]
    st = recompute_states(ids).astype(np.float16)
    out[s0:s0+len(ch)] = st[:len(ch)]
    if (s0 // 8) % 100 == 0: print(f"  [states {s0}/{n}]", flush=True)
out.flush(); del out
samples, ids2, mask, offsets = PH.tokenize(mixp)
gold = build_gold(samples, offsets)
sent = np.stack([sent_indices(s["text"], o, mask[i]) for i, (s, o) in enumerate(zip(samples, offsets))])
np.savez(".cache/phase1_alg_states_invpairs.npz", tokmask=mask.astype(np.uint8),
         sent=sent.astype(np.int8), **{f"g_{k}": v for k, v in gold.items()})
print("[inv] states + gold staged", flush=True)
PYEOF
SEED=102
for ARM in ctl inv; do
  SEED=$((SEED+1))
  if [ "$ARM" = "inv" ]; then export ALG_INV=1; else export ALG_INV=0; fi
  echo "=== INV FIRE [$ARM]: 4x4k from g22 (pair mix; delta = agreement term only) ==="
  for seg in 1 2 3 4; do
    if [ $seg -eq 1 ]; then W="WARM_FROM=.cache/g22.safetensors"; else W="RESUME=1"; fi
    env $W ALG_TRAIN=$MIX ALG_TRAIN_NAME=invpairs ALG_CKPT=.cache/g25_inv_${ARM}.safetensors \
        STEPS=4000 LR=1e-4 BATCH=8 SEED=${SEED}${seg} SNAP_EVERY=2000 \
        $PY scripts/phase1_algebra_head.py --train
  done
done
echo "=== THE INVARIANCE FIRE IS BURNED — reads next; g22 REMAINS THE GATE ==="
