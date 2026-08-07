#!/bin/bash
# inv_fire_v2.sh — THE INVARIANCE FIRE v2 (2026-08-07; substrate fixed:
# pairs EMBEDDED in the full mix; registration dd52e4eda739 stands, bars
# unmet-not-tested from v1). Assembly = patch A-changed + append B rows
# onto banked g22 base states; sentinels; then ctl/inv arms.
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_FTYPES=8 ALG_DUP=1 ALG_ALLOW_PEN_TRAIN=1
PY=.venv/bin/python3
MIX=.cache/inv2_mix.jsonl
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
tok=Tokenizer.from_file(TOKENIZER_JSON)
rows=[json.loads(l) for l in open(".cache/inv2_mix.jsonl")]
ch=json.load(open(".cache/augfire_invA_changed.json"))
need=sorted(ch)+list(range(82400,len(rows)))   # A-changed + all B-appended
base=np.load(".cache/phase1_alg_states_g22_states.npy", mmap_mode="r")
npyp=".cache/phase1_alg_states_inv2_states.npy"
out=np.lib.format.open_memmap(npyp, mode="w+", dtype=np.float16, shape=(len(rows),T_ALG,2048))
CH=4096
for s0 in range(0, base.shape[0], CH):
    out[s0:min(s0+CH,base.shape[0])]=base[s0:min(s0+CH,base.shape[0])]
for s0 in range(0,len(need),8):
    idxs=need[s0:s0+8]
    ids=np.zeros((8,T_ALG),np.int32)
    for i,ridx in enumerate(idxs):
        e=tok.encode(rows[ridx]["text"]); Ln=min(len(e.ids),T_ALG)
        ids[i,:Ln]=e.ids[:Ln]
    st=recompute_states(ids).astype(np.float16)
    for i,ridx in enumerate(idxs): out[ridx]=st[i]
    if (s0//8)%150==0: print(f"  [patch {s0}/{len(need)}]",flush=True)
out.flush(); del out
samples, ids2, mask, offsets = PH.tokenize(".cache/inv2_mix.jsonl")
gold=build_gold(samples, offsets)
sent=np.stack([sent_indices(s["text"],o,mask[i]) for i,(s,o) in enumerate(zip(samples,offsets))])
np.savez(".cache/phase1_alg_states_inv2.npz", tokmask=mask.astype(np.uint8),
         sent=sent.astype(np.int8), **{f"g_{k}":v for k,v in gold.items()})
st=np.load(npyp, mmap_mode="r")
picks=[0, need[0], need[len(need)//2], need[-1], 40000, len(rows)-1]
ids3=np.zeros((8,T_ALG),np.int32); msk=np.zeros((8,T_ALG),np.float32)
for i,ridx in enumerate(picks):
    e=tok.encode(rows[ridx]["text"]); Ln=min(len(e.ids),T_ALG)
    ids3[i,:Ln]=e.ids[:Ln]; msk[i,:Ln]=1.0
live=recompute_states(ids3).astype(np.float32)
for i,ridx in enumerate(picks):
    m_=msk[i]>0; a=live[i][m_]; b=np.asarray(st[ridx],np.float32)[m_]
    cos=float((a*b).sum()/(np.linalg.norm(a)*np.linalg.norm(b)))
    assert cos>0.9999, f"SENTINEL FAIL {ridx} {cos}"
print("[inv2] sentinels 6/6 — assembly TRUSTED",flush=True)
PYEOF
SEED=104
for ARM in ctl inv; do
  SEED=$((SEED+1))
  if [ "$ARM" = "inv" ]; then export ALG_INV=1 INV_PAIRS=.cache/inv2_pairs.npy; else export ALG_INV=0; fi
  echo "=== INV2 [$ARM]: 4x4k from g22 ==="
  for seg in 1 2 3 4; do
    if [ $seg -eq 1 ]; then W="WARM_FROM=.cache/g22.safetensors"; else W="RESUME=1"; fi
    env $W ALG_TRAIN=$MIX ALG_TRAIN_NAME=inv2 ALG_CKPT=.cache/g25v2_inv_${ARM}.safetensors \
        STEPS=4000 LR=1e-4 BATCH=8 SEED=${SEED}${seg} SNAP_EVERY=2000 \
        $PY scripts/phase1_algebra_head.py --train
  done
done
echo "=== INV2 BURNED — reads next; g22 REMAINS THE GATE ==="
