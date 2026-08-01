#!/bin/bash
# aug_fire.sh — THE AUGMENTATION FIRE (2026-08-01, the word given).
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_FTYPES=8 ALG_DUP=1 ALG_ALLOW_PEN_TRAIN=1
PY=.venv/bin/python3
SEED=97
for ARM in vlow vfull; do
  SEED=$((SEED+1))
  MIX=.cache/augfire_${ARM}_mix.jsonl
  CKPT=.cache/g24_${ARM}.safetensors
  echo "=== AUG [$ARM]: assemble (patch changed rows onto banked base states) ==="
  $PY - << PYEOF
import sys, os, json
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
os.environ.setdefault("ALG2","1"); os.environ.setdefault("ALG_FTYPES","8")
os.environ.setdefault("ALG_HW","512"); os.environ.setdefault("ALG_DUP","1")
import numpy as np
from phase1_algebra_head import T_ALG, sent_indices, TOKENIZER_JSON, build_gold
import phase1_algebra_head as PH
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer
ARM="${ARM}"
tok = Tokenizer.from_file(TOKENIZER_JSON)
mixp=f".cache/augfire_{ARM}_mix.jsonl"
rows=[json.loads(l) for l in open(mixp)]
changed=json.load(open(f".cache/augfire_{ARM}_changed.json"))
base=np.load(".cache/phase1_alg_states_g22_states.npy", mmap_mode="r")
npyp=f".cache/phase1_alg_states_g24{ARM}_states.npy"
out=np.lib.format.open_memmap(npyp, mode="w+", dtype=np.float16, shape=base.shape)
CH=4096
for s0 in range(0, base.shape[0], CH):
    out[s0:min(s0+CH,base.shape[0])]=base[s0:min(s0+CH,base.shape[0])]
for s0 in range(0, len(changed), 8):
    idxs=changed[s0:s0+8]
    ids=np.zeros((8,T_ALG),np.int32)
    for i,ridx in enumerate(idxs):
        e=tok.encode(rows[ridx]["text"]); Ln=min(len(e.ids),T_ALG)
        ids[i,:Ln]=e.ids[:Ln]
    st=recompute_states(ids).astype(np.float16)
    for i,ridx in enumerate(idxs): out[ridx]=st[i]
    if (s0//8)%100==0: print(f"  [patch {s0}/{len(changed)}]", flush=True)
out.flush(); del out
samples, ids2, mask, offsets = PH.tokenize(mixp)
gold=build_gold(samples, offsets)
sent=np.stack([sent_indices(s["text"],o,mask[i]) for i,(s,o) in enumerate(zip(samples,offsets))])
np.savez(f".cache/phase1_alg_states_g24{ARM}.npz", tokmask=mask.astype(np.uint8),
         sent=sent.astype(np.int8), **{f"g_{k}":v for k,v in gold.items()})
st=np.load(npyp, mmap_mode="r")
picks=[0, changed[0], changed[len(changed)//2], changed[-1], 40000, len(rows)-1]
ids3=np.zeros((8,T_ALG),np.int32); msk=np.zeros((8,T_ALG),np.float32)
for i,ridx in enumerate(picks):
    e=tok.encode(rows[ridx]["text"]); Ln=min(len(e.ids),T_ALG)
    ids3[i,:Ln]=e.ids[:Ln]; msk[i,:Ln]=1.0
live=recompute_states(ids3).astype(np.float32)
for i,ridx in enumerate(picks):
    m_=msk[i]>0; a=live[i][m_]; b=np.asarray(st[ridx],np.float32)[m_]
    cos=float((a*b).sum()/(np.linalg.norm(a)*np.linalg.norm(b)))
    assert cos>0.9999, f"SENTINEL FAIL {ridx} {cos}"
print(f"[{ARM}] sentinels 6/6 — assembly TRUSTED", flush=True)
PYEOF
  echo "=== AUG [$ARM]: fire (4x4k from g22, seed ${SEED}) ==="
  for seg in 1 2 3 4; do
    if [ $seg -eq 1 ]; then W="WARM_FROM=.cache/g22.safetensors"; else W="RESUME=1"; fi
    env $W ALG_TRAIN=$MIX ALG_TRAIN_NAME=g24${ARM} ALG_CKPT=$CKPT STEPS=4000 LR=1e-4 BATCH=8 SEED=${SEED}${seg} SNAP_EVERY=2000 $PY scripts/phase1_algebra_head.py --train
  done
  rm -f .cache/phase1_alg_states_g24${ARM}_states.npy .cache/phase1_alg_states_g24${ARM}.npz
  echo "=== [$ARM] trained + states cleaned ==="
done
echo "=== THE AUG FIRE IS BURNED — after-reads next; g22 REMAINS THE GATE ==="
