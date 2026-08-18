#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
PY=.venv/bin/python3
echo "== CATHEDRAL TRIGGER: per-breath profile, K5@74k =="
env DEV=AMD ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_SEPHASE=1 ALG_SIXWAVE=1 ALG_BREATH=5 BREATH_NORM=1 ALG_DEEPSUP=1 \
  PR_CK=.cache/gk5_arm.safetensors ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest $PY scripts/breath_profile.py
echo "== CENSUSES for the multi-hop bucketing =="
env DEV=AMD ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_SEPHASE=1 ALG_SIXWAVE=1 ALG_BREATH=5 BREATH_NORM=1 \
  CENSUS_CKPT=.cache/gk5_arm.safetensors CENSUS_OUT=.cache/mc_k5_74.json ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest $PY scripts/miss_census.py 2>/dev/null | grep census
env DEV=AMD ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_SIXWAVE=1 ALG_BREATH=3 BREATH_NORM=1 \
  CENSUS_CKPT=.cache/gnat_native.safetensors CENSUS_OUT=.cache/mc_k3_74.json ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest $PY scripts/miss_census.py 2>/dev/null | grep census
$PY - << 'PEOF'
import json
import numpy as np
rows=[json.loads(l) for l in open('.cache/algebra_nl_bigtest.jsonl')]
nf=np.array([len(r["factors"]) for r in rows]); qs=np.quantile(nf,[0.25,0.5,0.75])
m5=set(json.load(open('.cache/mc_k5_74.json'))["miss_idx"]); m3=set(json.load(open('.cache/mc_k3_74.json'))["miss_idx"])
print("quartile | K5 solve | K3 solve | margin")
for tag,lo,hi in (("Q1",0,qs[0]),("Q2",qs[0],qs[1]),("Q3",qs[1],qs[2]),("Q4",qs[2],99)):
    idx=[i for i in range(len(rows)) if (nf[i]>lo if lo>0 else True) and nf[i]<=hi]
    r5=sum(1 for i in idx if i not in m5)/len(idx); r3=sum(1 for i in idx if i not in m3)/len(idx)
    print(f"  {tag}    |  {r5:.3f}  |  {r3:.3f}  | {r5-r3:+.3f}")
PEOF
echo "== READS74 COMPLETE =="
