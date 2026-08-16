#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_FTYPES=8 ALG_HW=512 ALG_DUP=1 ALG_WIDE=1 ALG_INV=1 ALG_BREATH=3
export ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest
PY=.venv/bin/python3
echo "== RELAY READ pass-0 (straight) =="
env CENSUS_CKPT=.cache/g51_whisper.safetensors CENSUS_OUT=.cache/mc_g51_p0.json $PY scripts/miss_census.py 2>/dev/null | grep census
echo "== RELAY READ pass-1 (TWO_PASS: breaths engaged) =="
env TWO_PASS=1 CENSUS_CKPT=.cache/g51_whisper.safetensors CENSUS_OUT=.cache/mc_g51_p1.json $PY scripts/miss_census.py 2>/dev/null | grep census
$PY - << 'PEOF'
import json
p0=set(json.load(open('.cache/mc_g51_p0.json'))["miss_idx"])
p1=set(json.load(open('.cache/mc_g51_p1.json'))["miss_idx"])
rc=json.load(open('.cache/residue_census.json'))["rows"]
ref=[r["idx"] for r in rc if r["mode"]=="unforced"]; fw=[r["idx"] for r in rc if r["mode"]=="wrong"]
census=json.load(open('.cache/miss_census_gen41.json'))
conv=[i for i in census["miss_idx"] if i not in set(r["idx"] for r in rc)]
for tag,pop in (("refusal-65",ref),("wrong-9",fw),("converted-159",conv)):
    inb=[i for i in pop if i in p0]
    cv=[i for i in inb if i not in p1]
    print(f"[relay {tag:13s}] in-p0-miss {len(inb)}  RELAY-CONVERTS {len(cv)}  rate {len(cv)/max(len(inb),1):.3f}")
reg=len(p1-p0)
print(f"[relay] global: p0 misses {len(p0)}  p1 misses {len(p1)}  regressions {reg}")
PEOF
echo "== RELAY READ COMPLETE =="
