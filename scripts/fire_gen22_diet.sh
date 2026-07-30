#!/bin/bash
# fire_gen22_diet.sh — THE GEN-22 DIET FIRE (2026-07-30, on the word).
# One lever: the fdiv-on-derived diet (corpus v3, 400 uniques x10 reps,
# share 4.85%) on gen-21's promoted register. Recipe = fire_gen21.sh
# verbatim (SGDR 4x4k gentle continuation, LR 1e-4, RATION_W=1.75 on the
# unchanged gen21 ration indices — base order preserved by the mix
# builder). ALG_ALLOW_PEN_TRAIN=1: gen21_mix carries 4,810 pen rows,
# lawful diet members (custody law; supervision from states-file gold).
# Pre-flight per the word: states memmap verified against a live trunk
# forward on base AND diet rows BEFORE any segment burns.
# After-reads chained: diet after-fixture (before-baseline's exact
# probes) + the standing rehearsal, both tiers.
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_FTYPES=8 ALG_DUP=1 ALG_ALLOW_PEN_TRAIN=1
PY=.venv/bin/python3
MIX=.cache/gen22_mix.jsonl
echo "=== G22 0/6: build mix (dose declared) ==="
$PY scripts/build_gen22_mix.py
echo "=== G22 1/6: precompute ==="
ALG_TRAIN=$MIX ALG_TRAIN_NAME=g22 PRECOMPUTE_ONLY=g22 $PY scripts/phase1_algebra_head.py --precompute
echo "=== G22 1.5/6: states verification (live forward vs memmap) ==="
ALG_TRAIN=$MIX $PY - << 'EOF'
import sys, os, json
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from phase1_algebra_head import T_ALG, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer
rows = [json.loads(l) for l in open(".cache/gen22_mix.jsonl")]
st = np.load(".cache/phase1_alg_states_g22_states.npy", mmap_mode="r")
z = np.load(".cache/phase1_alg_states_g22.npz")
assert st.shape[0] == len(rows) == 82400, (st.shape, len(rows))
assert z["tokmask"].shape[0] == len(rows)
tok = Tokenizer.from_file(TOKENIZER_JSON)
picks = [0, 40000, 78399, 78400, 80000, 82399]   # base head/mid/tail + diet head/mid/tail
ids = np.zeros((8, T_ALG), np.int32); msk = np.zeros((8, T_ALG), np.float32)
for i, ridx in enumerate(picks):
    e = tok.encode(rows[ridx]["text"]); L = min(len(e.ids), T_ALG)
    ids[i, :L] = e.ids[:L]; msk[i, :L] = 1.0
live = recompute_states(ids).astype(np.float32)
worst = 0.0
for i, ridx in enumerate(picks):
    m = msk[i] > 0
    a = live[i][m]; b = np.asarray(st[ridx], np.float32)[m]
    rel = float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-6))
    cos = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))
    worst = max(worst, rel)
    print(f"  row {ridx} ({'base' if ridx < 78400 else 'DIET'}): max-rel-dev {rel:.5f}  cos {cos:.6f}")
    assert cos > 0.9999, f"states mismatch at row {ridx}"
print(f"[verify] 6/6 rows match (worst rel dev {worst:.5f}, fp16 storage) — states TRUSTED")
EOF
for seg in 1 2 3 4; do
  echo "=== G22 seg $seg/4 (RATION_W=1.75 hot-phase) ==="
  if [ $seg -eq 1 ]; then W="WARM_FROM=.cache/g21.safetensors"; else W="RESUME=1"; fi
  env $W ALG_TRAIN=$MIX ALG_TRAIN_NAME=g22 ALG_CKPT=.cache/g22.safetensors STEPS=4000 LR=1e-4 BATCH=8 SEED=8$seg SNAP_EVERY=500 RATION_FILE=.cache/gen21_ration_idx.json RATION_W=1.75 $PY scripts/phase1_algebra_head.py --train
  for st in 500 1000 1500 2000 2500 3000 3500 4000; do
    mv .cache/g22_s${st}.safetensors .cache/g22_seg${seg}_s${st}.safetensors 2>/dev/null || true
  done
done
echo "=== G22 6/6: AFTER-READS ==="
DIET_CKPT=.cache/g22.safetensors $PY scripts/diet_after_read.py
WFF_CKPT=.cache/g22.safetensors WFF_OUT=.cache/wild_frontier_fixture_g22.json $PY scripts/wild_frontier_fixture.py
echo "=== THE FIRE IS BURNED — g22 candidate on the bench (battery + verdict next; g21 remains the gate until PROMOTED) ==="
