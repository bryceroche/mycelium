#!/bin/bash
# v40 — Big architectural pivot from v39's A=8 collapse.
#
# Three changes from v39:
#   1. BREATHE_FRESH_INPUT=1  — each breath sees fresh embedding + REPLACE notebook
#                                context. NO rep-flow chain. Fixes A=8 collapse.
#   2. NOTEBOOK_V24=1 + NOTEBOOK_DUAL=1 + NOTEBOOK_ACCUMULATE_ENABLED=0
#                                — REPLACE notebook ONLY (RNN-style memory channel).
#   3. N_LOOKUP_ENTRIES=256    — 4 ops × 4 layers × 16 heads, initialized from
#                                per-head 64d outputs extracted from v38 step 1500
#                                on CORRECT examples, projected to 1024d via
#                                W_O column blocks.
#
# Keep from v39:
#   - BFIELD_WAIST=512 BFIELD_END_OF_BREATH=1 BFIELD_ENFORCED=1
#   - BFIELD_SIN_MOD=1 (heartbeat across breaths)
#   - BFIELD_AUX_WEIGHT=0.1 (op-CE on 512d compressed)
#
# Cold-start ARITH from Pythia.
set -e
cd "$(dirname "$0")/.."

# v40 fresh-input + REPLACE notebook (the A=8 fix)
export BREATHE_FRESH_INPUT=1
export NOTEBOOK_V24=1
export NOTEBOOK_DUAL=1
export NOTEBOOK_ACCUMULATE_ENABLED=0
export NOTEBOOK_INIT_SCALE=0.02

# v40 256-entry lookup with per-(op, layer, head) centroids
export N_LOOKUP_ENTRIES=256
export LOOKUP_VALUE_INJECT=1
export LOOKUP_VALUES_INIT_PATH=.cache/per_op_layer_head_centroids/centroids_per_op_layer_head_n256.npy
export LOOKUP_TEMP=20
export LOOKUP_VALUE_SCALE=1.0

# v39 photon-mode B-field (kept)
export BFIELD_WAIST=512
export BFIELD_END_OF_BREATH=1
export BFIELD_ENFORCED=1
export BFIELD_SIN_MOD=1
export BFIELD_AUX_WEIGHT=0.1

# v23a frozen per-head pitch (kept from baseline)
export PER_HEAD_PITCH=1

# Training
export DEV=PCI+AMD
export LEVEL=ARITH
export SPACE_DIGITS=1
export BATCH=16
export FIXED_LEN=32
export STEPS=1500
export LR=3e-5
export TRAIN_LOOPS=1,2,4
export EVAL_LOOPS=1,4,8
export ACC_EVAL_EVERY=250
export CKPT_EVERY=250
export LOOKUP_AUX_WEIGHT=0.1
export USE_JIT=1
export CKPT_LABEL=v40_replace_fresh_perhead

/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py
