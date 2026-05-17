#!/bin/bash
# v39 — Photon-mode B-field: end-of-breath enforced 512d + sin-modulated integral
# + aux op-CE supervision on compressed at "=" position.
#
# Five deltas vs v38:
#   1. BFIELD_END_OF_BREATH=1  — waist fires after L3 (not after L1)
#   2. BFIELD_ENFORCED=1       — no residual skip; bottleneck output IS breath output
#   3. BFIELD_WAIST=512         — wider waist (was 256)
#   4. BFIELD_AUX_WEIGHT=0.1    — op-CE on 512d compressed at "=" position
#   5. BFIELD_SIN_MOD=1         — heartbeat across breaths via sin envelope on integral
#
# Cold-start ARITH from Pythia. Same lookup config as v36/v38 baseline.
# Tests whether B-field becomes load-bearing at inference (α=0 should drop acc).
set -e
cd "$(dirname "$0")/.."

# v39 photon-mode B-field
export BFIELD_WAIST=512
export BFIELD_END_OF_BREATH=1
export BFIELD_ENFORCED=1
export BFIELD_SIN_MOD=1
export BFIELD_AUX_WEIGHT=0.1

# v36 baseline lookup config (per-(op, layer) centroids, T=20 sharp routing)
export LOOKUP_VALUE_INJECT=1
export LOOKUP_VALUES_INIT_PATH=.cache/ib_centroids_per_layer/centroids_n16.npy
export LOOKUP_TEMP=20
export LOOKUP_VALUE_SCALE=1.0
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
export CKPT_LABEL=v39_endbreath_enforced_512

/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py
