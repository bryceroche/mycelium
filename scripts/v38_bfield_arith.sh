#!/bin/bash
# v38 — B-field IB bottleneck (1024 -> 256 -> 1024) at the breath waist.
# Cold-start ARITH from Pythia. Same lookup config as v36 (per-(op, layer)
# centroids, LOOKUP_TEMP=20) so the only delta is the B-field.
#
# Photon analogy: rotation = E field (already on via PER_HEAD_PITCH), B-field
# = perpendicular compression axis. Single waist after L1 each breath.
# Zero-init proj_up means initial forward is bit-identical to v36 baseline;
# gradient learns the bottleneck from scratch.
set -e
cd "$(dirname "$0")/.."

# Core: B-field
export BFIELD_WAIST=256

# v36 baseline config (per-(op, layer) lookup library, T=20 sharp routing)
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
export CKPT_LABEL=v38_bfield_w256

/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py
