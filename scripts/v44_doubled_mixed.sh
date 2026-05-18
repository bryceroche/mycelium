#!/bin/bash
# v44 — DOUBLED-layers (8 layers, sine-alternated) + MIXED-level training.
#
# Two big architectural changes targeting overfitting:
#
# (1) DOUBLED_LAYERS=1 — 8 phase-layers in two sets. Set A active in breaths
#     [0, max_loops/2), Set B in [max_loops/2, max_loops). With ROPE_FULL_CIRCLE=1
#     (rotation 0→2π over max_loops breaths) this maps to E/B alternation in the
#     photon analogy — zero crossings at breath 0 and breath max_loops/2.
#     Both sets init from Pythia L0-L3; gradient differentiates them.
#     ~254M params vs 127M (forward compute unchanged — each breath still uses 4 layers).
#
# (2) MIXED_LEVELS=ARITH,L3,L4_MIXED — each training step samples a random
#     level. Forces the model to find abstract op representations rather than
#     format-specific shortcuts. Addresses the +/- blindspot we found in cold-start
#     ARITH (v36 & v38 both became */ specialists at the expense of +/-).
#
# Cold-start from Pythia. Both sets get Pythia L0-L3.
set -e
cd "$(dirname "$0")/.."

# v44 architectural changes
export DOUBLED_LAYERS=1
export ROPE_FULL_CIRCLE=1
export MIXED_LEVELS=ARITH,L3,L4_MIXED

# Keep v23a/v24c-era goodies
export PER_HEAD_PITCH=1
export SINE_TEMP=1
export SINE_TEMP_MAX=2.0
export SINE_TEMP_MIN=0.7
export CONSTANT_RADIUS=1
export BREATH_TIME_EMBED=1
export BREATH_TIME_INIT_SCALE=0.0
export CROSS_BREATH_HANDOFF=1
export ABLATE_BREATH_ROTATION=1

# v24c notebook (proven mechanism)
export NOTEBOOK_V24=1
export NOTEBOOK_DUAL=1
export NOTEBOOK_POOL_MODE=attn
export NOTEBOOK_INIT_SCALE=0.02

# B-field OFF for this run (item 14 / 15 deferred to v45 after we see v44 results)
export BFIELD_WAIST=0

# Training
export DEV=PCI+AMD
export LEVEL=L4_MIXED         # primary level used for eval
export SPACE_DIGITS=1
export BATCH=16
export FIXED_LEN=96           # primary fixed_len (mixed levels override per step)
export STEPS=1500
export LR=3e-5
export TRAIN_LOOPS=1,2,4,8
export EVAL_LOOPS=1,4,8
export ACC_EVAL_EVERY=250
export CKPT_EVERY=250
export LOOKUP_AUX_WEIGHT=0.1
export USE_JIT=1
export CKPT_LABEL=v44_doubled_mixed

/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py
