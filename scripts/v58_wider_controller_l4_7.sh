#!/bin/bash
# v58 — wider WaistController (2 cross-attn blocks), K=4 on L4.7.
#
# v57 K=4 on L4.7: aligned 25, segmented 33, gap +8.
# Scaling pattern:
#   K=2 (L4):   aligned 89, segmented 90  →  +1
#   K=3 (L4.5): aligned 75, segmented 89  →  +14
#   K=4 (L4.7): aligned 25, segmented 33  →  +8
# The K=3→K=4 gap shrunk. Two hypotheses: capacity ceiling or undertrained.
# v58 tests capacity: add a second cross-attn block to the WaistController.
# If aligned and segmented both lift (and gap grows/stabilizes), capacity was
# the bottleneck and we have a better foundation for GSM8K. If neither lifts,
# capacity isn't the issue and we look elsewhere (per-breath controllers,
# longer training, or step-target supervision).
#
# Warm-start from v56 step 500. Layer 0 of the controller loads from ckpt;
# layer 1 is random init from the model constructor (load_state_dict
# strict=False tolerates the missing keys).
#
# 750 steps (vs v57's 500) — wider controller needs more time to use layer 1.

set -e
cd "$(dirname "$0")/.."

# v55/v56/v57 architecture
export CONTROLLER_DECODE=1
export CONTROLLER_N_LAYERS=2                # WIDER controller — the v58 delta
export PER_BREATH_DECODE=1
export BFIELD_WAIST=512
export BFIELD_END_OF_BREATH=1
export BFIELD_ENFORCED=0
export BFIELD_ALPHA=1.0
export WAIST_CODEBOOK_N=64
export WAIST_CODEBOOK_INJECT_WEIGHT=1.0

# REPLACE-only notebook
export NOTEBOOK_V24=1
export NOTEBOOK_ACCUMULATE_ENABLED=0
export NOTEBOOK_DUAL=1
export NOTEBOOK_POOL_MODE=attn
export NOTEBOOK_INIT_SCALE=0.02

# Reg stack
export STOCH_DEPTH_P=0.10
export LABEL_SMOOTHING=0.1
export WEIGHT_DECAY=0.05

# v24c-era architecture
export PER_HEAD_PITCH=1
export SINE_TEMP=1
export SINE_TEMP_MAX=2.0
export SINE_TEMP_MIN=0.7
export CONSTANT_RADIUS=1
export BREATH_TIME_EMBED=1
export BREATH_TIME_INIT_SCALE=0.0
export CROSS_BREATH_HANDOFF=1
export ABLATE_BREATH_ROTATION=1

# Everything else off
export QUADRATURE_HEADS=0
export ACROSS_LAYER_PITCH_TARGET=0
export PER_LAYER_OFFSETS_RADIANS=""

# Training — L4.7 only (K=4 uniform)
export DEV=PCI+AMD
export LEVEL=L4.7
export SPACE_DIGITS=1
export BATCH=2                               # K=4 + fixed_len=200; tightest stable config
export FIXED_LEN=200
export STEPS=750                             # 50% more than v57 (wider controller needs more time)
export LR=3e-5
export TRAIN_LOOPS=4
export EVAL_LOOPS=1,2,3,4
export ACC_EVAL_EVERY=10000                  # off — misaligned eval is useless for this paradigm; run aligned+segmented at end
export CKPT_EVERY=125
export NUM_EVAL=100
export NUM_PROBLEMS=20000
export EVAL_BATCH=4
export EVAL_CACHE_LEN=208
export LOOKUP_AUX_WEIGHT=0.1
export USE_JIT=1                             # JIT'd per_breath path
export CKPT_LABEL=v58_wider_controller_l4_7
export RESUME_FROM=.cache/l4_5_ckpts/v56_controller_codebook_l4_5_step500.safetensors

/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py 2>&1 | tee .cache/v58_wider_controller_l4_7.log
