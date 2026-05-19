#!/bin/bash
# v56 — WaistController + codebook on L4.5 (K=3 inner breaths).
#
# Extends v55's K=2 architecture to K=3 on 3-cycle problems. Tests whether the
# K-breath paradigm genuinely scales with depth, which is the architectural
# reason for having inner breaths at all.
#
# v55 hit 89% aligned on L4 (K=2) — that's a leading indicator the structure
# helps. If v56 lifts on L4.5 vs v25's misaligned-paradigm 79/79/72, the
# rep-space-thinking paradigm scales with depth and v57 becomes K=4 on L4.7.
# If v56 plateaus near v55, we've found the depth ceiling of the current
# controller capacity — next moves are widening the controller (2 cross-attn
# blocks) or warming the codebook keys from extracted centroids.
#
# Warm-start from v55 step 500. L4.5 only (K=3 uniform). 500 steps.
# Aligned eval after training: scripts/eval_ckpt_controller_l4.py with
#   LEVEL=L4.5, K=3, FIXED_LEN=192, MAX_NEW=80.

set -e
cd "$(dirname "$0")/.."

# v55 architecture (carry forward)
export CONTROLLER_DECODE=1
export CONTROLLER_N_LAYERS=1
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

# Training — L4.5 only (K=3 uniform)
export DEV=PCI+AMD
export LEVEL=L4.5
export SPACE_DIGITS=1
export BATCH=8                              # v25 OOM'd at 16 on L4.5
export FIXED_LEN=160                        # matches v25
export STEPS=500
export LR=3e-5
export TRAIN_LOOPS=3
export EVAL_LOOPS=1,2,3
export ACC_EVAL_EVERY=125
export CKPT_EVERY=125
export NUM_EVAL=100
export NUM_PROBLEMS=20000
export EVAL_BATCH=16                         # tighter to keep KV cache under control at FIXED_LEN=160
export EVAL_CACHE_LEN=168
export LOOKUP_AUX_WEIGHT=0.1
export USE_JIT=1
export CKPT_LABEL=v56_controller_codebook_l4_5
export RESUME_FROM=.cache/l4_ckpts/v55_controller_codebook_step500.safetensors

/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py 2>&1 | tee .cache/v56_controller_codebook_l4_5.log
