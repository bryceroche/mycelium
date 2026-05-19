#!/bin/bash
# v59 — continue training v58 step 750 on L4.7 for 1500 more steps.
#
# Hypothesis: v58 hit 37% segmented on L4.7 with PERFECT step decomposition —
# the remaining errors are arithmetic execution, not orchestration. Per-step
# accuracy is 37^(1/4) = 78%, already breaking the prior 65% 4-layer
# arithmetic ceiling. More training should push per-step to 85-90%,
# giving final 50-66%.
#
# All perf wins active:
#   - JIT'd per_breath training (USE_JIT=1, flat 0.95s/step)
#   - argmax-in-JIT eval (built today)
#   - in-training acc eval OFF (ACC_EVAL_EVERY huge + SKIP_FINAL_ACC=1)
#   - Reg stack (STOCH_DEPTH_P=0.10, LABEL_SMOOTHING=0.1, WEIGHT_DECAY=0.05)
#
# Eval after: scripts/eval_ckpt_controller_segmented.py (segmented only).

set -e
cd "$(dirname "$0")/.."

# v58 architecture (carry forward)
export CONTROLLER_DECODE=1
export CONTROLLER_N_LAYERS=2                # WIDER controller (v58 wins)
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

# Training — continue v58 on L4.7, 1500 more steps, no acc eval at all
export DEV=PCI+AMD
export LEVEL=L4.7
export SPACE_DIGITS=1
export BATCH=2
export FIXED_LEN=200
export STEPS=1500
export LR=3e-5
export TRAIN_LOOPS=4
export EVAL_LOOPS=1,2,3,4
export ACC_EVAL_EVERY=10000                  # off (intermediate evals)
export SKIP_FINAL_ACC=1                      # off (final eval) — segmented eval runs separately
export CKPT_EVERY=250
export NUM_EVAL=100
export NUM_PROBLEMS=20000
export EVAL_BATCH=4
export EVAL_CACHE_LEN=208
export LOOKUP_AUX_WEIGHT=0.1
export USE_JIT=1
export CKPT_LABEL=v59_continue_l4_7
export RESUME_FROM=.cache/l4_7_ckpts/v58_wider_controller_l4_7_step750.safetensors

/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py 2>&1 | tee .cache/v59_continue_l4_7.log
