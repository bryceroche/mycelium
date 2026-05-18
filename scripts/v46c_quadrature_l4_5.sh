#!/bin/bash
# v46c — QUADRATURE RAMP on L4.5, warm-started from v46b control step 750
# (the new L4.5 champion at 92/92/88).
#
# Tests whether quadrature (heads 0..7 at base offset, heads 8..15 at base+π/2)
# adds anything beyond what plain L4.5 fine-tuning already achieved. Since v46b
# control step 1000 regressed from step 750 (84/85/78 vs 92/92/88), we re-warm
# from step 750 instead.
#
# Ramp: scale 0 → π/2 over the first 500 steps (so the model's W_O adapts
# slowly to the diverging head geometry). After step 500, full quadrature.
# Total 1000 steps, eval every 250.
#
# Comparison target: v46b step 750 (92/92/88). Quadrature signal = anything
# above that. v46b step 1000 control (84/85/78) shows what plain over-training
# gives — so we need quadrature to beat both 88 (A=8 peak) and avoid the
# overshoot pattern.

set -e
cd "$(dirname "$0")/.."

# v46c: quadrature ramp
export QUADRATURE_HEADS=1
export QUADRATURE_RAMP_STEPS=500

# Reg stack inherited from v45 (validated)
export STOCH_DEPTH_P=0.10
export LABEL_SMOOTHING=0.1
export WEIGHT_DECAY=0.05

# v24c-era architecture (matches v46b warm-start)
export PER_HEAD_PITCH=1
export SINE_TEMP=1
export SINE_TEMP_MAX=2.0
export SINE_TEMP_MIN=0.7
export CONSTANT_RADIUS=1
export BREATH_TIME_EMBED=1
export BREATH_TIME_INIT_SCALE=0.0
export CROSS_BREATH_HANDOFF=1
export ABLATE_BREATH_ROTATION=1
export NOTEBOOK_V24=1
export NOTEBOOK_DUAL=1
export NOTEBOOK_POOL_MODE=attn
export NOTEBOOK_INIT_SCALE=0.02

# Training — L4.5, warm-start from v46b control step 750
export DEV=PCI+AMD
export LEVEL=L4.5
export SPACE_DIGITS=1
export BATCH=8
export FIXED_LEN=160
export STEPS=1000
export LR=3e-5
export TRAIN_LOOPS=1,2,4,8
export EVAL_LOOPS=1,4,8
export ACC_EVAL_EVERY=250
export CKPT_EVERY=250
export LOOKUP_AUX_WEIGHT=0.1
export USE_JIT=1
export CKPT_LABEL=v46c_quadrature_l4_5
export RESUME_FROM=.cache/l4_5_ckpts/v46b_control_l4_5_step750.safetensors

/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py 2>&1 | tee .cache/l4_5_v46c_quadrature.log
