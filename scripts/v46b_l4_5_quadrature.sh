#!/bin/bash
# v46b — quadrature ramp on L4.5 (the harder benchmark with headroom).
#
# Two sequential runs from the v45 step-1000 warm-start:
#   1. CONTROL: fine-tune on L4.5, no quadrature. Establishes the "L4.5
#      adaptation alone" baseline. ~1 hour.
#   2. EXPERIMENT: same config + QUADRATURE_HEADS=1 + QUADRATURE_RAMP_STEPS=500.
#      Tests whether quadrature lifts the L4.5 numbers beyond what fine-tuning
#      alone achieves. ~1 hour.
#
# Comparison: experiment minus control == quadrature signal on L4.5. v45 step
# 1000 already saturates L4_MIXED at 96/94/93, so this is the first test where
# any new mechanism has room to actually move the needle.
#
# Note: L4.5 uses BATCH=8 (16 OOMs per v25 log) and FIXED_LEN=160.

set -e
cd "$(dirname "$0")/.."

# Reg stack inherited from v45 take 3
shared_env() {
    export STOCH_DEPTH_P=0.10
    export LABEL_SMOOTHING=0.1
    export WEIGHT_DECAY=0.05

    # v24c-era architecture (matches v45 warm-start)
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

    # L4.5 training/eval
    export DEV=PCI+AMD
    export LEVEL=L4.5
    export SPACE_DIGITS=1
    export BATCH=8                # L4.5 needs smaller batch — 16 OOMs (v25 log)
    export FIXED_LEN=160
    export STEPS=1000
    export LR=3e-5
    export TRAIN_LOOPS=1,2,4,8
    export EVAL_LOOPS=1,4,8
    export ACC_EVAL_EVERY=250     # bigger gap for L4.5 (eval is slower per breath)
    export CKPT_EVERY=250
    export LOOKUP_AUX_WEIGHT=0.1
    export USE_JIT=1
    export RESUME_FROM=.cache/l4_mixed_ckpts/v45_reg_take3_step1000.safetensors
}

reset_quadrature() {
    export QUADRATURE_HEADS=0
    export QUADRATURE_RAMP_STEPS=0
}

echo ""
echo "========================================"
echo "  v46b CONTROL: L4.5 fine-tune, no quadrature (baseline)"
echo "========================================"
shared_env
reset_quadrature
export CKPT_LABEL=v46b_control_l4_5
/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py 2>&1 | tee .cache/l4_5_v46b_control.log

echo ""
echo "========================================"
echo "  v46b EXPERIMENT: L4.5 + QUADRATURE_RAMP (500 steps to π/2)"
echo "========================================"
shared_env
reset_quadrature
export QUADRATURE_HEADS=1
export QUADRATURE_RAMP_STEPS=500
export CKPT_LABEL=v46b_quadrature_l4_5
/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py 2>&1 | tee .cache/l4_5_v46b_quadrature.log

echo ""
echo "========================================"
echo "  v46b complete. Compare step 1000 acc:"
echo "    CONTROL  (no quadrature):  $(grep 'acc @ A=' .cache/l4_5_v46b_control.log 2>/dev/null | tail -3 | tr -d '\n')"
echo "    EXPERIMENT (quadrature):   $(grep 'acc @ A=' .cache/l4_5_v46b_quadrature.log 2>/dev/null | tail -3 | tr -d '\n')"
echo "========================================"
