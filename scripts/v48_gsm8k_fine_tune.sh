#!/bin/bash
# v48 — GSM8K fine-tune from v46b step 750 (L4.5 champion).
#
# Two sequential runs, each on GSM8K-spaced (7.5k train, 1.3k test). Tests
# whether (a) the v45 reg stack continues to transfer beyond L4.5, and
# (b) per-layer phase diversity (triangle-small offsets validated by sweep)
# lifts the model beyond the v23a baseline architecture.
#
#   1. CONTROL (v48a): v23a baseline architecture, GSM8K-spaced fine-tune.
#   2. EXPERIMENT (v48b): triangle-small per-layer offsets {0, π/16, π/8, π/16}.
#
# Comparison: experiment - control acc on GSM8K test set = per-layer-phase
# signal on a benchmark with REAL HEADROOM (baseline pre-fine-tune is 1%).
#
# Architecture matches v46b warm-start (v24c-era) so v45 reg stack stays
# meaningful. BATCH=4 because FIXED_LEN=320 (GSM8K-spaced needs the room).

set -e
cd "$(dirname "$0")/.."

shared_env() {
    # Reg stack (validated)
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

    # Training
    export DEV=PCI+AMD
    export LEVEL=GSM8K_SPACED
    export BATCH=4              # FIXED_LEN=320 needs smaller batch
    export FIXED_LEN=320
    export STEPS=2000           # bigger since we're adapting to new distribution
    export LR=3e-5
    # 2026-05-18: dropped A=8 from training and eval. Across recent runs
    # (v45 96/94/93, v46b 92/92/88), A=8 was -3 to -4 below A=4 — depth was
    # decorative. Removing it saves ~30% training compute and concentrates
    # signal on the loop counts that actually pay back. A=4 is the operating
    # depth going forward.
    export TRAIN_LOOPS=1,2,4
    export EVAL_LOOPS=1,2,4
    export ACC_EVAL_EVERY=500
    export CKPT_EVERY=500
    export NUM_EVAL=100         # cap eval for speed during training
    export EVAL_BATCH=16        # FIXED_LEN=320 × A=8 OOMs at B=64; 16 keeps KV cache under control
    export EVAL_CACHE_LEN=400   # FIXED_LEN + max_new
    export LOOKUP_AUX_WEIGHT=0.1
    export USE_JIT=1
    export RESUME_FROM=.cache/l4_5_ckpts/v46b_control_l4_5_step750.safetensors
}

reset_per_layer() {
    export PER_LAYER_OFFSETS_RADIANS=""
    export ACROSS_LAYER_PITCH_TARGET=0
    export ACROSS_LAYER_PITCH_RAMP_STEPS=0
    export QUADRATURE_HEADS=0
    export QUADRATURE_RAMP_STEPS=0
}

# ----- v48a CONTROL: v23a baseline architecture -----
echo ""
echo "========================================"
echo "  v48a CONTROL: GSM8K-spaced, v23a baseline"
echo "========================================"
shared_env
reset_per_layer
export CKPT_LABEL=v48a_control_gsm8k
/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py 2>&1 | tee .cache/gsm8k_v48a_control.log

# ----- v48b EXPERIMENT: triangle-small per-layer offsets -----
echo ""
echo "========================================"
echo "  v48b EXPERIMENT: GSM8K-spaced + triangle-small {0, π/16, π/8, π/16}"
echo "========================================"
shared_env
reset_per_layer
# π/16 ≈ 0.19634954, π/8 ≈ 0.39269908
export PER_LAYER_OFFSETS_RADIANS="0,0.19634954,0.39269908,0.19634954"
export CKPT_LABEL=v48b_triangle_gsm8k
/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py 2>&1 | tee .cache/gsm8k_v48b_triangle.log

echo ""
echo "========================================"
echo "  v48 complete. Compare step 2000 acc:"
echo "    v48a CONTROL:    $(grep 'acc @ A=' .cache/gsm8k_v48a_control.log 2>/dev/null | tail -3 | tr -d '\n')"
echo "    v48b TRIANGLE:   $(grep 'acc @ A=' .cache/gsm8k_v48b_triangle.log 2>/dev/null | tail -3 | tr -d '\n')"
echo "  baseline (v46b step 750 on GSM8K, no fine-tune): 1/1/?"
echo "========================================"
