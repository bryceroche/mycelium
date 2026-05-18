#!/bin/bash
# v49 — MIXED CURRICULUM training including GSM8K.
#
# v48 fine-tune-on-GSM8K-only plateaued at val loss 2.7, acc 0-3%. The model
# couldn't bridge L4.5's templated 3-cycle → GSM8K's free-form natural-language
# variable-cycle in pure fine-tune. This run mixes all three distributions per
# training step (uniformly sampled), warm-starting from v46b step 750.
#
# Hypothesis: the v45 reg stack transferred L4_MIXED → L4.5 because L4.5 ≈
# L4_MIXED + one more cycle. GSM8K is too far away for pure fine-tune; needs
# to be IN the training mix from the start (like v44 designed but never ran).
#
# Per-step level sampling: ~33% L4_MIXED (refresh shallower), ~33% L4.5 (depth
# practice), ~33% GSM8K (the new distribution). Different fixed_len per level
# means JIT compiles separately for each (level, n_loops) combo (9 graphs).
#
# Eval is on GSM8K test set — the real benchmark, real headroom.

set -e
cd "$(dirname "$0")/.."

# Mixed curriculum (the test)
export MIXED_LEVELS=L4_MIXED,L4.5,GSM8K_SPACED

# Reg stack (validated)
export STOCH_DEPTH_P=0.10
export LABEL_SMOOTHING=0.1
export WEIGHT_DECAY=0.05

# v24c-era architecture (matches v46b warm-start) — no quadrature/triangle yet
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

# Quadrature off (clean curriculum-only baseline)
export QUADRATURE_HEADS=0
export ACROSS_LAYER_PITCH_TARGET=0
export PER_LAYER_OFFSETS_RADIANS=""

# Training
export DEV=PCI+AMD
export LEVEL=GSM8K_SPACED      # primary level (drives eval; GSM8K test set)
export SPACE_DIGITS=1
export BATCH=4                  # FIXED_LEN=320 (longest level) needs small batch
export FIXED_LEN=320            # PRIMARY for L=GSM8K_SPACED; mixed pool overrides per-step
export STEPS=3000               # ~1k effective steps per level
export LR=3e-5
export TRAIN_LOOPS=1,2,4        # A=8 dropped (decorative, validated 2026-05-18)
export EVAL_LOOPS=1,2,4
export ACC_EVAL_EVERY=500
export CKPT_EVERY=500
export NUM_EVAL=100             # cap for speed; full 1318 test set via eval_ckpt_on_gsm8k.py at end
export NUM_PROBLEMS=20000       # train set size per level (GSM8K caps at its actual train size)
export EVAL_BATCH=16            # avoid OOM at FIXED_LEN=320, A=8 eval
export EVAL_CACHE_LEN=400
export LOOKUP_AUX_WEIGHT=0.1
export USE_JIT=1
export CKPT_LABEL=v49_mixed
export RESUME_FROM=.cache/l4_5_ckpts/v46b_control_l4_5_step750.safetensors

/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py 2>&1 | tee .cache/v49_mixed_curriculum.log
