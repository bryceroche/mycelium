#!/bin/bash
# v54 Phase 1 — WaistController as supervision conduit.
#
# K=2 inner breaths (no outer cycles, REPLACE notebook for sequentiality).
# Per breath, a small WaistController (cross-attention from compressed
# waist 512d → prompt embeddings 1024d, tied embed_out) emits per-position
# vocab logits. CE supervises against step_k's gen_target tokens. Gradient
# flows back through the controller → into the waist → shapes the main
# model's rep space.
#
# Key difference from v53: the main model no longer decodes text from each
# breath's end-of-breath state. ALL text-level supervision flows through
# the controller. Main model's job becomes "produce a useful 512d waist";
# controller's job is "decode that waist into partial-answer text".
#
# Validation criteria at step 250:
#   PASS: pb_ce[0] and pb_ce[1] DIFFERENTIATE (>0.2 apart) AND drop below ~5.0
#         → controller is learning to read distinct waists per breath.
#   PARTIAL: both pb_ce drop but stay symmetric → waist isn't carrying
#            differentiable information; controller falls back on prompt only.
#   FAIL: pb_ce stuck at ~10 → controller can't bootstrap. Capacity issue.
#
# Warm-start from v46b step 750. L4 only (K=2 uniform). 500 steps.
# Controller is fresh random init at 0.02 (no warm-start for it).

set -e
cd "$(dirname "$0")/.."

# v54 Phase 1: enable the WaistController decode path
export CONTROLLER_DECODE=1
export CONTROLLER_N_LAYERS=1                # MVP: 1 cross-attn layer (~13M params)
export PER_BREATH_DECODE=1

# Waist mechanism active (controller reads compressed 512d)
export BFIELD_WAIST=512
export BFIELD_END_OF_BREATH=1
export BFIELD_ENFORCED=0
export BFIELD_ALPHA=1.0

# Codebook OFF (clean test of controller alone; can add codebook in Phase 2)
export WAIST_CODEBOOK_N=0
export WAIST_CODEBOOK_INJECT_WEIGHT=0

# REPLACE-only notebook (forces sequential breath state passing)
export NOTEBOOK_V24=1
export NOTEBOOK_ACCUMULATE_ENABLED=0
export NOTEBOOK_DUAL=1
export NOTEBOOK_POOL_MODE=attn
export NOTEBOOK_INIT_SCALE=0.02

# Reg stack (validated)
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

# Training — L4 only (K=2 uniform)
export DEV=PCI+AMD
export LEVEL=L4
export SPACE_DIGITS=1
export BATCH=8
export FIXED_LEN=96
export STEPS=500
export LR=3e-5
export TRAIN_LOOPS=2
export EVAL_LOOPS=1,2
export ACC_EVAL_EVERY=125
export CKPT_EVERY=125
export NUM_EVAL=100
export NUM_PROBLEMS=20000
export EVAL_BATCH=32
export EVAL_CACHE_LEN=136
export LOOKUP_AUX_WEIGHT=0.1
export USE_JIT=0                            # per_breath path non-JIT
export CKPT_LABEL=v54_controller
export RESUME_FROM=.cache/l4_5_ckpts/v46b_control_l4_5_step750.safetensors

/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py 2>&1 | tee .cache/v54_controller_decode.log
