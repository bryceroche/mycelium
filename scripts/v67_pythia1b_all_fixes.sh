#!/bin/bash
# v67 — Pythia-1B base + ALL v66 fixes.
#
# Architecture:
#   - Pythia-1B layers (HIDDEN=2048, N_HEADS=8, HEAD_DIM=256, FFN=8192)
#   - controller_hidden=1024 (decoupled — keeps controller compact)
#   - Waist=512d (UNCHANGED from 410M — same per-step bottleneck, now 4× compression)
#
# v66 fixes carried forward:
#   - PROMPT_REFRESH_ALPHA=0.2 — BUMPED from 0.1 to compensate for 4× compression at H=2048
#   - BOUNDARY_AUX_WEIGHT=0.1 with BOUNDARY_POS_WEIGHT=5.0
#   - SCHED_SAMPLE_RATE=0.3 (linear ramp warmup=500, full at step 1500)
#
# Stability (Sonnet's fixes, now in main):
#   - score.clip(-1e4, 1e4) before softmax at all attention sites
#   - total.isfinite() NaN-skip in JIT'd train step
#   - GRAD_CLIP=1.0 (global-norm)
#
# Cold-start: no warm-start (v62 step1000/2000 ckpts had NaN history; better fresh).
# ETA: ~10-12h for 3000 steps at BATCH=1.

set -e
cd "$(dirname "$0")/.."

if [ ! -f ".cache/pythia-1b/model.safetensors" ]; then
    echo "ERROR: Pythia-1B weights not found at .cache/pythia-1b/model.safetensors"
    exit 1
fi

# v67 — Pythia-1B base dims
export HIDDEN=2048
export N_HEADS=8
export HEAD_DIM=256
export FFN=8192
export CONTROLLER_HIDDEN=1024
export PYTHIA_WEIGHTS=.cache/pythia-1b/model.safetensors

# Architecture (matches v66 + Pythia-1B base)
export NOTEBOOK_DAG=0
export CONTROLLER_DECODE=1
export CONTROLLER_N_LAYERS=2
export PER_BREATH_DECODE=1
export BFIELD_WAIST=512
export BFIELD_END_OF_BREATH=1
export BFIELD_ENFORCED=0
export BFIELD_ALPHA=1.0
export WAIST_CODEBOOK_N=64
export WAIST_CODEBOOK_INJECT_WEIGHT=1.0
export NOTEBOOK_V24=1
export NOTEBOOK_ACCUMULATE_ENABLED=0
export NOTEBOOK_DUAL=1
export NOTEBOOK_POOL_MODE=attn
export NOTEBOOK_INIT_SCALE=0.02
export STOCH_DEPTH_P=0.10
export LABEL_SMOOTHING=0.1
export WEIGHT_DECAY=0.05
export PER_HEAD_PITCH=1
export SINE_TEMP=1
export SINE_TEMP_MAX=2.0
export SINE_TEMP_MIN=0.7
export CONSTANT_RADIUS=1
export BREATH_TIME_EMBED=1
export BREATH_TIME_INIT_SCALE=0.0
export CROSS_BREATH_HANDOFF=1
export ABLATE_BREATH_ROTATION=1
export QUADRATURE_HEADS=0
export ACROSS_LAYER_PITCH_TARGET=0
export PER_LAYER_OFFSETS_RADIANS=""

# v65/v66 fixes
export PROMPT_REFRESH_ALPHA=0.2     # BUMPED from 0.1 — 4× compression at H=2048 needs stronger skip
export BOUNDARY_AUX_WEIGHT=0.1
export BOUNDARY_POS_WEIGHT=5.0
export SCHED_SAMPLE_RATE=0.3
export GRAD_CLIP=1.0                # Pythia-1B stability (Sonnet's fix)

# Training
export DEV=PCI+AMD
export LEVEL=GSM8K_STEPS
export GSM8K_STEPS_PATH=.cache/gsm8k_steps_v1_train.jsonl
export GSM8K_STEPS_MIN_K=2
export GSM8K_STEPS_MAX_K=6
export SPACE_DIGITS=1
export BATCH=1                              # 4× per-token cost at H=2048
export FIXED_LEN=320
export STEPS=3000
export LR=3e-5                              # match v66; Sonnet's fixes prevent NaN at this rate
export TRAIN_LOOPS=4
export EVAL_LOOPS=1,2,3,4
export ACC_EVAL_EVERY=10000
export SKIP_FINAL_ACC=1
export CKPT_EVERY=500
export NUM_EVAL=100
export NUM_PROBLEMS=20000
export EVAL_BATCH=2
export EVAL_CACHE_LEN=328
export LOOKUP_AUX_WEIGHT=0.1
export USE_JIT=1
export USE_KV_CACHE=1
export CKPT_LABEL=v67_pythia1b_all_fixes
# Cold-start (no RESUME_FROM)

/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py 2>&1 | tee .cache/v67_pythia1b_all_fixes.log
