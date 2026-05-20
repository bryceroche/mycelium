#!/bin/bash
# v60-take-2 — GSM8K_STEPS with FULL train data + longer training.
#
# v60 (take 1) result: 2.3% on GSM8K test at step 3000. Failure mode was
# semantic confabulation (right format, wrong content). Two hypotheses for
# the cause: insufficient training-volume vs insufficient model capacity.
#
# v60-take-2 tests the training-volume hypothesis: same architecture,
# nearly 2× the train data (6026 in K=2..6 vs snapshot's 3457), 2× the
# steps (6000 vs 3000), slightly higher LR. Warm-start from v59 step 1500
# (not v60, to avoid baking in v60's confabulations).
#
# Outcome differentiates:
#   10%+ → training-volume was the limit; keep scaling training.
#   5-7% → partial; bigger model probably needed too.
#   2-3% → capacity is the lock; move to Pythia-1B (v62 design).

set -e
cd "$(dirname "$0")/.."

if [ ! -f ".cache/gsm8k_steps_v1_train.jsonl" ]; then
    echo "ERROR: train JSONL not found."
    exit 1
fi

# v55..v59 architecture (carry forward)
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

# Training — GSM8K_STEPS bucketed
export DEV=PCI+AMD
export LEVEL=GSM8K_STEPS
export GSM8K_STEPS_PATH=.cache/gsm8k_steps_v1_train.jsonl   # full 6705 → 6026 in K=2..6
export GSM8K_STEPS_MIN_K=2
export GSM8K_STEPS_MAX_K=6
export SPACE_DIGITS=1
export BATCH=2
export FIXED_LEN=320                       # ~8% truncation at K=6 worst case; acceptable
export STEPS=6000                          # 2× v60's 3000 — ~2 epochs over the K=2..6 data
export LR=5e-5                             # 1.67× v60's 3e-5 — bigger data → faster adaptation
export TRAIN_LOOPS=4
export EVAL_LOOPS=1,2,3,4
export ACC_EVAL_EVERY=10000
export SKIP_FINAL_ACC=1
export CKPT_EVERY=1000                     # save 6 ckpts for learning curve
export NUM_EVAL=100
export NUM_PROBLEMS=20000
export EVAL_BATCH=4
export EVAL_CACHE_LEN=328
export LOOKUP_AUX_WEIGHT=0.1
export USE_JIT=1
export USE_KV_CACHE=1
export CKPT_LABEL=v60_take2_gsm8k_steps
export RESUME_FROM=.cache/l4_7_ckpts/v59_continue_l4_7_step1500.safetensors

/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py 2>&1 | tee .cache/v60_take2_gsm8k_steps.log
