#!/bin/bash
# v66 — scheduled sampling + boundary pos_weight + waist norm logging.
#
# Three additions on top of v65's two fixes:
#   1. SCHED_SAMPLE_RATE=0.5 — linear ramp from 0 (steps 0-500) to 0.5 (step 1500+).
#      Each breath k predicts the next token via WaistController; a Bernoulli mask
#      replaces some gold tokens with argmax predictions for breath k+1's input.
#      Forces model to handle its own prediction errors (fixes teacher-forcing gap).
#   2. BOUNDARY_POS_WEIGHT=5.0 — up-weights positive class in the boundary BCE loss.
#      #### token appears ~1 per 10-30 positions; without pos_weight, the head learns
#      "always predict NOT-boundary" (95% acc, zero useful gradient).
#   3. Waist norm logged every 10 steps; warns if < 0.01 for 100 consecutive steps
#      (skip-connection leakage: model ignoring waist, relying on prompt refresh).
#
# Warm-start: v60-take-2 step 6000 (2.0% on GSM8K — same as v65 baseline).

set -e
cd "$(dirname "$0")/.."

if [ ! -f ".cache/gsm8k_steps_ckpts/v60_take2_gsm8k_steps_step6000.safetensors" ]; then
    echo "ERROR: v60-take-2 step 6000 ckpt not found."
    exit 1
fi

# v60-take-2 architecture (no DAG, no K-pos)
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

# v65 — entity preservation + segment timing
export PROMPT_REFRESH_ALPHA=0.1   # per-breath skip from raw prompt_emb
export BOUNDARY_AUX_WEIGHT=0.1    # BCE on #### prediction per breath

# v66 — scheduled sampling + boundary pos_weight + waist norm logging
# Conservative SCHED_SAMPLE_RATE=0.3 for first overnight (more aggressive 0.5 deferred to v67 if needed).
# Schedule: 0% steps 0-500 (warmup) → linear ramp 0→0.3 over steps 500-1500 → 0.3 sustained.
export SCHED_SAMPLE_RATE=0.3
export BOUNDARY_POS_WEIGHT=5.0    # up-weight #### positive class 5×

# Training
export DEV=PCI+AMD
export LEVEL=GSM8K_STEPS
export GSM8K_STEPS_PATH=.cache/gsm8k_steps_v1_train.jsonl
export GSM8K_STEPS_MIN_K=2
export GSM8K_STEPS_MAX_K=6
export SPACE_DIGITS=1
export BATCH=2
export FIXED_LEN=320
export STEPS=3000
export LR=3e-5
export TRAIN_LOOPS=4
export EVAL_LOOPS=1,2,3,4
export ACC_EVAL_EVERY=10000
export SKIP_FINAL_ACC=1
export CKPT_EVERY=500
export NUM_EVAL=100
export NUM_PROBLEMS=20000
export EVAL_BATCH=4
export EVAL_CACHE_LEN=328
export LOOKUP_AUX_WEIGHT=0.1
export USE_JIT=1
export USE_KV_CACHE=1
export CKPT_LABEL=v66_sched_sampling
export RESUME_FROM=.cache/gsm8k_steps_ckpts/v60_take2_gsm8k_steps_step6000.safetensors

/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py 2>&1 | tee .cache/v66_sched_sampling.log
