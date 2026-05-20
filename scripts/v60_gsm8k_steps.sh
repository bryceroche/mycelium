#!/bin/bash
# v60 — GSM8K with Haiku-distilled per-step gen_targets, K=2..6 bucketed.
#
# Bridges v59's working K-breath paradigm (45% on L4.7 templated, K=4)
# to real GSM8K word problems with variable step counts. Each batch comes
# from a single K bucket; per_breath_train_step compiles separate JIT
# kernels per K (the JIT cache key includes K).
#
# Prerequisite: .cache/gsm8k_steps_v1_train.jsonl exists (from
# scripts/generate_gsm8k_step_targets.py).
#
# Warm-start from v59 step 1500 (the K=4 L4.7 champion). Transfer the
# K-breath circuitry to real-problem distribution.

set -e
cd "$(dirname "$0")/.."

# Validate prerequisite
if [ ! -f ".cache/gsm8k_steps_v1_train.jsonl" ]; then
    echo "ERROR: .cache/gsm8k_steps_v1_train.jsonl not found."
    echo "Run scripts/generate_gsm8k_step_targets.py first."
    exit 1
fi

# v55/v56/v58/v59 architecture
export CONTROLLER_DECODE=1
export CONTROLLER_N_LAYERS=2                # v58+ wider controller
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

# Training — GSM8K_STEPS bucketed
export DEV=PCI+AMD
export LEVEL=GSM8K_STEPS
export GSM8K_STEPS_PATH=.cache/gsm8k_steps_v1_train.jsonl
export GSM8K_STEPS_MIN_K=2
export GSM8K_STEPS_MAX_K=6
export SPACE_DIGITS=1                        # ignored for GSM8K paths (already spaced)
export BATCH=2                               # K up to 6, FIXED_LEN=320 — drop to 1 if OOM
export FIXED_LEN=320                         # K=6 × ~50 tok/step + prompt ~30 ≈ 330; tight, may truncate longest K=6
export STEPS=3000                            # 6x examples seen across 5 K-buckets vs v59's 1500x4=6000 at K=4
export LR=3e-5                               # matches v59; preserves warm-start, gentle adaptation to GSM8K
export TRAIN_LOOPS=4                         # JIT key only; per_breath_train_step infers K from each batch
export EVAL_LOOPS=1,2,3,4
export ACC_EVAL_EVERY=10000                  # off — misaligned eval is structurally unfit for K-breath paradigm
export SKIP_FINAL_ACC=1                      # off — segmented eval runs separately via eval_ckpt_controller_segmented.py
export CKPT_EVERY=500                        # ~50 min/ckpt at 1s/step; saves disk vs 250-step cadence
export NUM_EVAL=100
export NUM_PROBLEMS=20000
export EVAL_BATCH=4
export EVAL_CACHE_LEN=328
export LOOKUP_AUX_WEIGHT=0.1
export USE_JIT=1
export USE_KV_CACHE=1                        # now default
export CKPT_LABEL=v60_gsm8k_steps
export RESUME_FROM=.cache/l4_7_ckpts/v59_continue_l4_7_step1500.safetensors

/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py 2>&1 | tee .cache/v60_gsm8k_steps.log
