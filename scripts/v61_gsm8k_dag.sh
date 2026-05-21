#!/bin/bash
# v61 — DAG notebook on top of v60-take-2.
#
# Architecture delta vs v60-take-2:
#   - NOTEBOOK_DAG=1: multi-entry attention-readable notebook (coexists with REPLACE notebook)
#   - 4-head causal cross-attention over (B, max_loops, 512) slot storage
#   - Slot positional embeddings enabled
#
# Warm-start: v60-take-2 step 6000 (2.0% on GSM8K test).
# DAG output proj is zero-init → byte-identical forward at step 0 → warm-start safe.
# Gradient flows once writes populate storage (LoRA-style asymmetric init).
#
# Hypothesis: DAG helps on K≥4 (multi-step problems with non-linear dependencies).
# At K=2/3 we expect parity with v60-take-2 (breaths converge).
# Outcome:
#   v60-take-2 6000 + 3000 with DAG > v60-take-2 → DAG wins (ship as new standard)
#   No lift → DAG decorative; pivot to other capacity ideas (DOUBLED_LAYERS, Pythia-1B).

set -e
cd "$(dirname "$0")/.."

if [ ! -f ".cache/gsm8k_steps_v1_train.jsonl" ]; then
    echo "ERROR: train JSONL not found."
    exit 1
fi

if [ ! -f ".cache/gsm8k_steps_ckpts/v60_take2_gsm8k_steps_step6000.safetensors" ]; then
    echo "ERROR: v60-take-2 step 6000 ckpt not found."
    exit 1
fi

# v55..v60 architecture (carry forward) + v61 DAG
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

# v61 — DAG notebook (the new piece)
export NOTEBOOK_DAG=1
export NOTEBOOK_DAG_N_HEADS=4
export NOTEBOOK_DAG_POS_EMBED=1

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

# Training
export DEV=PCI+AMD
export LEVEL=GSM8K_STEPS
export GSM8K_STEPS_PATH=.cache/gsm8k_steps_v1_train.jsonl
export GSM8K_STEPS_MIN_K=2
export GSM8K_STEPS_MAX_K=6
export SPACE_DIGITS=1
export BATCH=2
export FIXED_LEN=320
export STEPS=3000                          # 3000 more on top of v60-take-2's 6000 (effective total = 9000)
export LR=3e-5                             # lower than v60-take-2's 5e-5 — DAG adds ~2M new params; gentler refinement
export TRAIN_LOOPS=4
export EVAL_LOOPS=1,2,3,4
export ACC_EVAL_EVERY=10000
export SKIP_FINAL_ACC=1
export CKPT_EVERY=500                      # 6 ckpts for learning curve
export NUM_EVAL=100
export NUM_PROBLEMS=20000
export EVAL_BATCH=4
export EVAL_CACHE_LEN=328
export LOOKUP_AUX_WEIGHT=0.1
export USE_JIT=1
export USE_KV_CACHE=1
export CKPT_LABEL=v61_gsm8k_dag
export RESUME_FROM=.cache/gsm8k_steps_ckpts/v60_take2_gsm8k_steps_step6000.safetensors

/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py 2>&1 | tee .cache/v61_gsm8k_dag.log
