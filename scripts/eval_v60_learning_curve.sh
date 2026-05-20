#!/bin/bash
# Evaluate all v60 GSM8K_STEPS checkpoints sequentially and report a
# learning curve: per-ckpt segmented acc on the GSM8K test JSONL.
#
# Run after v60 training finishes (uses the same GPU; concurrent training
# would block on the AM driver lock).

set -e
cd "$(dirname "$0")/.."

# Match v59-era architecture (same flags used to train v60).
export DEV=PCI+AMD
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

# Eval config — GSM8K_STEPS auto-buckets by K.
export LEVEL=GSM8K_STEPS
export GSM8K_STEPS_PATH=.cache/gsm8k_steps_v1_test.jsonl
export GSM8K_STEPS_MIN_K=2
export GSM8K_STEPS_MAX_K=6
export NUM_EVAL=300                         # ~60/bucket × 5 buckets
export K=4                                  # ignored for GSM8K_STEPS (auto-bucketed)
export FIXED_LEN=320
export BATCH=4
export MAX_NEW=200
export USE_KV_CACHE=1

CKPT_DIR=".cache/gsm8k_steps_ckpts"
OUT=".cache/v60_learning_curve.txt"
echo "=== v60 learning curve on GSM8K test (NUM_EVAL=$NUM_EVAL) ===" > "$OUT"
echo "started: $(date)" >> "$OUT"
echo >> "$OUT"

for STEP in 500 1000 1500 2000 2500 3000; do
    CKPT_PATH="$CKPT_DIR/v60_gsm8k_steps_step${STEP}.safetensors"
    if [ ! -f "$CKPT_PATH" ]; then
        echo "--- step $STEP: ckpt missing, skipping ---" | tee -a "$OUT"
        continue
    fi
    echo "--- evaluating step $STEP ---" | tee -a "$OUT"
    CKPT="$CKPT_PATH" /home/bryce/mycelium/.venv/bin/python scripts/eval_ckpt_controller_segmented.py 2>&1 | \
        tee -a "$OUT" | \
        grep -E "(=== segmented acc:|per-K|K=[2-6]:)"
    echo >> "$OUT"
done

echo "=== finished: $(date) ===" >> "$OUT"
echo
echo "Full results in $OUT"
echo
echo "Summary:"
grep "=== segmented acc:" "$OUT"
