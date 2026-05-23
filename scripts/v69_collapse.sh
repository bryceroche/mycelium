#!/bin/bash
# v69 — JPEG/MP3-inspired lossy compression at the waist.
#
# What changed vs v66:
#   - Replace v66's B-field waist MLP (1024 → 512 → 1024) with the new collapse pipeline:
#     1. TRANSFORM: codebook match → prototype (256 entries, soft-VQ)
#     2. RESIDUAL: x - prototype  (what doesn't fit any prototype)
#     3. QUANTIZE: importance = sigmoid(gate(prototype)); residual × importance
#     4. ENCODE: proj_down to 128d (matching empirical signal dim from diagnostic)
#     5. RECONSTRUCT: out = x + α × (prototype + waist @ proj_up), α=0 init
#   - WaistController input dim 512 → 128 (reinit only the input projection; rest preserves from v66)
#
# Training signal:
#   - Per-breath CE through WaistController → gradient flows back through the entire
#     compression pipeline. Controller's "did the decode work?" trains the gate, codebook,
#     projections directly. No REINFORCE, no auxiliary controller loss.
#   - Optional entropy regularizer on match_weights (COLLAPSE_ENTROPY_REG=0.01).
#
# Warm-start: v66 step 3000.
#   - All v66 weights preserved EXCEPT wc.waist_up_w (shape mismatch, reinit) and the
#     new collapse_* params (default-init).
#   - α=0 init means breath output is identical to v66 (no compression contribution to
#     residual stream). Only the controller side is perturbed.
#   - Expect step 0 loss ~10 (controller reads garbage from random waist_up_w), descending
#     rapidly as the collapse pipeline trains.
#
# Eval: USE_KV_CACHE=0 (cached_generate_segmented Stage 1 JIT not v69-aware yet — TODO).

set -e
cd "$(dirname "$0")/.."

if [ ! -f ".cache/gsm8k_steps_ckpts/v66_sched_sampling_step3000.safetensors" ]; then
    echo "ERROR: v66 step 3000 ckpt not found."
    exit 1
fi

# v69 — THE NEW COLLAPSE PIPELINE
export COLLAPSE_V69=1
export COLLAPSE_WAIST_DIM=128             # matches the 96.2%-energy signal dim from diagnostic
export COLLAPSE_CODEBOOK_N=256            # 16 ops × 16 sub-types
export COLLAPSE_TAU=1.0                   # softmax temperature for codebook match
export COLLAPSE_GATE_BIAS=2.0             # sigmoid(2)≈0.88 init → keep most dims initially
export COLLAPSE_ENTROPY_REG=0.01          # reward sparse codebook use (small)

# v66 architecture preserved (single SharedWeights — diagnostic refuted two-set hypothesis)
export TWO_PHASE=0
export NOTEBOOK_DAG=0
export CONTROLLER_DECODE=1
export CONTROLLER_N_LAYERS=2
export PER_BREATH_DECODE=1
export BFIELD_WAIST=512                   # legacy state-dict shape (unused when COLLAPSE_V69=1)
export BFIELD_END_OF_BREATH=1             # ignored when COLLAPSE_V69=1 (collapse runs instead)
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

# v66 fixes (carry forward — all architectural, compatible with v69 collapse)
export PROMPT_REFRESH_ALPHA=0.1
export BOUNDARY_AUX_WEIGHT=0.1
export BOUNDARY_POS_WEIGHT=5.0
export SCHED_SAMPLE_RATE=0.3

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
export LR=2e-5                            # SAFER for new collapse params + reinit waist_up_w
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
export USE_KV_CACHE=0                     # Stage 1 JIT not yet v69-aware (TODO)
export CKPT_LABEL=v69_collapse
export RESUME_FROM=.cache/gsm8k_steps_ckpts/v66_sched_sampling_step3000.safetensors

/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py 2>&1 | tee .cache/v69_collapse.log
