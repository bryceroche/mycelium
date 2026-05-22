#!/bin/bash
# v68 — TWO_PHASE breath architecture.
#
# Structural shift: 4-layer symmetric breath → asymmetric EXPAND (4 warm) + COMPRESS (2 cool).
# Each phase has its own SharedWeights (V/O/FFN-out/LNs). Temperature is structural
# (not scheduled): EXPAND_TEMP=2.0, COMPRESS_TEMP=0.7. SINE_TEMP disabled.
#
# Inhale/exhale embodied in architecture rather than approximated through schedules.
#
# Warm-start strategy (hybrid):
#   - Set A (4 EXPAND layers) ← v66's 4 trained layers (full reading-comp stack)
#   - Set B (2 COMPRESS layers) ← v66's phases 2,3 cloned (later-layer distillation lean)
#   - WaistController, codebook, notebook, boundary head, lookup table → v66 unchanged
#   - per_head_pitch: 4→6 positions; v23a formula re-inits all 6 (small mismatch acceptable)
#
# Training: LR=2e-5 (gentler than v66's 3e-5; new component is Set B's independent shared weights).
# Eval: USE_KV_CACHE=0 (cached_generate_segmented Stage 1 closure not yet TWO_PHASE-aware).

set -e
cd "$(dirname "$0")/.."

if [ ! -f ".cache/gsm8k_steps_ckpts/v66_step3000_two_phase.safetensors" ]; then
    echo "ERROR: patched ckpt not found — run scripts/patch_v66_for_two_phase.py first"
    exit 1
fi

# v68 — TWO_PHASE structural architecture
export TWO_PHASE=1
export EXPAND_LAYERS=4
export COMPRESS_LAYERS=2
export EXPAND_TEMP=2.0
export COMPRESS_TEMP=0.7
export SINE_TEMP=0                    # DISABLED — temperature is structural now

# v66 fixes (carry forward — all architectural, all compatible)
export PROMPT_REFRESH_ALPHA=0.1
export BOUNDARY_AUX_WEIGHT=0.1
export BOUNDARY_POS_WEIGHT=5.0
export SCHED_SAMPLE_RATE=0.3

# Architecture (matches v66 except temperature)
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
export CONSTANT_RADIUS=1
export BREATH_TIME_EMBED=1
export BREATH_TIME_INIT_SCALE=0.0
export CROSS_BREATH_HANDOFF=1
export ABLATE_BREATH_ROTATION=1
export QUADRATURE_HEADS=0

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
export LR=2e-5                        # SAFER than v66's 3e-5; new Set B shared weights
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
export USE_KV_CACHE=0                 # disabled — Stage 1 JIT not TWO_PHASE-aware yet
export CKPT_LABEL=v68_two_phase
export RESUME_FROM=.cache/gsm8k_steps_ckpts/v66_step3000_two_phase_fixb.safetensors

/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py 2>&1 | tee .cache/v68_two_phase.log
