#!/bin/bash
# v62 — Pythia-1B base + decoupled WaistController + GSM8K_STEPS.
#
# "Bigger gateway, same bottleneck" design from memory:
#   - Base hidden: 1024 → 2048 (Pythia-1B L0-L3)
#   - n_heads: 16 → 8  (Pythia-1B has 8 heads × 256 head_dim)
#   - head_dim: 64 → 256
#   - FFN: 4096 → 8192
#   - controller_hidden: 1024 (DECOUPLED from base; keeps controller compact)
#   - Waist: stays at 512 (the per-step IB bottleneck — unchanged paradigm)
#
# Expected: ~400M trainable, ~4× per-step wallclock vs v60-take-2.
#
# Prerequisite: Pythia-1B weights at .cache/pythia-1b/model.safetensors.
# Download (one-time):
#   mkdir -p .cache/pythia-1b
#   /home/bryce/mycelium/.venv/bin/python -c "
#       from huggingface_hub import hf_hub_download
#       p = hf_hub_download(repo_id='EleutherAI/pythia-1b', filename='model.safetensors',
#                            local_dir='.cache/pythia-1b')
#       print(p)"
#
# Warm-start: v60-take-2 step 6000 (when ready) — BUT some weights won't transfer
# due to shape differences (transformer at H=2048, controller's K/V projs at
# H=2048). With strict=False, only matching-shape weights load. For the
# Pythia-1B layers themselves, _load_state pulls from .cache/pythia-1b/.

set -e
cd "$(dirname "$0")/.."

if [ ! -f ".cache/pythia-1b/model.safetensors" ]; then
    echo "ERROR: Pythia-1B weights not found at .cache/pythia-1b/model.safetensors"
    echo "Download via huggingface-cli or hf_hub_download (see comment above)."
    exit 1
fi

if [ ! -f ".cache/gsm8k_steps_v1_train.jsonl" ]; then
    echo "ERROR: GSM8K train JSONL not found."
    exit 1
fi

# v62-specific: Config overrides for Pythia-1B dims
export HIDDEN=2048
export N_HEADS=8
export HEAD_DIM=256
export FFN=8192
export CONTROLLER_HIDDEN=1024
export PYTHIA_WEIGHTS=.cache/pythia-1b/model.safetensors

# v55..v59 architecture (carry forward)
export CONTROLLER_DECODE=1
export CONTROLLER_N_LAYERS=2
export PER_BREATH_DECODE=1
export BFIELD_WAIST=512                    # waist STAYS at 512 — same per-step bottleneck
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

# Training — GSM8K_STEPS, bucketed by K
export DEV=PCI+AMD
export LEVEL=GSM8K_STEPS
export GSM8K_STEPS_PATH=.cache/gsm8k_steps_v1_train.jsonl
export GSM8K_STEPS_MIN_K=2
export GSM8K_STEPS_MAX_K=6
export SPACE_DIGITS=1
export BATCH=1                             # 4× per-token cost at H=2048; drop to B=1 for K=6 safety
export FIXED_LEN=320
export STEPS=3000                          # resume from step1000 (=effective step 3000) → 6000 effective
export LR=1e-5                             # gentle for the bigger model
export SEED=43                             # different from default (no effect on NaN though — issue was structural)
export GRAD_CLIP=1.0                       # NEW — global-norm gradient clip, prevents NaN spikes at H=2048
                                           # take-1 NaN'd at step 2420 with LR=3e-5
                                           # take-2 NaN'd at step 2420 with LR=1e-5 (seed=42)
                                           # take-3 NaN'd at step 1760 with LR=1e-5 (seed=43)
                                           # → not seed-specific. Need structural fix: gradient clipping.
export TRAIN_LOOPS=4
export EVAL_LOOPS=1,2,3,4
export ACC_EVAL_EVERY=10000
export SKIP_FINAL_ACC=1
export CKPT_EVERY=1000                     # 6 ckpts across the 6k-step run
export NUM_EVAL=100
export NUM_PROBLEMS=20000
export EVAL_BATCH=2
export EVAL_CACHE_LEN=328
export LOOKUP_AUX_WEIGHT=0.1
export USE_JIT=1
export USE_KV_CACHE=1
export CKPT_LABEL=v62_pythia1b_gsm8k
# Resume from step 2000 (the last healthy ckpt before NaN divergence at step 2420
# of the first take with LR=3e-5). With LR=1e-5 we should stay stable.
export RESUME_FROM=.cache/gsm8k_steps_ckpts/v62_pythia1b_gsm8k_step1000.safetensors
# This is the LATEST healthy ckpt (take-3's step 1000 = effective total step 3000:
# trained on Pythia-1B base for 2000 steps in take-1, then 1000 more in take-3
# with LR=1e-5 before take-3 NaN'd at step 1760).

/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py 2>&1 | tee .cache/v62_pythia1b_gsm8k.log
