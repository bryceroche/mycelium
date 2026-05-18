#!/bin/bash
# v45 ablations — isolate which of the three regularization knobs (WD bump,
# stoch depth, label smoothing) destabilizes the v24c step-500 warm-start.
#
# Background: v45 take 2 (all three knobs at once, with mask safeguard) still
# regressed to ~0% acc by step 125 despite the catastrophic-drop bug being
# fixed. Lookup-eval stayed at 100%, so internal classification is intact;
# generation/output-head must be the failure mode.
#
# Hypothesis: label smoothing pushes the sharply-converged v24c model toward
# a flatter output distribution, damaging autoregressive generation more than
# it regularizes. WD=0.05 (5× v24c's training-time WD=0.01) may compound.
#
# Three single-knob runs from v24c step 500. 250 steps each, eval at 125 and
# 250. Baseline (no knobs) is v24c step 500 itself: 96/94/91 on A=1/4/8.
#
# If a run lands NEAR 96/94/91, that knob is safe.
# If a run lands near 0%, that knob is the destabilizer.

set -e
cd "$(dirname "$0")/.."

# Shared config — exact v24c architecture so the only variable is the reg knob
shared_env() {
    export PER_HEAD_PITCH=1
    export SINE_TEMP=1
    export SINE_TEMP_MAX=2.0
    export SINE_TEMP_MIN=0.7
    export CONSTANT_RADIUS=1
    export BREATH_TIME_EMBED=1
    export BREATH_TIME_INIT_SCALE=0.0
    export CROSS_BREATH_HANDOFF=1
    export ABLATE_BREATH_ROTATION=1
    export NOTEBOOK_V24=1
    export NOTEBOOK_DUAL=1
    export NOTEBOOK_POOL_MODE=attn
    export NOTEBOOK_INIT_SCALE=0.02
    export DEV=PCI+AMD
    export LEVEL=L4_MIXED
    export SPACE_DIGITS=1
    export BATCH=16
    export FIXED_LEN=96
    export STEPS=250
    export LR=3e-5
    export TRAIN_LOOPS=1,2,4,8
    export EVAL_LOOPS=1,4,8
    export ACC_EVAL_EVERY=125
    export CKPT_EVERY=125
    export LOOKUP_AUX_WEIGHT=0.1
    export USE_JIT=1
    export RESUME_FROM=.cache/l4_mixed_ckpts/l4_mixed_v24c_dual_notebook_step500.safetensors
}

# Reset the reg knobs each run (so we don't leak state between runs)
reset_reg_env() {
    export WEIGHT_DECAY=0.01     # v24c baseline
    export LABEL_SMOOTHING=0.0
    export STOCH_DEPTH_P=0.0
}

# ----- Ablation v45c: WD bump only -----
echo ""
echo "========================================"
echo "  v45c: WD=0.05 only (LS=0, SD=0)"
echo "========================================"
shared_env
reset_reg_env
export WEIGHT_DECAY=0.05
export CKPT_LABEL=v45c_wd_only
/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py 2>&1 | tee .cache/l4_mixed_v45c_wd_only.log

# ----- Ablation v45d: stoch depth only -----
echo ""
echo "========================================"
echo "  v45d: SD=0.10 only (LS=0, WD=0.01)"
echo "========================================"
shared_env
reset_reg_env
export STOCH_DEPTH_P=0.10
export CKPT_LABEL=v45d_sd_only
/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py 2>&1 | tee .cache/l4_mixed_v45d_sd_only.log

# ----- Ablation v45e: label smoothing only -----
echo ""
echo "========================================"
echo "  v45e: LS=0.1 only (SD=0, WD=0.01)"
echo "========================================"
shared_env
reset_reg_env
export LABEL_SMOOTHING=0.1
export CKPT_LABEL=v45e_ls_only
/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py 2>&1 | tee .cache/l4_mixed_v45e_ls_only.log

echo ""
echo "========================================"
echo "  Ablations complete. Compare step-125 and step-250 acc:"
echo "    v45c (WD 0.05):  $(grep -A4 'accuracy at step 250' .cache/l4_mixed_v45c_wd_only.log 2>/dev/null | grep 'acc @' | head -3 | tr -d '\n')"
echo "    v45d (SD 0.10):  $(grep -A4 'accuracy at step 250' .cache/l4_mixed_v45d_sd_only.log 2>/dev/null | grep 'acc @' | head -3 | tr -d '\n')"
echo "    v45e (LS 0.1):   $(grep -A4 'accuracy at step 250' .cache/l4_mixed_v45e_ls_only.log 2>/dev/null | grep 'acc @' | head -3 | tr -d '\n')"
echo "  v24c step-500 baseline: 96/94/91 (A=1/4/8)"
echo "========================================"
