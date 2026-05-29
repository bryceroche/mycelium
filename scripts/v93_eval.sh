#!/bin/bash
# v93 eval — wrapper that calls v92_eval.sh on a v93 ckpt.
# The v93 architecture is byte-identical to v92 (only weights differ — cold start).
set -e
cd "$(dirname "$0")/.."

V93_CKPT="${V93_CKPT:-.cache/gsm8k_steps_ckpts/v93_cold_step500.safetensors}"
NUM_EVAL="${NUM_EVAL:-60}"

export V92_CKPT="$V93_CKPT"
export NUM_EVAL="$NUM_EVAL"
exec /home/bryce/mycelium/scripts/v92_eval.sh
