#!/usr/bin/env bash
# ib_run_when_ready.sh — wait for the factor-graph labeling to finish, then run
# the IB clustering pipeline on the completed data.
#
# Usage: bash scripts/ib_run_when_ready.sh [LABELING_PID]
#   LABELING_PID (optional) — PID of the running labeling process.
#                              If omitted, watches the JSONL record count instead.
#
# Env overrides:
#   SRC          — JSONL path (default: .cache/gsm8k_factor_graphs_train.jsonl)
#   MIN_RECORDS  — minimum records to consider labeling complete (default: 4500)
#   LOG          — log file (default: /tmp/ib_clustering_run.log)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"

SRC="${SRC:-${PROJECT_ROOT}/.cache/gsm8k_factor_graphs_train.jsonl}"
MIN_RECORDS="${MIN_RECORDS:-4500}"
LOG="${LOG:-/tmp/ib_clustering_run.log}"
LABELING_PID="${1:-}"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$LOG"
}

record_count() {
    wc -l < "$SRC" 2>/dev/null || echo 0
}

log "=== ib_run_when_ready.sh starting ==="
log "SRC=$SRC"
log "MIN_RECORDS=$MIN_RECORDS"
log "LOG=$LOG"
log "LABELING_PID=${LABELING_PID:-none}"

# ── Wait for labeling to finish ─────────────────────────────────────────────
if [[ -n "$LABELING_PID" ]]; then
    log "Waiting for PID $LABELING_PID to exit..."
    while kill -0 "$LABELING_PID" 2>/dev/null; do
        cnt=$(record_count)
        log "  still running — current records: $cnt"
        sleep 30
    done
    log "PID $LABELING_PID has exited."
else
    log "No PID given — polling record count until >= $MIN_RECORDS..."
    while true; do
        cnt=$(record_count)
        log "  current records: $cnt / $MIN_RECORDS"
        if [[ "$cnt" -ge "$MIN_RECORDS" ]]; then
            log "  -> threshold reached."
            break
        fi
        sleep 60
    done
fi

# ── Validate record count ────────────────────────────────────────────────────
cnt=$(record_count)
log "Final record count: $cnt"
if [[ "$cnt" -lt "$MIN_RECORDS" ]]; then
    log "[WARN] Only $cnt records found (expected >= $MIN_RECORDS). Proceeding anyway."
fi

# ── Run clustering ───────────────────────────────────────────────────────────
log "Running ib_cluster_factor_graphs.py (full run)..."
"$VENV_PYTHON" "$SCRIPT_DIR/ib_cluster_factor_graphs.py" \
    --src "$SRC" \
    2>&1 | tee -a "$LOG"

log "ib_cluster_factor_graphs.py done."

# ── Run verification ─────────────────────────────────────────────────────────
log "Running ib_verify_clustering.py..."
"$VENV_PYTHON" "$SCRIPT_DIR/ib_verify_clustering.py" \
    2>&1 | tee -a "$LOG"

log "ib_verify_clustering.py done."
log "=== IB clustering pipeline complete ==="
log "Outputs:"
log "  ${PROJECT_ROOT}/.cache/ib_tree_gsm8k.json"
log "  ${PROJECT_ROOT}/.cache/ib_centroids_gsm8k.npz"
log "  ${PROJECT_ROOT}/.cache/var_descriptions_to_leaf.jsonl"
log "  ${PROJECT_ROOT}/.cache/ib_clustering_report_gsm8k.md"
