#!/usr/bin/env bash
# Compile a comparison table across all 8 ablation runs (3 Phase 1 + 5 Phase 2/3).
# Reads .cache/ablate_*.log files. Outputs a markdown-ish table to stdout.

set -uo pipefail
cd "$(dirname "$0")/.."

LABELS=(
  ablate_baseline
  ablate_no_ctrl
  ablate_no_lookup
  ablate_const_temp
  ablate_const_step
  ablate_no_rotation
  ablate_no_integration
  ablate_no_notebook
)

extract_acc() {
  # extract a single acc % for the given A=N from the log
  local log=$1
  local n=$2
  grep -E "acc @ A=$n " "$log" 2>/dev/null | tail -1 | sed -E 's/.*: ([0-9.]+)%.*/\1/'
}

extract_lookup() {
  local log=$1
  grep -E "lookup-eval @" "$log" 2>/dev/null | tail -1 | sed -E 's/.*trained=([0-9.]+)%.*/\1/'
}

printf '%-25s | %6s | %6s | %6s | %6s | %s\n' "ablation" "A=1" "A=2" "A=4" "A=8" "lookup"
printf '%-25s-|-%6s-|-%6s-|-%6s-|-%6s-|-%s\n' "-------------------------" "------" "------" "------" "------" "------"
for label in "${LABELS[@]}"; do
  log=".cache/${label}.log"
  if [[ ! -f "$log" ]]; then
    printf '%-25s | %6s | %6s | %6s | %6s | %s\n' "$label" "—" "—" "—" "—" "(no log)"
    continue
  fi
  a1=$(extract_acc "$log" 1)
  a2=$(extract_acc "$log" 2)
  a4=$(extract_acc "$log" 4)
  a8=$(extract_acc "$log" 8)
  lk=$(extract_lookup "$log")
  printf '%-25s | %6s | %6s | %6s | %6s | %s\n' \
    "$label" "${a1:-—}" "${a2:-—}" "${a4:-—}" "${a8:-—}" "${lk:-—}"
done

echo
echo "Reference: ARITH_HARD v1 step 250 (the closest checkpointed comparison)"
echo "  A=1: 47.0%  A=2: 49.0%  A=4: 50.0%  A=8: 48.0%  lookup: 100%"
echo
echo "Reading guide:"
echo "  - Numbers materially below baseline → that component is load-bearing (in this regime)"
echo "  - Numbers tracking baseline → component is decorative / not load-bearing for this dataset"
echo "  - lookup column: drop = lookup-aux supervision / controller learning was doing the work"
