#!/usr/bin/bash
# THE QUEUE DRY-RUN (2026-10-11): every `grep ... || { echo "ABORT ...` check line in a queue script is run
# against a GATE LOG (the builder's own, or the post-merge one) with the log path substituted; each must pass
# or the queue must not be armed. Usage: queue_dryrun.sh <queue.sh> <gate.log> [<log-name-in-queue>]
# (the third arg = the log name the queue's checks reference, e.g. .cache/xxx_merge_gate.log; default: auto-detect)
Q=$1; LOG=$2; REF=${3:-$(grep -oE '\.cache/[a-z0-9_]+_merge_gate\.log' "$Q" | head -1)}
[ -f "$Q" ] && [ -f "$LOG" ] || { echo "usage: $0 <queue.sh> <gate.log> [log-name]"; exit 2; }
fail=0; n=0
while IFS= read -r line; do
  chk=$(echo "$line" | sed 's/ || {.*//; s/ && {.*//')
  [ -n "$REF" ] && chk=${chk//$REF/$LOG}
  n=$((n+1))
  if bash -o pipefail -c "$chk" 2>/dev/null; then echo "ok    $(echo "$chk" | cut -c1-110)"; else echo "FAIL  $(echo "$chk" | cut -c1-110)"; fail=$((fail+1)); fi
done < <(grep -E '^grep -a.*\|\| \{ echo "ABORT' "$Q")
echo "[queue-dryrun] $n checks, $fail failing against $LOG"; [ $fail -eq 0 ]
