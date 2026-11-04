#!/usr/bin/bash
# GPU QUEUE (ledger 11-04; word given): after lftwin ends (it takes gpu.lock per step, so a waiting flock could interleave), G0 then BT8b.
set -eo pipefail; cd /home/bryce/mycelium
while systemctl --user is-active -q lftwin; do sleep 120; done
bash .cache/run_lfg0.sh
bash .cache/run_lfbt8b.sh
