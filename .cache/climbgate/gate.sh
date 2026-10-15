# THE GOVERNOR CLIMB's CPU gate (2026-10-14). Fixture: breathstop_merge_gate.sh's COMMON + S8 (tiny64 /
# testtiny64, WARM_FROM balV242, BATCH=2, LR=1e-4 SEED=242, DEV=CPU). Climb targets for the fixture:
# .cache/climb_targets_tiny64.npz (scripts/climb_targets_sidecar.py over form_tiny64.jsonl; climb_targets.py's
# own functions). REFERENCES banked on the unmodified code e5053d99 (ref.log): role8 6.5535/1.1061, k3
# 6.6232/1.1444, sort4 60.7649/18.6915, sort4k3 52.7963/9.7296.
# GATES: (1) unset bit-identity (role8, k3); (2) ALG_SORT=4 reproduces its own pre-change losses (sort4,
# sort4k3); (3) ALG_SORT=4 ALG_CLIMB=1: finite, CE printed separately; additivity at FIXED weights (LR=0:
# post-update == pre-update, so total must equal the sort4 LR=0 loss + W x CE); nonzero grad on sort0/1/2
# wq/wk at step 0 (JIT=0 probe; ReZero zeroes the main loss's share, sort3 must read 0); the PLACEBO runs
# and gives a different CE; (4) the trainer refuses a DIAGNOSTIC target path (direct and via a symlink);
# (5) 20 CPU steps: the climb CE falls; the per-step cost vs sort4 alone (both K3, run side by side).
set -o pipefail
source /home/bryce/mycelium/.claude/worktrees/agent-a6fbba480f6d464d0/.cache/climbgate/env.sh
exec > $G/gate${GATE_TAG}.log 2>&1
echo "code at $(cd $WT && git rev-parse --short HEAD) dirty=$(cd $WT && git status --porcelain scripts | wc -l) $(date)"
CT="ALG_CLIMB_TARGETS=.cache/climb_targets_tiny64.npz ALG_CLIMB_LOG=1"
climbline() { grep -a "\[climb\] step" $G/gate_$1.log | tr -s ' ' | tr '\n' '|'; }

if [ -z "$SKIP12" ]; then
echo "== (1)+(2) UNSET / SORT4 BIT-IDENTITY (BATCH=2 STEPS=2) =="
run_cfg role8 "$S8" &
run_cfg k3 "$S8 ALG_LOOP_STOP=3" &
run_cfg sort4 "$S8 ALG_SORT=4" &
run_cfg sort4k3 "$S8 ALG_SORT=4 ALG_LOOP_STOP=3" &
wait
else echo "(1)+(2) banked in gate_run1.log (same unset/sort4 code path)"; fi

echo "== (3) THE CLIMB (STEPS=2) + THE PLACEBO; (4) THE DIAGNOSTIC FENCE =="
run_cfg climb "$S8 ALG_SORT=4 ALG_CLIMB=1 $CT" &
run_cfg climbk3 "$S8 ALG_SORT=4 ALG_LOOP_STOP=3 ALG_CLIMB=1 $CT" &
run_cfg placebok3 "$S8 ALG_SORT=4 ALG_LOOP_STOP=3 ALG_CLIMB=1 ALG_CLIMB_PLACEBO=1 $CT" &
ln -sf $WT/.cache/climb_targets_wild_DIAGNOSTIC_ONLY.npz $G/innocent_targets.npz
wait
run_cfg diagdirect "$S8 ALG_SORT=4 ALG_CLIMB=1 ALG_CLIMB_TARGETS=.cache/climb_targets_wild_DIAGNOSTIC_ONLY.npz" &
run_cfg diaglink "$S8 ALG_SORT=4 ALG_CLIMB=1 ALG_CLIMB_TARGETS=.cache/climbgate/innocent_targets.npz" &
wait
for N in climb climbk3 placebok3; do echo "  $N climb: $(climbline $N)"; grep -a "\[climb\] targets" $G/gate_$N.log | sed "s/^/    /"; done
for N in diagdirect diaglink; do echo "  $N refused: $(grep -a 'DIAGNOSTIC-ONLY' $G/gate_$N.log | head -1 | cut -c1-220)"; done

echo "== (3b) ADDITIVITY AT FIXED WEIGHTS (LR=0, STEPS=1) + THE GRAD PROBE (JIT=0, STEPS=1) =="
run_cfg lr0_sort4k3 "$S8 ALG_SORT=4 ALG_LOOP_STOP=3 LR=0" 1 &
run_cfg lr0_climbk3 "$S8 ALG_SORT=4 ALG_LOOP_STOP=3 ALG_CLIMB=1 $CT LR=0" 1 &
run_cfg lr0_placebok3 "$S8 ALG_SORT=4 ALG_LOOP_STOP=3 ALG_CLIMB=1 ALG_CLIMB_PLACEBO=1 $CT LR=0" 1 &
wait
run_cfg gradk3 "$S8 ALG_SORT=4 ALG_LOOP_STOP=3 ALG_CLIMB=1 $CT ALG_CLIMB_GRAD_PROBE=1 JIT=0" 1 &
wait
for N in lr0_climbk3 lr0_placebok3; do echo "  $N climb: $(climbline $N)"; done
grep -a "\[climb-grad\]" $G/gate_gradk3.log | sed "s/^/  /"

echo "== (5) 20 CPU STEPS: the climb CE falls; per-step cost vs sort4 alone (K3, side by side) =="
run_cfg cost20_sort4k3 "$S8 ALG_SORT=4 ALG_LOOP_STOP=3" 20 &
run_cfg cost20_climbk3 "$S8 ALG_SORT=4 ALG_LOOP_STOP=3 ALG_CLIMB=1 $CT" 20 &
wait
grep -a "\[climb\] step" $G/gate_cost20_climbk3.log | sed "s/^/  /"
echo "GATE COMPLETE $(date)"
