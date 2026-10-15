# REFERENCE bank on the UNMODIFIED code (e5053d99) before any climb edit.
source /home/bryce/mycelium/.claude/worktrees/agent-a6fbba480f6d464d0/.cache/climbgate/env.sh
exec > $G/ref.log 2>&1
cp $WT/scripts/phase1_algebra_head.py $G/ref_phase1_algebra_head.py   # freeze the code under test
echo "code at $(cd $WT && git rev-parse --short HEAD) dirty=$(cd $WT && git status --porcelain scripts | wc -l)"
run_cfg ref_role8 "$S8" &
run_cfg ref_k3 "$S8 ALG_LOOP_STOP=3" &
run_cfg ref_sort4 "$S8 ALG_SORT=4" &
run_cfg ref_sort4k3 "$S8 ALG_SORT=4 ALG_LOOP_STOP=3" &
wait
echo REF DONE
