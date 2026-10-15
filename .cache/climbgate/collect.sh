# Collect the climb gate's evidence into .cache/climb_gate.log (the deliverable).
WT=/home/bryce/mycelium/.claude/worktrees/agent-a6fbba480f6d464d0
G=$WT/.cache/climbgate
cd $WT
{
echo "### THE GOVERNOR CLIMB CPU GATE (2026-10-14) -- worktree branch $(git branch --show-current)"
echo "### REFERENCES on the unmodified code e5053d99 (ref.log)"
cat $G/ref.log
echo
echo "### GATE (gate.log; the first run, gate_run1.log, died at the feed-door fence -> fixed in f7c4e22a)"
grep -v "^    \[climb\] targets" $G/gate.log
echo
echo "### TARGETS on the fixture (climb_build_targets)"
grep -a "\[climb\] targets" $G/gate_climbk3.log
echo
echo "### READ SMOKE (read_smoke.log)"
grep -aE "climb-read\]   |READ SMOKE|^==" $G/read_smoke.log
} > $WT/.cache/climb_gate.log
wc -l $WT/.cache/climb_gate.log
