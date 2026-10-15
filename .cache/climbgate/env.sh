# THE GOVERNOR CLIMB gate env (2026-10-14): breathstop_merge_gate.sh's COMMON + S8 verbatim (tiny64 / testtiny64,
# WARM_FROM balV242, LR=1e-4 SEED=242), pointed at this worktree. CPU only (DEV=CPU; never the card).
WT=/home/bryce/mycelium/.claude/worktrees/agent-a6fbba480f6d464d0
PY=/home/bryce/mycelium/.venv/bin/python3
G=$WT/.cache/climbgate
S8="ALG_ROUTER=2 R_GAIN_INIT=1.0 ALG_FREEZE=r_gain ALG_ROUTER_PTR=0.0 ALG_SPAN_ALL=1 ALG_SPAN_ARGS=1 ALG_SPAN_OP=1 ALG_SPAN_RCUE=1 ALG_SPAN_ARCUE=1 ALG_PTR_SURF=role:add:2.0 BIND_CODES=.cache/bindbus_codes512r.npz"
COMMON="ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_BREATH=7 ALG_NOTEBOOK=1 ALG_SIXWAVE=1 NB_PERSLOT=1 ALG_BINDBUS=7 ALG_BIND_D=512 BIND_CODES=.cache/bindbus_codes512.npz ALG_BUSGARAGE=2 ALG_SHELF_CIRCLE=2 ALG_ALTMASK=1 ALG_ALT21=1 ALG_ALT2=1 ALG_MASKHEAD=1 ALG_FED=1 ALG_POLAR=1 ALG_POLAR_D=128 ALG_POLAR_EM=0.1 ALG_POLAR_D_INIT=.cache/polar_waist_init_d128u.npz ALG_PRUNE=pforms,s4,fednl0,lane2 ALG_SLOT_ALL=1 ALG_STELLAR=2 ALG_CLOCK_CANON=1 SC_EVAL=0 ALG_ALLOW_PEN_TRAIN=1 ALG_TRAIN=.cache/form_tiny64.jsonl ALG_TRAIN_NAME=tiny64 ALG_TEST=.cache/test_tiny64.jsonl ALG_TEST_NAME=testtiny64 ALG_MASKPREP_CACHE=1 ALG_MASKPREP_IGNORE=ALG_MASK_COOK,ALG_MASK_COOK_SKEL,ALG_TOK_COOK,ALG_TOK_COOK_S3,ALG_TG_INIT,ALG_BAL_COOK,ALG_NB2,ALG_ALT5,ALG_PRUNE,ALG_MASKPREP_JIT,ALG_MASKPREP_B,ALG_FACTS_POOL,ALG_FACTS_FLUSH,ALG_JIT_VAL LR=1e-4 SEED=242 WARM_FROM=.cache/sharp_balV242.safetensors SNAP_EVERY=4000"
# run_cfg NAME "ENV..." [STEPS] -> $G/gate_NAME.log, ckpt $G/ckpt_NAME.safetensors
run_cfg() { N=$1; V=$2; ST=${3:-2}
  cd $WT && env PYTHONPATH=$WT DEV=CPU $COMMON $V BATCH=2 STEPS=$ST ALG_CKPT=.cache/climbgate/ckpt_$N.safetensors $PY scripts/phase1_algebra_head.py --train > $G/gate_$N.log 2>&1
  echo "gate $N: $(grep -aoE 'step +[0-9]+ loss=[0-9.]+' $G/gate_$N.log | tr '\n' ';') $(grep -aoE '\([0-9.]+s/step' $G/gate_$N.log | tail -1) $(grep -aE 'Error|Traceback|assert' $G/gate_$N.log | head -1 | cut -c1-200)"
}
