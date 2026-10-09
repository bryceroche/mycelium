# THE DIRBIT gate (2026-10-08): the champion recipe's exact fixture (tiny64 / WARM_FROM balV242 /
# BATCH=2 STEPS=2, DEV=CPU), pointed at worktree mycelium-wt9 (branch dirbit) — reference configs
# that must reproduce to the digit: unset 5.2995/0.0279 (bit-identical, no ALG_DIR code on the
# default-off path); role8 6.5535/1.1061; hierd 10.8656/8.8945 (all three bit-identical — ALG_DIR=0
# everywhere, h_dir/h_dir_b never allocated, _heads_of's dir branch never taken). Then the new
# configs, which must RUN and DIFFER from their ALG_DIR-unset parents: dir = role8 + ALG_DIR=1 (THE
# DIRECTION BIT's own BCE term + the res-pointer structural mask, soft-by-p at train); dirhier =
# hierd + ALG_DIR=1 (same knob under THE HIERARCHICAL STATE).
# 2026-10-09 (FORM 2, word given after FORM 1 closed negative on three arms, ledger 03:1x): dir2 =
# role8 + ALG_DIR=2 (THE TEACHER-FORCED BIT: the res mask uses GOLD arg_dir/dir_mask/args at train
# via the host-fed dirgold buffer, b_dirgold; confidence-gated by the model's own p(inverse) at read,
# ALG_DIR_CONF default 0.8) — must RUN and DIFFER from dir (6.5288/1.0316), since the train-time mask
# is now gold-driven rather than p-driven.
cd /home/bryce/mycelium-wt9; exec > .cache/dirbit_gate.log 2>&1
PYTHONPATH=/home/bryce/mycelium-wt9 DEV=CPU .venv/bin/python3 scripts/complex_fence.py 2>&1 | tail -1
S8="ALG_ROUTER=2 R_GAIN_INIT=1.0 ALG_FREEZE=r_gain ALG_ROUTER_PTR=0.0 ALG_SPAN_ALL=1 ALG_SPAN_ARGS=1 ALG_SPAN_OP=1 ALG_SPAN_RCUE=1 ALG_SPAN_ARCUE=1 ALG_PTR_SURF=role:add:2.0 BIND_CODES=.cache/bindbus_codes512r.npz"
HD="ALG_HIER_READ=1 ALG_HIER_WAIST=1 ALG_HIER_DAMP=2,4,0 ALG_HIER_TAU=1.0"
for CFG in "unset:" "role8:$S8" "hierd:$S8 $HD" "dir:$S8 ALG_DIR=1" "dirhier:$S8 $HD ALG_DIR=1" "dir2:$S8 ALG_DIR=2"; do N=${CFG%%:*}; V=${CFG#*:}
  env PYTHONPATH=/home/bryce/mycelium-wt9 DEV=CPU ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_BREATH=7 ALG_NOTEBOOK=1 ALG_SIXWAVE=1 NB_PERSLOT=1 ALG_BINDBUS=7 ALG_BIND_D=512 BIND_CODES=.cache/bindbus_codes512.npz ALG_BUSGARAGE=2 ALG_SHELF_CIRCLE=2 ALG_ALTMASK=1 ALG_ALT21=1 ALG_ALT2=1 ALG_MASKHEAD=1 ALG_FED=1 ALG_POLAR=1 ALG_POLAR_D=128 ALG_POLAR_EM=0.1 ALG_POLAR_D_INIT=.cache/polar_waist_init_d128u.npz ALG_PRUNE=pforms,s4,fednl0,lane2 ALG_SLOT_ALL=1 ALG_STELLAR=2 ALG_CLOCK_CANON=1 SC_EVAL=0 ALG_ALLOW_PEN_TRAIN=1 ALG_TRAIN=.cache/form_tiny64.jsonl ALG_TRAIN_NAME=tiny64 ALG_TEST=.cache/test_tiny64.jsonl ALG_TEST_NAME=testtiny64 ALG_MASKPREP_CACHE=1 ALG_MASKPREP_IGNORE=ALG_MASK_COOK,ALG_MASK_COOK_SKEL,ALG_TOK_COOK,ALG_TOK_COOK_S3,ALG_TG_INIT,ALG_BAL_COOK,ALG_NB2,ALG_ALT5,ALG_PRUNE,ALG_MASKPREP_JIT,ALG_MASKPREP_B,ALG_FACTS_POOL,ALG_FACTS_FLUSH,ALG_JIT_VAL LR=1e-4 SEED=242 WARM_FROM=.cache/sharp_balV242.safetensors SNAP_EVERY=4000 $V BATCH=2 STEPS=2 ALG_CKPT=.cache/sharp_dirbitgate.safetensors .venv/bin/python3 scripts/phase1_algebra_head.py --train > .cache/dirbit_gate_$N.log 2>&1
  echo "gate $N: $(grep -aoE 'step +[01] loss=[0-9.]+' .cache/dirbit_gate_$N.log | tr '\n' ';') $(grep -aoE '\([0-9.]+s/step\)' .cache/dirbit_gate_$N.log | tr '\n' ' ') $(grep -aE 'Error|Traceback|assert' .cache/dirbit_gate_$N.log | head -1 | cut -c1-200)"
  if [ "$N" = dir ]; then cp .cache/sharp_dirbitgate.safetensors .cache/sharp_dirbitgate_dir.safetensors; fi   # kept for the read-path smoke
  if [ "$N" = dir2 ]; then cp .cache/sharp_dirbitgate.safetensors .cache/sharp_dirbitgate_dir2.safetensors; fi   # kept for the read-path smoke
done; rm -f .cache/sharp_dirbitgate.safetensors; echo "DIRBIT GATE COMPLETE"
