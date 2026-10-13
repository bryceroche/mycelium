"""decode_change_breath.py -- THE DECODE-CHANGE-AFTER-BREATH-2 COUNT (2026-10-11, PK_241's own
falsifiable read, Opus's spec via the ledger: "do the grown breaths refine (the drift falls, the
decode changes after breath 2) or repeat (nothing changes = the loop depth is decoration past
breath 2)"). No reader for this existed; written here rather than assumed.

NOT a new forward()/decode() reimplementation (that would risk silently drifting from the real
two-pass masked protocol every other read uses -- the open pass-1-then-build_slot_masks-then-
pass-2 cycle loop_val.py already runs). Instead: a thin comparator over TWO existing LV_DUMP
pickles from loop_val.py's OWN real read() -- one taken with ALG_LOOP_STOP_READ=<k_lo> (default
3: breaths 0,1,2 -- "after breath 2"), one with ALG_LOOP_STOP_READ=<k_hi> (default 7: the full
unroll) -- same checkpoint, same env, same fixture, only the breath count at THIS read differs.
Each pickle is loop_val.py's LV_DUMP list of (i, j, gold_ftype, gold_op, gold_args, gold_res,
gold_digits, pred_ftype, pred_op, pred_args_top2, pred_res, pred_dig, pred_pres, pred_dup) tuples,
one per (row, GOLD slot) -- so this counts decode changes restricted to gold slots, the exact
convention matched_read.py already uses for "the matched read" elsewhere in these chains.

usage:
  LV_DUMP=.cache/dump_wild_K3_241_lo.pkl ALG_LOOP_STOP_READ=3 ... .venv/bin/python3 scripts/loop_val.py
  LV_DUMP=.cache/dump_wild_K3_241_hi.pkl ALG_LOOP_STOP_READ=7 ... .venv/bin/python3 scripts/loop_val.py
  .venv/bin/python3 scripts/decode_change_breath.py .cache/dump_wild_K3_241_lo.pkl .cache/dump_wild_K3_241_hi.pkl
"""
import pickle
import sys

lo_path, hi_path = sys.argv[1], sys.argv[2]
lo = pickle.load(open(lo_path, "rb"))
hi = pickle.load(open(hi_path, "rb"))

lo_by_ij = {(t[0], t[1]): t for t in lo}
hi_by_ij = {(t[0], t[1]): t for t in hi}
common = set(lo_by_ij) & set(hi_by_ij)
assert common, "no (row, slot) keys in common between the two dumps -- mismatched fixtures?"
if len(lo_by_ij) != len(hi_by_ij) or len(common) != len(lo_by_ij):
    print(f"[decode-change] WARNING: dump sizes differ (lo={len(lo_by_ij)} hi={len(hi_by_ij)} "
          f"common={len(common)}) -- comparing the intersection only")

PRED_FIELDS = slice(7, 14)   # pred_ftype, pred_op, pred_args_top2, pred_res, pred_dig, pred_pres, pred_dup

n_changed = 0
rows_changed = set()
field_changed = {"ftype": 0, "op": 0, "args": 0, "res": 0, "dig": 0, "pres": 0, "dup": 0}
field_names = list(field_changed)
for key in common:
    a, b = lo_by_ij[key][PRED_FIELDS], hi_by_ij[key][PRED_FIELDS]
    if a != b:
        n_changed += 1
        rows_changed.add(key[0])
        for name, av, bv in zip(field_names, a, b):
            if av != bv:
                field_changed[name] += 1

n_rows = len({k[0] for k in common})
print(f"[decode-change] comparing {lo_path} (lo) vs {hi_path} (hi), {len(common)} gold slots, {n_rows} rows")
print(f"[decode-change] slots changed: {n_changed}/{len(common)} = {n_changed / len(common):.4f}")
print(f"[decode-change] rows with >=1 changed slot: {len(rows_changed)}/{n_rows} = {len(rows_changed) / n_rows:.4f}")
print(f"[decode-change] per-field change counts: " + " ".join(f"{k}={v}" for k, v in field_changed.items()))
