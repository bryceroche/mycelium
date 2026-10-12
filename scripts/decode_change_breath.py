"""decode_change_breath.py -- THE DECODE-CHANGE-AFTER-BREATH-N COUNT (registered in the K3_241/
PK_241 chains, ledger 2026-10-11 16:48; "scripts/decode_change_breath.py from the breathstop
builder, if merged by then -- else your own" -- not yet built by that lineage at the time this
RTB chain needed it, so built here, worktree mycelium-wt28, branch rtb).

Usage: decode_change_breath.py <dump_a.pkl> <dump_b.pkl>

Each pkl is a loop_val.py LV_DUMP list of per-gold-slot tuples:
  (i, j, gold_ftype, gold_op, gold_args, gold_res, gold_digits,
   pred_ftype, pred_op, pred_args_top2, pred_res, pred_digits, pred_pres, pred_dup)
from two reads of the SAME checkpoint on the SAME rows at two different breath counts (e.g.
ALG_LOOP_STOP_READ=3 vs =7, or breath-2 vs breath-6 -- whatever the caller dumped). Reports how
many (row, slot) predictions CHANGE discrete decode (ftype/op/res/args-top1/digits-argmax)
between the two dumps -- a nonzero count means the later breaths are doing real decode work on
this checkpoint; zero means the earlier breath already decided every slot.
"""
import pickle
import sys


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def key_pred(rec):
    (i, j, g_ft, g_op, g_args, g_res, g_dig,
     p_ft, p_op, p_args2, p_res, p_dig, p_pres, p_dup) = rec
    return (i, j), (p_ft, p_op, tuple(p_args2[:1]), p_res, tuple(p_dig), p_pres, p_dup)


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    a, b = load(a_path), load(b_path)
    am = {}
    for rec in a:
        k, v = key_pred(rec); am[k] = v
    bm = {}
    for rec in b:
        k, v = key_pred(rec); bm[k] = v
    common = sorted(set(am) & set(bm))
    missing_a = set(bm) - set(am); missing_b = set(am) - set(bm)
    n_diff = sum(1 for k in common if am[k] != bm[k])
    print(f"[decode-change] {a_path} vs {b_path}: {len(common)} (row,slot) pairs compared, "
          f"{n_diff} decode changes ({100.0 * n_diff / max(len(common), 1):.1f}%)")
    if missing_a or missing_b:
        print(f"  (note: {len(missing_a)} pairs only in b, {len(missing_b)} only in a -- "
              f"not compared)")


if __name__ == "__main__":
    main()
