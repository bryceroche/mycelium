"""dirrole_read_smoke.py -- THE DIRECTION ROLE's read-vs-train cue parity smoke (2026-10-09,
zero-GPU, CPU-only). Picks 2 rows from the test fixture (.cache/test_tiny64.jsonl, the gate's own
TEST split) and checks that the cue ids computed the TRAIN way (dirrole_build_array, called on a
`samples` list -- do_train's exact call) are byte-identical to the cue ids computed the READ way
(dirrole_row_features, called per-row on raw `text` alone -- loop_val's exact call) on those same
2 rows. Both call sites share ONE function underneath (there is no second matcher to drift out of
sync), so this smoke is proving the WIRING (do_train and loop_val both reach the same code with
the same input), not re-deriving the lexicon match twice independently -- the parity the teacher-
forced bit's death named as the rule: "a structural road trains only in the form the read can
supply."

usage: .venv/bin/python3 scripts/dirrole_read_smoke.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from phase1_algebra_head import T_ALG, dirrole_build_array, dirrole_row_features, DIRROLE_LEXICON

ROWS_PATH = ".cache/test_tiny64.jsonl"
N_ROWS = 2


def main():
    rows = []
    with open(ROWS_PATH) as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= N_ROWS:
                break
    print(f"[dirrole-smoke] {len(rows)} rows from {ROWS_PATH}")

    # THE TRAIN SIDE: do_train's exact call -- dirrole_build_array(samples, T_ALG)
    train_arr = dirrole_build_array(rows, T_ALG)

    # THE READ SIDE: loop_val's exact call -- dirrole_row_features(text, T_ALG) per row
    read_arr = np.stack([dirrole_row_features(r["text"], T_ALG) for r in rows])

    ok = True
    for i, r in enumerate(rows):
        same = np.array_equal(train_arr[i], read_arr[i])
        n_cue_train = int((train_arr[i] > 0).sum())
        n_cue_read = int((read_arr[i] > 0).sum())
        cue_words = sorted({DIRROLE_LEXICON[c - 1] for c in train_arr[i] if c > 0})
        print(f"[dirrole-smoke] row {i}: train/read IDENTICAL={same} "
              f"n_cue_tok train={n_cue_train} read={n_cue_read} cue_words={cue_words}")
        ok = ok and same
    assert ok, "train-side and read-side cue ids diverged on at least one row -- the parity rule is broken"
    print("[dirrole-smoke] PASS: read-side cue ids equal the training-side stamps on all rows")


if __name__ == "__main__":
    main()
