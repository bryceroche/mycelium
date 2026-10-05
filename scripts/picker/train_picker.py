"""scripts/picker/train_picker.py -- THE PANEL PICKER's training + 5-fold-by-row CV + ablation
(2026-10-05, delegate). Trains a small logistic ranker (scikit-learn) on the TRAINING-DIET candidate
feature table (never wild, never MATH-500 -- CLAUDE.md's standing rule) to score P(candidate is the
keyed graph); within a row, the picker's pick is argmax predicted score. Reports diet CV rows-right vs
the diet's own top-1 and consistency-judge baselines (computed identically on the diet), and an
ablation table (each angle group removed). Also fits the FULL model on all diet rows and saves it
(scaler + logistic weights + feature order) for scripts/picker/wild_read.py to apply, unchanged, to the
wild candidates.

Usage: .venv/bin/python3 scripts/picker/train_picker.py <diet_features.pkl> <out_model.pkl> <out_report.txt>
"""
import sys
import pickle

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from build_features import FEATURE_NAMES, ANGLE_GROUPS


def to_matrix(cands, feature_names):
    X = np.array([[c["X"][f] for f in feature_names] for c in cands], dtype=np.float64)
    y = np.array([1.0 if c["y"] else 0.0 for c in cands])
    rows = np.array([c["row"] for c in cands])
    return X, y, rows


def group_rows(cands):
    by_row = {}
    for c in cands:
        by_row.setdefault(c["row"], []).append(c)
    return by_row


def pick_argmax(cands_for_row, scores):
    j = int(np.argmax(scores))
    return cands_for_row[j]


def cv_rows_right(cands, feature_names, n_splits=5, C=1.0):
    """5-fold GroupKFold by row: fit on train folds' candidates, predict on test folds' candidates,
    within each test row pick argmax predicted P(correct); return (rows_right, n_rows, per_row_pick)."""
    X, y, rows = to_matrix(cands, feature_names)
    uniq_rows = np.unique(rows)
    gkf = GroupKFold(n_splits=min(n_splits, len(uniq_rows)))
    per_row_correct = {}
    for train_idx, test_idx in gkf.split(X, y, groups=rows):
        scaler = StandardScaler().fit(X[train_idx])
        Xtr = scaler.transform(X[train_idx]); Xte = scaler.transform(X[test_idx])
        clf = LogisticRegression(max_iter=2000, C=C, class_weight="balanced")
        clf.fit(Xtr, y[train_idx])
        scores_te = clf.decision_function(Xte)
        test_rows = rows[test_idx]
        by_row = {}
        for k, ridx in enumerate(test_idx):
            by_row.setdefault(test_rows[k], []).append((scores_te[k], y[test_idx[k]]))
        for r, lst in by_row.items():
            best = max(lst, key=lambda t: t[0])
            per_row_correct[r] = bool(best[1] > 0.5)
    n_rows = len(per_row_correct)
    rows_right = sum(1 for v in per_row_correct.values() if v)
    return rows_right, n_rows, per_row_correct


def top1_rows_right(cands):
    by_row = group_rows(cands)
    right = 0
    for r, lst in by_row.items():
        for c in lst:
            if c.get("is_top1"):
                if c["y"]:
                    right += 1
                break
    return right, len(by_row)


def judge_rows_right(cands):
    """combined_oracle.py's consistency judge (solved+unique preferred, tie by loglik), replicated here
    over the feature table's own fields (status_solved/unique/unique_known already encode this)."""
    by_row = group_rows(cands)
    right = 0
    for r, lst in by_row.items():
        solved = [c for c in lst if c["X"]["status_solved"] > 0.5]
        if not solved:
            continue   # refused -> not correct
        uniq = [c for c in solved if c["X"]["unique"] > 0.5]
        pool = uniq if uniq else solved
        best = max(pool, key=lambda c: c["loglik_raw"])
        if best["y"]:
            right += 1
    return right, len(by_row)


def ablation_table(cands, log=print):
    log("")
    log("ABLATION (each angle group removed; 5-fold CV rows-right on the diet):")
    full_right, full_n, _ = cv_rows_right(cands, FEATURE_NAMES)
    log(f"  {'FULL':10s} {full_right:4d}/{full_n}")
    for g, names in ANGLE_GROUPS.items():
        kept = [f for f in FEATURE_NAMES if f not in names]
        right, n, _ = cv_rows_right(cands, kept)
        log(f"  -{g:9s} {right:4d}/{n}  (delta {right - full_right:+d})")
    return full_right, full_n


def main():
    diet_pkl, out_model, out_report = sys.argv[1], sys.argv[2], sys.argv[3]
    cands = pickle.load(open(diet_pkl, "rb"))
    lines = []
    def log(s=""):
        print(s); lines.append(s)

    n_rows = len(group_rows(cands))
    log(f"[train-picker] diet: {len(cands)} candidates over {n_rows} rows (feature table {diet_pkl})")

    t1_right, t1_n = top1_rows_right(cands)
    j_right, j_n = judge_rows_right(cands)
    log(f"DIET BASELINE  top-1:              {t1_right}/{t1_n} = {t1_right/t1_n:.3f}")
    log(f"DIET BASELINE  consistency judge:  {j_right}/{j_n} = {j_right/j_n:.3f}")

    cv_right, cv_n, _ = cv_rows_right(cands, FEATURE_NAMES)
    log(f"DIET PICKER    5-fold CV by row:    {cv_right}/{cv_n} = {cv_right/cv_n:.3f}")

    full_right, full_n = ablation_table(cands, log)

    # fit the FULL model (all diet rows) for the wild read
    X, y, rows = to_matrix(cands, FEATURE_NAMES)
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(scaler.transform(X), y)
    pickle.dump(dict(scaler=scaler, clf=clf, feature_names=FEATURE_NAMES), open(out_model, "wb"))
    log(f"[train-picker] full model (fit on all {n_rows} diet rows) -> {out_model}")

    open(out_report, "w").write("\n".join(lines) + "\n")
    print(f"[train-picker] wrote {out_report}")


if __name__ == "__main__":
    main()
