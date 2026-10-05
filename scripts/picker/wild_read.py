"""scripts/picker/wild_read.py -- THE PANEL PICKER's ONE WILD READ (2026-10-05, delegate). Applies the
diet-trained picker (scripts/picker/train_picker.py's saved model, never refit here) to the wild
candidate feature table, picks the top-ranked candidate per row, and reports rows correct vs the key,
confusion against top-1 (fixes/regressions), and the precision table -- exactly as combined_oracle.py's
own consistency-judge report does. BAR (pinned, ledger 2026-10-05 11:50): rows >= 24 with regressions
<= 2 (floor 18 = the consistency judge; ceiling 43 = the oracle bound).

Usage: .venv/bin/python3 scripts/picker/wild_read.py <wild_features.pkl> <model.pkl> <out_report.txt>
"""
import sys
import pickle

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np

BAR_ROWS = 24
BAR_REGRESSIONS = 2
FLOOR = 18
CEILING = 43


def group_rows(cands):
    by_row = {}
    for c in cands:
        by_row.setdefault(c["row"], []).append(c)
    return by_row


def main():
    wild_pkl, model_pkl, out_report = sys.argv[1], sys.argv[2], sys.argv[3]
    cands = pickle.load(open(wild_pkl, "rb"))
    model = pickle.load(open(model_pkl, "rb"))
    scaler, clf, feature_names = model["scaler"], model["clf"], model["feature_names"]

    by_row = group_rows(cands)
    lines = []
    def log(s=""):
        print(s); lines.append(s)

    log("=" * 90); log("THE PANEL PICKER -- THE ONE WILD READ"); log("=" * 90)
    log(f"wild candidates: {len(cands)} over {len(by_row)} rows ({wild_pkl})")
    log(f"model: {model_pkl} (fit on the diet; never refit here) | features: {feature_names}")
    log("")

    pick_correct = top1_correct = 0
    fixes = []; regressions = []; unchanged_wrong = []; n_differ = 0
    confusion = {}
    for r, lst in sorted(by_row.items()):
        X = np.array([[c["X"][f] for f in feature_names] for c in lst], dtype=np.float64)
        scores = clf.decision_function(scaler.transform(X))
        j = int(np.argmax(scores))
        pick = lst[j]
        top1 = next((c for c in lst if c.get("is_top1")), None)
        if top1 is None:
            top1 = lst[0]
        top1_status = "correct" if top1["y"] else ("refused" if top1["status"] != "solved" else "wrong")
        pick_status = "correct" if pick["y"] else ("refused" if pick["status"] != "solved" else "wrong")
        confusion[(top1_status, pick_status)] = confusion.get((top1_status, pick_status), 0) + 1
        if pick["y"]:
            pick_correct += 1
        if top1["y"]:
            top1_correct += 1
        differs = pick["sig"] != top1["sig"]
        if differs:
            n_differ += 1
            if top1_status != "correct" and pick_status == "correct":
                fixes.append(r)
            elif top1_status == "correct" and pick_status != "correct":
                regressions.append(r)
            elif pick_status != "correct":
                unchanged_wrong.append(r)

    n = len(by_row)
    log(f"TOP-1 baseline (on this wild set, as this pipeline's own re-decode/re-solve sees it): {top1_correct}/{n}")
    log(f"PICKER (diet-trained, applied once):                                                  {pick_correct}/{n}")
    log(f"  floor (consistency judge, ledger 10-05): {FLOOR}   ceiling (oracle bound): {CEILING}")
    log("")
    log("CONFUSION (top1_status -> pick_status):")
    for (ts, ps), cnt in sorted(confusion.items()):
        log(f"    {ts:8s} -> {ps:8s} : {cnt}")
    log("")
    log(f"PRECISION: of {n_differ} rows where the picker's pick DIFFERS from top-1 --")
    log(f"  FIXES       (top1 not correct -> pick correct): {len(fixes)}  rows={fixes}")
    log(f"  REGRESSIONS (top1 correct -> pick not correct):  {len(regressions)}  rows={regressions}")
    log(f"  unchanged-wrong (differ, still not correct):     {len(unchanged_wrong)}")
    log("")
    bar_pass = pick_correct >= BAR_ROWS and len(regressions) <= BAR_REGRESSIONS
    log(f"BAR (pinned): rows >= {BAR_ROWS} with regressions <= {BAR_REGRESSIONS} -> "
        f"rows={pick_correct}, regressions={len(regressions)} -> {'PASS' if bar_pass else 'MISS'}")

    open(out_report, "w").write("\n".join(lines) + "\n")
    print(f"[wild-read] wrote {out_report}")


if __name__ == "__main__":
    main()
