"""scripts/jury_train.py -- THE CARICATURE JURY, training + CV (2026-10-06, zero-GPU, delegate;
form 2 of THE COURTROOM: docs/phase1_skeleton_spec.md 2026-10-06 13:35 "THE TRAINED JUROR").

Trains a small TRAINED PAIRWISE juror on (text, story A, story B) -> which is gold, on the diet
slice's own banked candidates (.cache/picker/cand_diet.pkl, gen_candidates.py's output -- NEVER
wild; wild is read once, by scripts/courtroom2.py, at the end). Reuses scripts/picker/
jury_features.py for every feature (the caricature embedding + the structural A-minus-B diff +
the three hand-jurors' own votes, straight off scripts/courtroom.py). Candidates and the answer
key are never regenerated or re-solved here -- gen_candidates.py's own b["correct"] flag (computed
against the dump's row["key"], itself the custody-gold answer -- mycelium/custody_gold.py's
row_gold(), already applied upstream of this pickle) is the ONLY label used ("the row's true
story").

PAIRS: for every row with >=1 solver-consistent (status=="solved") gold-correct candidate ("the
true story") AND >=1 solver-consistent WRONG candidate, form the FULL cross-product of (true,
wrong) pairs, each in both orderings (A=true/B=wrong, label=1; A=wrong/B=true, label=0 -- the
swap-augmented pair the shared caricature embedding does not itself distinguish, per the task's
own stated option). Rows with no consistent true story at all contribute nothing (counted,
reported, never silently dropped).

Usage:
  DEV=CPU .venv/bin/python3 scripts/jury_train.py [--cand .cache/picker/cand_diet.pkl]
      [--out-model .cache/picker/jury_model.pkl] [--out-report .cache/jury_train_PMS8_241.txt]
"""
import os
import sys
import time
import pickle
import argparse
from collections import Counter

sys.path.insert(0, "."); sys.path.insert(0, "scripts"); sys.path.insert(0, "scripts/picker")
import numpy as np

os.environ.setdefault("DEV", "CPU")
assert os.environ["DEV"] == "CPU", "jury_train: CPU-only, always"

import courtroom as CT
import jury_features as JF


# ================================================================================================
# 1. PAIR CONSTRUCTION
# ================================================================================================
def build_pairs(cand_path, log=print):
    rows = pickle.load(open(cand_path, "rb"))
    n_rows = len(rows)
    n_no_true = n_true_no_wrong = n_contrib = 0
    pairs = []   # (row_i, text, sigA, sigB, q) with A the TRUE story, B a WRONG one
    for row in rows:
        text, q = row["text"], row["q"]
        trues = [sig for sig, b in row["branches"].items() if b.get("status") == "solved" and b.get("correct")]
        wrongs = [sig for sig, b in row["branches"].items() if b.get("status") == "solved" and not b.get("correct")]
        if not trues:
            n_no_true += 1
            continue
        if not wrongs:
            n_true_no_wrong += 1
            continue
        n_contrib += 1
        for sa in trues:
            for sb in wrongs:
                pairs.append((row["i"], text, sa, sb, q))
    log(f"[jury-train] {n_rows} diet rows: no-true-story={n_no_true}  true-but-no-wrong={n_true_no_wrong}  "
        f"CONTRIBUTING(true+wrong)={n_contrib}  pairs(one direction)={len(pairs)}  "
        f"pairs(swap-augmented)={2*len(pairs)}")
    return rows, pairs, dict(n_rows=n_rows, n_no_true=n_no_true, n_true_no_wrong=n_true_no_wrong,
                              n_contrib=n_contrib, n_pairs_one_dir=len(pairs))


SIGNED_KEYS = [k for k in JF.STRUCT_ALL_KEYS if k not in ("n_diff_slots", "gran_is_span")]


def negate_signed(struct):
    out = dict(struct)
    for k in SIGNED_KEYS:
        out[k] = -out[k]
    return out


# ================================================================================================
# 2. FEATURE EXTRACTION (both orderings, swap-augmented)
# ================================================================================================
def extract(rows, pairs, log=print):
    by_sig = {row["i"]: row["branches"] for row in rows}
    q_text = {row["i"]: row["text"] for row in rows}
    texts = sorted({t for (_, t, _, _, _) in pairs})
    log(f"[jury-train] embedding {len(texts)} unique contributing-row texts through the frozen trunk (CPU)...")
    t0 = time.time()
    trunk_cache = JF.embed_texts(texts, batch_size=32, log=log)
    log(f"[jury-train] trunk embedding done in {time.time()-t0:.0f}s")

    groups, y, structs, span_pools, car_diffs, hand_votes = [], [], [], [], [], []
    t0 = time.time()
    for n, (ri, text, sig_a, sig_b, q) in enumerate(pairs):
        branches = by_sig[ri]
        cA, cB = branches[sig_a], branches[sig_b]
        rec = JF.pair_record(text, cA, cB, q, trunk_cache)
        # A=true, B=wrong -> label 1
        groups.append(ri); y.append(1); structs.append(rec["struct"])
        span_pools.append(rec["span_pool"]); car_diffs.append(rec["car_diff"]); hand_votes.append(rec["hand_votes"])
        # swap: A=wrong, B=true -> label 0 (embedding unchanged -- shared context; structural negated)
        groups.append(ri); y.append(0); structs.append(negate_signed(rec["struct"]))
        span_pools.append(rec["span_pool"]); car_diffs.append(rec["car_diff"])
        hand_votes.append([-v for v in rec["hand_votes"]])
        if (n + 1) % 2000 == 0 or n + 1 == len(pairs):
            log(f"[jury-train] features {n+1}/{len(pairs)} pairs ({time.time()-t0:.0f}s)")
    struct_mat = np.asarray([[s[k] for k in JF.STRUCT_ALL_KEYS] for s in structs], np.float32)
    span_mat = np.asarray(span_pools, np.float32)
    car_mat = np.asarray(car_diffs, np.float32)
    hand_mat = np.asarray(hand_votes, np.float32)
    y = np.asarray(y, np.int64)
    groups = np.asarray(groups, np.int64)
    gran_span_frac = float(np.mean([1.0 if s_["gran_is_span"] else 0.0 for s_ in structs]))
    log(f"[jury-train] granularity: span-restricted {gran_span_frac:.3f} of pairs")
    return dict(groups=groups, y=y, struct=struct_mat, span=span_mat, car=car_mat, hand=hand_mat,
                gran_span_frac=gran_span_frac)


# ================================================================================================
# 3. CV + ABLATION
# ================================================================================================
def hand_juror_accuracy(hand, y):
    """The baseline courtroom.py itself used: majority-of-3-signed-votes. Decisive pairs only, and
    with ties scored as a miss (both reported -- a tie is never a correct call)."""
    total = hand.sum(axis=1)
    pred = (total > 0).astype(np.int64)
    decisive = total != 0
    acc_all = float((pred == y).mean())
    acc_dec = float((pred[decisive] == y[decisive]).mean()) if decisive.any() else float("nan")
    return acc_all, acc_dec, float(decisive.mean())


def make_feats(data, pca_span=None, pca_car=None, use_struct=True, use_embed=True, fit=False):
    from sklearn.decomposition import PCA
    parts = []
    if use_embed:
        if fit:
            pca_span = PCA(n_components=16, random_state=0).fit(data["span"])
            pca_car = PCA(n_components=16, random_state=0).fit(data["car"])
        parts.append(pca_span.transform(data["span"]))
        parts.append(pca_car.transform(data["car"]))
    if use_struct:
        parts.append(data["struct"])
    X = np.concatenate(parts, axis=1).astype(np.float32)
    return X, pca_span, pca_car


def cv_eval(data, use_struct, use_embed, n_splits=5, log=print, label=""):
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    gkf = GroupKFold(n_splits=n_splits)
    accs = []
    for fold, (tr, va) in enumerate(gkf.split(data["struct"], data["y"], data["groups"])):
        tr_data = {k: v[tr] for k, v in data.items() if k != "gran_span_frac"}
        va_data = {k: v[va] for k, v in data.items() if k != "gran_span_frac"}
        Xtr, pca_s, pca_c = make_feats(tr_data, use_struct=use_struct, use_embed=use_embed, fit=True)
        Xva, _, _ = make_feats(va_data, pca_s, pca_c, use_struct=use_struct, use_embed=use_embed, fit=False)
        scaler = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=3000, C=1.0).fit(scaler.transform(Xtr), tr_data["y"])
        pred = clf.predict(scaler.transform(Xva))
        acc = float((pred == va_data["y"]).mean())
        accs.append(acc)
        log(f"[jury-train][cv:{label}] fold {fold}: n_train={len(tr)} n_val={len(va)} acc={acc:.4f}")
    log(f"[jury-train][cv:{label}] mean acc = {np.mean(accs):.4f}  (per-fold: {[round(a,4) for a in accs]})")
    return accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", default=".cache/picker/cand_diet.pkl")
    ap.add_argument("--out-model", default=".cache/picker/jury_model.pkl")
    ap.add_argument("--out-report", default=".cache/jury_train_PMS8_241.txt")
    ap.add_argument("--n-splits", type=int, default=5)
    args = ap.parse_args()

    lines = []
    def log(s=""):
        print(s, flush=True); lines.append(s)

    log("=" * 94); log("THE CARICATURE JURY -- training + CV on the diet slice"); log("=" * 94)
    rows, pairs, pair_stats = build_pairs(args.cand, log)
    data = extract(rows, pairs, log)

    hand_acc_all, hand_acc_dec, dec_frac = hand_juror_accuracy(data["hand"], data["y"])
    log("-" * 94)
    log(f"HAND-JUROR BASELINE (courtroom.py's majority-of-3, same pairs): "
        f"accuracy(all, ties=miss)={hand_acc_all:.4f}  accuracy(decisive-only, {dec_frac:.3f} of pairs)={hand_acc_dec:.4f}")

    log("-" * 94); log(f"5-FOLD GroupKFold BY ROW (n_splits={args.n_splits}):"); log("-" * 94)
    accs_full = cv_eval(data, use_struct=True, use_embed=True, n_splits=args.n_splits, log=log, label="full(struct+embed)")
    accs_embed = cv_eval(data, use_struct=False, use_embed=True, n_splits=args.n_splits, log=log, label="ABLATION:embed-only")
    accs_struct = cv_eval(data, use_struct=True, use_embed=False, n_splits=args.n_splits, log=log, label="ablation:struct-only")

    log("-" * 94)
    log("SUMMARY (pairwise accuracy, same diet pairs throughout):")
    log(f"  hand jurors (courtroom.py, majority-of-3, untrained): {hand_acc_all:.4f}  (decisive-only {hand_acc_dec:.4f})")
    log(f"  TRAINED JUROR, full (struct + caricature embedding):  {np.mean(accs_full):.4f}")
    log(f"  ablation, embedding-only (no structural diff):        {np.mean(accs_embed):.4f}")
    log(f"  ablation, structural-only (no trunk embedding):       {np.mean(accs_struct):.4f}")
    log(f"  JUROR 1 granularity (span-restricted fraction of pairs): {data['gran_span_frac']:.4f}")

    # final fit on ALL diet pairs, for deployment (courtroom2.py)
    log("-" * 94); log("Fitting the FINAL juror on all diet pairs (both orderings) for deployment..."); log("-" * 94)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Xall, pca_span, pca_car = make_feats(data, use_struct=True, use_embed=True, fit=True)
    scaler = StandardScaler().fit(Xall)
    clf = LogisticRegression(max_iter=5000, C=1.0).fit(scaler.transform(Xall), data["y"])
    train_acc = float((clf.predict(scaler.transform(Xall)) == data["y"]).mean())
    log(f"  final juror in-sample accuracy (NOT a CV number -- deployment fit only): {train_acc:.4f}")

    model = dict(pca_span=pca_span, pca_car=pca_car, scaler=scaler, clf=clf,
                 struct_keys=JF.STRUCT_ALL_KEYS, signed_keys=SIGNED_KEYS,
                 cv_acc_full=float(np.mean(accs_full)), cv_acc_embed_only=float(np.mean(accs_embed)),
                 cv_acc_struct_only=float(np.mean(accs_struct)),
                 hand_acc_all=hand_acc_all, hand_acc_decisive=hand_acc_dec,
                 pair_stats=pair_stats, gran_span_frac=data["gran_span_frac"])
    os.makedirs(os.path.dirname(args.out_model) or ".", exist_ok=True)
    with open(args.out_model, "wb") as f:
        pickle.dump(model, f)
    log(f"[jury-train] wrote {args.out_model}")

    with open(args.out_report, "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"[jury-train] wrote {args.out_report}")


if __name__ == "__main__":
    main()
