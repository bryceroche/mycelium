"""scripts/picker/pvalue_judge.py -- THE PERCEIVER AS THE VALUE FUNCTION, read 1 (2026-10-10,
CPU-only, delegate). Registered: docs/phase1_skeleton_spec.md 2026-10-09 (12:16) "THE PERCEIVER AS
THE VALUE FUNCTION"; build item (a). CLAUDE.md S4's Goodhart fence: the Welford atlas is never in
the head's training loss; here it is an INPUT to a decoder (this judge) trained on the KEY's rows
(the diet slice, gold-graded) -- the picker's existing, sanctioned protocol (train_picker.py). The
custody key gates the diet's labels and is NEVER consulted by the judge at wild read time: every
feature below is a function of (a candidate's own decoded parse, its fixed per-slot content state,
the diet-built Welford library) -- never of wild's gold factors/solution.

REUSE, NOT REIMPLEMENTATION: this script imports scripts/welford_atlas.py (WA: gold_kind,
WelfordLibrary, _kind_means_by_breath, _softmax_p, cos_to, tune_tau, KINDS, CONTENT_LIB_PATH,
CONTENT_BREATHS_LIB_PATH, DIET_STATES_BREATHS_PATH) and scripts/picker/{build_features,train_picker}.py
(FEATURE_NAMES/ANGLE_GROUPS/DIAG_KEYS; to_matrix/group_rows/cv_rows_right/top1_rows_right/
judge_rows_right/ablation_table) exactly as those modules define them -- nothing in welford_atlas.py,
build_features.py or train_picker.py is edited. The base 20-feature candidate table
(.cache/picker/feat_{diet,wild}_v2.pkl, the leak-free v2 panel picker's own output) is loaded as-is;
this script only ADDS the atlas angles on top and re-trains/re-reads once.

THE FEATURE-LIST DIFF (printed at run start, the task's own requirement): BASE_FEATURES (20, from
build_features.FEATURE_NAMES) are status/unique/loglik/length/nlc/flips -- none intersect
GOLD_FIELDS (the fixture row's own gold-bearing keys). ATLAS_FEATURES (4, new here) are each a
function of (candidate's own decoded fact's ftype/op -> WA.gold_kind, called on the DECODE, never
on a gold fact) x (the row+slot's fixed content STATE, read from an already-banked forward-pass
dump) x (the DIET-built Welford library, itself built once from gold-graded diet rows, offline,
per the registration's own ruling -- "under a loss that grades the GRAPH against the key" does not
apply here since nothing here is inside any training loss; this is the decoder-input use the fence
explicitly allows). No ATLAS_FEATURE reads row["factors"]/row["solution"]/row["decisions"]/
row["mentions"]/row["query_var"]/row["key"] at either diet or wild read time.

THE THREE NEW ANGLES (per the task brief):
  (a) atlas_cos_mean       -- mean over the candidate's own slots of cos(state_j, kind_mean[decoded
                              kind_j]) in the content-space Welford library (CONTENT_LIB_PATH) --
                              "the atlas monitor applied to the candidate's own kind assignments".
  (b) atlas_jsd_ent_mean/  -- entropy of the softmax-over-kind-centroids responsibility vector
      atlas_jsd_ent_min       (WA._softmax_p, tau tuned on the diet by WA.tune_tau, content_breaths_lib
                              at the final breath), mean and min over the candidate's own slots.
  (c) atlas_dsl_agree_share -- share of the candidate's own slots where the NEAREST centroid's kind
                              (argmax cosine over the present kind set) agrees with the candidate's
                              own decoded ftype/op kind.

Usage: DEV=CPU .venv/bin/python3 scripts/picker/pvalue_judge.py
Writes (ALL under .cache/pvalue_* -- this task's own write-scope fence, nothing banked touched):
  .cache/pvalue_judge_feat_diet_v3.pkl, .cache/pvalue_judge_feat_wild_v3.pkl (v2 features + atlas
  angles), .cache/pvalue_judge_model_v3.pkl, .cache/pvalue_judge_report.txt,
  .cache/pvalue_judge_PMS8_241.txt
"""
import os
import sys
import json
import pickle
import time

os.environ.setdefault("DEV", "CPU")
sys.path.insert(0, "."); sys.path.insert(0, "scripts"); sys.path.insert(0, "scripts/picker")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import welford_atlas as WA   # side effect: AS._build_family_env("PMS8_241", ""), DEV=CPU asserted
assert os.environ["DEV"] == "CPU", "pvalue_judge: zero-GPU, always"

from build_features import FEATURE_NAMES as BASE_FEATURES, ANGLE_GROUPS as BASE_GROUPS, DIAG_KEYS
from train_picker import (to_matrix, group_rows, cv_rows_right, top1_rows_right, judge_rows_right)

CAND_DIET = ".cache/picker/cand_diet.pkl"
CAND_WILD = ".cache/picker/cand_wild_PMS8_241.pkl"
FEAT_DIET_IN = ".cache/picker/feat_diet_v2.pkl"
FEAT_WILD_IN = ".cache/picker/feat_wild_v2.pkl"
# every NEW artifact this script writes lives under .cache/pvalue_* (this task's write-scope fence
# while the GPU trains AR_241/SR_242 -- nothing under .cache/picker/ or any banked path is touched).
FEAT_DIET_OUT = ".cache/pvalue_judge_feat_diet_v3.pkl"
FEAT_WILD_OUT = ".cache/pvalue_judge_feat_wild_v3.pkl"
MODEL_OUT = ".cache/pvalue_judge_model_v3.pkl"
REPORT_OUT = ".cache/pvalue_judge_report.txt"
OUT_TXT = ".cache/pvalue_judge_PMS8_241.txt"
WILD_CLOCK_STATES = ".cache/clock_band_states_PMS8_241.npz"

K_B = int(os.environ.get("ALG_BREATH", "7"))

# fields on a fixture row (diet's own vs[i], or the wild jsonl row) that carry GOLD -- the diff
# target. None of BASE_FEATURES or ATLAS_FEATURES may be a function of these at READ time (the
# library itself is built offline from them, once, per the registration -- see module docstring).
GOLD_FIELDS = ["factors", "solution", "decisions", "mentions", "query_var", "key", "n_vars", "m"]

ATLAS_FEATURES = ["atlas_cos_mean", "atlas_jsd_ent_mean", "atlas_jsd_ent_min", "atlas_dsl_agree_share"]
FULL_FEATURES = BASE_FEATURES + ATLAS_FEATURES
ALL_GROUPS = dict(BASE_GROUPS); ALL_GROUPS["atlas"] = ATLAS_FEATURES

BAR_ROWS = 24
BAR_REGRESSIONS = 2
KILL_ROWS = 20
FLOOR_CONSISTENCY_JUDGE = 18   # ledger 2026-10-05 11:50, combined_oracle.py's hand judge, 0 regressions
FLOOR_PANEL_V2 = 18            # ledger 2026-10-06 21:20/21:44, panel_picker v2 (no atlas), 4 regressions
CEILING = 43                   # ledger 2026-10-05 11:50, the combined oracle bound


def P(lines, s=""):
    print(s, flush=True)
    lines.append(s)


# ======================================================================================
# THE FEATURE-LIST DIFF (printed once, before anything is trained)
# ======================================================================================
def print_feature_diff(lines):
    P(lines, "=" * 100)
    P(lines, "THE FEATURE-LIST DIFF (base picker features vs the fixture's own GOLD fields)")
    P(lines, "=" * 100)
    overlap = sorted(set(BASE_FEATURES) & set(GOLD_FIELDS))
    P(lines, f"BASE_FEATURES ({len(BASE_FEATURES)}, build_features.py, leak-free since the 2026-10-06 fix):")
    P(lines, "  " + ", ".join(BASE_FEATURES))
    P(lines, f"GOLD_FIELDS ({len(GOLD_FIELDS)}, the fixture row's own gold-bearing keys): " + ", ".join(GOLD_FIELDS))
    P(lines, f"OVERLAP (must be empty): {overlap if overlap else 'EMPTY -- confirmed leak-free'}")
    assert not overlap, f"a base feature shares a name with a gold field: {overlap}"
    P(lines, f"DIAGNOSTIC-ONLY keys (never in any feature list below): {DIAG_KEYS}")
    P(lines, f"ATLAS_FEATURES ({len(ATLAS_FEATURES)}, NEW here): " + ", ".join(ATLAS_FEATURES))
    P(lines, "  each is a function of (candidate's OWN decoded fact -> WA.gold_kind) x (the row+slot's")
    P(lines, "  fixed content STATE, an already-banked forward-pass array) x (the diet-built Welford")
    P(lines, "  library) -- no ATLAS_FEATURE reads factors/solution/decisions/mentions/query_var/key at")
    P(lines, "  either diet or wild READ time (the library's own BUILD, once, offline, from gold-graded")
    P(lines, "  diet rows, is the registration's sanctioned decoder-input use, not a read-time leak).")
    overlap2 = sorted(set(ATLAS_FEATURES) & set(GOLD_FIELDS))
    assert not overlap2, overlap2
    P(lines, "")


# ======================================================================================
# per-row fixed content states (final breath, content dims only) -- diet and wild
# ======================================================================================
def load_diet_states():
    z = np.load(WA.DIET_STATES_BREATHS_PATH)
    states_all = z["states_all"].astype(np.float32)   # (n, K_B, L_FAC, C)
    kb = int(z["K_B"]) - 1
    return states_all[:, kb]   # (n, L_FAC, C)


def load_wild_states():
    import phase1_algebra_head as H
    bands, clock_dims = H._hier_band_dims()
    CONTENT = np.sort(np.concatenate(bands))
    z = np.load(WILD_CLOCK_STATES)
    S = z["states"].astype(np.float32)   # (n, 6, L_FAC, 512), loop breaths 1..6
    return S[:, 5][:, :, CONTENT]        # (n, L_FAC, C) -- final loop breath (kb=6 of K_B=7)


# ======================================================================================
# the atlas angles, per candidate, from its own decoded parse + the row's fixed state
# ======================================================================================
def candidate_kind(fac, j):
    """WA.gold_kind applied to a candidate's OWN decoded fact (not a gold fact) -- the function
    itself only reads ftype/op/result, so it is kind-agnostic to whether its input is gold or
    decoded; the registration's "atlas monitor applied to the candidate's own kind assignments"."""
    try:
        return WA.gold_kind(fac, j)
    except Exception:
        return None


def atlas_angles(parse, state_row, content_lib, present, means, tau):
    cosvals, ents, agree = [], [], []
    L_FAC = state_row.shape[0]
    for f in parse:
        j = f.get("_slot")
        if j is None or not (0 <= j < L_FAC):
            continue
        k = candidate_kind(f, j)
        if k is None:
            continue
        x = state_row[j]
        key = f"kind:{k}"
        if key in content_lib:
            cosvals.append(float(WA.cos_to(content_lib[key].mean, x[None, :])[0]))
        if means is not None:
            p, cos_all = WA._softmax_p(x[None, :], means, tau)
            ents.append(float(-(p[0] * np.log(p[0] + 1e-12)).sum()))
            nearest = present[int(np.argmax(cos_all[0]))]
            agree.append(1.0 if nearest == k else 0.0)
    return dict(
        atlas_cos_mean=float(np.mean(cosvals)) if cosvals else np.nan,
        atlas_jsd_ent_mean=float(np.mean(ents)) if ents else np.nan,
        atlas_jsd_ent_min=float(np.min(ents)) if ents else np.nan,
        atlas_dsl_agree_share=float(np.mean(agree)) if agree else np.nan,
    )


def attach_atlas(feat_records, cand_by_row, states, content_lib, present, means, tau, lines, side):
    n_nan = {f: 0 for f in ATLAS_FEATURES}
    n_no_parse = 0
    for rec in feat_records:
        row, sig = rec["row"], rec["sig"]
        branches = cand_by_row.get(row)
        parse = branches[sig]["parse"] if (branches is not None and sig in branches) else None
        if not parse or row >= states.shape[0]:
            n_no_parse += 1
            angles = {f: np.nan for f in ATLAS_FEATURES}
        else:
            angles = atlas_angles(parse, states[row], content_lib, present, means, tau)
        for f in ATLAS_FEATURES:
            if not np.isfinite(angles[f]):
                n_nan[f] += 1
        rec["X"].update(angles)
    P(lines, f"[atlas] {side}: {len(feat_records)} candidates, {n_no_parse} with no matched parse; "
              f"NaN counts (pre-impute): {n_nan}")
    return feat_records


def impute_nans(diet_records, wild_records, lines):
    """Fill NaN atlas values with the DIET's own per-feature mean (computed on the diet, applied to
    both sides) -- a candidate whose decoded parse carries no recognizable kind (rare: an
    "unbuildable"/empty-parse branch) gets the population average, never a wild-only statistic."""
    for f in ATLAS_FEATURES:
        vals = np.array([r["X"][f] for r in diet_records], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        fill = float(np.mean(finite)) if len(finite) else 0.0
        for recs in (diet_records, wild_records):
            for r in recs:
                if not np.isfinite(r["X"][f]):
                    r["X"][f] = fill
        P(lines, f"[impute] {f}: fill value {fill:.4f} (diet mean over {len(finite)}/{len(vals)} finite)")


# ======================================================================================
# main
# ======================================================================================
def main():
    t0 = time.time()
    lines = []
    print_feature_diff(lines)

    P(lines, "[load] candidate dumps + v2 (leak-free) feature tables...")
    cand_diet = pickle.load(open(CAND_DIET, "rb"))
    cand_wild = pickle.load(open(CAND_WILD, "rb"))
    cand_diet_by_row = {r["i"]: r["branches"] for r in cand_diet}
    cand_wild_by_row = {r["i"]: r["branches"] for r in cand_wild}
    diet_records = pickle.load(open(FEAT_DIET_IN, "rb"))
    wild_records = pickle.load(open(FEAT_WILD_IN, "rb"))
    P(lines, f"[load] diet: {len(diet_records)} candidates / {len(group_rows(diet_records))} rows; "
              f"wild: {len(wild_records)} candidates / {len(group_rows(wild_records))} rows")

    P(lines, "[load] diet/wild fixed content states (final breath) + the diet-built Welford library...")
    diet_states = load_diet_states()
    wild_states = load_wild_states()
    content_lib = WA.WelfordLibrary.load(WA.CONTENT_LIB_PATH)
    content_breaths_lib = WA.WelfordLibrary.load(WA.CONTENT_BREATHS_LIB_PATH)
    tau_star, tau_results, tau_rule = WA.tune_tau(content_breaths_lib, K_B)
    present, means = WA._kind_means_by_breath(content_breaths_lib, K_B)[K_B - 1]
    P(lines, f"[atlas] tau* = {tau_star} ({tau_rule}); present kinds at breath {K_B-1}: {present}")

    attach_atlas(diet_records, cand_diet_by_row, diet_states, content_lib, present, means, tau_star, lines, "diet")
    attach_atlas(wild_records, cand_wild_by_row, wild_states, content_lib, present, means, tau_star, lines, "wild")
    impute_nans(diet_records, wild_records, lines)

    pickle.dump(diet_records, open(FEAT_DIET_OUT, "wb"))
    pickle.dump(wild_records, open(FEAT_WILD_OUT, "wb"))
    P(lines, f"[save] {FEAT_DIET_OUT}, {FEAT_WILD_OUT}")

    # ---- DIET baselines (identical machinery/baselines as train_picker.py) ----
    P(lines, "")
    P(lines, "=" * 100)
    P(lines, "DIET: baselines + full-model CV + ablation (5-fold GroupKFold by row)")
    P(lines, "=" * 100)
    t1_right, t1_n = top1_rows_right(diet_records)
    j_right, j_n = judge_rows_right(diet_records)
    P(lines, f"DIET BASELINE  top-1:              {t1_right}/{t1_n} = {t1_right/t1_n:.3f}")
    P(lines, f"DIET BASELINE  consistency judge:  {j_right}/{j_n} = {j_right/j_n:.3f}")

    base_right, base_n, _ = cv_rows_right(diet_records, BASE_FEATURES)
    P(lines, f"DIET PICKER v2 (no atlas, 20 feats) 5-fold CV: {base_right}/{base_n} = {base_right/base_n:.3f}")

    full_right, full_n, _ = cv_rows_right(diet_records, FULL_FEATURES)
    P(lines, f"DIET PICKER v3 (+atlas, {len(FULL_FEATURES)} feats) 5-fold CV: {full_right}/{full_n} = {full_right/full_n:.3f}")

    P(lines, "")
    P(lines, "ABLATION (each angle group removed from the FULL v3 feature set; 5-fold CV rows-right, diet):")
    for g, names in ALL_GROUPS.items():
        kept = [f for f in FULL_FEATURES if f not in names]
        right, n, _ = cv_rows_right(diet_records, kept)
        P(lines, f"  -{g:9s} {right:4d}/{n}  (delta vs FULL {right - full_right:+d})")

    P(lines, "")
    P(lines, "SPECIAL ABLATION (the task's own ask): atlas angles alone + solver status, NO loglik/"
              "length/nlc/flips/unique:")
    status_atlas = BASE_GROUPS["status"] + ATLAS_FEATURES
    sa_right, sa_n, _ = cv_rows_right(diet_records, status_atlas)
    P(lines, f"  status+atlas ({len(status_atlas)} feats): {sa_right}/{sa_n} = {sa_right/sa_n:.3f} "
              f"(vs FULL {full_right}/{full_n}, vs v2-no-atlas {base_right}/{base_n})")

    # ---- fit the FULL v3 model on all diet rows ----
    X, y, rows = to_matrix(diet_records, FULL_FEATURES)
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(scaler.transform(X), y)
    pickle.dump(dict(scaler=scaler, clf=clf, feature_names=FULL_FEATURES), open(MODEL_OUT, "wb"))
    P(lines, f"[train] full v3 model (fit on all {len(group_rows(diet_records))} diet rows) -> {MODEL_OUT}")
    P(lines, "NOTE: the diet slice is IN-SAMPLE for PMS8_241 (per the 10-07 register/regime erratum) -- "
              "the judge's label is the key on these rows, the picker's existing, acknowledged protocol; "
              "the diet CV numbers above are calibration-on-memorized-rows, not a transfer estimate.")

    # ---- THE ONE WILD READ ----
    P(lines, "")
    P(lines, "=" * 100)
    P(lines, "THE ONE WILD READ (diet-trained v3 model, applied once to the 311-row wild holdout)")
    P(lines, "=" * 100)
    by_row = group_rows(wild_records)
    pick_correct = top1_correct = 0
    fixes, regressions, unchanged_wrong = [], [], []
    confusion = {}
    for r, lst in sorted(by_row.items()):
        Xr = np.array([[c["X"][f] for f in FULL_FEATURES] for c in lst], dtype=np.float64)
        scores = clf.decision_function(scaler.transform(Xr))
        j = int(np.argmax(scores))
        pick = lst[j]
        top1 = next((c for c in lst if c.get("is_top1")), None) or lst[0]
        top1_status = "correct" if top1["y"] else ("refused" if top1["status"] != "solved" else "wrong")
        pick_status = "correct" if pick["y"] else ("refused" if pick["status"] != "solved" else "wrong")
        confusion[(top1_status, pick_status)] = confusion.get((top1_status, pick_status), 0) + 1
        if pick["y"]:
            pick_correct += 1
        if top1["y"]:
            top1_correct += 1
        if pick["sig"] != top1["sig"]:
            if top1_status != "correct" and pick_status == "correct":
                fixes.append(r)
            elif top1_status == "correct" and pick_status != "correct":
                regressions.append(r)
            elif pick_status != "correct":
                unchanged_wrong.append(r)

    n = len(by_row)
    P(lines, f"TOP-1 baseline (this pipeline's own re-decode/re-solve): {top1_correct}/{n}")
    P(lines, f"PERCEIVER-AS-VALUE (v3, diet-trained, atlas+base, applied once): {pick_correct}/{n}")
    P(lines, f"  floor (consistency judge, ledger 2026-10-05 11:50): {FLOOR_CONSISTENCY_JUDGE} (0 regressions)")
    P(lines, f"  floor (panel picker v2, no atlas, ledger 2026-10-06 21:44): {FLOOR_PANEL_V2} (4 regressions)")
    P(lines, f"  ceiling (the combined oracle bound, ledger 2026-10-05 11:50): {CEILING}")
    P(lines, "")
    P(lines, "CONFUSION (top1_status -> pick_status):")
    for (ts, ps), cnt in sorted(confusion.items()):
        P(lines, f"    {ts:8s} -> {ps:8s} : {cnt}")
    P(lines, "")
    P(lines, f"FIXES       (top1 not correct -> pick correct): {len(fixes)}  rows={fixes}")
    P(lines, f"REGRESSIONS (top1 correct -> pick not correct):  {len(regressions)}  rows={regressions}")
    P(lines, f"unchanged-wrong (differ, still not correct):     {len(unchanged_wrong)}")
    P(lines, "")
    bar_pass = pick_correct >= BAR_ROWS and len(regressions) <= BAR_REGRESSIONS
    killed = pick_correct < KILL_ROWS
    P(lines, f"BAR (pinned): rows >= {BAR_ROWS} with regressions <= {BAR_REGRESSIONS} -> "
              f"rows={pick_correct}, regressions={len(regressions)} -> {'PASS' if bar_pass else 'MISS'}")
    P(lines, f"KILL (pinned): rows < {KILL_ROWS} -> {'KILLED' if killed else 'not killed'}")
    P(lines, "")
    P(lines, "SUMMARY TABLE:")
    P(lines, f"  {'judge':38} {'rows/311':>10} {'regressions':>12}")
    P(lines, f"  {'top-1 (this pipeline)':38} {top1_correct:10d} {'--':>12}")
    P(lines, f"  {'consistency judge (hand, no learning)':38} {FLOOR_CONSISTENCY_JUDGE:10d} {0:12d}")
    P(lines, f"  {'panel picker v2 (learned, no atlas)':38} {FLOOR_PANEL_V2:10d} {4:12d}")
    P(lines, f"  {'perceiver-as-value v3 (learned, +atlas)':38} {pick_correct:10d} {len(regressions):12d}")
    P(lines, f"  {'ceiling (oracle bound)':38} {CEILING:10d} {'--':>12}")

    P(lines, f"\n[done] {time.time()-t0:.0f}s")
    open(REPORT_OUT, "w").write("\n".join(lines) + "\n")
    open(OUT_TXT, "w").write("\n".join(lines) + "\n")
    print(f"[pvalue-judge] wrote {REPORT_OUT} and {OUT_TXT}")


if __name__ == "__main__":
    main()
