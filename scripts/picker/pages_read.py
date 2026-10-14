"""pages_read.py -- THE READ NOTEBOOK (2026-10-14, Opus): the page read.

Zero-GPU, pure numpy/sklearn. Reads banked wild artifacts (.cache/membrane_raw_wild_PMS8_241.npz,
.cache/ps_legal_wild_PMS8_241.npz) and the new valid2 artifacts this task's own
pages_collect_valid2.py wrote (.cache/pages_membrane_raw_valid2_PMS8_241.npz,
.cache/pages_ps_legal_valid2_PMS8_241.npz, 128 rows). No model, no tinygrad import.

THE IDEA: the slot's first look is the single highest-mass real token (the argmax of breath-0
pooled slot->token attention). When two numerals compete for a slot's attention, the first look
averages/collapses onto one winner and may pick the wrong twin. This script keeps the top-3
attended REAL TOKENS as separate "pages" per GIVEN slot, builds READ-TIME-ONLY features per page,
and fits a listwise-by-pointwise logistic reranker on the training diet's valid2 slice, then
measures it on wild (never fit on wild, except the explicitly-labeled LORO ceiling below).

PART 1 -- COVERAGE (wild only): reuses scripts/picker/goldmention_read.py's own _load_ok /
_collect_cells VERBATIM (import, not copy) -- those functions already compute, per wrong-at-
final-decode GIVEN slot, the gold mention's rank among real tokens by attention mass (1-indexed,
ties by strict-greater count) and the same-sentence("twin")/other-sentence class (breath-0 argmax
band vs the gold mention's mention/clause segmentation). This script only adds k=4 to goldmention's
own k<=3 table and reports both overall and by class.

PART 2 -- PAGES + FEATURES: for every GIVEN slot (gold-matched, any row), the top-3 real tokens by
breath-0 pooled attention mass are the candidate PAGES. Each page's VALUE is the decoded int of the
maximal digit-run containing that token (None if the token isn't part of any digit run -- an
"abstain" page, automatically wrong). Per-page features (READ-TIME ONLY -- no gold used to build
any of them):
  mass           - page's own attention mass / total real-token mass (softmax renormalized)
  gap_to_best    - (page1's mass) - (this page's mass); 0 for the top page
  rank           - 1/2/3 (categorical, one-hot'd as rank1/rank2/rank3 below)
  pos_frac       - token position / row's real-token count (0..1)
  sent_idx       - the token's sentence index (raw int)
  claimed        - this page's VALUE equals some OTHER given slot's own top-1 (first-look) page's
                   value, in the SAME row (read-time: uses other slots' first-look picks, never gold)
  same_sent_prev - this page's sentence == the PRECEDING given slot's (lower factor-list index, same
                   row) own top-1 page's sentence (read-time; "preceding" = the positional law's own
                   consecutive-letter order, i.e. slot index j)
  mag_band       - digit count of the page's value (1/2/3+; -1 if no value -- an abstain page)
LABEL (fit-time only): page.value == gold_val (gold from g_digits, the custody door; never pen
solution text).

NOT BUILT (said plainly, per the brief's "say what you ran" clause): the latest-before / governor-
chain owner feature (scripts/picker/latest_before_read.py's machinery) is NOT wired into the
per-page feature set -- that script computes ownership once per GOLD arg on a relation slot via a
fresh spaCy dependency parse; mapping it onto each of up to 3 CANDIDATE pages per GIVEN slot (a
different slot population entirely -- latest-before is a relation-arg question, this is a given-
value-location question) is a non-trivial second integration, not a cheap add, so it is skipped
here rather than rushed; its cost is named, not hidden.

PART 3 -- THE RERANKER: sklearn LogisticRegression (StandardScaler + balanced class weight),
POINTWISE fit (one training example per page, label = is-gold-page) on valid2's pages (n=128
rows), then at read time the slot's chosen page = argmax predicted P(gold) among its <=3 pages.
Measured on wild (311 rows, held out, never fit on): accuracy vs the first-look argmax (page1),
on ALL given slots / on wrong-at-first-look slots / by class (twin vs other-sentence) / the COST
on right-at-first-look slots (fraction that FLIPS from right to wrong).

PART 4 -- THE CEILING: the identical feature set + model family, fit by LEAVE-ONE-ROW-OUT on WILD
itself (one row held out, the rest of wild's own pages train the reranker, repeat for all 311
rows) -- labeled CEILING, NEVER deployable (it is not a valid2-trained model; it answers "can pages
be read at all with an in-domain-fit reader", not "does the diet teach a reader that transfers").

Usage: .venv/bin/python3 scripts/picker/pages_read.py
Outputs: .cache/pages_report_PMS8_241.txt / .json (new artifacts; nothing banked touched)
"""
import os
import sys
import json
import collections

sys.path.insert(0, "."); sys.path.insert(0, "scripts"); sys.path.insert(0, "scripts/picker")
import numpy as np

import goldmention_read as GM     # reused verbatim: _load_ok, _collect_cells
import membrane_scale as MS       # reused verbatim: _digit_runs

L_FAC = 24
T_ALG = 256

RAW_WILD = ".cache/membrane_raw_wild_PMS8_241.npz"
OK_WILD = ".cache/ps_legal_wild_PMS8_241.npz"
RAW_V2 = ".cache/pages_membrane_raw_valid2_PMS8_241.npz"
OK_V2 = ".cache/pages_ps_legal_valid2_PMS8_241.npz"
OUT_TXT = ".cache/pages_report_PMS8_241.txt"
OUT_JSON = ".cache/pages_report_PMS8_241.json"


def _tokenizer():
    import phase1_algebra_head as H
    from tokenizers import Tokenizer
    return Tokenizer.from_file(H.TOKENIZER_JSON)


def _load_ok_generic(path):
    ps = np.load(path)
    return {(int(r), int(c)): bool(o) for r, c, o in zip(ps["rows"], ps["slots"], ps["ok"])}


# ===========================================================================
# PART 1 -- COVERAGE (wild only), k=1..4, overall + by class
# ===========================================================================

def coverage_table(tok):
    ok = GM._load_ok()
    raw = np.load(RAW_WILD)
    n = raw["ids"].shape[0]
    fat0 = raw["fat_mass"][:, 0, :, :].astype(np.float32)
    cells, nomatch, notj = GM._collect_cells(
        raw["ids"], raw["tokmask"], raw["sent"], raw["g_presence"], raw["g_ftype"], raw["g_digits"],
        fat0, ok, n, L_FAC, T_ALG, tok)
    wrong = [c for c in cells if c["key"] == "wrong"]
    twin = [c for c in wrong if c["same_sent"]]
    other = [c for c in wrong if not c["same_sent"]]
    out = {}
    for name, pop in (("ALL_WRONG", wrong), ("twin", twin), ("other_sentence", other)):
        n_pop = len(pop)
        row = {"n": n_pop}
        for k in (1, 2, 3, 4):
            row[f"top{k}"] = (sum(c["rank"] <= k for c in pop) / n_pop) if n_pop else None
        out[name] = row
    return out, cells   # cells reused below to join 'same_sent' class onto the page tables


# ===========================================================================
# PART 2 -- PAGES: top-3 real-token candidates per GIVEN slot, read-time features
# ===========================================================================

def _digit_val(gdig_row_j):
    return int("".join(str(int(x)) for x in gdig_row_j))


def build_pages(raw, tok, same_sent_of):
    """raw: the npz (membrane_raw schema). same_sent_of: {(i,j): bool} from goldmention's cells
    (class join; independent of this function). Returns list of slot records:
    {i,j,gold_val,same_sent,pages:[{token,rank,mass,gap,pos_frac,sent_idx,claimed,same_sent_prev,
    mag_band,value,label}]}."""
    ids, tokmask, sent = raw["ids"], raw["tokmask"], raw["sent"]
    gpres, gftype, gdig = raw["g_presence"], raw["g_ftype"], raw["g_digits"]
    fat0 = raw["fat_mass"][:, 0, :, :].astype(np.float32)
    n = ids.shape[0]

    runs_cache = {}

    def runs_of(i):
        if i not in runs_cache:
            runs_cache[i] = MS._digit_runs(tok, ids[i], T_ALG)
        return runs_cache[i]

    def value_of_token(i, t):
        for a, b, val in runs_of(i):
            if a <= t < b:
                return val
        return None

    row_given = collections.defaultdict(list)
    top3_by_slot = {}
    for i in range(n):
        real = tokmask[i] > 0
        idxs = np.where(real)[0]
        if len(idxs) == 0:
            continue
        for j in range(L_FAC):
            if gpres[i, j] < 0.5 or int(gftype[i, j]) != 1:
                continue
            m = fat0[i, j]
            tot = float(m[idxs].sum())
            if tot <= 0:
                continue
            order = idxs[np.argsort(-m[idxs])]
            top = order[:3]
            top3_by_slot[(i, j)] = [(int(t), float(m[t]) / tot) for t in top]
            row_given[i].append(j)
    for i in row_given:
        row_given[i].sort()

    top1_value, top1_sent = {}, {}
    for (i, j), pages in top3_by_slot.items():
        t0 = pages[0][0]
        top1_value[(i, j)] = value_of_token(i, t0)
        top1_sent[(i, j)] = int(sent[i, t0])

    out = []
    for (i, j), pages in top3_by_slot.items():
        gold_val = _digit_val(gdig[i, j])
        other_vals = {top1_value[(i, j2)] for j2 in row_given[i]
                      if j2 != j and top1_value.get((i, j2)) is not None}
        prevs = [j2 for j2 in row_given[i] if j2 < j]
        prev_j = max(prevs) if prevs else None
        prev_sent = top1_sent.get((i, prev_j)) if prev_j is not None else None

        best_mass = pages[0][1]
        ntk = int(tokmask[i].sum())
        page_recs = []
        for rank0, (t, mass) in enumerate(pages):
            val = value_of_token(i, t)
            s_idx = int(sent[i, t])
            page_recs.append(dict(
                token=t, rank=rank0 + 1, mass=mass, gap_to_best=best_mass - mass,
                pos_frac=t / max(1, ntk), sent_idx=s_idx,
                claimed=bool(val is not None and val in other_vals),
                same_sent_prev=bool(prev_sent is not None and s_idx == prev_sent),
                mag_band=(len(str(val)) if val is not None else -1),
                value=val, label=bool(val is not None and val == gold_val),
            ))
        out.append(dict(i=int(i), j=int(j), gold_val=gold_val,
                         same_sent=bool(same_sent_of.get((int(i), int(j)), False)),
                         pages=page_recs))
    return out


FEATURE_NAMES = ["mass", "gap_to_best", "rank1", "rank2", "rank3", "pos_frac",
                 "sent_idx", "claimed", "same_sent_prev", "mag_band"]


def page_features(p):
    r1, r2, r3 = (1.0 if p["rank"] == k else 0.0 for k in (1, 2, 3))
    return [p["mass"], p["gap_to_best"], r1, r2, r3, p["pos_frac"],
            float(p["sent_idx"]), float(p["claimed"]), float(p["same_sent_prev"]), float(p["mag_band"])]


def flatten(records):
    X, y, owner = [], [], []
    for rec in records:
        for p in rec["pages"]:
            X.append(page_features(p)); y.append(int(p["label"])); owner.append((rec["i"], rec["j"]))
    return np.asarray(X, np.float64), np.asarray(y, np.int64), owner


# ===========================================================================
# PART 3/4 -- fit + evaluate
# ===========================================================================

def make_model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))])


def chosen_page(model, rec):
    X = np.asarray([page_features(p) for p in rec["pages"]], np.float64)
    probs = model.predict_proba(X)[:, 1]
    return rec["pages"][int(np.argmax(probs))]


def eval_records(model, records):
    rows = []
    for rec in records:
        p1 = rec["pages"][0]
        chosen = chosen_page(model, rec)
        rows.append(dict(i=rec["i"], j=rec["j"], same_sent=rec["same_sent"],
                          first_look_ok=p1["label"], chosen_ok=chosen["label"]))
    return rows


def summarize(rows, pred_field="chosen_ok"):
    n = len(rows)
    if n == 0:
        return dict(n=0, acc=None)
    return dict(n=n, acc=sum(r[pred_field] for r in rows) / n,
                first_look_acc=sum(r["first_look_ok"] for r in rows) / n)


def loro_ceiling(wild_records):
    X, y, owner = flatten(wild_records)
    rows_arr = np.array([o[0] for o in owner])
    uniq_rows = np.unique(rows_arr)
    out_rows = []
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    for ri in uniq_rows:
        train_mask = rows_arr != ri
        if y[train_mask].sum() == 0 or y[train_mask].sum() == train_mask.sum():
            continue   # degenerate (shouldn't happen at this n, guarded anyway)
        model = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))])
        model.fit(X[train_mask], y[train_mask])
        recs_i = [r for r in wild_records if r["i"] == ri]
        out_rows.extend(eval_records(model, recs_i))
    return out_rows


def main():
    tok = _tokenizer()

    print("[pages] PART 1: coverage (wild)", flush=True)
    cov, wild_cells = coverage_table(tok)
    same_sent_wild = {(c["i"], c["j"]): c["same_sent"] for c in wild_cells}

    print("[pages] PART 2: building pages (wild + valid2)", flush=True)
    raw_wild = np.load(RAW_WILD)
    wild_records = build_pages(raw_wild, tok, same_sent_wild)

    assert os.path.exists(RAW_V2), f"{RAW_V2} missing -- run scripts/picker/pages_collect_valid2.py first"
    raw_v2 = np.load(RAW_V2)
    ok_v2 = _load_ok_generic(OK_V2)
    # class join for valid2 pages too (via goldmention's own _collect_cells, reused on the new npz)
    n_v2 = raw_v2["ids"].shape[0]
    cells_v2, _, _ = GM._collect_cells(
        raw_v2["ids"], raw_v2["tokmask"], raw_v2["sent"], raw_v2["g_presence"], raw_v2["g_ftype"],
        raw_v2["g_digits"], raw_v2["fat_mass"][:, 0, :, :].astype(np.float32), ok_v2, n_v2, L_FAC, T_ALG, tok)
    same_sent_v2 = {(c["i"], c["j"]): c["same_sent"] for c in cells_v2}
    v2_records = build_pages(raw_v2, tok, same_sent_v2)

    print(f"[pages] wild slots={len(wild_records)}  valid2 slots={len(v2_records)}", flush=True)

    print("[pages] PART 3: fit reranker on valid2, evaluate on wild", flush=True)
    Xtr, ytr, _ = flatten(v2_records)
    model = make_model()
    model.fit(Xtr, ytr)
    coefs = dict(zip(FEATURE_NAMES, model.named_steps["lr"].coef_[0].tolist()))

    wild_eval = eval_records(model, wild_records)
    all_rows = wild_eval
    wrong_fl = [r for r in wild_eval if not r["first_look_ok"]]
    right_fl = [r for r in wild_eval if r["first_look_ok"]]
    twin_wrong_fl = [r for r in wrong_fl if r["same_sent"]]
    other_wrong_fl = [r for r in wrong_fl if not r["same_sent"]]

    sm_all = summarize(all_rows)
    sm_wrong = summarize(wrong_fl)
    sm_twin = summarize(twin_wrong_fl)
    sm_other = summarize(other_wrong_fl)
    cost_right = sum(not r["chosen_ok"] for r in right_fl) / max(len(right_fl), 1)

    print("[pages] PART 4: LORO ceiling on wild (fit-on-wild-minus-row, reported as CEILING)", flush=True)
    loro_rows = loro_ceiling(wild_records)
    loro_wrong_fl = [r for r in loro_rows if not r["first_look_ok"]]
    loro_twin = [r for r in loro_wrong_fl if r["same_sent"]]
    loro_other = [r for r in loro_wrong_fl if not r["same_sent"]]
    loro_right_fl = [r for r in loro_rows if r["first_look_ok"]]
    loro_sm_all = summarize(loro_rows)
    loro_sm_wrong = summarize(loro_wrong_fl)
    loro_sm_twin = summarize(loro_twin)
    loro_sm_other = summarize(loro_other)
    loro_cost = sum(not r["chosen_ok"] for r in loro_right_fl) / max(len(loro_right_fl), 1)

    bar_twin_gain = (sm_twin["acc"] - sm_twin["first_look_acc"]) if sm_twin["n"] else None
    bar_cost_ok = cost_right <= 0.02
    bar_twin_ok = (bar_twin_gain is not None and bar_twin_gain >= 0.10)
    verdict = "BARS MET" if (bar_twin_ok and bar_cost_ok) else "BARS NOT MET"

    out = dict(coverage=cov, feature_names=FEATURE_NAMES, coefficients=coefs,
               n_valid2_rows=int(raw_v2["ids"].shape[0]), n_valid2_slots=len(v2_records),
               n_wild_slots=len(wild_records),
               valid2_to_wild=dict(all=sm_all, wrong_at_first_look=sm_wrong,
                                    twin_wrong_at_first_look=sm_twin,
                                    other_sentence_wrong_at_first_look=sm_other,
                                    cost_on_right_at_first_look=cost_right,
                                    n_right_at_first_look=len(right_fl)),
               loro_ceiling_wild=dict(all=loro_sm_all, wrong_at_first_look=loro_sm_wrong,
                                       twin_wrong_at_first_look=loro_sm_twin,
                                       other_sentence_wrong_at_first_look=loro_sm_other,
                                       cost_on_right_at_first_look=loro_cost,
                                       n_right_at_first_look=len(loro_right_fl)),
               bars=dict(twin_gain=bar_twin_gain, twin_gain_bar=0.10, cost=cost_right, cost_bar=0.02,
                         verdict=verdict),
               not_done=["latest-before / governor-chain owner feature (per-page integration cost "
                         "named in the module docstring, skipped rather than rushed)"])

    lines = []
    P = lines.append
    P("=" * 90)
    P("PART 1 -- COVERAGE (wild, n=311), top-k attended-token share among WRONG-at-final-decode GIVEN slots")
    P("=" * 90)
    for name in ("ALL_WRONG", "twin", "other_sentence"):
        r = cov[name]
        P(f"  {name:16s} n={r['n']:4d}  top1={r['top1']:.3f}  top2={r['top2']:.3f}  "
          f"top3={r['top3']:.3f}  top4={r['top4']:.3f}" if r["n"] else f"  {name}: n=0")
    P("-" * 90)
    P("=" * 90)
    P("PART 2/3 -- RERANKER fit on valid2 (n=128 rows), measured on wild (n=311, never fit on)")
    P("=" * 90)
    P(f"  valid2 GIVEN slots (pages) = {len(v2_records)}  |  wild GIVEN slots (pages) = {len(wild_records)}")
    P(f"  ALL wild slots:              n={sm_all['n']:4d}  reranker={sm_all['acc']:.3f}  first-look={sm_all['first_look_acc']:.3f}")
    P(f"  wrong-at-first-look:         n={sm_wrong['n']:4d}  reranker={sm_wrong['acc']:.3f}  first-look={sm_wrong['first_look_acc']:.3f} (first-look is 0 by definition)")
    P(f"    twin class:                n={sm_twin['n']:4d}  reranker={sm_twin['acc'] if sm_twin['n'] else None}")
    P(f"    other-sentence class:      n={sm_other['n']:4d}  reranker={sm_other['acc'] if sm_other['n'] else None}")
    P(f"  cost on right-at-first-look: n={len(right_fl):4d}  flip-to-wrong rate={cost_right:.4f}")
    P("-" * 90)
    P(f"  BARS: twin gain >= 0.10 -> measured {bar_twin_gain if bar_twin_gain is not None else 'n/a'} ; "
      f"cost <= 0.02 -> measured {cost_right:.4f}  ==>  {verdict}")
    P("=" * 90)
    P("PART 4 -- LORO CEILING on wild itself (fit on wild-minus-row; NEVER deployable)")
    P("=" * 90)
    P(f"  ALL wild slots:              n={loro_sm_all['n']:4d}  reranker={loro_sm_all['acc']:.3f}  first-look={loro_sm_all['first_look_acc']:.3f}")
    P(f"  wrong-at-first-look:         n={loro_sm_wrong['n']:4d}  reranker={loro_sm_wrong['acc']:.3f}")
    P(f"    twin class:                n={loro_sm_twin['n']:4d}  reranker={loro_sm_twin['acc'] if loro_sm_twin['n'] else None}")
    P(f"    other-sentence class:      n={loro_sm_other['n']:4d}  reranker={loro_sm_other['acc'] if loro_sm_other['n'] else None}")
    P(f"  cost on right-at-first-look: n={len(loro_right_fl):4d}  flip-to-wrong rate={loro_cost:.4f}")
    P("-" * 90)
    P("FEATURES (valid2-fit logistic coefficients, StandardScaler'd):")
    for k, v in coefs.items():
        P(f"    {k:16s} {v:+.3f}")
    P("-" * 90)
    P("NOT DONE: " + "; ".join(out["not_done"]))

    txt = "\n".join(lines)
    print(txt)
    with open(OUT_TXT, "w") as f:
        f.write(txt + "\n")
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[pages] wrote {OUT_TXT} / {OUT_JSON}")


if __name__ == "__main__":
    main()
