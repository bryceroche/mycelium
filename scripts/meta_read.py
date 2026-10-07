"""scripts/meta_read.py -- THE META READ (2026-10-06, zero-GPU, delegate; ledger
docs/phase1_skeleton_spec.md 2026-10-06 17:08 "THE META").

QUESTION (Bryce's "meta"): is the wild wall (PMS8_241, the record body) on META rows (common
factor-graph motifs) or on TAIL rows (rare/unseen motifs)? PREDICTION PINNED IN THE LEDGER: the
wall is MOSTLY ON META ROWS, so a frequency prior over the first look's candidates is INERT.
MEASURED (see THE READING at the bottom of this file's output): the first clause is REFUTED --
the wall is mostly on TAIL rows (63.6% vs 36.4% meta), tracking the population split, because meta
rows are measurably easier (9.2% vs 1.6% correct) -- but the second clause HOLDS: the meta prior
is inert as a picker (15/311 vs top-1's 14/311), because the candidate lattice it re-ranks almost
never contains the gold motif as an alternative at all (oracle ceiling 7.7% coarse / 5.1% fine).

THE KNOT: mycelium.canonicalizer.canonical_digest(factors, query_var, n_vars) -- the one identity
door (scripts/hash_audit_iso.py::canon), values INCLUDED (givens/mod/fdiv/pct carry their literal
value; level-0 macro expansion first, per the floor-identity protocol). Checked by reading both
files: canonical_digest does NOT abstract values, so per the task's own instruction a SECOND,
COARSER knot is computed here (op/ftype skeleton only, values stripped -- the "common motif" the
ledger entry means: "total-then-difference, rate x time, percent-of", not "gave exactly 8 and 8 and
8"). The coarse canon below is a value-abstracted re-implementation of hash_audit_iso.canon's exact
WL loop (6 rounds, same bipartite var/factor refinement, same query-distinguished seed), reusing
hash_audit_iso.level0 for macro expansion (never editing hash_audit_iso.py or canonicalizer.py).

ARTIFACTS REUSED, NEVER EDITED, NEVER RE-SOLVED:
  - .cache/form_mix_pm35c.jsonl        -- the diet PMS8_241 trained on (50653 rows)
  - .cache/wild_admitted_holdout.jsonl -- the 311-row wild holdout (measured, never trained)
  - .cache/courtroom_PMS8_241.pkl      -- ['final']['results'][i] = {i, key, top1{status,correct,
                                           parse,...}, winner, verdict, ...} -- top1's status+correct
                                           gives the EXACT chain_acc-masked row verdict (confirmed:
                                           final.top1_correct == final.verdict counts == 14/311,
                                           202 wrong, 95 refused, matching the ledger's 10-06 13:34
                                           / 15:06 banked reads).
  - .cache/ps_legal_wild_PMS8_241.npz  -- rows/slots/ok: the masked (LV_LEGAL=num) per-GOLD-SLOT
                                           fact-exact read, 2051 gold slots over the 311 rows --
                                           grouped by row for the per-row wrong-slot share.
  - .cache/picker/cand_wild_PMS8_241.pkl -- the courtroom's own candidate lattice (THE LAWYERS): per
                                           row, every BAILIFF-SOLVED survivor's decoded parse +
                                           'correct' (already graded against the row's custody key
                                           by gen_candidates.generate(), no re-solving here) +
                                           'rank' (edits-from-top1, the bailiff's own order).
  - .cache/dump_wild_HS_241.pkl / dump_wild_RK_241.pkl + their ps_legal_wild_*.npz -- loop_val.py's
    LV_DUMP / LV_PER_SLOT reads: PER-GOLD-SLOT predictions only (no full 24-slot raw logits), so a
    full row-level solve+grade (chain_acc's top1 correct/refused/wrong) is NOT reconstructible from
    them without a fresh forward pass (GPU). Reported here: per-slot wrong share by knot bucket
    ONLY for HS_241/RK_241, explicitly flagged as not a row-verdict read.

Nothing is retrained, resolved, or re-decoded; every number below is a read off an existing file.
"""
import sys
import os
import json
import hashlib
import pickle
from collections import defaultdict, Counter

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

import numpy as np

from mycelium.canonicalizer import canonical_digest
from hash_audit_iso import level0   # macro expansion only; canon() itself is NOT reused (keeps values)

DIET_PATH = ".cache/form_mix_pm35c.jsonl"
WILD_PATH = ".cache/wild_admitted_holdout.jsonl"
COURTROOM_PKL = ".cache/courtroom_PMS8_241.pkl"
PS_LEGAL_NPZ = ".cache/ps_legal_wild_PMS8_241.npz"
CAND_WILD_PKL = ".cache/picker/cand_wild_PMS8_241.pkl"
OUT_TXT = ".cache/meta_read_PMS8_241.txt"

SECOND_BODIES = [
    ("HS_241", ".cache/dump_wild_HS_241.pkl", ".cache/ps_legal_wild_HS_241.npz"),
    ("RK_241", ".cache/dump_wild_RK_241.pkl", ".cache/ps_legal_wild_RK_241.npz"),
]


# ======================================================================================
# THE COARSE KNOT -- hash_audit_iso.canon's WL loop, verbatim in shape, values abstracted
# (op/ftype retained, exactly schema_miner.skind's own abstraction convention, applied to
# the WHOLE graph rather than mined subgraphs).
# ======================================================================================
def _fdesc_coarse(f):
    ft = f["ftype"]
    if ft == "rel":
        return (("rel", f["op"]),
                (("a", f["args"][0]), ("a", f["args"][1]), ("r", f["result"])))
    if ft == "given":
        return (("giv",), (("v", f["var"]),))
    if ft == "mod":
        return (("mod",), (("s", f["var"]), ("r", f["result"])))
    if ft == "fdiv":
        return (("fdv",), (("s", f["var"]), ("r", f["result"])))
    if ft == "pct":
        return (("pct",), (("p", f["args"][0]), ("b", f["args"][1])))
    if ft == "sel":
        return (("sel", f["sel"]),
                (("a", f["args"][0]), ("a", f["args"][1]), ("r", f["result"])))
    raise ValueError(ft)


def coarse_digest(factors, query, n_vars=24):
    facs_l0, nv = level0({"factors": factors, "n_vars": n_vars})
    facs = [_fdesc_coarse(f) for f in facs_l0]
    col = {v: ("Q" if v == query else ".") for v in range(nv)}
    for _ in range(6):
        fcols = []
        for kind, mem in facs:
            aa = tuple(sorted(col[m] for r, m in mem if r == "a"))
            rest = tuple((r, col[m]) for r, m in mem if r != "a")
            fcols.append((kind, aa, rest))
        inc = defaultdict(list)
        for (kind, mem), fc in zip(facs, fcols):
            for r, m in mem:
                inc[m].append((fc, r))
        col = {v: (col[v], tuple(sorted(map(repr, inc[v]))))
               for v in range(nv)}
        ranks = {c: i for i, c in enumerate(sorted(set(map(repr, col.values()))))}
        col = {v: ranks[repr(c)] for v, c in col.items()}
    sig = sorted(repr((kind,
                       tuple(sorted(col[m] for r, m in mem if r == "a")),
                       tuple((r, col[m]) for r, m in mem if r != "a")))
                 for kind, mem in facs)
    return hashlib.sha256(("|".join(sig) + f"#q{col[query]}").encode()).hexdigest()[:16]


def fine_knot(row):
    return canonical_digest(row["factors"], row["query_var"], row.get("n_vars", 24))[:16]


def coarse_knot(row):
    return coarse_digest(row["factors"], row["query_var"], row.get("n_vars", 24))


def coarse_knot_of_parse(parse, query, n_vars=24):
    facs = [f for f in parse]  # extra keys (_slot) are ignored by _fdesc_coarse's field reads
    return coarse_digest(facs, query, n_vars)


# ======================================================================================
# 1. THE DIET KNOT TABLE
# ======================================================================================
def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


def knot_table(rows, keyfn):
    counts = Counter()
    for r in rows:
        counts[keyfn(r)] += 1
    return counts


def coverage_curve(counts, total, fractions=(0.50, 0.80, 0.95)):
    ordered = sorted(counts.values(), reverse=True)
    out = {}
    cum = 0
    idx = 0
    targets = sorted(fractions)
    ti = 0
    for n_knots, c in enumerate(ordered, start=1):
        cum += c
        while ti < len(targets) and cum >= targets[ti] * total:
            out[targets[ti]] = n_knots
            ti += 1
    for f in targets:
        out.setdefault(f, len(ordered))
    return out


def fmt_pct(x):
    return f"{100.0 * x:.1f}%"


def main():
    lines = []

    def P(s=""):
        print(s)
        lines.append(s)

    P("=" * 96)
    P("THE META READ (2026-10-06) -- PMS8_241 -- zero-GPU -- .venv/bin/python3 scripts/meta_read.py")
    P("=" * 96)

    diet = load_jsonl(DIET_PATH)
    wild = load_jsonl(WILD_PATH)
    P(f"diet rows: {len(diet)} ({DIET_PATH})")
    P(f"wild rows: {len(wild)} ({WILD_PATH})")

    # ---- knot computation (diet) ----
    diet_fine = []
    diet_coarse = []
    bad = 0
    for r in diet:
        try:
            diet_fine.append(fine_knot(r))
            diet_coarse.append(coarse_knot(r))
        except Exception:
            bad += 1
            diet_fine.append(None)
            diet_coarse.append(None)
    if bad:
        P(f"  WARNING: {bad} diet rows failed knot computation (excluded below)")
    fine_counts = Counter(k for k in diet_fine if k is not None)
    coarse_counts = Counter(k for k in diet_coarse if k is not None)
    n_diet_ok = sum(1 for k in diet_fine if k is not None)

    P("")
    P("-" * 96)
    P("(1a) DIET KNOT TABLE -- FINE (canonical_digest; values INCLUDED -- the exact knot-ID door)")
    P("-" * 96)
    P(f"distinct fine knots: {len(fine_counts)} over {n_diet_ok} rows "
      f"(mean reps/knot {n_diet_ok / max(len(fine_counts), 1):.2f})")
    n_singletons = sum(1 for c in fine_counts.values() if c == 1)
    P(f"singleton fine knots (count==1): {n_singletons} ({fmt_pct(n_singletons / max(len(fine_counts), 1))} of distinct knots)")
    cov_fine = coverage_curve(fine_counts, n_diet_ok)
    P(f"coverage curve (fine): {cov_fine[0.50]} knots -> 50%, {cov_fine[0.80]} knots -> 80%, "
      f"{cov_fine[0.95]} knots -> 95% of diet rows")
    P("top-20 fine knots by count:")
    for k, c in fine_counts.most_common(20):
        P(f"    {k}  count={c:6d}  share={fmt_pct(c / n_diet_ok)}")

    P("")
    P("-" * 96)
    P("(1b) DIET KNOT TABLE -- COARSE (op/ftype skeleton, values stripped -- THE META's own motif)")
    P("-" * 96)
    P(f"distinct coarse knots: {len(coarse_counts)} over {n_diet_ok} rows "
      f"(mean reps/knot {n_diet_ok / max(len(coarse_counts), 1):.2f})")
    n_singletons_c = sum(1 for c in coarse_counts.values() if c == 1)
    P(f"singleton coarse knots (count==1): {n_singletons_c} ({fmt_pct(n_singletons_c / max(len(coarse_counts), 1))} of distinct knots)")
    cov_coarse = coverage_curve(coarse_counts, n_diet_ok)
    P(f"coverage curve (coarse): {cov_coarse[0.50]} knots -> 50%, {cov_coarse[0.80]} knots -> 80%, "
      f"{cov_coarse[0.95]} knots -> 95% of diet rows")
    P("top-20 coarse knots by count:")
    for k, c in coarse_counts.most_common(20):
        P(f"    {k}  count={c:6d}  share={fmt_pct(c / n_diet_ok)}")

    top_k_meta = cov_coarse[0.50]
    meta_knot_set = set(k for k, _ in coarse_counts.most_common(top_k_meta))
    P("")
    P(f"TOP-K META (coarse knots covering 50% of the diet): k={top_k_meta} knots")

    # ======================================================================================
    # 2. WILD ROWS: knot + diet frequency + row verdict (PMS8_241) + per-slot wrong share
    # ======================================================================================
    P("")
    P("-" * 96)
    P("(2) WILD ROWS x KNOT FREQUENCY x ROW VERDICT (PMS8_241)")
    P("-" * 96)

    courtroom = pickle.load(open(COURTROOM_PKL, "rb"))["final"]
    results = {r["i"]: r for r in courtroom["results"]}
    P(f"courtroom bank: n={courtroom['n']} top1_correct={courtroom['top1_correct']} "
      f"verdict_wrong={courtroom['verdict_wrong']} verdict_refused={courtroom['verdict_refused']}")

    ps = np.load(PS_LEGAL_NPZ, allow_pickle=True)
    ps_rows, ps_ok = ps["rows"], ps["ok"]
    per_row_slot_total = Counter()
    per_row_slot_wrong = Counter()
    for r, ok in zip(ps_rows, ps_ok):
        per_row_slot_total[int(r)] += 1
        per_row_slot_wrong[int(r)] += 0 if ok else 1

    def row_verdict(i):
        res = results.get(i)
        if res is None:
            return "missing"
        t1 = res["top1"]
        if t1.get("status") != "solved":
            return "refused"
        return "correct" if t1.get("correct") else "wrong"

    def freq_bucket(n):
        if n == 0:
            return "unseen"
        if n < 10:
            return "1-9"
        if n < 100:
            return "10-99"
        return "100+"

    wild_recs = []
    for idx, r in enumerate(wild):
        fk = fine_knot(r)
        ck = coarse_knot(r)
        ffreq = fine_counts.get(fk, 0)
        cfreq = coarse_counts.get(ck, 0)
        verdict = row_verdict(idx)
        slot_tot = per_row_slot_total.get(idx, 0)
        slot_wrong = per_row_slot_wrong.get(idx, 0)
        slot_wrong_share = (slot_wrong / slot_tot) if slot_tot else None
        wild_recs.append(dict(i=idx, fine_knot=fk, coarse_knot=ck, fine_freq=ffreq,
                               coarse_freq=cfreq, verdict=verdict,
                               slot_total=slot_tot, slot_wrong=slot_wrong,
                               slot_wrong_share=slot_wrong_share,
                               is_meta=ck in meta_knot_set))

    P(f"wild rows with a courtroom verdict: {sum(1 for w in wild_recs if w['verdict'] != 'missing')}/{len(wild_recs)}")
    n_fine_unseen = sum(1 for w in wild_recs if w["fine_freq"] == 0)
    n_coarse_unseen = sum(1 for w in wild_recs if w["coarse_freq"] == 0)
    P(f"wild rows whose FINE knot is unseen in the diet: {n_fine_unseen}/{len(wild_recs)} "
      f"({fmt_pct(n_fine_unseen / len(wild_recs))}) -- values rarely repeat exactly by construction")
    P(f"wild rows whose COARSE knot is unseen in the diet: {n_coarse_unseen}/{len(wild_recs)} "
      f"({fmt_pct(n_coarse_unseen / len(wild_recs))})")

    P("")
    P("per-row table (first 20 rows shown; full table is the pinned deliverable text below / use the pkl)")
    P(f"{'i':>4} {'coarse_knot':16} {'coarse_freq':>11} {'bucket':>7} {'verdict':>8} {'slot_wrong_share':>16}")
    for w in wild_recs[:20]:
        sw = f"{w['slot_wrong_share']:.2f}" if w["slot_wrong_share"] is not None else "n/a"
        P(f"{w['i']:>4} {w['coarse_knot']:16} {w['coarse_freq']:>11} {freq_bucket(w['coarse_freq']):>7} "
          f"{w['verdict']:>8} {sw:>16}")

    # ---- row verdict x frequency bucket table ----
    P("")
    P("-" * 96)
    P("ROW VERDICT x KNOT-FREQUENCY BUCKET (coarse knot; bucket = unseen / 1-9 / 10-99 / 100+)")
    P("-" * 96)
    buckets = ["unseen", "1-9", "10-99", "100+"]
    tab = {b: Counter() for b in buckets}
    for w in wild_recs:
        tab[freq_bucket(w["coarse_freq"])][w["verdict"]] += 1
    P(f"{'bucket':>8} {'n':>5} {'correct':>8} {'wrong':>7} {'refused':>8} {'correct_rate':>13}")
    for b in buckets:
        n = sum(tab[b].values())
        cr = tab[b]["correct"]
        wr = tab[b]["wrong"]
        rf = tab[b]["refused"]
        rate = fmt_pct(cr / n) if n else "n/a"
        P(f"{b:>8} {n:>5} {cr:>8} {wr:>7} {rf:>8} {rate:>13}")

    # ---- meta vs tail (meta = coarse knot in top-k covering 50% of diet) ----
    P("")
    P("-" * 96)
    P(f"META (top-{top_k_meta} coarse knots, 50% of diet mass) vs TAIL -- share of ROWS and of THE WALL")
    P("-" * 96)
    n_meta = sum(1 for w in wild_recs if w["is_meta"])
    n_tail = len(wild_recs) - n_meta
    P(f"wild rows on a META knot: {n_meta}/{len(wild_recs)} ({fmt_pct(n_meta / len(wild_recs))})")
    P(f"wild rows on a TAIL knot: {n_tail}/{len(wild_recs)} ({fmt_pct(n_tail / len(wild_recs))})")

    wall_rows = [w for w in wild_recs if w["verdict"] in ("wrong", "refused")]
    n_wall_meta = sum(1 for w in wall_rows if w["is_meta"])
    n_wall_tail = len(wall_rows) - n_wall_meta
    P(f"THE WALL (wrong+refused rows): {len(wall_rows)}/{len(wild_recs)} "
      f"({fmt_pct(len(wall_rows) / len(wild_recs))})")
    P(f"  of the wall: on META knots {n_wall_meta} ({fmt_pct(n_wall_meta / max(len(wall_rows), 1))}), "
      f"on TAIL knots {n_wall_tail} ({fmt_pct(n_wall_tail / max(len(wall_rows), 1))})")

    right_rows = [w for w in wild_recs if w["verdict"] == "correct"]
    n_right_meta = sum(1 for w in right_rows if w["is_meta"])
    P(f"RIGHT rows (correct): {len(right_rows)}/{len(wild_recs)}; of those on META knots "
      f"{n_right_meta} ({fmt_pct(n_right_meta / max(len(right_rows), 1))})")

    slot_wrong_total = sum(w["slot_wrong"] for w in wild_recs)
    slot_wrong_meta = sum(w["slot_wrong"] for w in wild_recs if w["is_meta"])
    P(f"WRONG SLOTS (ps_legal, masked fac-exact): total {slot_wrong_total}; "
      f"on META-knot rows {slot_wrong_meta} ({fmt_pct(slot_wrong_meta / max(slot_wrong_total, 1))}); "
      f"on TAIL-knot rows {slot_wrong_total - slot_wrong_meta} "
      f"({fmt_pct((slot_wrong_total - slot_wrong_meta) / max(slot_wrong_total, 1))})")

    correct_rate_meta = n_right_meta / max(n_meta, 1)
    correct_rate_tail = (len(right_rows) - n_right_meta) / max(n_tail, 1)
    P(f"correctness rate on META rows: {fmt_pct(correct_rate_meta)}  |  on TAIL rows: {fmt_pct(correct_rate_tail)}")

    # ======================================================================================
    # 3. HS_241 / RK_241 -- per-slot wrong share by knot bucket ONLY (no full-row verdict
    #    reconstructible from the banked LV_DUMP/ps_legal artifacts without a GPU forward pass)
    # ======================================================================================
    P("")
    P("=" * 96)
    P("(3) SECOND BODIES -- HS_241 / RK_241 (per-slot wrong share by knot bucket ONLY)")
    P("=" * 96)
    for tag, dump_path, ps_path in SECOND_BODIES:
        if not (os.path.exists(dump_path) and os.path.exists(ps_path)):
            P(f"{tag}: MISSING ({dump_path} exists={os.path.exists(dump_path)}, "
              f"{ps_path} exists={os.path.exists(ps_path)}) -- skipped")
            continue
        P(f"{tag}: dump={dump_path} ps_legal={ps_path}")
        P(f"  CAVEAT: {os.path.basename(dump_path)} is loop_val.py's LV_DUMP format (per-GOLD-SLOT "
          f"predictions only, no full 24-slot raw logits for the model's own top-1 PARSE) -- a "
          f"row-level correct/refused/wrong verdict (chain_acc's masked read) is NOT reconstructible "
          f"from it without a fresh GPU forward pass. Only the per-slot wrong share (ps_legal's own "
          f"fact-exact 'ok' column) is reported here, bucketed by the WILD ROW's gold coarse knot "
          f"frequency in the SAME PM35c diet (the knot identity does not depend on which body is read).")
        psx = np.load(ps_path, allow_pickle=True)
        pr, po = psx["rows"], psx["ok"]
        tot2 = Counter(); wrong2 = Counter()
        for r, ok in zip(pr, po):
            tot2[int(r)] += 1
            wrong2[int(r)] += 0 if ok else 1
        tab2 = {b: [0, 0] for b in buckets}   # [slot_total, slot_wrong]
        for w in wild_recs:
            b = freq_bucket(w["coarse_freq"])
            tab2[b][0] += tot2.get(w["i"], 0)
            tab2[b][1] += wrong2.get(w["i"], 0)
        P(f"  {'bucket':>8} {'slots':>7} {'wrong':>7} {'wrong_share':>12}")
        for b in buckets:
            st, sw = tab2[b]
            share = fmt_pct(sw / st) if st else "n/a"
            P(f"  {b:>8} {st:>7} {sw:>7} {share:>12}")
        n2 = sum(v for v in tot2.values())
        wtot2 = sum(v for v in wrong2.values())
        P(f"  TOTAL: {n2} gold slots, {wtot2} wrong ({fmt_pct(wtot2 / max(n2, 1))}) "
          f"[c.f. PMS8_241's own ps_legal: {sum(per_row_slot_total.values())} slots, "
          f"{sum(per_row_slot_wrong.values())} wrong "
          f"({fmt_pct(sum(per_row_slot_wrong.values()) / max(sum(per_row_slot_total.values()), 1))})]")

    # ======================================================================================
    # 4. THE META PRIOR AS A PICKER ANGLE
    # ======================================================================================
    P("")
    P("=" * 96)
    P("(4) THE META PRIOR AS A PICKER ANGLE (re-rank the courtroom's solved survivors by COARSE-")
    P("    KNOT diet frequency; ties by the bailiff's own rank order)")
    P("=" * 96)
    if not os.path.exists(CAND_WILD_PKL):
        P(f"MISSING: {CAND_WILD_PKL} -- picker angle skipped")
    else:
        cand = pickle.load(open(CAND_WILD_PKL, "rb"))
        cand_by_i = {c["i"]: c for c in cand}
        meta_correct = meta_wrong = meta_refused = 0
        fixes, regressions = [], []
        oracle_coarse_hits = 0
        oracle_fine_hits = 0
        oracle_denom = 0
        for w in wild_recs:
            i = w["i"]
            top1_status = w["verdict"]
            c = cand_by_i.get(i)
            if c is None or not c["branches"]:
                pick_status = top1_status   # bailiff-empty / no candidates -- keep top1, per courtroom's own rule
            else:
                branches = list(c["branches"].values())
                gold_ck = w["coarse_knot"]
                gold_fk = w["fine_knot"]
                oracle_denom += 1
                any_coarse = False
                any_fine = False
                for b in branches:
                    try:
                        bck = coarse_knot_of_parse(b["parse"], c["q"])
                    except Exception:
                        bck = None
                    if bck == gold_ck:
                        any_coarse = True
                    try:
                        bfk = canonical_digest(b["parse"], c["q"], 24)[:16]
                    except Exception:
                        bfk = None
                    if bfk == gold_fk:
                        any_fine = True
                oracle_coarse_hits += int(any_coarse)
                oracle_fine_hits += int(any_fine)

                def freq_of(b):
                    try:
                        ck = coarse_knot_of_parse(b["parse"], c["q"])
                    except Exception:
                        return (-1, b.get("rank", 1 << 30))
                    return (coarse_counts.get(ck, 0), b.get("rank", 1 << 30))

                best = max(branches, key=lambda b: (freq_of(b)[0], -freq_of(b)[1]))
                pick_status = "correct" if best.get("correct") else "wrong"
            if pick_status == "correct":
                meta_correct += 1
            elif pick_status == "refused":
                meta_refused += 1
            else:
                meta_wrong += 1
            if top1_status == "correct" and pick_status != "correct":
                regressions.append(i)
            if top1_status != "correct" and pick_status == "correct":
                fixes.append(i)
        P(f"META-PRIOR PICK: correct {meta_correct}/{len(wild_recs)} | wrong {meta_wrong} | refused {meta_refused}")
        P(f"  (c.f. top-1 baseline {courtroom['top1_correct']}/311; judge/picker bars in the ledger: "
          f"combined_oracle consistency judge 18/311, panel picker 19/311)")
        P(f"  FIXES   (top1 wrong/refused -> meta-prior correct): {len(fixes)}  rows={fixes}")
        P(f"  REGRESSIONS (top1 correct -> meta-prior not correct): {len(regressions)}  rows={regressions}")
        P("")
        P(f"THE ORACLE (upper bound of a knot-level prior, over rows with >=1 solved candidate, "
          f"n={oracle_denom}):")
        P(f"  ANY survivor's COARSE knot matches the gold coarse knot: {oracle_coarse_hits}/{oracle_denom} "
          f"({fmt_pct(oracle_coarse_hits / max(oracle_denom, 1))})")
        P(f"  ANY survivor's FINE knot (exact isomorph) matches the gold: {oracle_fine_hits}/{oracle_denom} "
          f"({fmt_pct(oracle_fine_hits / max(oracle_denom, 1))})")

    # ======================================================================================
    # THE READING
    # ======================================================================================
    P("")
    P("=" * 96)
    P("THE READING (6 lines)")
    P("=" * 96)
    reading = [
        f"1. THE PINNED PREDICTION IS REFUTED ON ITS FIRST CLAUSE: the wall (wrong+refused, {len(wall_rows)}/311) "
        f"sits {fmt_pct(n_wall_tail / max(len(wall_rows), 1))} on TAIL knots vs only "
        f"{fmt_pct(n_wall_meta / max(len(wall_rows), 1))} on META knots (top-{top_k_meta} coarse knots, 50% of "
        f"diet mass) -- the wall is MOSTLY ON TAIL ROWS, roughly tracking the population split "
        f"({fmt_pct(n_tail / len(wild_recs))} tail / {fmt_pct(n_meta / len(wild_recs))} meta), not concentrated on meta.",
        f"2. FREQUENCY DOES PREDICT CORRECTNESS, directionally: correct rate on META rows "
        f"{fmt_pct(correct_rate_meta)} is ~{correct_rate_meta / max(correct_rate_tail, 1e-9):.1f}x TAIL rows' "
        f"{fmt_pct(correct_rate_tail)} -- meta rows are measurably EASIER, which is exactly why they are "
        f"under-represented in the wall relative to tail, not over-represented as predicted.",
        "3. The bucket table is non-monotonic though (unseen 1.6%, 1-9 0.0%, 10-99 9.8%, 100+ 3.6%): frequency "
        "is a real but WEAK and NOISY correctness signal, not a clean ladder -- consistent with a thin lift, not a strong prior.",
        "4. THE FINE knot (values included) is near-useless for grouping by construction -- 96.8% of wild rows' "
        "fine knots are UNSEEN in the diet (numbers rarely repeat exactly); the COARSE (op/ftype-only) knot is "
        "the one that carries real repetition and is the one graded above, exactly as THE META entry's mapping says.",
        f"5. THE META PRIOR AS A PICKER IS INERT ON ROWS, as predicted (second clause holds even though the "
        f"first doesn't): re-ranking the courtroom's survivors by coarse-knot diet frequency buys 15/311 vs "
        f"top-1's 14/311 (4 fixes, 3 regressions, net +1) -- nowhere near the judge's 18 or the picker's 19, "
        f"and bounded low: ANY survivor's coarse knot matches gold in only {fmt_pct(oracle_coarse_hits / max(oracle_denom, 1))} "
        f"of rows (fine/exact-isomorph oracle lower still, {fmt_pct(oracle_fine_hits / max(oracle_denom, 1))}) -- "
        f"the courtroom's candidate lattice rarely EXPLORES a structurally different (gold-matching) motif at all.",
        "6. READING: the model is not failing by reaching for rare motifs -- it fails MOST on tail rows (both in "
        "share of the wall and in raw correctness), and on the common motifs it already gets more right than it "
        "gets wrong relative to its own tail performance; a frequency prior over the EXISTING candidate lattice has "
        "little to bite on because that lattice almost never contains the gold motif as an alternative in the first place.",
    ]
    for r in reading:
        P(r)

    with open(OUT_TXT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[meta-read] wrote {OUT_TXT}")


if __name__ == "__main__":
    main()
