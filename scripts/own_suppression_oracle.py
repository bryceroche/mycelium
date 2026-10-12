"""scripts/own_suppression_oracle.py -- THE OWN-SUPPRESSION ORACLE (2026-10-09, zero training,
reads of banked raw-logit dumps; word given in the ledger entry "PC_241 (THE PAIRWISE COMPARATOR
ROAD...) DOES NOT FIRE", 2026-10-09 16:46). PC_241 removed both of the previous nulls' excuses
(scale, self-driven mask) and still left inverse res pinned at 0.253 -- the entry re-derives what
the inverse res pointer must DO under THE POSITIONAL LAW: for an inverse relation the text's
subtraction/division is stored as add(C, B)->own (reencode_ops), so args = {own, C} (own is
structurally among the args -- confirmed 372/372) and the gold res is a THIRD variable B, the
minuend, neither own nor C. The direction (ALG_DIR's whole project) can only ever say "not own" --
to be RIGHT the pointer must then find B, a binding read of the same class as the args wall. This
script asks the one question that tells "the direction is the lever" (B is the oracle's #2 choice
most of the time, ceiling >= 0.6) apart from "BINDING B is the lever" (B's rank scattered, ceiling
< 0.4): what is the ceiling of ANY direction-aware fix on this body's actual res logits?

Five tasks, both bodies (PMS8_241, PC_241), gold relation slots split fwd/inv by polarity_census's
own classify_row (replays each row's variable-introduction order; imported, not reimplemented):
  (1) baseline res accuracy by form (must reproduce 0.87/0.25 PMS8_241, 0.867/0.253 PC_241)
  (2) THE ORACLE -- inverse slots: clamp the own-index (=k, the structural tell) res logit to
      -inf and re-argmax; accuracy vs gold res B = the ceiling of any direction-aware fix. Forward
      slots, symmetric sanity check: does "always predict own" hit ~1.0 (it must, by definition).
  (3) rank histogram of gold res B among the 24 raw res logits, inverse and forward; on inverse
      slots wrong even under the oracle, classify the oracle's new argmax: own (impossible, it's
      clamped) / the other arg C / a given variable / a derived (rel) variable / B itself (right).
  (4) under the oracle, the margin (winner's logit - B's logit) distribution on inverse slots that
      stay wrong; does it correlate with B being a given vs a derived variable, or with B sharing
      a sentence with the relation's own clause (reuses polarity_census.clause_of + args_census's
      sentence utilities -- no reimplementation).
  (5) the SAME oracle idea applied to ARGS: on inverse slots, is C (the non-own gold arg) found by
      the args head's own top-2/dup decode? Decomposes the already-known inverse args gap (0.51
      fwd/0.31 inv-ish) into "own missing from decode" vs "C wrong" vs "both wrong".

Inputs (raw args/res LOGITS, pre-sigmoid/pre-softmax, chain_acc.py's CA_RAWDUMP convention --
d["args"][j] (24,) BCE logits, d["res"][j] (24,) softmax logits, d["dup"][j] scalar logit, per
gold slot j; d["key"] unused here):
  PMS8_241: .cache/rawslots_wild_PMS8_241.pkl           (banked, unmodified, read-only)
  PC_241:   .cache/ownsup_rawslots_wild_PC_241.pkl      (this read's own collection,
            .cache/ownsup_collect_PC.sh -- NOT a banked artifact, never overwrites one)
Gold factors + classify_row + clause_of (THE POSITIONAL LAW's own fwd/inv/other labeler and the
clause-resolution heuristic): polarity_census.py, read-only import.

usage: .venv/bin/python3 scripts/own_suppression_oracle.py
outputs: .cache/ownsup_oracle_report.txt
"""
import json
import os
import pickle
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

import numpy as np

from polarity_census import classify_row, clause_of
from args_census import sentence_bounds, sentence_spans, sent_of_char, find_numeral_sentence

WILD_JSONL = ".cache/wild_admitted_holdout.jsonl"
DUMPS = {"PMS8_241": ".cache/rawslots_wild_PMS8_241.pkl",
         "PC_241": ".cache/ownsup_rawslots_wild_PC_241.pkl"}
OUT = ".cache/ownsup_oracle_report.txt"

LOG = []


def P(s=""):
    print(s, flush=True)
    LOG.append(s)


def load_dump(path):
    D = pickle.load(open(path, "rb"))
    return {int(d["i"]): d for d in D}


def rank_of(logits, idx):
    """rank 1 = argmax (the highest). ties broken toward a better (lower) rank for idx, matching
    an optimistic read of the model's own decode (np.argsort already breaks ties by index, not
    relevant here since logits are float and ties are measure-zero in practice)."""
    return int((logits > logits[idx]).sum()) + 1


def decode_args(args_logit, dup_logit):
    """THE SAME decode sm_autopsy_selfmatch_census.py / head_fields use: dup>0 -> a single
    repeated index (argmax); else the top-2 raw-logit indices (the 2-hot BCE decode)."""
    if dup_logit is not None and dup_logit > 0:
        return {int(np.argmax(args_logit))}
    return set(np.argsort(-args_logit)[:2].tolist())


def main():
    rows = [json.loads(l) for l in open(WILD_JSONL)]
    P("THE OWN-SUPPRESSION ORACLE -- PMS8_241 vs PC_241, wild holdout, zero training")
    P(f"wild fixture: {WILD_JSONL} ({len(rows)} rows)")

    # pre-build sentence structure per row (task 4's "same sentence" read)
    bounds_cache = {}
    spans_cache = {}
    for i, r in enumerate(rows):
        b = sentence_bounds(r["text"])
        bounds_cache[i] = b
        spans_cache[i] = sentence_spans(r["text"], b)

    per_body = {}
    for tag, path in DUMPS.items():
        if not os.path.exists(path):
            P(f"\n[SKIP] {tag}: {path} not found")
            continue
        d = load_dump(path)
        P(f"\n{'='*86}\nBODY {tag}  (dump: {path}, {len(d)} rows)\n{'='*86}")

        # ---- per-slot collection ----
        fwd = {"res_ok": 0, "n": 0}
        inv = {"res_ok": 0, "n": 0}
        oracle_hits = 0
        oracle_n = 0
        fwd_own_hits = 0
        fwd_own_n = 0
        rank_inv = []
        rank_fwd = []
        oracle_argmax_class = {"B_right": 0, "other_arg_C": 0, "given_var": 0, "derived_var": 0}
        margins_wrong = []          # (margin, B_is_given, B_same_sentence)
        # args oracle, inverse slots only
        args_inv_n = 0
        args_inv_ok = 0
        args_inv_own_missing = 0
        args_inv_C_wrong_own_found = 0
        args_inv_both_wrong = 0
        args_fwd_n = 0
        args_fwd_ok = 0

        for i, r in enumerate(rows):
            if i not in d:
                continue
            factors = r["factors"]
            cls = classify_row(factors)
            dd = d[i]
            bounds = bounds_cache[i]
            spans = spans_cache[i]
            for k, f in enumerate(factors):
                c = cls[k]
                if c not in ("fwd", "inv"):
                    continue
                res_logit = dd["res"][k].astype(np.float64)
                gres = int(f.get("result"))
                pred = int(np.argmax(res_logit))

                # ---------- (1) baseline ----------
                if c == "fwd":
                    fwd["n"] += 1
                    fwd["res_ok"] += int(pred == gres)
                    fwd_own_n += 1
                    fwd_own_hits += int(k == gres)   # "always predict own" sanity check
                    rank_fwd.append(rank_of(res_logit, gres))
                else:
                    inv["n"] += 1
                    inv["res_ok"] += int(pred == gres)

                # ---------- gold arg structure for inverse slots ----------
                gargs = list(f.get("args", []))
                gargs_set = set(gargs)
                if c == "inv":
                    # structural fact: own (k) should be among the gold args
                    others = [a for a in gargs if a != k]
                    C = others[0] if others else None   # None only if args==[k,k] (degenerate)

                    # ---------- (2) THE ORACLE: clamp own, re-argmax ----------
                    clamped = res_logit.copy()
                    clamped[k] = -np.inf
                    oracle_pred = int(np.argmax(clamped))
                    oracle_n += 1
                    hit = (oracle_pred == gres)
                    oracle_hits += int(hit)
                    rank_inv.append(rank_of(res_logit, gres))

                    if not hit:
                        if oracle_pred == C:
                            oracle_argmax_class["other_arg_C"] += 1
                        else:
                            vtype = factors[oracle_pred]["ftype"] if 0 <= oracle_pred < len(factors) else None
                            if vtype == "given":
                                oracle_argmax_class["given_var"] += 1
                            else:
                                oracle_argmax_class["derived_var"] += 1

                        # ---------- (4) margin + correlates ----------
                        margin = float(clamped[oracle_pred] - clamped[gres])
                        b_is_given = (0 <= gres < len(factors)) and (factors[gres]["ftype"] == "given")
                        # "same sentence as the relation's own clause"
                        a0, b0, _src = clause_of(r, k, f, bounds, spans)
                        same_sent = None
                        if a0 is not None:
                            si_rel = sent_of_char(bounds, a0)
                            sol = r.get("solution") or []
                            bval = sol[gres] if 0 <= gres < len(sol) else None
                            si_b = find_numeral_sentence(r["text"], bounds, bval) if bval is not None else None
                            if si_b is not None:
                                same_sent = (si_rel == si_b)
                        margins_wrong.append((margin, b_is_given, same_sent))
                    else:
                        oracle_argmax_class["B_right"] += 1

                    # ---------- (5) args oracle on inverse slots ----------
                    args_logit = dd["args"][k].astype(np.float64)
                    dup_logit = float(dd["dup"][k]) if "dup" in dd else None
                    decoded = decode_args(args_logit, dup_logit)
                    args_inv_n += 1
                    if decoded == gargs_set:
                        args_inv_ok += 1
                    else:
                        own_found = k in decoded
                        C_found = (C in decoded) if C is not None else None
                        if not own_found and (C_found is False or C_found is None):
                            args_inv_both_wrong += 1
                        elif not own_found:
                            args_inv_own_missing += 1
                        else:   # own found, C not found (since not ok and own found)
                            args_inv_C_wrong_own_found += 1
                else:
                    # forward args, for the baseline comparison in the table only
                    args_logit = dd["args"][k].astype(np.float64)
                    dup_logit = float(dd["dup"][k]) if "dup" in dd else None
                    decoded = decode_args(args_logit, dup_logit)
                    args_fwd_n += 1
                    args_fwd_ok += int(decoded == gargs_set)

        per_body[tag] = dict(fwd=fwd, inv=inv, oracle_hits=oracle_hits, oracle_n=oracle_n,
                              fwd_own_hits=fwd_own_hits, fwd_own_n=fwd_own_n,
                              rank_inv=np.array(rank_inv), rank_fwd=np.array(rank_fwd),
                              oracle_argmax_class=oracle_argmax_class, margins_wrong=margins_wrong,
                              args_inv_n=args_inv_n, args_inv_ok=args_inv_ok,
                              args_inv_own_missing=args_inv_own_missing,
                              args_inv_C_wrong_own_found=args_inv_C_wrong_own_found,
                              args_inv_both_wrong=args_inv_both_wrong,
                              args_fwd_n=args_fwd_n, args_fwd_ok=args_fwd_ok)

        # ---------------- report ----------------
        P(f"\n(1) BASELINE res accuracy by form:")
        P(f"    fwd n={fwd['n']:4d}  res={fwd['res_ok']/max(fwd['n'],1):.4f}")
        P(f"    inv n={inv['n']:4d}  res={inv['res_ok']/max(inv['n'],1):.4f}")

        P(f"\n(2) THE ORACLE (inverse slots, own-index res logit clamped to -inf, re-argmax):")
        P(f"    inverse res accuracy under the oracle = {oracle_hits}/{oracle_n} = "
          f"{oracle_hits/max(oracle_n,1):.4f}   (baseline {inv['res_ok']/max(inv['n'],1):.4f}; "
          f"THE CEILING of any direction-aware fix)")
        P(f"    forward sanity check ('always predict own' == gold res): "
          f"{fwd_own_hits}/{fwd_own_n} = {fwd_own_hits/max(fwd_own_n,1):.4f}  (must be ~1.0 by "
          f"the positional law's own definition of 'forward')")

        P(f"\n(3) RANK HISTOGRAM of gold res B among the 24 raw res logits (rank 1 = argmax):")
        for label, ranks in (("inverse", per_body[tag]["rank_inv"]), ("forward", per_body[tag]["rank_fwd"])):
            n = len(ranks)
            if n == 0:
                P(f"    {label}: no slots")
                continue
            r1 = int((ranks == 1).sum()); r2 = int((ranks == 2).sum()); r3 = int((ranks == 3).sum())
            rgt = int((ranks > 3).sum())
            P(f"    {label:8s} n={n:4d}  rank1={r1/n:.3f}  rank2={r2/n:.3f}  rank3={r3/n:.3f}  "
              f"rank>3={rgt/n:.3f}   (counts: {r1}/{r2}/{r3}/{rgt})")

        oac = per_body[tag]["oracle_argmax_class"]
        n_or_wrong = sum(oac.values()) - oac["B_right"]
        P(f"    on inverse slots WRONG under the oracle (n={n_or_wrong}), the oracle's new argmax is:")
        for key, label in (("other_arg_C", "the other gold arg C"), ("given_var", "some OTHER given variable"),
                            ("derived_var", "some OTHER derived (rel) variable")):
            v = oac[key]
            P(f"      {label:32s} {v:4d}  ({v/max(n_or_wrong,1):.3f})")

        P(f"\n(4) MARGIN under the oracle, on inverse slots still wrong (winner_logit - B_logit):")
        margins = per_body[tag]["margins_wrong"]
        if margins:
            marr = np.array([m[0] for m in margins])
            P(f"    n={len(marr)}  mean={marr.mean():.3f}  median={np.median(marr):.3f}  "
              f"p25={np.percentile(marr,25):.3f}  p75={np.percentile(marr,75):.3f}  max={marr.max():.3f}")
            given_mask = np.array([bool(m[1]) for m in margins])
            P(f"    B is a GIVEN variable: {given_mask.sum()}/{len(margins)} ({given_mask.mean():.3f}) "
              f"of the still-wrong rows; mean margin given={marr[given_mask].mean() if given_mask.any() else float('nan'):.3f} "
              f"vs derived={marr[~given_mask].mean() if (~given_mask).any() else float('nan'):.3f}")
            same_vals = [m[2] for m in margins if m[2] is not None]
            if same_vals:
                same_arr = np.array(same_vals, dtype=bool)
                marr_known = np.array([m[0] for m in margins if m[2] is not None])
                P(f"    B resolvable to a sentence: {len(same_vals)}/{len(margins)}; same-sentence-as-clause "
                  f"share={same_arr.mean():.3f}; mean margin same-sentence={marr_known[same_arr].mean() if same_arr.any() else float('nan'):.3f} "
                  f"vs different-sentence={marr_known[~same_arr].mean() if (~same_arr).any() else float('nan'):.3f}")
            else:
                P("    B resolvable to a sentence: 0 (clause_of/find_numeral_sentence could not place B)")
        else:
            P("    no still-wrong inverse slots under the oracle")

        P(f"\n(5) ARGS ORACLE (inverse slots): is C (the non-own gold arg) found by the args head's own decode?")
        ain = per_body[tag]["args_inv_n"]
        P(f"    inverse args accuracy (exact set match) = {per_body[tag]['args_inv_ok']}/{ain} = "
          f"{per_body[tag]['args_inv_ok']/max(ain,1):.4f}   (forward: "
          f"{per_body[tag]['args_fwd_ok']}/{per_body[tag]['args_fwd_n']} = "
          f"{per_body[tag]['args_fwd_ok']/max(per_body[tag]['args_fwd_n'],1):.4f})")
        n_args_wrong = ain - per_body[tag]["args_inv_ok"]
        P(f"    decomposition of the {n_args_wrong} inverse args errors:")
        P(f"      own(k) missing from decode (both effectively wrong): {per_body[tag]['args_inv_own_missing']} "
          f"({per_body[tag]['args_inv_own_missing']/max(n_args_wrong,1):.3f})")
        P(f"      own found, C wrong:                                 {per_body[tag]['args_inv_C_wrong_own_found']} "
          f"({per_body[tag]['args_inv_C_wrong_own_found']/max(n_args_wrong,1):.3f})")
        P(f"      both wrong / degenerate (gold args == {{k,k}}):           {per_body[tag]['args_inv_both_wrong']} "
          f"({per_body[tag]['args_inv_both_wrong']/max(n_args_wrong,1):.3f})")

    # ---------------- cross-body verdict ----------------
    P(f"\n{'='*86}\nTHE VERDICT\n{'='*86}")
    for tag in DUMPS:
        if tag not in per_body:
            continue
        d = per_body[tag]
        ceiling = d["oracle_hits"] / max(d["oracle_n"], 1)
        rank2_share = float((d["rank_inv"] == 2).sum()) / max(len(d["rank_inv"]), 1)
        P(f"  {tag}: baseline inv res={d['inv']['res_ok']/max(d['inv']['n'],1):.4f}  "
          f"oracle ceiling={ceiling:.4f}  rank-2 share={rank2_share:.4f}")
    P("")
    any_ceiling_high = any(
        (per_body[t]["oracle_hits"] / max(per_body[t]["oracle_n"], 1)) >= 0.6 for t in per_body)
    any_ceiling_low = any(
        (per_body[t]["oracle_hits"] / max(per_body[t]["oracle_n"], 1)) < 0.4 for t in per_body)
    P("  READING: the oracle's ceiling (inverse res accuracy with the own index removed and the body's")
    P("  OWN remaining logits re-argmaxed -- the best any direction-aware fix on this body's existing")
    P("  res geometry could achieve) decides between the two framings the ledger posed. If B sits at")
    P("  rank 2 on most inverse rows (ceiling >= 0.6), the direction IS the lever: the body already")
    P("  ranks B highly and a trained 'not own' bit would mostly suffice. If B's rank is scattered")
    P("  across the oracle's wrong-argmax classes (ceiling < 0.4), BINDING B is the lever -- removing")
    P("  the own-index default does not reliably surface B, and the direction line closes into the")
    P("  args wall (section 5's decomposition says whether C itself is even found).")
    P(f"  -> ceiling >= 0.6 on any body: {any_ceiling_high}; ceiling < 0.4 on any body: {any_ceiling_low}")

    os.makedirs(".cache", exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(LOG) + "\n")
    print(f"\n[own-suppression-oracle] wrote {OUT}")


if __name__ == "__main__":
    main()
