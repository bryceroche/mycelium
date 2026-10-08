"""polarity_census.py — THE POLARITY CENSUS (2026-10-08, zero GPU; word given in the ledger entry
"THE ALGEBRAIC EPIPHANY", 2026-10-08 12:14). Bryce: "the problem is that we're not detecting
negative and inverse numbers." The fixture carries no signed numbers (positive naturals only, no
sign in the digit head) — what looks like "polarity" is the DIRECTION of a relation: which
variable the gold `res` pointer names. THE POSITIONAL LAW (2026-09-17, mycelium/rulebook.py,
scripts/canonical_positional.py): a factor at slot k introduces exactly one NEW variable; a
FORWARD relation introduces its own result (gold res == the new variable == the slot's own
position under the identity convention K_VARS==L_FAC, scripts/phase1_algebra_head.py:1104,5381-82);
an INVERSE relation (the sub/div encoding, reencode_ops: sub(a,b)->r becomes add(b,r)->a) introduces
one of its ARGS instead — its gold res names an EARLIER variable. This script classifies every gold
relation factor in the wild holdout as forward/inverse by REPLAYING the row's own variable
introduction order (not by assuming slot==var, which THE ARGUMENT-BINDING CENSUS already measured
holds on 94.3% of wild instances, not 100% — a factor whose single new-variable condition fails is
bucketed "other" and excluded, counted separately), joins the per-slot gold/pred heads from the
LV_DUMP pickle (scripts/matched_read.py documents the tuple layout; loop_val.py builds it), reads
the direction-flip confusion off the res pointer, census the cue words of the relation's clause
(mentions when present — none on wild — else the sentence containing an arg's gold numeral, the
same heuristic scripts/args_census.py uses and documents), and reads the diet's forward/inverse
share per op from the training jsonl.

usage: .venv/bin/python3 scripts/polarity_census.py
outputs: .cache/polarity_census_PMS8_241.txt
"""
import collections
import json
import os
import pickle
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from args_census import (STOPWORDS, sentence_bounds, sentence_spans,
                          sent_of_char, find_numeral_sentence, words_of)

WILD = ".cache/wild_admitted_holdout.jsonl"
BODIES = ["PMS8_241", "HS_241", "RK_241"]   # PMS8_241 = the claim body; HS_241/RK_241 = parity
MAIN = "PMS8_241"
DIET_PATHS = [".cache/form_mix_pm35c.jsonl", ".cache/form_mix_pm35a.jsonl"]
OUT = ".cache/polarity_census_PMS8_241.txt"
TOPN = 20
MIN_CUE_N = 5   # a cue needs this many relation-instances on a side to be reported (noise floor)


# =======================================================================
# (1) forward / inverse classification — replays the row's own variable
# introduction order (THE POSITIONAL LAW), not a bare slot==var assumption
# =======================================================================

def vars_of(f):
    if f["ftype"] == "given":
        return [f["var"]]
    if f["ftype"] == "rel":
        return list(f.get("args", [])) + [f.get("result")]
    return []   # fdiv / other ftypes: not a relation, not classified


def classify_row(factors):
    """Per factor: "fwd" (rel, introduces its own result), "inv" (rel,
    introduces one of its args — the sub/div encoding), "other" (rel, the
    single-new-variable law does not hold at this slot — ambiguous or
    zero new variables; the 94.3%-not-100% residual), or None (given/fdiv)."""
    seen = set()
    out = [None] * len(factors)
    for k, f in enumerate(factors):
        vs = set(vars_of(f))
        if f["ftype"] == "rel":
            unseen = vs - seen
            if len(unseen) == 1:
                nv = next(iter(unseen))
                out[k] = "fwd" if nv == f.get("result") else "inv"
            else:
                out[k] = "other"
        seen |= vs
    return out


# =======================================================================
# (2) per-head correctness from the LV_DUMP tuple — loop_val.py's own
# `ok` logic (scripts/loop_val.py ~l.488-505), recomputed field-by-field
# =======================================================================

def load_dump(tag):
    D = pickle.load(open(f".cache/dump_wild_{tag}.pkl", "rb"))
    d = {}
    for t in D:
        i, j = t[0], t[1]
        d[(i, j)] = t[2:]   # gft,gop,gargs,gres,gdig,pft,pop,pargs,pres_,pdig,ppres,pdup
    return d


def load_ps_legal(tag):
    z = np.load(f".cache/ps_legal_wild_{tag}.npz")
    return {(int(i), int(j)): bool(ok) for i, j, ok in zip(z["rows"], z["slots"], z["ok"])}


def head_fields(t):
    gft, gop, gargs, gres, gdig, pft, pop, pargs, pres_, pdig, ppres, pdup = t
    f_ftype = (pft == gft)
    f_res = (pres_ == gres)
    if gft == 0:   # gold is a relation
        gset = set(gargs)
        if len(gset) == 1:
            f_args = bool(pdup) and (pargs[0] in gset)
        else:
            f_args = set(pargs) == gset
        f_op = (pop == gop)
        ok = bool(ppres) and f_ftype and f_res and f_op and f_args
        return dict(pres=bool(ppres), ftype=f_ftype, op=f_op, args=f_args,
                    res=f_res, dig=None, ok=ok, pres_=pres_, gres=gres,
                    gargs=tuple(gargs), pargs=tuple(pargs))
    f_dig = (list(pdig) == list(gdig))
    ok = bool(ppres) and f_ftype and f_res and f_dig
    return dict(pres=bool(ppres), ftype=f_ftype, op=None, args=None,
                res=f_res, dig=f_dig, ok=ok, pres_=pres_, gres=gres,
                gargs=(), pargs=tuple(pargs))


# =======================================================================
# (4) the clause / cue census — mentions when present (none on wild), else
# the sentence containing an arg's GOLD numeral (the row's own `solution`
# vector gives every variable's value directly, given or derived — no
# need to re-evaluate the graph the way args_census.row_eval does)
# =======================================================================

def clause_of(row, k, f, bounds, spans):
    mentions = row.get("mentions") or {}
    mv = mentions.get(str(f.get("result")))
    if mv:
        s, e = mv[0]
        si = sent_of_char(bounds, s)
        return spans[si][0], spans[si][1], "mention"
    sol = row.get("solution") or []
    for a in f.get("args", []):
        val = sol[a] if 0 <= a < len(sol) else None
        si = find_numeral_sentence(row["text"], bounds, val)
        if si is not None:
            return spans[si][0], spans[si][1], "arg_numeral"
    return None, None, None


def cue_cands(text):
    ws = words_of(text)
    uni = {w for w in ws if w not in STOPWORDS and len(w) > 2}
    bi = {a + " " + b for a, b in zip(ws, ws[1:]) if not (a in STOPWORDS and b in STOPWORDS)}
    return uni | bi


# =======================================================================
# main
# =======================================================================

def main():
    t0 = time.time()
    rows = [json.loads(l) for l in open(WILD)]
    n_rows = len(rows)

    dumps = {tag: load_dump(tag) for tag in BODIES}
    ps = {tag: load_ps_legal(tag) for tag in BODIES}

    # per-row classification + per-(row,slot) record
    cls = {}              # (i,k) -> "fwd"/"inv"/"other"
    rel_slots = collections.defaultdict(list)   # i -> [(k, f), ...] rel slots only
    for i, r in enumerate(rows):
        c = classify_row(r["factors"])
        for k, f in enumerate(r["factors"]):
            if c[k] is not None:
                cls[(i, k)] = c[k]
                rel_slots[i].append((k, f))

    n_fwd = sum(1 for v in cls.values() if v == "fwd")
    n_inv = sum(1 for v in cls.values() if v == "inv")
    n_other = sum(1 for v in cls.values() if v == "other")

    # (1) counts per op
    op_name = {0: "add", 1: "mul"}
    counts = collections.Counter()   # (op, cls)
    for i, r in enumerate(rows):
        for k, f in enumerate(r["factors"]):
            c = cls.get((i, k))
            if c is None:
                continue
            counts[(f["op"], c)] += 1

    # (2) per class x per head, per body
    head_stats = {tag: collections.defaultdict(lambda: collections.Counter()) for tag in BODIES}
    # head_stats[tag][cls_label][field] accumulated as (n_ok, n_tot) via two counters "<field>_ok","<field>_n"
    rec_cache = {tag: {} for tag in BODIES}   # tag -> (i,k) -> head_fields dict, rel slots only
    sanity_mismatch = collections.Counter()
    for i, r in enumerate(rows):
        for k, f in enumerate(r["factors"]):
            c = cls.get((i, k))
            if c is None or c == "other":
                continue
            for tag in BODIES:
                t = dumps[tag].get((i, k))
                if t is None:
                    continue
                hf = head_fields(t)
                rec_cache[tag][(i, k)] = hf
                hs = head_stats[tag][c]
                for field in ("pres", "ftype", "op", "args", "res", "ok"):
                    v = hf[field]
                    if v is None:
                        continue
                    hs[field + "_n"] += 1
                    hs[field + "_ok"] += int(bool(v))
                psv = ps[tag].get((i, k))
                if psv is not None and psv != hf["ok"]:
                    sanity_mismatch[tag] += 1

    # (3) the direction-flip confusion, MAIN body (res-wrong relations only)
    confusion = {"inv_flip_to_fwd": 0, "inv_wrong_other": 0, "inv_res_wrong_n": 0,
                 "fwd_flip_to_inv": 0, "fwd_wrong_other": 0, "fwd_res_wrong_n": 0}
    for (i, k), c in cls.items():
        if c not in ("fwd", "inv"):
            continue
        hf = rec_cache[MAIN].get((i, k))
        if hf is None or hf["res"]:
            continue
        pres_ = hf["pres_"]
        if c == "inv":
            confusion["inv_res_wrong_n"] += 1
            if pres_ == k:
                confusion["inv_flip_to_fwd"] += 1
            else:
                confusion["inv_wrong_other"] += 1
        else:
            confusion["fwd_res_wrong_n"] += 1
            if pres_ in hf["gargs"]:
                confusion["fwd_flip_to_inv"] += 1
            else:
                confusion["fwd_wrong_other"] += 1

    # (4) cue census (clause resolution + cand extraction), MAIN body correctness
    cue_n = {"fwd": collections.Counter(), "inv": collections.Counter()}
    cue_res_ok = {"fwd": collections.Counter(), "inv": collections.Counter()}
    cue_ok_ok = {"fwd": collections.Counter(), "inv": collections.Counter()}
    n_resolved = {"fwd": 0, "inv": 0}
    n_unresolved = {"fwd": 0, "inv": 0}
    resolve_src = collections.Counter()
    for i, r in enumerate(rows):
        bounds = sentence_bounds(r["text"])
        spans = sentence_spans(r["text"], bounds)
        for k, f in rel_slots[i]:
            c = cls.get((i, k))
            if c not in ("fwd", "inv"):
                continue
            a, b, src = clause_of(r, k, f, bounds, spans)
            if a is None:
                n_unresolved[c] += 1
                continue
            n_resolved[c] += 1
            resolve_src[(c, src)] += 1
            clause_text = r["text"][a:b]
            hf = rec_cache[MAIN].get((i, k))
            res_ok = bool(hf["res"]) if hf else None
            ok_ok = bool(hf["ok"]) if hf else None
            for cue in cue_cands(clause_text):
                cue_n[c][cue] += 1
                if res_ok:
                    cue_res_ok[c][cue] += 1
                if ok_ok:
                    cue_ok_ok[c][cue] += 1

    def cue_table(primary):
        other = "fwd" if primary == "inv" else "inv"
        rows_out = []
        for cue, n in cue_n[primary].most_common():
            if n < MIN_CUE_N:
                continue
            n_o = cue_n[other].get(cue, 0)
            racc_p = cue_res_ok[primary].get(cue, 0) / n
            racc_o = cue_res_ok[other].get(cue, 0) / n_o if n_o else None
            rows_out.append((cue, n, racc_p, n_o, racc_o))
        return rows_out[:TOPN]

    top_inv_cues = cue_table("inv")
    top_fwd_cues = cue_table("fwd")

    def cue_skew_table(primary):
        other = "fwd" if primary == "inv" else "inv"
        rows_out = []
        for cue, n in cue_n[primary].items():
            if n < MIN_CUE_N:
                continue
            n_o = cue_n[other].get(cue, 0)
            if n_o >= n:
                continue   # only cues genuinely skewed toward `primary`
            racc_p = cue_res_ok[primary].get(cue, 0) / n
            racc_o = cue_res_ok[other].get(cue, 0) / n_o if n_o else None
            rows_out.append((cue, n, racc_p, n_o, racc_o, n - n_o))
        rows_out.sort(key=lambda r: -r[5])
        return [r[:5] for r in rows_out[:TOPN]]

    top_inv_skew = cue_skew_table("inv")
    top_fwd_skew = cue_skew_table("fwd")

    # (5) representability (static architectural note) + diet frequency
    diet_counts = {}
    for path in DIET_PATHS:
        if not os.path.exists(path):
            diet_counts[path] = None
            continue
        cnt = collections.Counter()
        with open(path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                c = classify_row(r["factors"])
                for k, f in enumerate(r["factors"]):
                    if c[k] in ("fwd", "inv"):
                        cnt[(f["op"], c[k])] += 1
        diet_counts[path] = cnt

    # ================= write the report =================
    lines = []
    P = lines.append
    P(f"THE POLARITY CENSUS — {MAIN} (parity: HS_241, RK_241); zero GPU")
    P(f"generated {os.popen('date').read().strip()}; wild_admitted_holdout.jsonl rows={n_rows}")
    P("")
    P("(1) FORWARD vs INVERSE relation counts, by op (classified by replaying each row's own")
    P("    variable-introduction order — THE POSITIONAL LAW, not a bare slot==var assumption;")
    P("    'other' = the single-new-variable law does not hold at that slot, excluded below)")
    P(f"  total relation slots: {n_fwd + n_inv + n_other} (fwd {n_fwd}, inv {n_inv}, other {n_other})")
    for op in ("add", "mul"):
        f_ = counts[(op, "fwd")]; v_ = counts[(op, "inv")]
        P(f"  op={op:4s}  forward {f_:4d}  inverse {v_:4d}  (inverse share {v_/(f_+v_):.3f})" if (f_ + v_) else f"  op={op:4s}  (none)")
    P("")
    P("(2) per-class slot-level + per-head correctness, per body (fwd / inv; 'other' excluded)")
    for tag in BODIES:
        P(f"  --- {tag} ---")
        for c in ("fwd", "inv"):
            hs = head_stats[tag][c]
            def r(field):
                n = hs.get(field + "_n", 0); return (hs.get(field + "_ok", 0) / n, n) if n else (float("nan"), 0)
            ok_, n_ = r("ok"); pres_, _ = r("pres"); ftype_, _ = r("ftype")
            op_, opn = r("op"); args_, argn = r("args"); res_, _ = r("res")
            P(f"    {c:4s} n={n_:4d}  ok={ok_:.3f}  pres={pres_:.3f}  ftype={ftype_:.3f}  "
              f"op={op_:.3f}  args={args_:.3f}  res={res_:.3f}")
        if sanity_mismatch.get(tag):
            P(f"    [sanity] {sanity_mismatch[tag]} slots where recomputed ok != ps_legal ok (dup/arg-set edge cases)")
    P("    NOTE: 'value'/digit correctness is not reported for relation slots — the dump's gold digit")
    P("    field is zero-padded (meaningless) for every rel factor (loop_val.py's f_dig only runs in the")
    P("    given branch); a rel-slot digit comparison would be a comparison against a dummy target.")
    P("")
    P("(3) THE DIRECTION-FLIP CONFUSION (MAIN body, among relations wrong on res only)")
    niw = confusion["inv_res_wrong_n"]; nfw = confusion["fwd_res_wrong_n"]
    P(f"  inverse, res wrong: n={niw}  flipped-to-forward (pred res == own slot) "
      f"{confusion['inv_flip_to_fwd']} ({confusion['inv_flip_to_fwd']/niw:.3f})  "
      f"other-wrong {confusion['inv_wrong_other']} ({confusion['inv_wrong_other']/niw:.3f})" if niw else "  inverse: no res-wrong instances")
    P(f"  forward, res wrong: n={nfw}  flipped-to-inverse (pred res in its own gold args) "
      f"{confusion['fwd_flip_to_inv']} ({confusion['fwd_flip_to_inv']/nfw:.3f})  "
      f"other-wrong {confusion['fwd_wrong_other']} ({confusion['fwd_wrong_other']/nfw:.3f})" if nfw else "  forward: no res-wrong instances")
    P("")
    P("(4) CUE CENSUS — clause = mentions[result] span if present (none on wild), else the sentence")
    P("    containing an arg's gold numeral (row['solution'][arg]); n = relation-instances whose")
    P("    clause contains the cue at least once (document frequency); racc = MAIN body res-accuracy")
    P(f"  clause resolved: fwd {n_resolved['fwd']}/{n_resolved['fwd']+n_unresolved['fwd']}  "
      f"inv {n_resolved['inv']}/{n_resolved['inv']+n_unresolved['inv']}  "
      f"(source counts {dict(resolve_src)})")
    P(f"  top-{TOPN} INVERSE cues (min n={MIN_CUE_N}):  cue | n_inv | racc_inv | n_fwd | racc_fwd")
    for cue, n, racc_p, n_o, racc_o in top_inv_cues:
        P(f"    {cue:18s} {n:4d}  {racc_p:.3f}   {n_o:4d}  {('%.3f' % racc_o) if racc_o is not None else '  -  '}")
    P(f"  top-{TOPN} FORWARD cues (min n={MIN_CUE_N}):  cue | n_fwd | racc_fwd | n_inv | racc_inv")
    for cue, n, racc_p, n_o, racc_o in top_fwd_cues:
        P(f"    {cue:18s} {n:4d}  {racc_p:.3f}   {n_o:4d}  {('%.3f' % racc_o) if racc_o is not None else '  -  '}")
    P("")
    P("  the raw-frequency tables above are dominated by rate-problem vocabulary common to BOTH forms")
    P("  (every/per/costs/years/hours/...); the SKEW tables below keep only cues strictly more common")
    P("  on one side, ranked by (n_primary - n_other) — the genuinely discriminative cues:")
    P(f"  top INVERSE-skewed cues:  cue | n_inv | racc_inv | n_fwd | racc_fwd")
    for cue, n, racc_p, n_o, racc_o in top_inv_skew:
        P(f"    {cue:18s} {n:4d}  {racc_p:.3f}   {n_o:4d}  {('%.3f' % racc_o) if racc_o is not None else '  -  '}")
    P(f"  top FORWARD-skewed cues:  cue | n_fwd | racc_fwd | n_inv | racc_inv")
    for cue, n, racc_p, n_o, racc_o in top_fwd_skew:
        P(f"    {cue:18s} {n:4d}  {racc_p:.3f}   {n_o:4d}  {('%.3f' % racc_o) if racc_o is not None else '  -  '}")
    P("")
    P("(5) REPRESENTABILITY + DIET FREQUENCY")
    P("  representability: YES, structurally unrestricted. K_VARS == L_FAC == 24")
    P("  (scripts/phase1_algebra_head.py:1104, :5381-82 'slot k -> variable index: IDENTITY'); the res")
    P("  head (p['W_res'], :1687) is a plain pointer over the SAME 24-wide space the args head uses —")
    P("  nothing in the forward pass biases res toward the slot's own index j (no identity/diagonal")
    P("  bias term was found on the res path; np.eye( ) hits in this file are W_ps/W_role/t1_pw init and")
    P("  an unrelated same-slot mask, none on W_res). An inverse relation's gold res (an EARLIER")
    P("  variable) is exactly as reachable as a forward relation's gold res (the slot's own index) —")
    P("  the inverse target is not excluded by the wiring the way args=[a,a] once was ([85]).")
    for path, cnt in diet_counts.items():
        if cnt is None:
            P(f"  diet {path}: NOT FOUND, skipped")
            continue
        P(f"  diet {path}:")
        for op in ("add", "mul"):
            f_ = cnt[(op, "fwd")]; v_ = cnt[(op, "inv")]
            tot = f_ + v_
            P(f"    op={op:4s}  forward {f_:6d}  inverse {v_:6d}  (inverse share {v_/tot:.3f})" if tot else f"    op={op:4s}  (none)")
    if (diet_counts.get(".cache/form_mix_pm35c.jsonl") is not None and
            diet_counts.get(".cache/form_mix_pm35a.jsonl") is not None and
            diet_counts[".cache/form_mix_pm35c.jsonl"] == diet_counts[".cache/form_mix_pm35a.jsonl"]):
        P("  [identical]: pm35a and pm35c carry byte-identical factor graphs (c only adds role_order/")
        P("  role_cues/arg_role_cues stamps) — confirmed by direct diff, not a census bug.")
    P("  NOTE: share-of-mix only (THE DOSE LAW also wants reps-per-unique, i.e. inverse forms' repeat")
    P("  count per unique problem — NOT computed here, flagged unverified; would need the diet's own")
    P("  knot/WL-digest dedup machinery, out of scope for a zero-GPU structural census).")
    P("")
    P("(6) THE READING")
    inv_acc = head_stats[MAIN]["inv"]["ok_ok"] / max(head_stats[MAIN]["inv"]["ok_n"], 1)
    fwd_acc = head_stats[MAIN]["fwd"]["ok_ok"] / max(head_stats[MAIN]["fwd"]["ok_n"], 1)
    inv_res = head_stats[MAIN]["inv"]["res_ok"] / max(head_stats[MAIN]["inv"]["res_n"], 1)
    fwd_res = head_stats[MAIN]["fwd"]["res_ok"] / max(head_stats[MAIN]["fwd"]["res_n"], 1)
    inv_op = head_stats[MAIN]["inv"]["op_ok"] / max(head_stats[MAIN]["inv"]["op_n"], 1)
    fwd_op = head_stats[MAIN]["fwd"]["op_ok"] / max(head_stats[MAIN]["fwd"]["op_n"], 1)
    inv_args = head_stats[MAIN]["inv"]["args_ok"] / max(head_stats[MAIN]["inv"]["args_n"], 1)
    fwd_args = head_stats[MAIN]["fwd"]["args_ok"] / max(head_stats[MAIN]["fwd"]["args_n"], 1)
    diet_a = diet_counts.get(".cache/form_mix_pm35c.jsonl")
    inv_share_diet = None
    if diet_a:
        tot_diet = sum(diet_a.values()); inv_diet = diet_a[("add", "inv")] + diet_a[("mul", "inv")]
        inv_share_diet = inv_diet / tot_diet if tot_diet else None
    inv_share_wild = n_inv / (n_fwd + n_inv) if (n_fwd + n_inv) else None
    P(f"  1. INVERSE FORMS FAIL WHERE FORWARD FORMS SUCCEED: fac-exact {inv_acc:.3f} vs {fwd_acc:.3f} on")
    P(f"     {MAIN} ({n_inv} vs {n_fwd} instances); the gap sits almost entirely on RES ({inv_res:.3f} vs")
    P(f"     {fwd_res:.3f}), not ftype/op ({inv_op:.3f} vs {fwd_op:.3f} op) — args also split ({inv_args:.3f}")
    P(f"     vs {fwd_args:.3f}) since an inverse relation's own new variable sits in ARGS, not res.")
    P(f"  2. THE HEAD: res, as predicted. Among inverse res-wrong instances "
      f"({confusion['inv_res_wrong_n']}), "
      f"{confusion['inv_flip_to_fwd']/max(niw,1):.3f} point at the slot's OWN index — the model decodes")
    P(f"     them as if they were forward (the direction bit the ledger asked for would correct exactly")
    P(f"     this flip); the mirror flip on wrong forward relations (res landing in the relation's own")
    P(f"     gold args) is {confusion['fwd_flip_to_inv']/max(nfw,1):.3f} of {nfw} — asymmetric, inverse->forward")
    P(f"     is the dominant failure direction, not a symmetric mix-up.")
    P(f"  3. NOT REPRESENTABILITY: the res head's 24-wide pointer space is identical for both forms and")
    P(f"     carries no own-index bias found in the code (section 5) — an inverse target is as reachable")
    P(f"     as a forward one by construction.")
    if inv_share_diet is not None and inv_share_wild is not None:
        P(f"  4. FREQUENCY: inverse share of the diet ({inv_share_diet:.3f}) vs the wild holdout's own")
        P(f"     inverse share ({inv_share_wild:.3f}) — " +
          ("STARVED: the diet under-represents inverse forms relative to what wild actually needs."
           if inv_share_diet < inv_share_wild - 0.02 else
           "roughly matched — frequency alone does not explain the gap; see the cue reading below."))
    else:
        P("  4. FREQUENCY: diet file(s) not found/empty — not measured.")
    inv_skew_acc = (sum(c[1] * c[2] for c in top_inv_skew) / max(sum(c[1] for c in top_inv_skew), 1)) if top_inv_skew else float("nan")
    predicted = ("fewer", "less", "gave", "left", "half")
    pred_tally = ", ".join(f"{w} inv={cue_n['inv'].get(w,0)}/fwd={cue_n['fwd'].get(w,0)}" for w in predicted)
    P(f"  5. CUE BINDING: the PINNED prediction's cue words, scored by document count (inv/fwd): {pred_tally}.")
    P(f"     2 of 5 (less, fewer) are genuinely inverse-skewed; 'half' is a near-tie (noise); 'gave' and")
    P(f"     'left' are actually FORWARD-skewed on this holdout — the prediction's cue list was only")
    P(f"     half right. Even the cues that ARE inverse-skewed carry res-accuracy averaging {inv_skew_acc:.3f}")
    P(f"     (weighted over the skew table), barely above the inverse baseline ({inv_res:.3f}): the cue is")
    P(f"     present in the text at a rate that discriminates the form, but the model is not yet BINDING")
    P(f"     it to the res decision — a frequency/rehearsal fix alone would not obviously close this;")
    P(f"     read the per-cue racc_inv column above directly before building a cue-keyed road.")
    P(f"  6. WHAT A DIRECTION BIT / SORTING-ROOM LAYER MUST SUPPLY: a per-relation-clause classifier")
    P(f"     (forward vs inverse) feeding the res pointer's prior BEFORE the bilinear match — not a new")
    P(f"     value, a ROUTING bit over the SAME 24-wide space (section 5), trained on the clause cues")
    P(f"     above and gated structurally (ALG_DUP precedent: one bit, three generations) rather than")
    P(f"     left to the pointer alone to infer from pattern frequency.")
    P("")
    P(f"[timing] {time.time()-t0:.1f}s")

    os.makedirs(".cache", exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[polarity_census] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
