"""twin_gap.py -- THE TWIN-GAP READ (2026-09-27, zero-GPU, zero-training).

Baseline for THE CARICATURE MARGIN (a hard-negative margin loss on the
pointer's twins): before building a margin loss we need the margin
DISTRIBUTION the raw args logits already produce, split by whether the
decision is right/wrong and by the mixed_bucket_census KEY-INH class
(SAME-NOUN / DIFFERENT-NOUN / MIXED-NO-KEY).

Reuses, never reimplements: scripts/mixed_bucket_census.py's
arg_closure/inherited_keys/classify_sets (the KEY-INH classification) and
scripts/lexical_identity_census.py's build_candidate_tables/same_sentence
(the same-sentence competitor definition). This script does NOT modify or
import-drive any training script; it only reads the banked raw per-slot
head dump .cache/rawslots_wild_PMS8_241.pkl and the wild holdout fixture.

THE POINTER, per _decode_slots / decode() in scripts/phase1_algebra_head.py
(ft==0, "rel"): args[j] is a 24-wide bilinear-pointer logit vector indexed
by VARIABLE id (not slot id) -- decode()'s own gset/vg["args"] convention
confirms this (gold factors' "args" list and "result"/"var" fields all
live in the 24-variable space; K_VARS == L_FAC == 24 numerically but they
are two different index spaces, and a slot's OWN variable is NOT its own
index in general -- e.g. row 0 slot 4 introduces variable 2, not 4 -- so
a mixed_bucket "competitor" slot c's competing pointer target is
own_var(factors[c]), read via mycelium's stamp_arg_mentions.own_var,
not the slot index c itself).

THE DECODE (what "correct" and the dup branch mean here): for ALG_DUP
slots (row["dup"][j] > 0, the family env's fallback -- no "dargs" bank in
the raw dump so decode() falls back to argmax(args[j]) exactly as its own
"dargs" absent branch does) the actual pick is the SINGLE argmax variable,
duplicated for both argument positions -- not a top-2 set. For a plain
relation slot the actual pick is the top-2 argsort of args[j]. "correct"
here means the gold argument variable v is in that decoded pick set --
this is the "matching the machine's actual pick" convention, a stricter,
dup-aware version of the plain "v in top-2" convention the other census
scripts use (which the LV_DUMP dump also always uses top-2, ignoring dup).

THE GAP is explicitly NOT the full decode threshold (which is "beat the
2nd-best of all 23 other variables", or, for a dup slot, "beat all 23
others outright"): it is the margin over the TWIN-SPECIFIC competitor set
mixed_bucket_census defines (same-sentence slots), i.e. how hard the local
selection problem specifically is, which is what a hard-negative margin
loss on the twins would move. Because that competitor set is a SUBSET of
all 23 other variables, GAP is an upper bound on the true full-decode
margin -- a decision can show GAP>0 here and still lose to some
non-competitor slot elsewhere; the correctness field is decoded exactly
(all 23 candidates), so the two views are reported side by side, never
conflated.

SCOPE: only gold ftype=="rel" slots (the args-logit top-2/dup bilinear
pointer this GAP construction was built for); pct/sel/mod/fdiv have
different pointer shapes and are out of scope for this read (reported as
excluded, not silently folded in).

Run: DEV=CPU + the family env (see docs/NEXT_SESSION.md-adjacent commit
message / the invoking prompt) with
  .venv/bin/python3 scripts/twin_gap.py
Never calls forward/build_params -- only phase1_algebra_head.decode /
_decode_slots (pure numpy) on the banked raw dump.
"""
import json
import os
import pickle
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

import numpy as np

import phase1_algebra_head as H          # decode / _decode_slots only (no forward, no build_params)
import mixed_bucket_census as MBC        # arg_closure, inherited_keys, classify_sets (KEY-INH, reused verbatim)
import lexical_identity_census as LIC    # build_candidate_tables, same_sentence
import stamp_arg_mentions as SAM         # own_var

WILD_A = MBC.WILD_A                      # ".cache/wild_admitted_holdout_a.jsonl" -- same 311 rows/order as the plain fixture
RAWSLOTS = ".cache/rawslots_wild_PMS8_241.pkl"
OUT = ".cache/twin_gap_PMS8_241.txt"
CLASSES = MBC.CLASSES                    # ("DIFFERENT-NOUN", "SAME-NOUN", "MIXED/NO-KEY")


def is_twin(k, comp_idx, derived_i, inh_k, comp_inh_i):
    """per-competitor (not aggregate) SAME-NOUN membership: derived-from-k,
    or its inherited key set intersects k's -- classify_sets' predicate
    applied to ONE competitor instead of "any" over all of them."""
    if derived_i:
        return True
    if inh_k and (comp_inh_i & inh_k):
        return True
    return False


def gap_records(row, row_idx, raw):
    (text, factors, solution, bounds, sspans, intro_map, memo,
     cand_key, cand_sent) = LIC.build_candidate_tables(row)
    cmemo = {}
    closure = {i: MBC.arg_closure(i, factors, intro_map, cmemo) for i in range(len(factors))}
    inh = {i: MBC.inherited_keys(i, factors, cand_key, closure) for i in range(len(factors))}
    n_dup = raw["dup"] if "dup" in raw else np.zeros(len(factors), np.float32)
    recs = []
    n_nonrel_skipped = 0
    for j, fac in enumerate(factors):
        args = fac.get("args")
        if not args:
            continue
        if fac["ftype"] != "rel":
            n_nonrel_skipped += 1
            continue
        args_logits = np.asarray(raw["args"][j], dtype=np.float64)
        dup_flag = bool(n_dup[j] > 0)
        if dup_flag:
            a0 = int(np.argmax(args_logits))
            decoded_set = {a0}
        else:
            decoded_set = set(int(x) for x in np.argsort(-args_logits)[:2])
        for pos, v in enumerate(args):
            k = intro_map.get(v)
            if k is None or k == j:
                continue
            k_sent = cand_sent.get(k, set())
            competitors = [c for c in range(len(factors))
                           if c != j and c != k and LIC.same_sentence(cand_sent.get(c, set()), k_sent)]
            if not competitors:
                continue
            comp_derived = [k in closure[c] for c in competitors]
            comp_inh = [inh[c] for c in competitors]
            twin_flags = [is_twin(k, c, d, inh[k], ci) for c, d, ci in zip(competitors, comp_derived, comp_inh)]
            new_cls = MBC.classify_sets(inh[k], comp_inh, comp_derived)
            # ---- pointer-space competitor variables (own_var, NOT slot index) ----
            comp_vars = []; comp_var_twin = []; comp_var_derived = []
            for c, tw, d in zip(competitors, twin_flags, comp_derived):
                cv = SAM.own_var(factors[c])
                if cv is None or cv == v:
                    continue
                comp_vars.append(cv); comp_var_twin.append(tw); comp_var_derived.append(d)
            if not comp_vars:
                continue
            gold_logit = float(args_logits[v])
            comp_logits = args_logits[comp_vars]
            mx = int(np.argmax(comp_logits))
            max_comp_logit = float(comp_logits[mx])
            gap = gold_logit - max_comp_logit
            strongest_is_twin = bool(comp_var_twin[mx])
            strongest_is_derived = bool(comp_var_derived[mx])
            correct = v in decoded_set
            pick_class = None
            if not correct:
                pick_class = "OTHER"
                for pv in decoded_set:
                    if pv in comp_vars:
                        ci = comp_vars.index(pv)
                        pick_class = "TWIN" if comp_var_twin[ci] else "SAME-SENT-NONTWIN"
                        if pick_class == "TWIN":
                            break
            recs.append(dict(row=row_idx, j=j, pos=pos, v=v, k=k, cls=new_cls,
                              any_derived=any(comp_derived), gap=gap, correct=correct,
                              dup=dup_flag, n_comp=len(comp_vars),
                              strongest_is_twin=strongest_is_twin,
                              strongest_is_derived=strongest_is_derived,
                              pick_class=pick_class))
    return recs, n_nonrel_skipped


def pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else float("nan")


def dist_line(P, label, xs):
    xs = np.asarray(xs, dtype=np.float64)
    if len(xs) == 0:
        P(f"    {label:28s} n=0")
        return
    P(f"    {label:28s} n={len(xs):5d}  mean={xs.mean():7.3f}  median={np.median(xs):7.3f}  "
      f"p10={pct(xs,10):7.3f}  p25={pct(xs,25):7.3f}  p75={pct(xs,75):7.3f}  p90={pct(xs,90):7.3f}  "
      f"share<0={(xs < 0).mean():.3f}")


def main():
    lines = []

    def P(s=""):
        print(s)
        lines.append(s)

    P("=" * 88)
    P("THE TWIN-GAP READ (2026-09-27)")
    P("=" * 88)
    P("scope: gold ftype=='rel' argument decisions with >=1 same-sentence competitor")
    P("(mixed_bucket_census's competitor definition); GAP = args-logit(gold variable)")
    P("- max(args-logit(competitor's own variable)) over those same-sentence competitors,")
    P("in the pointer's own VARIABLE-id space (own_var(slot), not the slot index itself --")
    P("see the module docstring: a slot's own variable is NOT its own index in general).")
    P("correct := gold variable in the decode()-faithful pick (dup slots: single argmax;")
    P("else top-2 argsort) -- the ACTUAL decode, over all 23 other variables, not just the")
    P("twin subset GAP is scored against (GAP is an upper bound on the true decode margin).")
    P("")

    rows = [json.loads(l) for l in open(WILD_A)]
    raw = pickle.load(open(RAWSLOTS, "rb"))
    assert len(rows) == len(raw), f"row count mismatch: {len(rows)} vs {len(raw)}"
    mism = sum(1 for r, d in zip(rows, raw) if r["text"] != d["text"])
    P(f"loaded {len(rows)} rows from {WILD_A}, raw slots from {RAWSLOTS} "
      f"(text mismatches: {mism})")

    all_recs = []
    n_nonrel = 0
    for i, (row, rw) in enumerate(zip(rows, raw)):
        recs, nnr = gap_records(row, i, rw)
        all_recs.extend(recs)
        n_nonrel += nnr
    n = len(all_recs)
    P(f"scored {n} rel-slot argument decisions with >=1 scoreable same-sentence competitor "
      f"(non-rel args-bearing slots excluded: {n_nonrel})")
    n_correct = sum(1 for r in all_recs if r["correct"])
    P(f"decode-faithful accuracy on this set: {n_correct}/{n} = {n_correct/max(n,1):.3f} "
      f"(dup slots: {sum(1 for r in all_recs if r['dup'])})")
    P("")

    # ---- 1. GAP distribution, correct vs wrong, overall and by class ----
    P("-" * 88)
    P("1. GAP DISTRIBUTION (correct vs wrong; overall and by KEY-INH class)")
    P("-" * 88)
    corr = [r["gap"] for r in all_recs if r["correct"]]
    wrong = [r["gap"] for r in all_recs if not r["correct"]]
    P("  OVERALL:")
    dist_line(P, "correct", corr)
    dist_line(P, "wrong", wrong)
    P("")
    for c in CLASSES:
        P(f"  CLASS {c}:")
        cc = [r["gap"] for r in all_recs if r["correct"] and r["cls"] == c]
        cw = [r["gap"] for r in all_recs if not r["correct"] and r["cls"] == c]
        dist_line(P, "correct", cc)
        dist_line(P, "wrong", cw)
        P("")

    # ---- 2. is the wrong pick a twin? ----
    P("-" * 88)
    P("2. AMONG WRONG DECISIONS: IS THE MODEL'S PICK A TWIN?")
    P("-" * 88)
    wrongs = [r for r in all_recs if not r["correct"]]
    nw = len(wrongs)
    for lab in ("TWIN", "SAME-SENT-NONTWIN", "OTHER"):
        cnt = sum(1 for r in wrongs if r["pick_class"] == lab)
        P(f"  pick_class={lab:20s} n={cnt:5d}  share={cnt/max(nw,1):.3f}")
    P(f"  (n_wrong={nw}; TWIN = the model's actual pick matches a same-sentence competitor")
    P(f"   that is itself derived-from-k or key-overlapping with k -- a same-noun twin --")
    P(f"   OTHER = the pick is neither that competitor set's twins nor its non-twins, i.e.")
    P(f"   some slot entirely outside the same-sentence competitor pool.)")
    P("")
    P("  by class (share of that class's wrong instances where the pick is a twin):")
    for c in CLASSES:
        ww = [r for r in wrongs if r["cls"] == c]
        if not ww:
            P(f"    {c:16s} n_wrong=0")
            continue
        tw = sum(1 for r in ww if r["pick_class"] == "TWIN")
        P(f"    {c:16s} n_wrong={len(ww):5d}  pick_is_twin_share={tw/len(ww):.3f}")
    P("")

    # ---- 3. near-miss mass ----
    P("-" * 88)
    P("3. NEAR-MISS MASS (what a margin of size m would have to move)")
    P("-" * 88)
    fragile = [r for r in all_recs if r["correct"] and r["gap"] < 1.0]
    near_win = [r for r in all_recs if not r["correct"] and r["gap"] > -1.0]
    P(f"  fragile wins (correct, GAP<1.0):  n={len(fragile):5d}  "
      f"({len(fragile)/max(len(corr),1):.3f} of correct)")
    P(f"  near wins   (wrong,  GAP>-1.0):   n={len(near_win):5d}  "
      f"({len(near_win)/max(len(wrong),1):.3f} of wrong)")
    for m in (0.5, 1.0, 2.0, 3.0):
        f_m = sum(1 for r in all_recs if r["correct"] and r["gap"] < m)
        n_m = sum(1 for r in all_recs if not r["correct"] and r["gap"] > -m)
        P(f"    m={m:4.1f}: fragile(correct,gap<m)={f_m:5d} ({f_m/max(len(corr),1):.3f})  "
          f"near-win(wrong,gap>-m)={n_m:5d} ({n_m/max(len(wrong),1):.3f})")
    P("")

    # ---- 4. derived-from-k vs not, mean gap per class ----
    P("-" * 88)
    P("4. MEAN GAP: STRONGEST COMPETITOR DERIVED-FROM-k VS NOT (per class)")
    P("-" * 88)
    for c in CLASSES:
        xs_d = [r["gap"] for r in all_recs if r["cls"] == c and r["strongest_is_derived"]]
        xs_nd = [r["gap"] for r in all_recs if r["cls"] == c and not r["strongest_is_derived"]]
        P(f"  {c}:")
        P(f"    strongest competitor DERIVED-FROM-k:     n={len(xs_d):5d}  "
          f"mean_gap={np.mean(xs_d) if xs_d else float('nan'):7.3f}")
        P(f"    strongest competitor NOT derived-from-k: n={len(xs_nd):5d}  "
          f"mean_gap={np.mean(xs_nd) if xs_nd else float('nan'):7.3f}")
    P("")

    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[twin-gap] wrote {OUT}")


if __name__ == "__main__":
    main()
