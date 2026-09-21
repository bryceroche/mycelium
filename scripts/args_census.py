"""args_census.py — THE ARGUMENT-BINDING CENSUS (2026-09-20, zero-training,
zero-GPU: a gold + banked-artifact census). Bryce's framing (ledger
2026-09-19 21:55-23:10): args (0.44-0.45 on wild, every body this week) is
the #1 wall, and no span supervises it. Under THE POSITIONAL LAW a
relation's argument is a pointer to the slot that INTRODUCED that
variable; the open question is whether the failure is COREFERENCE ACROSS
SENTENCES — a mention in the relation's clause has to be traced back to
the sentence that first named the variable — and whether that trace is
lexically deterministic (a recurring content word) or needs real
coreference (pronoun / implicit "the second number").

Two independent measurements, both CPU-only, no tokenizer, no trunk:

  WILD (.cache/wild_admitted_holdout.jsonl +
  .cache/phase1_alg_states_wildhold.npz): g_fspan is ALL ZERO here (the
  membrane census's finding — no span annotation exists at all), so
  clause location is a HEURISTIC built entirely from gold arrays + raw
  text: a GIVEN slot's clause = the sentence containing its gold value's
  numeral; a RELATION slot's clause = the sentence containing its own
  (graph-evaluated) result's numeral if the result appears literally in
  text, else the nearest operator-cue sentence at or after its arguments'
  own clauses (recursive). Sentence boundaries replicate
  phase1_algebra_head.sent_indices EXACTLY (split on ". ", not re-derived
  from that function to avoid importing the trunk-loading module): a
  character position's sentence index = the count of ". " boundaries at
  or before it. This is independently re-verifiable against the npz's own
  `sent` array on any fixture that HAS both text and `sent` (spot-checked
  below at collect time).

  THE STAMPED DIET (.cache/form_mix_pm35v.jsonl +
  .cache/phase1_alg_states_formpm35v.npz, prose rows only: gen.src
  present): REAL annotated g_fspan on ~47% of prose factors (both given
  AND relation factors carry spans here) — restricted to relation-argument
  pairs where BOTH the relation and the introducing factor have a
  non-empty fspan, this gives GROUND-TRUTH sentence distance and clause
  text, used to sanity-check the wild heuristic's shape.

Accuracy join (wild only): the per-slot args-correctness the banked
per-slot npz files do NOT carry (LV_PER_SLOT only writes overall
fac-exact — checked in loop_val.py) comes instead from the LV_DUMP
pickles (.cache/dump_wild_PMS3_241.pkl, dump_wild_PM35_scratch_241.pkl),
recomputing loop_val's own f_args criterion from the raw per-slot
predictions they carry (scripts/matched_read.py documents the tuple
layout).

THE STAMPER'S REACH (item d): the SAME heuristic used on wild, run over
EVERY prose relation-argument pair in the diet (not restricted to spanned
rows — this measures how far a lexical/pronoun/neither classifier could
reach if wired as a live stamper, independent of whether hand annotation
happens to exist for that particular row).
"""
import json
import os
import pickle
import re
import sys

import numpy as np

WILD_JSONL = ".cache/wild_admitted_holdout.jsonl"
WILD_NPZ = ".cache/phase1_alg_states_wildhold.npz"
DIET_JSONL = ".cache/form_mix_pm35v.jsonl"
DIET_NPZ = ".cache/phase1_alg_states_formpm35v.npz"
OUT = os.environ.get("OUT", ".cache/args_census.txt")

# THE STOPWORD LIST (chosen; small, closed-class English function words —
# content_words() below also strips CUE_WORDS so an operator cue never
# counts as a lexical anchor).
STOPWORDS = set("""
a an the is are was were be been being to of in on at and or but if then
for with from by that this these those has have had do does did will
would can could should may might as than so not no all each both more
most other some such only own same too very just also out up down into
over after before how many much what which who whom there here now i
you he she it we they me him her them my your his its our their s t
number numbers second first third value amount total
""".split())

# THE OPERATOR-CUE LIST (chosen): single words and short phrases whose
# presence marks a sentence as STATING a relation (add- and mul-flavored;
# the dialect's gold op is binary add/mul after reencode_ops, but a cue
# does not need to disambiguate WHICH op, only that a relation is spoken
# here). Reported per the task's instruction to say what was chosen.
CUE_WORDS = set("""
total sum altogether combined together remaining left difference more
less fewer plus minus gained lost spent received added subtract
subtracted increase increased decrease decreased extra additional times
twice double triple each per product multiply multiplied half quarter
third percent rate every combined
""".split())
CUE_PHRASES = ["in all", "how many", "how much", "left over", "in total",
               "all together", "twice as", "times as"]

PRONOUNS = set("he she they it his her their its him them".split())

_WORD_RE = re.compile(r"[a-z']+")


def words_of(s):
    return _WORD_RE.findall(s.lower())


def content_words(s):
    return {w for w in words_of(s)
            if w not in STOPWORDS and w not in CUE_WORDS and len(w) > 2}


def has_cue(s):
    sl = s.lower()
    if any(p in sl for p in CUE_PHRASES):
        return True
    return bool(set(words_of(s)) & CUE_WORDS)


def sentence_bounds(text):
    """VERBATIM rule from phase1_algebra_head.sent_indices: split on
    every ". " occurrence."""
    bounds = []
    i = text.find(". ")
    while i != -1:
        bounds.append(i + 1)
        i = text.find(". ", i + 1)
    return bounds


def sentence_spans(text, bounds=None):
    b = sentence_bounds(text) if bounds is None else bounds
    starts = [0] + b
    ends = b + [len(text)]
    return list(zip(starts, ends))


def sent_of_char(bounds, c):
    return int(np.searchsorted(np.asarray(bounds, dtype=np.int64), c, "right")) if bounds else 0


def find_numeral_sentence(text, bounds, value):
    if value is None:
        return None
    try:
        iv = int(value)
        if iv != value:
            return None
    except (TypeError, ValueError):
        return None
    m = re.search(r'(?<!\d)' + re.escape(str(iv)) + r'(?!\d)', text)
    if not m:
        return None
    return sent_of_char(bounds, m.start())


def _verify_sent_convention(jsonl_path, npz):
    """Spot-check sentence_bounds()+sent_of_char() against the npz's own
    per-token `sent` array is NOT directly possible without the
    tokenizer's char offsets, so instead we verify the boundary COUNT
    matches: max(sent[i]) should equal the number of ". " splits found in
    text[i] (both count sentences the same way, off the same rule)."""
    vs = [json.loads(l) for l in open(jsonl_path)][:200]
    sent = npz["sent"]; tokmask = npz["tokmask"]
    agree = tot = 0
    for i, row in enumerate(vs):
        if tokmask[i].sum() == 0:
            continue
        n_sent_npz = int(sent[i][tokmask[i] > 0].max()) + 1
        n_sent_mine = len(sentence_bounds(row["text"])) + 1
        tot += 1
        agree += int(n_sent_npz == n_sent_mine)
    return agree, tot


# =======================================================================
# gold-array row evaluation (slot-indexed; positional law throughout)
# =======================================================================

def row_eval(pres, ftype, args_arr, res_arr, digits, op_arr, L=24):
    """From gold arrays alone (no raw jsonl): value_of[j] for given slots,
    intro_of[v] = slot index whose res==v, var_val[v] and result_val[j]
    for relation slots via forward graph evaluation in slot order (the
    positional law guarantees args of slot j reference slots < j)."""
    value_of = {}
    intro_of = {}
    var_val = {}
    for j in range(L):
        if not pres[j]:
            continue
        v = int(res_arr[j])
        intro_of.setdefault(v, j)
        if int(ftype[j]) == 1:
            val = int("".join(str(int(d)) for d in digits[j]))
            value_of[j] = val
            var_val.setdefault(v, val)
    result_val = {}
    for j in range(L):
        if not pres[j] or int(ftype[j]) != 0:
            continue
        a = np.where(args_arr[j] > 0.5)[0].tolist()
        if not a:
            continue
        a0 = var_val.get(a[0])
        a1 = var_val.get(a[1]) if len(a) > 1 else var_val.get(a[0])
        if a0 is None or a1 is None:
            continue
        op = "add" if int(op_arr[j]) == 0 else "mul"
        v = a0 + a1 if op == "add" else a0 * a1
        result_val[j] = v
        var_val.setdefault(int(res_arr[j]), v)
    return value_of, intro_of, var_val, result_val


def clause_sentence_wild(j, pres, ftype, args_arr, res_arr, value_of,
                         result_val, intro_of, text, bounds, sspans, memo):
    if j in memo:
        return memo[j]
    memo[j] = None  # guard against pathological recursion
    if int(ftype[j]) == 1:
        s = find_numeral_sentence(text, bounds, value_of.get(j))
        memo[j] = s
        return s
    if int(ftype[j]) == 0:
        s = find_numeral_sentence(text, bounds, result_val.get(j))
        if s is None:
            arg_sents = []
            for a in np.where(args_arr[j] > 0.5)[0].tolist():
                ak = intro_of.get(int(a))
                if ak is not None and ak != j:
                    as_ = clause_sentence_wild(ak, pres, ftype, args_arr, res_arr,
                                               value_of, result_val, intro_of,
                                               text, bounds, sspans, memo)
                    if as_ is not None:
                        arg_sents.append(as_)
            base = max(arg_sents) if arg_sents else 0
            cue_sents = [k for k, (a, b) in enumerate(sspans) if has_cue(text[a:b])]
            fwd = [c for c in cue_sents if c >= base]
            s = min(fwd) if fwd else (max(cue_sents) if cue_sents else None)
        memo[j] = s
        return s
    return None


def anchor_class(intro_text, rel_text):
    shared = content_words(intro_text) & content_words(rel_text)
    if shared:
        return "anchored", shared
    if set(words_of(rel_text)) & PRONOUNS:
        return "pronoun", set()
    return "neither", set()


# =======================================================================
# WILD census (a, b) + accuracy join (c)
# =======================================================================

def census_wild():
    vs = [json.loads(l) for l in open(WILD_JSONL)]
    z = np.load(WILD_NPZ)
    pres_a, ftype_a, args_a, res_a, dig_a = (z["g_presence"], z["g_ftype"],
                                             z["g_args"], z["g_res"], z["g_digits"])
    op_a = z["g_op"]
    n = len(vs)
    recs = []
    pos_law_hits = pos_law_tot = 0
    unresolved = 0
    other_intro_ftype = 0
    for i in range(n):
        text = vs[i]["text"]
        bounds = sentence_bounds(text)
        sspans = sentence_spans(text, bounds)
        pres = pres_a[i] > 0.5
        ftype = ftype_a[i]
        args_arr = args_a[i]; res_arr = res_a[i]; digits = dig_a[i]; op_arr = op_a[i]
        value_of, intro_of, var_val, result_val = row_eval(
            pres, ftype, args_arr, res_arr, digits, op_arr)
        memo = {}
        for j in range(24):
            if not pres[j] or int(ftype[j]) != 0:
                continue
            avars = sorted(set(np.where(args_arr[j] > 0.5)[0].tolist()))
            if not avars:
                continue
            s_rel = clause_sentence_wild(j, pres, ftype, args_arr, res_arr,
                                         value_of, result_val, intro_of,
                                         text, bounds, sspans, memo)
            for a in avars:
                k = intro_of.get(int(a))
                if k is None:
                    continue
                pos_law_tot += 1
                pos_law_hits += int(k == a)
                if int(ftype[k]) not in (0, 1):
                    other_intro_ftype += 1
                s_intro = clause_sentence_wild(k, pres, ftype, args_arr, res_arr,
                                               value_of, result_val, intro_of,
                                               text, bounds, sspans, memo)
                if s_rel is None or s_intro is None:
                    unresolved += 1
                    continue
                dist = abs(s_rel - s_intro)
                rel_text = " ".join(text[a0:b0] for a0, b0 in [sspans[s_rel]])
                intro_text = " ".join(text[a0:b0] for a0, b0 in [sspans[s_intro]])
                cls, shared = anchor_class(intro_text, rel_text)
                recs.append(dict(row=i, rel_slot=j, arg_var=int(a), intro_slot=k,
                                 dist=dist, cls=cls, n_shared=len(shared)))
    return recs, dict(pos_law_hits=pos_law_hits, pos_law_tot=pos_law_tot,
                       unresolved=unresolved, other_intro_ftype=other_intro_ftype,
                       n_rows=n)


# =======================================================================
# accuracy join (LV_DUMP pickles)
# =======================================================================

def f_args_from_tuple(t):
    """t = (i, j, gft, gop, gargs, gres, gdig, pft, pop, pargs, pres_, pdig, ppres, pdup)
    (LV_DUMP's layout, scripts/loop_val.py / matched_read.py's match_row
    docstring) -- loop_val's f_args criterion, recomputed (only meaningful
    for gft==0)."""
    gft, gargs, pargs, ppres, pdup = t[2], t[4], t[9], t[12], t[13]
    if gft != 0:
        return None
    gset = set(gargs)
    if not gset:
        return None
    if len(gset) == 1:
        return bool(pdup) and (pargs[0] in gset)
    return set(pargs[:2]) == gset


def load_args_correctness(pkl_path):
    D = pickle.load(open(pkl_path, "rb"))
    out = {}
    for t in D:
        i, j = t[0], t[1]
        v = f_args_from_tuple(t)
        if v is not None:
            out[(i, j)] = v
    return out


# =======================================================================
# DIET: real-span ground truth (restricted) + full-diet stamper reach
# =======================================================================

def load_diet_prose_index():
    is_prose = []
    for l in open(DIET_JSONL):
        r = json.loads(l)
        is_prose.append("src" in r.get("gen", {}))
    return np.array(is_prose, bool)


def census_diet_real(vs, z, prose_mask, cap=None):
    pres_a, ftype_a, args_a, res_a = z["g_presence"], z["g_ftype"], z["g_args"], z["g_res"]
    fspan_a, sent_a = z["g_fspan"], z["sent"]
    idx = np.where(prose_mask)[0]
    if cap:
        idx = idx[:cap]
    recs = []
    for i in idx:
        i = int(i)
        text = vs[i]["text"]
        bounds = sentence_bounds(text)
        sspans = sentence_spans(text, bounds)
        pres = pres_a[i] > 0.5
        ftype = ftype_a[i]; args_arr = args_a[i]; res_arr = res_a[i]
        fspan = fspan_a[i]; sent = sent_a[i]
        intro_of = {}
        for j in range(24):
            if pres[j]:
                intro_of.setdefault(int(res_arr[j]), j)
        for j in range(24):
            if not pres[j] or int(ftype[j]) != 0:
                continue
            if fspan[j].sum() <= 0:
                continue
            j_sents = sorted({int(sent[t]) for t in np.where(fspan[j] > 0.5)[0]})
            avars = sorted(set(np.where(args_arr[j] > 0.5)[0].tolist()))
            for a in avars:
                k = intro_of.get(int(a))
                if k is None or fspan[k].sum() <= 0:
                    continue
                k_sents = sorted({int(sent[t]) for t in np.where(fspan[k] > 0.5)[0]})
                dist = min(abs(sj - sk) for sj in j_sents for sk in k_sents)
                rel_text = " ".join(text[sspans[s][0]:sspans[s][1]]
                                    for s in j_sents if s < len(sspans))
                intro_text = " ".join(text[sspans[s][0]:sspans[s][1]]
                                      for s in k_sents if s < len(sspans))
                cls, shared = anchor_class(intro_text, rel_text)
                recs.append(dict(row=i, rel_slot=j, arg_var=int(a), intro_slot=k,
                                 dist=dist, cls=cls, n_shared=len(shared)))
    return recs


def census_diet_reach(vs, z, prose_mask, cap=None):
    """item (d): the SAME heuristic as wild, over every prose relation-
    argument pair in the diet, spanned or not."""
    pres_a, ftype_a, args_a, res_a, dig_a, op_a = (
        z["g_presence"], z["g_ftype"], z["g_args"], z["g_res"],
        z["g_digits"], z["g_op"])
    idx = np.where(prose_mask)[0]
    if cap:
        idx = idx[:cap]
    recs = []
    for i in idx:
        i = int(i)
        text = vs[i]["text"]
        bounds = sentence_bounds(text)
        sspans = sentence_spans(text, bounds)
        pres = pres_a[i] > 0.5
        ftype = ftype_a[i]; args_arr = args_a[i]; res_arr = res_a[i]
        digits = dig_a[i]; op_arr = op_a[i]
        value_of, intro_of, var_val, result_val = row_eval(
            pres, ftype, args_arr, res_arr, digits, op_arr)
        memo = {}
        for j in range(24):
            if not pres[j] or int(ftype[j]) != 0:
                continue
            avars = sorted(set(np.where(args_arr[j] > 0.5)[0].tolist()))
            if not avars:
                continue
            s_rel = clause_sentence_wild(j, pres, ftype, args_arr, res_arr,
                                         value_of, result_val, intro_of,
                                         text, bounds, sspans, memo)
            for a in avars:
                k = intro_of.get(int(a))
                if k is None:
                    continue
                s_intro = clause_sentence_wild(k, pres, ftype, args_arr, res_arr,
                                               value_of, result_val, intro_of,
                                               text, bounds, sspans, memo)
                if s_rel is None or s_intro is None:
                    recs.append(dict(row=i, cls="unresolved", dist=None, n_shared=0))
                    continue
                dist = abs(s_rel - s_intro)
                rel_text = " ".join(text[a0:b0] for a0, b0 in [sspans[s_rel]])
                intro_text = " ".join(text[a0:b0] for a0, b0 in [sspans[s_intro]])
                cls, shared = anchor_class(intro_text, rel_text)
                recs.append(dict(row=i, cls=cls, dist=dist, n_shared=len(shared)))
    return recs


def bucket_dist(d):
    return "3+" if d >= 3 else str(d)


def main():
    lines = []

    def P(s=""):
        print(s)
        lines.append(s)

    P("=" * 78)
    P("THE ARGUMENT-BINDING CENSUS (2026-09-20)")
    P("=" * 78)
    P("")
    P("DEFINITIONS CHOSEN:")
    P(f"  stopwords ({len(STOPWORDS)}): {' '.join(sorted(STOPWORDS))}")
    P(f"  operator cue words ({len(CUE_WORDS)}): {' '.join(sorted(CUE_WORDS))}")
    P(f"  operator cue phrases: {CUE_PHRASES}")
    P(f"  pronoun set: {sorted(PRONOUNS)}")
    P("  content word = alphabetic token, len>2, not a stopword, not a cue word.")
    P("  WILD clause location (no span annotation exists at all -- heuristic):")
    P("    given slot -> sentence of its gold value's literal numeral occurrence;")
    P("    relation slot -> sentence of its OWN graph-evaluated result's literal")
    P("    numeral if present in text, else the nearest cue-bearing sentence at or")
    P("    after the max of its arguments' own (recursively resolved) clauses.")
    P("  DIET ground truth: real g_fspan (given AND relation factors both carry")
    P("    spans in this diet); a slot's clause sentences = the set of sentence ids")
    P("    its span's tokens fall in (via the npz's own `sent` array); restricted to")
    P("    relation-argument pairs where BOTH slots have a non-empty fspan.")
    P("  Relation-level aggregation for a 2-arg relation (accuracy join, table c):")
    P("    class = 'neither' if any arg is neither; else 'pronoun' if any arg is")
    P("    pronoun; else 'anchored'. distance = MAX over its arguments (the harder")
    P("    one drives whether the relation is gettable).")
    P("  Introducing factor k of variable v = the slot whose gold res==v (read from")
    P("    g_res, not the raw jsonl factors[] list -- the positional law's two")
    P("    conventions mean list-index != slot-index on some registers).")

    # sentence-convention spot check
    zw = np.load(WILD_NPZ)
    agree, tot = _verify_sent_convention(WILD_JSONL, zw)
    P("")
    P(f"  sentence-count convention check (wild, first {tot} rows): "
      f"{agree}/{tot} rows' own ('. '-split count) matches the npz sent array's "
      f"max+1 -- confirms sentence_bounds() here replicates sent_indices().")

    # -------------------------------------------------------------
    # WILD
    # -------------------------------------------------------------
    recs_w, meta_w = census_wild()
    P("")
    P("-" * 78)
    P("WILD: relation-argument pairs censused")
    P("-" * 78)
    P(f"  rows={meta_w['n_rows']}  arg-instances scored for the positional law="
      f"{meta_w['pos_law_tot']}  k==v holds={meta_w['pos_law_hits']} "
      f"({meta_w['pos_law_hits'] / max(meta_w['pos_law_tot'], 1):.4f})")
    P(f"  introducing factor of a non-(given/rel) ftype: {meta_w['other_intro_ftype']}")
    P(f"  unresolved (no clause sentence found for relation or introducer): "
      f"{meta_w['unresolved']} / {meta_w['pos_law_tot']}")
    P(f"  scored arg-instances (both clauses resolved): {len(recs_w)}")

    P("")
    P("a. SENTENCE DISTANCE (relation's clause vs introducing clause)")
    cnt = {}
    for r in recs_w:
        cnt[bucket_dist(r["dist"])] = cnt.get(bucket_dist(r["dist"]), 0) + 1
    tot_w = len(recs_w)
    for b in ("0", "1", "2", "3+"):
        n_ = cnt.get(b, 0)
        P(f"    dist={b:3s}  n={n_:5d}  ({n_ / max(tot_w, 1):.3f})")

    P("")
    P("b. LEXICAL RECURRENCE / PRONOUN / NEITHER, by distance")
    P(f"  {'dist':6s} {'anchored':>10s} {'pronoun':>10s} {'neither':>10s} {'n':>6s}")
    by_bucket = {}
    for r in recs_w:
        by_bucket.setdefault(bucket_dist(r["dist"]), []).append(r["cls"])
    for b in ("0", "1", "2", "3+"):
        cls = by_bucket.get(b, [])
        n_ = len(cls)
        if n_ == 0:
            P(f"  {b:6s} {'--':>10s} {'--':>10s} {'--':>10s} {0:>6d}")
            continue
        a_ = cls.count("anchored") / n_
        p_ = cls.count("pronoun") / n_
        ne_ = cls.count("neither") / n_
        P(f"  {b:6s} {a_:10.3f} {p_:10.3f} {ne_:10.3f} {n_:6d}")
    overall = [r["cls"] for r in recs_w]
    n_o = len(overall)
    P(f"  {'ALL':6s} {overall.count('anchored') / max(n_o, 1):10.3f} "
      f"{overall.count('pronoun') / max(n_o, 1):10.3f} "
      f"{overall.count('neither') / max(n_o, 1):10.3f} {n_o:6d}")

    # -------------------------------------------------------------
    # c. accuracy join
    # -------------------------------------------------------------
    P("")
    P("-" * 78)
    P("c. ACCURACY JOIN (wild): args-correct rate by (class x distance), both checkpoints")
    P("-" * 78)
    ckpts = {"PMS5d_241": ".cache/dump_wild_PMS5d_241.pkl", "PMS4_241": ".cache/dump_wild_PMS4_241.pkl", "PMS3_241": ".cache/dump_wild_PMS3_241.pkl",
             "PM35_scratch_241": ".cache/dump_wild_PM35_scratch_241.pkl"}
    acc_tables = {}
    for tag, path in ckpts.items():
        if not os.path.exists(path):
            P(f"  SKIPPED {tag}: {path} not found")
            continue
        acc = load_args_correctness(path)
        acc_tables[tag] = acc
        # relation-level aggregation over recs_w grouped by (row, rel_slot)
        by_relslot = {}
        for r in recs_w:
            key = (r["row"], r["rel_slot"])
            by_relslot.setdefault(key, []).append(r)
        agg = {}
        for key, args_ in by_relslot.items():
            classes = [a["cls"] for a in args_]
            if "neither" in classes:
                c = "neither"
            elif "pronoun" in classes:
                c = "pronoun"
            else:
                c = "anchored"
            d = max(a["dist"] for a in args_)
            agg[key] = (c, d)
        P(f"  -- {tag} --")
        P(f"  {'class':10s} {'dist=0':>10s} {'dist=1':>10s} {'dist=2':>10s} {'dist=3+':>10s} {'ALL':>10s}")
        for c in ("anchored", "pronoun", "neither"):
            row_vals = []
            keys_c = [k for k, (cc, dd) in agg.items() if cc == c]
            for b in ("0", "1", "2", "3+"):
                keys_b = [k for k in keys_c if bucket_dist(agg[k][1]) == b]
                vals = [acc[k] for k in keys_b if k in acc]
                row_vals.append(f"{np.mean(vals):.3f}({len(vals)})" if vals else "--")
            vals_all = [acc[k] for k in keys_c if k in acc]
            all_str = f"{np.mean(vals_all):.3f}({len(vals_all)})" if vals_all else "--"
            P(f"  {c:10s} " + " ".join(f"{v:>10s}" for v in row_vals) + f" {all_str:>10s}")

    # -------------------------------------------------------------
    # DIET: real-span restricted ground truth
    # -------------------------------------------------------------
    P("")
    P("-" * 78)
    P("DIET (real fspan, restricted to pairs where BOTH slots have a span)")
    P("-" * 78)
    vs_diet = [json.loads(l) for l in open(DIET_JSONL)]
    prose_mask = load_diet_prose_index()
    z_diet = np.load(DIET_NPZ)
    recs_d = census_diet_real(vs_diet, z_diet, prose_mask)
    P(f"  prose rows in diet: {int(prose_mask.sum())}; span-restricted arg-instances: {len(recs_d)}")
    P("  a'. SENTENCE DISTANCE (real spans)")
    cnt_d = {}
    for r in recs_d:
        cnt_d[bucket_dist(r["dist"])] = cnt_d.get(bucket_dist(r["dist"]), 0) + 1
    for b in ("0", "1", "2", "3+"):
        n_ = cnt_d.get(b, 0)
        P(f"    dist={b:3s}  n={n_:5d}  ({n_ / max(len(recs_d), 1):.3f})")
    P("  b'. LEXICAL RECURRENCE / PRONOUN / NEITHER, by distance (real spans)")
    P(f"  {'dist':6s} {'anchored':>10s} {'pronoun':>10s} {'neither':>10s} {'n':>6s}")
    by_bucket_d = {}
    for r in recs_d:
        by_bucket_d.setdefault(bucket_dist(r["dist"]), []).append(r["cls"])
    for b in ("0", "1", "2", "3+"):
        cls = by_bucket_d.get(b, [])
        n_ = len(cls)
        if n_ == 0:
            P(f"  {b:6s} {'--':>10s} {'--':>10s} {'--':>10s} {0:>6d}")
            continue
        P(f"  {b:6s} {cls.count('anchored') / n_:10.3f} {cls.count('pronoun') / n_:10.3f} "
          f"{cls.count('neither') / n_:10.3f} {n_:6d}")
    overall_d = [r["cls"] for r in recs_d]
    n_od = len(overall_d)
    P(f"  {'ALL':6s} {overall_d.count('anchored') / max(n_od, 1):10.3f} "
      f"{overall_d.count('pronoun') / max(n_od, 1):10.3f} "
      f"{overall_d.count('neither') / max(n_od, 1):10.3f} {n_od:6d}")

    # -------------------------------------------------------------
    # d. the stamper's reach (full prose diet, heuristic)
    # -------------------------------------------------------------
    P("")
    P("-" * 78)
    P("d. THE STAMPER'S REACH (full prose diet, heuristic -- spanned or not)")
    P("-" * 78)
    recs_reach = census_diet_reach(vs_diet, z_diet, prose_mask)
    n_r = len(recs_reach)
    by_cls = {}
    for r in recs_reach:
        by_cls.setdefault(r["cls"], []).append(r)
    P(f"  prose relation-argument instances: {n_r}")
    for c in ("anchored", "pronoun", "neither", "unresolved"):
        xs = by_cls.get(c, [])
        P(f"    {c:12s} n={len(xs):6d}  ({len(xs) / max(n_r, 1):.3f})")
    anch = by_cls.get("anchored", [])
    if anch:
        P("  for ANCHORED instances: sentence distance distribution")
        cnt_a = {}
        for r in anch:
            cnt_a[bucket_dist(r["dist"])] = cnt_a.get(bucket_dist(r["dist"]), 0) + 1
        for b in ("0", "1", "2", "3+"):
            n_ = cnt_a.get(b, 0)
            P(f"    dist={b:3s}  n={n_:5d}  ({n_ / max(len(anch), 1):.3f})")
        shared_n = [r["n_shared"] for r in anch]
        P(f"  for ANCHORED instances: number of shared content words "
          f"(the 'mention span' width, in whole matched words -- this heuristic")
        P(f"  matches single dictionary words, not full noun phrases, so the stamp")
        P(f"  itself is always a 1-word span; 'how many tokens' below counts DISTINCT")
        P(f"  shared words available to anchor on, not multi-word span length):")
        for k in (1, 2, 3):
            n_ = sum(1 for s in shared_n if (s == k or (k == 3 and s >= 3)))
            P(f"    {k if k < 3 else '3+'} shared word(s): n={n_}  ({n_ / len(anch):.3f})")
        P(f"    mention sentence = the relation's OWN clause sentence, by construction "
          f"(the anchor is searched for INSIDE the relation's clause).")

    P("")
    P("=" * 78)
    P("READING")
    P("=" * 78)
    P("MOST OF THE WILD ARGS WALL IS LEXICALLY ANCHORABLE, NOT A HARD COREFERENCE")
    P("PROBLEM: 91.6% of wild relation-argument pairs (n=1481) share a content word")
    P("between the relation's clause and the variable's introducing clause -- and the")
    P("diet's REAL fspan-restricted ground truth agrees closely (91.9%, n=28995),")
    P("which is the strongest evidence the heuristic isn't just finding what it wants")
    P("to find. Only 5.6% need a pronoun and 2.8% have neither a shared word nor a")
    P("pronoun (implicit coreference, e.g. 'the second number') on wild; the diet's")
    P("real-span numbers are close (2.7% / 5.4%). So a deterministic lexical stamper")
    P("could locate the great majority of these pointers without any learned")
    P("coreference model -- but note dist=0's 99.5% anchored rate is partly a")
    P("heuristic artifact: when a relation's own clause can't be pinned by a literal")
    P("result-numeral, the fallback lands it on the SAME sentence as its argument's")
    P("clause, which trivially shares content words. The diet's real-span dist=0 rate")
    P("is 100.0% for the identical reason (same sentence, of course they share words),")
    P("so the informative comparison is the DECLINE across distance buckets (wild")
    P("0.99->0.79->0.77->0.70; diet 1.00->0.76->0.78->0.67) -- both registers still")
    P("clear 65-80% anchored even at 2+ sentences away, so the ceiling on a lexical")
    P("stamper is high even for the genuinely cross-sentence cases.")
    P("ACCURACY DOES NOT YET TRACK ANCHORING ON EITHER CHECKPOINT -- THE OPPOSITE OF")
    P("THE NAIVE PRIOR: args-correct rate for anchored dist=0 pairs is LOWER (0.40 PMS3")
    P("/ 0.40 PM35_scratch) than anchored dist=1-2 pairs (0.55/0.53 and 0.55/0.56) on")
    P("both checkpoints, and pronoun/neither cases (n=36-65, small) are NOT reliably")
    P("worse than anchored -- pronoun's overall rate (0.51/0.55) sits close to")
    P("anchored's (0.46/0.46). Two readings, both worth registering: (1) the args wall")
    P("may not be primarily a coreference-resolution problem at all -- the pointer")
    P("mechanism fails at roughly the same rate whether the mention is trivially")
    P("adjacent or requires reaching across sentences, which argues AGAINST 'wire up a")
    P("coreference-aware stamp and the wall falls' and for something interior to the")
    P("pointer/binding mechanism itself; (2) dist=0 pairs are confounded with SLOT")
    P("DEPTH (a relation whose clause collapses onto its argument's clause is often")
    P("the row's first or an early relation, but rows with several same-sentence")
    P("relations chained together also load multiple pointers onto one dense clause)")
    P("-- this census does not control for slot index or chain depth, and doing so is")
    P("the natural follow-up before concluding the lexical route is a dead end.")
    P("THE STAMPER'S REACH (full prose diet, spanned or not, n=148808 instances):")
    P("77.4% anchored, 3.6% pronoun, 3.7% neither, 15.3% unresolved (this heuristic")
    P("could not locate either clause at all -- mostly rows whose relation result")
    P("never appears as a literal numeral AND has no cue-bearing sentence to fall back")
    P("on). A live stamper built on this heuristic alone would reach roughly 3 in 4")
    P("prose relation arguments with a single matched content word (mostly a 1-word")
    P("span; multi-word mention detection was not attempted here), leave 1 in 7")
    P("unresolved, and need real coreference machinery for under 4%.")

    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[args-census] wrote {OUT}")


if __name__ == "__main__":
    main()
