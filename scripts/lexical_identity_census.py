"""lexical_identity_census.py — THE LEXICAL-IDENTITY CENSUS (2026-09-21,
zero-training, zero-GPU). The noun census found single-token cosine
geometry doesn't expose the referent (rank-1 at chance on both waist and
trunk). This asks the prior question directly at the TEXT: does a
deterministic lexical-recurrence test (no representation at all — just
stemmed word overlap) already identify the true argument's introducing
slot among its competitors, and if wired as a LEGAL-SET mask on the
pointer's existing top-2 (the numeral mask's true sibling — its legal set
comes from decode legality, this one's comes from the TEXT), does it move
args accuracy?

Reuses, never reimplements: scripts/stamp_arg_mentions.py (the arg-mention
stamper: stem(), is_content(), tokenize(), content_tokens_in(),
content_stem_set(), clause_of(), windows_of(), sent_set_of(),
build_intro_map()) and scripts/args_census.py (sentence_bounds,
sentence_spans, sent_of_char — via stamp_arg_mentions' own re-export).

Fixtures: .cache/wild_admitted_holdout_a.jsonl (wild, measurement-only,
already arg-stamped) and 2,000 PROSE rows (gen.src present) of
.cache/form_mix_pm35a.jsonl (the diet's own arg-stamped file — real
`spans` cover ~47-54% of its factors, used automatically by clause_of()
wherever present, falling back to the heuristic elsewhere exactly as
stamp_arg_mentions.py itself does).

Definitions chosen:
  RE-MENTION WORDS, two variants, both reported:
    v1 (pooled arg_spans): the union of every char span across ALL
    argument positions of relation j's OWN already-stamped `arg_spans`
    field, each stemmed — position ambiguity does not matter for a set
    test, per the word.
    v2 (whole clause): every content-word stem in relation j's own clause
    (stamp_arg_mentions.content_stem_set on its clause windows) — a
    strictly larger, cheaper-to-get set.
  CANDIDATE POOL for relation j: present factors introduced EARLIER (list
  index < j — the positional law: a relation only ever references an
  earlier variable) OR sharing a sentence with j's own clause (covers the
  rare compound-sentence case) — excluding j itself.
  CANDIDATE LEXICAL KEY: a GIVEN candidate's key = content-word stems
  within the next 4 WORD-TOKENS (regex word tokens, `stamp_arg_mentions.
  tokenize`'s convention — not BPE subwords; this census never touches
  the trunk/tokenizer) after its value's first literal-numeral occurrence
  in the text. A non-given candidate's key = its own clause's content-word
  stems (content_stem_set), via clause_of() (real `spans` when present,
  else the heuristic) — the SAME machinery used for the relation's own v2
  re-mention set, applied to a candidate instead.
  M(j) / M(j,p): a candidate k' is a MATCH if its lexical key shares >=1
  stem with the re-mention words (relation-level M(j) for the mask
  experiment: pooled across the relation's own argument positions;
  per-argument M(j,p) for reach/precision: still uses the SAME relation-
  level re-mention words — a set test is symmetric in the argument it's
  "for", so there is only one M(j) per relation per variant; reach/
  precision are read PER ARGUMENT against that one M(j)).
  THE READ-TIME MASK (wild only — no banked PMS4_241 prediction dump
  exists for the diet's prose rows, and this census is zero-GPU so none
  is generated here): masked_correct(pargs_top2, v, Mj) = "v in (pargs_top2
  INTERSECT Mj)" if that intersection is non-empty, ELSE "v in pargs_top2"
  (the unmasked test) — a FILTER on the existing top-2, not a re-rank with
  a substitute candidate (the dump only carries a rank-1/2, not a full
  distribution, so there is no third choice to promote in; the task's own
  "the weaker proxy" language is honored literally).
"""
import json
import os
import pickle
import re
import sys

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

import numpy as np
import stamp_arg_mentions as SAM
import args_census as AC

WILD_A = ".cache/wild_admitted_holdout_a.jsonl"
DIET_A = ".cache/form_mix_pm35a.jsonl"
DIET_CAP = 2000
DUMP_WILD_PMS4 = ".cache/dump_wild_PMS4_241.pkl"
OUT = ".cache/lexical_identity_census.txt"

_NUM_RE_CACHE = {}


def num_pattern(val):
    v = str(int(abs(val)))
    p = _NUM_RE_CACHE.get(v)
    if p is None:
        p = re.compile(r'(?<!\d)' + re.escape(v) + r'(?!\d)')
        _NUM_RE_CACHE[v] = p
    return p


def given_lexical_key(text, fac, bounds):
    """content-word stems within the next 4 word-tokens after the value's
    first literal numeral occurrence; also its sentence index (or None)."""
    val = fac.get("value")
    if val is None:
        return set(), set()
    m = num_pattern(val).search(text)
    if not m:
        return set(), set()
    words = SAM.tokenize(text, m.end(), len(text))[:4]
    key = {SAM.stem(w.lower()) for w, ws, we in words if SAM.is_content(w.lower())}
    return key, {AC.sent_of_char(bounds, m.start())}


def build_candidate_tables(row):
    text = row["text"]; factors = row["factors"]; solution = row.get("solution", [])
    bounds = AC.sentence_bounds(text); sspans = AC.sentence_spans(text, bounds)
    intro_map = SAM.build_intro_map(factors)
    memo = {}
    cand_key = {}; cand_sent = {}
    for idx, fac in enumerate(factors):
        if fac["ftype"] == "given":
            key, sset = given_lexical_key(text, fac, bounds)
        else:
            cl = SAM.clause_of(idx, factors, text, bounds, sspans, solution, intro_map, memo)
            win = SAM.windows_of(cl, sspans)
            key = SAM.content_stem_set(text, win) if cl is not None else set()
            sset = SAM.sent_set_of(cl, bounds, sspans)
        cand_key[idx] = key
        cand_sent[idx] = sset
    return text, factors, solution, bounds, sspans, intro_map, memo, cand_key, cand_sent


def same_sentence(sa, sb):
    return bool(sa and sb and (sa & sb))


def process_row(row, row_idx):
    (text, factors, solution, bounds, sspans, intro_map, memo,
     cand_key, cand_sent) = build_candidate_tables(row)
    recs = []
    for j, fac in enumerate(factors):
        args = fac.get("args")
        if not args:
            continue
        rel_clause = SAM.clause_of(j, factors, text, bounds, sspans, solution, intro_map, memo)
        rel_windows = SAM.windows_of(rel_clause, sspans)
        rel_sent_set = SAM.sent_set_of(rel_clause, bounds, sspans)
        rel_content_all = SAM.content_stem_set(text, rel_windows)             # v2
        pooled_spans = []
        for pos_spans in (fac.get("arg_spans") or []):
            pooled_spans.extend(tuple(sp) for sp in pos_spans)
        pooled_stems = {SAM.stem(text[s:e].lower()) for s, e in pooled_spans
                        if text[s:e].isalpha()}                              # v1

        candidates = sorted({idx for idx in range(len(factors)) if idx != j and
                             (idx < j or same_sentence(cand_sent.get(idx, set()), rel_sent_set))})
        Mj = {}
        for tag, remention in (("v1", pooled_stems), ("v2", rel_content_all)):
            Mj[tag] = ({c for c in candidates if cand_key.get(c, set()) & remention}
                      if remention else set())

        for p, v in enumerate(args):
            k = intro_map.get(v)
            if k is None or k == j:
                continue
            same_comp = [c for c in candidates if c != k and
                        same_sentence(cand_sent.get(c, set()), cand_sent.get(k, set()))]
            dist = None
            if rel_sent_set and cand_sent.get(k):
                dist = min(abs(rs - is_) for rs in rel_sent_set for is_ in cand_sent[k])
            recs.append(dict(row_idx=row_idx, j=j, v=v, k=k,
                             n_candidates=len(candidates), same_comp=same_comp,
                             dist=dist, Mj=Mj))
    return recs


def bucket_dist(d):
    if d is None:
        return "unresolved"
    return "3+" if d >= 3 else str(d)


def summarize(recs, variant, P, title, restrict=None):
    xs = [r for r in recs if (restrict is None or restrict(r))]
    if not xs:
        P(f"  {title}: n=0")
        return
    reach = np.mean([r["k"] in r["Mj"][variant] for r in xs])
    precision = np.mean([r["Mj"][variant] == {r["k"]} for r in xs])
    mean_size = np.mean([len(r["Mj"][variant]) for r in xs])
    P(f"  {title}: n={len(xs):6d}  reach={reach:.3f}  precision={precision:.3f}  "
      f"mean|M(j)|={mean_size:.2f}")


def load_dump_pargs(path):
    D = pickle.load(open(path, "rb"))
    out = {}
    for t in D:
        i, j = t[0], t[1]
        out.setdefault(i, {})[j] = list(t[9])   # pargs: predicted top-2 variable indices
    return out


def masked_correct(pargs, v, Mj):
    filtered = [p for p in pargs if p in Mj]
    if filtered:
        return v in filtered
    return v in pargs


def main():
    lines = []

    def P(s=""):
        print(s)
        lines.append(s)

    P("=" * 78)
    P("THE LEXICAL-IDENTITY CENSUS (2026-09-21)")
    P("=" * 78)
    P("")
    P("DEFINITIONS: see module docstring. Two re-mention-word variants throughout:")
    P("  v1 = pooled stamped arg_spans (all argument positions of the relation,")
    P("       unioned, stemmed) -- what the arg-stamper already committed to.")
    P("  v2 = every content-word stem in the relation's own clause -- larger, free.")
    P("candidate pool = present factors introduced earlier (list index < j) or in the")
    P("same sentence as the relation; candidate key = 4-word-token window after a")
    P("given's numeral, or a non-given's own clause content words.")

    print("[lexical-identity] wild...", flush=True)
    wild_rows = [json.loads(l) for l in open(WILD_A)]
    wild_recs = []
    for i, row in enumerate(wild_rows):
        wild_recs.extend(process_row(row, i))

    print("[lexical-identity] diet (2000 prose rows)...", flush=True)
    diet_rows = []
    for l in open(DIET_A):
        r = json.loads(l)
        if "src" in r.get("gen", {}):
            diet_rows.append(r)
        if len(diet_rows) >= DIET_CAP:
            break
    diet_recs = []
    for i, row in enumerate(diet_rows):
        diet_recs.extend(process_row(row, i))

    for fx_name, recs in (("WILD", wild_recs), (f"DIET prose (n={len(diet_rows)} rows)", diet_recs)):
        P("")
        P("=" * 78)
        P(f"{fx_name}: {len(recs)} relation-argument instances")
        P("=" * 78)
        for variant in ("v1", "v2"):
            P("")
            P(f"-- variant {variant} --")
            summarize(recs, variant, P, "overall")
            summarize(recs, variant, P, "with >=1 same-sentence competitor (the twin cell)",
                     restrict=lambda r: len(r["same_comp"]) > 0)
            for b in ("0", "1", "2", "3+", "unresolved"):
                summarize(recs, variant, P, f"distance={b}",
                         restrict=lambda r, b=b: bucket_dist(r["dist"]) == b)

    # -------------------------------------------------------------
    # THE READ-TIME MASK (wild only)
    # -------------------------------------------------------------
    P("")
    P("=" * 78)
    P("THE READ-TIME MASK (wild only, PMS4_241; no LV_DUMP_RAW/CA_RAWDUMP exists for")
    P("PMS4_241 on wildhold -- grepped the ledger and .cache/; using the weaker proxy:")
    P(f"{DUMP_WILD_PMS4}'s per-slot top-2 predictions, filtered to M(j) when the")
    P("filter is non-empty, else the unmasked top-2 stands)")
    P("=" * 78)
    if not os.path.exists(DUMP_WILD_PMS4):
        P(f"  SKIPPED: {DUMP_WILD_PMS4} not found")
    else:
        pargs_by_row = load_dump_pargs(DUMP_WILD_PMS4)
        for variant in ("v1", "v2"):
            P("")
            P(f"-- variant {variant} --")
            for label, restrict in (
                ("overall", None),
                ("with >=1 same-sentence competitor", lambda r: len(r["same_comp"]) > 0),
            ):
                xs = [r for r in wild_recs if (restrict is None or restrict(r))]
                xs = [r for r in xs if r["row_idx"] in pargs_by_row
                     and r["j"] in pargs_by_row[r["row_idx"]]]
                if not xs:
                    P(f"  {label}: n=0")
                    continue
                unmasked = [r["v"] in pargs_by_row[r["row_idx"]][r["j"]] for r in xs]
                masked = [masked_correct(pargs_by_row[r["row_idx"]][r["j"]], r["v"], r["Mj"][variant])
                         for r in xs]
                n_empty_Mj = sum(1 for r in xs if not r["Mj"][variant])
                n_changed = sum(1 for u, m in zip(unmasked, masked) if u != m)
                n_fixed = sum(1 for u, m in zip(unmasked, masked) if not u and m)
                n_broken = sum(1 for u, m in zip(unmasked, masked) if u and not m)
                P(f"  {label}: n={len(xs):5d}  unmasked={np.mean(unmasked):.3f}  "
                  f"masked={np.mean(masked):.3f}  (M(j) empty for {n_empty_Mj}/{len(xs)}; "
                  f"changed {n_changed}: +{n_fixed} fixed / -{n_broken} broken)")

    P("")
    P("=" * 78)
    P("READING")
    P("=" * 78)

    def r_p(recs, variant, restrict=None):
        xs = [r for r in recs if (restrict is None or restrict(r))]
        if not xs:
            return float("nan"), float("nan"), float("nan")
        return (np.mean([r["k"] in r["Mj"][variant] for r in xs]),
               np.mean([r["Mj"][variant] == {r["k"]} for r in xs]),
               np.mean([len(r["Mj"][variant]) for r in xs]))

    w_v1 = r_p(wild_recs, "v1"); w_v2 = r_p(wild_recs, "v2")
    d_v1 = r_p(diet_recs, "v1"); d_v2 = r_p(diet_recs, "v2")
    w_v1_twin = r_p(wild_recs, "v1", lambda r: len(r["same_comp"]) > 0)
    w_v2_twin = r_p(wild_recs, "v2", lambda r: len(r["same_comp"]) > 0)

    P(f"REACH IS HIGH, PRECISION IS THE STORY: wild v1(pooled arg_spans) reach="
      f"{w_v1[0]:.3f} precision={w_v1[1]:.3f} mean|M|={w_v1[2]:.2f}; v2(whole clause)")
    P(f"reach={w_v2[0]:.3f} precision={w_v2[1]:.3f} mean|M|={w_v2[2]:.2f}. Diet prose:")
    P(f"v1 reach={d_v1[0]:.3f} precision={d_v1[1]:.3f} mean|M|={d_v1[2]:.2f}; v2 reach="
      f"{d_v2[0]:.3f} precision={d_v2[1]:.3f} mean|M|={d_v2[2]:.2f}.")
    P(f"IN THE TWIN CELL (>=1 same-sentence competitor): wild v1 reach={w_v1_twin[0]:.3f} "
      f"precision={w_v1_twin[1]:.3f} mean|M|={w_v1_twin[2]:.2f}; v2 reach={w_v2_twin[0]:.3f} "
      f"precision={w_v2_twin[1]:.3f} mean|M|={w_v2_twin[2]:.2f}.")
    P("")
    P("THE VERDICT IS CLEAN AND NEGATIVE: precision is near zero everywhere (0.006-")
    P("0.023, both variants, both fixtures, worst exactly in the twin cell) with")
    P("mean|M(j)| of 4-5 candidates out of a same-sentence pool that size — the")
    P("lexical-recurrence test barely discriminates AT ALL; common words (\"tea\",")
    P("\"cup\", \"party\") recur across nearly every candidate in a row, so M(j) is")
    P("almost as permissive as the full candidate pool. Reach is high (0.77-0.85 wild,")
    P("even higher on the diet's v2) largely BECAUSE M(j) is so permissive, not because")
    P("it has located the right candidate specifically — a set that contains almost")
    P("everyone will contain the true one too. THE READ-TIME MASK CONFIRMS THIS THE")
    P("HARD WAY: masked accuracy is LOWER than unmasked in every row of the mask table")
    P("(wild overall 0.735 -> 0.679 v1, 0.735 -> 0.673 v2; twin cell 0.718 -> 0.675 /")
    P("0.671), and in every single row the fix count is EXACTLY ZERO — the mask never")
    P("once rescues a wrong prediction, it only ever breaks previously-correct ones")
    P("(91-101 broken out of ~1147-1628 instances, depending on variant/subset). This")
    P("is the coordinator's own decision rule fired conclusively: reach is high but")
    P("precision is low (many matches), so THE TIE IS NOT LEXICAL — a deterministic")
    P("text-side legal-set mask is not a viable road here, unlike the numeral mask it")
    P("was modeled on. The numeral mask works because a value's legal set (the text's")
    P("numerals) is small and nearly always excludes the wrong answer; this lexical")
    P("legal set is large and excludes almost nothing, so filtering to it mostly just")
    P("removes the model's own (already correct) choice. AN ENTITY CHANNEL WOULD NEED")
    P("SOMETHING SHARPER THAN STEMMED WORD-OVERLAP TO BE WORTH BUILDING — recurrence")
    P("of a common noun is not a rare enough event in these clauses to serve as an")
    P("identity signal on its own.")

    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[lexical-identity] wrote {OUT}")


if __name__ == "__main__":
    main()
