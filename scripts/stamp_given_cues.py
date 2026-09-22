"""stamp_given_cues.py -- THE ROLE SIGNATURE, data half (2026-09-22,
zero-token, zero-GPU, CPU-only). Ledger 2026-09-22 "THE TWIN-SPLIT
VERDICT": SAME-NOUN twins (the same entity at different roles/times --
"bought 3 tickets ... 5 more tickets"; "had 12 apples, ate 4, has 8
left") carry 2.8x the wrong-argument mass of DIFFERENT-NOUN twins, 11x
at the classic same-sentence tie (distance 0). No noun distinguishes a
same-noun twin pair -- the discriminator has to be the CUE attached to
each quantity's own clause (bought / has / more / left / each / then /
first). This script stamps that cue, per factor, so a future arm can
bind a role phasor to a slot's cue + order-of-introduction instead of
its (non-existent) noun identity.

Reuses, never reimplements: scripts/args_census.py (STOPWORDS, CUE_WORDS,
CUE_PHRASES, sentence_bounds/sentence_spans/sent_of_char, has_cue) and
scripts/stamp_arg_mentions.py (WORD_RE/tokenize, stem, is_content,
build_intro_map, own_value, clause_of, windows_of, sent_set_of --
imported, not reimplemented, so no drift between the three stampers).
scripts/twin_split_census.py's classify() and
scripts/lexical_identity_census.py's build_candidate_tables()/
one_token_key() are reused verbatim for the same-noun discriminator
census at the bottom of main() (they define what a "same-noun twin
pair" IS; this script only asks whether the cue differs across it).

THE ROLE-CUE LEXICON (printed in the report): the existing operator-cue
lexicon (args_census.CUE_WORDS / CUE_PHRASES -- total, more, left,
each, twice, ...) UNIONED with new temporal/possessive/state cues:
  acquisition/loss verbs: has had have got gets bought buys received
    earned found made gave gives sold spent lost ate eats used uses
    needs wants paid
  state/remainder:        left remaining remain rest still now total
    altogether together (+ phrase "in all")
  temporal/order:         first then next later after before initially
    originally finally every per each (+ phrases "at first", "each day")
  comparatives:           more fewer less than twice half double triple
    times
  collectives:             both all together combined
Matching is WHOLE-WORD, stemmed exactly as stem() stems (so plural/verb
forms not spelled out literally above still collapse onto a listed
form, e.g. "boughts" -> "bought" would if it existed -- most forms here
are irregular verbs so are spelled out explicitly instead); phrases are
lower-case substring matches, ported verbatim from
stamp_arg_mentions.cue_spans_in's phrase-matching loop.

PER-FACTOR FIELDS ADDED (every factor, beside the existing arg_spans/
cue_spans which are left untouched):
  role_cues: list of [char_start, char_end] role-cue spans.
    - GIVEN factors: WINDOWED -- only cues within 6 tokens before and 6
      after the factor's OWN value-numeral occurrence (its `spans` field
      if present, else the first occurrence in `mentions[str(var)]`,
      which is itself the leftmost occurrence -- the same convention
      args_census.find_numeral_sentence uses via re.search). This is
      the point of the window: a given's value numeral can recur
      elsewhere in the text (a neighbour's own quantity), and an
      unrestricted clause-wide cue scan would attach a cue that belongs
      to that neighbour instead.
    - All other factors (rel/sel/pct/mod/fdiv/macro): cues anywhere in
      the factor's own CLAUSE (stamp_arg_mentions.clause_of/windows_of,
      identical definition to the existing cue_spans field, just over
      the widened lexicon).
  role_order: the factor's 0-based index in the row's factor list (the
    positional order it already carries under THE POSITIONAL LAW --
    stamped here purely for the gold builder's convenience).
  arg_role_cues: relation factors only (has `args`) get a per-argument-
    position list: the entries of this factor's own role_cues that sit
    within 4 TOKENS of that position's re-mention span(s) (arg_spans[p],
    already stamped by stamp_arg_mentions -- an empty gap between a
    word token and the re-mention counts as 0 tokens between them);
    when arg_spans[p] is empty (no re-mention resolved), the whole
    factor's role_cues are used as the fallback. Non-relation factors
    get `arg_role_cues: []` for uniformity (a downstream reader never
    has to check `"arg_role_cues" in factor`).

Usage:
  .venv/bin/python3 scripts/stamp_given_cues.py
    (fixed IO: .cache/form_mix_pm35a.jsonl -> .cache/form_mix_pm35c.jsonl,
    plus the wild measurement fixture .cache/wild_admitted_holdout_a.jsonl
    -> .cache/wild_admitted_holdout_c.jsonl -- NEVER trained on, produced
    every run for parity with the diet.)
"""
import bisect
import json
import sys
from collections import Counter

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import args_census as AC          # noqa: E402  (reuse, never reimplement)
import stamp_arg_mentions as SAM  # noqa: E402
import lexical_identity_census as LIC  # noqa: E402
import twin_split_census as TSC   # noqa: E402

DIET_IN = ".cache/form_mix_pm35a.jsonl"
DIET_OUT = ".cache/form_mix_pm35c.jsonl"
WILD_IN = ".cache/wild_admitted_holdout_a.jsonl"
WILD_OUT = ".cache/wild_admitted_holdout_c.jsonl"
REPORT = ".cache/given_cues_report.txt"

TWIN_DIET_CAP = 2000  # matches twin_split_census.DIET_CAP -- same fixture, same bound

# =======================================================================
# THE ROLE-CUE LEXICON
# =======================================================================

ROLE_CUE_WORDS_NEW = """
has had have got gets bought buys received earned found made gave gives
sold spent lost ate eats used uses needs wants paid
left remaining remain rest still now total altogether together
first then next later after before initially originally finally every
per each
more fewer less than twice half double triple times
both all together combined
""".split()

ROLE_CUE_PHRASES_NEW = ["in all", "at first", "each day"]

ROLE_CUE_WORDS_RAW = sorted(set(AC.CUE_WORDS) | set(ROLE_CUE_WORDS_NEW))
ROLE_CUE_PHRASES = sorted(set(AC.CUE_PHRASES) | set(ROLE_CUE_PHRASES_NEW))
# stemmed exactly as stem() stems, so the matcher below (which stems each
# candidate token the same way) never has to special-case plural/verb forms
ROLE_CUE_WORDS = {SAM.stem(w) for w in ROLE_CUE_WORDS_RAW}


def role_cue_spans_in(text, windows):
    """Ported from stamp_arg_mentions.cue_spans_in (whole-word matching for
    single words, lower-case substring matching for phrases) with the
    widened role-cue lexicon and stemmed word comparison."""
    spans = set()
    for (s, e) in windows:
        seg = text[s:e]
        segl = seg.lower()
        for ph in ROLE_CUE_PHRASES:
            start = 0
            while True:
                pos = segl.find(ph, start)
                if pos == -1:
                    break
                spans.add((s + pos, s + pos + len(ph)))
                start = pos + 1
        for w, ws, we in SAM.tokenize(text, s, e):
            if SAM.stem(w.lower()) in ROLE_CUE_WORDS:
                spans.add((ws, we))
    return sorted(spans)


# =======================================================================
# GIVEN-factor windowing: 6 tokens before / 6 after the value numeral
# =======================================================================

def given_anchor_spans(fac, mentions, text):
    sp = fac.get("spans")
    if sp:
        return [tuple(x) for x in sp]
    var = fac.get("var")
    if var is not None:
        ms = mentions.get(str(var))
        if ms:
            return [tuple(ms[0])]  # leftmost occurrence -- matches
                                    # find_numeral_sentence's re.search convention
    # WILD FALLBACK: `mentions` is never stamped on the wild holdout
    # (stamp_value_spans.py only ever ran on the diet pipeline -- confirmed
    # empty {} on every wild row), so re-derive the leftmost standalone
    # numeral occurrence directly, via the IDENTICAL regex convention
    # args_census.find_numeral_sentence uses (ported here only for its span,
    # not its sentence index, which find_numeral_sentence already gives
    # clause_of elsewhere).
    val = fac.get("value")
    return first_numeral_span(text, val)


def first_numeral_span(text, value):
    if value is None:
        return []
    try:
        iv = int(value)
        if iv != value:
            return []
    except (TypeError, ValueError):
        return []
    import re
    m = re.search(r'(?<!\d)' + re.escape(str(abs(iv))) + r'(?!\d)', text)
    if not m:
        return []
    return [(m.start(), m.end())]


def token_window(a, b, toks, before_n=6, after_n=6):
    before = [t for t in toks if t[2] <= a]
    after = [t for t in toks if t[1] >= b]
    before = before[-before_n:] if before else []
    after = after[:after_n] if after else []
    win_s = before[0][1] if before else a
    win_e = after[-1][2] if after else b
    return (win_s, win_e)


def given_role_cues(fac, text, mentions, toks):
    anchors = given_anchor_spans(fac, mentions, text)
    if not anchors:
        return []
    windows = [token_window(a, b, toks) for (a, b) in anchors]
    return role_cue_spans_in(text, windows)


# =======================================================================
# token-distance (arg_role_cues' "within 4 tokens of the re-mention")
# =======================================================================

def tokens_between(a_s, a_e, b_s, b_e, starts):
    if a_e <= b_s:
        gap_s, gap_e = a_e, b_s
    elif b_e <= a_s:
        gap_s, gap_e = b_e, a_s
    else:
        return 0  # overlapping spans
    i = bisect.bisect_left(starts, gap_s)
    j = bisect.bisect_left(starts, gap_e)
    return j - i  # count of word tokens starting strictly inside the gap


def arg_role_cues_for(fac, rel_role_cues, starts):
    args = fac.get("args", [])
    arg_spans = fac.get("arg_spans") or [[] for _ in args]
    out = []
    for p in range(len(args)):
        mentions_p = arg_spans[p] if p < len(arg_spans) else []
        if not mentions_p:
            out.append([list(sp) for sp in rel_role_cues])
            continue
        near = set()
        for (cs, ce) in rel_role_cues:
            for (ms, me) in mentions_p:
                if tokens_between(cs, ce, ms, me, starts) <= 4:
                    near.add((cs, ce))
                    break
        out.append([list(sp) for sp in sorted(near)])
    return out


# =======================================================================
# per-row stamping
# =======================================================================

def stamp_row(row):
    text = row["text"]
    factors = row["factors"]
    solution = row.get("solution", [])
    mentions = row.get("mentions", {})

    bounds = AC.sentence_bounds(text)
    sspans = AC.sentence_spans(text, bounds)
    intro_map = SAM.build_intro_map(factors)
    memo = {}
    toks = SAM.tokenize(text, 0, len(text))
    starts = [t[1] for t in toks]

    for idx, fac in enumerate(factors):
        fac["role_order"] = idx
        if fac["ftype"] == "given":
            rc = given_role_cues(fac, text, mentions, toks)
            fac["role_cues"] = [list(sp) for sp in rc]
            fac["arg_role_cues"] = []
        else:
            clause = SAM.clause_of(idx, factors, text, bounds, sspans,
                                    solution, intro_map, memo)
            windows = SAM.windows_of(clause, sspans)
            rc = role_cue_spans_in(text, windows)
            fac["role_cues"] = [list(sp) for sp in rc]
            if "args" in fac:
                fac["arg_role_cues"] = arg_role_cues_for(fac, rc, starts)
            else:
                fac["arg_role_cues"] = []
    return factors


# =======================================================================
# coverage / distribution stats
# =======================================================================

def new_stats():
    return dict(n_given=0, n_given_with_cue=0,
                n_rel=0, n_rel_with_cue=0,
                given_cue_words=Counter(), rel_cue_words=Counter())


def update_stats(st, text, factors):
    for fac in factors:
        rc = fac["role_cues"]
        if fac["ftype"] == "given":
            st["n_given"] += 1
            if rc:
                st["n_given_with_cue"] += 1
            for s, e in rc:
                st["given_cue_words"][text[s:e].lower()] += 1
        elif "args" in fac:
            st["n_rel"] += 1
            if rc:
                st["n_rel_with_cue"] += 1
            for s, e in rc:
                st["rel_cue_words"][text[s:e].lower()] += 1


# =======================================================================
# THE SAME-NOUN TWIN DISCRIMINATOR CENSUS (reuses twin_split_census /
# lexical_identity_census verbatim for the classification; this script
# only asks whether role_cues differs across an already-identified pair)
# =======================================================================

def cue_word_set(fac, text):
    return {text[s:e].lower() for s, e in fac.get("role_cues", [])}


def twin_pairs_for_row(row):
    """Every (k, c) pair where c is a same-sentence competitor of k
    (the introducing slot of some relation argument) sharing k's
    one-token head-noun key -- i.e. an actual SAME-NOUN twin, per
    twin_split_census.classify()'s SAME-NOUN branch."""
    (text, factors, solution, bounds, sspans, intro_map, memo,
     cand_key, cand_sent) = LIC.build_candidate_tables(row)
    pairs = []
    for j, fac in enumerate(factors):
        args = fac.get("args")
        if not args:
            continue
        for pos, v in enumerate(args):
            k = intro_map.get(v)
            if k is None or k == j:
                continue
            k_sent = cand_sent.get(k, set())
            competitors = [c for c in range(len(factors))
                           if c != j and c != k and
                           LIC.same_sentence(cand_sent.get(c, set()), k_sent)]
            if not competitors:
                continue
            key_k = cand_key.get(k)
            if key_k is None:
                continue
            for c in competitors:
                if cand_key.get(c) == key_k:
                    pairs.append((k, c))
    return pairs, factors, text


def twin_discriminator_census(rows):
    counts = Counter()
    examples = {"different": [], "same": [], "none": []}
    for row in rows:
        pairs, factors, text = twin_pairs_for_row(row)
        for (k, c) in pairs:
            k_cues = cue_word_set(factors[k], text)
            c_cues = cue_word_set(factors[c], text)
            if not k_cues and not c_cues:
                cls = "none"
            elif k_cues == c_cues:
                cls = "same"
            else:
                cls = "different"
            counts[cls] += 1
            if len(examples[cls]) < 5:
                examples[cls].append((k, c, sorted(k_cues), sorted(c_cues), text[:160]))
    return counts, examples


def load_prose_cap(path, cap):
    out = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "src" in r.get("gen", {}):
                out.append(r)
            if len(out) >= cap:
                break
    return out


# =======================================================================
# identity check
# =======================================================================

NEW_KEYS = ("role_cues", "role_order", "arg_role_cues")


def identity_check(in_path, out_path):
    n_rows = n_fac_mismatch = n_text_mismatch = n_mentions_mismatch = n_other_mismatch = 0
    with open(in_path) as fin, open(out_path) as fout:
        for li_line, lo_line in zip(fin, fout):
            ri = json.loads(li_line)
            ro = json.loads(lo_line)
            n_rows += 1
            if ri["text"] != ro["text"]:
                n_text_mismatch += 1
            if ri.get("mentions", {}) != ro.get("mentions", {}):
                n_mentions_mismatch += 1
            for k in ri:
                if k == "factors":
                    continue
                if ri[k] != ro.get(k):
                    n_other_mismatch += 1
            fi_list = ri["factors"]
            fo_list = ro["factors"]
            if len(fi_list) != len(fo_list):
                n_fac_mismatch += 1
                continue
            for fi, fo in zip(fi_list, fo_list):
                fo_stripped = {k: v for k, v in fo.items() if k not in NEW_KEYS}
                if fo_stripped != fi:
                    n_fac_mismatch += 1
    return dict(n_rows=n_rows, n_fac_mismatch=n_fac_mismatch,
                n_text_mismatch=n_text_mismatch,
                n_mentions_mismatch=n_mentions_mismatch,
                n_other_mismatch=n_other_mismatch)


# =======================================================================
# spot check rendering
# =======================================================================

def mark_row(text, factors):
    """De-duplicates identical spans across factors first (the same cue span
    is often reachable from more than one factor's window/clause), then
    greedily selects a NON-OVERLAPPING left-to-right subset (a cue span can
    exactly coincide with a given's own lexicon-word value span, e.g. "half"
    -- both are real, but only one renders here) so the bracket rendering
    never nests or corrupts."""
    cue_spans = set()
    v_spans = {}
    for fac in factors:
        for (s, e) in fac["role_cues"]:
            cue_spans.add((s, e))
        if fac["ftype"] == "given":
            for (s, e) in fac.get("spans") or []:
                v_spans.setdefault((s, e), fac["var"])
    for k in list(v_spans):
        if k in cue_spans:
            del v_spans[k]
    all_spans = [(s, e, "CUE") for (s, e) in cue_spans]
    all_spans += [(s, e, f"V{var}") for (s, e), var in v_spans.items()]
    all_spans.sort(key=lambda t: (t[0], t[1]))
    selected = []
    last_end = -1
    for s, e, tag in all_spans:
        if s >= last_end:
            selected.append((s, e, tag))
            last_end = e
    out = []
    prev = 0
    for s, e, tag in selected:
        out.append(text[prev:s])
        out.append(f"[{tag}:")
        out.append(text[s:e])
        out.append(f":{tag}]")
        prev = e
    out.append(text[prev:])
    return "".join(out)


# =======================================================================
# main pass over one file
# =======================================================================

def stamp_file(in_path, out_path, stats):
    out_rows = []
    spot = []
    with open(in_path) as f:
        for row in map(json.loads, f):
            factors = stamp_row(row)
            is_prose = "src" in row.get("gen", {})
            register = "prose" if is_prose else "mint"
            update_stats(stats[register], row["text"], factors)
            if is_prose and len(spot) < 40:
                spot.append((row["text"], factors))
            out_rows.append(row)
    with open(out_path, "w") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")
    return spot


def main():
    lines = []

    def P(s=""):
        print(s)
        lines.append(s)

    P("=" * 78)
    P("THE ROLE SIGNATURE -- STAMP_GIVEN_CUES (2026-09-22)")
    P("=" * 78)

    P("")
    P("THE ROLE-CUE LEXICON")
    P("-" * 78)
    P(f"single words ({len(ROLE_CUE_WORDS_RAW)}, union of args_census.CUE_WORDS "
      "and the new temporal/possessive/state list):")
    P("  " + ", ".join(ROLE_CUE_WORDS_RAW))
    P(f"phrases ({len(ROLE_CUE_PHRASES)}, union of args_census.CUE_PHRASES and the new list):")
    P("  " + ", ".join(ROLE_CUE_PHRASES))
    P(f"(matched stemmed -- {len(ROLE_CUE_WORDS)} distinct stems)")

    stats = {"prose": new_stats(), "mint": new_stats()}
    P("")
    P(f"Stamping {DIET_IN} -> {DIET_OUT} ...")
    spot = stamp_file(DIET_IN, DIET_OUT, stats)
    P("done.")

    wild_stats = {"prose": new_stats(), "mint": new_stats()}  # wild rows read as "prose" register
    P(f"Stamping {WILD_IN} -> {WILD_OUT} (measurement fixture, never trained on) ...")
    wild_spot = stamp_file(WILD_IN, WILD_OUT, wild_stats)
    P("done.")

    P("")
    P("-" * 78)
    P("COVERAGE -- GIVEN factors with >=1 role cue")
    P("-" * 78)
    for name, st in (("diet PROSE", stats["prose"]), ("diet MINT", stats["mint"]),
                      ("wild (all prose-register)", wild_stats["prose"])):
        n = st["n_given"]
        c = st["n_given_with_cue"]
        P(f"  {name:26s} n_given={n:7d}  with_cue={c:7d}  coverage={c / max(n,1):.3f}")
    P("")
    P("COVERAGE -- RELATION-type factors (has 'args') with >=1 role cue")
    for name, st in (("diet PROSE", stats["prose"]), ("diet MINT", stats["mint"]),
                      ("wild (all prose-register)", wild_stats["prose"])):
        n = st["n_rel"]
        c = st["n_rel_with_cue"]
        P(f"  {name:26s} n_rel={n:7d}  with_cue={c:7d}  coverage={c / max(n,1):.3f}")

    P("")
    P("-" * 78)
    P("TOP 25 CUE WORDS/PHRASES ON GIVENS (diet + wild combined)")
    P("-" * 78)
    given_words = Counter()
    for st in (stats["prose"], stats["mint"], wild_stats["prose"]):
        given_words += st["given_cue_words"]
    tot_given = sum(given_words.values())
    for w, n in given_words.most_common(25):
        P(f"  {w:16s} n={n:6d}  ({n / max(tot_given,1):.3f})")

    P("")
    P("-" * 78)
    P("TOP 25 CUE WORDS/PHRASES ON RELATIONS (diet + wild combined)")
    P("-" * 78)
    rel_words = Counter()
    for st in (stats["prose"], stats["mint"], wild_stats["prose"]):
        rel_words += st["rel_cue_words"]
    tot_rel = sum(rel_words.values())
    for w, n in rel_words.most_common(25):
        P(f"  {w:16s} n={n:6d}  ({n / max(tot_rel,1):.3f})")

    # -------------------------------------------------------------
    # THE DISCRIMINATOR CENSUS
    # -------------------------------------------------------------
    P("")
    P("=" * 78)
    P("THE SAME-NOUN TWIN DISCRIMINATOR CENSUS")
    P("=" * 78)
    P("Every same-sentence SAME-NOUN twin pair (k, c) -- the introducing slot k of a")
    P("relation argument and a same-sentence competitor c sharing k's one-token")
    P("head-noun key (twin_split_census.classify()'s SAME-NOUN branch) -- classified")
    P("by whether their stamped role_cues (the set of matched cue words/phrases in")
    P("each factor's own clause/window) are DIFFERENT (the discriminator exists),")
    P("SAME (no discriminator: both carry the identical cue set), or NONE (neither")
    P(f"carries any role cue). Diet: first {TWIN_DIET_CAP} prose rows of {DIET_OUT}")
    P("(freshly stamped, matching twin_split_census.load_rows_diet's cap/filter).")
    P("")

    diet_rows_for_twin = load_prose_cap(DIET_OUT, TWIN_DIET_CAP)
    counts, examples = twin_discriminator_census(diet_rows_for_twin)
    n_pairs = sum(counts.values())
    P(f"diet prose (first {len(diet_rows_for_twin)} prose rows): {n_pairs} same-noun twin pairs")
    for cls in ("different", "same", "none"):
        n = counts.get(cls, 0)
        P(f"  {cls:10s} n={n:6d}  ({n / max(n_pairs,1):.3f})")
    P("")

    wild_rows_all = [json.loads(l) for l in open(WILD_OUT)]
    counts_w, examples_w = twin_discriminator_census(wild_rows_all)
    n_pairs_w = sum(counts_w.values())
    P(f"wild ({len(wild_rows_all)} rows): {n_pairs_w} same-noun twin pairs")
    for cls in ("different", "same", "none"):
        n = counts_w.get(cls, 0)
        P(f"  {cls:10s} n={n:6d}  ({n / max(n_pairs_w,1):.3f})")
    P("")
    P("THE NUMBER (diet+wild combined): fraction of same-noun twin pairs where the")
    P("cue DIFFERS -- the role signature's ceiling as a discriminator:")
    counts_all = counts + counts_w
    n_all = sum(counts_all.values())
    P(f"  different={counts_all.get('different',0)}  same={counts_all.get('same',0)}  "
      f"none={counts_all.get('none',0)}  total={n_all}")
    if n_all:
        P(f"  DIFFERENT share: {counts_all.get('different',0) / n_all:.3f}")

    P("")
    P("examples (diet), up to 5 per class:")
    for cls in ("different", "same", "none"):
        P(f"  [{cls}]")
        for (k, c, kc, cc, snippet) in examples.get(cls, []):
            P(f"    k={k} cues={kc}  vs  c={c} cues={cc}")
            P(f"      {snippet!r}")
        if not examples.get(cls):
            P("    (none)")

    # -------------------------------------------------------------
    # identity check
    # -------------------------------------------------------------
    P("")
    P("-" * 78)
    P("IDENTITY CHECK")
    P("-" * 78)
    for name, ip, op in (("diet", DIET_IN, DIET_OUT), ("wild", WILD_IN, WILD_OUT)):
        chk = identity_check(ip, op)
        P(f"  {name}: rows={chk['n_rows']}  fac_mismatch={chk['n_fac_mismatch']}  "
          f"text_mismatch={chk['n_text_mismatch']}  mentions_mismatch={chk['n_mentions_mismatch']}  "
          f"other_mismatch={chk['n_other_mismatch']}  " +
          ("PASS" if chk['n_fac_mismatch'] == chk['n_text_mismatch'] ==
           chk['n_mentions_mismatch'] == chk['n_other_mismatch'] == 0 else "FAIL"))

    # -------------------------------------------------------------
    # 12 example rows, cues bracketed beside the numerals
    # -------------------------------------------------------------
    P("")
    P("=" * 78)
    P("12 EXAMPLE ROWS (cues bracketed [CUE:...:CUE], given values bracketed [Vn:...:Vn])")
    P("=" * 78)
    shown = 0
    for text, factors in spot:
        if not any(fac["role_cues"] for fac in factors):
            continue
        P(f"\n{mark_row(text, factors)}")
        shown += 1
        if shown >= 12:
            break

    P("")
    P("=" * 78)
    P("done.")

    with open(REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[stamp_given_cues] wrote {REPORT}")


if __name__ == "__main__":
    main()
