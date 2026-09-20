"""stamp_arg_mentions.py -- THE ARG-MENTION STAMP (2026-09-20, zero-token,
zero-GPU, CPU-only): the census (scripts/args_census.py, ledger 2026-09-20
00:20 "THE ARGUMENT-BINDING CENSUS") found that 91.6% (wild) / 91.9%
(diet, real spans) of relation arguments share a lexical anchor with their
introducing factor's clause, and that the args wall is worst at SAME-
SENTENCE distance -- exactly where every existing surface road is blind
(sentence granularity only). This script stamps the actual char spans of
that lexical anchor -- the argument's MENTION inside the relation's own
clause -- so a future arm can supervise the args pointer with a real span
target instead of nothing.

Definitions REUSED VERBATIM from scripts/args_census.py (stopword list, cue
word/phrase list, pronoun set, sentence-boundary convention, the ". "-split
sentence walk, has_cue(), find_numeral_sentence()): imported, not
reimplemented, so the two scripts never drift.

THE INTRODUCING FACTOR of variable v = the factor whose own `var` (given)
or `result` (rel/sel/mod/fdiv/macro) equals v -- built once per row as
`intro_map`, scanned in list order (THE POSITIONAL LAW: unique per row).
`pct` factors are the one exception in this dialect: their `args[0]` IS
their own result (verified 364/364 in the diet -- no other factor ever
introduces a pct's args[0]), so pct registers itself as the introducer of
args[0] too (letting a LATER relation that references a pct's output find
its clause), and pct's own arg-mention pass always skips position 0 as a
self-reference (empty span, not "neither" -- there is nothing to point at).

A factor's CLAUSE (used both when it is the relation being stamped and
when it is another factor's introducer):
  - its own `spans` field if present and non-empty (the diet's real
    annotation, ~47% of relation factors, ~86% of given factors);
  - else a HEURISTIC, ported from args_census.clause_sentence_wild but
    reading values off the row's own `solution` array (solution[v] = the
    graph-evaluated value of variable v -- already computed at generation
    time, so no re-implementation of forward add/mul evaluation is needed
    here) instead of the census's dense gold arrays:
      given      -> the sentence containing its |value|'s literal numeral;
      rel/sel    -> the sentence containing its own result's literal
                    numeral if present, else the nearest operator-cue
                    sentence at or after the max of its two arguments'
                    (recursively resolved) clause sentences;
      pct        -> same rule, using args[1] (the base) as its one
                    recursion argument (args[0] excluded: self);
      mod/fdiv   -> same rule, recursing through their single `var`;
      macro      -> same rule, recursing through `a` and `x`.
  A clause with no numeral, no cue sentence and no resolvable recursion is
  UNRESOLVED (None) -- most often a given whose value is spelled only as a
  lexicon word (stamp_value_spans.py's lexicon path; this heuristic is the
  literal-numeral-only census rule, not extended for lexicon words, since
  the census being reused does not either) or a relation whose result
  numeral was itself simplified away and which sits after every cue.

ARG-MENTION (per argument position, RELATION factors only -- any factor
carrying an `args` key: rel, sel, pct):
  1. content_words(introducing clause) & content_words(relation's clause)
     (case-insensitive, trailing 's'/'es' stemmed, stopwords/cue words/
     numerals excluded, len>2) -- if any stem recurs, every OCCURRENCE of
     a recurring content word INSIDE the relation's clause is the mention
     (there can be more than one span -- e.g. a word repeated twice in a
     multi-sentence relation clause).
  2. SAME-SENTENCE RESTRICTION: when the relation's clause and the
     introducing clause share at least one sentence index, any candidate
     span that falls inside the introducing factor's own VALUE span
     (`mentions[str(v)]`, already stamped by stamp_value_spans.py -- the
     numeral/lexicon-word occurrence(s) of the introducing GIVEN's value;
     empty for a relation-type introducer, so the restriction is a no-op
     there) is dropped, so the stamp points at the RE-mention, not the
     introduction. Counted separately: how often this empties an
     otherwise non-empty match list.
  3. If step 1-2 leaves nothing and the relation's clause contains a
     pronoun (he/she/they/it/his/her/their/them/him), every pronoun token
     in the relation's clause is the mention.
  4. Otherwise the mention is empty ("neither", or "unresolved" if the
     introducing clause itself could not be located, or "no_introducer"
     for a pct's self-referencing position 0).

CUE_SPANS: every occurrence of a census CUE_WORD or CUE_PHRASE inside the
relation's own clause (whole-word for single words, substring for
phrases); empty if none. Stamped on every relation factor regardless of
whether any arg resolved.

FIELD FORMAT (see the report for the exact spec quoted back): every factor
in the output gets two new keys, `arg_spans` and `cue_spans`.
  - Relation factors (has `args`): `arg_spans` is a list with ONE entry
    per argument position (same length/order as `args`); each entry is a
    list of [char_start, char_end) pairs, absolute offsets into `text`,
    possibly empty. `cue_spans` is a flat list of [char_start, char_end)
    pairs for the whole clause, possibly empty.
  - All OTHER factors (given, fdiv, mod, macro): `arg_spans: []` and
    `cue_spans: []` -- present on every factor for uniformity (a
    downstream reader never has to check `"arg_spans" in factor`), not
    just given factors.
No other field is touched; `text` and `mentions` are carried through
byte-identical; factor list order is never changed (THE POSITIONAL LAW).

Usage:
  .venv/bin/python3 scripts/stamp_arg_mentions.py [IN] [OUT]
    (defaults: .cache/form_mix_pm35v.jsonl -> .cache/form_mix_pm35a.jsonl)

The wild holdout (measurement fixture only, see docstring at the bottom of
main()) is stamped by a second invocation:
  .venv/bin/python3 scripts/stamp_arg_mentions.py \
      .cache/wild_admitted_holdout.jsonl .cache/wild_admitted_holdout_a.jsonl
"""
import json
import re
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import args_census as census  # noqa: E402  (reuse, never reimplement)

STOPWORDS = census.STOPWORDS
CUE_WORDS = census.CUE_WORDS
CUE_PHRASES = census.CUE_PHRASES
PRONOUNS = census.PRONOUNS
sentence_bounds = census.sentence_bounds
sentence_spans = census.sentence_spans
sent_of_char = census.sent_of_char
find_numeral_sentence = census.find_numeral_sentence
has_cue = census.has_cue

IN = sys.argv[1] if len(sys.argv) > 1 else ".cache/form_mix_pm35v.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else ".cache/form_mix_pm35a.jsonl"

WORD_RE = re.compile(r"[A-Za-z']+")


def stem(w):
    """Simple stemming per spec: strip a trailing 's'/'es'."""
    if w.endswith("es") and len(w) > 3:
        return w[:-2]
    if w.endswith("s") and len(w) > 2:
        return w[:-1]
    return w


def is_content(wl):
    return wl not in STOPWORDS and wl not in CUE_WORDS and len(wl) > 2


def tokenize(text, s, e):
    return [(m.group(), m.start(), m.end()) for m in WORD_RE.finditer(text, s, e)]


# =======================================================================
# introducing-factor map, own value, recursion args (jsonl-native; the
# census's dense-array row_eval is not reused here because `solution`
# already carries every variable's graph-evaluated value)
# =======================================================================

def build_intro_map(factors):
    m = {}
    for idx, fac in enumerate(factors):
        ft = fac["ftype"]
        if ft == "given":
            m.setdefault(fac["var"], idx)
        elif "result" in fac:
            m.setdefault(fac["result"], idx)
        elif ft == "pct":
            a = fac.get("args") or []
            if a:
                m.setdefault(a[0], idx)
    return m


def own_var(fac):
    ft = fac["ftype"]
    if ft == "given":
        return fac["var"]
    if "result" in fac:
        return fac["result"]
    if ft == "pct":
        a = fac.get("args") or []
        return a[0] if a else None
    return None


def own_value(fac, solution):
    v = own_var(fac)
    if v is None or v >= len(solution):
        return None
    return solution[v]


def recursion_args(fac):
    ft = fac["ftype"]
    if ft in ("rel", "sel"):
        return fac.get("args", [])
    if ft == "pct":
        return (fac.get("args") or [])[1:]
    if ft in ("mod", "fdiv"):
        return [fac["var"]] if "var" in fac else []
    if ft == "macro":
        return [fac[k] for k in ("a", "x") if k in fac]
    return []


# =======================================================================
# clause resolution (ports args_census.clause_sentence_wild to the jsonl
# list-of-factors representation; adds the real-`spans` branch)
# =======================================================================

def clause_of(idx, factors, text, bounds, sspans, solution, intro_map, memo):
    if idx in memo:
        return memo[idx]
    memo[idx] = None  # recursion guard (pathological cycles)
    fac = factors[idx]
    spans = fac.get("spans")
    if spans:
        result = ("spans", [tuple(sp) for sp in spans])
        memo[idx] = result
        return result
    val = own_value(fac, solution)
    s = find_numeral_sentence(text, bounds, abs(val)) if val is not None else None
    if s is not None:
        result = ("sent", s)
        memo[idx] = result
        return result
    if fac["ftype"] == "given":
        memo[idx] = None
        return None
    arg_sents = []
    for a in recursion_args(fac):
        k = intro_map.get(a)
        if k is not None and k != idx:
            r = clause_of(k, factors, text, bounds, sspans, solution, intro_map, memo)
            sset = sent_set_of(r, bounds, sspans)
            if sset:
                arg_sents.append(max(sset))
    base = max(arg_sents) if arg_sents else 0
    cue_sents = [k for k, (a, b) in enumerate(sspans) if has_cue(text[a:b])]
    fwd = [c for c in cue_sents if c >= base]
    s = min(fwd) if fwd else (max(cue_sents) if cue_sents else None)
    result = ("sent", s) if s is not None else None
    memo[idx] = result
    return result


def sent_set_of(clause, bounds, sspans):
    if clause is None:
        return set()
    kind, val = clause
    if kind == "sent":
        return {val} if 0 <= val < len(sspans) else set()
    out = set()
    for (s, e) in val:
        out.add(sent_of_char(bounds, s))
        out.add(sent_of_char(bounds, max(s, e - 1)))
    return out


def windows_of(clause, sspans):
    if clause is None:
        return []
    kind, val = clause
    if kind == "sent":
        return [sspans[val]] if 0 <= val < len(sspans) else []
    return [list(sp) for sp in val]


# =======================================================================
# lexical matching within a clause window
# =======================================================================

def content_tokens_in(text, windows):
    out = []
    for (s, e) in windows:
        for w, ws, we in tokenize(text, s, e):
            wl = w.lower()
            if is_content(wl):
                out.append((stem(wl), ws, we))
    return out


def content_stem_set(text, windows):
    return {stm for (stm, _, _) in content_tokens_in(text, windows)}


# THE LETTER-NAME MAP (mint's "letters register" only -- gated behind
# `not is_prose` at every call site so prose rows are byte-for-byte
# unaffected regardless of this regex's behavior): mint's gold builder
# writes an explicit naming preamble ("Let a, b, c, ... be whole numbers.",
# "Consider the numbers ...", "The following facts hold about ...", and
# scrambled-order variants of the same) whose comma-separated single-letter
# list assigns variable id v -> its v-th listed letter (verified against
# `mentions`' own letter-occurrence spans on rows where both are present).
# A general, phrase-agnostic pattern is used instead of anchoring on those
# three lead-ins, so any future lead-in phrase is still caught: the first
# run of >=2 comma-separated single-letter tokens anywhere in the text.
LETTER_LIST_RE = re.compile(r"\b([a-zA-Z](?:\s*,\s*[a-zA-Z])+)\b")


def build_letter_map(text):
    m = LETTER_LIST_RE.search(text)
    if not m:
        return {}
    letters = [tok.strip() for tok in m.group(1).split(",")]
    return {idx: ch for idx, ch in enumerate(letters)}


def standalone_letter_spans(text, windows, letter):
    """Case-sensitive, word-boundary occurrences of a single letter token
    (never inside another word: blocked by a letter on either side, but a
    digit/punctuation neighbor is fine -- 'a.' or '(a)' or 'a,' all count)."""
    if not letter:
        return []
    pat = re.compile(r"(?<![A-Za-z])" + re.escape(letter) + r"(?![A-Za-z])")
    out = []
    for (s, e) in windows:
        for m in pat.finditer(text, s, e):
            out.append((m.start(), m.end()))
    return out


def pronoun_spans_in(text, windows):
    out = []
    for (s, e) in windows:
        for w, ws, we in tokenize(text, s, e):
            if w.lower() in PRONOUNS:
                out.append([ws, we])
    return out


def cue_spans_in(text, windows):
    spans = set()
    for (s, e) in windows:
        seg = text[s:e]
        segl = seg.lower()
        for ph in CUE_PHRASES:
            start = 0
            while True:
                pos = segl.find(ph, start)
                if pos == -1:
                    break
                spans.add((s + pos, s + pos + len(ph)))
                start = pos + 1
        for w, ws, we in tokenize(text, s, e):
            if w.lower() in CUE_WORDS:
                spans.add((ws, we))
    return [list(sp) for sp in sorted(spans)]


def overlaps(span, ranges):
    s, e = span
    for (a, b) in ranges:
        if s < b and a < e:
            return True
    return False


def bucket_dist(d):
    if d is None:
        return "unresolved"
    return "3+" if d >= 3 else str(d)


# =======================================================================
# per-relation-factor stamping
# =======================================================================

def process_relation(i, fac, factors, text, bounds, sspans, solution,
                      intro_map, mentions, memo, letter_of):
    args = fac.get("args", [])
    rel_clause = clause_of(i, factors, text, bounds, sspans, solution, intro_map, memo)
    rel_windows = windows_of(rel_clause, sspans)
    rel_sent_set = sent_set_of(rel_clause, bounds, sspans)
    rel_content = content_tokens_in(text, rel_windows)
    rel_pronoun_spans = pronoun_spans_in(text, rel_windows)
    cue_spans = cue_spans_in(text, rel_windows)

    arg_spans = []
    infos = []
    for p, v in enumerate(args):
        k = intro_map.get(v)
        if k is None or k == i:
            arg_spans.append([])
            infos.append(dict(cls="no_introducer", dist=None, restricted_emptied=False))
            continue
        intro_clause = clause_of(k, factors, text, bounds, sspans, solution, intro_map, memo)
        intro_windows = windows_of(intro_clause, sspans)
        intro_sent_set = sent_set_of(intro_clause, bounds, sspans)
        intro_stems = content_stem_set(text, intro_windows) if intro_clause is not None else set()

        matches = sorted({(s, e) for (stm, s, e) in rel_content if stm in intro_stems})
        same_sent = bool(rel_sent_set and intro_sent_set and (rel_sent_set & intro_sent_set))
        restricted_emptied = False
        if same_sent and matches:
            value_ranges = mentions.get(str(v), [])
            filtered = [sp for sp in matches if not overlaps(sp, value_ranges)]
            if not filtered:
                restricted_emptied = True
            matches = filtered

        dist = None
        if rel_sent_set and intro_sent_set:
            dist = min(abs(rs - is_) for rs in rel_sent_set for is_ in intro_sent_set)

        # THE LETTER FALLBACK (mint's letters register): only reached when
        # the content-word rule found nothing (words of len==1 are never
        # content words) -- gated on letter_of being non-empty, which is
        # itself gated on `not is_prose` by the caller, so prose rows never
        # take this branch. The introducing factor's OWN clause must
        # literally contain the candidate letter as a standalone token
        # (confirms the name is actually tied to this factor here, not
        # just assumed from the row-level map) before it is used to search
        # the relation's clause.
        letter_spans = []
        if not matches and letter_of:
            letter = letter_of.get(v)
            if letter and standalone_letter_spans(text, intro_windows, letter):
                cand = standalone_letter_spans(text, rel_windows, letter)
                if same_sent:
                    intro_letter_occ = standalone_letter_spans(text, intro_windows, letter)
                    cand = [sp for sp in cand if not overlaps(sp, intro_letter_occ)]
                letter_spans = cand

        if matches:
            arg_spans.append([[s, e] for (s, e) in matches])
            infos.append(dict(cls="content", dist=dist, restricted_emptied=restricted_emptied))
        elif letter_spans:
            arg_spans.append([list(sp) for sp in sorted(set(letter_spans))])
            infos.append(dict(cls="letter", dist=dist, restricted_emptied=restricted_emptied))
        elif rel_pronoun_spans:
            arg_spans.append([list(sp) for sp in rel_pronoun_spans])
            infos.append(dict(cls="pronoun", dist=dist, restricted_emptied=restricted_emptied))
        else:
            cls = "neither" if intro_clause is not None else "unresolved"
            arg_spans.append([])
            infos.append(dict(cls=cls, dist=dist, restricted_emptied=restricted_emptied))
    return arg_spans, cue_spans, infos


# =======================================================================
# main stamping pass
# =======================================================================

def stamp_file(in_path, out_path, stats):
    with open(in_path) as f:
        lines = f.readlines()
    out_rows = []
    spot_prose = []
    spot_mint = []
    for li, line in enumerate(lines):
        row = json.loads(line)
        text = row["text"]
        factors = row["factors"]
        solution = row.get("solution", [])
        mentions = row.get("mentions", {})
        is_prose = "src" in row.get("gen", {})
        register = "prose" if is_prose else "mint"

        bounds = sentence_bounds(text)
        sspans = sentence_spans(text, bounds)
        intro_map = build_intro_map(factors)
        memo = {}
        # THE LETTER MAP is built ONLY for non-prose rows -- guarantees
        # prose output/stats are byte-identical to the pre-letter-rule run
        # regardless of the regex's behavior on prose text.
        letter_of = build_letter_map(text) if not is_prose else {}

        row_spot = []
        row_has_letter = False
        for i, fac in enumerate(factors):
            if "args" in fac:
                arg_spans, cue_spans, infos = process_relation(
                    i, fac, factors, text, bounds, sspans, solution,
                    intro_map, mentions, memo, letter_of)
                fac["arg_spans"] = arg_spans
                fac["cue_spans"] = cue_spans
                st = stats[register]
                st["n_rel"] += 1
                if cue_spans:
                    st["n_rel_with_cue"] += 1
                for p, (a, info) in enumerate(zip(arg_spans, infos)):
                    st["n_args"] += 1
                    st["by_cls"][info["cls"]] += 1
                    st["by_dist"][bucket_dist(info["dist"])] += 1
                    st["by_cls_dist"][(info["cls"], bucket_dist(info["dist"]))] += 1
                    if info["restricted_emptied"]:
                        st["n_restricted_emptied"] += 1
                    if info["cls"] in ("content", "letter", "pronoun"):
                        for (s, e) in a:
                            st["mention_chars"].append(e - s)
                            st["mention_words"].append(len(text[s:e].split()))
                    if info["cls"] == "letter":
                        row_has_letter = True
                row_spot.append((i, fac, arg_spans, cue_spans, infos))
            else:
                fac["arg_spans"] = []
                fac["cue_spans"] = []
        if is_prose and row_spot and len(spot_prose) < 20:
            spot_prose.append((li, text, row_spot))
        if (not is_prose) and row_has_letter and len(spot_mint) < 20:
            spot_mint.append((li, text, row_spot))
        out_rows.append(row)

    with open(out_path, "w") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")
    return spot_prose, spot_mint


def new_stats():
    from collections import Counter
    return dict(n_rel=0, n_args=0, n_rel_with_cue=0, n_restricted_emptied=0,
                by_cls=Counter(), by_dist=Counter(), by_cls_dist=Counter(),
                mention_chars=[], mention_words=[])


def print_stats(P, name, st):
    import numpy as np
    P(f"-- {name} --")
    P(f"  relation factors (has 'args'): {st['n_rel']}")
    P(f"  argument instances: {st['n_args']}")
    P(f"  relations with >=1 cue span: {st['n_rel_with_cue']} "
      f"({st['n_rel_with_cue'] / max(st['n_rel'], 1):.3f})")
    P("")
    P("  by class:")
    n = max(st["n_args"], 1)
    for c in ("content", "letter", "pronoun", "neither", "unresolved", "no_introducer"):
        v = st["by_cls"].get(c, 0)
        P(f"    {c:14s} n={v:7d}  ({v / n:.3f})")
    P("")
    P("  by sentence distance:")
    for b in ("0", "1", "2", "3+", "unresolved"):
        v = st["by_dist"].get(b, 0)
        P(f"    dist={b:11s} n={v:7d}  ({v / n:.3f})")
    P("")
    P("  class x distance:")
    P(f"  {'class':14s} {'d0':>8s} {'d1':>8s} {'d2':>8s} {'d3+':>8s} {'unres':>8s}")
    for c in ("content", "letter", "pronoun", "neither"):
        row = []
        for b in ("0", "1", "2", "3+", "unresolved"):
            row.append(st["by_cls_dist"].get((c, b), 0))
        P(f"  {c:14s} " + " ".join(f"{v:8d}" for v in row))
    P("")
    n_matched_before_restriction = st["by_cls"].get("content", 0) + st["n_restricted_emptied"]
    P(f"  same-sentence restriction emptied an otherwise non-empty match list: "
      f"{st['n_restricted_emptied']} times "
      f"(of {n_matched_before_restriction} same-sentence lexical matches before the "
      f"restriction, {st['n_restricted_emptied'] / max(n_matched_before_restriction, 1):.3f})")
    if st["mention_chars"]:
        P(f"  mean mention length: {np.mean(st['mention_chars']):.2f} chars, "
          f"{np.mean(st['mention_words']):.2f} words  (n={len(st['mention_chars'])} spans)")
    else:
        P("  mean mention length: n/a (no non-empty mentions)")
    P("")


def mark_spans(text, args, arg_spans, cue_spans):
    """Overlap-safe bracket rendering for the spot check: a sweep-line pass
    (never mutates positions by inserting into a growing list, so spans
    that share a start/end -- e.g. two argument positions that both fall
    back to the SAME pronoun occurrences -- render without corruption)."""
    events = []  # (pos, priority, text) -- priority 0 = closes, 1 = opens,
                 # so a close at position p is emitted before an open at p
    for c_s, c_e in cue_spans:
        events.append((c_s, 1, "[CUE:"))
        events.append((c_e, 0, ":CUE]"))
    for p, spans in enumerate(arg_spans):
        v = args[p]
        for (a_s, a_e) in spans:
            events.append((a_s, 1, f"[ARG{v}:"))
            events.append((a_e, 0, f":ARG{v}]"))
    events.sort(key=lambda t: (t[0], t[1]))
    out = []
    prev = 0
    for pos, _, tag in events:
        out.append(text[prev:pos])
        out.append(tag)
        prev = pos
    out.append(text[prev:])
    return "".join(out)


def identity_check(in_path, out_path):
    n_rows = 0
    n_fac_mismatch = 0
    n_text_mismatch = 0
    n_mentions_mismatch = 0
    n_other_mismatch = 0
    with open(in_path) as fin, open(out_path) as fout:
        for li, (li_line, lo_line) in enumerate(zip(fin, fout)):
            ri = json.loads(li_line)
            ro = json.loads(lo_line)
            n_rows += 1
            if ri["text"] != ro["text"]:
                n_text_mismatch += 1
            if ri.get("mentions", {}) != ro.get("mentions", {}):
                n_mentions_mismatch += 1
            for k in ri:
                if k in ("factors",):
                    continue
                if ri[k] != ro.get(k):
                    n_other_mismatch += 1
            fi_list = ri["factors"]
            fo_list = ro["factors"]
            if len(fi_list) != len(fo_list):
                n_fac_mismatch += 1
                continue
            for fi, fo in zip(fi_list, fo_list):
                fo_stripped = {k: v for k, v in fo.items() if k not in ("arg_spans", "cue_spans")}
                if fo_stripped != fi:
                    n_fac_mismatch += 1
    return dict(n_rows=n_rows, n_fac_mismatch=n_fac_mismatch,
                n_text_mismatch=n_text_mismatch,
                n_mentions_mismatch=n_mentions_mismatch,
                n_other_mismatch=n_other_mismatch)


def main():
    lines = []

    def P(s=""):
        print(s)
        lines.append(s)

    P("=" * 78)
    P("THE ARG-MENTION STAMP (2026-09-20)")
    P("=" * 78)

    stats = {"prose": new_stats(), "mint": new_stats()}
    P(f"\nStamping {IN} -> {OUT} ...")
    spot_prose, spot_mint = stamp_file(IN, OUT, stats)
    P("done.")

    P("")
    P("-" * 78)
    P("COVERAGE (diet)")
    P("-" * 78)
    print_stats(P, "PROSE rows", stats["prose"])
    print_stats(P, "MINT rows", stats["mint"])

    P("-" * 78)
    P("IDENTITY CHECK")
    P("-" * 78)
    chk = identity_check(IN, OUT)
    P(f"  rows compared: {chk['n_rows']}")
    P(f"  factor-list mismatches (beyond arg_spans/cue_spans): {chk['n_fac_mismatch']}")
    P(f"  text mismatches: {chk['n_text_mismatch']}")
    P(f"  mentions mismatches: {chk['n_mentions_mismatch']}")
    P(f"  other top-level field mismatches: {chk['n_other_mismatch']}")
    P(f"  PASS" if chk["n_fac_mismatch"] == chk["n_text_mismatch"] ==
      chk["n_mentions_mismatch"] == chk["n_other_mismatch"] == 0 else "  FAIL")

    P("")
    P("-" * 78)
    P("SPOT CHECK (prose rows, first 8 of 20 collected)")
    P("-" * 78)
    for li, text, row_spot in spot_prose[:8]:
        P(f"\nROW {li}: {text!r}")
        for i, fac, arg_spans, cue_spans, infos in row_spot:
            args = fac.get("args", [])
            marked_text = mark_spans(text, args, arg_spans, cue_spans)
            P(f"  factor[{i}] {fac['ftype']} op={fac.get('op', fac.get('sel', ''))} "
              f"args={args} result={fac.get('result')}: "
              f"classes={[inf['cls'] for inf in infos]}")
            P(f"    {marked_text}")

    P("")
    P("-" * 78)
    P("SPOT CHECK (mint rows with >=1 letter-class mention, first 4 of "
      f"{len(spot_mint)} collected)")
    P("-" * 78)
    for li, text, row_spot in spot_mint[:4]:
        P(f"\nROW {li}: {text!r}")
        for i, fac, arg_spans, cue_spans, infos in row_spot:
            args = fac.get("args", [])
            marked_text = mark_spans(text, args, arg_spans, cue_spans)
            P(f"  factor[{i}] {fac['ftype']} op={fac.get('op', fac.get('sel', ''))} "
              f"args={args} result={fac.get('result')}: "
              f"classes={[inf['cls'] for inf in infos]}")
            P(f"    {marked_text}")

    P("")
    P("=" * 78)
    P("done.")

    report_path = OUT.rsplit(".jsonl", 1)[0] + "_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[stamp_arg_mentions] wrote {report_path}")


if __name__ == "__main__":
    main()
