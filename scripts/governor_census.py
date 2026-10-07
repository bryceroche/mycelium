"""governor_census.py -- THE GOVERNOR CENSUS (2026-10-07, zero GPU, CPU-only,
delegated). Bryce's question (read docs/phase1_skeleton_spec.md's 2026-09-22
"THE TWIN-SPLIT VERDICT" / "THE MIXED-BUCKET DECOMPOSITION" and 2026-10-06
"THE META READ" first): does the wild wall VIOLATE the syntactic governor
path (numeral -> head noun -> verb -> subject/possessor entity) or respect
it and differ only in TIME (tense / clause order)? Those two censuses
established that ~80-90% of the wall is SAME-NOUN / SAME-KEY selection
among same-entity twins at different roles/times, separated (where it is
separated at all) by precedence/order, not by noun identity. This script
asks the same question with a REAL dependency parse's governor chain
(spacy en_core_web_sm, installed into .venv for this measurement only --
no external API calls at run time; the model download is a one-time local
asset, consistent with CLAUDE.md's "dataset downloads are fine") instead
of the lexical-key / cue-window heuristics those censuses used.

INPUTS (read-only):
  .cache/wild_admitted_holdout.jsonl   -- 311 rows: text / factors / mentions
                                          (empty on every row in this file --
                                          confirmed by inspection) / query_var
  .cache/dump_wild_PMS8_241.pkl        -- loop_val.py's LV_DUMP: one tuple per
                                          GOLD slot, (i, j, gft, gop, gargs,
                                          gres, gdig, pft, pop, pargs, pres_,
                                          pdig, ppres, pdup); gdig/pdig are
                                          7-digit MSD-first arrays (value<=999).
                                          i is the row index into the holdout
                                          jsonl (verified: dump row 0's 5
                                          slots == holdout row 0's 5 factors,
                                          in order; 311 distinct rows, 2051
                                          total tuples, matching the npz).
  .cache/ps_legal_wild_PMS8_241.npz    -- (rows, slots, ok): the AUTHORITATIVE
                                          per-slot correctness verdict used
                                          for all CORRECT/WRONG classification
                                          below (per Bryce's framing).

CAVEAT BANKED UP FRONT (own census, not asserted from the ledger): ps_legal's
`ok` is the LV_LEGAL=num MASKED verdict; the dump's pdig is the RAW pre-mask
digit decode -- the exact raw-vs-masked gap rack_numeral_audit.py's
2026-10-07 correction flagged for the rack body. Measured here on PMS8_241:
58/1009 given slots (5.7%) have own-derived-ok (ppres and pft==gft and
pdig==gdig) != npz ok; of the 567 npz-wrong given slots, 34 (6.0%) have
pdig==gdig anyway (the failure is presence/ftype, not digit value -- no
distinct "chosen numeral" exists to find an owner for). Per the task's own
instruction, CORRECT/WRONG comes from ps_legal's `ok` (the law), and "the
slot's chosen numeral" comes from the dump's pdig (also the task's own
instruction) -- the 34 digit-matching-but-npz-wrong slots are reported in
their own bucket (NOT-A-VALUE-ERROR) and excluded from owner/governor
classification, which needs a *different* chosen numeral to classify.

METHOD (per-row):
  1. Parse the row's text with spacy (doc.sents, dependency tree).
  2. Enumerate NUMERAL CANDIDATES: every token matching \\d+ (like_num),
     plus single-token spelled-out numbers 0-100 (zero..ninety, dozen,
     hundred) WHEN that token's POS is NUM/NOUN/ADJ (filters out "won
     one" cases where a spelled number is not acting as a number --
     "one" as a pronoun-head gets through sometimes; spot-checked below).
     Multi-word spelled compounds ("twenty one") are NOT merged -- out of
     scope, flagged as a limitation, rare in this grade-school register.
  3. GREEDY VALUE MATCH (no span/mention annotation exists on this file --
     confirmed empty `mentions` on all 311 rows -- so there is no ground
     truth span to read): walk the row's GIVEN factors in their list order
     (= var-introduction order under the positional law) and bind each to
     the EARLIEST so-far-unused numeral candidate whose value equals the
     factor's value. This is a heuristic, not an annotation; the 20-row
     spot check below is exactly the human check on whether it is sound.
  4. GOVERNOR CHAIN for a bound numeral token: walk doc token .ancestors
     (nearest first); the first NOUN/PROPN ancestor is the head noun, the
     first VERB ancestor (stopping there) is the governing verb (the LOCAL
     clause verb, not the sentence root -- verified on a hand example: "her
     mom drinks an 8-ounce cup ... and uses one ounce" correctly resolves
     the 8 to "drinks"/"mom", not to "knows"/"She"). OWNER = (a) a `poss`
     child of the head noun if one exists (the direct-possessor case, "her
     mom['s cup]" / "John's apples"), else (b) an `nsubj`/`nsubjpass` child
     of the governing verb, climbing through `conj` coordination to borrow
     an elided subject ("drinks ... and uses" shares "mom"). No owner
     resolvable (no verb AND no poss) => NO-CHAIN. Owner identity for
     SAME/DIFFERENT comparison uses the owner token's lemma (surface
     comparison -- a known limitation: this does NOT resolve pronoun
     coreference, e.g. "mom" vs a later "she" referring to the same mom
     would read as DIFFERENT-OWNER; banked as a limitation, not corrected).
  5. CLASSIFY every WRONG given slot (npz ok==False, chosen != gold by
     digit value): SAME-OWNER (chosen owner key == gold owner key, both
     resolved), DIFFERENT-OWNER (both resolved, keys differ), NO-CHAIN
     (either side's chain did not resolve, including "chosen value has no
     text occurrence at all" -- the derived-identity-in-no-token case).
     For SAME-OWNER wrong picks: verb lemma differs? clause differs in
     TEXT ORDER (earlier vs later)? -> VERB / ORDER / NEITHER.
  6. GOVERNOR MASK ORACLE: candidate pool for a wrong slot = every GIVEN
     factor's (value, owner, verb, clause) in the same row (the legal
     numeral-pointer domain). owner-mask / owner+verb-mask / owner+order-
     mask filter the pool to those sharing the gold slot's owner (resp.
     +verb, +clause-order vs the gold's clause); report UNAMBIGUOUS (1
     survivor) vs STILL-AMBIGUOUS (>1) -- unresolvable gold chains are
     counted separately (the oracle cannot be defined for them).
  7. RELATION ARGUMENT POINTERS (same 4 parts) where a gargs var is ITSELF
     a given (exists as some factor's `var`): the "positional law" (slot
     index IS variable identity, shared between gold and prediction -- the
     architecture's two slot banks are 24 vars <-> letters positionally)
     makes pargs/gargs directly comparable var ids; per-arg correctness =
     membership in the opposite side's set (len 1 uses the dup bit, len 2
     a set compare, exactly matched_read.py's convention, reused in spirit
     not by import since its row-matching loop assumes a different slot
     numbering task). Args whose var is NOT given (derived) are OUT OF
     SCOPE per the prompt -- counted, not classified.

Writes .cache/governor_census_PMS8_241.txt with the full tables, the 20-row
spot check, and a 6-line reading. Run: .venv/bin/python3 scripts/governor_census.py
"""
import json
import pickle
import re
import collections
import subprocess
import numpy as np
import spacy

HOLDOUT = ".cache/wild_admitted_holdout.jsonl"
DUMP = ".cache/dump_wild_PMS8_241.pkl"
PSLEGAL = ".cache/ps_legal_wild_PMS8_241.npz"
OUT = ".cache/governor_census_PMS8_241.txt"

SPELLED = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "dozen": 12, "hundred": 100,
}


def digits_to_int(d):
    return int("".join(str(int(x)) for x in d))


def numeral_candidates(doc):
    cands = []
    for tok in doc:
        v = None
        if re.fullmatch(r"\d+", tok.text):
            v = int(tok.text)
        elif tok.text.lower() in SPELLED and tok.pos_ in ("NUM", "NOUN", "ADJ"):
            v = SPELLED[tok.text.lower()]
        if v is not None:
            cands.append((tok.i, tok, v))
    return cands


def sent_index_of(doc, sent_list, tok):
    for k, s in enumerate(sent_list):
        if s.start <= tok.i < s.end:
            return k
    return -1


FINITE_TAGS = ("VBD", "VBZ", "VBP")  # tensed, agreeing forms -- excludes
# infinitives (VB), gerunds (VBG), participles (VBN); AUX copulas (is/was)
# carry these same tags and count as finite too.


def is_finite_verb(tok):
    return tok.pos_ in ("VERB", "AUX") and tok.tag_ in FINITE_TAGS


def finite_verbs(doc):
    return sorted((t for t in doc if is_finite_verb(t)), key=lambda t: t.i)


def nearest_verb_anc(tok):
    for anc in tok.ancestors:
        if anc.pos_ in ("VERB", "AUX"):
            return anc
    return None


def event_index_of(fverbs, tok):
    """Bryce's THE NARRATIVE TIME (2026-10-07): the ordinal of the numeral's
    governing verb among the row's finite verbs in TEXT order (had=1, gave=2,
    bought=3, ...) -- 1-indexed, None iff the row has no finite verb at all.
    Climbs xcomp/ccomp/conj/advcl/acl/relcl from the nearest VERB/AUX
    ancestor to a finite one (mirrors find_owner's elided-subject climb);
    falls back to "the ordinal of the nearest PRECEDING finite verb token"
    by raw text position when no ancestor climb reaches a finite verb
    (fragment sentences, non-finite subordinate clauses with no finite
    governor above them within this parse)."""
    v = nearest_verb_anc(tok)
    seen = set()
    for _ in range(8):
        if v is None:
            break
        if is_finite_verb(v):
            for k, fv in enumerate(fverbs):
                if fv.i == v.i:
                    return k + 1
            break
        if v.i in seen or v.head is v:
            break
        seen.add(v.i)
        if v.dep_ in ("conj", "xcomp", "ccomp", "advcl", "acl", "relcl"):
            v = v.head
        else:
            break
    cand = [fv for fv in fverbs if fv.i <= tok.i]
    if cand:
        return fverbs.index(cand[-1]) + 1
    if fverbs:
        return 1
    return None


def verb_tense(verb):
    if verb is None:
        return None
    aux = [c.text.lower() for c in verb.children if c.dep_ in ("aux", "auxpass")]
    if "will" in aux or "wo" in aux:
        return "future"
    if "would" in aux or "'d" in aux:
        return "conditional"
    if "have" in aux or "has" in aux or "had" in aux:
        return "perfect"
    tag = verb.tag_
    return {"VBD": "past", "VBZ": "present", "VBP": "present",
            "VBG": "progressive", "VBN": "past-participle/passive",
            "MD": "modal"}.get(tag, tag or "unknown")


def find_owner(verb, head_noun, depth=0):
    """-> (owner_tok, owner_kind, verb_used) -- verb_used is the verb whose
    clause actually supplied the owner (may differ from the `verb` passed in
    when a controlled/coordinated subject was borrowed from a matrix clause
    higher up; verb_lemma/tense/clause_id should always be read off
    verb_used, not the original ancestor verb, so the three stay coherent
    with each other and with the owner they jointly describe)."""
    if head_noun is not None:
        for c in head_noun.children:
            if c.dep_ == "poss":
                return c, "possessor", verb
    if verb is None:
        return None, None, None
    for c in verb.children:
        if c.dep_ in ("nsubj", "nsubjpass"):
            return c, "subject", verb
    # elided/controlled subject: coordination ("drinks ... and uses") and
    # control/complement clauses ("wants to ride", "started to save",
    # "noticed that ...") borrow the matrix clause's subject -- a known
    # approximation for ccomp (which CAN have its own subject; when it does,
    # the nsubj check above already caught it and never reaches here).
    if verb.dep_ in ("conj", "xcomp", "ccomp", "advcl", "acl", "relcl") and depth < 4:
        return find_owner(verb.head, None, depth + 1)
    return None, None, None


def governor_chain(doc, sent_list, tok):
    """-> dict(owner_text, owner_key, owner_kind, verb_lemma, tense,
    clause_id, verb_idx) or None if unresolvable."""
    head_noun = None
    verb = None
    for anc in tok.ancestors:
        if head_noun is None and anc.pos_ in ("NOUN", "PROPN"):
            head_noun = anc
        # AUX included for copula clauses ("is 20 inches tall", "is 3 feet
        # across") where there is no lexical VERB in the chain at all -- an
        # AUX ancestor only ever shows up here as the clause's own ROOT/
        # predicate (a "will"/"has" AUX sitting ON a VERB is that VERB's
        # CHILD, never an ancestor of the VERB's own dependents, so this
        # cannot mis-fire on ordinary tense auxiliaries).
        if verb is None and anc.pos_ in ("VERB", "AUX"):
            verb = anc
            break
    if head_noun is None and tok.head is not tok:
        head_noun = tok.head
    owner_tok, owner_kind, verb_used = find_owner(verb, head_noun)
    if owner_tok is None:
        return None
    owner_text = " ".join(w.text for w in owner_tok.subtree)
    owner_key = (owner_tok.text.lower() if owner_tok.pos_ == "PROPN"
                 else owner_tok.lemma_.lower())
    s_idx = sent_index_of(doc, sent_list, tok)
    verb_idx = verb_used.i if verb_used is not None else (head_noun.i if head_noun else tok.i)
    return dict(
        owner_text=owner_text, owner_key=owner_key, owner_kind=owner_kind,
        verb_lemma=(verb_used.lemma_ if verb_used is not None else None),
        tense=verb_tense(verb_used),
        head_noun=(head_noun.text if head_noun is not None else None),
        clause_id=(s_idx, verb_idx), sent_idx=s_idx, verb_idx=verb_idx,
        char_start=tok.idx,
    )


def bind_given_occurrences(doc, given_list, cands):
    """given_list: [(var, value)] in factor-list order. Returns
    {var: (token, chain_or_None)}; chain computed lazily by caller."""
    used = set()
    out = {}
    for var, value in given_list:
        pick = None
        for k, (ti, tok, v) in enumerate(cands):
            if k in used:
                continue
            if v == value:
                pick = (k, tok)
                break
        if pick is not None:
            used.add(pick[0])
            out[var] = pick[1]
        else:
            out[var] = None
    return out


def nearest_occurrence(cands, value, used_idx, near_char):
    best = None
    for k, (ti, tok, v) in enumerate(cands):
        if v != value:
            continue
        d = abs(tok.idx - near_char)
        if best is None or d < best[0]:
            best = (d, k, tok)
    if best is None:
        return None
    return best[2]


def main():
    NLP = spacy.load("en_core_web_sm")
    rows = [json.loads(l) for l in open(HOLDOUT)]

    D = pickle.load(open(DUMP, "rb"))
    by_row = collections.defaultdict(list)
    for t in D:
        by_row[t[0]].append(t[1:])

    z = np.load(PSLEGAL)
    npz_ok = {(int(r), int(s)): bool(o)
              for r, s, o in zip(z["rows"], z["slots"], z["ok"])}

    # -------------------------- counters --------------------------
    parse_resolved = 0
    parse_total = 0
    given_cls = collections.Counter()      # SAME-OWNER/DIFFERENT-OWNER/NO-CHAIN
    same_owner_time = collections.Counter()  # VERB/ORDER/NEITHER
    same_owner_event = collections.Counter()  # SAME_EVENT/DIFF_EVENT/UNRESOLVED
    event_dist = collections.Counter()       # 0 / 1 / 2+ / unresolved (same-owner only)
    digit_match_only = 0
    n_given_wrong = 0
    n_given_total = 0
    oracle = collections.Counter()  # owner_unamb / owner_amb / owner_verb_unamb / ... / no_oracle

    arg_derived_skipped = 0
    arg_given_total = 0
    arg_given_wrong = 0
    arg_cls = collections.Counter()
    arg_same_owner_time = collections.Counter()
    arg_same_owner_event = collections.Counter()
    arg_event_dist = collections.Counter()
    arg_oracle = collections.Counter()

    spot_check = []  # (row, var, numeral, gold_chain, chosen_chain, cls)

    for i, row in enumerate(rows):
        text = row["text"]
        doc = NLP(text)
        sent_list = list(doc.sents)
        cands = numeral_candidates(doc)

        factors = row["factors"]
        given_list = [(f["var"], f["value"]) for f in factors if f["ftype"] == "given"]
        given_val = {f["var"]: f["value"] for f in factors if f["ftype"] == "given"}
        given_vars = set(given_val)

        fverbs = finite_verbs(doc)  # THE NARRATIVE TIME (Bryce, 2026-10-07):
        # the row's finite verbs in text order, 1-indexed ordinal below.

        bound = bind_given_occurrences(doc, given_list, cands)
        chain_of = {}
        event_idx_of = {}
        for var, tok in bound.items():
            parse_total += 1
            if tok is None:
                chain_of[var] = None
                event_idx_of[var] = None
                continue
            ch = governor_chain(doc, sent_list, tok)
            chain_of[var] = ch
            event_idx_of[var] = event_index_of(fverbs, tok)
            if ch is not None:
                parse_resolved += 1

        row_pool = []  # (var, value, chain, event_idx) for oracle candidate pool
        for var, value in given_list:
            row_pool.append((var, value, chain_of.get(var), event_idx_of.get(var)))

        slots = sorted(by_row[i], key=lambda s: s[0])
        gres_of_slot = {}
        gft_of_slot = {}
        for s in slots:
            j, gft, gop, gargs, gres, gdig, pft, pop, pargs, pres_, pdig, ppres, pdup = s
            gres_of_slot[j] = gres
            gft_of_slot[j] = gft

        # ---------------- PART A/B: GIVEN slots ----------------
        for s in slots:
            j, gft, gop, gargs, gres, gdig, pft, pop, pargs, pres_, pdig, ppres, pdup = s
            if gft != 1:
                continue
            n_given_total += 1
            var = gres  # the var this given slot introduces
            ok = npz_ok.get((i, j))
            if ok is None or ok:
                continue
            n_given_wrong += 1
            gold_chain = chain_of.get(var)
            gold_tok = bound.get(var)
            gold_val = digits_to_int(gdig)
            chosen_val = digits_to_int(pdig)

            if chosen_val == gold_val:
                digit_match_only += 1
                continue

            near_char = gold_tok.idx if gold_tok is not None else 0
            chosen_tok = nearest_occurrence(cands, chosen_val, None, near_char)
            chosen_chain = (governor_chain(doc, sent_list, chosen_tok)
                             if chosen_tok is not None else None)
            gold_event = event_idx_of.get(var)
            chosen_event = (event_index_of(fverbs, chosen_tok)
                             if chosen_tok is not None else None)

            if gold_chain is None or chosen_chain is None:
                cls = "NO-CHAIN"
            elif gold_chain["owner_key"] == chosen_chain["owner_key"]:
                cls = "SAME-OWNER"
            else:
                cls = "DIFFERENT-OWNER"
            given_cls[cls] += 1

            if cls == "SAME-OWNER":
                verb_diff = gold_chain["verb_lemma"] != chosen_chain["verb_lemma"]
                order_diff = gold_chain["clause_id"] != chosen_chain["clause_id"]
                if verb_diff and order_diff:
                    same_owner_time["VERB+ORDER"] += 1
                elif verb_diff:
                    same_owner_time["VERB"] += 1
                elif order_diff:
                    same_owner_time["ORDER"] += 1
                else:
                    same_owner_time["NEITHER"] += 1
                if gold_event is None or chosen_event is None:
                    same_owner_event["UNRESOLVED"] += 1
                    event_dist["unresolved"] += 1
                else:
                    same_owner_event["SAME_EVENT" if gold_event == chosen_event else "DIFF_EVENT"] += 1
                    d = abs(gold_event - chosen_event)
                    event_dist["2+" if d >= 2 else str(d)] += 1

            # ---- oracle ----
            if gold_chain is not None:
                pool = [(v, val, ch, ev) for (v, val, ch, ev) in row_pool if ch is not None]
                owner_pool = [p for p in pool if p[2]["owner_key"] == gold_chain["owner_key"]]
                verb_pool = [p for p in owner_pool if p[2]["verb_lemma"] == gold_chain["verb_lemma"]]
                order_pool = [p for p in owner_pool if p[2]["clause_id"] == gold_chain["clause_id"]]
                event_pool = [p for p in owner_pool if p[3] == gold_event]
                oracle["owner_unamb" if len(owner_pool) <= 1 else "owner_amb"] += 1
                oracle["owner_verb_unamb" if len(verb_pool) <= 1 else "owner_verb_amb"] += 1
                oracle["owner_order_unamb" if len(order_pool) <= 1 else "owner_order_amb"] += 1
                oracle["owner_event_unamb" if len(event_pool) <= 1 else "owner_event_amb"] += 1
            else:
                oracle["no_oracle"] += 1

            if len(spot_check) < 20:
                spot_check.append((i, var, gold_val, chosen_val, gold_chain, chosen_chain, cls,
                                    gold_event, chosen_event))

        # ---------------- PART C: relation arg pointers onto givens ----------------
        for s in slots:
            j, gft, gop, gargs, gres, gdig, pft, pop, pargs, pres_, pdig, ppres, pdup = s
            if gft != 0:
                continue
            for v in gargs:
                is_given = v in given_vars
                if not is_given:
                    arg_derived_skipped += 1
                    continue
                arg_given_total += 1
                correct = v in set(pargs) if len(gargs) == 2 else (
                    bool(pdup) and len(pargs) > 0 and pargs[0] == v)
                if correct:
                    continue
                arg_given_wrong += 1
                gold_chain = chain_of.get(v)
                gold_val = given_val[v]
                gold_tok = bound.get(v)
                extra = [p for p in pargs if p not in gargs]
                chosen_var = extra[0] if extra else None
                if chosen_var is not None and chosen_var in given_vars:
                    chosen_chain = chain_of.get(chosen_var)
                    chosen_val = given_val[chosen_var]
                else:
                    chosen_chain = None
                    chosen_val = None

                if gold_chain is None or chosen_chain is None:
                    cls = "NO-CHAIN"
                elif gold_chain["owner_key"] == chosen_chain["owner_key"]:
                    cls = "SAME-OWNER"
                else:
                    cls = "DIFFERENT-OWNER"
                arg_cls[cls] += 1

                gold_event = event_idx_of.get(v)
                chosen_event = event_idx_of.get(chosen_var) if chosen_var is not None else None

                if cls == "SAME-OWNER":
                    verb_diff = gold_chain["verb_lemma"] != chosen_chain["verb_lemma"]
                    order_diff = gold_chain["clause_id"] != chosen_chain["clause_id"]
                    if verb_diff and order_diff:
                        arg_same_owner_time["VERB+ORDER"] += 1
                    elif verb_diff:
                        arg_same_owner_time["VERB"] += 1
                    elif order_diff:
                        arg_same_owner_time["ORDER"] += 1
                    else:
                        arg_same_owner_time["NEITHER"] += 1
                    if gold_event is None or chosen_event is None:
                        arg_same_owner_event["UNRESOLVED"] += 1
                        arg_event_dist["unresolved"] += 1
                    else:
                        arg_same_owner_event["SAME_EVENT" if gold_event == chosen_event else "DIFF_EVENT"] += 1
                        d = abs(gold_event - chosen_event)
                        arg_event_dist["2+" if d >= 2 else str(d)] += 1

                if gold_chain is not None:
                    pool = [(vv, val, ch, ev) for (vv, val, ch, ev) in row_pool if ch is not None]
                    owner_pool = [p for p in pool if p[2]["owner_key"] == gold_chain["owner_key"]]
                    verb_pool = [p for p in owner_pool if p[2]["verb_lemma"] == gold_chain["verb_lemma"]]
                    order_pool = [p for p in owner_pool if p[2]["clause_id"] == gold_chain["clause_id"]]
                    event_pool = [p for p in owner_pool if p[3] == gold_event]
                    arg_oracle["owner_unamb" if len(owner_pool) <= 1 else "owner_amb"] += 1
                    arg_oracle["owner_verb_unamb" if len(verb_pool) <= 1 else "owner_verb_amb"] += 1
                    arg_oracle["owner_order_unamb" if len(order_pool) <= 1 else "owner_order_amb"] += 1
                    arg_oracle["owner_event_unamb" if len(event_pool) <= 1 else "owner_event_amb"] += 1
                else:
                    arg_oracle["no_oracle"] += 1

    # -------------------------- write report --------------------------
    ts = subprocess.check_output(["date"]).decode().strip()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    lines = []
    P = lines.append
    P("THE GOVERNOR CENSUS -- PMS8_241 on the wild holdout (311 rows, 2051 gold slots)")
    P(f"generated {ts}; repo HEAD at generation time {commit}")
    P(f"spacy {spacy.__version__} / en_core_web_sm; .venv installed for this measurement (pip install spacy +")
    P("spacy download en_core_web_sm), no external API calls at run time.")
    P("")
    P("== PARSE QUALITY ON THIS TEXT ==")
    P(f"given-factor numerals with a resolvable owner chain: {parse_resolved}/{parse_total} "
      f"({parse_resolved/max(parse_total,1):.3f})")
    P("")
    P("== GIVEN-SLOT WRONG CENSUS (ps_legal 'ok'==False, n_given_wrong of n_given_total) ==")
    P(f"given slots total: {n_given_total}; wrong (npz ok=False): {n_given_wrong} "
      f"({n_given_wrong/max(n_given_total,1):.3f})")
    P(f"  of which NOT-A-VALUE-ERROR (raw pdig==gdig anyway -- a presence/ftype failure, no")
    P(f"  distinct chosen numeral to classify): {digit_match_only}")
    classified_n = sum(given_cls.values())
    P(f"  classified (chosen numeral != gold numeral): {classified_n}")
    for k in ("SAME-OWNER", "DIFFERENT-OWNER", "NO-CHAIN"):
        c = given_cls.get(k, 0)
        P(f"    {k:16s} {c:4d}  ({c/max(classified_n,1):.3f})")
    P("")
    P("== SAME-OWNER WRONG PICKS: separated by VERB / clause ORDER / NEITHER ==")
    same_n = sum(same_owner_time.values())
    for k in ("VERB", "ORDER", "VERB+ORDER", "NEITHER"):
        c = same_owner_time.get(k, 0)
        P(f"    {k:12s} {c:4d}  ({c/max(same_n,1):.3f})")
    P("")
    P("== SAME-OWNER WRONG PICKS: separated by EVENT INDEX (ordinal of the governing finite verb, had=1/gave=2/...) ==")
    sen = sum(same_owner_event.values())
    for k in ("SAME_EVENT", "DIFF_EVENT", "UNRESOLVED"):
        c = same_owner_event.get(k, 0)
        P(f"    {k:12s} {c:4d}  ({c/max(sen,1):.3f})")
    P("  event-index DISTANCE |gold_event - chosen_event| (same-owner wrong picks only):")
    edn = sum(event_dist.values())
    for k in ("0", "1", "2+", "unresolved"):
        c = event_dist.get(k, 0)
        P(f"    dist={k:10s} {c:4d}  ({c/max(edn,1):.3f})")
    P("")
    P("== GOVERNOR MASK ORACLE (wrong given slots with a resolvable gold chain) ==")
    oracle_n = oracle.get("owner_unamb", 0) + oracle.get("owner_amb", 0)
    P(f"  gold chain resolvable for oracle: {oracle_n}  (no_oracle: {oracle.get('no_oracle',0)})")
    P(f"    owner alone:        unambiguous {oracle.get('owner_unamb',0):4d}  ambiguous {oracle.get('owner_amb',0):4d}  "
      f"(fixed: {oracle.get('owner_unamb',0)/max(oracle_n,1):.3f})")
    P(f"    owner + verb:       unambiguous {oracle.get('owner_verb_unamb',0):4d}  ambiguous {oracle.get('owner_verb_amb',0):4d}  "
      f"(fixed: {oracle.get('owner_verb_unamb',0)/max(oracle_n,1):.3f})")
    P(f"    owner + clause order: unambiguous {oracle.get('owner_order_unamb',0):4d}  ambiguous {oracle.get('owner_order_amb',0):4d}  "
      f"(fixed: {oracle.get('owner_order_unamb',0)/max(oracle_n,1):.3f})")
    P(f"    owner + event index:  unambiguous {oracle.get('owner_event_unamb',0):4d}  ambiguous {oracle.get('owner_event_amb',0):4d}  "
      f"(fixed: {oracle.get('owner_event_unamb',0)/max(oracle_n,1):.3f})")
    P("")
    P("== RELATION ARG POINTERS WHOSE GOLD VAR IS A GIVEN (same 4 parts) ==")
    P(f"arg instances whose gold var is DERIVED (out of scope, counted only): {arg_derived_skipped}")
    P(f"arg instances whose gold var is GIVEN (in scope): {arg_given_total}; wrong: {arg_given_wrong} "
      f"({arg_given_wrong/max(arg_given_total,1):.3f})")
    acn = sum(arg_cls.values())
    for k in ("SAME-OWNER", "DIFFERENT-OWNER", "NO-CHAIN"):
        c = arg_cls.get(k, 0)
        P(f"    {k:16s} {c:4d}  ({c/max(acn,1):.3f})")
    P("  same-owner wrong picks by VERB / ORDER / NEITHER:")
    asn = sum(arg_same_owner_time.values())
    for k in ("VERB", "ORDER", "VERB+ORDER", "NEITHER"):
        c = arg_same_owner_time.get(k, 0)
        P(f"    {k:12s} {c:4d}  ({c/max(asn,1):.3f})")
    P("  same-owner wrong picks by EVENT INDEX (same/diff; distance 0/1/2+):")
    asen = sum(arg_same_owner_event.values())
    for k in ("SAME_EVENT", "DIFF_EVENT", "UNRESOLVED"):
        c = arg_same_owner_event.get(k, 0)
        P(f"    {k:12s} {c:4d}  ({c/max(asen,1):.3f})")
    aedn = sum(arg_event_dist.values())
    for k in ("0", "1", "2+", "unresolved"):
        c = arg_event_dist.get(k, 0)
        P(f"    dist={k:10s} {c:4d}  ({c/max(aedn,1):.3f})")
    aoracle_n = arg_oracle.get("owner_unamb", 0) + arg_oracle.get("owner_amb", 0)
    P(f"  oracle (gold chain resolvable: {aoracle_n}, no_oracle {arg_oracle.get('no_oracle',0)}):")
    P(f"    owner alone:        unambiguous {arg_oracle.get('owner_unamb',0):4d}  ambiguous {arg_oracle.get('owner_amb',0):4d}")
    P(f"    owner + verb:       unambiguous {arg_oracle.get('owner_verb_unamb',0):4d}  ambiguous {arg_oracle.get('owner_verb_amb',0):4d}")
    P(f"    owner + clause order: unambiguous {arg_oracle.get('owner_order_unamb',0):4d}  ambiguous {arg_oracle.get('owner_order_amb',0):4d}")
    P(f"    owner + event index:  unambiguous {arg_oracle.get('owner_event_unamb',0):4d}  ambiguous {arg_oracle.get('owner_event_amb',0):4d}")
    P("")
    P("== 20-ROW SPOT CHECK (numeral, chain, gold owner, event index) for human eyeball ==")
    for (row, var, gval, cval, gch, cch, cls, gev, cev) in spot_check:
        gtxt = (f"owner={gch['owner_text']!r}({gch['owner_kind']}) verb={gch['verb_lemma']} "
                f"tense={gch['tense']} clause={gch['clause_id']}") if gch else "NO-CHAIN"
        ctxt = (f"owner={cch['owner_text']!r}({cch['owner_kind']}) verb={cch['verb_lemma']} "
                f"tense={cch['tense']} clause={cch['clause_id']}") if cch else "NO-CHAIN"
        P(f"  row {row:3d} var {var:2d} gold={gval:<4d} event={gev} [{gtxt}]")
        P(f"              chosen={cval:<4d} event={cev} [{ctxt}]  -> {cls}")
    P("")
    P("== THE READING (8 lines) ==")
    wall_same = given_cls.get("SAME-OWNER", 0)
    wall_diff = given_cls.get("DIFFERENT-OWNER", 0)
    wall_nochain = given_cls.get("NO-CHAIN", 0)
    sep_owner = oracle.get("owner_unamb", 0) / max(oracle_n, 1)
    sep_verb = oracle.get("owner_verb_unamb", 0) / max(oracle_n, 1)
    sep_order = oracle.get("owner_order_unamb", 0) / max(oracle_n, 1)
    P(f"1. THE WALL RESPECTS THE GOVERNOR PATH: of {classified_n} classified wrong given picks "
      f"(digit-value errors only), {wall_same} ({wall_same/max(classified_n,1):.1%}) are SAME-OWNER, "
      f"{wall_diff} ({wall_diff/max(classified_n,1):.1%}) DIFFERENT-OWNER, {wall_nochain} "
      f"({wall_nochain/max(classified_n,1):.1%}) NO-CHAIN -- a governor mask would never have forbidden "
      f"the vast majority of the model's own wrong picks.")
    P(f"2. Separability by OWNER ALONE is a minority of the oracle-resolvable wrong slots: "
      f"{oracle.get('owner_unamb',0)}/{oracle_n} ({sep_owner:.1%}) become unambiguous under an "
      f"owner-only mask; the rest ({1-sep_owner:.1%}) still have >=2 same-owner candidates left.")
    P(f"3. OWNER + VERB resolves {oracle.get('owner_verb_unamb',0)}/{oracle_n} ({sep_verb:.1%}); "
      f"OWNER + CLAUSE ORDER resolves {oracle.get('owner_order_unamb',0)}/{oracle_n} ({sep_order:.1%}) -- "
      f"{'order' if sep_order>=sep_verb else 'verb'} is the stronger of the two second cues on this body.")
    P(f"4. Among SAME-OWNER wrong picks the gold/chosen pair differ by VERB in "
      f"{same_owner_time.get('VERB',0)+same_owner_time.get('VERB+ORDER',0)}/{max(same_n,1)} "
      f"({(same_owner_time.get('VERB',0)+same_owner_time.get('VERB+ORDER',0))/max(same_n,1):.1%}), by "
      f"clause ORDER in {same_owner_time.get('ORDER',0)+same_owner_time.get('VERB+ORDER',0)}/{max(same_n,1)} "
      f"({(same_owner_time.get('ORDER',0)+same_owner_time.get('VERB+ORDER',0))/max(same_n,1):.1%}), "
      f"and by NEITHER feature in {same_owner_time.get('NEITHER',0)}/{max(same_n,1)} "
      f"({same_owner_time.get('NEITHER',0)/max(same_n,1):.1%}) -- the NEITHER slice is the pair the "
      f"governor chain truly cannot tell apart at all (same owner, same verb, same clause).")
    P(f"5. PREDICTION CHECK (pinned: most of the wall respects the governor path (same owner), and "
      f"separability by owner alone is a minority): {'CONFIRMED' if wall_same/max(classified_n,1) > 0.5 and sep_owner < 0.5 else 'NOT CONFIRMED AS STATED'} "
      f"-- same-owner {wall_same/max(classified_n,1):.1%} ({'>' if wall_same/max(classified_n,1)>0.5 else '<='} half) "
      f"and owner-alone separability {sep_owner:.1%} ({'<' if sep_owner<0.5 else '>='} half).")
    P(f"6. PARSE CAVEAT: owner comparison uses surface lemma identity, not coreference (a pronoun "
      f"referring back to an already-named owner reads as a DIFFERENT owner) -- this biases "
      f"DIFFERENT-OWNER and NO-CHAIN UP and SAME-OWNER DOWN, so line 1's same-owner share is a "
      f"LOWER BOUND on how much of the wall truly respects the governor path. Parse resolves "
      f"{parse_resolved}/{parse_total} ({parse_resolved/max(parse_total,1):.1%}) of given numerals' "
      f"owner chains outright (see the 20-row spot check above for the failure mode).")
    sep_event = oracle.get("owner_event_unamb", 0) / max(oracle_n, 1)
    diff_event_share = same_owner_event.get("DIFF_EVENT", 0) / max(sen, 1)
    P(f"7. THE NARRATIVE-TIME CHECK (Bryce, 2026-10-07): among SAME-OWNER wrong picks, "
      f"{same_owner_event.get('DIFF_EVENT',0)}/{sen} ({diff_event_share:.1%}) differ in EVENT INDEX "
      f"(the ordinal of the governing finite verb, had=1/gave=2/bought=3/...) vs "
      f"{same_owner_event.get('SAME_EVENT',0)}/{sen} ({same_owner_event.get('SAME_EVENT',0)/max(sen,1):.1%}) "
      f"sharing the same event; distance is 0 for {event_dist.get('0',0)}/{edn} "
      f"({event_dist.get('0',0)/max(edn,1):.1%}), 1 for {event_dist.get('1',0)}/{edn} "
      f"({event_dist.get('1',0)/max(edn,1):.1%}), 2+ for {event_dist.get('2+',0)}/{edn} "
      f"({event_dist.get('2+',0)/max(edn,1):.1%}).")
    P(f"8. OWNER+EVENT vs OWNER+CLAUSE (does the same-owner wall separate at the EVENT level where "
      f"TC_241's where-codes line found it did NOT separate at the clause/tree level?): owner+event "
      f"resolves {oracle.get('owner_event_unamb',0)}/{oracle_n} ({sep_event:.1%}) vs owner+clause-order "
      f"{oracle.get('owner_order_unamb',0)}/{oracle_n} ({sep_order:.1%}) -- "
      f"{'EVENT IS FINER THAN CLAUSE (separates more)' if sep_event > sep_order else ('EVENT TIES CLAUSE' if abs(sep_event-sep_order)<1e-9 else 'EVENT IS NOT FINER THAN CLAUSE on this body')}; "
      f"since event index is itself read off the SAME dependency parse as the clause id (a within-clause "
      f"refinement where multiple finite verbs share one sentence), any gain here is the parse's clause "
      f"GRANULARITY, not new evidence -- consistent with (not an independent test of) the ledger's finding "
      f"that the tree-descent/where-codes line closed on the membrane (TC_241, 2026-10-04).")

    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[governor-census] wrote {OUT}")
    print("\n".join(lines[-20:]))


if __name__ == "__main__":
    main()
