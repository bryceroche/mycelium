"""latest_before_read.py -- READ 2: LATEST-BEFORE (delegated task, 2026-10-13, zero-GPU).

Question (Bryce's framing, via the delegated brief): among multi-mention entities (an owner
with >=2 GIVEN numerals in one row), does the text pick the entity's MOST RECENT state as of
the governing relation, and does PMS8_241's wrong same-entity arg pick violate that rule?

Zero-GPU, zero-model: reads ONLY .cache/wild_admitted_holdout.jsonl (text+gold factors) and
.cache/dump_wild_PMS8_241.pkl (gold vs predicted per-slot args, loop_val.py's LV_DUMP format --
see governor_census.py's own INPUTS docstring for the tuple layout and the i==row/j==factor-
list-index identity, confirmed there by inspection). Entity identity and sentence index come
from spaCy's dependency parse (en_core_web_sm), reusing governor_census.py's own helpers
VERBATIM by import (numeral_candidates, bind_given_occurrences, governor_chain, sent_index_of,
digits_to_int) -- never from the twin-split tables, which read gold to build their splits; here
the owner/sentence machinery is built once per row from the TEXT alone, then gold and predicted
args are laid on top of it afterward.

DEFINITIONS (stated once, used throughout -- see the deliverable for why each was chosen):
  ENTITY    = a governor_chain() owner_key (surface lemma of the subject/possessor of a given
              numeral's local clause) -- the SAME owner identity governor_census.py's own
              SAME-OWNER/DIFFERENT-OWNER classification uses.
  STATE     = a GIVEN variable in the row whose governor chain resolves to that owner_key.
              "Multi-state" entities are owner_keys with >=2 such given vars in the SAME row.
              (Multiple given numerals sharing one owner are the mechanism the ledger's wall
              calls "the rate family"/"chained-fdiv" -- Alice's apple count at two points in
              the story is the textbook case.)
  RELATION'S OWN SENTENCE = max(sent_idx(v) for every GIVEN var v whose position in the row's
              `factors` list is < the relation's own position), i.e. the latest sentence
              boundary reached by ANY given fact revealed so far in text order, by the time
              the relation's factor-list slot is reached. This is INDEPENDENT of which
              candidate arg (gold or chosen) is under test -- it depends only on the row's
              given-var sentence positions and the relation's own list index -- so testing
              "is the gold/chosen arg the latest state AS OF this sentence" never leaks the
              answer being tested. (Factor list order mirrors text revelation order by
              construction -- the positional law / consecutive-letters convention, CLAUDE.md
              section 5 -- so list position is a sound proxy for "already stated by now";
              there is no per-relation text span to read directly, no `mentions` annotation
              exists on this file (confirmed empty on all 311 rows, same as governor_census.py
              found), and relations are not bound to a single text token the way given numerals
              are.)
  BEFORE    = a state's sent_idx <= the relation's own sentence (inclusive: a state introduced
              in the SAME sentence as the relation is available to it; this also guarantees the
              arg actually being tested is never spuriously excluded from its own "before" set,
              since every given var used as an arg necessarily precedes the relation in the
              factor list and therefore contributes to the very max that defines "own sentence").
  LATEST-BEFORE PICK for a multi-state entity at a given relation = the state(s) in BEFORE with
              the maximum sent_idx (ties possible -> a set, not always a single var).
  VIOLATION (part ii) = the chosen (wrongly-predicted) arg is NOT a member of that latest set --
              covers both "chose a state introduced after the relation's sentence" (which, by
              BEFORE's definition, simply means the chosen var isn't in BEFORE at all) and
              "chose an older state when a fresher one (already in BEFORE) exists".
  FIX (part iii) = among violations, the LATEST-BEFORE rule is APPLIED (pick the state(s) in
              BEFORE with max sent_idx) and that pick is UNAMBIGUOUS (exactly one state at the
              max) AND equals gold -- a tie at the max does not count as a deterministic fix.

OUTPUTS: .cache/latest_before_read_PMS8_241.json / .txt
"""
import os
import sys
import json
import pickle
import collections

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/picker")

import spacy

import governor_census as GC

HOLDOUT = ".cache/wild_admitted_holdout.jsonl"
DUMP = ".cache/dump_wild_PMS8_241.pkl"
OUT_JSON = ".cache/latest_before_read_PMS8_241.json"
OUT_TXT = ".cache/latest_before_read_PMS8_241.txt"


def build_row_chains(nlp, text, factors):
    """-> (doc, sent_list, given_vars set, given_val {var:value}, given_kfac {var:list_idx},
    chain_of {var: chain_dict or None}, entity_states {owner_key: [(var, sent_idx), ...]})"""
    doc = nlp(text)
    sent_list = list(doc.sents)
    cands = GC.numeral_candidates(doc)
    given_list = [(f["var"], f["value"]) for f in factors if f["ftype"] == "given"]
    given_val = {v: val for v, val in given_list}
    given_vars = set(given_val)
    given_kfac = {}
    for idx, f in enumerate(factors):
        if f["ftype"] == "given":
            given_kfac[f["var"]] = idx
    bound = GC.bind_given_occurrences(doc, given_list, cands)
    chain_of = {}
    for var, tok in bound.items():
        chain_of[var] = GC.governor_chain(doc, sent_list, tok) if tok is not None else None
    entity_states = collections.defaultdict(list)
    for var in given_vars:
        ch = chain_of.get(var)
        if ch is not None:
            entity_states[ch["owner_key"]].append((var, ch["sent_idx"]))
    return doc, sent_list, given_vars, given_val, given_kfac, chain_of, entity_states


def relation_sentence(idx, given_kfac, chain_of):
    sents = [chain_of[v]["sent_idx"] for v, k in given_kfac.items()
             if k < idx and chain_of.get(v) is not None]
    return max(sents) if sents else None


def latest_set_before(states, rel_sent):
    if rel_sent is None:
        return None
    before = [(v, s) for v, s in states if s is not None and s <= rel_sent]
    if not before:
        return None
    max_s = max(s for _, s in before)
    return {v for v, s in before if s == max_s}


def main():
    nlp = spacy.load("en_core_web_sm")
    rows = [json.loads(l) for l in open(HOLDOUT)]
    D = pickle.load(open(DUMP, "rb"))
    by_row = collections.defaultdict(list)
    for t in D:
        by_row[t[0]].append(t[1:])

    # ---- part (i): on GOLD graphs, is the gold arg the entity's latest-before state? ----
    n_eligible_i = 0
    n_hit_i = 0
    per_row_i = []

    # ---- parts (ii)/(iii): PMS8_241's wrong same-entity arg picks ----
    n_same_entity_wrong = 0
    n_violate = 0
    n_fixed = 0
    examples = []

    n_rows_multistate = 0
    n_entities_multistate = 0

    for i, row in enumerate(rows):
        factors = row["factors"]
        doc, sent_list, given_vars, given_val, given_kfac, chain_of, entity_states = \
            build_row_chains(nlp, row["text"], factors)

        multi = {k: v for k, v in entity_states.items() if len(v) >= 2}
        if multi:
            n_rows_multistate += 1
            n_entities_multistate += len(multi)

        for idx, f in enumerate(factors):
            if f["ftype"] != "rel":
                continue
            rel_sent = relation_sentence(idx, given_kfac, chain_of)
            gargs = [int(a) for a in f["args"]]

            for v in gargs:
                if v not in given_vars or chain_of.get(v) is None:
                    continue
                owner = chain_of[v]["owner_key"]
                states = entity_states.get(owner, [])
                if len(states) < 2:
                    continue
                lset = latest_set_before(states, rel_sent)
                if lset is None:
                    continue
                n_eligible_i += 1
                hit = v in lset
                n_hit_i += int(hit)
                per_row_i.append((i, idx, v, hit))

        # ---- part (ii)/(iii): join the dump's predicted args for this row's rel slots ----
        slots = {s[0]: s for s in by_row.get(i, [])}   # j -> (gft,gop,gargs,gres,gdig,pft,pop,pargs,pres_,pdig,ppres,pdup)
        for idx, f in enumerate(factors):
            if f["ftype"] != "rel":
                continue
            s = slots.get(idx)
            if s is None:
                continue
            gft, gop, gargs_d, gres, gdig, pft, pop, pargs, pres_, pdig, ppres, pdup = s
            gargs_d = [int(a) for a in gargs_d]
            pargs = [int(a) for a in pargs]
            rel_sent = relation_sentence(idx, given_kfac, chain_of)
            for v in gargs_d:
                if v not in given_vars or chain_of.get(v) is None:
                    continue
                owner = chain_of[v]["owner_key"]
                states = entity_states.get(owner, [])
                if len(states) < 2:
                    continue
                correct = (v in set(pargs)) if len(gargs_d) == 2 else (
                    bool(pdup) and len(pargs) > 0 and pargs[0] == v)
                if correct:
                    continue
                extra = [p for p in pargs if p not in gargs_d]
                chosen_var = extra[0] if extra else None
                if chosen_var is None or chosen_var not in given_vars:
                    continue
                chosen_chain = chain_of.get(chosen_var)
                if chosen_chain is None or chosen_chain["owner_key"] != owner:
                    continue   # not a same-entity wrong pick (DIFFERENT-OWNER / NO-CHAIN)
                n_same_entity_wrong += 1
                lset = latest_set_before(states, rel_sent)
                if lset is None:
                    continue
                violates = chosen_var not in lset
                n_violate += int(violates)
                fixed = False
                if violates and len(lset) == 1 and v in lset:
                    fixed = True
                    n_fixed += 1
                if len(examples) < 20:
                    examples.append(dict(row=i, slot=idx, gold_var=v, chosen_var=chosen_var,
                                          owner=owner, states=states, rel_sent=rel_sent,
                                          latest_set=sorted(lset), violates=violates, fixed=fixed))

    rate_i = n_hit_i / max(n_eligible_i, 1)
    rate_ii = n_violate / max(n_same_entity_wrong, 1)
    rate_iii = n_fixed / max(n_violate, 1)

    report = dict(
        tag="PMS8_241", fixture="wildhold",
        n_rows=len(rows), n_rows_with_multistate_entity=n_rows_multistate,
        n_multistate_entities=n_entities_multistate,
        part_i=dict(n_eligible=n_eligible_i, n_hit=n_hit_i, rate=rate_i, bar=0.9),
        part_ii=dict(n_same_entity_wrong=n_same_entity_wrong, n_violate=n_violate, rate=rate_ii, bar=0.5),
        part_iii=dict(n_violate=n_violate, n_fixed=n_fixed, rate=rate_iii),
        examples=examples,
    )
    json.dump(report, open(OUT_JSON, "w"), indent=1, default=str)

    lines = []
    L = lines.append
    L("READ 2 -- LATEST-BEFORE (PMS8_241, wildhold, n_rows=%d)" % len(rows))
    L("")
    L(f"Rows with >=1 multi-state entity: {n_rows_multistate}/{len(rows)}; total multi-state entities: {n_entities_multistate}")
    L("")
    L(f"(i) ON GOLD GRAPHS: eligible (relation arg = given var of a multi-state entity, rel_sent resolvable) "
      f"n={n_eligible_i}; gold arg IS the latest-before state in {n_hit_i} ({rate_i:.3f}); "
      f"PREDICTION >= 0.90: {'MET' if rate_i >= 0.9 else 'NOT MET'}")
    L("")
    L(f"(ii) PMS8_241 WRONG SAME-ENTITY ARG PICKS: n={n_same_entity_wrong}; violate latest-before "
      f"n={n_violate} ({rate_ii:.3f}); BAR >= 0.50: {'MET' if rate_ii >= 0.5 else 'NOT MET'}")
    L("")
    L(f"(iii) OF THE {n_violate} VIOLATIONS: latest-before rule (unambiguous pick) == gold for {n_fixed} "
      f"({rate_iii:.3f})")
    L("")
    L("DEFINITIONS: entity=governor_chain owner_key; state=a given var resolving to that owner; "
      "relation's own sentence=max sent_idx over given vars preceding the relation in factor-list order; "
      "before=sent_idx <= relation's own sentence; latest-before pick=max-sent_idx state(s) in 'before'; "
      "fix=unambiguous (singleton) latest-before pick equal to gold.")
    L("")
    L("20 EXAMPLES (row, slot, gold_var, chosen_var, owner, states, rel_sent, latest_set, violates, fixed):")
    for e in examples:
        L(f"  row={e['row']:3d} slot={e['slot']:2d} gold={e['gold_var']} chosen={e['chosen_var']} "
          f"owner={e['owner']!r} states={e['states']} rel_sent={e['rel_sent']} "
          f"latest_set={e['latest_set']} violates={e['violates']} fixed={e['fixed']}")
    txt = "\n".join(lines)
    open(OUT_TXT, "w").write(txt + "\n")
    print(txt)
    print(f"\n[latest-before] wrote {OUT_JSON} and {OUT_TXT}")


if __name__ == "__main__":
    main()
