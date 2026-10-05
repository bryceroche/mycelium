"""scripts/picker/angles.py -- THE PANEL PICKER's feature angles (2026-10-05, delegate).

Computes, per CANDIDATE parse (not just the top-1 decode), the angles listed in the task brief:
  - solver status (solved / inconsistent / refused) -- read straight off gen_candidates.py's branch dict
  - unique (bool) -- ditto
  - the model's log-likelihood -- ditto (CO.candidate_loglik, already computed by gen_candidates.py)
  - the candidate's rank by likelihood within its row -- computed here (argsort over the row's candidates)
  - graph length and its difference from the row's top-1 length -- computed here (len(parse))
  - the NL certifier's certificates (coverage, implied-value confirmation, collisions, cue agreement,
    chain-reaches-query) -- adapted from scripts/nl_certifier.py's per-row body (lines ~100-131 as of
    2026-09-24), generalized here to score an ARBITRARY (parse, assignment) pair instead of only the
    dump's own top-1 decode. REUSE, NOT REIMPLEMENTATION: the cue-word sets (ADD_CUES/MUL_CUES) and the
    five certificates' formulas are copied verbatim from nl_certifier.py; stamp_arg_mentions (SAM) and
    args_census (AC) are imported exactly as nl_certifier.py imports them.
  - args / value / type-op flip counts vs the row's top-1 candidate -- computed here by diffing facts
    keyed by their decode-time "_slot" id (the positional law: a slot's own fact never moves between
    candidates in the SAME row, only its CONTENT does -- args/value/ftype/op -- so a straight per-slot
    field diff is exact, not an approximation).
  - whether a flipped argument picked a SAME-NOUN TWIN -- adapted from scripts/twin_gap.py's own
    is_twin/classify_sets machinery (scripts/mixed_bucket_census.py's arg_closure/inherited_keys,
    scripts/lexical_identity_census.py's build_candidate_tables/same_sentence) and scripts/
    stamp_arg_mentions.py's own_var. twin_gap.py asks, for a GOLD argument decision, whether the
    MODEL's actual (possibly wrong) pick is a twin of the correct one. This script does not have "the
    correct one" to compare against at read time (that is what the picker is FOR) -- it asks the
    symmetric, key-free question twin_gap's own machinery already answers for any two variables: are
    the two fillers a candidate branch chooses among (the row's top-1 pick vs this candidate's pick,
    at the SAME argument position of the SAME relation slot) each other's SAME-NOUN TWINS under the
    row's factor-graph annotation (same sentence + (derived-from-each-other or inherited-key overlap))?
    This is a STATED INTERPRETATION, not a mechanical transcription of twin_gap.py (which is scored
    against gold, not against a top-1-vs-alternate pair) -- see TWIN_FLAG_NOTE below.
"""
import sys

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np

import stamp_arg_mentions as SAM
import args_census as AC
import mixed_bucket_census as MBC
import lexical_identity_census as LIC

TWIN_FLAG_NOTE = (
    "twin_flip_flag asks: among this candidate's ARGUMENT flips vs the row's top-1 candidate, is the "
    "newly-picked variable a SAME-NOUN TWIN (same sentence as, and either derived-from or inherited-key-"
    "overlapping with) the variable it replaced? This is the top1-vs-alternate symmetric form of twin_gap.py's "
    "gold-vs-model-pick is_twin predicate, not a verbatim reuse (there is no gold at read time)."
)

ADD_CUES = set("total sum altogether combined together more plus gained added increase increased extra "
               "additional left remaining fewer less minus lost spent subtract subtracted decrease decreased "
               "difference".split())
MUL_CUES = set("times twice double triple each per product multiply multiplied half quarter third rate "
               "every".split())


# ----------------------------------------------------------------------------------------------------------
# THE NL CERTIFIER, generalized to an arbitrary (parse, assignment) pair (adapted from nl_certifier.py's
# main(), lines ~100-131, attributed above).
# ----------------------------------------------------------------------------------------------------------
import re as _re


def nlc_certificates(text, parse, asg, q):
    givens = [f for f in parse if f["ftype"] == "given"]
    rels = [f for f in parse if f["ftype"] == "rel"]
    nums = [int(x) for x in _re.findall(r"(?<![\d.])\d{1,3}(?![\d])", text)]
    numset = set(nums)
    gvals = [int(f["value"]) for f in givens if f.get("value") is not None]
    value_present = (float(np.mean([v in numset for v in gvals])) if gvals else 0.0)
    coverage = (float(np.mean([n in set(gvals) for n in nums])) if nums else 1.0)
    no_double = (1.0 - (len(gvals) - len(set(gvals))) / len(gvals)) if gvals else 1.0
    bounds = AC.sentence_bounds(text); sspans = AC.sentence_spans(text, bounds)
    intro = SAM.build_intro_map(parse); memo = {}
    sol = list(asg) if asg is not None else []
    agree = []
    for j, f in enumerate(parse):
        if f["ftype"] != "rel":
            continue
        cl = SAM.clause_of(j, parse, text, bounds, sspans, sol, intro, memo)
        words = set(w.lower() for w in _re.findall(r"[a-z']+", " ".join(text[a:b] for a, b in SAM.windows_of(cl, sspans)))) if cl is not None else set()
        cues = ADD_CUES if f.get("op") == "add" else MUL_CUES
        agree.append(1.0 if (words & cues) else 0.0)
    cue_agree = float(np.mean(agree)) if agree else 1.0
    reach = {f["result"] for f in rels} | {f["var"] for f in givens}
    chain_reaches = 1.0 if q in reach else 0.0
    if asg is not None and rels:
        derived_stated = float(np.mean([int(asg[f["result"]]) in numset for f in rels if f["result"] < len(asg)]))
    else:
        derived_stated = 0.0
    certs = dict(value_present=value_present, coverage=coverage, no_double=no_double, cue_agree=cue_agree,
                 chain_reaches=chain_reaches, derived_stated=derived_stated)
    certs["cert"] = float(np.mean(list(certs.values())))
    return certs


NLC_KEYS = ("value_present", "coverage", "no_double", "cue_agree", "chain_reaches", "derived_stated", "cert")


# ----------------------------------------------------------------------------------------------------------
# flip counts vs top-1 (by _slot id -- the positional law: a slot's fact never moves rows, only content).
# ----------------------------------------------------------------------------------------------------------
def diff_from_top1(parse, top1_parse):
    top1_by_slot = {f["_slot"]: f for f in top1_parse}
    cur_by_slot = {f["_slot"]: f for f in parse}
    n_args = n_val = n_typeop = 0
    for slot, f1 in top1_by_slot.items():
        f2 = cur_by_slot.get(slot)
        if f2 is None:
            n_typeop += 1   # the slot vanished from the decode entirely (a ftype flip into "no fact")
            continue
        if f1.get("ftype") != f2.get("ftype") or f1.get("op") != f2.get("op"):
            n_typeop += 1
        if f1.get("ftype") == "rel" and f2.get("ftype") == "rel" and f1.get("args") != f2.get("args"):
            n_args += 1
        val1 = f1.get("value", f1.get("k", f1.get("p", f1.get("a"))))
        val2 = f2.get("value", f2.get("k", f2.get("p", f2.get("a"))))
        if val1 != val2 and f1.get("ftype") == f2.get("ftype"):
            n_val += 1
    for slot in cur_by_slot:
        if slot not in top1_by_slot:
            n_typeop += 1   # a fact appeared that wasn't in top-1 at all
    return n_args, n_val, n_typeop


def graph_len(parse):
    return len(parse)


# ----------------------------------------------------------------------------------------------------------
# twin-pick flag (adapted from twin_gap.py; see TWIN_FLAG_NOTE above). Cached per-row tables.
# ----------------------------------------------------------------------------------------------------------
def build_row_tables(fixture_row):
    """LIC.build_candidate_tables + MBC's closure/inherited-keys tables, for ONE row's gold factor
    annotation (fixture_row must carry 'factors'/'solution'/'text' -- the valid2 slice and the "_a"
    wild fixture both do). Cache this per row; it is the expensive part."""
    (text, factors, solution, bounds, sspans, intro_map, memo,
     cand_key, cand_sent) = LIC.build_candidate_tables(fixture_row)
    cmemo = {}
    closure = {i: MBC.arg_closure(i, factors, intro_map, cmemo) for i in range(len(factors))}
    inh = {i: MBC.inherited_keys(i, factors, cand_key, closure) for i in range(len(factors))}
    return dict(text=text, factors=factors, intro_map=intro_map, cand_sent=cand_sent, closure=closure, inh=inh)


def _are_twins(tables, k_a, k_b):
    """symmetric SAME-NOUN test between two factor-graph slots k_a, k_b (both the INTRODUCING slot of
    one of the two variables a candidate branch chose): same sentence, and (one derived-from the other,
    or their inherited key sets overlap) -- MBC's own classify_sets predicate, applied pairwise."""
    if k_a is None or k_b is None or k_a == k_b:
        return False
    sa = tables["cand_sent"].get(k_a, set()); sb = tables["cand_sent"].get(k_b, set())
    if not LIC.same_sentence(sa, sb):
        return False
    derived = (k_b in tables["closure"].get(k_a, set())) or (k_a in tables["closure"].get(k_b, set()))
    inh_overlap = bool(tables["inh"].get(k_a, set()) & tables["inh"].get(k_b, set()))
    return bool(derived or inh_overlap)


def twin_flip_flag(tables, parse, top1_parse):
    """True iff >=1 RELATION slot's argument differs between `parse` and `top1_parse` (by slot id) AND,
    for at least one such differing argument position, the newly-picked variable is a SAME-NOUN TWIN
    (per _are_twins) of the variable it replaced. See TWIN_FLAG_NOTE. Returns (flag:bool, n_checked:int)
    -- n_checked lets a caller distinguish 'no flip at all' from 'flip(s), none a twin'."""
    if tables is None:
        return False, 0
    top1_by_slot = {f["_slot"]: f for f in top1_parse if f.get("ftype") == "rel"}
    cur_by_slot = {f["_slot"]: f for f in parse if f.get("ftype") == "rel"}
    n_checked = 0; any_twin = False
    intro_map = tables["intro_map"]
    for slot, f1 in top1_by_slot.items():
        f2 = cur_by_slot.get(slot)
        if f2 is None or f1.get("args") == f2.get("args"):
            continue
        a1 = list(f1.get("args") or []); a2 = list(f2.get("args") or [])
        removed = [v for v in a1 if v not in a2]
        added = [v for v in a2 if v not in a1]
        for v_from, v_to in zip(removed, added):
            n_checked += 1
            k_from = intro_map.get(v_from); k_to = intro_map.get(v_to)
            if _are_twins(tables, k_from, k_to):
                any_twin = True
    return any_twin, n_checked
