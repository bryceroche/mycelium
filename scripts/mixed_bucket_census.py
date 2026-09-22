"""mixed_bucket_census.py -- THE MIXED-BUCKET DECOMPOSITION (2026-09-22, zero-GPU,
zero-training; queue item 3 of the 09-22 handoff, word given).

The twin-split census (scripts/twin_split_census.py) keyed every candidate by
LIC.one_token_key: the first content token after the candidate's value numeral
in the text. A RELATION candidate's value is derived, almost never literal, so
79-87% of relation-type candidates carried no key and 65% of the pointer wall
fell into MIXED/NO-KEY -- a key-coverage gap, not a third kind of tie. This
census gives a relation a key and re-splits.

THE RELATION KEY (per the handoff: "its result's noun / its cue"):
  KEY-INH (inherited): a relation is ABOUT what its arguments are about. Its
    key SET is the union of its arguments' key sets, transitively through
    intro_map (SAM.recursion_args gives each ftype's argument variables; a
    given's set is {its one-token key} or {} if none). This is the census
    form of the registered identity-port design ("a slot accumulates the
    identity of what it attended").
  DERIVED-FROM-k: a competitor c whose argument closure contains k itself --
    c is a later value of THE SAME quantity ("bought 12 ... has 12-5 left").
    This is the "same entity at a different time" tie, named structurally.
  KEY-LEX (lexical, secondary): the content stems inside the relation's own
    clause window (its stamped spans on wild, else its resolved sentence),
    minus cue words. Reported beside KEY-INH, never combined with it.

RE-CLASSIFICATION of every argument instance (row, j, pos, v, k) with >= 1
same-sentence competitor (the twin-split definitions, reused verbatim):
  k's key set K = KEY-INH(k) (a given's literal key; a relation's inherited set).
  SAME-NOUN: some competitor is DERIVED-FROM-k, or some competitor's key set
    intersects K (K non-empty).
  DIFFERENT-NOUN: K non-empty, every competitor's key set non-empty, none
    intersects K, none derived from k.
  MIXED/NO-KEY: everything else (K empty, or some competitor's set empty).
  Sub-buckets are reported for the old MIXED bucket: where each instance went.
args-correct := v in the dump's top-2 predicted args for slot j (the
twin-split convention). THE WALL SHARE = fraction of all WRONG classified
instances per class. Dumps: TS_DUMPS (comma list; default the chassis pair
and the record body). Output: .cache/mixed_bucket_census.txt
"""
import json
import os
import pickle
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

import numpy as np
import stamp_arg_mentions as SAM
import lexical_identity_census as LIC
import twin_split_census as TS

WILD_A = TS.WILD_A
DUMPS = os.environ.get("TS_DUMPS", ".cache/dump_wild_PMS4_241.pkl,.cache/dump_wild_PMS4_242.pkl,"
                       ".cache/dump_wild_PMS8_241.pkl").split(",")
OUT = ".cache/mixed_bucket_census.txt"
CLASSES = TS.CLASSES
CUES = set(SAM.stem(w) for w in SAM.CUE_WORDS)


def arg_closure(idx, factors, intro_map, memo):
    """the set of slots reachable from idx through its argument variables."""
    if idx in memo:
        return memo[idx]
    memo[idx] = set()  # cycle guard
    out = set()
    for a in SAM.recursion_args(factors[idx]):
        k = intro_map.get(a)
        if k is None or k == idx:
            continue
        out.add(k)
        out |= arg_closure(k, factors, intro_map, memo)
    memo[idx] = out
    return out


def inherited_keys(idx, factors, cand_key, closure):
    """KEY-INH: a given's literal key; a relation's = the union over its closure's givens."""
    fac = factors[idx]
    if fac["ftype"] == "given":
        return {cand_key[idx]} if cand_key.get(idx) is not None else set()
    out = set()
    if cand_key.get(idx) is not None:  # a relation whose result IS literal in the text keeps it
        out.add(cand_key[idx])
    for c in closure[idx]:
        if factors[c]["ftype"] == "given" and cand_key.get(c) is not None:
            out.add(cand_key[c])
    return out


def lexical_keys(idx, factors, text, sspans, clause_memo, bounds, solution, intro_map):
    cl = SAM.clause_of(idx, factors, text, bounds, sspans, solution, intro_map, clause_memo)
    if cl is None:
        return set()
    stems = set(s for s, _, _ in SAM.content_tokens_in(text, SAM.windows_of(cl, sspans)))
    return stems - CUES


def classify_sets(K, comp_sets, comp_derived):
    if any(comp_derived):
        return "SAME-NOUN"
    if not K:
        return "MIXED/NO-KEY"
    if any(cs & K for cs in comp_sets):
        return "SAME-NOUN"
    if comp_sets and all(cs for cs in comp_sets):
        return "DIFFERENT-NOUN"
    return "MIXED/NO-KEY"


def process_row(row, row_idx):
    (text, factors, solution, bounds, sspans, intro_map, memo,
     cand_key, cand_sent) = LIC.build_candidate_tables(row)
    cmemo = {}
    closure = {i: arg_closure(i, factors, intro_map, cmemo) for i in range(len(factors))}
    inh = {i: inherited_keys(i, factors, cand_key, closure) for i in range(len(factors))}
    lex = {}
    recs = []
    for j, fac in enumerate(factors):
        args = fac.get("args")
        if not args:
            continue
        rel_clause = SAM.clause_of(j, factors, text, bounds, sspans, solution, intro_map, memo)
        rel_sent_set = SAM.sent_set_of(rel_clause, bounds, sspans)
        for pos, v in enumerate(args):
            k = intro_map.get(v)
            if k is None or k == j:
                continue
            k_sent = cand_sent.get(k, set())
            competitors = [c for c in range(len(factors))
                           if c != j and c != k and LIC.same_sentence(cand_sent.get(c, set()), k_sent)]
            if not competitors:
                continue
            dist = None
            if rel_sent_set and k_sent:
                dist = min(abs(rs - ks) for rs in rel_sent_set for ks in k_sent)
            old_cls = TS.classify(cand_key.get(k), [cand_key.get(c) for c in competitors])
            comp_derived = [k in closure[c] for c in competitors]
            comp_inh = [inh[c] for c in competitors]
            new_cls = classify_sets(inh[k], comp_inh, comp_derived)
            # the lexical reading: literal keys where they exist, else the clause's content stems
            for c in competitors + [k]:
                if c not in lex:
                    lex[c] = ({cand_key[c]} if cand_key.get(c) is not None
                              else lexical_keys(c, factors, text, sspans, memo, bounds, solution, intro_map))
            lex_cls = classify_sets(lex[k], [lex[c] for c in competitors], [False] * len(competitors))
            k_is_rel = factors[k]["ftype"] != "given"
            n_rel_comp = sum(1 for c in competitors if factors[c]["ftype"] != "given")
            recs.append(dict(row=row_idx, j=j, pos=pos, v=v, k=k, competitors=competitors,
                             old=old_cls, new=new_cls, lex=lex_cls, derived=any(comp_derived),
                             k_is_rel=k_is_rel, n_rel_comp=n_rel_comp, n_comp=len(competitors),
                             dist=dist, K=inh[k], comp_inh=comp_inh,
                             sent_text=TS.sentence_text_of(k_sent, sspans, text)))
    return recs


def table(P, recs, pargs, field, title):
    P(f"  {title}")
    P(f"  {'class':16s} {'n':>6s} {'share':>7s} {'args-correct':>13s} {'n_wrong':>8s} {'wall':>7s}")
    corr = {c: [] for c in CLASSES}
    for r in recs:
        pr = pargs.get(r["row"], {}).get(r["j"])
        if pr is None:
            continue
        corr[r[field]].append(r["v"] in pr)
    n_tot = sum(len(x) for x in corr.values())
    n_wrong_tot = sum(sum(1 for c_ in x if not c_) for x in corr.values())
    for c in CLASSES:
        x = corr[c]
        nw = sum(1 for c_ in x if not c_)
        acc = np.mean(x) if x else float("nan")
        P(f"  {c:16s} {len(x):6d} {len(x) / max(n_tot, 1):7.3f} {acc:13.3f} {nw:8d} {nw / max(n_wrong_tot, 1):7.3f}")
    P(f"  (n={n_tot}, n_wrong={n_wrong_tot})")
    P("")
    return corr


def main():
    lines = []

    def P(s=""):
        print(s)
        lines.append(s)

    P("=" * 78)
    P("THE MIXED-BUCKET DECOMPOSITION (2026-09-22)")
    P("=" * 78)
    P("")
    P("A relation candidate's key = KEY-INH, the union of its arguments' literal keys")
    P("(transitively); a competitor whose argument closure contains k is DERIVED-FROM-k")
    P("(the same quantity at a later time) and counts SAME-NOUN. KEY-LEX (the clause's")
    P("content stems minus cue words) is the secondary reading. Same-sentence competitor,")
    P("dist, args-correct (top-2) exactly as twin_split_census.py.")
    P("")
    rows = [json.loads(l) for l in open(WILD_A)]
    recs = []
    for i, row in enumerate(rows):
        recs.extend(process_row(row, i))
    n = len(recs)
    P(f"WILD ({len(rows)} rows): {n} argument instances with >= 1 same-sentence competitor")
    P(f"  k is itself a relation (a derived argument): {sum(1 for r in recs if r['k_is_rel'])} "
      f"({sum(1 for r in recs if r['k_is_rel']) / n:.3f}); instances with >= 1 relation competitor: "
      f"{sum(1 for r in recs if r['n_rel_comp'])} ({sum(1 for r in recs if r['n_rel_comp']) / n:.3f}); "
      f"with a DERIVED-FROM-k competitor: {sum(1 for r in recs if r['derived'])} "
      f"({sum(1 for r in recs if r['derived']) / n:.3f})")
    P("")
    # structural migration of the old MIXED bucket
    P("WHERE THE OLD MIXED/NO-KEY BUCKET WENT (structural, no dump):")
    mixed = [r for r in recs if r["old"] == "MIXED/NO-KEY"]
    P(f"  old MIXED/NO-KEY n={len(mixed)} ({len(mixed) / n:.3f} of instances)")
    for c in CLASSES:
        xs = [r for r in mixed if r["new"] == c]
        d = sum(1 for r in xs if r["derived"])
        krel = sum(1 for r in xs if r["k_is_rel"])
        P(f"    -> {c:16s} n={len(xs):5d} ({len(xs) / max(len(mixed), 1):.3f})  of which derived-from-k {d}, k-is-relation {krel}")
    still = [r for r in mixed if r["new"] == "MIXED/NO-KEY"]
    P(f"  still MIXED under KEY-INH: k has no key {sum(1 for r in still if not r['K'])}; "
      f"a competitor has no key {sum(1 for r in still if r['K'])}")
    P(f"  under KEY-LEX the old MIXED bucket goes: "
      + ", ".join(f"{c} {sum(1 for r in mixed if r['lex'] == c)}" for c in CLASSES))
    P("")
    for dp in DUMPS:
        if not os.path.exists(dp):
            P(f"  (no dump {dp})")
            continue
        pargs = TS.load_dump_pargs(dp)
        name = os.path.basename(dp).replace("dump_wild_", "").replace(".pkl", "")
        P("-" * 78)
        P(f"DUMP {name}")
        P("-" * 78)
        table(P, recs, pargs, "old", "OLD KEY (twin_split_census, literal one-token key):")
        table(P, recs, pargs, "new", "KEY-INH (inherited + derived-from-k):")
        table(P, recs, pargs, "lex", "KEY-LEX (literal key else clause content stems):")
        # the derived-from-k cell on its own, and the same-noun cell split by derived vs shared-key
        for lab, pred in (("DERIVED-FROM-k competitor present", lambda r: r["derived"]),
                          ("SAME-NOUN(inh) without a derived competitor", lambda r: r["new"] == "SAME-NOUN" and not r["derived"]),
                          ("k is a relation (derived argument)", lambda r: r["k_is_rel"]),
                          ("k is a given, all competitors givens", lambda r: not r["k_is_rel"] and r["n_rel_comp"] == 0),
                          ("k is a given, >= 1 relation competitor", lambda r: not r["k_is_rel"] and r["n_rel_comp"] > 0)):
            x = [r["v"] in pargs[r["row"]][r["j"]] for r in recs if pred(r) and r["j"] in pargs.get(r["row"], {})]
            nw_all = sum(1 for r in recs if r["j"] in pargs.get(r["row"], {}) and r["v"] not in pargs[r["row"]][r["j"]])
            nw = sum(1 for c_ in x if not c_)
            P(f"  {lab:46s} n={len(x):5d} args-correct={np.mean(x) if x else float('nan'):.3f} "
              f"n_wrong={nw:4d} wall={nw / max(nw_all, 1):.3f}")
        P("")
        d0 = [r for r in recs if r["dist"] == 0]
        table(P, d0, pargs, "new", "KEY-INH at dist=0 (the relation stated in k's sentence):")
    # examples from the last dump: the wrong instances per new class
    P("=" * 78)
    P(f"WRONG EXAMPLES PER KEY-INH CLASS ({name})")
    P("=" * 78)
    for c in CLASSES:
        P(f"  [{c}]")
        shown = 0
        for r in recs:
            if r["new"] != c:
                continue
            pr = pargs.get(r["row"], {}).get(r["j"])
            if pr is None or r["v"] in pr:
                continue
            shown += 1
            if shown > 8:
                break
            P(f"    row={r['row']} j={r['j']} pos={r['pos']} v={r['v']} k={r['k']} old={r['old']} derived={r['derived']} k_is_rel={r['k_is_rel']} pred_top2={pr}")
            P(f"      sentence: {r['sent_text'][:170]}")
            P(f"      K(k)={sorted(r['K'])}  competitors: " + ", ".join(f"slot{c_}={sorted(s)}" for c_, s in zip(r["competitors"], r["comp_inh"])))
        P("")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[mixed-bucket] wrote {OUT}")


if __name__ == "__main__":
    main()
