"""candidate_census.py -- THE CANDIDATE CENSUS (2026-10-21, zero GPU, CPU-only;
Opus's question after the read race; ledger 2026-10-21 "THE CANDIDATE
CENSUS"). The read race found that a beam must be as wide as the live
ambiguity (top-2 failed with three live candidates; beam-4 = exact
marginalization). How wide is the live ambiguity on the MAIN LINE? For each
wild GOLD given lookup (a given slot whose value a numeral in the text
supplies), count the candidates still LIVE after the parse's filters:

  F0  none                      every numeral in the text
  F1  type                      same head-noun lemma as the gold's numeral
  F2  owner                     same governor-chain owner
  F3  owner + type
  F4  owner + type + event      + same event index (governing finite verb)

Counted in ANSWER space (distinct VALUES among the live tokens: the read
race marginalized over answers, and two tokens with one value are one
answer) and in token space. A filter that cannot read a candidate (no owner
chain / no head noun / no event) cannot exclude it: LENIENT = unknown is
live (the primary count); STRICT = unknown excluded (reported beside it).

What this is NOT: the model's live mass (the mass beam's count needs the
pointer distribution; the dump keeps only the decoded digits). It is the
ambiguity a perfect owner/type/event filter would leave, i.e. what a beam
behind such filters would have to carry. The parse is spaCy en_core_web_sm,
for measurement only (the governor census's machinery, imported unchanged:
greedy value-to-numeral binding, no coreference).

INPUTS: .cache/wild_admitted_holdout.jsonl; .cache/dump_wild_{CC_TAG}.pkl and
.cache/ps_legal_wild_{CC_TAG}.npz (only to split right / wrong slots).
Writes .cache/candidate_census_{CC_TAG}.txt (overwrite guard).
"""
import collections
import json
import os
import pickle
import sys

import numpy as np
import spacy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import governor_census as G  # noqa: E402

TAG = os.environ.get("CC_TAG", "CLK3_241")
OUT = f".cache/candidate_census_{TAG}.txt"
FILTERS = ["F0 none", "F1 type", "F2 owner", "F3 owner+type", "F4 owner+type+event"]


def head_noun_key(tok):
    for anc in tok.ancestors:
        if anc.pos_ in ("NOUN", "PROPN"):
            return anc.lemma_.lower()
    if tok.head is not tok and tok.head.pos_ in ("NOUN", "PROPN"):
        return tok.head.lemma_.lower()
    return None


def live(gold, cand, level, strict):
    """gold / cand: dict(owner, type, event) (None = unreadable)."""
    keys = [[], ["type"], ["owner"], ["owner", "type"], ["owner", "type", "event"]][level]
    for k in keys:
        g, c = gold[k], cand[k]
        if g is None or c is None:
            if strict:
                return False
            continue
        if g != c:
            return False
    return True


def summarize(counts):
    a = np.array(counts)
    if len(a) == 0:
        return "n 0"
    return (f"n {len(a):4d}  mean {a.mean():5.2f}  median {np.median(a):4.1f}  "
            f"=1 {np.mean(a == 1):.3f}  <=2 {np.mean(a <= 2):.3f}  "
            f"<=4 {np.mean(a <= 4):.3f}  >=5 {np.mean(a >= 5):.3f}  max {a.max()}")


def main():
    if os.path.exists(OUT):
        sys.exit(f"overwrite guard: {OUT} exists")
    nlp = spacy.load("en_core_web_sm")
    rows = [json.loads(l) for l in open(G.HOLDOUT)]
    D = pickle.load(open(f".cache/dump_wild_{TAG}.pkl", "rb"))
    by_row = collections.defaultdict(list)
    for t in D:
        by_row[t[0]].append(t[1:])
    z = np.load(f".cache/ps_legal_wild_{TAG}.npz")
    npz_ok = {(int(r), int(s)): bool(o) for r, s, o in zip(z["rows"], z["slots"], z["ok"])}

    # counts[space][strict][subset][level] -> list of live counts
    counts = {sp: {st: {ss: [[] for _ in FILTERS] for ss in ("all", "right", "wrong")}
                   for st in (False, True)} for sp in ("values", "tokens")}
    buckets = collections.Counter()
    gold_unreadable = collections.Counter()
    examples = []

    for i, row in enumerate(rows):
        doc = nlp(row["text"])
        sents = list(doc.sents)
        cands = G.numeral_candidates(doc)
        fverbs = G.finite_verbs(doc)
        feat = []
        for _, tok, v in cands:
            ch = G.governor_chain(doc, sents, tok)
            feat.append(dict(tok=tok, value=v,
                             owner=(ch["owner_key"] if ch else None),
                             type=head_noun_key(tok),
                             event=G.event_index_of(fverbs, tok)))
        given_list = [(f["var"], f["value"]) for f in row["factors"] if f["ftype"] == "given"]
        bound = G.bind_given_occurrences(doc, given_list, cands)
        tok2feat = {f["tok"].i: f for f in feat}

        for s in by_row[i]:
            j, gft, gop, gargs, gres, gdig = s[:6]
            if gft != 1:
                continue
            buckets["given slots"] += 1
            gtok = bound.get(gres)
            if gtok is None:
                buckets["no text occurrence (excluded)"] += 1
                continue
            buckets["lookups counted"] += 1
            gold = tok2feat[gtok.i]
            for k in ("owner", "type", "event"):
                if gold[k] is None:
                    gold_unreadable[k] += 1
            ok = npz_ok.get((i, j))
            subsets = ["all"] + ([] if ok is None else ["right" if ok else "wrong"])
            for strict in (False, True):
                for lv in range(len(FILTERS)):
                    lv_feats = [f for f in feat if f["tok"].i == gtok.i or live(gold, f, lv, strict)]
                    nv = len({f["value"] for f in lv_feats})
                    nt = len(lv_feats)
                    for ss in subsets:
                        counts["values"][strict][ss][lv].append(nv)
                        counts["tokens"][strict][ss][lv].append(nt)
                    if not strict and lv == 4 and nv >= 5 and len(examples) < 12:
                        examples.append((i, gold["value"], nv, row["text"][:220]))

    L = [f"THE CANDIDATE CENSUS -- body {TAG} (right/wrong split from its ps_legal)",
         "lookups: " + ", ".join(f"{k} {v}" for k, v in buckets.items()),
         "gold features unreadable: " + ", ".join(f"{k} {v}" for k, v in gold_unreadable.items()),
         ""]
    for sp in ("values", "tokens"):
        for strict in (False, True):
            L.append(f"== live {sp.upper()} per lookup, {'STRICT (unknown excluded)' if strict else 'LENIENT (unknown is live) [PRIMARY]'} ==")
            for ss in ("all", "right", "wrong"):
                L.append(f"  -- {ss} slots --")
                for lv, name in enumerate(FILTERS):
                    L.append(f"    {name:22s} {summarize(counts[sp][strict][ss][lv])}")
            L.append("")
    a4 = np.array(counts["values"][False]["all"][4])
    p2, p5 = float(np.mean(a4 <= 2)), float(np.mean(a4 >= 5))
    verdict = "CHEAP" if (p2 >= 0.80 and p5 < 0.20) else ("COSTLY" if p5 >= 0.20 else "MIXED")
    L.append(f"PINNED READING (F4, values, lenient, all lookups): <=2 {p2:.3f} (CHEAP needs >= 0.80), "
             f">=5 {p5:.3f} (COSTLY at >= 0.20) -> {verdict}")
    L.append("")
    L.append("F4 lookups with >= 5 live values (first 12):")
    for e in examples:
        L.append(f"  row {e[0]} gold {e[1]} live {e[2]}: {e[3]!r}")
    txt = "\n".join(L)
    print(txt)
    open(OUT, "w").write(txt + "\n")


if __name__ == "__main__":
    main()
