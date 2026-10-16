"""who_when_census.py -- THE (WHO, WHEN) CEILING (2026-10-16; registered + bars pinned in the ledger before this ran).
Zero GPU, zero model: wild GOLD only (.cache/wild_admitted_holdout.jsonl) + spaCy (governor_census.py's helpers by import).

QUESTION: can every gold relation argument be written as a (who, when) address -- an entity register plus an event
phase -- so that a per-problem register tree (entity -> event -> quantity) is the right shape for PLANNING?

LABELS (text-only; gold supplies only the graph's own wiring, never the labels' content):
  GIVEN var    who = its governor owner_key (None if the numeral is unbound or its chain unresolvable);
               when = its event index (the governing finite verb's ordinal in text order).
  DERIVED var  labeled when the relation that INTRODUCES it is reached (gold order; the introduced var = the first
               not-yet-seen var in args + [result]); its operands = the relation's other vars:
               who = the operands' common owner (None operands ignored); different owners -> MIXED(frozenset);
               all None -> None. when = the max operand event (None ignored).
  PRONOUN PASS (the 'coref' column): a pronoun owner (he/she/they/it/his/her/their/...) is replaced by the nearest
               PRECEDING non-pronoun owner in the text (a crude resolver; reported beside the raw labels).
ADDRESSABLE  an argument a of relation r: among the vars introduced BEFORE r, exactly one carries a's (who, when).
LATEST-ADDR  a is the unique max-when var among the vars introduced before r with a's who (the 'latest-before' rule).
EITHER       ADDRESSABLE or LATEST-ADDR.
FAILURE ANATOMY: WHO-NONE (no owner) / WHEN-NONE / MIXED (a cross-entity total) / COLLIDE-GG, -GD, -DD (another var
  shares the exact (who, when): given-given, given-derived, derived-derived).
Reported over ALL args, GIVEN args and DERIVED args. Out: .cache/who_when_census.txt (+ .json).
"""
import collections
import json
import sys

sys.path.insert(0, "scripts")
import spacy  # noqa: E402

import governor_census as G  # noqa: E402

PRON = {"he", "she", "they", "it", "him", "her", "them", "his", "hers", "their", "theirs", "its", "i", "we", "you",
        "me", "us", "my", "our", "your", "this", "that", "these", "those", "each", "everyone", "who", "which"}


def label_row(nlp, row):
    doc = nlp(row["text"]); sl = list(doc.sents); cands = G.numeral_candidates(doc); fverbs = G.finite_verbs(doc)
    gl = [(f["var"], f["value"]) for f in row["factors"] if f["ftype"] == "given"]
    bound = G.bind_given_occurrences(doc, gl, cands)
    raw, cor = {}, {}
    named = []   # (char_start, owner_key) of non-pronoun owners, for the pronoun pass
    chains = {}
    for v, tok in bound.items():
        ch = G.governor_chain(doc, sl, tok) if tok is not None else None
        chains[v] = (tok, ch)
        if ch is not None and ch["owner_key"] not in PRON:
            named.append((ch["char_start"], ch["owner_key"]))
    named.sort()
    for v, (tok, ch) in chains.items():
        when = G.event_index_of(fverbs, tok) if tok is not None else None
        who = ch["owner_key"] if ch is not None else None
        raw[v] = (who, when, "G")
        if who in PRON:
            prev = [k for c, k in named if c <= ch["char_start"]]
            who = prev[-1] if prev else (named[0][1] if named else who)
        cor[v] = (who, when, "G")
    return raw, cor


def derive(labels, operands):
    whos = {labels[o][0] for o in operands if o in labels and labels[o][0] is not None}
    whens = [labels[o][1] for o in operands if o in labels and labels[o][1] is not None]
    who = None if not whos else (next(iter(whos)) if len(whos) == 1 else ("MIXED", frozenset(whos)))
    return (who, max(whens) if whens else None, "D")


VARIANT = "BASE"   # BASE (pinned) / KIND (+ a given-vs-derived role level) / GROUP (KIND + group entities for MIXED)


def key_of(lab):
    who, when, kind = lab
    if VARIANT == "BASE":
        return (who, when)
    return (who, when, kind)


def census(rows, nlp):
    C = {k: collections.Counter() for k in ("raw", "coref")}
    for row in rows:
        raw, cor = label_row(nlp, row)
        for tag, lab0 in (("raw", raw), ("coref", cor)):
            lab = dict(lab0); seen = [v for v in lab]   # givens are 'introduced' at their own slot
            introduced = []
            for f in row["factors"]:
                if f["ftype"] == "given":
                    introduced.append(f["var"]); continue
                vs = list(f["args"]) + [f["result"]]
                before = list(introduced)
                for a in f["args"]:
                    if a not in before:
                        continue   # the relation introduces this var itself (inverse form): not a lookup
                    kind = "G" if lab[a][2] == "G" else "D"
                    who, when, _ = lab[a]
                    C[tag]["n"] += 1; C[tag][f"n_{kind}"] += 1
                    same = [b for b in before if b != a and key_of(lab[b]) == key_of(lab[a])]
                    same_who = [b for b in before if lab[b][0] == who]
                    grp_ok = VARIANT == "GROUP"
                    ok_addr = who is not None and when is not None and (grp_ok or not isinstance(who, tuple)) and not same
                    whens = [lab[b][1] for b in same_who if lab[b][1] is not None]
                    _sw = [b for b in same_who if VARIANT == "BASE" or lab[b][2] == lab[a][2]]
                    _ws = [lab[b][1] for b in _sw if lab[b][1] is not None]
                    ok_latest = (who is not None and (grp_ok or not isinstance(who, tuple)) and when is not None
                                 and when == max(_ws) and sum(1 for b in _sw if lab[b][1] == when) == 1)
                    for k_, ok in (("addr", ok_addr), ("latest", ok_latest), ("either", ok_addr or ok_latest)):
                        C[tag][k_] += ok; C[tag][f"{k_}_{kind}"] += ok
                    if not (ok_addr or ok_latest):
                        if who is None:
                            why = "WHO-NONE"
                        elif isinstance(who, tuple):
                            why = "MIXED"
                        elif when is None:
                            why = "WHEN-NONE"
                        elif same:
                            why = "COLLIDE-" + kind + ("G" if lab[same[0]][2] == "G" else "D")
                        else:
                            why = "NOT-LATEST-NOT-UNIQUE"
                        C[tag]["why_" + why] += 1; C[tag][f"why_{kind}_" + why] += 1
                new = [v for v in vs if v not in introduced]
                for v in new:
                    lab[v] = derive(lab, [u for u in vs if u != v])
                    introduced.append(v)
    return C


def main():
    global VARIANT
    rows = [json.loads(l) for l in open(".cache/wild_admitted_holdout.jsonl")]
    nlp = spacy.load("en_core_web_sm")
    lines, out = [], {}
    for VARIANT in ("BASE", "KIND", "GROUP"):
        C = census(rows, nlp)
        _l, _o = report(C)
        lines += [f"\n################ VARIANT {VARIANT}{' (PINNED)' if VARIANT == 'BASE' else ' (diagnostic: the anatomy step the pinned prediction named)'}"] + _l
        out[VARIANT] = _o
    txt = "\n".join(lines)
    print(txt)
    open(".cache/who_when_census.txt", "w").write(txt + "\n")
    json.dump(out, open(".cache/who_when_census.json", "w"), indent=1)


def report(C):
    lines = ["THE (WHO, WHEN) CEILING on wild GOLD (311 rows; text-only labels; ledger 2026-10-16 bars: >= 0.85 = the tree "
             "is the right shape; coordinator's pinned prediction 0.55-0.70, derived args weaker)"]
    out = {}
    for tag in ("raw", "coref"):
        c = C[tag]; n = c["n"]; out[tag] = dict(c)
        lines.append(f"\n== {tag.upper()} labels: {n} relation-argument lookups (given {c['n_G']}, derived {c['n_D']})")
        for k_ in ("addr", "latest", "either"):
            lines.append(f"  {k_:7s} ALL {c[k_] / n:.3f} | GIVEN {c[k_ + '_G'] / max(c['n_G'], 1):.3f} | "
                         f"DERIVED {c[k_ + '_D'] / max(c['n_D'], 1):.3f}")
        fails = n - c["either"]
        lines.append(f"  failure anatomy ({fails} lookups neither addressable nor latest):")
        for k_, v in sorted(((k, v) for k, v in c.items() if k.startswith("why_") and k.count("_") == 1), key=lambda x: -x[1]):
            gk = c.get("why_G_" + k_[4:], 0); dk = c.get("why_D_" + k_[4:], 0)
            lines.append(f"    {k_[4:]:22s} {v:4d} ({v / max(fails, 1):.3f})  given {gk} / derived {dk}")
    return lines, out


if __name__ == "__main__":
    main()
