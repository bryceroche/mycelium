"""panama_guard.py — THE PANAMA-HAT WATCHLIST GUARD (2026-07-20, built on
the word). The constructional-novelty wall for the species the panel exam
industrialized: style-native text carrying constructions the trained
register never taught.

JURISDICTION (the OOD decomposition's split, instrumented at the input
skin): the mouth asks 'is this text's STYLE familiar?' (continuous,
distance); the guard asks 'does this text contain CONSTRUCTIONS whose
bindings I was never taught?' (discrete, coverage). Zero-parameter,
input-side, selection-safe by position. ECONOMICS: a flag bars the
CERTIFY tier and routes to answer/abstain — it can cost coverage, never
precision.

MECHANISM: word-level n-grams (n=2..4) with values abstracted (numbers ->
<NUM>, single letters -> <VAR>) over the trained register (TRAIN_SOURCES
texts + book-4 dialects) = the TAUGHT-CONSTRUCTION LEXICON. FLAG RULE
(pinned before any read): input contains >=1 novel abstracted n-gram of
length >=3. VALIDATION (pinned predictions): the 20 adversarial scope
specimens flag 20/20; bigtest flags near zero (its constructions ARE the
register); the flag rate on native fixtures is the guard's priced
coverage cost.
"""
import json, sys, os, re
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from collections import Counter

TRAIN_SOURCES = {
    "nl":      ".cache/algebra_nl_train.jsonl",
    "alg2":    ".cache/algebra2_nl_train.jsonl",
    "alg3":    ".cache/algebra3_nl_train.jsonl",
    "alg4":    ".cache/algebra4_nl_train.jsonl",
    "verbose": ".cache/algv_train_verbose.jsonl",
    "dag6":    ".cache/dag_train.jsonl",
    "dag7":    ".cache/dag7_train.jsonl",
    "dag8":    ".cache/dag8_train.jsonl",
    "dag10":   ".cache/dag10_train.jsonl",
    "dag11":   ".cache/dag11_train.jsonl",
}
LEXICON_PATH = ".cache/panama_guard_lexicon.json"


def abstract(text):
    ws = re.findall(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]", text)
    out = []
    for w in ws:
        if w.isdigit():
            out.append("<NUM>")
        elif len(w) == 1 and w.isalpha():
            out.append("<VAR>")
        else:
            out.append(w.lower())
    return out


def ngrams(toks, lo=2, hi=4):
    for n in range(lo, hi + 1):
        for i in range(len(toks) - n + 1):
            yield " ".join(toks[i:i + n])


def build_lexicon():
    lex = set()
    n_rows = 0
    for src, path in TRAIN_SOURCES.items():
        try:
            for line in open(path):
                t = json.loads(line).get("text", "")
                lex.update(ngrams(abstract(t)))
                n_rows += 1
        except FileNotFoundError:
            print(f"  [warn] missing {path}")
    try:
        for line in open(".cache/book4_prose_pairs.jsonl"):
            d = json.loads(line)["gen"].get("dialect", "")
            if d:
                lex.update(ngrams(abstract(d)))
                n_rows += 1
    except FileNotFoundError:
        pass
    print(f"[guard] lexicon: {len(lex)} abstracted n-grams from {n_rows} register rows")
    json.dump(sorted(lex), open(LEXICON_PATH, "w"))
    return lex


def load_lexicon():
    if os.path.exists(LEXICON_PATH):
        return set(json.load(open(LEXICON_PATH)))
    return build_lexicon()


def guard(text, lex):
    """Returns (flagged, novel_constructions) — the deployable read."""
    toks = abstract(text)
    novel = [g for g in ngrams(toks, 3, 4) if g not in lex]
    return (len(novel) > 0), novel


def validate():
    lex = load_lexicon()
    # population 1: the adversarial scope fixture (must flag 20/20)
    VALS = [(7, 4), (9, 5), (8, 3), (11, 6), (12, 7), (13, 4), (10, 3),
            (15, 8), (14, 5), (9, 2)]
    D = "Consider the numbers "
    spec = []
    for a, b in VALS:
        spec.append(D + f"a, b, c. a is {a}. b is {b}. The difference of the squares of a and b equals c. What is c?")
        spec.append(D + f"a, b, c. a is {a}. b is {b}. The square of the difference of a and b equals c. What is c?")
    sf = [guard(t, lex) for t in spec]
    n_flag = sum(1 for f, _ in sf if f)
    ex_novel = next((nv for f, nv in sf if f), [])[:4]
    print(f"[guard] SPECIMENS: flagged {n_flag}/20 (pinned: 20)  "
          f"example novel constructions: {ex_novel}")

    # population 2: bigtest (native fixture — the coverage cost)
    rows = [json.loads(l) for l in open(".cache/algebra_nl_bigtest.jsonl")]
    bf = [guard(r["text"], lex)[0] for r in rows]
    print(f"[guard] BIGTEST: flagged {sum(bf)}/{len(bf)} "
          f"({sum(bf)/len(bf):.2%}) — the certify-coverage price")

    # population 3: book-4 banked dialects (taught pages — should pass)
    b4 = [json.loads(l) for l in open(".cache/book4_prose_pairs.jsonl")]
    b4d = [g["gen"]["dialect"] for g in b4 if g["gen"].get("dialect")]
    b4f = [guard(t, lex)[0] for t in b4d]
    print(f"[guard] BOOK-4 DIALECTS: flagged {sum(b4f)}/{len(b4f)} "
          f"(in-lexicon by construction — sanity)")

    ok = n_flag == 20
    print(f"[guard] VALIDATION: {'PASS — the guard catches the species' if ok else 'FAIL'}")
    json.dump({"specimens_flagged": n_flag, "bigtest_flagged": int(sum(bf)),
               "bigtest_n": len(bf), "book4_flagged": int(sum(b4f)),
               "book4_n": len(b4f), "lexicon_size": len(lex),
               "pass": bool(ok)},
              open(".cache/panama_guard_validation.json", "w"))
    return ok


if __name__ == "__main__":
    if "--build" in sys.argv:
        build_lexicon()
    sys.exit(0 if validate() else 1)
