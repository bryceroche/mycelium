"""units_parse_read.py -- THE UNITS READ, REOPENED IN PARSE FORM (2026-10-18; the 10-13 surface-lexicon units census closed
NULL with the trigger 'a parse-based unit extractor'; Opus's ask; Bryce's word for the queue; bars pinned in the ledger first).
Zero GPU. Fixtures: the wild gold (wild_admitted_holdout, 311 rows; THE BAR) and the 3,672 unique verbatim GSM8K diet rows
(reported). A GIVEN's mention = the text numeral / number word with its value (all candidates must agree on the unit, else
uncovered). Its UNIT from the dependency parse (spaCy): '$' before -> dollar; 'cent(s)' -> cent; else the head noun of a
nummod / the noun the number modifies (hyphen compounds: '8-ounce cup' -> ounce); a RATE when the unit phrase is followed by
'per Y' / 'a(n) Y' / 'each' / 'every Y' (-> unit/Y, 'each' = unit/*). Relations are read ONLY where their vars are GIVENS
(no propagation through gold ops: that would use the answer's own wiring).
  OP FROM UNITS: a relation with >= 2 unit-known vars; predict add if every known unit is equal, else mul; vs the gold op.
  UNIT ALGEBRA CONSISTENCY (the 10-13 bar's form): a gold mul with all 3 vars known is legal if one is X/Y and another Y
  (or X/*) and the result X, or the result's unit equals one factor's (count x count); a gold add is legal if all equal.
Out: .cache/units_parse_read.txt"""
import collections, json, re
import spacy

nlp = spacy.load("en_core_web_sm")
OUT = []


def say(s=""):
    print(s, flush=True); OUT.append(s)


WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
         "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100}
CANON = {"buck": "dollar", "$": "dollar", "dollar": "dollar", "cent": "cent", "minute": "minute", "min": "minute", "hr": "hour"}
RATE_W = {"per", "a", "an", "every", "each"}


def tval(tok):
    s = tok.text.replace(",", "").lstrip("$")
    try:
        return float(s)
    except ValueError:
        return float(WORDS[tok.lower_]) if tok.lower_ in WORDS else None


def unit_of(doc, i):
    t = doc[i]
    if i > 0 and doc[i - 1].text == "$":
        u = "dollar"; j = i
    else:
        u = None; j = i
        h = t.head
        if t.dep_ in ("nummod", "quantmod") and h.pos_ in ("NOUN", "PROPN") and h.i > i:
            # '8-ounce cup': the number's head is 'cup' via 'ounce' (compound); take the first noun after the number
            k = i + 1
            while k < len(doc) and doc[k].text in ("-",):
                k += 1
            u = doc[k].lemma_.lower() if k < len(doc) and doc[k].pos_ in ("NOUN", "PROPN") else h.lemma_.lower(); j = k
        elif i + 1 < len(doc) and doc[i + 1].pos_ == "NOUN":
            u = doc[i + 1].lemma_.lower(); j = i + 1
        elif i + 2 < len(doc) and doc[i + 1].pos_ == "ADJ" and doc[i + 2].pos_ == "NOUN":
            u = doc[i + 2].lemma_.lower(); j = i + 2
    if u is None:
        return None
    u = CANON.get(u, u)
    # a rate: the unit phrase followed by per / a(n) / every Y, or 'each'
    k = j + 1
    while k < len(doc) and doc[k].pos_ in ("ADJ", "NOUN") and k <= j + 2 and doc[k].lower_ not in RATE_W:
        k += 1
    if k < len(doc) and doc[k].lower_ in RATE_W:
        if doc[k].lower_ == "each":
            return (u, "*")
        if k + 1 < len(doc) and doc[k + 1].pos_ == "NOUN":
            return (u, CANON.get(doc[k + 1].lemma_.lower(), doc[k + 1].lemma_.lower()))
    return (u, None)


def read(rows, name):
    docs = nlp.pipe([r["text"] for r in rows], batch_size=128)
    C = collections.Counter()
    for r, doc in zip(rows, docs):
        nums = [(tval(t), t.i) for t in doc if t.like_num or re.fullmatch(r"\$?\d[\d,]*(\.\d+)?", t.text)]
        U = {}
        for f in r["factors"]:
            if f["ftype"] != "given":
                continue
            C["givens"] += 1
            cand = {unit_of(doc, i) for v, i in nums if v is not None and abs(v - float(f["value"])) < 1e-9}
            if len(cand) == 1 and None not in cand:
                U[f["var"]] = cand.pop(); C["given_unit"] += 1
            elif not cand:
                C["given_no_mention"] += 1
            elif None in cand and len(cand) == 1:
                C["given_no_unit"] += 1
            else:
                C["given_ambiguous"] += 1
        for f in r["factors"]:
            if f["ftype"] != "rel":
                continue
            C["rels"] += 1; C["rels_" + f["op"]] += 1
            vs = list(f["args"]) + [f["result"]]
            known = [U[v] for v in vs if v in U]
            if len(known) >= 2:
                C["op_cov"] += 1
                pred = "add" if len({k for k in known}) == 1 else "mul"
                C["op_ok"] += pred == f["op"]; C[f"op_cov_{f['op']}"] += 1; C[f"op_ok_{f['op']}"] += pred == f["op"]
            if all(v in U for v in vs):
                C["alg_cov"] += 1; C["alg_cov_" + f["op"]] += 1
                a, b, c = (U[v] for v in vs)
                if f["op"] == "add":
                    legal = a == b == c
                else:
                    legal = (c in (a, b)) or any(x[1] in ("*", y[0]) and x[0] == c[0] and y[1] is None for x, y in ((a, b), (b, a)))
                C["alg_ok_" + f["op"]] += legal
    cov = C["op_cov"] / max(1, C["rels"]); acc = C["op_ok"] / max(1, C["op_cov"])
    say(f"{name}: rows {len(rows)} | givens {C['givens']}: unit {C['given_unit']} ({C['given_unit'] / C['givens']:.3f}), no unit "
        f"{C['given_no_unit']}, ambiguous {C['given_ambiguous']}, no mention (hidden) {C['given_no_mention']}")
    say(f"  OP FROM UNITS: relations {C['rels']} (add {C['rels_add']} / mul {C['rels_mul']}) | covered {C['op_cov']} ({cov:.3f}) | "
        f"ACCURACY {acc:.3f} (add {C['op_ok_add']}/{C['op_cov_add']}, mul {C['op_ok_mul']}/{C['op_cov_mul']}) | majority-class "
        f"baseline on covered {max(C['op_cov_add'], C['op_cov_mul']) / max(1, C['op_cov']):.3f}")
    for op in ("add", "mul"):
        n = C["alg_cov_" + op]
        say(f"  UNIT ALGEBRA: gold {op} with all 3 units known {n} | legal {C['alg_ok_' + op]} ({C['alg_ok_' + op] / max(1, n):.3f})")
    return cov, acc, C


wild = [json.loads(l) for l in open(".cache/wild_admitted_holdout.jsonl")]
cov, acc, Cw = read(wild, "WILD (the bar)")
seen, diet = set(), []
for l in open(".cache/form_mix_real.jsonl"):
    r = json.loads(l); g = r.get("gen") or {}
    if g.get("src") == "gsm8k" and not g.get("resample") and r["text"] not in seen:
        seen.add(r["text"]); diet.append(r)
read(diet, "DIET GSM8K (reported)")
leg_mul = Cw["alg_ok_mul"] / max(1, Cw["alg_cov_mul"])
v1 = ("UNITS CARRY THE OP" if (acc >= 0.85 and cov >= 0.30) else "NULL" if acc <= 0.74 else "WEAK (between the bars)")
say(f"VERDICT op-from-units (pinned: >= 0.85 on >= 0.30 coverage; NULL <= 0.74 = the op head's wild field): {v1}")
say(f"VERDICT unit-algebra consistency, gold mul (pinned 10-13 bar >= 0.95): " + ("VOID (n = 0: a relation's RESULT is never a given, "
    "so without propagation no relation has all three units; a design flaw of this read, not a measurement)" if not Cw["alg_cov_mul"]
    else f"{'MET' if leg_mul >= 0.95 else 'MISS'} ({leg_mul:.3f})"))
open(".cache/units_parse_read.txt", "w").write("\n".join(OUT) + "\n")
