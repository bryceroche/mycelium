"""op_phrase_census.py -- THE HARVESTED PHOTOBOOTH, a census (2026-10-17; Bryce's concept-primitive table + Opus: "mine the
corpus for the real ways GSM8K says add, subtract, multiply, divide, with the context attached"). Zero GPU; no loss.
For every GSM8K TRAIN calculator step <<a op b = c>> with a single binary op whose BOTH operands are numerals of the
QUESTION, and both sit in ONE question sentence, the sentence is that step's wording. Two views of the cue:
  SENTENCE  the content lemmas of that sentence (spaCy; numbers, names, stopwords dropped)
  BETWEEN   the lemmas strictly between the two operand mentions (the tightest context)
Reported: coverage (steps anchored / all single-op steps), per-op distinct phrasings, and THE AMBIGUITY TABLE: for each
frequent cue lemma, its op distribution and op-entropy -- a rule table 'word -> op' is only as good as this table says.
Out: .cache/op_phrase_census.txt + .cache/op_phrase_table.json (lemma -> op counts, both views)."""
import collections, glob, json, math, re
import pyarrow.parquet as pq
import spacy

snap = glob.glob(".cache/gsm8k/datasets--openai--gsm8k/snapshots/*/main/")[0]
T = pq.read_table(snap + "train-00000-of-00001.parquet").to_pydict()
CALC = re.compile(r"<<\s*([\d\.,]+)\s*([\+\-\*/])\s*([\d\.,]+)\s*=\s*([\d\.,\-]+)\s*>>")
NUM = re.compile(r"(?<![\w.])\$?(\d[\d,]*(?:\.\d+)?)")
OPN = {"+": "add", "-": "sub", "*": "mul", "/": "div"}
nlp = spacy.load("en_core_web_sm")
OUT = []


def say(s=""):
    print(s, flush=True); OUT.append(s)


def val(s):
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


C = collections.Counter()
SENT = collections.defaultdict(collections.Counter)
BETW = collections.defaultdict(collections.Counter)
PHR = collections.defaultdict(set)
docs = nlp.pipe(T["question"], batch_size=256)
for q, a, d in zip(T["question"], T["answer"], docs):
    lits = [(m.start(1), m.end(1), val(m.group(1))) for m in NUM.finditer(q)]
    for m in re.finditer(r"<<[^<>]*>>", a):
        C["steps"] += 1
        cm = CALC.fullmatch(m.group(0))
        if not cm:
            C["multi_op_or_unparsed"] += 1; continue
        x, op, y = val(cm.group(1)), cm.group(2), val(cm.group(3))
        C["single_op"] += 1
        px = [l for l in lits if l[2] == x]; py = [l for l in lits if l[2] == y and l not in px[:1]] if x == y else [l for l in lits if l[2] == y]
        if not px or not py:
            C["operand_not_in_question"] += 1; continue
        best = None
        for sent in d.sents:
            ix = [l for l in px if sent.start_char <= l[0] < sent.end_char]
            iy = [l for l in py if sent.start_char <= l[0] < sent.end_char and (not ix or l != ix[0])]
            if ix and iy:
                best = (sent, ix[0], iy[0]); break
        if best is None:
            C["operands_in_different_sentences"] += 1; continue
        sent, lx, ly = best
        C["anchored"] += 1; C["anchored_" + OPN[op]] += 1
        lo, hi = sorted([lx[1], ly[0]]) if lx[0] < ly[0] else sorted([ly[1], lx[0]])
        keep = lambda t: t.is_alpha and not t.is_stop and t.ent_type_ not in ("PERSON", "CARDINAL", "MONEY") and len(t) > 1
        sl = sorted({t.lemma_.lower() for t in sent if keep(t)})
        bl = [t.lemma_.lower() for t in sent if keep(t) and lo <= t.idx < hi]
        for w in sl: SENT[w][OPN[op]] += 1
        for w in set(bl): BETW[w][OPN[op]] += 1
        PHR[OPN[op]].add(" ".join(bl) if bl else "(adjacent)")
ent = lambda c: -sum(v / sum(c.values()) * math.log2(v / sum(c.values())) for v in c.values())
s1 = C["single_op"]
say(f"GSM8K train calculator steps {C['steps']} | single binary op {s1} | ANCHORED (both operands numerals of one question sentence) "
    f"{C['anchored']} ({C['anchored'] / s1:.3f} of single-op)")
say(f"  not anchored: an operand not a question numeral {C['operand_not_in_question']} (a derived value or an implied constant) | "
    f"operands in different sentences {C['operands_in_different_sentences']} | multi-op / unparsed {C['multi_op_or_unparsed']}")
say("  anchored by op: " + " | ".join(f"{o} {C['anchored_' + o]}" for o in ("add", "sub", "mul", "div")))
say("  distinct BETWEEN phrasings per op: " + " | ".join(f"{o} {len(PHR[o])}" for o in ("add", "sub", "mul", "div")))
for name, tab in (("BETWEEN (lemmas between the operands)", BETW), ("SENTENCE (content lemmas of the step's sentence)", SENT)):
    say(f"\nAMBIGUITY TABLE, {name}: top 40 lemmas by count | op shares | entropy (bits; 0 = one op, 2 = uniform over 4)")
    top = sorted(tab, key=lambda w: -sum(tab[w].values()))[:40]
    for w in top:
        c = tab[w]; n = sum(c.values())
        say(f"  {w:14s} n {n:5d} | " + " ".join(f"{o} {c[o] / n:.2f}" for o in ("add", "sub", "mul", "div")) + f" | H {ent(c):.2f}")
    mass = sum(sum(c.values()) for c in tab.values()); pure = sum(sum(c.values()) for c in tab.values() if max(c.values()) / sum(c.values()) >= 0.9)
    say(f"  share of cue mass on lemmas >= 90% one op: {pure / mass:.3f}")
for w in ("receive", "give", "more", "less", "each", "per", "total", "leave", "spend", "buy", "share", "split", "twice", "half", "every", "cost"):
    for name, tab in (("between", BETW), ("sentence", SENT)):
        c = tab.get(w)
        if c:
            n = sum(c.values()); say(f"  WATCH {w:8s} ({name:8s}) n {n:5d} | " + " ".join(f"{o} {c[o] / n:.2f}" for o in ("add", "sub", "mul", "div")))
open(".cache/op_phrase_census.txt", "w").write("\n".join(OUT) + "\n")
json.dump({"between": {w: dict(c) for w, c in BETW.items()}, "sentence": {w: dict(c) for w, c in SENT.items()}}, open(".cache/op_phrase_table.json", "w"))
