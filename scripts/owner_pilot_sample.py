"""owner_pilot_sample.py -- zero-GPU: the LABELER PILOT's sample (2026-10-22; Bryce's word "have sonnet or haiku curate").
100 numerals from the GSM8K TRAIN split (question text only -- custody-gold law: the answer field is never read), with every
text that appears in a wild holdout / measurement fixture EXCLUDED (the 10-22 ruling: never prompt an LLM with eval items --
the first sample, owner_label_sample, came from the wild holdout and so cannot be the pilot). spaCy's owner per numeral via
governor_census's exact extraction, as in owner_label_sample.py. Writes:
  .cache/owner_pilot_train100.jsonl     spaCy labels (HIDDEN from the LLM labelers and from the hand-check sheet)
  .cache/owner_pilot_blind100.txt       numbered texts with [[numeral]] marked, nothing else (the labelers' input)
  .cache/owner_pilot_handcheck30.txt    30 of the 100, stratified by spaCy's construction, blank OWNER lines (Bryce's gold)
"""
import glob, json, os, random, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyarrow.parquet as pq
import spacy
from governor_census import numeral_candidates, find_owner
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUTJ, OUTB, OUTH = (os.path.join(R, ".cache", f) for f in ("owner_pilot_train100.jsonl", "owner_pilot_blind100.txt", "owner_pilot_handcheck30.txt"))
for p in (OUTJ, OUTB, OUTH):
    assert not os.path.exists(p), f"refuse overwrite {p}"
norm = lambda s: re.sub(r"\s+", " ", s.strip().lower())
fixtures = set()
for f in glob.glob(os.path.join(R, ".cache/wild_*holdout*.jsonl")):
    fixtures |= {norm(json.loads(l)["text"]) for l in open(f)}
train = pq.read_table(glob.glob(os.path.join(R, ".cache/gsm8k/datasets--openai--gsm8k/snapshots/*/main/train-*.parquet"))[0], columns=["question"]).to_pydict()
texts = [q for q in train["question"] if norm(q) not in fixtures]
print("train questions", len(train["question"]), "after fixture exclusion", len(texts), "fixture texts", len(fixtures))
random.seed(1022)
texts = random.sample(texts, 400)                                   # a pool; numerals are sampled from it
nlp = spacy.load("en_core_web_sm", disable=["ner"])
allnum = [(i, doc, tok) for i, doc in enumerate(nlp.pipe(texts, batch_size=64)) for _, tok, _ in numeral_candidates(doc)]
pick = sorted(random.sample(range(len(allnum)), 100))
recs = []
for n, k in enumerate(pick):
    i, doc, tok = allnum[k]
    head_noun = verb = None
    for anc in tok.ancestors:
        if head_noun is None and anc.pos_ in ("NOUN", "PROPN"):
            head_noun = anc
        if verb is None and anc.pos_ in ("VERB", "AUX"):
            verb = anc
            break
    if head_noun is None and tok.head is not tok:
        head_noun = tok.head
    o, kind, _ = find_owner(verb, head_noun)
    recs.append(dict(n=n, text=texts[i], numeral=tok.text, char_offset=tok.idx,
                     spacy_owner=(" ".join(w.text for w in o.subtree) if o is not None else None), spacy_kind=kind,
                     spacy_pronoun=bool(o is not None and o.pos_ == "PRON")))
mark = lambda r: r["text"][:r["char_offset"]] + "[[" + r["numeral"] + "]]" + r["text"][r["char_offset"] + len(r["numeral"]):]
with open(OUTJ, "w") as f:
    for r in recs:
        f.write(json.dumps(r) + "\n")
with open(OUTB, "w") as f:
    for r in recs:
        f.write(f"#{r['n']}\n{mark(r)}\n\n")
strata = {"pronoun": [r for r in recs if r["spacy_pronoun"]], "none": [r for r in recs if r["spacy_owner"] is None],
          "plain": [r for r in recs if r["spacy_owner"] is not None and not r["spacy_pronoun"]]}
print({k: len(v) for k, v in strata.items()})
quota = {"pronoun": 10, "none": 5}; quota["plain"] = 30 - sum(min(q, len(strata[k])) for k, q in quota.items())
hc = [r for k, q in quota.items() for r in random.sample(strata[k], min(q, len(strata[k])))]
random.shuffle(hc)
with open(OUTH, "w") as f:
    f.write("HAND-CHECK (Bryce): for each [[numeral]] write its TRUE OWNER -- the entity the quantity belongs to; resolve a pronoun "
            "to the name; NONE if the quantity belongs to no entity (a unit, a generic price/rate, an ordinal). Machine labels hidden.\n\n")
    for r in hc:
        f.write(f"#{r['n']}\n{mark(r)}\nOWNER: \n\n")
print("wrote", OUTJ, OUTB, OUTH)
