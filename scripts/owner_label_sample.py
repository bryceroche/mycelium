"""owner_label_sample.py -- zero-GPU: sample 100 numerals from the wild holdout,
apply governor_census's exact owner extraction (as climb_targets.climb_links does),
write .cache/owner_label_sample.jsonl (+ .txt for reading). Judgments are added later
by hand into the jsonl. Reads only the `text` field (custody-gold law)."""
import json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spacy
from governor_census import numeral_candidates, find_owner
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUTJ = os.path.join(R, ".cache/owner_label_sample.jsonl")
OUTT = os.path.join(R, ".cache/owner_label_sample.txt")
for p in (OUTJ, OUTT):
    assert not os.path.exists(p), f"refuse overwrite {p}"
nlp = spacy.load("en_core_web_sm", disable=["ner"])
texts = [json.loads(l)["text"] for l in open(os.path.join(R, ".cache/wild_admitted_holdout.jsonl"))]
allnum = []
for i, doc in enumerate(nlp.pipe(texts, batch_size=64)):
    for ti, tok, v in numeral_candidates(doc):
        allnum.append((i, doc, tok))
print("total numerals", len(allnum), "rows", len(texts))
random.seed(0)
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
    o, kind, vu = find_owner(verb, head_noun)
    recs.append(dict(n=n, row=i, text=texts[i], numeral=tok.text, char_offset=tok.idx,
        head_noun=head_noun.text if head_noun is not None else None,
        verb=verb.text if verb is not None else None,
        owner=(" ".join(w.text for w in o.subtree) if o is not None else None),
        owner_tok=o.text if o is not None else None, owner_kind=kind,
        is_pronoun=bool(o is not None and o.pos_ == "PRON")))
with open(OUTJ, "w") as f:
    for r in recs: f.write(json.dumps(r) + "\n")
with open(OUTT, "w") as f:
    for r in recs:
        t = r["text"]; a = r["char_offset"]
        m = t[:a] + "[[" + t[a:a+len(r["numeral"])] + "]]" + t[a+len(r["numeral"]):]
        f.write(f"#{r['n']} row{r['row']} owner={r['owner']!r}({r['owner_kind']},pron={r['is_pronoun']}) noun={r['head_noun']} verb={r['verb']}\n{m}\n\n")
