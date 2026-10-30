"""gsm_graph_keys.py -- the GRAPH arm's per-problem graph (ledger 2026-10-29, Bryce "word given"): for every item of every split in
.cache/gsmb_data.json, each numeral word's (OWNER, EVENT, KIND) key as gsm_graph_ceiling.keys reads it, turned into LOCAL ids
(owners / events numbered in first-appearance order within the problem; -1 = unknown = a wildcard the walk's switches pass) + the
MENTION words that carry the same phase tag (an owner's own word(s): the subject / possessor token; an event's verb). KIND = the k=3
type-forest class id (global; -1 unknown). CPU, spaCy. Output .cache/gsmb_graph_keys.json {split: [{own, ev, kind, own_m, ev_m}]}
(per bench word; own_m / ev_m = the id a mention word carries, -1 elsewhere)."""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
import gsm_bench_data as GD
import gsm_graph_ceiling as G
import gsm_trunk_states as GT
import gsm_type_forest as F

OUT = GD.V1.C("gsmb_graph_keys.json")


def graph(q, nlp, cls, cid):
    doc = nlp(q); spans = [m.span() for m in GD.TOK.finditer(q)]; nn = F.nouns(q, nlp); n = len(spans)
    persons = [(e.start_char, e.text.split()[0].lower()) for e in doc.ents if e.label_ == "PERSON"]
    def word_of(t):                                                      # the bench word holding spaCy token t
        return next((i for i, (a, b) in enumerate(spans) if a < t.idx + len(t) and b > t.idx), None)
    def who(t):
        w = t.text.lower()
        if w in G.NARR: return "narrator", None
        if w in G.PRON:
            prev = [p for c, p in persons if c < t.idx]
            return (prev[-1] if prev else None), None
        return t.lemma_.lower(), word_of(t)
    own, ev, kind = [-1] * n, [-1] * n, [-1] * n; own_m, ev_m = [-1] * n, [-1] * n; O, E = {}, {}
    for i, ((a, b), noun) in enumerate(zip(spans, nn)):
        if noun is None: continue
        tk = [t for t in doc if t.idx < b and t.idx + len(t) > a][-1]
        hd = tk.head if tk.dep_ in ("nummod", "quantmod") else tk
        o = next((who(c) for c in hd.children if c.dep_ == "poss"), None)
        v = hd
        while v.pos_ not in ("VERB", "AUX") and v.head.i != v.i: v = v.head
        e = v.lemma_.lower() if v.pos_ in ("VERB", "AUX") else None
        if o is None and e is not None: o = next((who(c) for c in v.children if c.dep_ in ("nsubj", "nsubjpass")), None)
        if o is not None and o[0] is not None:
            own[i] = O.setdefault(o[0], len(O))
            if o[1] is not None: own_m[o[1]] = own[i]
        if e is not None:
            ev[i] = E.setdefault(e, len(E)); w = word_of(v)
            if w is not None: ev_m[w] = ev[i]
        k = cls.get(noun); kind[i] = cid[k] if k is not None else -1
    for i, (a, b) in enumerate(spans):                                   # every other mention of a known owner name carries its tag
        w = q[a:b].lower()
        if w in O and own_m[i] < 0 and nn[i] is None: own_m[i] = O[w]
    return dict(own=own, ev=ev, kind=kind, own_m=own_m, ev_m=ev_m)


if __name__ == "__main__":
    import spacy
    assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
    nlp = spacy.load("en_core_web_sm"); fo = json.load(open(GD.V1.C("gsm_type_forest_k3.json")))
    cls = fo["parent"]; cid = {r: i for i, r in enumerate(sorted(set(cls.values())))}
    Dt, Q = GT.texts(); memo = {}; out = {}
    for sp in Dt:
        out[sp] = []
        for q in Q[sp]:
            if q not in memo: memo[q] = graph(q, nlp, cls, cid)
            out[sp].append(memo[q])
        print(f"[keys] {sp} {len(out[sp])} items | numerals with owner {sum(o >= 0 for g in out[sp] for o in g['own'])} | "
              f"with event {sum(e >= 0 for g in out[sp] for e in g['ev'])}", flush=True)
    json.dump(dict(n_kind=len(cid), keys=out), open(OUT, "w"))
