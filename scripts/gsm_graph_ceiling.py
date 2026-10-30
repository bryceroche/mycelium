"""gsm_graph_ceiling.py -- G-CEIL (ledger 2026-10-29 18:28, idea C reopened as a GRAPH WALK): would a per-problem graph key
(OWNER, KIND, EVENT) on every numeral separate VANILLA's wrong picks from gold? Zero-training, CPU.
OWNER = the possessor of the numeral's noun, else the nsubj of the governing verb (he/she/his/her/him/they/their/them -> the last
PERSON entity before it; I/my/me -> 'narrator'); KIND = the k=3 type-forest class (gsm_type_forest); EVENT = the governing verb
lemma. Reads on TEST: (i) KEY SUFFICIENCY = share of gold numeral operands whose key no question numeral of a DIFFERENT value holds
(the walk could land on gold by key alone); (ii) KEY SEPARABILITY = share of VANILLA's wrong numeral picks whose key differs from
every gold member's in a field known on both sides. Writes .cache/gsm_graph_ceiling.txt."""
import collections, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
import gsm_bench_data as GD
import gsm_trunk_states as GT
import gsm_type_forest as F

PRON = {"he", "she", "his", "her", "him", "they", "their", "them", "hers", "theirs"}
NARR = {"i", "my", "me", "we", "our", "us"}


def keys(q, nlp, cls):
    """-> per bench word: (owner, kind, event) for numeral words, else None."""
    doc = nlp(q); spans = [m.span() for m in GD.TOK.finditer(q)]; nn = F.nouns(q, nlp); out = []
    persons = [(e.start_char, e.text.split()[0].lower()) for e in doc.ents if e.label_ == "PERSON"]
    def who(t):
        w = t.text.lower()
        if w in NARR: return "narrator"
        if w in PRON:
            prev = [p for c, p in persons if c < t.idx]
            return prev[-1] if prev else None
        return t.lemma_.lower()
    for (a, b), noun in zip(spans, nn):
        if noun is None: out.append(None); continue
        tk = [t for t in doc if t.idx < b and t.idx + len(t) > a][-1]
        hd = tk.head if tk.dep_ in ("nummod", "quantmod") else tk
        own = next((who(c) for c in hd.children if c.dep_ == "poss"), None)
        v = hd
        while v.pos_ not in ("VERB", "AUX") and v.head.i != v.i: v = v.head          # spaCy tokens are fresh objects: compare by index
        ev = v.lemma_.lower() if v.pos_ in ("VERB", "AUX") else None
        if own is None and ev is not None: own = next((who(c) for c in v.children if c.dep_ in ("nsubj", "nsubjpass")), None)
        kind = cls.get(noun, noun if noun == F.DIM else None)
        out.append((own, kind, ev))
    return out


if __name__ == "__main__":
    import spacy
    nlp = spacy.load("en_core_web_sm")
    fo = json.load(open(GD.V1.C("gsm_type_forest_k3.json"))); cls = fo["parent"]
    P = json.load(open(GD.V1.C("gsmb_picks_VANILLA_s0.json"))); Dt, Q = GT.texts()
    suff, sep, fields = [], [], collections.Counter()
    for it, q, pk in zip(Dt["TEST"], Q["TEST"], P["TEST"]):
        K = keys(q, nlp, cls); vals = it["vals"]; n = len(it["toks"])
        for k, (gop, A, B) in enumerate(it["prog"]):
            for S in (A, B):
                if S["t"]:
                    g = S["t"][0]; kg = K[g]
                    if kg is not None and all(x is not None for x in kg):
                        suff.append(not any(K[j] == kg and vals[j] is not None and not GD.V1.near(vals[j], vals[g]) for j in range(n)))
            if k >= len(pk["slots"]): continue
            op, a, b = pk["slots"][k]
            for o, S in ((a, A), (b, B)):
                if o[0] != "t" or o[1] >= n or o[1] in S["t"] or not S["t"] or K[o[1]] is None: continue
                kp = K[o[1]]; diff = []
                for g in S["t"]:
                    kg = K[g] or (None, None, None)
                    d = [f for f, x, y in zip(("owner", "kind", "event"), kp, kg) if x is not None and y is not None and x != y]
                    diff.append(d)
                s = all(diff); sep.append(s)
                if s: fields.update({f for d in diff for f in d})
    L = [f"(i) KEY SUFFICIENCY: {np.mean(suff):.4f} of gold numeral operands with a full key are unique by key (n {len(suff)}; bar >= 0.50)",
         f"(ii) KEY SEPARABILITY: {np.mean(sep):.4f} of VANILLA's wrong numeral picks differ from every gold member in a known field "
         f"(n {len(sep)}; bar >= 0.30)",
         "read: separating fields among separable picks: " + "; ".join(f"{f} {c / max(sum(sep), 1):.3f}" for f, c in fields.most_common())]
    L.append(f"G-CEIL: {'PASS' if np.mean(sep) >= 0.30 and np.mean(suff) >= 0.50 else 'FAIL'}")
    print("\n".join(L)); open(GD.V1.C("gsm_graph_ceiling.txt"), "w").write("\n".join(L) + "\n")
