"""gsm_slot_ceiling.py -- S-CEIL (ledger 2026-10-30 11:14, Bryce's movie-database brainstorm): THE SLOT AS A NODE. A gold step's result
INHERITS a key from its operands (owners / events = the union over each operand's first gold member: numeral -> its G-CEIL owner /
event, slot -> that slot's inherited key, constant -> nothing); the QUESTION sentence supplies the first hop. Zero-training, CPU, TEST.
S1 SLOT-KEY SUFFICIENCY: gold slot operands with a non-empty key held by no non-valid candidate (earlier slot or question numeral).
S2 QUESTION ANCHOR: the question sentence's owner is among the final gold step's owners (placebo: a random owner of the problem).
Reads: VANILLA's slot-confusion errors -- picked key != gold key, beside a random-candidate placebo. Writes .cache/gsm_slot_ceiling.txt."""
import collections, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
import gsm_bench_data as GD
import gsm_graph_ceiling as G
import gsm_trunk_states as GT

E0 = (frozenset(), frozenset())


def first(S):
    return ("t", S["t"][0]) if S["t"] else (("s", S["s"][0]) if S["s"] else ("c", S["c"][0]))


def numkey(K, j):
    k = K[j] if j < len(K) else None
    if k is None: return E0
    return frozenset({k[0]} - {None}), frozenset({k[2]} - {None})


def slot_keys(prog, K):
    out = []
    for op, A, B in prog:
        o, e = set(), set()
        for S in (A, B):
            kd, i = first(S)
            ko = numkey(K, i) if kd == "t" else (out[i] if kd == "s" else E0)
            o |= ko[0]; e |= ko[1]
        out.append((frozenset(o), frozenset(e)))
    return out


def q_owner(q, nlp, owners):
    doc = nlp(q); persons = [(e.start_char, e.text.split()[0].lower()) for e in doc.ents if e.label_ == "PERSON"]
    sents = [s for s in doc.sents if "?" in s.text]
    if not sents: return None
    for t in sents[-1]:
        if t.dep_ not in ("nsubj", "nsubjpass", "poss"): continue
        w = t.text.lower()
        o = "narrator" if w in G.NARR else (([p for c, p in persons if c < t.idx] or [None])[-1] if w in G.PRON else t.lemma_.lower())
        if o in owners: return o
    return None


if __name__ == "__main__":
    import spacy
    nlp = spacy.load("en_core_web_sm"); rng = np.random.default_rng(0)
    cls = json.load(open(GD.V1.C("gsm_type_forest_k3.json")))["parent"]
    P = json.load(open(GD.V1.C("gsmb_picks_VANILLA_s0.json"))); Dt, Q = GT.texts()
    s1, s1_all, s2, s2p, s2_n, err, errp = [], 0, [], [], 0, collections.defaultdict(list), collections.defaultdict(list)
    for it, q, pk in zip(Dt["TEST"], Q["TEST"], P["TEST"]):
        K = G.keys(q, nlp, cls); prog = it["prog"]; n = len(it["toks"]); SK = slot_keys(prog, K)
        key = lambda kd, i: numkey(K, i) if kd == "t" else (SK[i] if kd == "s" else E0)
        for k, (gop, A, B) in enumerate(prog):
            for S in (A, B):
                if not S["s"]: continue
                s1_all += 1; gk = SK[S["s"][0]]
                if gk == E0: continue
                cands = [key("s", j) for j in range(k) if j not in S["s"]] + [numkey(K, j) for j in range(n) if K[j] is not None and j not in S["t"]]
                s1.append(gk not in cands)
            if k >= len(pk["slots"]): continue
            _, a, b = pk["slots"][k]
            for o, S in ((a, A), (b, B)):
                kd, i = o
                if kd not in ("t", "s") or i in S[kd] or (kd == "t" and i >= n) or (kd == "s" and i >= k): continue
                gkd = "s" if S["s"] else ("t" if S["t"] else None)
                if gkd is None or gkd == kd == "t": continue                  # the numeral-vs-numeral row is G-CEIL's
                gk, pkk = key(*first(S)), key(kd, i)
                row = f"picked {kd} / gold {gkd}"
                if gk == E0 or pkk == E0: continue
                err[row].append(pkk != gk)
                pool = [j for j in (range(k) if kd == "s" else [j for j in range(n) if K[j] is not None]) if j not in S[kd]]
                if pool: errp[row].append(key(kd, int(rng.choice(pool))) != gk)
        owners = sorted({x[0] for x in K if x is not None and x[0] is not None})
        fo = SK[-1][0] if SK else frozenset()
        if owners and fo:
            s2_n += 1; qo = q_owner(q, nlp, set(owners))
            if qo is not None:
                s2.append(qo in fo); s2p.append(owners[int(rng.integers(len(owners)))] in fo)
    a1, a2, p2 = np.mean(s1), np.mean(s2), np.mean(s2p)
    L = [f"S1 SLOT-KEY SUFFICIENCY: {a1:.4f} of gold slot operands with a non-empty key are unique by key (n {len(s1)} of {s1_all} slot operands; bar >= 0.50)",
         f"S2 QUESTION ANCHOR: {a2:.4f} vs placebo {p2:.4f} (margin {a2 - p2:+.4f}; n {len(s2)} of {s2_n} problems with owners and a keyed final step; bar >= 0.50 and margin >= +0.15)"]
    for row in sorted(err):
        L.append(f"read: VANILLA {row}: key differs {np.mean(err[row]):.3f} (n {len(err[row])}) vs random-candidate placebo {np.mean(errp[row]):.3f}")
    ok1, ok2 = a1 >= 0.50, a2 >= 0.50 and a2 - p2 >= 0.15
    L.append(f"S1 {'PASS' if ok1 else 'FAIL'} | S2 {'PASS' if ok2 else 'FAIL'} -> S-CEIL {'PASS' if ok1 and ok2 else 'FAIL'}")
    print("\n".join(L)); open(GD.V1.C("gsm_slot_ceiling.txt"), "w").write("\n".join(L) + "\n")
