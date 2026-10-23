"""mention_type_slice.py -- THE MENTION-TYPE SLICE (2026-10-23, zero GPU; ledger same date).
Does the deployed verdict on GIVEN slots (ps_legal ok) lag when the gold numeral's owner mention is a
PRONOUN or a DESCRIPTION rather than a NAME?  The write-side join (T2h/T2i) seen on the main line.
Reuses governor_census's binder and find_owner.  Wild holdout = measure-only.
Outputs .cache/mention_type_slice.{txt,json} + .cache/mention_type_handcheck40.txt (the validity sample).
"""
import collections, json, os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import spacy  # noqa: E402
import governor_census as GC  # noqa: E402

BODIES = ["BPK3_241", "BPK3_242", "CBK3_241", "CLK3_241", "CRK3_241", "SRK3_241"]
OUT = ".cache/mention_type_slice.txt"
PERS = {"he", "she", "they", "it", "him", "her", "them", "his", "hers", "its", "their", "theirs",
        "i", "me", "my", "we", "us", "our", "you", "your"}
CLASSES = ["NAME", "PRONOUN", "DESC", "OTHER", "NO-CHAIN"]


def mention_type(doc, tok):
    head_noun = verb = None
    for anc in tok.ancestors:
        if head_noun is None and anc.pos_ in ("NOUN", "PROPN"):
            head_noun = anc
        if anc.pos_ in ("VERB", "AUX"):
            verb = anc; break
    if head_noun is None and tok.head is not tok:
        head_noun = tok.head
    o, _, _ = GC.find_owner(verb, head_noun)
    if o is None:
        return "NO-CHAIN", None
    if any(w.pos_ == "PROPN" for w in o.subtree):
        return "NAME", o
    if o.text.lower() in PERS:
        return "PRONOUN", o
    if o.pos_ in ("NOUN",):
        return "DESC", o
    return "OTHER", o


def main():
    assert not os.path.exists(OUT), "refusing to overwrite a banked read"
    nlp = spacy.load("en_core_web_sm")
    rows = [json.loads(l) for l in open(GC.HOLDOUT)]
    # per (row, var): mention type, sentence index, owner text
    info = {}
    for i, row in enumerate(rows):
        doc = nlp(row["text"]); sents = list(doc.sents)
        given = [(f["var"], f["value"]) for f in row["factors"] if f["ftype"] == "given"]
        bound = GC.bind_given_occurrences(doc, given, GC.numeral_candidates(doc))
        for var, tok in bound.items():
            if tok is None:
                info[(i, var)] = ("UNBOUND", -1, None, None); continue
            mt, o = mention_type(doc, tok)
            info[(i, var)] = (mt, GC.sent_index_of(doc, sents, tok), o.text if o is not None else None,
                              (row["text"], tok.idx, len(tok.text)))
    R = {}
    for b in BODIES:
        D = pickle.load(open(f".cache/dump_wild_{b}.pkl", "rb"))
        z = np.load(f".cache/ps_legal_wild_{b}.npz")
        ok = {(int(r), int(s)): bool(o) for r, s, o in zip(z["rows"], z["slots"], z["ok"])}
        recs = []
        for t in D:
            i, j, gft, gres = t[0], t[1], t[2], t[5]
            if gft != 1 or (i, j) not in ok:
                continue
            mt, si, _, _ = info.get((i, gres), ("UNBOUND", -1, None, None))
            recs.append((i, mt, min(si, 3), ok[(i, j)]))
        R[b] = recs
    lines = []

    def say(s):
        print(s, flush=True); lines.append(s)

    def acc(rs):
        return float(np.mean([r[3] for r in rs])) if rs else float("nan")

    def matched(rs, cls):
        """NAME accuracy re-weighted to cls's sentence-index distribution."""
        tgt = collections.Counter(r[2] for r in rs if r[1] == cls); n = sum(tgt.values()); tot = 0.0
        for k, c in tgt.items():
            nm = [r for r in rs if r[1] == "NAME" and r[2] == k]
            if not nm:
                return float("nan")
            tot += c / n * acc(nm)
        return tot

    res = {}
    say("THE MENTION-TYPE SLICE -- given-slot ok (ps_legal) by the gold numeral's owner mention type, wild holdout")
    for b in BODIES:
        rs = R[b]; cnt = collections.Counter(r[1] for r in rs)
        row = {c: dict(n=cnt[c], acc=acc([r for r in rs if r[1] == c])) for c in CLASSES + ["UNBOUND"]}
        for c in ("PRONOUN", "DESC"):
            row[f"gap_raw_{c}"] = row["NAME"]["acc"] - row[c]["acc"]
            row[f"gap_pos_{c}"] = matched(rs, c) - row[c]["acc"]
        res[b] = row
        say(f"{b}: " + " | ".join(f"{c} {row[c]['acc']:.3f} (n {row[c]['n']})" for c in CLASSES)
            + f" || NAME-PRON raw {row['gap_raw_PRONOUN']:+.3f} pos {row['gap_pos_PRONOUN']:+.3f}"
            + f" | NAME-DESC raw {row['gap_raw_DESC']:+.3f} pos {row['gap_pos_DESC']:+.3f}")
    pooled = [r for b in BODIES for r in R[b]]
    gp = {c: matched(pooled, c) - acc([r for r in pooled if r[1] == c]) for c in ("PRONOUN", "DESC")}
    gr = {c: acc([r for r in pooled if r[1] == "NAME"]) - acc([r for r in pooled if r[1] == c]) for c in ("PRONOUN", "DESC")}
    # row bootstrap SE of the position-matched gaps on the first body
    rng = np.random.default_rng(0); rs0 = R[BODIES[0]]; byrow = collections.defaultdict(list)
    for r in rs0:
        byrow[r[0]].append(r)
    keys = list(byrow); bs = {c: [] for c in gp}
    for _ in range(1000):
        samp = [r for k in rng.choice(keys, len(keys)) for r in byrow[k]]
        for c in gp:
            bs[c].append(matched(samp, c) - acc([r for r in samp if r[1] == c]))
    se = {c: float(np.nanstd(bs[c])) for c in gp}
    signs = {c: sum(res[b][f"gap_pos_{c}"] > 0 for b in BODIES) for c in gp}
    say(f"POOLED (6 bodies): NAME-PRONOUN pos-matched {gp['PRONOUN']:+.3f} (raw {gr['PRONOUN']:+.3f}; SE {se['PRONOUN']:.3f} on {BODIES[0]}; sign + in {signs['PRONOUN']}/6)")
    say(f"                   NAME-DESC    pos-matched {gp['DESC']:+.3f} (raw {gr['DESC']:+.3f}; SE {se['DESC']:.3f}; sign + in {signs['DESC']}/6)")
    w1 = all(gp[c] >= 0.15 and signs[c] >= 5 for c in gp); w0 = all(abs(gp[c]) < 0.05 for c in gp)
    say("VERDICT (pinned 2026-10-23): " + ("W1 -- THE WRITE-SIDE HOLE IS VISIBLE DOWNSTREAM (both gaps >= 0.15)" if w1 else
        "W0 -- NULL: the given pick does not see the hole (both gaps < 0.05)" if w0 else
        "PARTIAL -- " + ", ".join(f"{c} {'>= 0.15' if gp[c] >= 0.15 else '< 0.05' if abs(gp[c]) < 0.05 else 'between'}" for c in gp))
        + "  [VALID only if the 40-numeral hand-check agrees >= 0.80]")
    # the validity sample: 10 per class, stratified, seed 1023
    rng = np.random.default_rng(1023); hc = []
    for c in ["NAME", "PRONOUN", "DESC", "NO-CHAIN"]:
        ks = [k for k, v in info.items() if v[0] == c]
        for idx in rng.choice(len(ks), min(10, len(ks)), replace=False):
            k = ks[idx]; text, ci, ln = info[k][3]; hc.append((k, c, info[k][2], text[:ci] + "[[" + text[ci:ci + ln] + "]]" + text[ci + ln:]))
    rng.shuffle(hc)
    with open(".cache/mention_type_handcheck40.txt", "w") as f:
        for n, (k, c, o, t) in enumerate(hc):
            f.write(f"#{n}\n{t}\nTYPE: \n\n")
    json.dump(dict(bodies=res, pooled_pos=gp, pooled_raw=gr, se=se, signs=signs,
                   handcheck_key=[dict(n=n, row=k[0], var=k[1], machine=c, owner=o) for n, (k, c, o, t) in enumerate(hc)]),
              open(OUT.replace(".txt", ".json"), "w"), indent=1)
    open(OUT, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
