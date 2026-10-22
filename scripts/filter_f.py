"""filter_f.py -- THE PER-LOOKUP FILTER ACCURACY f (2026-10-21, zero GPU; ledger same day). Per wild given lookup: does
the body's pick share the gold numeral's OWNER / TYPE / EVENT (the candidate census's three filters)? Over all lookups,
and over wrong picks against chance. spaCy for measurement only (governor_census.py's machinery, unchanged).
Usage: FF_TAGS=CLK3_241,K3_242 .venv/bin/python3 scripts/filter_f.py  -> .cache/filter_f.txt (overwrite guard)."""
import collections, json, os, pickle, sys
import numpy as np
import spacy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import governor_census as G  # noqa: E402
from candidate_census import head_noun_key  # noqa: E402

OUT = ".cache/filter_f.txt"
FEATS = ("owner", "type", "event")


def main():
    assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
    nlp = spacy.load("en_core_web_sm")
    rows = [json.loads(l) for l in open(G.HOLDOUT)]
    parsed = []
    for row in rows:
        doc = nlp(row["text"]); sents = list(doc.sents); cands = G.numeral_candidates(doc); fv = G.finite_verbs(doc)
        feat = {}
        for _, tok, v in cands:
            ch = G.governor_chain(doc, sents, tok)
            feat[tok.i] = dict(value=v, owner=(ch["owner_key"] if ch else None), type=head_noun_key(tok),
                               event=G.event_index_of(fv, tok))
        given = [(f["var"], f["value"]) for f in row["factors"] if f["ftype"] == "given"]
        parsed.append((cands, feat, G.bind_given_occurrences(doc, given, cands)))
    L = []
    for tag in os.environ.get("FF_TAGS", "CLK3_241,K3_242").split(","):
        D = pickle.load(open(f".cache/dump_wild_{tag}.pkl", "rb"))
        z = np.load(f".cache/ps_legal_wild_{tag}.npz")
        ok = {(int(r), int(s)): bool(o) for r, s, o in zip(z["rows"], z["slots"], z["ok"])}
        C = collections.Counter(); hit = collections.defaultdict(list); hitw = collections.defaultdict(list)
        chance = collections.defaultdict(list); joint, jointw = [], []
        for t in D:
            i, j, gft, gres, gdig, pdig = t[0], t[1], t[2], t[5], t[6], t[11]
            if gft != 1:
                continue
            cands, feat, bound = parsed[i]
            gtok = bound.get(gres)
            if gtok is None:
                C["gold has no text occurrence"] += 1; continue
            g = feat[gtok.i]; pv = G.digits_to_int(pdig); right = ok.get((i, j))
            if right is None:
                C["no verdict"] += 1; continue
            if pv == g["value"]:
                ptok_f = g; C["pick = gold value"] += 1
            else:
                pt = G.nearest_occurrence(cands, pv, None, gtok.idx)
                if pt is None:
                    C["pick has no text occurrence"] += 1; continue
                ptok_f = feat[pt.i]; C["pick elsewhere in text"] += 1
            same = {}
            for k in FEATS:
                if g[k] is None or ptok_f[k] is None:
                    continue
                same[k] = float(g[k] == ptok_f[k]); hit[k].append(same[k])
                if pv != g["value"]:
                    hitw[k].append(same[k])
                    others = [f[k] for ti, f in feat.items() if ti != gtok.i and f[k] is not None]
                    if others:
                        chance[k].append(float(np.mean([o == g[k] for o in others])))
            if len(same) == 3:
                joint.append(float(all(same.values())))
                if pv != g["value"]:
                    jointw.append(float(all(same.values())))
        L.append(f"== body {tag} ==  " + ", ".join(f"{k} {v}" for k, v in C.items()))
        for k in FEATS:
            L.append(f"  {k:6s} f (all lookups) {np.mean(hit[k]):.3f} (n {len(hit[k])}) | wrong picks {np.mean(hitw[k]):.3f} "
                     f"(n {len(hitw[k])}) vs chance {np.mean(chance[k]):.3f} -> {np.mean(hitw[k]) - np.mean(chance[k]):+.3f}")
        L.append(f"  joint  f (all three, all lookups) {np.mean(joint):.3f} (n {len(joint)}) | wrong picks {np.mean(jointw):.3f} (n {len(jointw)})")
        L.append(f"  F^H with the joint f: H1 {np.mean(joint):.3f} H2 {np.mean(joint) ** 2:.3f} H3 {np.mean(joint) ** 3:.3f}")
    print("\n".join(L)); open(OUT, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
