"""AG3 part 1, THE VIEW DATA (ledger 2026-11-04, bars pinned): zero-GPU, CPU. A second VIEW of every ORIG and TEST bench item.

The view = the problem text with its NON-QUESTION sentences reordered (the final sentence = the question, pinned last), re-tokenized
(gsm_bench_data.tokenize, the item's own PERSON names), and the gold program RE-DERIVED on the view's tokens from the SAME harvest trace
(gsm_bench_data.program; the valid pointer sets are recomputed by value, the answer key is unchanged). Splitter: whitespace after [.?!]
followed by an uppercase / digit / quote / $ start, abbreviations (Mr. Mrs. Dr. St. a.m. ...) never split. Permutation: seeded per
(source, gid) -- ORIG and TRAIN share a gid, so they share a view -- a random NON-identity order; <= 1 non-question sentence = identity.
FALLBACK = the identity view (the item copied, text unchanged) when the re-derivation fails, the final differs, or the word multiset
differs; every fallback is COUNTED. Item ORDER and COUNT per split equal .cache/gsmb_data.json's (the downstream join is positional).
Splits written: ORIG, TEST (the AG3 reads; TRAIN / CF are not needed: the dump reuses view 0's rho0 / lam).
Items gain: text (GSM8K text; .cache only, never committed), ident (bool), why (fallback reason or 'permuted' / 'single').
-> .cache/gsmb_data_v1.json, .cache/gsmb_trunk_index_v1.json (the word-row start per item in .cache/gsmb_mbr8_states_v1.npy),
   .cache/gsmb_view_census_v1.txt.   env: AG3_OUT_DIR (default .cache; the smoke redirects), AG3_N (first n items per split, smoke).
"""
import collections, glob, json, os, re, sys, zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
import gsm_bench_data as GD
import gsm_trunk_states as GTS

OUTD = os.environ.get("AG3_OUT_DIR", ".cache"); NLIM = int(os.environ.get("AG3_N", "0")) or None
ABBR = {"mr", "mrs", "ms", "dr", "st", "jr", "sr", "vs", "mt", "no", "inc", "co", "etc", "approx"}
SPLIT = re.compile(r"(?<=[.?!])\s+(?=[A-Z0-9\"'$(])")


def sentences(q):
    """raw text -> sentence strings (no abbreviation / decimal splits)"""
    parts, cur = [], ""
    for seg in SPLIT.split(q.strip()):
        cur = f"{cur} {seg}" if cur else seg
        last = re.findall(r"([A-Za-z]+)\.$", cur)
        if last and (last[-1].lower() in ABBR or re.search(r"\b[ap]\.m\.$|\b[A-Z]\.$", cur)): continue
        parts.append(cur); cur = ""
    if cur: parts.append(cur)
    return parts


def view_text(q, key):
    """-> (text, why): the non-question sentences in a seeded non-identity order, the last sentence pinned"""
    s = sentences(q)
    if len(s) < 3: return q, "single"                                   # <= 1 non-question sentence: nothing to reorder
    body, last = s[:-1], s[-1]; rng = np.random.default_rng(zlib.crc32(key.encode()))
    for _ in range(50):
        p = rng.permutation(len(body))
        if (p != np.arange(len(body))).any(): return " ".join([body[i] for i in p] + [last]), "permuted"
    return q, "single"


def main():
    import pyarrow.parquet as pq
    rd = lambda sp: pq.read_table(glob.glob(GD.V1.C(f"gsm8k/datasets--openai--gsm8k/snapshots/*/main/{sp}-*.parquet"))[0]).to_pydict()
    D0 = json.load(open(GD.OUT)); src = {"ORIG": ("train", rd("train")), "TEST": ("test", rd("test"))}
    Dv, census, lines = {}, {}, []
    for sp in ("ORIG", "TEST"):
        nm, pq_ = src[sp]; census[sp] = collections.Counter(); Dv[sp] = []
        for it in D0[sp][:NLIM]:
            q, a = pq_["question"][it["gid"]], pq_["answer"][it["gid"]]
            assert GD.tokenize(q, set(it["names"]))[0] == it["toks"], f"{sp} gid {it['gid']}: view-0 word sequence mismatch"
            vt, why = view_text(q, f"{nm}|{it['gid']}")
            new = None
            if why == "permuted":
                r, new = GD.program(vt, a, set(it["names"]))
                if new is None: why = f"fallback_{r}"
                elif not GD.V1.near(new["final"], it["final"]): new, why = None, "fallback_final"
                elif sorted(new["toks"]) != sorted(it["toks"]): new, why = None, "fallback_words"
                elif len(new["prog"]) != len(it["prog"]): new, why = None, "fallback_nsteps"   # a view whose step count shifts is a different task
            if new is None: new, vt = dict(it), q; new["ident"] = True            # the identity view (text unchanged)
            else: new = dict(new, gid=it["gid"], names=it["names"], ident=False)
            new["text"], new["why"] = vt, why; census[sp][why] += 1; Dv[sp].append(new)
        c = census[sp]; n = len(Dv[sp])
        lines.append(f"{sp:5s} n {n} | permuted {c['permuted']} | identity: single/<=1 non-question sentence {c['single']}, "
                     f"re-derivation failures " + ", ".join(f"{k} {v}" for k, v in c.items() if k.startswith("fallback")) + f" | identity total {n - c['permuted']}")
    # the word-row index: laid out exactly as scripts/distill_reader.py --gsm lays out the states (one block per DISTINCT text, in split order)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(".cache/llama-3.2-1b-weights/tokenizer.json")
    Q = {sp: [it["text"] for it in Dv[sp]] for sp in Dv}; uniq = list(dict.fromkeys(q for sp in Q.values() for q in sp)); enc = [tok.encode(q) for q in uniq]
    assert max(len(e.ids) for e in enc) <= GTS.T_ALG
    al = [GTS.align(q, e.offsets, len(e.ids)) for q, e in zip(uniq, enc)]; start = np.cumsum([0] + [len(x) for x in al]); pos = {q: int(start[i]) for i, q in enumerate(uniq)}
    lines.append(f"{len(uniq)} distinct texts, {int(start[-1])} words")
    os.makedirs(OUTD, exist_ok=True); sfx = "" if OUTD == ".cache" else "_smoke"
    json.dump(Dv, open(f"{OUTD}/gsmb_data_v1.json", "w")); json.dump({sp: [pos[q] for q in Q[sp]] for sp in Q}, open(f"{OUTD}/gsmb_trunk_index_v1.json", "w"))
    open(f"{OUTD}/gsmb_view_census_v1.txt", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    assert os.environ.get("AG3_FORCE") == "1" or not os.path.exists(f"{OUTD}/gsmb_data_v1.json"), "refusing to overwrite the view data"
    main()
