"""hmm_audition.py — THE VITERBI FRONT-END AUDITION (2026-08-24, word
given; option 3). Emissions: per-token waist features (computed under
gsb227_real — the SAME coordinates the v2 centroids were mined in;
never-mix-generations honored by using the original instrument) scored
against waist_patterns_sent centroids, max-pooled to KIND states.
Decode: 8-state Viterbi with switching penalty (uniform transitions;
the v3 empirical prior is the registered refinement). Grade: decoded
kind-multiset vs the 143 annotated rows' GOLD factor kinds (wild
annotations carry no spans — multiset F1 is the honest gradeable).
BARS (pinned pre-read): mean kind-multiset F1 >= 0.5 on the 143 = the
front-end exists; below = emissions too weak at sentence-grain
centroids, span-grain re-mine becomes the next rung.
"""
import os, sys, json, glob
os.environ.update({"DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "8",
                   "ALG_DUP": "1", "ALG_HW": "512", "ALG_WIDE": "1",
                   "ALG_TEST": ".cache/algebra_nl_bigtest.jsonl",
                   "ALG_TEST_NAME": "bigtest"})
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import sqlite3
import numpy as np
from phase1_algebra_head import T_ALG, TOKENIZER_JSON, sent_indices, load_alg
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer
from tinygrad.nn.state import safe_load

_ = load_alg("test")
tok = Tokenizer.from_file(TOKENIZER_JSON)
KINDS = ["rel", "given", "mod", "sel", "pct", "fdiv", "macro", "frac"]
SW_PEN = float(os.environ.get("HMM_SW_PEN", "0.35"))

def corpus143():
    byid = {}
    for f in sorted(glob.glob('.cache/book*_t*_batch*.jsonl')):
        for l in open(f):
            r = json.loads(l); byid[r["src_idx"]] = r
    for l in open('.cache/book12_anchor_batch1.jsonl'):
        r = json.loads(l); byid[r["src_idx"]] = r
    sk = set(json.load(open('.cache/book12_anchor_skips.json')))
    return [v for k, v in sorted(byid.items()) if k not in sk]

def main():
    rows = corpus143()
    print(f"[hmm] gold rows: {len(rows)}", flush=True)
    sd = safe_load('.cache/gsb227_real.safetensors')
    W = sd["waist_w"].to("CPU").cast('float').numpy()
    B = sd["waist_b"].to("CPU").cast('float').numpy()
    SE = sd["sent_emb"].to("CPU").cast('float').numpy()
    # centroids -> kind-labeled bank
    c = sqlite3.connect('.cache/campaign.db')
    cents = []; ckinds = []
    for cid, cnt, mean, kc in c.execute(
            "SELECT cluster_id,count,mean,kind_counts FROM waist_patterns_sent"):
        if cnt < 5: continue
        v = np.frombuffer(mean, np.float64).astype(np.float32)
        kcd = json.loads(kc) if kc else {}
        if not kcd: continue
        k = max(kcd, key=kcd.get)
        if k not in KINDS: continue
        cents.append(v); ckinds.append(KINDS.index(k))
    c.close()
    C = np.stack(cents); C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
    ckinds = np.array(ckinds)
    print(f"[hmm] centroid bank: {len(C)} clusters over "
          f"{len(set(ckinds.tolist()))} kinds", flush=True)
    f1s = []
    for s0 in range(0, len(rows), 8):
        sl = rows[s0:s0 + 8]
        ids = np.zeros((8, T_ALG), np.int32); msk = np.zeros((8, T_ALG), np.float32)
        snt = np.zeros((8, T_ALG), np.int32)
        for li, r in enumerate(sl):
            e = tok.encode(r["original"])
            if len(e.ids) > T_ALG: continue
            ids[li, :len(e.ids)] = e.ids; msk[li, :len(e.ids)] = 1.0
            snt[li] = sent_indices(r["original"], list(e.offsets), msk[li])
        sts = np.asarray(recompute_states(ids)).astype(np.float32)
        for li, r in enumerate(sl):
            ntk = int(msk[li].sum())
            if ntk == 0: f1s.append(0.0); continue
            h = sts[li, :ntk] @ W + B
            h = 0.5 * h * (1 + np.tanh(0.7978845608 * (h + 0.044715 * h ** 3)))
            h = h + SE[snt[li, :ntk]]
            h = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-9)
            sim = h @ C.T                              # (ntk, n_cent)
            em = np.full((ntk, len(KINDS)), -1.0, np.float32)
            for ki in range(len(KINDS)):
                m = ckinds == ki
                if m.any(): em[:, ki] = sim[:, m].max(1)
            # Viterbi, uniform transitions + switching penalty
            V = em[0].copy(); back = np.zeros((ntk, len(KINDS)), np.int32)
            for t in range(1, ntk):
                stay = V
                sw = V.max() - SW_PEN
                for ki in range(len(KINDS)):
                    if stay[ki] >= sw:
                        back[t, ki] = ki
                    else:
                        back[t, ki] = int(V.argmax())
                V = np.maximum(stay, sw) + em[t]
            path = [int(V.argmax())]
            for t in range(ntk - 1, 0, -1):
                path.append(int(back[t, path[-1]]))
            path = path[::-1]
            # decoded segments -> kind multiset (collapse runs; drop 'given'
            # is NOT done — givens are ops too at this grain)
            dec = []
            for t, ki in enumerate(path):
                if t == 0 or ki != path[t - 1]:
                    dec.append(KINDS[ki])
            from collections import Counter
            d = Counter(dec)
            g = Counter(f["ftype"] for f in r["factors"])
            inter = sum((d & g).values())
            f1 = 2 * inter / max(sum(d.values()) + sum(g.values()), 1)
            f1s.append(f1)
    f1s = np.array(f1s)
    print(f"[hmm] KIND-MULTISET F1: mean {f1s.mean():.3f} median "
          f"{np.median(f1s):.3f} (bar >= 0.5)  rows>=0.5: "
          f"{(f1s >= 0.5).sum()}/{len(f1s)}", flush=True)
    print("[hmm] VERDICT: " + ("THE FRONT-END EXISTS — assembly rules are "
          "the remaining game" if f1s.mean() >= 0.5 else
          "emissions too weak at sentence-grain — span-grain re-mine is "
          "the next rung"), flush=True)

if __name__ == "__main__":
    main()
