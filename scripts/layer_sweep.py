"""layer_sweep.py — THE LAYER-SWEEP DECODABILITY READ (2026-08-27; the
five-phase map's required follow-up; subsumes the L2 probe). Per-layer
pooled states for the 143 golds across ALL 16 taps + embed; at each
tap, ridge-probe decodability of: (a) op-multiset F1 (per-class count
regression, rounded — the offensive's grain), (b) register (gold vs
20 wild-val rows: AUC of a 1-D LDA), (c) presence-count (n_factors
regression R2). 5-fold CV on the golds. The question: does op signal
PEAK in the mixed zone (L8-L11) above our L3 tap?
BARS (pinned): a tap-depth finding = op F1 at best deep tap exceeds
L3's by >= 0.05 (CV mean) — then the deep-tap pilot (research lineage,
matched L3 twin) earns registration. Below: tap depth is not the wall.
"""
import os, sys, json, glob
os.environ.update({"DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "9",
                   "ALG_DUP": "1", "ALG_HW": "512", "ALG_WIDE": "1",
                   "ALG_TEST": ".cache/algebra_nl_bigtest.jsonl",
                   "ALG_TEST_NAME": "bigtest"})
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
from collections import Counter
from phase1_algebra_head import T_ALG, TOKENIZER_JSON, load_alg
from mycelium.llama_loader import (attach_llama_layers, load_llama_weights,
                                   LLAMA_3_2_1B_CFG)
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes

_ = load_alg("test")
tok = Tokenizer.from_file(TOKENIZER_JSON)
OPS = ["add", "sub", "mul", "div", "sq", "opa", "fr"]
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(0)

def pooled_all_layers(texts):
    class _H: pass
    host = _H()
    sd = load_llama_weights(os.path.join(_ROOT, ".cache/llama-3.2-1b-weights/model.safetensors"))
    attach_llama_layers(host, n_layers=16, sd=sd, cfg=LLAMA_3_2_1B_CFG)
    del sd
    out = []
    for s0 in range(0, len(texts), 4):
        sl = texts[s0:s0 + 4]
        ids = np.zeros((4, T_ALG), np.int32); msk = np.zeros((4, T_ALG), np.float32)
        for li, t in enumerate(sl):
            e = tok.encode(t)
            if len(e.ids) > T_ALG: continue
            ids[li, :len(e.ids)] = e.ids; msk[li, :len(e.ids)] = 1.0
        x = host.llama_embed[Tensor(ids, dtype=dtypes.int)]
        m = msk[:, :, None]
        traj = [(x.cast(dtypes.float).numpy() * m).sum(1) / np.maximum(m.sum(1), 1)]
        for layer in host.llama_layers:
            x = layer(x, host.llama_rope_cos, host.llama_rope_sin)
            traj.append((x.cast(dtypes.float).numpy() * m).sum(1)
                        / np.maximum(m.sum(1), 1))
        T2 = np.stack(traj, axis=1)
        for li in range(len(sl)):
            out.append(T2[li])
    return np.stack(out)      # (n, 17, H)

def gmeta(r):
    c = Counter()
    for f in r["factors"]:
        if f["ftype"] == "rel":
            if f.get("op") == "mul" and len(set(f.get("args", []))) == 1:
                c["sq"] += 1
            else: c[f.get("op", "add")] += 1
        elif f["ftype"] == "macro":
            c["opa" if f.get("name") == "OP_APPLY" else "fr"] += 1
        elif f["ftype"] == "frac": c["fr"] += 1
    return np.array([min(c.get(k, 0), 7) for k in OPS], np.float32)

def main():
    byid = {}
    for f in sorted(glob.glob('.cache/book*_t*_batch*.jsonl')):
        for l in open(f): r = json.loads(l); byid[r["src_idx"]] = r
    for l in open('.cache/book12_anchor_batch1.jsonl'):
        r = json.loads(l); byid[r["src_idx"]] = r
    sk = set(json.load(open('.cache/book12_anchor_skips.json')))
    rows = [v for k, v in sorted(byid.items()) if k not in sk]
    Y = np.stack([gmeta(r) for r in rows])          # (143, 7) counts
    NF = np.array([len(r["factors"]) for r in rows], np.float32)
    wv = [json.loads(l) for l in open('.cache/g55_wildval.jsonl')]
    texts = [r["original"] for r in rows] + [r["original"] for r in wv]
    P = pooled_all_layers(texts)
    Pg, Pw = P[:len(rows)], P[len(rows):]
    n, LT, H = Pg.shape
    print(f"[ls] states banked: {P.shape}", flush=True)
    folds = np.array_split(rng.permutation(n), 5)
    lam = 10.0
    for L in range(LT):
        X = Pg[:, L]; Xw = Pw[:, L]
        mu, sd2 = X.mean(0), X.std(0) + 1e-6
        Xn = (X - mu) / sd2
        f1s = []; r2s = []
        for k in range(5):
            te = folds[k]; tr = np.concatenate([folds[j] for j in range(5) if j != k])
            A = Xn[tr]; G = A.T @ A + lam * np.eye(H)
            Wr = np.linalg.solve(G, A.T @ Y[tr])
            pred = np.clip(np.round(Xn[te] @ Wr), 0, 7)
            for i, t in enumerate(te):
                d = Counter({OPS[c]: int(pred[i, c]) for c in range(7) if pred[i, c] > 0})
                g = Counter({OPS[c]: int(Y[t, c]) for c in range(7) if Y[t, c] > 0})
                inter = sum((d & g).values())
                f1s.append(2 * inter / max(sum(d.values()) + sum(g.values()), 1))
            wn = np.linalg.solve(G, A.T @ NF[tr])
            pr = Xn[te] @ wn
            ss = 1 - ((pr - NF[te]) ** 2).sum() / (((NF[te] - NF[tr].mean()) ** 2).sum() + 1e-9)
            r2s.append(ss)
        # register separation: gold vs wild-val, 1-D LDA AUC
        Xwn = (Xw - mu) / sd2
        d = Xn.mean(0) - Xwn.mean(0)
        sg = Xn @ d; sw = Xwn @ d
        auc = (sg[:, None] > sw[None, :]).mean()
        print(f"[ls] L{L:02d}: opF1 {np.mean(f1s):.3f}  nfacR2 {np.mean(r2s):+.2f}"
              f"  regAUC {auc:.2f}", flush=True)
    print("[ls] (constant op-only F1 reference: 0.457; our tap = L04 row "
          "[embed=L00]; mixed zone = L09-L12 rows)", flush=True)

if __name__ == "__main__":
    main()
