"""cue_tagger.py — RUNG 1: THE OP-CUE TAGGER on the frozen trunk
(2026-08-26, word given). Per-token 8-way head (2048 -> 256 -> 8) on
banked frozen states; gold = fspan x op (cue_prep). Read: tag the 143
golds' tokens, connected-components per class -> cue counts -> op
multiset. Detection-and-counting = the extensive token-grain form the
linear probe could not test (nonlinear located detection; each detected
cue clicks the counter — the winding intuition's practical cousin).
BARS (pinned, the reckoning's, computed in-script for the op-only
grain): OP-ONLY exact > best-constant op-only exact (17/143) AND
op-only F1 > best-constant op-only F1. KILL below both.
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
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.optim import AdamW

CLS = ["none", "add", "sub", "mul", "div", "sq", "opa", "fr"]
OPS = tuple(CLS[1:])
NONE_W = float(os.environ.get("CUE_NONE_W", "0.1"))
STEPS = int(os.environ.get("CUE_STEPS", "3000"))

_ = load_alg("test")
tok = Tokenizer.from_file(TOKENIZER_JSON)

def main():
    Y = np.load('.cache/cue_y_form8.npy')
    ok = np.load('.cache/cue_rows_form8.npy')
    z = np.load('.cache/phase1_alg_states_form8.npz')
    tkm = z['tokmask'].astype(np.float32)
    st = np.load('.cache/phase1_alg_states_form8_states.npy', mmap_mode='r')
    rows_ok = np.where(ok)[0]
    print(f"[cue] training rows {len(rows_ok)}", flush=True)
    rng = np.random.default_rng(0)
    W1 = Tensor(rng.standard_normal((2048, 256)).astype(np.float32) * 0.02)
    b1 = Tensor(np.zeros(256, np.float32))
    W2 = Tensor(rng.standard_normal((256, 8)).astype(np.float32) * 0.02)
    b2 = Tensor(np.zeros(8, np.float32))
    for t_ in (W1, b1, W2, b2): t_.requires_grad = True
    opt = AdamW([W1, b1, W2, b2], lr=3e-4, weight_decay=0.0)
    B = 8
    Tensor.training = True
    for s in range(STEPS):
        idx = rng.choice(rows_ok, B, replace=False)
        X = Tensor(np.asarray(st[np.sort(idx)]).astype(np.float32))
        idx = np.sort(idx)
        y = Y[idx].astype(np.int32)
        m = tkm[idx]
        w = m * np.where(y > 0, 1.0, NONE_W).astype(np.float32)
        h = (X @ W1 + b1).relu()
        lg = h @ W2 + b2
        lsm = lg.log_softmax(-1)
        yt = Tensor(y[..., None])
        nll = -lsm.gather(-1, yt).squeeze(-1)
        loss = (nll * Tensor(w)).sum() / (float(w.sum()) + 1e-6)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 500 == 0:
            print(f"[cue] step {s} loss {float(loss.numpy()):.4f}", flush=True)
    np.savez('.cache/cue_head.npz', W1=W1.numpy(), b1=b1.numpy(),
             W2=W2.numpy(), b2=b2.numpy())

    # ---- read: the 143 golds ----
    byid = {}
    for f in sorted(glob.glob('.cache/book*_t*_batch*.jsonl')):
        for l in open(f): r = json.loads(l); byid[r["src_idx"]] = r
    for l in open('.cache/book12_anchor_batch1.jsonl'):
        r = json.loads(l); byid[r["src_idx"]] = r
    sk = set(json.load(open('.cache/book12_anchor_skips.json')))
    rows = [v for k, v in sorted(byid.items()) if k not in sk]
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
        return Counter({k: min(v, 7) for k, v in c.items() if k in OPS and v > 0})
    golds = [gmeta(r) for r in rows]
    # in-script constant baselines at the op-only grain
    keys = Counter(tuple(sorted(g.items())) for g in golds)
    cexact = keys.most_common(1)[0][1]
    def f1(a, b):
        i = sum((a & b).values())
        return 2 * i / max(sum(a.values()) + sum(b.values()), 1)
    cands = [Counter(dict(k)) for k in keys]
    cf1 = max(np.mean([f1(c, g) for g in golds]) for c in cands)
    preds = []
    W1n, b1n, W2n, b2n = W1.numpy(), b1.numpy(), W2.numpy(), b2.numpy()
    for s0 in range(0, len(rows), 8):
        sl = rows[s0:s0 + 8]
        ids = np.zeros((8, T_ALG), np.int32); msk = np.zeros((8, T_ALG), np.float32)
        for li, r in enumerate(sl):
            e = tok.encode(r["original"])
            if len(e.ids) > T_ALG: continue
            ids[li, :len(e.ids)] = e.ids; msk[li, :len(e.ids)] = 1.0
        sts = np.asarray(recompute_states(ids)).astype(np.float32)
        h = np.maximum(sts @ W1n + b1n, 0)
        lab = (h @ W2n + b2n).argmax(-1) * (msk > 0)
        for li in range(len(sl)):
            c = Counter(); prev = 0
            for t in range(T_ALG):
                v = int(lab[li, t])
                if v > 0 and v != prev:
                    c[CLS[v]] += 1
                prev = v
            preds.append(Counter({k: min(v, 7) for k, v in c.items()}))
    exo = 0; f1s = []; distinct = set()
    for d, g in zip(preds, golds):
        distinct.add(tuple(sorted(d.items())))
        if d == g: exo += 1
        f1s.append(f1(d, g))
    print(f"[cue] GOLD143 OP-ONLY exact {exo}/143 (constant {cexact})  "
          f"F1 {np.mean(f1s):.3f} (constant {cf1:.3f})  "
          f"distinct {len(distinct)}", flush=True)
    print("[cue] VERDICT: " + ("ABOVE THE ROCK — first above-constant op "
          "skill in the campaign" if exo > cexact and np.mean(f1s) > cf1
          else "at or below the constant — rung 2 (lean LoRA + wild-surface "
          "mints) is the registered next form"), flush=True)

if __name__ == "__main__":
    main()
