"""cue_pair.py — RUNG 2 PAIRED FIRE (2026-08-27, word given): canonical
grammar (addf=add+sub, mul, sq, opa, fr — div retired, the schism
honored) on the BACKFILLED diet.
  ARM T: token-span tagger (rung 1's architecture, diet+grammar
         repaired — the clean diet-effect twin), span-count read.
  ARM W: THE OP-CARRIER WINDING HEAD — per-token per-class amplitude
         a_ct = MLP(state); count features per class [sum(a), cos(th),
         sin(th)] with th = step_c * sum(a) (learnable step: phase
         winding, end-to-end to ROW COUNTS — no span supervision at
         all; a different supervision grain; the rotation form honest).
BARS (pinned, in-script): canonical-grain constant baselines on the
143 golds (exact + F1); PASS per arm = exact > constant AND F1 >
constant; the pairing question = W vs T on identical states.
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

CLS = ["none", "addf", "mul", "sq", "opa", "fr"]
NC = 5; NB = 8
STEPS_T = int(os.environ.get("CP_STEPS_T", "3000"))
STEPS_W = int(os.environ.get("CP_STEPS_W", "3000"))
NONE_W = 0.1
_ = load_alg("test")
tok = Tokenizer.from_file(TOKENIZER_JSON)
rng = np.random.default_rng(0)

def gold143():
    byid = {}
    for f in sorted(glob.glob('.cache/book*_t*_batch*.jsonl')):
        for l in open(f): r = json.loads(l); byid[r["src_idx"]] = r
    for l in open('.cache/book12_anchor_batch1.jsonl'):
        r = json.loads(l); byid[r["src_idx"]] = r
    sk = set(json.load(open('.cache/book12_anchor_skips.json')))
    rows = [v for k, v in sorted(byid.items()) if k not in sk]
    def canon(r):
        c = Counter()
        for f in r["factors"]:
            if f["ftype"] == "rel":
                if f.get("op") == "mul" and len(set(f.get("args", []))) == 1:
                    c["sq"] += 1
                elif f.get("op") in ("add", "sub"):
                    c["addf"] += 1
                elif f.get("op") == "mul":
                    c["mul"] += 1
                elif f.get("op") == "div":
                    c["fr"] += 1          # schism mapping: division = fr
            elif f["ftype"] == "macro":
                c["opa" if f.get("name") == "OP_APPLY" else "fr"] += 1
            elif f["ftype"] == "frac": c["fr"] += 1
        return Counter({k: min(v, NB - 1) for k, v in c.items() if v > 0})
    return rows, [canon(r) for r in rows]

def f1(a, b):
    i = sum((a & b).values())
    return 2 * i / max(sum(a.values()) + sum(b.values()), 1)

def grade(preds, golds, tag, cex, cf1):
    exo = 0; f1s = []; distinct = set()
    for d, g in zip(preds, golds):
        distinct.add(tuple(sorted(d.items())))
        if d == g: exo += 1
        f1s.append(f1(d, g))
    print(f"[cp {tag}] CANON exact {exo}/143 (constant {cex})  "
          f"F1 {np.mean(f1s):.3f} (constant {cf1:.3f})  distinct {len(distinct)}",
          flush=True)
    return exo, np.mean(f1s)

def main():
    Y = np.load('.cache/cue_y2_form8.npy').astype(np.int32)
    # merge sub(2)->addf(1); shift mul4? cue_y2 CLS was [none,add,sub,mul,sq,opa,fr]
    Y2 = Y.copy()
    Y2[Y == 2] = 1
    for old, new in ((3, 2), (4, 3), (5, 4), (6, 5)):
        Y2[Y == old] = new
    ok = np.load('.cache/cue_rows2_form8.npy')
    z = np.load('.cache/phase1_alg_states_form8.npz')
    tkm = z['tokmask'].astype(np.float32)
    G = z['g_opc']                      # [add,sub,mul,div,sq,opa,fr,...]
    CC = np.stack([np.minimum(G[:, 0] + G[:, 1], NB - 1), G[:, 2],
                   G[:, 4], G[:, 5], np.minimum(G[:, 6] + G[:, 3], NB - 1)],
                  axis=1).astype(np.int32)   # canonical row counts
    st = np.load('.cache/phase1_alg_states_form8_states.npy', mmap_mode='r')
    rows_ok = np.where(ok)[0]
    all_rows = np.arange(len(tkm))
    rows, golds = gold143()
    keys = Counter(tuple(sorted(g.items())) for g in golds)
    cex = keys.most_common(1)[0][1]
    cands = [Counter(dict(k)) for k in keys]
    cf1 = max(np.mean([f1(c, g) for g in golds]) for c in cands)
    print(f"[cp] canonical constant: exact {cex}/143, F1 {cf1:.3f}", flush=True)

    # ---------- ARM T ----------
    W1 = Tensor(rng.standard_normal((2048, 256)).astype(np.float32) * 0.02)
    b1 = Tensor(np.zeros(256, np.float32))
    W2 = Tensor(rng.standard_normal((256, 6)).astype(np.float32) * 0.02)
    b2 = Tensor(np.zeros(6, np.float32))
    for t_ in (W1, b1, W2, b2): t_.requires_grad = True
    opt = AdamW([W1, b1, W2, b2], lr=3e-4, weight_decay=0.0)
    Tensor.training = True
    for s in range(STEPS_T):
        idx = np.sort(rng.choice(rows_ok, 8, replace=False))
        X = Tensor(np.asarray(st[idx]).astype(np.float32))
        y = Y2[idx]; m = tkm[idx]
        w = m * np.where(y > 0, 1.0, NONE_W).astype(np.float32)
        lg = ((X @ W1 + b1).relu() @ W2 + b2).log_softmax(-1)
        nll = -lg.gather(-1, Tensor(y[..., None])).squeeze(-1)
        loss = (nll * Tensor(w)).sum() / (float(w.sum()) + 1e-6)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 1000 == 0: print(f"[cp T] step {s} loss {float(loss.numpy()):.4f}", flush=True)
    W1n, b1n, W2n, b2n = W1.numpy(), b1.numpy(), W2.numpy(), b2.numpy()

    # ---------- ARM W ----------
    V1 = Tensor(rng.standard_normal((2048, 256)).astype(np.float32) * 0.02)
    c1 = Tensor(np.zeros(256, np.float32))
    V2 = Tensor(rng.standard_normal((256, NC)).astype(np.float32) * 0.02)
    c2 = Tensor(np.full(NC, -3.0, np.float32))       # amplitudes start low
    stp = Tensor(np.full(NC, 1.0, np.float32))       # learnable phase step
    V3 = Tensor(rng.standard_normal((NC, 3, NB)).astype(np.float32) * 0.1)
    c3 = Tensor(np.zeros((NC, NB), np.float32))
    for t_ in (V1, c1, V2, c2, stp, V3, c3): t_.requires_grad = True
    optw = AdamW([V1, c1, V2, c2, stp, V3, c3], lr=3e-4, weight_decay=0.0)
    for s in range(STEPS_W):
        idx = np.sort(rng.choice(all_rows, 8, replace=False))
        X = Tensor(np.asarray(st[idx]).astype(np.float32))
        m = Tensor(tkm[idx][:, :, None])
        a = (((X @ V1 + c1).relu() @ V2 + c2).sigmoid()) * m   # (B,T,NC)
        S = a.sum(1)                                            # (B,NC)
        th = S * stp.reshape(1, NC)
        F = Tensor.stack(S, th.cos(), th.sin(), dim=2)          # (B,NC,3)
        Zl = (F.reshape(8, NC, 1, 3) * V3.reshape(1, NC, NB, 3).transpose(2, 3)
              ).sum(-1) if False else None
        Z = (F.unsqueeze(2) * V3.transpose(1, 2).reshape(1, NC, NB, 3)).sum(-1) + c3
        lsm = Z.log_softmax(-1)
        yt = Tensor(CC[idx][..., None].astype(np.int32))
        loss = -lsm.gather(-1, yt).squeeze(-1).mean()
        optw.zero_grad(); loss.backward(); optw.step()
        if s % 1000 == 0: print(f"[cp W] step {s} loss {float(loss.numpy()):.4f}", flush=True)
    V1n, c1n, V2n, c2n = V1.numpy(), c1.numpy(), V2.numpy(), c2.numpy()
    stpn, V3n, c3n = stp.numpy(), V3.numpy(), c3.numpy()

    # ---------- read: the 143 golds ----------
    predsT = []; predsW = []
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
        aw = 1 / (1 + np.exp(-(np.maximum(sts @ V1n + c1n, 0) @ V2n + c2n)))
        aw = aw * msk[:, :, None]
        for li in range(len(sl)):
            c = Counter(); prev = 0
            for t in range(T_ALG):
                v = int(lab[li, t])
                if v > 0 and v != prev: c[CLS[v]] += 1
                prev = v
            predsT.append(Counter({k: min(v, NB - 1) for k, v in c.items()}))
            S = aw[li].sum(0)
            th = S * stpn
            F = np.stack([S, np.cos(th), np.sin(th)], axis=1)   # (NC,3)
            Z = np.einsum('cf,cfk->ck', F, V3n) + c3n
            pk = Z.argmax(-1)
            predsW.append(Counter({CLS[c2i + 1]: int(pk[c2i])
                                   for c2i in range(NC) if pk[c2i] > 0}))
    grade(predsT, golds, "T", cex, cf1)
    grade(predsW, golds, "W", cex, cf1)

if __name__ == "__main__":
    main()
