"""nl_metric.py — THE METRIC HEAD (2026-08-28, word given): supervised-
contrastive projection h_t -> z_t (2048->128) shaped on the banked span
anchors (positives per class) + background tokens (their own cluster =
the rejection region); Welford/leader centroids rebuilt IN z-SPACE;
token routing with distance abstention; hybrid with the formal-register
scanner. FLOOR: v0's 29/143. Custody: anchors from mint+book3 only.
"""
import json, re, glob, os, sys
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
import op_witness as OW

_ = load_alg("test")
tok = Tokenizer.from_file(TOKENIZER_JSON)
OPS = ("addf", "mul", "sq", "fr")
STEPS = int(os.environ.get("NM_STEPS", "2000"))
THR = float(os.environ.get("NM_THR", "0.35"))

def cue_sites():
    out = []
    for l in open('.cache/book3.jsonl'):
        r = json.loads(l)
        sp = [(s['span'][0], s['span'][1], s['op'])
              for s in (r.get('op_spans') or []) if s['op'] in OPS]
        if sp: out.append((r['raw'], sp))
    def opg(f):
        if f["ftype"] == "rel":
            if f.get("op") == "mul" and len(set(f.get("args", []))) == 1: return "sq"
            return {"add": "addf", "sub": "addf", "mul": "mul"}.get(f.get("op"))
        if f["ftype"] == "macro": return None if f.get("name") == "OP_APPLY" else "fr"
        if f["ftype"] == "frac": return "fr"
        return None
    n = 0
    for l in open('.cache/form_mix8.jsonl'):
        if n >= 1500: break
        r = json.loads(l); txt = r.get('text') or ''
        sp = []
        for f in r.get('factors', []):
            c = opg(f)
            if c is None: continue
            for (a, b) in (f.get('spans') or []):
                sp.append((a, b, c))
        if sp: out.append((txt, sp)); n += 1
    return out

def states_for(texts):
    S = []; OFF = []
    for s0 in range(0, len(texts), 8):
        sl = texts[s0:s0+8]
        ids = np.zeros((8, T_ALG), np.int32); msk = np.zeros((8, T_ALG), np.float32)
        offs = []
        for i, t in enumerate(sl):
            e = tok.encode(t)
            tid = e.ids[:T_ALG]
            ids[i, :len(tid)] = tid; msk[i, :len(tid)] = 1.0
            offs.append(list(e.offsets)[:T_ALG])
        st = np.asarray(recompute_states(ids)).astype(np.float32)
        for i in range(len(sl)):
            S.append(st[i]); OFF.append(offs[i])
    return S, OFF

def main():
    sites = cue_sites()
    print(f"[nm] site rows {len(sites)}", flush=True)
    S, OFF = states_for([t for t, _ in sites])
    X = []; Y = []
    rngn = np.random.default_rng(0)
    for (txt, sp), st, off in zip(sites, S, OFF):
        cue_tok = set()
        for (a, b, c) in sp:
            idxs = [t for t, o in enumerate(off) if o[1] > a and o[0] < b]
            for t in idxs: cue_tok.add(t)
            if idxs:
                X.append(st[idxs].mean(0)); Y.append(OPS.index(c))
        ntoks = [t for t, o in enumerate(off) if o[1] > 0 and t not in cue_tok]
        for t in rngn.choice(ntoks, min(3, len(ntoks)), replace=False):
            X.append(st[t]); Y.append(4)          # background class
    X = np.stack(X).astype(np.float32); Y = np.array(Y)
    print(f"[nm] anchors {len(X)}; class dist {np.bincount(Y).tolist()}", flush=True)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd
    rng = np.random.default_rng(1)
    W1 = Tensor(rng.standard_normal((2048, 256)).astype(np.float32) * 0.02)
    b1 = Tensor(np.zeros(256, np.float32))
    W2 = Tensor(rng.standard_normal((256, 128)).astype(np.float32) * 0.05)
    for t_ in (W1, b1, W2): t_.requires_grad = True
    opt = AdamW([W1, b1, W2], lr=3e-4, weight_decay=0.0)
    Tensor.training = True
    B = 256; TAU = 0.1
    for s in range(STEPS):
        idx = rng.choice(len(Xn), B, replace=False)
        xb = Tensor(Xn[idx]); yb = Y[idx]
        z = ((xb @ W1 + b1).relu() @ W2)
        z = z / (z.pow(2).sum(-1, keepdim=True).sqrt() + 1e-6)
        sim = (z @ z.T) / TAU
        eye = Tensor(np.eye(B, dtype=np.float32))
        same = Tensor((yb[:, None] == yb[None, :]).astype(np.float32)) - eye
        exps = (sim - sim.max(axis=1, keepdim=True)).exp() * (1 - eye)
        denom = exps.sum(1)
        pos = (exps * same).sum(1) + 1e-9
        loss = -(pos / (denom + 1e-9)).log().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 500 == 0:
            print(f"[nm] step {s} loss {float(loss.numpy()):.4f}", flush=True)
    Tensor.training = False
    W1n, b1n, W2n = W1.numpy(), b1.numpy(), W2.numpy()
    def proj(v):
        z = np.maximum((v - mu) / sd @ W1n + b1n, 0) @ W2n
        return z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-9)
    # z-space leader centroids from class anchors (background excluded)
    CENT = {}
    for ci, c in enumerate(OPS):
        Z = proj(X[Y == ci])
        m = []; n = []
        for v in Z:
            if m:
                M = np.stack(m); d = 1 - (M / np.linalg.norm(M, axis=1, keepdims=True)) @ v
                j = int(d.argmin())
                if d[j] < 0.10:
                    m[j] = (m[j] * n[j] + v) / (n[j] + 1); n[j] += 1; continue
            m.append(v.copy()); n.append(1)
        M = np.stack(m); CENT[c] = M / np.linalg.norm(M, axis=1, keepdims=True)
        print(f"[nm] {c}: {len(M)} z-centroids", flush=True)
    BG = proj(X[Y == 4])
    bgm = []; bgn = []
    for v in BG:
        if bgm:
            M = np.stack(bgm); d = 1 - (M / np.linalg.norm(M, axis=1, keepdims=True)) @ v
            j = int(d.argmin())
            if d[j] < 0.10:
                bgm[j] = (bgm[j] * bgn[j] + v) / (bgn[j] + 1); bgn[j] += 1; continue
        if len(bgm) < 3000: bgm.append(v.copy()); bgn.append(1)
    BGC = np.stack(bgm); BGC = BGC / np.linalg.norm(BGC, axis=1, keepdims=True)
    print(f"[nm] background: {len(BGC)} z-centroids", flush=True)

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
                if f.get("op") == "mul" and len(set(f.get("args", []))) == 1: c["sq"] += 1
                elif f.get("op") in ("add", "sub"): c["addf"] += 1
                elif f.get("op") == "mul": c["mul"] += 1
                elif f.get("op") == "div": c["fr"] += 1
            elif f["ftype"] == "macro":
                if f.get("name") != "OP_APPLY": c["fr"] += 1
            elif f["ftype"] == "frac": c["fr"] += 1
        return Counter({k: min(v, 7) for k, v in c.items() if v > 0})
    golds = [canon(r) for r in rows]
    gS, gOFF = states_for([r['original'] for r in rows])
    def f1(a, b):
        i = sum((a & b).values()); return 2 * i / max(sum(a.values()) + sum(b.values()), 1)
    exo = ex_ne = 0; f1s = []
    for r, g, st, off in zip(rows, golds, gS, gOFF):
        text = r['original']
        c = Counter()
        for m in OW.MATH.finditer(text):
            c += OW._scan_math(m.group(1) or m.group(2) or '')
        mm = np.zeros(len(text) + 1, bool)
        for m in OW.MATH.finditer(text):
            mm[m.start():m.end()] = True
        Zt = proj(st)
        lab = np.full(T_ALG, -1, np.int8)
        for t, o in enumerate(off):
            if o[1] == 0 or mm[min(o[0], len(text) - 1)]: continue
            v = Zt[t]
            dbg = float((1 - BGC @ v).min())
            best, bc = 10.0, -1
            for ci, cls in enumerate(OPS):
                d = float((1 - CENT[cls] @ v).min())
                if d < best: best, bc = d, ci
            if best < THR and best < dbg: lab[t] = bc
        prev = -1
        for t in range(T_ALG):
            if lab[t] >= 0 and lab[t] != prev: c[OPS[lab[t]]] += 1
            prev = lab[t]
        d = Counter({k: min(v, 7) for k, v in c.items() if v > 0})
        if d == g:
            exo += 1
            if g: ex_ne += 1
        f1s.append(f1(d, g))
    print(f"[nm] METRIC-HEAD HYBRID (thr {THR}): exact {exo}/143 "
          f"(nonempty {ex_ne})  F1 {np.mean(f1s):.3f}  (floor: v0 29/25)", flush=True)

if __name__ == "__main__":
    main()
