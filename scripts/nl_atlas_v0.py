"""nl_atlas_v0.py — THE NL-ATLAS (2026-08-28, the emergency brake honored):
prose cue matching by GEOMETRY, not literals. Mine: frozen-trunk token
states at banked cue sites (171 wild + mint spans) -> leader-clustered
Welford centroids per op class. Route: golds token-wise by nearest
centroid under a distance threshold -> cue regions -> counts. The
formal register (notation) stays with the symbol scanner — LaTeX is a
formal language; rules ARE its correct parser. Hybrid = scanner (math
regions) + atlas routing (prose). Floor to beat: the literal-match
front door at 24/143.
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
import op_witness as OW

_ = load_alg("test")
tok = Tokenizer.from_file(TOKENIZER_JSON)
OPS = ("addf", "mul", "sq", "fr")
CLS = {c: i for i, c in enumerate(OPS)}
THR = float(os.environ.get("NLA_THR", "0.35"))
MAXC = 4000

def cue_sites():
    """(text, [(a,b,cls)]) from book3 wild spans + mint spans (canonical)."""
    out = []
    for l in open('.cache/book3.jsonl'):
        r = json.loads(l)
        sp = [(s['span'][0], s['span'][1], s['op'])
              for s in (r.get('op_spans') or []) if s['op'] in CLS]
        if sp: out.append((r['raw'], sp))
    def opg(f):
        if f["ftype"] == "rel":
            if f.get("op") == "mul" and len(set(f.get("args", []))) == 1: return "sq"
            return {"add": "addf", "sub": "addf", "mul": "mul"}.get(f.get("op"))
        if f["ftype"] == "macro": return None if f.get("name") == "OP_APPLY" else "fr"
        if f["ftype"] == "frac": return "fr"
        return None
    n_mint = 0
    for i, l in enumerate(open('.cache/form_mix8.jsonl')):
        if n_mint >= 1500: break
        r = json.loads(l); txt = r.get('text') or ''
        sp = []
        for f in r.get('factors', []):
            c = opg(f)
            if c is None: continue
            for (a, b) in (f.get('spans') or []):
                sp.append((a, b, c))
        if sp:
            out.append((txt, sp)); n_mint += 1
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
    print(f"[nla] cue-site rows: {len(sites)}", flush=True)
    texts = [t for t, _ in sites]
    S, OFF = states_for(texts)
    banks = {c: {"m": [], "n": []} for c in OPS}
    for (txt, sp), st, off in zip(sites, S, OFF):
        for (a, b, c) in sp:
            idxs = [t for t, o in enumerate(off) if o[1] > a and o[0] < b]
            if not idxs: continue
            v = st[idxs].mean(0)
            v = v / (np.linalg.norm(v) + 1e-9)
            B = banks[c]
            if B["m"]:
                M = np.stack(B["m"])
                Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
                d = 1 - Mn @ v
                j = int(d.argmin())
                if d[j] < 0.15:
                    B["m"][j] = (B["m"][j] * B["n"][j] + v) / (B["n"][j] + 1)
                    B["n"][j] += 1
                    continue
            if len(B["m"]) < MAXC:
                B["m"].append(v.copy()); B["n"].append(1)
    CENT = {}
    for c in OPS:
        M = np.stack(banks[c]["m"])
        CENT[c] = M / np.linalg.norm(M, axis=1, keepdims=True)
        print(f"[nla] {c}: {len(M)} centroids", flush=True)

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
    keys = Counter(tuple(sorted(g.items())) for g in golds)
    cex = keys.most_common(1)[0][1]
    exo = ex_ne = 0; f1s = []
    for r, g, st, off in zip(rows, golds, gS, gOFF):
        text = r['original']
        c = Counter()
        for m in OW.MATH.finditer(text):
            c += OW._scan_math(m.group(1) or m.group(2) or '')
        # prose: token-wise atlas routing OUTSIDE math regions
        mathmask = np.zeros(len(text) + 1, bool)
        for m in OW.MATH.finditer(text):
            mathmask[m.start():m.end()] = True
        lab = np.full(T_ALG, -1, np.int8)
        for t, o in enumerate(off):
            if o[1] == 0 or mathmask[min(o[0], len(text) - 1)]:
                continue
            v = st[t] / (np.linalg.norm(st[t]) + 1e-9)
            best, bc = 10.0, -1
            for ci, cls in enumerate(OPS):
                d = float((1 - CENT[cls] @ v).min())
                if d < best: best, bc = d, ci
            if best < THR: lab[t] = bc
        prev = -1
        for t in range(T_ALG):
            if lab[t] >= 0 and lab[t] != prev:
                c[OPS[lab[t]]] += 1
            prev = lab[t]
        d = Counter({k: min(v, 7) for k, v in c.items() if v > 0})
        if d == g:
            exo += 1
            if g: ex_ne += 1
        f1s.append(f1(d, g))
    print(f"[nla] HYBRID (scanner + NL-ATLAS routing, thr {THR}): "
          f"exact {exo}/143 (nonempty {ex_ne})  F1 {np.mean(f1s):.3f}  "
          f"(literal front door: 24/19; rock {cex})", flush=True)

if __name__ == "__main__":
    main()
