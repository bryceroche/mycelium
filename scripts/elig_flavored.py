"""elig_flavored.py — THE FLAVORED WITNESS (2026-08-29, word given): concrete add/sub where the source knows the flavor (the scanner ALWAYS knows: + vs binary −; mint spans carry op; wild cues flavor by semantics), addf-wildcard ONLY where genuinely ambiguous -> branching collapses -> uniqueness returns honestly. Original: elig_read.py — THE ELIGIBILITY WIRING (2026-08-29): the standing witness (scanner + NL-atlas geometry, the 29/143 instrument) drives ENUMERATION: witness multiset -> reachable() with ADDF-WILDCARD (each addf application branches add|sub) -> unique-root EMISSION graded by the key. Fixtures: 143 golds + wv 20 + held 20. Baseline: the chain-based eligibility era (~9% eligible, lies-dominated unique). Original: nl_atlas_v0.py — THE NL-ATLAS (2026-08-28, the emergency brake honored):
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
OPS = ("add", "sub", "mul", "sq", "fr")
ADDW = ('sum','total','plus','more','gain','increas','combin','altogether','older','taller','add','together','both')
SUBW = ('differ','less','fewer','minus','subtract','younger','left','remain','lost','exceed','shorter','decreas','away','neither','not ')
CLS = {c: i for i, c in enumerate(OPS)}
THR = float(os.environ.get("NLA_THR", "0.35"))
MAXC = 4000

def flavor(cue, fallback):
    cl = cue.lower()
    a = any(w in cl for w in ADDW); b = any(w in cl for w in SUBW)
    if a and not b: return "add"
    if b and not a: return "sub"
    return fallback

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
            return {"add": "add", "sub": "sub", "mul": "mul"}.get(f.get("op"))
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

NUM = re.compile(r"(?<![\d.])(\d+)(?![\d.])")

def reachable_w(leaves, ops, cap=200000):
    """addf-wildcard enum: 'addf' tries add AND sub per application."""
    calls = [0]; seen = set(); roots = set()
    def rec(avail, left):
        calls[0] += 1
        if calls[0] > cap: return
        key = (tuple(sorted(avail)), tuple(sorted(left)))
        if key in seen: return
        seen.add(key)
        if not left:
            for v in avail: roots.add(v)
            return
        n = len(avail)
        for op in set(left):
            rest = list(left); rest.remove(op)
            if op == "sq":
                for i in range(n):
                    v = avail[i] * avail[i]
                    if v > 300: continue
                    rec(avail[:i] + (v,) + avail[i+1:], tuple(rest))
            elif op == "fr":
                for i in range(n):
                    for j in range(n):
                        if i == j: continue
                        a, k = avail[i], avail[j]
                        if k < 2 or a % k: continue
                        na = tuple(x for t2, x in enumerate(avail) if t2 not in (i, j)) + (a // k,)
                        rec(na, tuple(rest))
            else:
                for i in range(n):
                    for j in range(n):
                        if i == j: continue
                        a, b = avail[i], avail[j]
                        cands = []
                        if op == "mul": cands = [a * b]
                        elif op == "add": cands = [a + b]
                        elif op == "sub": cands = [a - b]
                        elif op == "addf": cands = [a + b, a - b]
                        for v in cands:
                            if not (0 <= v <= 300): continue
                            na = tuple(x for t2, x in enumerate(avail) if t2 not in (i, j)) + (v,)
                            rec(na, tuple(rest))
    rec(tuple(leaves), tuple(ops))
    return roots, calls[0] > cap

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
    wv = [json.loads(l) for l in open('.cache/g55_wildval.jsonl')]
    never = [{"original": r["original"], "answer": r["answer"], "tag": "wv"} for r in wv]
    dd = [json.loads(l) for l in open('.cache/base_t7self_deeds.jsonl')]
    h = [json.loads(l) for l in open('.cache/math_harvest_v0.jsonl')]
    drafted = set(byid) | sk | set(x["src_idx"] for x in dd)
    rng2 = np.random.default_rng(99)
    for seed in (99, 299):
        rg = np.random.default_rng(seed)
        never += [{"original": h[i]["problem"], "answer": int(str(h[i]["answer"]).strip()), "tag": "held"}
                  for i in rg.permutation(len(h)) if i not in drafted
                  and str(h[i]["answer"]).strip().isdigit()][:10]
    allrows = [dict(original=r['original'], answer=r['answer'], tag='gold') for r in rows] + never
    gS, gOFF = states_for([r['original'] for r in allrows])
    def f1(a, b):
        i = sum((a & b).values()); return 2 * i / max(sum(a.values()) + sum(b.values()), 1)
    keys = Counter(tuple(sorted(g.items())) for g in golds)
    cex = keys.most_common(1)[0][1]
    T = {t: {"n": 0, "wit": 0, "elig": 0, "cover": 0, "uniq": 0, "ur": 0, "ul": 0}
         for t in ("gold", "wv", "held")}
    for r, st, off in zip(allrows, gS, gOFF):
        text = r['original']
        c = Counter()
        for m in OW.MATH.finditer(text):
            e = (m.group(1) or m.group(2) or '')
            e2 = __import__('re').sub(r'\\(?:text|mathrm|hphantom|underline)\{[^}]*\}', ' ', e)
            import re as _re
            c['add'] += len(_re.findall(r'\+', e2))
            c['sub'] += len(_re.findall(r'(?<=[\w)\}])\s*-\s*(?=[\w(\\])', e2))
            c['mul'] += len(_re.findall(r'\\cdot|\\times', e2))
            c['mul'] += len(_re.findall(r'(?<=\d)(?=[a-z]\b)', e2))
            c['mul'] += len(_re.findall(r'(?<=\d)\(', e2))
            c['sq'] += len(_re.findall(r'\^\{?2\}?(?![0-9])', e2))
            nfr = len(_re.findall(r'\\d?frac\s*\{\s*\d+\s*\}\s*\{\s*\d+\s*\}', e2))
            afr = len(_re.findall(r'\\d?frac', e2))
            c['fr'] += (afr - nfr) + len(_re.findall(r'\\div|(?<=[\w}])/(?=[\w{])', e2))
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
        t = T[r['tag']]; t['n'] += 1
        if not d: continue
        t['wit'] += 1
        ops = []
        for cls, k in d.items(): ops += [cls] * k
        nums = [int(m.group(1)) for m in NUM.finditer(r['original'])]
        if not nums or len(nums) > 8 or len(ops) > 6: continue
        t['elig'] += 1
        roots, blown = reachable_w(nums, ops)
        key = r['answer']
        if key in roots: t['cover'] += 1
        if len(roots) == 1:
            t['uniq'] += 1
            if key in roots: t['ur'] += 1
            else: t['ul'] += 1
    for tag, t in T.items():
        print(f"[elig {tag}] n={t['n']} witnessed {t['wit']} eligible {t['elig']} "
              f"coverage {t['cover']}/{t['elig']} unique {t['uniq']} "
              f"(right {t['ur']} lies {t['ul']})", flush=True)

if __name__ == "__main__":
    main()
