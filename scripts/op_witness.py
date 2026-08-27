"""op_witness.py — THE TWO-REGISTER FRONT DOOR (2026-08-28, word given:
LETS GO): symbol scanner (guarded) inside math regions + prose registry
outside them + abstention everywhere. Deterministic, auditable, every
rule a permanent unit of capability. Grades vs the canonical rock.
Exports op_witness(text) -> Counter for downstream (enum eligibility).
"""
import json, re, glob, os
import numpy as np
from collections import Counter

OPS = ("addf", "mul", "sq", "fr")
MATH = re.compile(r'\$([^$]+)\$|\\begin\{align\*?\}(.+?)\\end\{align\*?\}', re.S)

def _build_prose():
    lex = {c: set() for c in OPS}
    if os.path.exists('.cache/book3.jsonl'):
        for l in open('.cache/book3.jsonl'):
            r = json.loads(l)
            for s in (r.get('op_spans') or []):
                if s['op'] in lex and 2 <= len(s['cue']) <= 60:
                    lex[s['op']].add(s['cue'].lower())
    def opg(f):
        if f["ftype"] == "rel":
            if f.get("op") == "mul" and len(set(f.get("args", []))) == 1: return "sq"
            return {"add": "addf", "sub": "addf", "mul": "mul"}.get(f.get("op"))
        if f["ftype"] == "macro": return None if f.get("name") == "OP_APPLY" else "fr"
        if f["ftype"] == "frac": return "fr"
        return None
    NUMRE = re.compile(r"\d+")
    mint = {c: Counter() for c in OPS}
    for i, l in enumerate(open('.cache/form_mix8.jsonl')):
        if i >= 30000: break
        r = json.loads(l); txt = r.get('text') or ''
        for f in r.get('factors', []):
            c = opg(f)
            if c is None: continue
            for (a, b) in (f.get('spans') or []):
                ph = NUMRE.sub('#', txt[a:b].lower()).strip()
                if 3 <= len(ph) <= 60: mint[c][ph] += 1
    for c in OPS:
        for ph, n in mint[c].items():
            if n >= 5 and max(mint[c2][ph] for c2 in OPS if c2 != c) * 3 <= n:
                lex[c].add(ph)
    return {c: [re.compile(re.escape(p).replace(r'\#', r'\d+'))
                for p in sorted(v, key=len, reverse=True)] for c, v in lex.items()}

_PROSE = None

def _scan_math(e):
    c = Counter()
    e = re.sub(r'\\(?:text|mathrm|hphantom|underline)\{[^}]*\}', ' ', e)
    c['addf'] += len(re.findall(r'\+', e))
    c['addf'] += len(re.findall(r'(?<=[\w)\}])\s*-\s*(?=[\w(\\])', e))
    c['mul'] += len(re.findall(r'\\cdot|\\times', e))
    c['mul'] += len(re.findall(r'(?<=\d)(?=[a-z]\b)', e))           # 3x only
    c['mul'] += len(re.findall(r'(?<=\d)\(', e))                    # 3( only (kills f( )
    c['sq'] += len(re.findall(r'\^\{?2\}?(?![0-9])', e))            # ^{2} not ^{28}
    # numeric-only \frac{3}{5} is a VALUE (a rational literal), not a div op;
    # variable-bearing \frac{3}{x} stays an operation
    nfr = len(re.findall(r'\\d?frac\s*\{\s*\d+\s*\}\s*\{\s*\d+\s*\}', e))
    afr = len(re.findall(r'\\d?frac', e))
    c['fr'] += (afr - nfr)
    c['fr'] += len(re.findall(r'\\div|(?<=[\w}])/(?=[\w{])', e))
    return c

TYPE_ABSTAIN = re.compile(r'\bsimplify\b|\bsolve for\b|in the form|minimum value|maximum value|expressed? as', re.I)

def op_witness(text):
    global _PROSE
    if _PROSE is None: _PROSE = _build_prose()
    if TYPE_ABSTAIN.search(text):
        return Counter()          # transformation-heavy type: surface != graph; abstain
    c = Counter()
    for m in MATH.finditer(text):
        c += _scan_math(m.group(1) or m.group(2) or '')
    prose = MATH.sub(' ', text).lower()
    claimed = []
    for cls in OPS:
        for rex in _PROSE[cls]:
            for m in rex.finditer(prose):
                if not any(m.start() < b and m.end() > a for a, b in claimed):
                    claimed.append((m.start(), m.end())); c[cls] += 1
    return Counter({k: min(v, 7) for k, v in c.items() if v > 0})

def main():
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
    def f1(a, b):
        i = sum((a & b).values()); return 2 * i / max(sum(a.values()) + sum(b.values()), 1)
    keys = Counter(tuple(sorted(g.items())) for g in golds)
    cex = keys.most_common(1)[0][1]
    cf1 = max(np.mean([f1(Counter(dict(k)), g) for g in golds]) for k in keys)
    exo = ex_ne = 0; f1s = []
    tp = Counter(); fp = Counter(); fn = Counter()
    for r, g in zip(rows, golds):
        d = op_witness(r['original'])
        if d == g:
            exo += 1
            if g: ex_ne += 1
        f1s.append(f1(d, g))
        for cls in OPS:
            tp[cls] += min(d[cls], g[cls]); fp[cls] += max(d[cls]-g[cls], 0); fn[cls] += max(g[cls]-g[cls] if False else g[cls]-d[cls] if g[cls]>d[cls] else 0, 0)
    print(f"[fuse] TWO-REGISTER WITNESS: exact {exo}/143 (nonempty {ex_ne})  "
          f"F1 {np.mean(f1s):.3f}   ROCK: {cex} / {cf1:.3f}", flush=True)
    for cls in OPS:
        P = tp[cls]/max(tp[cls]+fp[cls],1); R = tp[cls]/max(tp[cls]+fn[cls],1)
        print(f"  {cls:5s} P {P:.2f} R {R:.2f}", flush=True)
    print("[fuse] VERDICT: " + ("ABOVE THE ROCK — the front door stands" if exo > cex and np.mean(f1s) > cf1
          else ("exact above, F1 below — partial" if exo > cex else "not above — iterate guards")), flush=True)

if __name__ == "__main__":
    main()
