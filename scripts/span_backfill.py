"""span_backfill.py — RUNG 2 JOB 2: SPAN BACKFILL BY LEXICON PROJECTION
(2026-08-27, word given). Census: sub 4.5%, opa 4.7%, fr 9.5%, mul 52%
spanned (div: zero factors — division lives as fr in the mints; class
retired from the tagger vocabulary). Method: mine cue phrases from the
SPANNED exemplars per class (normalized substrings, class-distinctive),
project onto unspanned op factors of the same class by text match.
Labels honestly tagged PROJECTED (lexicon), vs EMITTED (construction).
Output: .cache/cue_y2_form8.npy + rows mask + a lexicon report.
"""
import sys, json, re
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
from collections import Counter
from phase1_algebra_head import T_ALG, TOKENIZER_JSON
from tokenizers import Tokenizer

CLS = ["none", "add", "sub", "mul", "sq", "opa", "fr"]   # div retired
tok = Tokenizer.from_file(TOKENIZER_JSON)
NUMRE = re.compile(r"\d+")

def opg(f):
    if f["ftype"] == "rel":
        if f.get("op") == "mul" and len(set(f.get("args", []))) == 1:
            return "sq"
        return {"add": "add", "sub": "sub", "mul": "mul"}.get(f.get("op"))
    if f["ftype"] == "macro":
        return "opa" if f.get("name") == "OP_APPLY" else "fr"
    if f["ftype"] == "frac":
        return "fr"
    return None

rows = [json.loads(l) for l in open('.cache/form_mix8.jsonl')]

# ---- pass 1: mine the lexicon from spanned exemplars ----
lex = {c: Counter() for c in CLS[1:]}
for r in rows:
    txt = r.get("text") or r.get("original", "")
    for f in r.get("factors", []):
        c = opg(f)
        if c is None: continue
        for (a, b) in (f.get("spans") or []):
            ph = NUMRE.sub("#", txt[a:b].lower()).strip()
            if 2 <= len(ph) <= 60:
                lex[c][ph] += 1
# distinctive: a phrase belongs to the class where it is >=3x more common
own = {}
for c in CLS[1:]:
    for ph, n in lex[c].items():
        if n < 5: continue
        others = max(lex[c2][ph] for c2 in CLS[1:] if c2 != c) if len(CLS) > 2 else 0
        if n >= 3 * max(others, 1):
            own.setdefault(c, []).append((ph, n))
for c in own:
    own[c].sort(key=lambda t: -t[1])
    print(f"[bf] lexicon {c}: {len(own[c])} phrases; top: "
          f"{[p for p, _ in own[c][:5]]}", flush=True)

# templates with '#' back to regex for projection
def to_re(ph):
    return re.compile(re.escape(ph).replace(r"\#", r"\d+"))
LEXRE = {c: [(to_re(p), len(p)) for p, _ in own.get(c, [])[:400]] for c in CLS[1:]}

# ---- pass 2: emit token labels (emitted spans first, then projection) ----
Y = np.zeros((len(rows), T_ALG), np.int8)
ok = np.zeros(len(rows), bool)
n_proj = n_emit = 0
for i, r in enumerate(rows):
    txt = r.get("text") or r.get("original", "")
    e = tok.encode(txt)
    offs = list(e.offsets)[:T_ALG]
    low = txt.lower()
    claimed = []
    def mark(a, b, ci):
        for t, (oa, ob) in enumerate(offs):
            if ob > a and oa < b:
                Y[i, t] = ci
    for f in r.get("factors", []):
        c = opg(f)
        if c is None or c not in CLS: continue
        ci = CLS.index(c)
        sp = f.get("spans") or []
        if sp:
            for (a, b) in sp:
                mark(a, b, ci); claimed.append((a, b))
            n_emit += 1; ok[i] = True
    for f in r.get("factors", []):
        c = opg(f)
        if c is None or c not in CLS or (f.get("spans")): continue
        ci = CLS.index(c)
        best = None
        for rex, plen in LEXRE.get(c, []):
            m = rex.search(low)
            if m and not any(m.start() < b and m.end() > a for a, b in claimed):
                best = (m.start(), m.end()); break
        if best:
            mark(best[0], best[1], ci); claimed.append(best)
            n_proj += 1; ok[i] = True
    if i % 20000 == 0: print(f"[bf] {i}/{len(rows)}", flush=True)
np.save('.cache/cue_y2_form8.npy', Y)
np.save('.cache/cue_rows2_form8.npy', ok)
bc = np.bincount(Y[ok].ravel(), minlength=7).tolist()
print(f"[bf] rows usable {ok.sum()}/{len(rows)}; emitted-factors {n_emit} "
      f"projected-factors {n_proj}; per-class tokens {bc}", flush=True)
