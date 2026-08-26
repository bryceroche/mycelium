"""cue_prep.py — token-grain op-cue labels for form8 (2026-08-26, word
given; rung 1 of the op-cue tagger). fspan x op, computed straight from
the jsonl factors (spans + op fields — construction-known, custody-
clean). Classes: 0=none, 1..7 = add/sub/mul/div/sq/opa/fr. Rows with no
op-factor spans are flagged out of training.
Output: .cache/cue_y_form8.npy (n, T_ALG) int8 + cue_rows_form8.npy mask.
"""
import sys, json
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
from phase1_algebra_head import T_ALG, TOKENIZER_JSON
from tokenizers import Tokenizer

CLS = ["none", "add", "sub", "mul", "div", "sq", "opa", "fr"]
tok = Tokenizer.from_file(TOKENIZER_JSON)

def opg(f):
    if f["ftype"] == "rel":
        if f.get("op") == "mul" and len(set(f.get("args", []))) == 1:
            return "sq"
        return f.get("op")
    if f["ftype"] == "macro":
        return "opa" if f.get("name") == "OP_APPLY" else "fr"
    if f["ftype"] == "frac":
        return "fr"
    return None

rows = [json.loads(l) for l in open('.cache/form_mix8.jsonl')]
n = len(rows)
Y = np.zeros((n, T_ALG), np.int8)
ok = np.zeros(n, bool)
for i, r in enumerate(rows):
    e = tok.encode(r.get("text") or r["original"])
    offs = list(e.offsets)[:T_ALG]
    any_span = False
    for f in r.get("factors", []):
        c = opg(f)
        if c is None or c not in CLS: continue
        ci = CLS.index(c)
        for (a, b) in (f.get("spans") or []):
            for t, (oa, ob) in enumerate(offs):
                if ob > a and oa < b:
                    Y[i, t] = ci; any_span = True
    ok[i] = any_span
    if i % 20000 == 0: print(f"[cue] {i}/{n}", flush=True)
np.save('.cache/cue_y_form8.npy', Y)
np.save('.cache/cue_rows_form8.npy', ok)
frac = Y[ok].astype(bool).mean() if ok.any() else 0
print(f"[cue] rows with op spans: {ok.sum()}/{n}; cue-token fraction "
      f"{frac:.4f}; per-class {np.bincount(Y[ok].ravel(), minlength=8).tolist()}",
      flush=True)
