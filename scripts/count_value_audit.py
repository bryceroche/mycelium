"""count_value_audit.py — THE SET-VS-COUNT FALSE-TRIP AUDIT
(2026-07-31, bench work, audit-before-install per the value-check
precedent). The binding-swap blind spot: the installed value check uses
SET membership — a parse asserting a text number MORE TIMES than the
text licenses passes (26% of book rows carry repeated digits, the
live precondition). The COUNT rule candidate: for each numeric v, the
number of ASSERTING factors (given.value, fdiv/mod.k, pct.p == v) must
not exceed v's occurrence count in the text (digits + number-words).
INSTALL BAR (pinned, the precedent's own): 0 false trips on the 263
gold-certified rows -> install-eligible in the shared trace layer;
any false trip -> refuted or redesigned, never installed silently."""
import sys, os, json, glob, re
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_HW", "512"); os.environ.setdefault("ALG_DUP", "1")
import numpy as np
from collections import Counter
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_views import permuted_view
from tta_alg2_dials import solve2
from mycelium.trace_layer import WORDNUM
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

MAN = json.load(open(".cache/GENERATION.json"))
CKPT = MAN["parser_ckpt"]
tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
sd = safe_load(CKPT)
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
print(f"[count-audit] gate from manifest: {CKPT}")

def parse_batch(texts):
    n = len(texts); N = ((n+7)//8)*8
    ids = np.zeros((N, T_ALG), np.int32); msk = np.zeros((N, T_ALG), np.float32); snt = np.zeros((N, T_ALG), np.int32)
    for i, t in enumerate(texts):
        e = tok.encode(t); Ln = min(len(e.ids), T_ALG)
        ids[i, :Ln] = e.ids[:Ln]; msk[i, :Ln] = 1.0
        snt[i] = sent_indices(t, list(e.offsets), msk[i])
    st = recompute_states(ids)
    out_r = []
    for s0 in range(0, N, 8):
        out = forward(p, Tensor(st[s0:s0+8].astype(np.float32), dtype=dtypes.float),
                      Tensor(msk[s0:s0+8].astype(np.float32), dtype=dtypes.float),
                      Tensor(snt[s0:s0+8].astype(np.int32), dtype=dtypes.int))
        keys = ("pres","ftype","op","islit","dig","args","res","query") + (("sel",) if "sel" in out else ()) + (("dup",) if "dup" in out else ())
        o = {k: out[k].realize().numpy() for k in keys}
        for bi in range(8):
            if s0+bi < n: out_r.append(decode({k: o[k][bi] for k in o}))
    return out_r

def text_number_counts(text):
    c = Counter(int(n) for n in re.findall(r"\d+", text))
    tl = text.lower()
    for w, v in WORDNUM.items():
        c[v] += len(re.findall(r"\b" + w + r"\b", tl))
    return c

def count_trips(text, parse):
    tc = text_number_counts(text)
    asserted = Counter()
    for fa in parse:
        for key in ("value", "k", "p"):
            if key in fa:
                asserted[int(fa[key])] += 1
    return [(v, asserted[v], tc.get(v, 0)) for v in asserted if asserted[v] > tc.get(v, 0)]

false_trips = []
n_done = 0
key_counts = Counter()
for draft in sorted(glob.glob(".cache/book8_*prose_pairs_draft.jsonl")):
    certf = draft.replace("prose_pairs_draft.jsonl", "certification.json")
    if not os.path.exists(certf): certf = ".cache/book8_certification.json"
    rows = [json.loads(l) for l in open(draft)]
    for e in json.load(open(certf))["certified"]:
        i = e["i"]; r = rows[i]; dialect = r["gen"]["dialect"]
        gold = r["solution"][r["query_var"]]
        vt = [dialect] + [permuted_view(dialect, 91000 + 10*i + k) for k in range(1, 5)]
        parsed = parse_batch(vt)
        win = [f_ for (f_, q_) in parsed
               if solve2(f_, q_, {"n_vars": 24, "m": r["m"]}) == gold]
        for f_ in win:
            trips = count_trips(dialect, f_)
            if trips:
                false_trips.append({"draft": draft, "i": i,
                                    "src_idx": r["gen"]["src_idx"],
                                    "trips": trips[:4], "dialect": dialect[:110]})
                print(f"  FALSE TRIP {draft.split('/')[-1]} i={i}: {trips[:4]}", flush=True)
                break
        n_done += 1
        if n_done % 50 == 0: print(f"  [{n_done}]", flush=True)
print(f"\n[count-audit] certified rows checked {n_done}  FALSE TRIPS {len(false_trips)}")
verdict = ("INSTALL-ELIGIBLE (0 false trips — the count rule joins the shared trace layer on the word)"
           if not false_trips else
           f"REFUSED — {len(false_trips)} false trips; the rule is redesigned or dropped, never installed silently")
print(f"=== VERDICT: {verdict} ===")
json.dump({"n_checked": n_done, "false_trips": false_trips, "verdict": verdict},
          open(".cache/count_value_audit.json", "w"), indent=1)
print("[saved] .cache/count_value_audit.json")
