"""structural_rate_audit.py — THE STRUCTURAL-RATE AUDIT (2026-07-31,
registered before firing). Turns [428]'s >=1/263 floor into a RATE.
METRIC 1 (headline): numeric surplus — any asserted value/k/p exceeding
its text license (count-based), text-anchored, convention-free.
METRICS 2-3 (characterization only, synonymy folded at design time):
winning-parse kind multiset vs the annotation's (sub->add before
comparing; divergence split KNOWN-SYNONYMY vs OTHER) and factor-count
deltas. Pre-stated interpretation: ~1% -> footnote; ~15% -> systematic
structural noise the gate learns from."""
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
print(f"[rate-audit] gate from manifest: {CKPT}")

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

def numeric_surplus(text, parse):
    tc = text_number_counts(text)
    asserted = Counter()
    for fa in parse:
        for key in ("value", "k", "p"):
            if key in fa:
                asserted[int(fa[key])] += 1
    return [(v, asserted[v], tc.get(v, 0)) for v in asserted if asserted[v] > tc.get(v, 0)]

def kind(f):
    ft = f["ftype"]
    if ft == "rel":
        op = f.get("op", "?")
        return f"rel-{'add' if op in ('add', 'sub') else op}"   # synonymy folded
    if ft == "macro": return f"macro-{f.get('name','?')}"
    return ft

surplus_rows = []
div_known = div_other = exact = 0
count_deltas = Counter()
n_done = 0
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
        # metric 1: numeric surplus (any winning view)
        row_surplus = None
        for f_ in win:
            s = numeric_surplus(dialect, f_)
            if s:
                row_surplus = s[:4]; break
        if row_surplus:
            surplus_rows.append({"draft": draft.split("/")[-1], "i": i,
                                 "src_idx": r["gen"]["src_idx"], "surplus": row_surplus})
        # metrics 2-3 (view 0's winning parse vs annotation, characterization)
        if win:
            pk = Counter(kind(f) for f in win[0])
            ak = Counter(kind(f) for f in r["factors"])
            had_sub = any(f.get("op") == "sub" for f in r["factors"])
            if pk == ak:
                exact += 1
            elif had_sub:
                div_known += 1
            else:
                div_other += 1
            count_deltas[sum(pk.values()) - sum(ak.values())] += 1
        n_done += 1
        if n_done % 50 == 0: print(f"  [{n_done}]", flush=True)

rate = len(surplus_rows) / max(n_done, 1)
print(f"\n=== METRIC 1 — THE HEADLINE RATE: numeric surplus "
      f"{len(surplus_rows)}/{n_done} = {rate:.1%} ===")
interp = ("~footnote territory (scope line stands as written)" if rate <= 0.03
          else "SYSTEMATIC — structural noise in the training corpus, consequences downstream"
          if rate >= 0.10 else "intermediate — the scope line hardens; a bench read prices next steps")
print(f"=== pre-stated interpretation: {interp} ===")
for s in surplus_rows[:12]:
    print(f"  {s['draft']} i={s['i']} src={s['src_idx']}: {s['surplus']}")
print(f"\n[metric 2 — characterization] kind-multiset vs annotation "
      f"(sub folded): exact {exact}  known-synonymy {div_known}  OTHER {div_other}"
      f"  (of {exact+div_known+div_other} with winning view-0)")
print(f"[metric 3 — characterization] factor-count deltas (parse - annotation): "
      f"{dict(sorted(count_deltas.items()))}")
json.dump({"n": n_done, "surplus_rows": surplus_rows, "rate": rate,
           "kind_exact": exact, "kind_known_syn": div_known, "kind_other": div_other,
           "count_deltas": {str(k): v for k, v in count_deltas.items()}},
          open(".cache/structural_rate_audit.json", "w"), indent=1)
print("[saved] .cache/structural_rate_audit.json")
