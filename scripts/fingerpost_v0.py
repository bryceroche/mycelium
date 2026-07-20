"""fingerpost_v0.py — GUT #33, READ (a) v0 (2026-07-20): THE FINGERPOST
PROBE. On banked vote-split items (two distinct answers at counts (3,2)
or (2,2) among the gen-14 gate's five views), reconstruct the two
witness parses, render each to canonical dialect via the DETERMINISTIC
writer, and let the frozen trunk adjudicate: pooled-state similarity of
the original text to each restatement. Grade against gold (gold GRADES,
never gates): does the trunk prefer the TRUE reading's restatement?

PINNED BARS: preference-for-truth >=60% = the fingerpost points;
<=55% = dies for the price of a probe. Lengths logged per the law
(near-cancelling by construction). THE LOCUS RIDER: every adjudication
logs the factor-kind diff — the contested-binding census, free.
"""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from collections import Counter, defaultdict
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "6")
os.environ.setdefault("ALG_DUP", "1")
from phase1_algebra_head import T_ALG, L_FAC, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_views import permuted_view
from tta_alg2_dials import solve2
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

rows = [json.loads(l) for l in open(".cache/algebra_nl_bigtest.jsonl")]
gold = [r["solution"][r["query_var"]] for r in rows]
gate_votes = json.load(open(".cache/lattice_gate.json"))["bigtest"]
z = np.load(".cache/phase1_alg_states_bigtest.npz")
states0, tokmask0, sent0 = z["states"], z["tokmask"], z["sent"]

# split items: exactly two distinct non-None answers, counts (3,2)/(2,2),
# with GOLD among them (gradable)
splits = []
for i, v in enumerate(gate_votes):
    c = Counter(x for x in v if x is not None)
    top = c.most_common()
    if len(top) == 2 and top[0][1] + top[1][1] >= 4 and top[1][1] >= 2:
        a1, a2 = top[0][0], top[1][0]
        if gold[i] in (a1, a2) and a1 != a2:
            splits.append((i, a1, a2))
print(f"[fingerpost] gradable split items: {len(splits)}")

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
CKPT = os.environ.get("GATE_CKPT", ".cache/phase1_gen14_head.safetensors")
sd = safe_load(CKPT)
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()


def head_parse(sts, msk, snt):
    N = ((len(sts) + 7) // 8) * 8
    pad = N - len(sts)
    if pad:
        sts = np.concatenate([sts, sts[:1].repeat(pad, 0)])
        msk = np.concatenate([msk, msk[:1].repeat(pad, 0)])
        snt = np.concatenate([snt, snt[:1].repeat(pad, 0)])
    res = []
    for s0 in range(0, N, 8):
        out = forward(p, Tensor(sts[s0:s0+8].astype(np.float32), dtype=dtypes.float),
                      Tensor(msk[s0:s0+8].astype(np.float32), dtype=dtypes.float),
                      Tensor(snt[s0:s0+8].astype(np.int32), dtype=dtypes.int))
        keys = [k for k in ("pres", "ftype", "op", "islit", "dig", "args", "res",
                            "query", "sel", "dup") if k in out]
        o = {k: out[k].realize().numpy() for k in keys}
        for bi in range(8):
            res.append(decode({k: o[k][bi] for k in o}))
    return res[:len(res) - pad if pad else None]


# reconstruct per-view parses for split items (view 0 banked; 1-4 recomputed)
def views_of(i):
    texts = [rows[i]["text"]] + [permuted_view(rows[i]["text"], 40000 + 10 * i + k)
                                 for k in range(1, 5)]
    ids = np.zeros((5, T_ALG), np.int32); msk = np.zeros((5, T_ALG), np.float32)
    snt = np.zeros((5, T_ALG), np.int32)
    for vi, t in enumerate(texts):
        e = tok.encode(t)
        Ln = min(len(e.ids), T_ALG)
        ids[vi, :Ln] = e.ids[:Ln]; msk[vi, :Ln] = 1.0
        snt[vi] = sent_indices(t, list(e.offsets), msk[vi])
    sts = recompute_states(ids)
    return head_parse(sts, msk, snt)


LET = "abcdefghijklmnopqrstuvwx"


def render(facs, q):
    """The deterministic dialect writer: says only what the parse says."""
    used = sorted({v for f in facs for v in
                   (list(f.get("args", [])) + [f[k] for k in ("result", "var")
                    if k in f])})
    L = {v: LET[j] for j, v in enumerate(used)}
    s = ["Consider the numbers " + ", ".join(L[v] for v in used) + "."]
    for f in facs:
        if f["ftype"] == "given":
            s.append(f"{L[f['var']]} is {f['value']}.")
        elif f["ftype"] == "rel":
            w = "plus" if f["op"] == "add" else "times"
            s.append(f"{L[f['args'][0]]} {w} {L[f['args'][1]]} equals {L[f['result']]}.")
        elif f["ftype"] == "mod":
            s.append(f"When {L[f['var']]} is divided by {f['k']}, the remainder is {L[f['result']]}.")
        elif f["ftype"] == "fdiv":
            s.append(f"When {L[f['var']]} is divided by {f['k']}, the quotient is {L[f['result']]}.")
        elif f["ftype"] == "pct":
            s.append(f"{L[f['args'][0]]} percent of {L[f['args'][1]]} is taken.")
        else:
            s.append(f"the larger of {L[f['args'][0]]} and {L[f['args'][1]]} is {L[f['result']]}.")
    s.append(f"What is {L.get(q, 'a')}?")
    return " ".join(s)


def fkey(f):
    if f["ftype"] == "rel":
        return ("rel", f["op"], tuple(sorted(f["args"])), f["result"])
    if f["ftype"] == "given":
        return ("given", f["var"], f["value"])
    return (f["ftype"], f.get("var"), f.get("k", f.get("p")), f.get("result"))


def pooled(texts):
    ids = np.zeros((len(texts), T_ALG), np.int32)
    msk = np.zeros((len(texts), T_ALG), np.float32)
    Ls = []
    for i, t in enumerate(texts):
        e = tok.encode(t)
        Ln = min(len(e.ids), T_ALG)
        ids[i, :Ln] = e.ids[:Ln]; msk[i, :Ln] = 1.0
        Ls.append(Ln)
    sts = recompute_states(ids)
    V = (sts.astype(np.float32) * msk[:, :, None]).sum(1) / \
        np.maximum(msk.sum(1)[:, None], 1)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return V, Ls


right = wrong = skipped = 0
locus = Counter()
details = []
for i, a1, a2 in splits:
    parses = views_of(i)
    P = {}
    for facs, q in parses:
        a = solve2(facs, q, {"n_vars": 24, "m": rows[i].get("m", 60)})
        if a in (a1, a2) and a not in P:
            P[a] = (facs, q)
    if len(P) < 2:
        skipped += 1
        continue
    (fa, qa), (fb, qb) = P[a1], P[a2]
    da, db = render(fa, qa), render(fb, qb)
    # original pooled from banked states
    v0 = (states0[i].astype(np.float32) * tokmask0[i][:, None]).sum(0) / \
        max(tokmask0[i].sum(), 1)
    v0 /= np.linalg.norm(v0)
    V, Ls = pooled([da, db])
    s1, s2 = float(V[0] @ v0), float(V[1] @ v0)
    point = a1 if s1 > s2 else a2
    ok = point == gold[i]
    right += ok
    wrong += (not ok)
    ka, kb = set(map(fkey, fa)), set(map(fkey, fb))
    for f in (ka ^ kb):
        locus[f[0]] += 1
    details.append({"i": i, "gold": gold[i], "a1": a1, "a2": a2,
                    "s1": s1, "s2": s2, "point": point, "ok": bool(ok),
                    "len_d1": Ls[0], "len_d2": Ls[1]})

n = right + wrong
acc = right / max(n, 1)
verdict = ("THE FINGERPOST POINTS" if acc >= 0.60 else
           "DIES FOR THE PRICE OF A PROBE" if acc <= 0.55 else "BETWEEN BARS")
print(f"\n=== THE FINGERPOST v0 (n={n} adjudications, {skipped} skipped — "
      f"witness parse unrecoverable) ===")
print(f"  preference-for-truth: {right}/{n} = {acc:.3f}  => {verdict}")
print(f"  contested-binding census (the locus rider): {dict(locus.most_common())}")
json.dump({"n": n, "right": right, "acc": acc, "verdict": verdict,
           "skipped": skipped, "locus": dict(locus),
           "details": details},
          open(".cache/fingerpost_v0.json", "w"))
print("[fingerpost] banked -> .cache/fingerpost_v0.json (per-item, per the law)")
