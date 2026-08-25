"""dialect_ladder.py — THE DIALECT-LADDER SURVEY (2026-08-24, word
given; diffusion-parallel framing). One breath sweep over gold-labeled
form8 rows; per (cycle x label-family) nearest-class-centroid accuracy
on a held split -> the 7x6 RELIABILITY SURFACE. Families (slot-grain,
all free from gold): ftype, op, dup(sq-ness), depth, terminality,
row-shape (global-context probe). Deliverable: per-family cycle-weight
PROFILES (the measured noise schedule) — no stage decrees; bars = each
cell vs its family's majority-class chance.
"""
import os, sys, json
os.environ.setdefault("ALG_MINE_BREATHS", "1")
os.environ.setdefault("ALG_BREATH", "7")
os.environ.setdefault("ALG_NOTEBOOK", "1")
os.environ.setdefault("ALG_SIXWAVE", "1")
os.environ.update({"DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "9",
                   "ALG_DUP": "1", "ALG_HW": "512", "ALG_WIDE": "1",
                   "ALG_TRAIN": ".cache/form_mix8.jsonl",
                   "ALG_TRAIN_NAME": "form8",
                   "ALG_ALLOW_PEN_TRAIN": "1",
                   "ALG_TEST": ".cache/algebra_nl_test.jsonl",
                   "ALG_TEST_NAME": "test23"})
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
from collections import Counter
from phase1_algebra_head import (build_params, forward, load_alg,
                                 build_slot_masks, L_FAC)
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

samples, states, tokmask, gold, sent = load_alg("train")
KINDS = ["rel", "given", "mod", "sel", "pct", "fdiv", "macro", "frac", "chain"]
OPS = ["add", "sub", "mul", "div"]
p = build_params(0)
sd = safe_load('.cache/gsb227_real.safetensors')
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
CAP = int(os.environ.get("LADDER_CAP", "600"))
rng = np.random.RandomState(11)
rows = rng.choice(states.shape[0], CAP, replace=False)

def fac_meta(r):
    """per-factor (depth, terminal) from the row's factor list."""
    facs = r["factors"]; q = r["query_var"]
    depth = {}
    def var_of(f):
        return f.get("result", f.get("var", 0))
    def dep(fi, seen):
        f = facs[fi]
        if f["ftype"] == "given": return 0
        srcs = f.get("args", []) if "args" in f else [f.get("var", 0)]
        ds = []
        for v in srcs:
            for fj, g in enumerate(facs):
                if fj != fi and var_of(g) == v and fj not in seen:
                    ds.append(dep(fj, seen | {fj}))
        return 1 + max(ds, default=0)
    out = []
    for fi, f in enumerate(facs):
        d = min(dep(fi, {fi}), 5)
        term = 1 if var_of(f) == q else 0
        out.append((d, term))
    return out

X = {c: [] for c in range(7)}
Y = {fam: [] for fam in ("ftype", "op", "dup", "depth", "term", "shape")}
for s0 in range(0, CAP, 8):
    sl = [int(r) for r in rows[s0:s0 + 8]]
    pad = 8 - len(sl); slp = sl + sl[:1] * pad
    ts = Tensor(states[slp].astype(np.float32), dtype=dtypes.float)
    tk = Tensor(tokmask[slp].astype(np.float32), dtype=dtypes.float)
    se = Tensor(sent[slp].astype(np.int32), dtype=dtypes.int)
    o0 = forward(p, ts, tk, se)
    o0n = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
    mk = build_slot_masks(o0n, sent[slp])
    o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float))
    B = [b.realize().numpy() for b in o["breaths_all"]]
    for bi, ri in enumerate(sl):
        r = samples[ri]
        meta = fac_meta(r)
        shape = hash(tuple(sorted(Counter(
            f.get("op", f["ftype"]) for f in r["factors"]).items()))) % 16
        for j in range(min(L_FAC, len(r["factors"]))):
            if gold["presence"][ri, j] <= 0: continue
            for c in range(min(7, len(B))):
                v = B[c][bi, j].astype(np.float32)
                X[c].append(v / max(np.linalg.norm(v), 1e-9))
            Y["ftype"].append(int(gold["ftype"][ri, j]))
            Y["op"].append(int(gold["op"][ri, j])
                           if KINDS[int(gold["ftype"][ri, j])] == "rel" else -1)
            Y["dup"].append(int(gold["arg_dup"][ri, j] > 0.5)
                            if "arg_dup" in gold else -1)
            d, t = meta[j] if j < len(meta) else (0, 0)
            Y["depth"].append(d); Y["term"].append(t)
            Y["shape"].append(shape)

n = len(Y["ftype"])
print(f"[ladder] {n} labeled slot-instances swept", flush=True)
idx = np.random.RandomState(3).permutation(n)
tr, te = idx[:int(n * 0.7)], idx[int(n * 0.7):]
print("[ladder] cycle x family accuracy (vs chance):", flush=True)
hdr = "cyc  " + "  ".join(f"{f:>6s}" for f in Y)
print(hdr, flush=True)
for c in range(7):
    Xc = np.stack(X[c])
    cells = []
    for fam, lab in Y.items():
        lab = np.array(lab)
        ok = lab >= 0
        tr_f = [i for i in tr if ok[i]]; te_f = [i for i in te if ok[i]]
        if len(te_f) < 20: cells.append("   -  "); continue
        cents = {}
        for cl in set(lab[tr_f].tolist()):
            m = Xc[[i for i in tr_f if lab[i] == cl]].mean(0)
            cents[cl] = m / (np.linalg.norm(m) + 1e-9)
        cls = list(cents)
        C = np.stack([cents[c2] for c2 in cls])
        pred = np.array(cls)[np.argmax(Xc[te_f] @ C.T, 1)]
        acc = float((pred == lab[te_f]).mean())
        chance = float(Counter(lab[tr_f].tolist()).most_common(1)[0][1] / len(tr_f))
        cells.append(f"{acc:.2f}/{chance:.2f}"[:11].ljust(6))
    print(f"  {c}  " + "  ".join(cells), flush=True)
