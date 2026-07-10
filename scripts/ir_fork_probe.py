"""ir_fork_probe.py — the three-outcome IR probe reads (2026-07-10).
Zero-shot register gap + state decodability [(b)/(c)] + mouth column.
(Outcome (a)'s mixed-training eval runs in the chain after this.)"""
import json, os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "6")
from phase1_algebra_head import (T_ALG, L_FAC, N_DIG, build_params, forward,
                                 decode, _spans_to_tokmask, tokenize)
from survivor_depth_probe import train_probe, eval_probe
from tta_alg2_dials import solve2
from survivor_multiplicity import midrank_auc
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

def load(split):
    z = np.load(f".cache/phase1_alg_states_{split}.npz")
    return z["states"], z["tokmask"], z["sent"]

def answers(p, samples, st, tk, se):
    n = len(samples); out = []
    for s0 in range(0, n, 8):
        sl = np.arange(s0, min(s0 + 8, n))
        o_ = forward(p, Tensor(st[sl].astype(np.float32), dtype=dtypes.float),
                     Tensor(tk[sl].astype(np.float32), dtype=dtypes.float),
                     Tensor(se[sl].astype(np.int32), dtype=dtypes.int))
        keys = ("pres","ftype","op","islit","dig","args","res","query") + \
            (("sel",) if "sel" in o_ else ())
        o = {k: o_[k].realize().numpy() for k in keys}
        for bi, i in enumerate(sl):
            facs, q = decode({k: o[k][bi] for k in o})
            a = solve2(facs, q, samples[int(i)])
            g = samples[int(i)]["solution"][samples[int(i)]["query_var"]]
            out.append((a is not None, a == g if a is not None else False))
    return out

p = build_params(0)
sd = safe_load(".cache/phase1_algebra4_head.safetensors")
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

vs = [json.loads(l) for l in open(".cache/algv_test_verbose.jsonl")]
ts = [json.loads(l) for l in open(".cache/algv_test_terse.jsonl")]
sv, kv, ev = load("vtest"); st_, kt, et = load("vttest")
av = answers(p, vs, sv, kv, ev); at = answers(p, ts, st_, kt, et)
print(f"ZERO-SHOT REGISTER GAP (gen-4 head, matched graphs, n={len(vs)}):")
print(f"  terse  : answered-correct {sum(c for _,c in at)} | forced {sum(f for f,_ in at)}")
print(f"  verbose: answered-correct {sum(c for _,c in av)} | forced {sum(f for f,_ in av)}")

# (b)/(c): value decodability at gold given spans, verbose states
def span_rows(path, split):
    samples, ids, mask, offs = tokenize(path)
    st, tk, se = load(split)
    X, Y = [], []
    for i, s in enumerate(samples):
        for f in s["factors"]:
            if f["ftype"] != "given": continue
            m = np.zeros((T_ALG,), np.float32)
            _spans_to_tokmask(f["spans"], offs[i], m)
            if m.sum() == 0: continue
            m /= m.sum()
            X.append((st[i].astype(np.float32) * m[:, None]).sum(0))
            v = int(f["value"])
            Y.append([(v // 10 ** (N_DIG - 1 - d)) % 10 for d in range(N_DIG)])
    return np.stack(X), np.array(Y)
Xtr, Ytr = span_rows(".cache/algv_train_verbose.jsonl", "vtrain")
Xte, Yte = span_rows(".cache/algv_test_verbose.jsonl", "vtest")
mu, sd_ = Xtr.mean(0), Xtr.std(0) + 1e-6
_, fwd = train_probe((Xtr - mu) / sd_, Ytr, seed=1)
acc = eval_probe(fwd, (Xte - mu) / sd_, Yte)
print(f"\n(b)/(c) DISCRIMINATOR: verbose given-value decodability = {acc:.3f}"
      f"  (terse precedent 0.998; >0.9 = states fine, (c) dead)")

# mouth column
mo = np.load(".cache/recognition_mouth.npz")
def pool(st, tk):
    v = (st.astype(np.float32) * tk[:, :, None].astype(np.float32)).sum(1) / \
        np.maximum(tk.sum(1)[:, None], 1)
    return v / np.linalg.norm(v, axis=1, keepdims=True)
pv = pool(sv, kv)
d = 1.0 - pv @ mo["bank"].T
knn = np.sort(d, axis=1)[:, :8].mean(1)
print(f"MOUTH COLUMN: generator-verbose mean kNN {knn.mean():.4f} "
      f"(native thr {float(mo['thr_knn']):.4f}; MATH-500 was ~0.25) | "
      f"read-foreign {float((knn > mo['thr_knn']).mean()):.1%}")
