"""interference_matrix.py — THE INTERFERENCE INSTRUMENT (2026-07-11,
registered; rides the capacity probe). Crowding = destructive gradient
interference in shared weights. Per-register gradient cosine matrix on
a checkpoint: for each register, assemble batches from ITS rows of the
mixed7b training set (provenance by exact text match), run forward +
the training loss, backward WITHOUT stepping, flatten grads over all
head params, average over microbatches; report pairwise cosines.

Joint verdict table with the capacity probe — four pre-written
quadrants (see ledger 2026-07-11): the sign of the off-diagonal
cosines x whether bigtest erosion reverses at 2x attributes the kill
completely.

Env: IM_CKPT (default gen-7b), IM_BATCHES (microbatches/register, def 4).
"""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "6")
os.environ.setdefault("ALG_TRAIN", ".cache/algebra_mixed7b_train.jsonl")
os.environ.setdefault("ALG_TRAIN_NAME", "m7btrain")
os.environ.setdefault("ALG_TEST", ".cache/dag7b_test.jsonl")
os.environ.setdefault("ALG_TEST_NAME", "dag7btest")
import phase1_algebra_head as H
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

CKPT = os.environ.get("IM_CKPT", ".cache/phase1_gen7b_head.safetensors")
NB = int(os.environ.get("IM_BATCHES", "4"))

SOURCES = {
    "nl-core":  [".cache/algebra_nl_train.jsonl"],
    "alg2":     [".cache/algebra2_nl_train.jsonl"],
    "alg3":     [".cache/algebra3_nl_train.jsonl"],
    "alg4":     [".cache/algebra4_nl_train.jsonl"],
    "verbose":  [".cache/algv_train_verbose.jsonl"],
    "dag6":     [".cache/dag_train.jsonl"],
    "dag7":     [".cache/dag7_train.jsonl", ".cache/dag7b_train.jsonl"],
}
text2reg = {}
for reg, paths in SOURCES.items():
    for path in paths:
        for l in open(path):
            text2reg[json.loads(l)["text"]] = reg

samples, states, tokmask, gold, sent = H.load_alg("train")
reg_idx = {r: [] for r in SOURCES}
for i, s in enumerate(samples):
    r = text2reg.get(s["text"])
    if r:
        reg_idx[r].append(i)
print("[im] register rows:", {r: len(v) for r, v in reg_idx.items()})

p = H.build_params(0)
sd = safe_load(CKPT)
assert set(sd.keys()) == set(p.keys())
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
keys = sorted(p.keys())

FEED = (("presence", "presence", dtypes.float), ("is_lit_f", "is_lit", dtypes.float),
        ("args", "args", dtypes.float), ("fspan", "fspan", dtypes.float),
        ("vspan", "vspan", dtypes.float), ("ftype", "ftype", dtypes.int),
        ("op", "op", dtypes.int), ("res", "res", dtypes.int),
        ("digits", "digits", dtypes.int), ("query", "query", dtypes.int),
        ("sel", "sel", dtypes.int), ("is_rel", "is_rel", dtypes.float),
        ("is_mod", "is_mod", dtypes.float), ("is_sel", "is_sel", dtypes.float),
        ("is_pct", "is_pct", dtypes.float), ("is_fdiv", "is_fdiv", dtypes.float))

def register_grad(idxs, seed):
    rng = np.random.RandomState(seed)
    acc = None
    for b in range(NB):
        idx = rng.choice(idxs, 8, replace=False)
        Tensor.training = True
        for k in p:
            p[k].grad = None
        o = H.forward(p,
                      Tensor(states[idx].astype(np.float32), dtype=dtypes.float),
                      Tensor(tokmask[idx].astype(np.float32), dtype=dtypes.float),
                      Tensor(sent[idx].astype(np.int32), dtype=dtypes.int))
        bg = {}
        for bk, gk, dt in FEED:
            npdt = np.float32 if dt == dtypes.float else np.int32
            bg[bk] = Tensor(gold[gk][idx].astype(npdt), dtype=dt)
        l = H.loss_fn(o, bg)
        l.backward()
        vec = np.concatenate([p[k].grad.numpy().ravel() for k in keys
                              if p[k].grad is not None])
        acc = vec if acc is None else acc + vec
        Tensor.training = False
    return acc / NB

regs = [r for r in SOURCES if len(reg_idx[r]) >= 32]
grads = {}
for ri, r in enumerate(regs):
    grads[r] = register_grad(reg_idx[r], 1000 + ri)
    print(f"[im] {r}: grad norm {np.linalg.norm(grads[r]):.4f}", flush=True)

print(f"\n=== GRADIENT COSINE MATRIX ({CKPT}) ===")
print("          " + "".join(f"{r:>9s}" for r in regs))
for a in regs:
    row = [float(np.dot(grads[a], grads[b]) /
                 (np.linalg.norm(grads[a]) * np.linalg.norm(grads[b]) + 1e-12))
           for b in regs]
    print(f"  {a:8s}" + "".join(f"{v:9.3f}" for v in row))
off = [float(np.dot(grads[a], grads[b]) /
             (np.linalg.norm(grads[a]) * np.linalg.norm(grads[b]) + 1e-12))
       for i, a in enumerate(regs) for b in regs[i+1:]]
print(f"\n  off-diagonal: mean {np.mean(off):+.3f} min {np.min(off):+.3f} "
      f"(anti-aligned = destructive interference)")
