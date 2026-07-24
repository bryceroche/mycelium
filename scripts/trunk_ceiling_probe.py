"""trunk_ceiling_probe.py — the trunk-ceiling diagnostic's first
administration (2026-07-24; bars pinned in the ledger first).

Decodability-by-depth: pooled trunk states at prefix depths
L2/L4/L6/L8 on the hundreds-held rows, linear softmax probe for the
given value's hundreds digit (7-way, {3..9}), 150 train / 50 test.
DEEPER-BETTER (>= +0.10 over L4) -> the L0-L3 cut loses held
information; FLAT -> the cut is innocent.
"""
import json, sys, os, re
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np

os.environ.setdefault("DEV", "AMD")
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes

T_ALG = 256
rows = [json.loads(l) for l in open(".cache/gen17_hundreds_held.jsonl")]
tok = Tokenizer.from_file(".cache/llama-3.2-1b-weights/tokenizer.json")

# gold: the FIRST given's hundreds digit (the row's contested content)
items = []
for r in rows:
    g = [f for f in r["factors"] if f["ftype"] == "given" and f["value"] >= 300]
    if g:
        items.append((r["text"], g[0]["value"] // 100))
print(f"[ceiling] items with >=300 given: {len(items)}")
items = items[:200]
labels = np.array([d for _, d in items]) - 3          # {3..9} -> {0..6}

from mycelium.llama_loader import (attach_llama_layers, load_llama_weights,
                                   LLAMA_3_2_1B_CFG)


class Host:
    pass


host = Host()
sd = load_llama_weights(".cache/llama-3.2-1b-weights/model.safetensors")
attach_llama_layers(host, n_layers=8, sd=sd, cfg=LLAMA_3_2_1B_CFG)
embed = host.llama_embed
rc, rs = host.llama_rope_cos, host.llama_rope_sin

DEPTHS = [2, 4, 6, 8]
pooled = {d: [] for d in DEPTHS}
for bi in range(0, len(items), 8):
    batch = items[bi:bi + 8]
    ids = np.zeros((8, T_ALG), np.int32)
    msk = np.zeros((8, T_ALG), np.float32)
    for i, (t, _) in enumerate(batch):
        e = tok.encode(t)
        L = min(len(e.ids), T_ALG)
        ids[i, :L] = e.ids[:L]; msk[i, :L] = 1.0
    x = embed[Tensor(ids, dtype=dtypes.int)]
    m = Tensor(msk, dtype=dtypes.float)
    for li, layer in enumerate(host.llama_layers):
        x = layer(x, rc, rs)
        if (li + 1) in DEPTHS:
            p = ((x * m.unsqueeze(-1)).sum(1) /
                 (m.sum(1, keepdim=True) + 1e-6)).realize().numpy()
            for i in range(len(batch)):
                pooled[li + 1].append(p[i])
    print(f"[ceiling] {bi + len(batch)}/{len(items)}", flush=True)

res = {}
rng = np.random.RandomState(0)
perm = rng.permutation(len(items))
tr, te = perm[:150], perm[150:]
for d in DEPTHS:
    X = np.array(pooled[d], np.float64)
    X = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-6)
    W = np.zeros((X.shape[1], 7)); b = np.zeros(7)
    y = labels
    for it in range(400):
        z = X[tr] @ W + b
        z -= z.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        g = p.copy(); g[np.arange(len(tr)), y[tr]] -= 1
        W -= 0.1 * (X[tr].T @ g / len(tr) + 1e-3 * W)
        b -= 0.1 * g.mean(0)
    acc = float((np.argmax(X[te] @ W + b, 1) == y[te]).mean())
    res[d] = acc
    print(f"[ceiling] prefix L0-L{d-1}: test acc {acc:.3f}")

gain = max(res[6], res[8]) - res[4]
if gain >= 0.10:
    verdict = "DEEPER-BETTER — the L0-L3 cut loses held information; the prefix arm opens"
elif abs(max(res.values()) - min(res.values())) <= 0.05:
    verdict = "FLAT — the cut is innocent; the budget is bound at the read"
else:
    verdict = f"MIXED — gain {gain:+.3f} under the +0.10 bar; the cut holds most of it"
print(f"[ceiling] VERDICT: {verdict}")
json.dump(dict(res={str(k): v for k, v in res.items()}, gain=gain,
               verdict=verdict), open(".cache/trunk_ceiling_probe.json", "w"),
          indent=1)
print("[ceiling] banked -> .cache/trunk_ceiling_probe.json")
