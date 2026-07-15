"""kv_audit.py — gut #18: pointer-entropy vs graph size + two-way ballast."""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
os.environ.setdefault("ALG2","1"); os.environ.setdefault("ALG_FTYPES","6")
from phase1_algebra_head import STATES_NPZ, STATES_NPY, T_ALG, TOKENIZER_JSON, build_params, forward, sent_indices
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

def load_head(ckpt, dup):
    os.environ["ALG_DUP"] = dup
    p = build_params(0)
    sd = safe_load(ckpt)
    assert set(sd.keys()) == set(p.keys())
    for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    return p

rows = [json.loads(l) for l in open(".cache/dag8_test.jsonl")]
z = np.load(STATES_NPZ.format(split="dag8test"))
st = np.load(STATES_NPY.format(split="dag8test"), mmap_mode="r")
tk = z["tokmask"].astype(np.float32); se = z["sent"].astype(np.int32)

def entropy_curve(p, name):
    bins = {}
    for s0 in range(0, 400, 8):
        out = forward(p, Tensor(np.asarray(st[s0:s0+8], np.float32), dtype=dtypes.float),
                      Tensor(tk[s0:s0+8], dtype=dtypes.float),
                      Tensor(se[s0:s0+8], dtype=dtypes.int))
        args = out["args"].realize().numpy()
        pres = out["pres"].realize().numpy()
        for bi in range(8):
            r = rows[s0+bi]; nv = r["n_vars"]
            for j in range(24):
                if pres[bi, j] > 0:
                    x = args[bi, j] - args[bi, j].max()
                    pr = np.exp(x); pr /= pr.sum()
                    H = float(-(pr * np.log(pr + 1e-12)).sum())
                    bins.setdefault(min(nv // 4 * 4, 20), []).append(H)
    print(f"  {name}: " + " ".join(f"nv{k}-{k+3}:{np.mean(v):.3f}" for k, v in sorted(bins.items())))

print("=== READ 2: args-softmax entropy vs graph size ===")
g13 = load_head(".cache/phase1_gen13_head.safetensors", "1")
entropy_curve(g13, "gen-13")
g9b = load_head(".cache/phase1_gen9b_head.safetensors", "1")
entropy_curve(g9b, "gen-9b")

print("=== TWO-WAY BALLAST SPLIT ([90]'s base, under gen-13) ===")
tok = Tokenizer.from_file(TOKENIZER_JSON)
texts = {
 "bare":  "Consider the numbers a, b. a times a equals b. It is known that b is 49. What is a?",
 "var-ballast": "Consider the numbers a, b, c, d, e. a times a equals b. It is known that b is 49. c is 3. d is 5. c plus d equals e. What is a?",
 "text-ballast": "Consider the numbers a, b. a times a equals b. It is known that b is 49. Work carefully through each fact. Numbers can be written in many ways. What is a?",
}
os.environ["ALG_DUP"] = "1"
for name, t in texts.items():
    ids = np.zeros((8, T_ALG), np.int32); msk = np.zeros((8, T_ALG), np.float32); snt = np.zeros((8, T_ALG), np.int32)
    e = tok.encode(t); L = len(e.ids)
    ids[0,:L] = e.ids; msk[0,:L] = 1.0; snt[0] = sent_indices(t, list(e.offsets), msk[0])
    s = recompute_states(ids)
    out = forward(g13, Tensor(s.astype(np.float32), dtype=dtypes.float),
                  Tensor(msk, dtype=dtypes.float), Tensor(snt, dtype=dtypes.int))
    args = out["args"].realize().numpy()[0]; pres = out["pres"].realize().numpy()[0]
    Hs = []
    for j in range(24):
        if pres[j] > 0:
            x = args[j] - args[j].max(); pr = np.exp(x); pr /= pr.sum()
            Hs.append(float(-(pr*np.log(pr+1e-12)).sum()))
    print(f"  {name:12s}: mean args-entropy {np.mean(Hs):.3f} over {len(Hs)} factors")
