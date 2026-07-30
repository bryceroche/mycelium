"""diet_after_read.py — the fdiv-on-derived AFTER read (2026-07-30).

The EXACT before-fixture instrument (diet_prep.py section A, 2026-07-29):
same probe generator, same RandomState(4000+depth), same view seeds
5000+20*made+k — apples to apples against the banked before
(.cache/diet_prep.json: d0 7/8 basin 0.00, d1 1/8 basin 0.53,
d2 8/8 basin 0.00). Ckpt via DIET_CKPT (default g22). The attractor is
measured, not the pass rate: quorum AND sel-basin view-fraction both
print, per the registered bar."""
import sys, os, json
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_HW", "512"); os.environ.setdefault("ALG_DUP", "1")
import numpy as np
from collections import Counter
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_views import permuted_view
from tta_alg2_dials import solve2
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

CKPT = os.environ.get("DIET_CKPT", ".cache/g22.safetensors")
tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
sd = safe_load(CKPT)
assert set(sd.keys()) == set(p.keys())
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

def make_row(rng, depth):
    K = int(rng.choice([2, 3, 4, 5, 6, 7]))
    if depth == 0:
        V = K * int(rng.randint(2, 40))
        if V > 300: return None
        text = (f"Consider the numbers a, b. a is {V}. "
                f"When a is divided by {K}, the quotient is b. What is b?")
        facs = [{"ftype": "given", "var": 0, "value": V},
                {"ftype": "fdiv", "var": 0, "k": K, "result": 1}]
        return text, facs, 1, V // K
    if depth == 1:
        A = int(rng.randint(2, 12)); B = int(rng.randint(2, 12))
        C = A * B
        if C % K or C > 300: return None
        text = (f"Consider the numbers a, b, c, d. a is {A}. b is {B}. "
                f"a times b equals c. When c is divided by {K}, the quotient is d. What is d?")
        facs = [{"ftype": "given", "var": 0, "value": A}, {"ftype": "given", "var": 1, "value": B},
                {"ftype": "rel", "op": "mul", "args": [0, 1], "result": 2},
                {"ftype": "fdiv", "var": 2, "k": K, "result": 3}]
        return text, facs, 3, C // K
    A = int(rng.randint(2, 12)); B = int(rng.randint(2, 12)); E = int(rng.randint(2, 60))
    C = A * B; F = C + E
    if F % K or F > 300: return None
    text = (f"Consider the numbers a, b, c, d, e, f. a is {A}. b is {B}. "
            f"a times b equals c. d is {E}. c plus d equals e. "
            f"When e is divided by {K}, the quotient is f. What is f?")
    facs = [{"ftype": "given", "var": 0, "value": A}, {"ftype": "given", "var": 1, "value": B},
            {"ftype": "rel", "op": "mul", "args": [0, 1], "result": 2},
            {"ftype": "given", "var": 3, "value": E},
            {"ftype": "rel", "op": "add", "args": [2, 3], "result": 4},
            {"ftype": "fdiv", "var": 4, "k": K, "result": 5}]
    return text, facs, 5, F // K

def parse_batch(texts):
    n = len(texts); N = ((n+7)//8)*8
    ids = np.zeros((N, T_ALG), np.int32); msk = np.zeros((N, T_ALG), np.float32); snt = np.zeros((N, T_ALG), np.int32)
    for i, t in enumerate(texts):
        e = tok.encode(t); L = min(len(e.ids), T_ALG)
        ids[i, :L] = e.ids[:L]; msk[i, :L] = 1.0
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

after = {}
for depth in (0, 1, 2):
    rng = np.random.RandomState(4000 + depth)
    n_ok = n_basin = made = tries = 0
    while made < 8 and tries < 200:
        tries += 1
        pr = make_row(rng, depth)
        if pr is None: continue
        text, facs, q, gold = pr
        assert solve2(facs, q, {"n_vars": 24, "m": 300}) == gold
        texts = [text] + [permuted_view(text, 5000 + 20*made + k) for k in range(1, 5)]
        views = [(f, qq, solve2(f, qq, {"n_vars": 24, "m": 300})) for f, qq in parse_batch(texts)]
        nn = [a for _, _, a in views if a is not None]
        c = Counter(nn).most_common(1); plur, cnt = c[0] if c else (None, 0)
        n_ok += (cnt >= 3 and plur == gold)
        n_basin += sum(1 for f, _, _ in views if any(fa.get("ftype") == "sel" for fa in f)) / 5
        made += 1
    after[depth] = {"quorum_pass": f"{n_ok}/8", "mean_sel_basin_views": round(n_basin/8, 2)}
    print(f"[AFTER depth-{depth}] quorum {n_ok}/8   sel-basin view-fraction {n_basin/8:.2f}")
before = json.load(open(".cache/diet_prep.json"))["before_fixture"]
print(f"[before, banked] " + "  ".join(f"d{d}: {before[str(d)]['quorum_pass']} basin {before[str(d)]['mean_sel_basin_views']}" for d in (0, 1, 2)))
json.dump({"ckpt": CKPT, "after_fixture": after, "before_fixture": before},
          open(".cache/diet_after_read.json", "w"), indent=1)
print("[saved] .cache/diet_after_read.json")
