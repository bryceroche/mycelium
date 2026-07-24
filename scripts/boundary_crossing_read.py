"""boundary_crossing_read.py — gut #62's read (2026-07-24, fired on the
word). Do the stagnant items MOVE at restart boundaries, or sit frozen
through every scatter?

Population: the dark zone (no view finds gold, from the gen-18 armA
lattice) + the 65 residue. Control: a size-matched sample of stable
majority-correct items. Checkpoints: arm A's boundary pairs
(segN_s4000 -> segN+1_s500, N=1..3) vs within-segment pairs
(segN_s2000 -> segN_s2500). Metric: answer-change rate per pair.
Bars (pinned at #62): stagnant boundary-move >= 2x stagnant
within-move -> TARGETED SCATTER opens; else SCATTER-PROOF.
"""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from collections import Counter
os.environ.setdefault("ALG2", "1"); os.environ["ALG_FTYPES"] = "8"
os.environ.setdefault("ALG_DUP", "1")
from phase1_algebra_head import build_params, forward, decode
from tta_alg2_dials import solve2
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

z = np.load(".cache/phase1_alg_states_bigtest.npz")
st, tk, se = z["states"], z["tokmask"], z["sent"]
rows = [json.loads(l) for l in open(".cache/algebra_nl_bigtest.jsonl")]
gold = [r["solution"][r["query_var"]] for r in rows]
votes = json.load(open(".cache/lattice_gen18_A.json"))["bigtest"]
resid = set(json.load(open(".cache/residue_portrait.json"))["residue_items"])


def maj(v):
    vs = [x for x in v if x is not None]
    return Counter(vs).most_common(1)[0] if vs else (None, 0)


dark, correct_pool = [], []
for i in range(1500):
    vs = [x for x in votes[i] if x is not None]
    t_, c_ = maj(votes[i])
    if gold[i] not in vs:
        dark.append(i)
    elif c_ >= 3 and t_ == gold[i]:
        correct_pool.append(i)
stag = sorted(set(dark) | resid)
rng = np.random.RandomState(62)
ctrl = sorted(rng.choice(correct_pool, size=len(stag), replace=False))
print(f"[boundary] stagnant {len(stag)} (dark {len(dark)} + residue {len(resid)},"
      f" overlap {len(set(dark) & resid)}) | control {len(ctrl)}")

PAIRS = {
    "boundary": [("seg1_s4000", "seg2_s500"), ("seg2_s4000", "seg3_s500"),
                 ("seg3_s4000", "seg4_s500")],
    "within":   [("seg1_s2000", "seg1_s2500"), ("seg2_s2000", "seg2_s2500"),
                 ("seg3_s2000", "seg3_s2500")],
}
CKPTS = sorted({c for ps in PAIRS.values() for p in ps for c in p})
items = stag + list(ctrl)


def answers(ckpt_tag):
    p = build_params(0)
    sd = safe_load(f".cache/g18_armA_{ckpt_tag}.safetensors")
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    ans = {}
    arr = np.array(items)
    for s0 in range(0, len(arr), 8):
        sl = arr[s0:s0 + 8]
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        out = forward(p, Tensor(st[sl_p].astype(np.float32), dtype=dtypes.float),
                      Tensor(tk[sl_p].astype(np.float32), dtype=dtypes.float),
                      Tensor(se[sl_p].astype(np.int32), dtype=dtypes.int))
        keys = [k for k in ("pres", "ftype", "op", "islit", "dig", "dig2", "args",
                            "res", "query", "sel", "dup", "y") if k in out]
        o = {k: out[k].realize().numpy() for k in keys}
        for bi, ri in enumerate(sl):
            if bi < len(sl):
                facs, q = decode({k: o[k][bi] for k in o})
                ans[int(ri)] = solve2(facs, q, {"n_vars": 24, "m": rows[ri]["m"]})
    return ans


A = {c: answers(c) for c in CKPTS}
res = {}
for pop_name, pop in (("stagnant", stag), ("control", list(ctrl))):
    res[pop_name] = {}
    for kind, prs in PAIRS.items():
        moves = tot = 0
        for a, b in prs:
            for i in pop:
                tot += 1
                if A[a][i] != A[b][i]:
                    moves += 1
        res[pop_name][kind] = moves / max(tot, 1)
        print(f"[boundary] {pop_name} {kind}: move-rate {moves}/{tot} = {moves/max(tot,1):.3f}")

sb, sw = res["stagnant"]["boundary"], res["stagnant"]["within"]
lights = sb >= 2 * sw and sb > 0.02
verdict = ("MOVES AT BOUNDARIES — the targeted-scatter registration OPENS"
           if lights else
           "SCATTER-PROOF — stagnation survives controlled explosion; the walls close their last question")
print(f"[boundary] stagnant boundary {sb:.3f} vs within {sw:.3f} (bar: >=2x and >0.02)")
print(f"[boundary] VERDICT: {verdict}")
json.dump(dict(res=res, stag_n=len(stag), verdict=verdict, lights=bool(lights)),
          open(".cache/boundary_crossing_read.json", "w"), indent=1)
print("[boundary] banked -> .cache/boundary_crossing_read.json")
