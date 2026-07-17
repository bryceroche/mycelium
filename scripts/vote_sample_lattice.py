"""vote_sample_lattice.py — GUT #26 FOLLOW-UP 2 (2026-07-17): THE
VOTE-x-SAMPLE LATTICE. Pins (ledger, follow-up-1 entry): permutation
views with the STANDING seeds (40000+10*i+k, view-matched to the banked
lattice artifact) x temperature samples (T=1.0, K=4 per view, 20
candidates/item) on the full 320-item fixture; disposal unchanged —
solver-consistent plurality, gold GRADES never GATES. Reads: (i) does
view-diversity add a third recovery axis beyond the union's 175?
(ii) the decomposition: view-sampled vs the banked per-view
deterministic votes (lattice_gate) and vs the sample-only 136.
"""
import json, os, sys, re, random
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("ALG2", "1")
os.environ.setdefault("ALG_FTYPES", "6")
os.environ.setdefault("ALG_DUP", "1")
import numpy as np
from collections import Counter

from phase1_algebra_head import (T_ALG, L_FAC, N_DIG, build_params, forward,
                                 sent_indices, TOKENIZER_JSON)
from beacon_closing_arm import recompute_states
from tta_alg2_dials import solve2
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

GATE = ".cache/phase1_gen14_head.safetensors"
K_PER_VIEW, T_SAMP, N_VIEWS = 4, 1.0, 5


def permuted_view(text, seed):
    rng = random.Random(seed)
    parts = re.split(r"(?<=\.)\s+", text.strip())
    if len(parts) <= 3:
        return text
    mid = parts[1:-1]
    rng.shuffle(mid)
    return " ".join([parts[0]] + mid + [parts[-1]])


fixture = json.load(open(".cache/flux_continuity_audit.json"))["repair_intake"]
rows = [json.loads(l) for l in open(".cache/algebra_nl_bigtest.jsonl")]
gold = {i: rows[i]["solution"][rows[i]["query_var"]] for i in fixture}
prior = json.load(open(".cache/nack_incumbent_read.json"))["per_item"]
z = np.load(".cache/phase1_alg_states_bigtest.npz")
states0, tokmask0, sent0 = z["states"], z["tokmask"], z["sent"]
tok = Tokenizer.from_file(TOKENIZER_JSON)

p = build_params(0)
sd = safe_load(GATE)
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
print(f"[lattice] gate={GATE} | fixture n={len(fixture)} | views={N_VIEWS} K/view={K_PER_VIEW} T={T_SAMP}")


def head_out(sts, msk, snt, idx):
    outs = {}
    for s0 in range(0, len(idx), 8):
        sl = np.arange(s0, min(s0 + 8, len(idx)))
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        out = forward(p, Tensor(sts[sl_p].astype(np.float32), dtype=dtypes.float),
                      Tensor(msk[sl_p].astype(np.float32), dtype=dtypes.float),
                      Tensor(snt[sl_p].astype(np.int32), dtype=dtypes.int))
        keys = [k for k in ("pres", "ftype", "op", "islit", "dig", "args", "res",
                            "query", "sel", "dup") if k in out]
        o = {k: out[k].realize().numpy() for k in keys}
        for bi, j in enumerate(sl):
            outs[idx[int(j)]] = {k: o[k][bi] for k in o}
    return outs


def softmax_T(logits, T):
    x = np.asarray(logits, np.float64) / T
    x -= x.max()
    e = np.exp(x)
    return e / e.sum()


def samp(logits, T, rng):
    return int(rng.choice(len(logits), p=softmax_T(logits, T)))


def samp2(logits, T, rng):
    p1 = softmax_T(logits, T)
    a = int(rng.choice(len(logits), p=p1))
    m = p1.copy(); m[a] = 0.0
    return sorted((a, int(rng.choice(len(logits), p=m / m.sum()))))


def sampled_decode(o, T, rng):
    from mycelium.csp_domains import ID_TO_SEL
    facs = []
    for j in range(L_FAC):
        if o["pres"][j] <= 0:
            continue
        res = samp(o["res"][j], T, rng)
        ft = samp(o["ftype"][j], T, rng)

        def digval():
            return int(sum(samp(o["dig"][j][d], T, rng) * 10 ** (N_DIG - 1 - d)
                           for d in range(N_DIG)))
        if ft == 1:
            facs.append({"ftype": "given", "var": res, "value": digval()})
        elif ft == 0:
            op = "add" if samp(o["op"][j], T, rng) == 0 else "mul"
            if "dup" in o and o["dup"][j] > 0:
                a0 = samp(o["args"][j], T, rng)
                args = [a0, a0]
            else:
                args = samp2(o["args"][j], T, rng)
            facs.append({"ftype": "rel", "op": op, "args": args, "result": res})
        elif ft == 2:
            facs.append({"ftype": "mod", "var": samp(o["args"][j], T, rng),
                         "k": max(digval(), 2), "result": res})
        elif ft == 4:
            facs.append({"ftype": "pct", "args": [samp(o["args"][j], T, rng), res],
                         "p": max(digval(), 1)})
        elif ft == 5:
            facs.append({"ftype": "fdiv", "var": samp(o["args"][j], T, rng),
                         "k": max(digval(), 2), "result": res})
        else:
            facs.append({"ftype": "sel", "sel": ID_TO_SEL[samp(o["sel"][j], T, rng)],
                         "args": samp2(o["args"][j], T, rng), "result": res})
    return facs, int(o["query"].argmax())


# per-view head outputs: view 0 from banked states; views 1-4 recomputed
view_outs = [head_out(states0[fixture], tokmask0[fixture], sent0[fixture], fixture)]
for v in range(1, N_VIEWS):
    ids = np.zeros((len(fixture), T_ALG), np.int32)
    msk = np.zeros((len(fixture), T_ALG), np.float32)
    snt = np.zeros((len(fixture), T_ALG), np.int32)
    for r_, i in enumerate(fixture):
        t = permuted_view(rows[i]["text"], 40000 + 10 * i + v)
        e = tok.encode(t)
        L = min(len(e.ids), T_ALG)
        ids[r_, :L] = e.ids[:L]
        msk[r_, :L] = 1.0
        snt[r_] = sent_indices(t, list(e.offsets), msk[r_])
    sts = recompute_states(ids)
    view_outs.append(head_out(sts, msk, snt, fixture))
    print(f"[lattice] view {v} states + head outputs done")

lat = plur = 0
per_item = {}
for i in fixture:
    answers = []
    for v in range(N_VIEWS):
        for k in range(K_PER_VIEW):
            rng = np.random.RandomState(70000 + i * 100 + v * 10 + k)
            facs, q = sampled_decode(view_outs[v][i], T_SAMP, rng)
            a = solve2(facs, q, {"n_vars": 24, "m": rows[i].get("m", 60)})
            if a is not None:
                answers.append(a)
    ok = bool(answers) and Counter(answers).most_common(1)[0][0] == gold[i]
    lat += ok
    per_item[i] = "recovered" if ok else "unrecovered"

n = len(fixture)
tri_union = sum(1 for i in fixture
                if prior[str(i)]["incumbent"] != "unrecovered"
                or prior[str(i)]["sampled_T1"] == "recovered"
                or per_item[i] == "recovered")
lat_only = sum(1 for i in fixture
               if per_item[i] == "recovered"
               and prior[str(i)]["incumbent"] == "unrecovered"
               and prior[str(i)]["sampled_T1"] != "recovered")
print(f"\n=== THE VOTE-x-SAMPLE LATTICE (n={n}) ===")
print(f"  lattice deployable (5 views x {K_PER_VIEW} samples, consistent plurality): "
      f"{lat}/{n} ({lat/n:.1%})")
print(f"  vs sample-only 136 | vs incumbent 151 | vs union 175")
print(f"  TRIPLE UNION (incumbent | sampled | lattice): {tri_union}/{n} "
      f"({tri_union/n:.1%})  [lattice-only adds {lat_only}]")
json.dump({"n": n, "lattice": lat, "triple_union": tri_union,
           "lattice_only": lat_only,
           "per_item": {str(i): per_item[i] for i in fixture}},
          open(".cache/vote_sample_lattice.json", "w"))
print("[lattice] banked -> .cache/vote_sample_lattice.json (per-item)")
