"""sampled_retry_probe.py — GUT #26, READ (a): THE SAMPLED-RETRY PROBE
(2026-07-16, fired on Bryce's word). WIDTH, NOT DEPTH: the anatomy killed
four-rounds-deep (same voice, 44->16->4->0); this is one round WIDE — K
distinct utterances sampled from the head's own softmax at T>0, disposal
picks. The parser's determinism is the re-emission mechanism twenty-four
measured; this asks whether the voice was one temperature away all along.

FIVE PINS (ledger, gut #26): (i) gold GRADES, never GATES — disposal is
solver consistency (unique forced answer) + plurality across consistent
samples (the vote wearing sample clothes; view-revote arm deferred,
deviation stated); (ii) population = the 320 banked vote-abstain items
(the continuity audit's fixture); (iii) control = the deterministic
straight parse on the same 320 (view-0 of the banked lattice votes);
(iv) grid T in {0.3, 0.7, 1.0}, K=8, ONE round; (v) bars: deployable
recovery <= control+1pt -> the wall is CONTENT-DEEP (the writer's
charter inherits a measured floor); >= control+5pt -> the cheapest
voice in the universe takes the chair's first shift.

Sampling scope (pinned here, before the run): CONTENT fields sampled
(ftype, op, sel, args, result var, digits); STRUCTURE deterministic
(presence, dup gate, query pointer) — the wall is content, and sampled
presence/query would change the question rather than the answer.
"""
import json, os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("ALG2", "1")
os.environ.setdefault("ALG_FTYPES", "6")
os.environ.setdefault("ALG_DUP", "1")
import numpy as np
from collections import Counter

from phase1_algebra_head import build_params, forward, L_FAC, N_DIG
from tta_alg2_dials import solve2
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

CKPT = ".cache/phase1_gen14_head.safetensors"
K, TEMPS = 8, (0.3, 0.7, 1.0)

fixture = json.load(open(".cache/flux_continuity_audit.json"))["repair_intake"]
rows = [json.loads(l) for l in open(".cache/algebra_nl_bigtest.jsonl")]
gold = [r["solution"][r["query_var"]] for r in rows]
gate_votes = json.load(open(".cache/lattice_gate.json"))["bigtest"]
z = np.load(".cache/phase1_alg_states_bigtest.npz")
states, tokmask, sent = z["states"], z["tokmask"], z["sent"]

p = build_params(0)
sd = safe_load(CKPT)
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

print(f"[probe] gate={CKPT} | fixture n={len(fixture)} | K={K} T={TEMPS}")

# one forward pass over the fixture, batched at 8 (the do_eval pattern)
keys_wanted = ("pres", "ftype", "op", "islit", "dig", "args", "res",
               "query", "sel", "dup")
O = {}
idx = np.array(fixture)
for s0 in range(0, len(idx), 8):
    sl = idx[s0:s0 + 8]
    pad = 8 - len(sl)
    sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
    out = forward(p, Tensor(states[sl_p].astype(np.float32), dtype=dtypes.float),
                  Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float),
                  Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int))
    o = {k: out[k].realize().numpy() for k in keys_wanted if k in out}
    for bi, i in enumerate(sl):
        O[int(i)] = {k: o[k][bi] for k in o}
print(f"[probe] forward pass complete on {len(O)} items")


def softmax_T(logits, T, rng):
    x = np.asarray(logits, np.float64) / T
    x -= x.max()
    e = np.exp(x)
    return e / e.sum()


def samp(logits, T, rng):
    return int(rng.choice(len(logits), p=softmax_T(logits, T, rng)))


def samp2_distinct(logits, T, rng):
    p1 = softmax_T(logits, T, rng)
    a = int(rng.choice(len(logits), p=p1))
    m = p1.copy(); m[a] = 0.0
    b = int(rng.choice(len(logits), p=m / m.sum()))
    return sorted((a, b))


def sampled_decode(o, T, rng):
    from mycelium.csp_domains import ID_TO_SEL
    facs = []
    for j in range(L_FAC):
        if o["pres"][j] <= 0:                      # structure: deterministic
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
            if "dup" in o and o["dup"][j] > 0:     # dup gate: deterministic
                a0 = samp(o["args"][j], T, rng)
                args = [a0, a0]
            else:
                args = samp2_distinct(o["args"][j], T, rng)
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
                         "args": samp2_distinct(o["args"][j], T, rng),
                         "result": res})
    return facs, int(o["query"].argmax())          # query: deterministic


# control: the deterministic straight parse (view 0) on the same items
ctrl = sum(1 for i in fixture if gate_votes[i][0] == gold[i])
print(f"\n[control] deterministic view-0 correct on fixture: "
      f"{ctrl}/{len(fixture)} ({ctrl/len(fixture):.1%})")

results = {}
for T in TEMPS:
    dep = orc = abst = 0
    for i in fixture:
        m = rows[i].get("m", 60)
        q = None
        answers = []
        for k in range(K):
            rng = np.random.RandomState(9000 + i * 100 + k)
            facs, q = sampled_decode(O[i], T, rng)
            a = solve2(facs, q, {"n_vars": 24, "m": m})
            if a is not None:
                answers.append(a)
        if not answers:
            abst += 1
            continue
        pick = Counter(answers).most_common(1)[0][0]
        dep += (pick == gold[i])
        orc += (gold[i] in answers)
    n = len(fixture)
    results[T] = {"deployable": dep, "oracle": orc, "no_consistent": abst}
    print(f"  T={T}: deployable {dep}/{n} ({dep/n:.1%})  "
          f"oracle-any {orc}/{n} ({orc/n:.1%})  no-consistent-sample {abst}")

best = max(r["deployable"] for r in results.values())
d = best - ctrl
verdict = ("THE VOICE TAKES THE SHIFT" if d >= 0.05 * len(fixture) else
           "CONTENT-DEEP — the writer inherits a measured floor"
           if d <= 0.01 * len(fixture) else "BETWEEN BARS — partial signal")
print(f"\n[verdict] best deployable {best} vs control {ctrl} "
      f"(delta {d:+d} on n={len(fixture)}) => {verdict}")
json.dump({"control_view0": ctrl, "n": len(fixture), "K": K,
           "results": {str(t): r for t, r in results.items()},
           "verdict": verdict},
          open(".cache/sampled_retry_probe.json", "w"))
print("[probe] banked -> .cache/sampled_retry_probe.json")
