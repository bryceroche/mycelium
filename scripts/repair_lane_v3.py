"""repair_lane_v3.py — THE THREE-VOICE REPAIR LANE (integration,
2026-07-20). The deployable organ carrying the adopted + drawn tiers:

  vote-abstain item -> vote-x-sample lattice (5 views x 4 samples, T=1.0,
  standing seeds) -> solver-consistent candidates -> plurality count c:
    TIER 1 (adopted 2026-07-17): c >= 8            -> EMIT plurality
    TIER 2 (drawn 2026-07-20):   5 <= c <= 7 AND the fingerpost CONFIRMS
            (trunk prefers the plurality's canonical restatement over the
            runner-up's) -> EMIT plurality
    else -> ABSTAIN
  Gold-free throughout: solver consistency + plurality + trunk paraphrase
  preference. The key grades; it never gates. Certification untouched
  (the lane consumes only vote-abstain items).

VALIDATION MODE (--validate, the machine-first law's requirement): run
the full fixture, assert byte-reproduction of the banked tier numbers
(tier1 36/36; tier2 31/31), print the adoption re-statement or refuse.
Standalone by design: a production organ does not re-run research
scripts on import.
"""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from collections import Counter
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "6")
os.environ.setdefault("ALG_DUP", "1")
from phase1_algebra_head import (T_ALG, L_FAC, N_DIG, build_params, forward,
                                 sent_indices, TOKENIZER_JSON)
from beacon_closing_arm import recompute_states
from tta_alg2_dials import solve2
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
import re, random

T_SAMP, K_PER_VIEW, N_VIEWS = 1.0, 4, 5
TIER1_MIN, TIER2_MIN, TIER2_MAX = 8, 5, 7
LET = "abcdefghijklmnopqrstuvwx"

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
CKPT = os.environ.get("GATE_CKPT", ".cache/phase1_gen14_head.safetensors")
sd = safe_load(CKPT)
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()


def permuted_view(text, seed):
    rng = random.Random(seed)
    parts = re.split(r"(?<=\.)\s+", text.strip())
    if len(parts) <= 3:
        return text
    mid = parts[1:-1]
    rng.shuffle(mid)
    return " ".join([parts[0]] + mid + [parts[-1]])


def head_out_batch(sts, msk, snt):
    N = ((len(sts) + 7) // 8) * 8
    pad = N - len(sts)
    if pad:
        sts = np.concatenate([sts, sts[:1].repeat(pad, 0)])
        msk = np.concatenate([msk, msk[:1].repeat(pad, 0)])
        snt = np.concatenate([snt, snt[:1].repeat(pad, 0)])
    outs = []
    for s0 in range(0, N, 8):
        out = forward(p, Tensor(sts[s0:s0+8].astype(np.float32), dtype=dtypes.float),
                      Tensor(msk[s0:s0+8].astype(np.float32), dtype=dtypes.float),
                      Tensor(snt[s0:s0+8].astype(np.int32), dtype=dtypes.int))
        keys = [k for k in ("pres", "ftype", "op", "islit", "dig", "args", "res",
                            "query", "sel", "dup") if k in out]
        o = {k: out[k].realize().numpy() for k in keys}
        for bi in range(8):
            outs.append({k: o[k][bi] for k in o})
    return outs[:len(outs) - pad if pad else None]


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


def render(facs, q):
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


def pooled_batch(texts):
    ids = np.zeros((len(texts), T_ALG), np.int32)
    msk = np.zeros((len(texts), T_ALG), np.float32)
    for i, t in enumerate(texts):
        e = tok.encode(t)
        Ln = min(len(e.ids), T_ALG)
        ids[i, :Ln] = e.ids[:Ln]; msk[i, :Ln] = 1.0
    sts = recompute_states(ids)
    V = (sts.astype(np.float32) * msk[:, :, None]).sum(1) / \
        np.maximum(msk.sum(1)[:, None], 1)
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def lane(item_idx, text, m, orig_pooled, view_outs_i, seed_base):
    """The three-voice lane on one vote-abstain item. Returns
    (decision, answer|None, tier)."""
    cands = {}
    for v in range(N_VIEWS):
        for k in range(K_PER_VIEW):
            rng = np.random.RandomState(seed_base + v * 10 + k)
            facs, q = sampled_decode(view_outs_i[v], T_SAMP, rng)
            a = solve2(facs, q, {"n_vars": 24, "m": m})
            if a is not None:
                cands.setdefault(a, {"count": 0, "parse": (facs, q)})
                cands[a]["count"] += 1
    if not cands:
        return "abstain", None, None
    top = sorted(cands.items(), key=lambda kv: -kv[1]["count"])[:2]
    a1, c1 = top[0]
    if c1["count"] >= TIER1_MIN:
        return "emit", a1, 1
    if TIER2_MIN <= c1["count"] <= TIER2_MAX and len(top) == 2:
        a2, c2 = top[1]
        V = pooled_batch([render(*c1["parse"]), render(*c2["parse"])])
        s1, s2 = float(V[0] @ orig_pooled), float(V[1] @ orig_pooled)
        if s1 > s2:                       # the fingerpost CONFIRMS the plurality
            return "emit", a1, 2
    return "abstain", None, None


def validate():
    rows = [json.loads(l) for l in open(".cache/algebra_nl_bigtest.jsonl")]
    gold = [r["solution"][r["query_var"]] for r in rows]
    fixture = json.load(open(".cache/flux_continuity_audit.json"))["repair_intake"]
    z = np.load(".cache/phase1_alg_states_bigtest.npz")
    states0, tokmask0, sent0 = z["states"], z["tokmask"], z["sent"]

    # view head-outputs (view 0 banked states; 1-4 recomputed) — standing seeds
    view_outs = [dict()] * 0
    vo = [None] * N_VIEWS
    vo[0] = {i: o for i, o in zip(fixture, head_out_batch(
        states0[fixture], tokmask0[fixture], sent0[fixture]))}
    for v in range(1, N_VIEWS):
        ids = np.zeros((len(fixture), T_ALG), np.int32)
        msk = np.zeros((len(fixture), T_ALG), np.float32)
        snt = np.zeros((len(fixture), T_ALG), np.int32)
        for r_, i in enumerate(fixture):
            t = permuted_view(rows[i]["text"], 40000 + 10 * i + v)
            e = tok.encode(t)
            Ln = min(len(e.ids), T_ALG)
            ids[r_, :Ln] = e.ids[:Ln]
            msk[r_, :Ln] = 1.0
            snt[r_] = sent_indices(t, list(e.offsets), msk[r_])
        sts = recompute_states(ids)
        vo[v] = {i: o for i, o in zip(fixture, head_out_batch(sts, msk, snt))}
        print(f"[lane] view {v} ready")

    t1e = t1r = t2e = t2r = abst = 0
    for i in fixture:
        v0 = (states0[i].astype(np.float32) * tokmask0[i][:, None]).sum(0) / \
            max(tokmask0[i].sum(), 1)
        v0 /= np.linalg.norm(v0)
        dec, ans, tier = lane(i, rows[i]["text"], rows[i].get("m", 60), v0,
                              [vo[v][i] for v in range(N_VIEWS)], 70000 + i * 100)
        if dec == "abstain":
            abst += 1
        elif tier == 1:
            t1e += 1; t1r += (ans == gold[i])
        else:
            t2e += 1; t2r += (ans == gold[i])
    print(f"\n=== THREE-VOICE LANE VALIDATION (n={len(fixture)}) ===")
    print(f"  tier 1 (count>=8):            emit {t1e}  right {t1r}")
    print(f"  tier 2 (5-7 + fingerpost):    emit {t2e}  right {t2r}")
    print(f"  abstain:                      {abst}")
    comp_r, comp_e = 1179 + t1r + t2r, 1180 + t1e + t2e
    print(f"  composite: {comp_r}/{comp_e} = {comp_r/comp_e:.5f} "
          f"(bar {1179/1180:.5f}) -> {'PASS' if comp_r/comp_e >= 1179/1180 else 'FAIL'}")
    ok = (t1e, t1r) == (36, 36) and (t2e, t2r) == (31, 31)
    print(f"  banked-number reproduction: "
          f"{'EXACT — THE LANE IS THE MACHINE THE LEDGER DESCRIBES' if ok else 'MISMATCH — DO NOT ADOPT'}")
    json.dump({"tier1": [t1e, t1r], "tier2": [t2e, t2r], "abstain": abst,
               "composite": [comp_r, comp_e], "reproduced": bool(ok)},
              open(".cache/repair_lane_v3_validation.json", "w"))
    return ok


if __name__ == "__main__":
    if "--validate" in sys.argv:
        sys.exit(0 if validate() else 1)
    print("repair_lane_v3: import `lane()` for deployment; --validate to "
          "reproduce the banked fixture numbers")
