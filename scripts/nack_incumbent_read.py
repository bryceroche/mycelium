"""nack_incumbent_read.py — GUT #26 FOLLOW-UP 1 (2026-07-17): THE
NACK-INCUMBENT READ, owed before any repair lane switches voices.

Head-to-head on the 320-item vote-abstain fixture, one grading frame
(unique forced answer == gold; disposal gold-free):
  INCUMBENT      = the composed two-checkpoint stack (gen-14 gate parses,
                   withhold-2-and-solve, suspects->flags, gen-13 specialist
                   retransmits, withhold-2 again). PRIMARY ARM = field_only
                   (fully deployable — the span channel consults gold fspans
                   and reports separately as ceiling). Specialist wears the
                   manifest's ONE-GENERATION waiver (entourage-14 owed).
  CHALLENGER     = sampled retry, T=1.0 K=8, solver-consistent plurality
                   (yesterday's verdict arm, re-derived per-item — seeds
                   identical, byte-reproducible).
  UNION          = items either mechanism recovers (the deployment-relevant
                   composition ceiling for the repair lane).
Per-item outcomes BANKED this time (the continuity audit's per-item law).
"""
import json, os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("ALG2", "1")
os.environ.setdefault("ALG_FTYPES", "6")
os.environ.setdefault("ALG_DUP", "1")
import numpy as np
from collections import Counter

from phase1_algebra_head import T_ALG, L_FAC, N_DIG, build_params, decode, ALG_CKPT
from phase1_algebra_nack import N_FIELDS, build_cond_params, forward_cond
from tta_alg2_dials import solve2
from tier0_incumbent import softmax, sig
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

GATE = ".cache/phase1_gen14_head.safetensors"
NACK = ".cache/phase1_gen13_nack.safetensors"
K_WH, K_SAMP, T_SAMP = 2, 8, 1.0

fixture = json.load(open(".cache/flux_continuity_audit.json"))["repair_intake"]
rows = [json.loads(l) for l in open(".cache/algebra_nl_bigtest.jsonl")]
z = np.load(".cache/phase1_alg_states_bigtest.npz")
states, tokmask, sent, g_fspan = z["states"], z["tokmask"], z["sent"], z["g_fspan"]

p_plat = build_params(0)
sd = safe_load(GATE)
for k in p_plat:
    p_plat[k].assign(sd[k].to(p_plat[k].device).cast(p_plat[k].dtype)).realize()
p_re = build_params(1)
c_re = build_cond_params(1)
sd2 = safe_load(NACK)
for d in (p_re, c_re):
    for k in d:
        d[k].assign(sd2[k].to(d[k].device).cast(d[k].dtype)).realize()
c_zero = build_cond_params(9)
print(f"[incumbent] gate={GATE} | specialist={NACK} (one-gen waiver) | n={len(fixture)}")


def run(model_p, cond_c, idx, ftok, fbit, ffld):
    outs = {}
    for s0 in range(0, len(idx), 8):
        sl = np.array(idx[s0:s0 + 8])
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        f1, f2, f3 = ftok[s0:s0 + len(sl)], fbit[s0:s0 + len(sl)], ffld[s0:s0 + len(sl)]
        if pad:
            f1 = np.concatenate([f1, f1[:1].repeat(pad, 0)])
            f2 = np.concatenate([f2, f2[:1].repeat(pad, 0)])
            f3 = np.concatenate([f3, f3[:1].repeat(pad, 0)])
        out = forward_cond(model_p, cond_c,
                           Tensor(states[sl_p].astype(np.float32), dtype=dtypes.float),
                           Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float),
                           Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int),
                           Tensor(f1, dtype=dtypes.float), Tensor(f2, dtype=dtypes.float),
                           Tensor(f3, dtype=dtypes.float))
        o = {k: out[k].realize().numpy() for k in
             ("pres", "ftype", "op", "islit", "dig", "args", "res", "query")}
        for bi, i in enumerate(sl):
            outs[int(i)] = {k: o[k][bi] for k in o}
    return outs


def slot_conf(o, j):
    cc = float(sig(o["pres"][j]))
    cc *= float(softmax(o["ftype"][j][None])[0].max())
    cc *= float(softmax(o["res"][j][None])[0].max())
    if sig(o["islit"][j]) > 0.5:
        cc *= float(np.mean(softmax(o["dig"][j]).max(-1)))
    else:
        cc *= float(softmax(o["op"][j][None])[0].max())
        cc *= float(np.sort(softmax(o["args"][j][None])[0])[-2:].sum())
    return cc


def forced(facs, q, i):
    return solve2(facs, q, {"n_vars": 24, "m": rows[i].get("m", 60)})


def withheld(o, facs, k_wh):
    confs, fi = [], 0
    for j in range(L_FAC):
        if o["pres"][j] <= 0:
            continue
        confs.append((fi, j, slot_conf(o, j)))
        fi += 1
    order = sorted(confs, key=lambda t: t[2])[:k_wh]
    wh_fi = {x for x, _, _ in order}
    wh_j = [j for _, j, _ in order]
    return [f for x, f in enumerate(facs) if x not in wh_fi], wh_j


gold = {i: rows[i]["solution"][rows[i]["query_var"]] for i in fixture}
per_item = {i: {} for i in fixture}

# ---- INCUMBENT: stage 0 parse, stage 1 withhold, specialist round ----
zf = (np.zeros((len(fixture), T_ALG), np.float32),
      np.zeros((len(fixture), 1), np.float32),
      np.zeros((len(fixture), L_FAC, N_FIELDS), np.float32))
blank = run(p_plat, c_zero, fixture, *zf)
stage0 = stage1 = 0
survivors, surv_wh = [], {}
for i in fixture:
    o = blank[i]
    facs, q = decode(o)
    if forced(facs, q, i) == gold[i]:
        stage0 += 1
        per_item[i]["incumbent"] = "stage0"
        continue
    sub, wh_j = withheld(o, facs, K_WH)
    if forced(sub, q, i) == gold[i]:
        stage1 += 1
        per_item[i]["incumbent"] = "stage1-withhold"
        continue
    survivors.append(i)
    surv_wh[i] = wh_j
print(f"[incumbent] stage0 (straight parse solves-to-gold): {stage0}")
print(f"[incumbent] stage1 (withhold-{K_WH}): +{stage1} | survivors to specialist: {len(survivors)}")

for arm in ("field_only", "both"):
    ftok_s = np.zeros((len(survivors), T_ALG), np.float32)
    ffld_s = np.zeros((len(survivors), L_FAC, N_FIELDS), np.float32)
    for r_, i in enumerate(survivors):
        for j in surv_wh[i]:
            ffld_s[r_, j, :] = 1.0
            if arm == "both":
                ftok_s[r_] = np.maximum(ftok_s[r_], g_fspan[i, j].astype(np.float32))
    fbit_s = np.ones((len(survivors), 1), np.float32)
    re = run(p_re, c_re, survivors, ftok_s, fbit_s, ffld_s)
    rec = 0
    for i in survivors:
        o = re[i]
        facs, q = decode(o)
        ok = forced(facs, q, i) == gold[i]
        if not ok:
            sub, _ = withheld(o, facs, K_WH)
            ok = forced(sub, q, i) == gold[i]
        rec += ok
        if arm == "field_only":
            per_item[i]["incumbent"] = "specialist" if ok else "unrecovered"
    tot = stage0 + stage1 + rec
    dep = " (FULLY DEPLOYABLE)" if arm == "field_only" else " (span=gold — ceiling arm)"
    print(f"[incumbent] ARM={arm}{dep}: specialist +{rec} -> TOTAL {tot}/{len(fixture)} ({tot/len(fixture):.1%})")
    if arm == "field_only":
        incumbent_total = tot

# ---- CHALLENGER: sampled retry T=1.0 K=8 (yesterday's seeds, per-item) ----
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


# sel head not in blank keys above — refetch with sel for sampling parity
blank_s = {}
for s0 in range(0, len(fixture), 8):
    sl = np.array(fixture[s0:s0 + 8])
    pad = 8 - len(sl)
    sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
    from phase1_algebra_head import forward
    out = forward(p_plat, Tensor(states[sl_p].astype(np.float32), dtype=dtypes.float),
                  Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float),
                  Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int))
    keys = [k for k in ("pres", "ftype", "op", "islit", "dig", "args", "res",
                        "query", "sel", "dup") if k in out]
    o = {k: out[k].realize().numpy() for k in keys}
    for bi, i in enumerate(sl):
        blank_s[int(i)] = {k: o[k][bi] for k in o}

chal = 0
for i in fixture:
    answers = []
    for k in range(K_SAMP):
        rng = np.random.RandomState(9000 + i * 100 + k)
        facs, q = sampled_decode(blank_s[i], T_SAMP, rng)
        a = forced(facs, q, i)
        if a is not None:
            answers.append(a)
    ok = bool(answers) and Counter(answers).most_common(1)[0][0] == gold[i]
    chal += ok
    per_item[i]["sampled_T1"] = "recovered" if ok else "unrecovered"

union = sum(1 for i in fixture
            if per_item[i]["incumbent"] != "unrecovered"
            or per_item[i]["sampled_T1"] == "recovered")
only_inc = sum(1 for i in fixture if per_item[i]["incumbent"] != "unrecovered"
               and per_item[i]["sampled_T1"] != "recovered")
only_sam = sum(1 for i in fixture if per_item[i]["incumbent"] == "unrecovered"
               and per_item[i]["sampled_T1"] == "recovered")
n = len(fixture)
print(f"\n=== HEAD-TO-HEAD on the {n} (deployable arms, gold grades never gates) ===")
print(f"  INCUMBENT (stack, field_only): {incumbent_total}/{n} ({incumbent_total/n:.1%})")
print(f"  CHALLENGER (sampled T=1.0):    {chal}/{n} ({chal/n:.1%})")
print(f"  UNION:                         {union}/{n} ({union/n:.1%})  "
      f"[incumbent-only {only_inc} | sampled-only {only_sam}]")
json.dump({"n": n, "incumbent_field_only": incumbent_total, "sampled_T1": chal,
           "union": union, "incumbent_only": only_inc, "sampled_only": only_sam,
           "per_item": {str(i): per_item[i] for i in fixture}},
          open(".cache/nack_incumbent_read.json", "w"))
print("[incumbent] banked -> .cache/nack_incumbent_read.json (per-item, per the law)")
