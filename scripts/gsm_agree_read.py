"""AG part 2, THE AGREEMENT READ (ledger 2026-11-04, word given): what certifies a discovered structure without the key?

Zero training, on two dumps (scripts/gsm_beam_dump.py) of the SAME items from two independently trained BTs (the landscape axis:
the two seeds are two views of each problem). Per seed, per item, three selectors over that seed's finished beam programs:
  BEAM-1  the banked selector: max log p - the soft factors (lam fit on TRAIN)
  SELF    one view's mass vote: group programs by final value, logsumexp(log p) - the soft factors (the placebo: no second view)
  AGREE   the cross-view vote: values present in BOTH seeds' beams, score = mass here + mass there - the soft factors;
          no shared value -> fall back to BEAM-1 (agreement only biases the choice; nothing crosses hard: the bridge rule)
Beside: the certificate tier = items where the two seeds' BEAM-1 answers agree (coverage, precision).
usage: gsm_agree_read.py TAG0 TAG1 -> .cache/gsmb_agree_{TAG0}_{TAG1}.json
"""
import json, math, sys

import numpy as np

T0, T1 = sys.argv[1:3]
D = [json.load(open(f".cache/gsmb_agdump_{t}.json")) for t in (T0, T1)]
key = lambda v: round(v * 1e6)                                           # the near() tolerance's grain


def pen(v, lam):
    return lam[0] * (abs(v - round(v)) > 1e-6) + lam[1] * (v < -1e-9)


def mass(r):
    """value key -> (logsumexp log p, value, right?)"""
    m = {}
    for lp, v, ok in r["beam"]:
        k = key(v); a = m.get(k, (-math.inf, v, ok))[0]
        m[k] = (np.logaddexp(a, lp), v, ok)
    return m


def beam1(r, lam):
    if not r["beam"]: return r["g_ok"], (None if r["g"] is None else key(r["g"]))
    i = int(np.argmax([lp - pen(v, lam) for lp, v, _ in r["beam"]])); return r["beam"][i][2], key(r["beam"][i][1])


def self_vote(r, lam):
    m = mass(r)
    if not m: return r["g_ok"]
    return max(m.values(), key=lambda x: x[0] - pen(x[1], lam))[2]


def agree(r, o, lam):
    m, mo = mass(r), mass(o); sh = [k for k in m if k in mo]
    if not sh: return beam1(r, lam)[0], False
    k = max(sh, key=lambda k: m[k][0] + mo[k][0] - pen(m[k][1], lam)); return m[k][2], True


out = dict(tags=[T0, T1])
for split in ("ORIG", "TEST"):
    A, B = D[0][split], D[1][split]; assert len(A) == len(B)
    o = out[split] = dict(n=len(A))
    b1 = [[beam1(r, D[s]["lam"]) for r in D[s][split]] for s in (0, 1)]
    for s in (0, 1):
        me, ot, lam = D[s][split], D[1 - s][split], D[s]["lam"]
        g = np.array([r["g_ok"] for r in me], float); bm = np.array([x[0] for x in b1[s]], float)
        sv = np.array([self_vote(r, lam) for r in me], float)
        ag_ = [agree(r, q, lam) for r, q in zip(me, ot)]; ag = np.array([x[0] for x in ag_], float)
        d, dp = ag - bm, ag - sv
        o[f"s{s}"] = dict(greedy=g.mean(), beam1=bm.mean(), self=sv.mean(), agree=ag.mean(), shared=float(np.mean([x[1] for x in ag_])),
                          diff=d.mean(), se=d.std(ddof=1) / math.sqrt(len(d)), diff_self=dp.mean(), se_self=dp.std(ddof=1) / math.sqrt(len(dp)))
        x = o[f"s{s}"]
        print(f"[ag] {D[s]['tag']} {split} n {len(me)} ans greedy {x['greedy']:.4f} | BEAM-1 {x['beam1']:.4f} | SELF {x['self']:.4f} | AGREE {x['agree']:.4f} "
              f"(vs BEAM-1 {x['diff']:+.4f} SE {x['se']:.4f}; vs SELF {x['diff_self']:+.4f} SE {x['se_self']:.4f}) | a shared value {x['shared']:.3f}", flush=True)
    cert = np.array([a[1] is not None and a[1] == b[1] for a, b in zip(*b1)])
    right = np.array([a[0] for a in b1[0]], float)
    o["cert"] = dict(coverage=float(cert.mean()), precision=float(right[cert].mean()) if cert.any() else float("nan"),
                     base=float(np.mean([o["s0"]["beam1"], o["s1"]["beam1"]])))
    c = o["cert"]
    print(f"[ag] {split} certificate (the seeds' BEAM-1 answers agree): coverage {c['coverage']:.3f} precision {c['precision']:.4f} "
          f"(BEAM-1 mean {c['base']:.4f}; x{c['precision'] / max(c['base'], 1e-9):.2f})", flush=True)
t = out["TEST"]; d = [t["s0"]["diff"], t["s1"]["diff"]]; dp = [t["s0"]["diff_self"], t["s1"]["diff_self"]]
ag1 = np.mean(d) >= 0.010 and min(d) > 0
agp = np.mean(dp) >= 0.005 and min(dp) > 0
agc = t["cert"]["precision"] >= 3 * t["cert"]["base"]
agc_s = 'VOID (coverage < 0.02)' if t['cert']['coverage'] < 0.02 else ('PASS' if agc else 'FAIL')
V = [f"AG-1: TEST ans AGREE - BEAM-1 {d[0]:+.4f} / {d[1]:+.4f} (mean {np.mean(d):+.4f}; bar +0.010, both > 0) -> {'PASS' if ag1 else 'FAIL'}",
     f"AG-P: TEST ans AGREE - SELF {dp[0]:+.4f} / {dp[1]:+.4f} (mean {np.mean(dp):+.4f}; bar +0.005, both > 0; read only on AG-1 PASS) -> {'PASS' if agp else 'FAIL'}",
     f"AG-C: TEST certificate precision {t['cert']['precision']:.4f} vs 3 x BEAM-1 {3 * t['cert']['base']:.4f} (coverage {t['cert']['coverage']:.3f}; VOID below 0.02) -> {agc_s}"]
print("\n".join(V)); out["verdict"] = V
json.dump(out, open(f".cache/gsmb_agree_{T0}_{T1}.json", "w"), indent=1, default=float)
