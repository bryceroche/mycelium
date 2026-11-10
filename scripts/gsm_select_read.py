"""gsm_select_read.py -- THE SELECT READ (ledger 2026-11-10 pin): keyless selection among Gumbel draws. Pool per problem = the greedy
pass + N draws at tau 1.0 (the SAMPLE read's rng seeds 1000+i, so the draws are the same draws). Each candidate program is judged
WITHOUT the key:
  FCERT  the program executes and its final value is a non-negative integer (the executor's certificate on the answer)
  CERT   the share of its steps whose certificate holds (executed, integer, non-negative) -- the notebook's own cert
  COVER  the share of the problem's text numbers its pointers consume
  LL     its log-likelihood (sum of log commit radii)
SEL (barred) = argmax of (FCERT, CERT, COVER, LL) lexicographically; descriptive: LL only, (FCERT, CERT, LL), (COVER, LL).
The chosen candidate's steps and answer are graded as the scorer grades them. Arm: SR_ARM (+ its flags).
-> .cache/gsm_select_{ARM}.txt + .npz"""
import json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gsm_cj_probe as CP
BTC, BR = CP.BTC, CP.BR
import numpy as np
from tinygrad.nn.state import safe_load

ARM = os.environ["SR_ARM"]; N = int(os.environ.get("SR_N", "16")); K = BR.K
Dt, states = BR.TR.load(); tree = BR.TR.load_tree(); BR.RF = np.load(BR.GD.V1.C(BR.rolef_name()), mmap_mode="r")
seen = {CP.skeleton(it) for it in Dt["TRAIN"]}; its = Dt["TEST"]
t0_, E0 = BR.batch(Dt["TRAIN"][:2], states, tree, np.random.default_rng(0)); P0 = BTC.port_batch(t0_, E0, Dt["TRAIN"][:2], 64)
bt = BTC.BTM.BreathingTransformer(BTC.config(P0["x"].shape[-1], P0["aux"].shape[-1]))
sd = safe_load(f".cache/gsmb_btc_{ARM}.safetensors"); own = bt.state_dict()
assert set(sd) == set(own), f"key mismatch: {sorted(set(sd) ^ set(own))[:8]}"
for k, v in own.items(): v.assign(sd[k].to(v.device)).realize()
atoms = BTC.Atoms(BTC.NA, BTC.DB, BR.NCAP); atoms.C = np.load(f".cache/gsmb_btc_{ARM}_atoms.npy")
m = [min(len(it["prog"]), K) for it in its]; off = np.concatenate([[0], np.cumsum(m)])
sn = np.concatenate([[CP.skeleton(it) in seen] * mm for it, mm in zip(its, m)]).astype(bool)
banked = json.load(open(f".cache/gsmb_btc_{ARM}.json"))["TEST_step_all"]


def feats(prog, it):
    vals = it["vals"] + [None] * K
    nums = {i for i, v in enumerate(it["vals"]) if v is not None}
    used = {i for _, a, b in prog for kd, i in (a, b) if kd == "t"} & nums
    out = BR.TR.run_steps(prog, vals) if prog else []
    cs = [BR.TR.cert(v) for v in out]
    fin = cs[-1] if cs else (0.0, 0.0, 0.0)
    return (float(all(fin)), float(np.mean([all(c) for c in cs])) if cs else 0.0, len(used) / max(len(nums), 1))


def run(noise):
    t = time.time(); r = BTC.score(bt, atoms, its, states, tree, noise=noise); assert len(r["step_all"]) == sum(m)
    print(f"[sel] {ARM} {noise and noise[0]} step {np.mean(r['step_all']):.4f} ({time.time() - t:.0f}s)", flush=True)
    return r


rs = [run(None)] + [run((1.0, np.random.default_rng(1000 + i))) for i in range(N)]
S = np.stack([np.asarray(r["step_all"], bool) for r in rs]); A = np.stack([np.asarray(r["ans"], bool) for r in rs])
F = np.array([[feats(r["progp"][j], its[j]) for j in range(len(its))] for r in rs])            # (N+1, items, 3)
LL = np.array([r["ll"] for r in rs])                                                           # (N+1, items)
RULES = {"SEL": (0, 1, 2, "ll"), "LL": ("ll",), "CERT+LL": (0, 1, "ll"), "COVER+LL": (2, "ll")}
out = [f"{ARM} GREEDY TEST step {S[0].mean():.4f} (banked {banked:.4f} {'VALID' if abs(S[0].mean() - banked) <= 0.003 else 'VOID'}) "
       f"TEST-U {S[0][~sn].mean():.4f} ans {A[0].mean():.4f} | draws mean step {S[1:].mean():.4f} | BEST@{N + 1} step "
       f"{S.any(0).mean():.4f} ans {A.any(0).mean():.4f}",
       f"  greedy feats: FCERT {F[0, :, 0].mean():.3f} CERT {F[0, :, 1].mean():.3f} COVER {F[0, :, 2].mean():.3f} | draws: FCERT "
       f"{F[1:, :, 0].mean():.3f} CERT {F[1:, :, 1].mean():.3f} COVER {F[1:, :, 2].mean():.3f}"]
sv = dict(S=S, A=A, F=F, LL=LL, seen=sn)
for name, keys in RULES.items():
    pick = np.zeros(len(its), int)
    for j in range(len(its)):
        cand = [tuple(F[c, j, q] if q != "ll" else LL[c, j] for q in keys) for c in range(N + 1)]
        pick[j] = max(range(N + 1), key=lambda c: (cand[c], -c))                               # ties: the earliest (greedy first)
    st = np.concatenate([S[pick[j], off[j]:off[j + 1]] for j in range(len(its))]); an = A[pick, np.arange(len(its))]
    sv[f"pick_{name}"] = pick
    out.append(f"  {name}: step {st.mean():.4f} ({st.mean() - S[0].mean():+.4f}) TEST-U {st[~sn].mean():.4f} "
               f"({st[~sn].mean() - S[0][~sn].mean():+.4f}) ans {an.mean():.4f} ({an.mean() - A[0].mean():+.4f}) | greedy kept {np.mean(pick == 0):.3f}")
np.savez(f".cache/gsm_select_{ARM}.npz", **sv)
if os.environ.get("SR_PROGS") == "1":   # (11-10 order audit) every candidate's program, for the slot-order audit
    json.dump([[r["progp"][j] for j in range(len(its))] for r in rs], open(f".cache/gsm_select_{ARM}_progs.json", "w"))
txt = "\n".join(out); print(txt, flush=True); open(f".cache/gsm_select_{ARM}.txt", "w").write(txt + "\n")
