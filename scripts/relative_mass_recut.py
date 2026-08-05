"""relative_mass_recut.py — THE RELATIVE-MASS RE-CUT (2026-08-05; bars
pinned in ledger BEFORE this ran: AUC >=0.70 primary / >=0.60 weak on
the wild 124; size-artifact strata control; one-signal scope).

The raking screen's (#149) prospective call: a DIFFERENCE separates at
the frontier where the LEVEL saturated (abs min-mass p50 0.9991 in
register, AUC 0.650 wild). Contrast forms DECLARED HERE, before any
number prints — the two pinned arms per the gut's own wording:
  C1 (row-normalized, PRIMARY): min_mass - mean_mass over present slots
  C3 (rival-relative, PRIMARY): answer-factor mass - max other present
      slot mass (answer factor = rel whose result == query var;
      coverage reported where undefined)
  exploratory (reported, not confirmatory): C2 = min - median,
      C4 = -(max - min) (spread, sign-aligned so higher = more correct)
Size control: Spearman(contrast, n_present) + per-stratum AUC by
n_present. Instrument check: absolute min-mass AUC must reproduce the
banked 0.6504. Per-slot masses banked (the wild artifact's min-only
mass was the aggregates offense at the slot grain — fixed here)."""
import os, sys, json
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_HW", "512"); os.environ.setdefault("ALG_DUP", "1")
os.environ.setdefault("ALG_WIDE", "1")
os.environ["ALG_BREATH"] = "3"; os.environ["ALG_RINGS"] = "1"
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON, build_slot_masks
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
from scipy.stats import mannwhitneyu, spearmanr
tok = Tokenizer.from_file(TOKENIZER_JSON)

def load(ck):
    p = build_params(0); sd = safe_load(ck)
    for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    return p

def slot_masses(p, text):
    ids = np.zeros((8, T_ALG), np.int32); msk = np.zeros((8, T_ALG), np.float32); snt = np.zeros((8, T_ALG), np.int32)
    e = tok.encode(text); Ln = min(len(e.ids), T_ALG)
    ids[0, :Ln] = e.ids[:Ln]; msk[0, :Ln] = 1.0
    snt[0] = sent_indices(text, list(e.offsets), msk[0])
    st = recompute_states(ids)
    o0 = forward(p, Tensor(st.astype(np.float32), dtype=dtypes.float),
                 Tensor(msk, dtype=dtypes.float), Tensor(snt, dtype=dtypes.int))
    onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
    mk = build_slot_masks(onp0, snt)
    o = forward(p, Tensor(st.astype(np.float32), dtype=dtypes.float),
                Tensor(msk, dtype=dtypes.float), Tensor(snt, dtype=dtypes.int),
                slot_mask=Tensor(mk, dtype=dtypes.float))
    keys = ["pres", "ftype", "op", "islit", "dig", "sgn", "args", "res", "query", "cmt_m"]
    if "sel" in o: keys.append("sel")
    if "dup" in o: keys.append("dup")
    onp = {k: o[k].realize().numpy() for k in keys}
    d = decode({k: onp[k][0] for k in onp if k != "cmt_m"})
    pres = onp["pres"][0]; cm = onp["cmt_m"][0]
    slots = [(j, float(cm[j])) for j in range(24) if float(pres[j]) > 0]
    return d, slots

def auc(x, y):
    x = np.array(x, np.float64); y = np.array(y, bool)
    a, b = x[y], x[~y]
    if not len(a) or not len(b): return float("nan"), 1.0
    u, pv = mannwhitneyu(a, b, alternative="greater")
    return u / (len(a) * len(b)), pv

recs = [json.loads(l) for l in open(".cache/wild_ledger_v1.jsonl")]
h = [json.loads(l) for l in open(".cache/math_harvest_v0.jsonl")]
wild = [(h[r["harvest_idx"]]["problem"], bool(r["correct"])) for r in recs if r["tier"] == "answered"]
print(f"[recut] wild n={len(wild)} ({sum(1 for _,ok in wild if ok)} correct)", flush=True)

rp = load(".cache/g24_rings_rings.safetensors")
rows = []
for j, (t, ok) in enumerate(wild):
    (f, q), slots = slot_masses(rp, t)
    ms = [m for _, m in slots]
    ans_slot = [i for i, fa in enumerate(f) if fa.get("ftype") == "rel" and fa.get("result") == q]
    c3 = None
    if len(ans_slot) == 1 and len(ms) >= 2:
        aj = ans_slot[0]
        pos = [k for k, (sj, _) in enumerate(slots) if sj == aj]
        if pos:
            am = slots[pos[0]][1]
            riv = [m for k, (sj, m) in enumerate(slots) if sj != aj]
            c3 = am - max(riv)
    rows.append({"correct": ok, "n": len(ms), "slots": slots,
                 "minm": min(ms) if ms else 0.0,
                 "c1": (min(ms) - float(np.mean(ms))) if ms else None,
                 "c2": (min(ms) - float(np.median(ms))) if ms else None,
                 "c3": c3,
                 "c4": -(max(ms) - min(ms)) if ms else None})
    if (j + 1) % 25 == 0: print(f"[recut] {j+1}/{len(wild)}", flush=True)

lab = [r["correct"] for r in rows]
a_abs, p_abs = auc([r["minm"] for r in rows], lab)
print(f"[check] absolute min-mass AUC={a_abs:.4f} (banked 0.6504) p={p_abs:.3g}", flush=True)

for name, key, tag in [("C1 row-normalized (PINNED)", "c1", "PINNED"),
                       ("C3 rival-relative (PINNED)", "c3", "PINNED"),
                       ("C2 min-median (exploratory)", "c2", "expl"),
                       ("C4 -spread (exploratory)", "c4", "expl")]:
    sub = [(r[key], r["correct"], r["n"]) for r in rows if r[key] is not None]
    if not sub:
        print(f"[{name}] NO COVERAGE", flush=True); continue
    xs, ys, ns = zip(*sub)
    a, pv = auc(xs, ys)
    rho, prho = spearmanr(xs, ns)
    print(f"[{name}] AUC={a:.4f} p={pv:.3g} n={len(xs)} | size-corr rho={rho:+.3f} p={prho:.3g}", flush=True)
    strata = sorted(set(ns))
    parts = []
    for s in strata:
        sx = [x for x, y, n in sub if n == s]; sy = [y for x, y, n in sub if n == s]
        if len(set(sy)) == 2:
            sa, _ = auc(sx, sy)
            parts.append(f"n={s}:{sa:.3f}({len(sx)})")
        else:
            parts.append(f"n={s}:—({len(sx)})")
    print(f"    strata: {' '.join(parts)}", flush=True)

a1 = auc([r["c1"] for r in rows if r["c1"] is not None], [r["correct"] for r in rows if r["c1"] is not None])[0]
sub3 = [(r["c3"], r["correct"]) for r in rows if r["c3"] is not None]
a3 = auc([x for x, _ in sub3], [y for _, y in sub3])[0] if sub3 else float("nan")
best = np.nanmax([a1, a3])
verdict = ("PASS-PRIMARY" if best >= 0.70 else "PASS-WEAK" if best >= 0.60 else "FAIL")
print(f"VERDICT (pinned bars 0.70/0.60 on pinned arms only): {verdict} (best pinned {best:.4f})", flush=True)
json.dump({"rows": rows, "auc_abs_check": a_abs,
           "auc_c1": a1, "auc_c3": a3, "verdict": verdict},
          open(".cache/relative_mass_recut.json", "w"), indent=0)
print("[saved] .cache/relative_mass_recut.json (per-SLOT banked)", flush=True)
