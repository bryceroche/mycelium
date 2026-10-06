"""THE ATLAS RADIUS READ (2026-10-05, Bryce's gut; zero GPU): does an op-kind centroid's RADIUS tighten
breath by breath while its ANGLE stays put? Data: the clock-band probe's collected states
(.cache/clock_band_states_<tag>.npz: (311 rows, 6 loop breaths, 24 factor slots, 512 dims); breath index
0..5 = loop breaths 1..6) on the wild holdout. CONTENT dims only (the head's own _hier_band_dims(): the
union of the root/branch/leaf bands, 384 of 512; the 128 clock dims excluded). The content planes are
already in the canon frame — only the clock planes turn — so centroids across breaths are comparable as-is.
Gold kind per slot = ftype x op x form from the fixture's annotated factor graph (slot k introduces var k;
rel with result == k is the forward form add/mul, otherwise the inverse form sub/div), cross-checked
against the LV_DUMP's gold fields and the probe's meta; the custody door (row_gold) is asserted per row,
never a pen-written solution field. RIGHT slots = ps_legal_wild_<tag>.npz ok.
usage: .venv/bin/python3 scripts/atlas_radius_read.py [tags...]   (DEV=CPU family env set here)"""
import os, sys, json, pickle, collections
for _k, _v in (("DEV", "CPU"), ("ALG2", "1"), ("ALG_FTYPES", "9"), ("ALG_DUP", "1"), ("ALG_WIDE", "1"), ("ALG_HW", "512"),
               ("ALG_BREATH", "7"), ("ALG_NOTEBOOK", "1"), ("ALG_SIXWAVE", "1"), ("NB_PERSLOT", "1"), ("ALG_BINDBUS", "7"),
               ("ALG_BIND_D", "512"), ("BIND_CODES", ".cache/bindbus_codes512r.npz"), ("ALG_BUSGARAGE", "2"),
               ("ALG_SHELF_CIRCLE", "2"), ("ALG_ALTMASK", "1"), ("ALG_ALT21", "1"), ("ALG_ALT2", "1"), ("ALG_MASKHEAD", "1"),
               ("ALG_FED", "1"), ("ALG_POLAR", "1"), ("ALG_POLAR_D", "128"), ("ALG_POLAR_EM", "0.1"),
               ("ALG_POLAR_D_INIT", ".cache/polar_waist_init_d128u.npz"), ("ALG_PRUNE", "pforms,s4,fednl0,lane2"),
               ("ALG_SLOT_ALL", "1"), ("ALG_STELLAR", "2"), ("ALG_CLOCK_CANON", "1"), ("SC_EVAL", "0")):
    os.environ.setdefault(_k, _v)
assert os.environ["DEV"] == "CPU", "zero-GPU read"
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
import phase1_algebra_head as H
from mycelium.custody_gold import row_gold

TAGS = sys.argv[1:] or ["PMS8_241", "HS_241", "HSd_241"]
FIX = ".cache/wild_admitted_holdout.jsonl"; OUT = ".cache/atlas_radius_read.txt"; MIN_N = 40
LINES = []
def P(s=""): print(s, flush=True); LINES.append(s)

bands, clock = H._hier_band_dims()
CONTENT = np.sort(np.concatenate(bands)); assert len(np.intersect1d(CONTENT, clock)) == 0
rows = [json.loads(l) for l in open(FIX)]; assert len(rows) == 311, len(rows)
for r in rows: assert isinstance(row_gold(r), int)          # the custody door passes on every row (harvest key)
def gold_kind(fac, j):
    if fac["ftype"] == "given": return "given"
    assert fac["ftype"] == "rel", fac["ftype"]
    return "rel:" + ({"add": "add", "mul": "mul"} if fac["result"] == j else {"add": "sub", "mul": "div"})[fac["op"]]
def cos(a, b): return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)
def fmt(name, kinds, tab, nb=6):
    P(f"  {name:<34}" + "".join(f"{'b'+str(b+1):>8}" for b in range(nb)))
    for k in kinds: P(f"  {k:<34}" + "".join(f"{tab[k][b]:8.3f}" for b in range(nb)))

P(f"THE ATLAS RADIUS READ — content dims {len(CONTENT)}/512 (bands {[len(b) for b in bands]}; clock {len(clock)} excluded; "
  f"content planes are the canon frame: only the clock planes turn). wild holdout n={len(rows)}; kinds with >= {MIN_N} slots.")
P("radius = mean ||x-mu_b||/||mu_b||; cos2mu = mean cos(x, mu_b); ang_prev = cos(mu_b, mu_{b-1}); ang_fin = cos(mu_b, mu_b6);")
P("NC acc = leave-one-out nearest-centroid (cosine) accuracy among the kept kinds; margin = cos(own) - best other (LOO).")
VERDICT = {}
for tag in TAGS:
    z = np.load(f".cache/clock_band_states_{tag}.npz"); S, M = z["states"], z["meta"]
    assert S.shape[0] == len(rows) and S.shape[1] == 6, S.shape
    D = pickle.load(open(f".cache/dump_wild_{tag}.pkl", "rb")); ps = np.load(f".cache/ps_legal_wild_{tag}.npz")
    okmap = {(int(r), int(j)): bool(o) for r, j, o in zip(ps["rows"], ps["slots"], ps["ok"])}
    ri, ji, kind, ok = [], [], [], []
    for t in D:                                     # (row, j, gft, gop, gargs, gres, gdig, pft, pop, pargs, pres, pdig, ppres, pdup)
        r, j, gft, gop, gres = t[0], t[1], t[2], t[3], t[5]
        k = gold_kind(rows[r]["factors"][j], j)
        assert M[r, j, 0] == 1 and (M[r, j, 1] == 0) == (k == "given") and (M[r, j, 1] == 1) == (k != "given"), (tag, r, j)
        if k != "given": assert gft == 0 and gop == ("mul" in k or "div" in k) and (gres == j) == (k in ("rel:add", "rel:mul")), (tag, r, j, k, t[:6])
        else: assert gft == 1
        ri.append(r); ji.append(j); kind.append(k); ok.append(okmap[(r, j)])
    ri, ji, kind, ok = np.array(ri), np.array(ji), np.array(kind), np.array(ok)
    X = S[ri, :, ji, :][:, :, CONTENT].astype(np.float32)     # (N, 6, C)
    P(f"\n=== {tag}: {len(ri)} gold slots ({int(ok.sum())} RIGHT, slot-exact {ok.mean():.4f}); kinds: "
      + ", ".join(f"{k} {n}" for k, n in sorted(collections.Counter(kind.tolist()).items())))
    for sub in ("RIGHT", "ALL"):
        sel = ok if sub == "RIGHT" else np.ones_like(ok)
        cnt = collections.Counter(kind[sel].tolist()); kinds = [k for k in sorted(cnt) if cnt[k] >= MIN_N]
        keep = sel & np.isin(kind, kinds); Xk, kk = X[keep], kind[keep]
        mu = {k: Xk[kk == k].mean(0) for k in kinds}                                   # (6, C) each
        rad, c2m, ap, af, acc, mg = ({k: np.zeros(6) for k in kinds} for _ in range(6))
        tot = np.zeros(6)
        for k in kinds:
            xs = Xk[kk == k]; n = len(xs)
            rad[k] = (np.linalg.norm(xs - mu[k], axis=-1) / np.linalg.norm(mu[k], axis=-1)).mean(0)
            c2m[k] = cos(xs, mu[k][None]).mean(0)
            ap[k] = np.r_[np.nan, cos(mu[k][1:], mu[k][:-1])]; af[k] = cos(mu[k], mu[k][5:6])
            own = cos(xs, ((n * mu[k])[None] - xs) / (n - 1))                           # LOO own centroid
            oth = np.max(np.stack([cos(xs, mu[o][None]) for o in kinds if o != k], 0), 0) if len(kinds) > 1 else own * 0 - 1
            acc[k] = (own > oth).mean(0); mg[k] = (own - oth).mean(0); tot += (own > oth).sum(0)
        P(f"\n--- {tag} / {sub} slots ({int(keep.sum())} slots over {len(kinds)} kinds: " + ", ".join(f"{k} {cnt[k]}" for k in kinds) + ")")
        fmt("radius (rel. spread)", kinds, rad); fmt("cos2mu", kinds, c2m); fmt("ang_prev", kinds, ap); fmt("ang_fin", kinds, af)
        fmt("NC acc (LOO)", kinds, acc); fmt("NC margin (LOO)", kinds, mg)
        P(f"  {'NC acc pooled':<34}" + "".join(f"{tot[b] / keep.sum():8.3f}" for b in range(6)))
        mono = {k: bool(np.all(np.diff(rad[k][1:]) <= 0)) for k in kinds}
        d16 = {k: (rad[k][5] - rad[k][0]) / rad[k][0] for k in kinds}
        settle = {k: next((b + 1 for b in range(6) if np.all(af[k][b:] > 0.99)), None) for k in kinds}
        P(f"  SUMMARY {tag}/{sub}: radius monotone after b1: {sum(mono.values())}/{len(kinds)} kinds "
          f"[{', '.join(f'{k} {'yes' if mono[k] else 'no'} {100*d16[k]:+.1f}%' for k in kinds)}]; "
          f"angle settles (ang_fin > 0.99 from): {', '.join(f'{k} b{settle[k]}' for k in kinds)}; "
          f"NC acc pooled b1 {tot[0]/keep.sum():.3f} -> b6 {tot[5]/keep.sum():.3f}")
        VERDICT[(tag, sub)] = (np.mean(list(d16.values())), tot[0] / keep.sum(), tot[5] / keep.sum())
P("\nVERDICT (mean over kinds): " + " | ".join(f"{t}/{s}: radius b1->b6 {100*d:+.1f}%, NC acc {a0:.3f}->{a5:.3f}" for (t, s), (d, a0, a5) in VERDICT.items()))
open(OUT, "w").write("\n".join(LINES) + "\n"); print(f"[atlas-radius] wrote {OUT}")
