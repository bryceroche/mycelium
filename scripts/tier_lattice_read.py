"""tier_lattice_read.py -- TIER-LATTICE (ledger 2026-11-12 19:22 "THE QUEUE PINNED", item 3; theory: tiers = invariance contracts 11-07).
Zero training. Views of each TRAIN problem with >=1 number AND >=1 name token: N (every digit run re-drawn, every digit CHANGED, same
length/format, leading digit of a multi-digit run nonzero), W (every name swapped for a different lexicon name, consistent per problem),
NW (both, same draws). Never reorders sentences; never touches the CF fixture.
  --views OUT.json      write {"texts": [unique texts], "meta": {...,"rows": [[orig,N,W,NW] text indices per problem]}} for the GPU step
                        (distill_reader.py --local: json "texts" list -> OUT.npy row i = pooled state of texts[i], plus OUT.npy.json sidecar)
  --measure STATES.npy [--vjson views.json]  Sigma_T = cov(h - h_T); O(N,W); ceiling = O(N_A,N_B) on two random disjoint halves; bootstrap SE; s_T; BL share;
                        additivity; LAT-x verdict -> .cache/tier_lattice.txt (override --out). Dim is 768 << n, covariances well-conditioned: no PCA."""
import argparse, json, os, re, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--views"); ap.add_argument("--measure"); ap.add_argument("--out", default=".cache/tier_lattice.txt")
ap.add_argument("--vjson", default=".cache/tier_lattice_views.json"); ap.add_argument("--boot", type=int, default=1000); ap.add_argument("--seed", type=int, default=0); args = ap.parse_args()
assert bool(args.views) != bool(args.measure), "give exactly one of --views / --measure"
VIEWS = ("N", "W", "NW")


def build():
    import gsm_bench_data as GD, gsm_trunk_reader as TR, gsm_trunk_states as GTS
    D, _ = TR.load(); _, Q = GTS.texts(); items = D["TRAIN"]; qs = Q["TRAIN"]; assert len(items) == len(qs)
    for it, q in zip(items, qs): it["ch"] = [m.span() for m in GD.TOK.finditer(q)]; assert len(it["ch"]) == len(it["toks"])
    lex = sorted({q[a:b] for it, q in zip(items, qs) for (a, b), n in zip(it["ch"], it["isn"]) if n})   # the data's name lexicon (surface forms)
    ntok = lambda q: len(GD.tokenize(q, set())[0])
    pop = [i for i, it in enumerate(items) if any(v is not None for v in it["vals"]) and any(it["isn"])]
    texts, idx, rows, skip = [], {}, [], dict(retok=0, same=0)
    def put(t):
        if t not in idx: idx[t] = len(texts); texts.append(t)
        return idx[t]
    def redraw(q, i):   # every digit run: each digit changed, leading digit of a multi-digit run nonzero; per-problem rng so N and NW share draws
        rng = np.random.default_rng([args.seed, i])
        def f(m):
            s = m.group(); o = []
            for k, c in enumerate(s):
                lo = 1 if (k == 0 and len(s) > 1) else 0; ch = [d for d in range(lo, 10) if d != int(c)]; o.append(str(ch[rng.integers(len(ch))]))
            return "".join(o)
        return re.sub(r"\d+", f, q)
    for i in pop:
        it, q = items[i], qs[i]; rng = np.random.default_rng([args.seed, 1, i]); mp = {}
        spans = [((a, b), q[a:b]) for (a, b), n in zip(it["ch"], it["isn"]) if n]
        for _, s in spans:
            if s not in mp: c = [x for x in lex if x != s]; mp[s] = c[rng.integers(len(c))]
        w = q
        for (a, b), s in sorted(spans, reverse=True): w = w[:a] + mp[s] + w[b:]   # right to left keeps offsets valid
        n_, nw = redraw(q, i), redraw(w, i)
        if n_ == q or w == q: skip["same"] += 1; continue
        if any(ntok(v) != len(it["toks"]) for v in (n_, w, nw)): skip["retok"] += 1; continue
        rows.append([put(t) for t in (q, n_, w, nw)])
        it["_v"] = (q, n_, w, nw)
    return texts, rows, pop, skip, items, lex


if args.views:
    texts, rows, pop, skip, items, lex = build()
    json.dump(dict(texts=texts, meta=dict(order=["orig", "N", "W", "NW"], rows=rows, seed=args.seed, population=len(pop), kept=len(rows), skipped=skip, lexicon=len(lex))), open(args.views, "w"))
    print(f"population {len(pop)} (>=1 number & >=1 name), kept {len(rows)}, skipped {skip}, name lexicon {len(lex)}, unique texts {len(texts)} -> {args.views}")
    for it in [items[i] for i in pop if "_v" in items[i]][:2]:
        for lab, t in zip(("orig", "N", "W", "NW"), it["_v"]): print(f"  {lab:4s}: {t}")
        print()
    sys.exit(0)

# ---------------- measure ----------------
Hs = np.load(args.measure).astype(np.float64); vj = json.load(open(args.measure + ".json")) if os.path.exists(args.measure + ".json") else None
V = json.load(open(args.vjson)); rows = np.array(V["meta"]["rows"]); assert Hs.shape[0] == len(V["texts"]), (Hs.shape, len(V["texts"]))
if vj is not None: assert vj["texts"] == V["texts"], "states sidecar texts != views texts (row order mismatch)"
assert np.isfinite(Hs).all(); n = len(rows)
h = Hs[rows[:, 0]]; d = {t: h - Hs[rows[:, 1 + k]] for k, t in enumerate(VIEWS)}   # (n,D) difference vectors
cov = lambda x: (x - x.mean(0)).T @ (x - x.mean(0)) / len(x)
tr = lambda A, B: float((A * B).sum())   # symmetric: tr(AB) = sum(A*B)
ov = lambda A, B: tr(A, B) / np.sqrt(tr(A, A) * tr(B, B))
f32 = lambda x: x.astype(np.float32)
def stats(ix, rng):
    S = {t: cov(f32(d[t][ix])).astype(np.float64) for t in VIEWS}; o = ov(S["N"], S["W"])
    p = rng.permutation(len(ix)); a, b = p[:len(p) // 2], p[len(p) // 2:]
    c = ov(cov(f32(d["N"][ix][a])).astype(np.float64), cov(f32(d["N"][ix][b])).astype(np.float64)); return S, o, c
rng = np.random.default_rng(args.seed); S, O, C = stats(np.arange(n), rng)
bo, bc, bd = [], [], []
for _ in range(args.boot):
    _, o, c = stats(rng.integers(0, n, n), rng); bo.append(o); bc.append(c); bd.append(c - o)
perm = np.random.default_rng(args.seed + 1).permutation(n)
while (perm == np.arange(n)).any(): perm = np.where(perm == np.arange(n), (perm + 1) % n, perm)   # derangement-ish: h' always a different problem
scale = ((h - h[perm]) ** 2).sum(1).mean(); s = {t: (d[t] ** 2).sum(1).mean() / scale for t in VIEWS}
add = np.linalg.norm(S["NW"] - S["N"] - S["W"]) / np.linalg.norm(S["NW"]); se = float(np.std(bd, ddof=1)); diff = C - O
ver = "PASS" if O < C - 2 * se else "FAIL"
rep = [f"TIER-LATTICE read (ledger 11-12 pin item 3) -- {args.measure}; n problems {n}, dim {Hs.shape[1]} (no PCA), bootstrap {args.boot} resamples over problems (seed {args.seed})",
       f"sensitivities s_N {s['N']:.4f}  s_W {s['W']:.4f}  s_NW {s['NW']:.4f}   (scale E|h-h'|^2 {scale:.4f})   BL share 1-s_NW = {1 - s['NW']:.4f}",
       f"O(N,W) {O:.4f} (boot SE {np.std(bo, ddof=1):.4f})   ceiling O(N_A,N_B) {C:.4f} (boot SE {np.std(bc, ddof=1):.4f})   ceiling-O {diff:.4f} (boot SE {se:.4f})",
       f"additivity |S_NW - S_N - S_W|_F/|S_NW|_F = {add:.4f}",
       f"LAT-x: O(N,W) {O:.4f} vs ceiling {C:.4f} (diff {diff:.4f}, 2SE {2 * se:.4f}) -> {ver}"]
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True); open(args.out, "w").write("\n".join(rep) + "\n"); print("\n".join(rep))
