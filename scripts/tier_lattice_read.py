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
ap = argparse.ArgumentParser(); ap.add_argument("--views"); ap.add_argument("--measure"); ap.add_argument("--out")
ap.add_argument("--vjson", default=".cache/tier_lattice_views.json"); ap.add_argument("--boot", type=int, default=1000); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--units", action="store_true", help="LAT-U (ledger 11-15): add the units view U (+ NWU); rows need U != orig")
args = ap.parse_args()
args.out = args.out or (".cache/tier_lattice_u.txt" if args.units else ".cache/tier_lattice.txt")
assert bool(args.views) != bool(args.measure), "give exactly one of --views / --measure"
VIEWS = ("N", "W", "NW", "U", "NWU") if args.units else ("N", "W", "NW")
UNIT_OUT = {"fewer", "million", "percent"}   # (LAT-U pin) post-numeral words that are not units: a comparative and two scale operators


def build():
    import gsm_bench_data as GD, gsm_trunk_reader as TR, gsm_trunk_states as GTS
    D, _ = TR.load(); _, Q = GTS.texts(); items = D["TRAIN"]; qs = Q["TRAIN"]; assert len(items) == len(qs)
    for it, q in zip(items, qs): it["ch"] = [m.span() for m in GD.TOK.finditer(q)]; assert len(it["ch"]) == len(it["toks"])
    lex = sorted({q[a:b] for it, q in zip(items, qs) for (a, b), n in zip(it["ch"], it["isn"]) if n})   # the data's name lexicon (surface forms)
    ntok = lambda q: len(GD.tokenize(q, set())[0])
    pop = [i for i, it in enumerate(items) if any(v is not None for v in it["vals"]) and any(it["isn"])]
    ulex = []
    if args.units:   # (LAT-U) the data's unit lexicon: lowercase words of >= 3 letters, >= 5 times right after a numeral, >= 0.6 of their uses there
        from collections import Counter
        aft, tot = Counter(), Counter()
        for it, q in zip(items, qs):
            ms = list(GD.TOK.finditer(q))
            for k, m in enumerate(ms):
                w = m.group(0)
                if w.isalpha() and not it["isn"][k]:
                    tot[w] += 1; aft[w] += k > 0 and bool(GD.NUMRX.fullmatch(ms[k - 1].group(0)))
        ulex = sorted(w for w in aft if aft[w] >= 5 and aft[w] / tot[w] >= 0.6 and len(w) >= 3 and w.islower() and w not in UNIT_OUT)
    def uswap(q, i):   # every occurrence of a lexicon unit -> a distinct other lexicon unit, consistent (injective) per problem
        rng = np.random.default_rng([args.seed, 2, i]); ms = [m for m in GD.TOK.finditer(q) if m.group(0) in ulex]; mp = {}
        for m in ms:
            w = m.group(0)
            if w not in mp: c = [x for x in ulex if x != w and x not in mp.values() and x not in {m2.group(0) for m2 in ms}]; mp[w] = c[rng.integers(len(c))]
        for m in sorted(ms, key=lambda m: -m.start()): q = q[:m.start()] + mp[m.group(0)] + q[m.end():]
        return q
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
        vs = (q, n_, w, nw)
        if args.units:
            u = uswap(q, i); vs = vs + (u, uswap(nw, i))
            if u == q: skip.setdefault("u_same", 0); skip["u_same"] += 1; continue
        if n_ == q or w == q: skip["same"] += 1; continue
        if any(ntok(v) != len(it["toks"]) for v in vs[1:]): skip["retok"] += 1; continue
        rows.append([put(t) for t in vs])
        it["_v"] = vs
    return texts, rows, pop, skip, items, (lex, ulex)


if args.views:
    texts, rows, pop, skip, items, lex = build()
    lex, ulex = lex; order = ["orig", *VIEWS]
    json.dump(dict(texts=texts, meta=dict(order=order, rows=rows, seed=args.seed, population=len(pop), kept=len(rows), skipped=skip, lexicon=len(lex), unit_lexicon=ulex)), open(args.views, "w"))
    print(f"population {len(pop)} (>=1 number & >=1 name), kept {len(rows)}, skipped {skip}, name lexicon {len(lex)}, unit lexicon {len(ulex)}, unique texts {len(texts)} -> {args.views}")
    if ulex: print("unit lexicon:", " ".join(ulex))
    for it in [items[i] for i in pop if "_v" in items[i]][:2]:
        for lab, t in zip(order, it["_v"]): print(f"  {lab:4s}: {t}")
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
def stats(ix, rng, full=True):
    S = {t: cov(f32(d[t][ix])).astype(np.float64) for t in (VIEWS if full else ("N", "W"))}; o = ov(S["N"], S["W"])
    p = rng.permutation(len(ix)); a, b = p[:len(p) // 2], p[len(p) // 2:]
    c = ov(cov(f32(d["N"][ix][a])).astype(np.float64), cov(f32(d["N"][ix][b])).astype(np.float64)); return S, o, c
rng = np.random.default_rng(args.seed); S, O, C = stats(np.arange(n), rng)
bo, bc, bd = [], [], []
for _ in range(args.boot):
    _, o, c = stats(rng.integers(0, n, n), rng, full=False); bo.append(o); bc.append(c); bd.append(c - o)
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
if args.units:   # LAT-U (ledger 11-15): is the units face a third direction set, or inside the span of the number + name faces?
    nt = lambda A: A / np.trace(A)
    def ustats(ix, rng):
        Sx = {t: cov(f32(d[t][ix])).astype(np.float64) for t in ("N", "W", "U")}; NW_ = nt(Sx["N"]) + nt(Sx["W"])
        p = rng.permutation(len(ix)); a, b = p[:len(p) // 2], p[len(p) // 2:]
        cu = ov(cov(f32(d["U"][ix][a])).astype(np.float64), cov(f32(d["U"][ix][b])).astype(np.float64))
        return dict(span=ov(Sx["U"], NW_), un=ov(Sx["U"], Sx["N"]), uw=ov(Sx["U"], Sx["W"]), cu=cu)
    rng = np.random.default_rng(args.seed + 2); U0 = ustats(np.arange(n), rng); B = [ustats(rng.integers(0, n, n), rng) for _ in range(args.boot)]
    sd = lambda k: float(np.std([x[k] for x in B], ddof=1)); seu = float(np.std([x["cu"] - x["span"] for x in B], ddof=1))
    addu = np.linalg.norm(S["NWU"] - S["N"] - S["W"] - S["U"]) / np.linalg.norm(S["NWU"])
    if U0["cu"] < 0.5: vu = "VOID (C_U < 0.5: no reproducible units face to count)"
    elif U0["span"] < U0["cu"] - 2 * seu: vu = "SEPARATE (a third face: the 3-transform lattice has 2^3 - 1 = 7 cells)"
    else: vu = "MERGE-" + ("N" if U0["un"] >= U0["uw"] else "W") + " (inside the number + name span: the lattice stays at 3 cells)"
    rep += [f"--- LAT-U (ledger 11-15): unit lexicon {len(V['meta']['unit_lexicon'])} words; s_U {s['U']:.4f}  s_NWU {s['NWU']:.4f}  BL share 1-s_NWU = {1 - s['NWU']:.4f}",
            f"O(U, N+W) {U0['span']:.4f} (SE {sd('span'):.4f})   O(U,N) {U0['un']:.4f} (SE {sd('un'):.4f})   O(U,W) {U0['uw']:.4f} (SE {sd('uw'):.4f})",
            f"ceiling O(U_A,U_B) {U0['cu']:.4f} (SE {sd('cu'):.4f})   ceiling - O(U,N+W) {U0['cu'] - U0['span']:.4f} (SE {seu:.4f})",
            f"additivity |S_NWU - S_N - S_W - S_U|_F/|S_NWU|_F = {addu:.4f}",
            f"LAT-U: O(U,N+W) {U0['span']:.4f} vs ceiling {U0['cu']:.4f} (2SE {2 * seu:.4f}) -> {vu}"]
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True); open(args.out, "w").write("\n".join(rep) + "\n"); print("\n".join(rep))
