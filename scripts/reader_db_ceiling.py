"""reader_db_ceiling.py -- M-CEIL (ledger 2026-10-30 15:24, Bryce: "distill llama into our own breathing transformer reader;
mapreduce; leave the MLP behind; memorize facts into the database"). Zero-GPU. How much of the teacher's L0-L3 BP state can a
key tree of BIRCH clustering features (N, LS, SS) over the token and its neighbours carry, with no attention and no weights?
MAP = per-chunk CFs (bincount sums), REDUCE = sum (CFs are additive; tree_db.chan is the same merge in Welford form).
Teacher formpm35cBP; held out 5% of rows by text hash; targets in a fixed PCA-256 basis (variance outside it = unexplained).
Levels L0 tok / L1 (prev,tok) / L2 (prev,tok,next) / L3 (prev2,prev,tok,next); PLACEBO L2 = (tok, two random same-row tokens).
Units: 1 - R^2 = pooled MSE / teacher variance (the distill loss's mse units). Writes .cache/reader_db_ceiling.txt."""
import hashlib, os, sys, time

import numpy as np

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("ALG2", "1")
from phase1_algebra_head import tokenize, STATES_NPZ, STATES_NPY

SPLIT, JSONL, P, MINN, CH = "formpm35cBP", ".cache/form_mix_pm35c.jsonl", 256, 3, 256
OUT = ".cache/reader_db_ceiling.txt"
LEVELS = ("L0", "L1", "L2", "L3", "L2P")
M64 = np.uint64(0x9E3779B97F4A7C15)


def mix(a, b):
    with np.errstate(over="ignore"):
        return (a * M64) ^ (b.astype(np.uint64) + np.uint64(0x632BE59BD9B4E019))


def row_keys(ids, n, rng):
    t = ids[:n].astype(np.uint64); pad = np.uint64(1 << 20)
    prev = np.r_[pad, t[:-1]]; nxt = np.r_[t[1:], pad]; prev2 = np.r_[pad, pad, t[:-2]][:n]
    r1, r2 = t[rng.integers(0, n, n)], t[rng.integers(0, n, n)]
    k0 = t; k1 = mix(k0, prev); k2 = mix(k1, nxt); k3 = mix(k2, prev2)
    kp = mix(mix(mix(k0, np.full(n, 7, np.uint64)), r1), r2)
    return np.stack([k0, k1, k2, k3, kp])


if __name__ == "__main__":
    t0 = time.time(); rng = np.random.default_rng(0)
    samples, ids, mask, _ = tokenize(JSONL)
    tm = np.load(STATES_NPZ.format(split=SPLIT))["tokmask"]; A = np.load(STATES_NPY.format(split=SPLIT), mmap_mode="r")
    assert tm.shape == mask.shape and (tm.astype(bool) == mask.astype(bool)).all(), "tokmask != tokenizer mask"
    ntok = mask.sum(1).astype(int)
    held = np.array([int(hashlib.md5(s["text"].encode()).hexdigest(), 16) % 20 == 0 for s in samples])
    print(f"[mceil] rows {len(samples)} (held {held.sum()}), real tokens {ntok.sum()} ({time.time() - t0:.0f}s)", flush=True)

    # PCA-256 basis + mean on a sample of TRAIN tokens
    sr = rng.choice(np.flatnonzero(~held), 1500, replace=False)
    S = np.concatenate([np.asarray(A[i, :ntok[i]], np.float64) for i in sorted(sr)])
    mu = S.mean(0); C = np.cov(S - mu, rowvar=False); ev, U = np.linalg.eigh(C)
    U = U[:, ::-1][:, :P].astype(np.float32); pca_frac = ev[::-1][:P].sum() / ev.sum()
    print(f"[mceil] PCA-{P} on {len(S)} tokens: {pca_frac:.4f} of variance ({time.time() - t0:.0f}s)", flush=True)
    mu32 = mu.astype(np.float32)

    # one streaming pass: project, keys; train Z kept for the reduce, held-out kept for the read
    Ztr, Ktr, Zho, Kho, Rho = [], [], [], [], []
    for s0 in range(0, len(samples), CH):
        s1 = min(s0 + CH, len(samples)); L = int(ntok[s0:s1].max())
        blk = np.asarray(A[s0:s1, :L], np.float32)
        for j, i in enumerate(range(s0, s1)):
            x = blk[j, :ntok[i]] - mu32; z = x @ U; k = row_keys(ids[i], ntok[i], rng)
            if held[i]: Zho.append(z); Kho.append(k); Rho.append((x * x).sum(1) - (z * z).sum(1))
            else: Ztr.append(z.astype(np.float16)); Ktr.append(k)
        if (s0 // CH) % 40 == 0: print(f"[mceil] map {s1}/{len(samples)} ({time.time() - t0:.0f}s)", flush=True)
    Ztr = np.concatenate(Ztr); Ktr = np.concatenate(Ktr, 1); Zho = np.concatenate(Zho); Kho = np.concatenate(Kho, 1)
    Rho = np.concatenate(Rho).astype(np.float64)
    SST = Rho.sum() + (Zho.astype(np.float64) ** 2).sum()                 # held-out variance about the TRAIN mean
    g_spread = float((Ztr.astype(np.float64) ** 2).sum(1).mean())
    print(f"[mceil] train tokens {len(Ztr)}, held-out tokens {len(Zho)}; reduce ({time.time() - t0:.0f}s)", flush=True)

    # REDUCE per level: CF = (N, LS, SS) by key; held-out lookup by searchsorted
    CF = {}
    sq = (Ztr.astype(np.float32) ** 2).sum(1)
    for li, lv in enumerate(LEVELS):
        uk, inv = np.unique(Ktr[li], return_inverse=True)
        N = np.bincount(inv, minlength=len(uk)).astype(np.float64); keep = N >= MINN     # only nodes with N >= MINN are stored
        LS = np.stack([np.bincount(inv, weights=Ztr[:, d].astype(np.float64), minlength=len(uk))[keep] for d in range(P)], 1)
        SS = np.bincount(inv, weights=sq, minlength=len(uk))[keep]; nk = len(uk); uk, N = uk[keep], N[keep]
        mean = (LS / N[:, None]).astype(np.float32); spread = SS / N - (LS / N[:, None]).__pow__(2).sum(1); del LS
        pos = np.clip(np.searchsorted(uk, Kho[li]), 0, len(uk) - 1); hit = uk[pos] == Kho[li]
        CF[lv] = dict(n_keys=nk, mean=mean, N=N, spread=spread, pos=pos, hit=hit)
        print(f"[mceil] {lv}: {nk} keys ({len(uk)} with N>={MINN}); held-out hit with N>={MINN} {hit.mean():.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    del Ztr

    Zh = Zho.astype(np.float64)
    def r2(pred):
        return 1 - (Rho.sum() + ((Zh - pred) ** 2).sum()) / SST
    def lookup(lv, minn=MINN):
        c = CF[lv]; ok = c["hit"] & (c["N"][c["pos"]] >= minn); return ok, c["mean"][c["pos"]]
    def backoff(levels, minn=MINN):
        pred = np.zeros_like(Zh); depth = np.full(len(Zh), -1)
        for d, lv in enumerate(levels):
            ok, m = lookup(lv, minn); pred[ok] = m[ok]; depth[ok] = d
        return pred, depth

    L = [f"M-CEIL (teacher {SPLIT}; train tokens {Ktr.shape[1]}, held-out tokens {len(Zh)} from {held.sum()} rows; PCA-{P} keeps "
         f"{pca_frac:.4f} of variance = the R^2 ceiling of any lookup here)"]
    res = {}
    for lv in LEVELS:
        ok, m = lookup(lv); pred = np.where(ok[:, None], m, 0.0)
        # fall back to L0 where the level misses, so every level is graded on every token
        ok0, m0 = lookup("L0"); pred = np.where((~ok & ok0)[:, None], m0, pred)
        res[lv] = r2(pred); L.append(f"  {lv:4s} keys {CF[lv]['n_keys']:>9d}  R2 {res[lv]:.4f}  (1-R2 {1 - res[lv]:.4f})")
    for name, lv in (("BACKOFF L0>L1>L2>L3", ("L0", "L1", "L2", "L3")), ("BACKOFF L0>L1>L2", ("L0", "L1", "L2"))):
        for minn in (MINN, 10):
            pred, depth = backoff(lv, minn); v = r2(pred); res[f"{name}|{minn}"] = v
            L.append(f"  {name} (N>={minn}): R2 {v:.4f} (1-R2 {1 - v:.4f}); depth shares "
                     + " ".join(f"{d}:{(depth == d).mean():.3f}" for d in range(-1, len(lv))))
    best = max(v for k, v in res.items() if k.startswith("BACKOFF"))

    # BIRCH descent: go deeper only while the current node's spread exceeds T x the global spread
    L.append(f"  BIRCH descent (global spread {g_spread:.4f}):")
    order = ("L0", "L1", "L2", "L3"); birch = []
    for T in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
        pred = np.zeros_like(Zh); stop = np.full(len(Zh), -1); live = np.ones(len(Zh), bool)
        for d, lv in enumerate(order):
            ok, m = lookup(lv); take = live & ok; pred[take] = m[take]; stop[take] = d
            c = CF[lv]; live = take & (c["spread"][c["pos"]] > T * g_spread)
        v = r2(pred); above = ((stop >= 0) & (stop < len(order) - 1)).mean(); birch.append((v, T, above))
        L.append(f"    T {T:.1f}: R2 {v:.4f}; stop shares " + " ".join(f"{lv}:{(stop == d).mean():.3f}" for d, lv in enumerate(order))
                 + f" | unseen {(stop == -1).mean():.3f}")
    bv, bT, babove = max(birch)
    m1a = best - res["L0"]; m1b = res["L2"] - res["L2P"]
    M1 = m1a >= 0.05 and m1b >= 0.03; M2 = (1 - best) <= 0.31; M3 = babove >= 0.30
    L += [f"M1 CONTEXT PAYS: best backoff - token-only {m1a:+.4f} (bar +0.05); L2 - placebo L2 {m1b:+.4f} (bar +0.03) -> {'PASS' if M1 else 'FAIL'}",
          f"M2 THE DB IS AN ORGAN: lookup-only 1-R2 {1 - best:.4f} (bar <= 0.31; SwiGLU student mse 0.154) -> {'PASS' if M2 else 'FAIL'}",
          f"M3 PLATEAU: at the best T ({bT:.1f}, R2 {bv:.4f}) {babove:.3f} of tokens stop above L3 (bar >= 0.30) -> {'PASS' if M3 else 'FAIL'}",
          f"({time.time() - t0:.0f}s)"]
    print("\n".join(L)); open(OUT, "w").write("\n".join(L) + "\n")
