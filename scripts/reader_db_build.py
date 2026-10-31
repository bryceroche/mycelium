"""reader_db_build.py -- STEP 1 of THE MLP-FREE STUDENT (ledger 2026-10-30, word given): the reader's DATABASE organ.
MAP: per-chunk BIRCH clustering features (N, LS, SS) of the teacher's formpm35cBP states in a PCA-256 basis, keyed L0 tok /
L1 (prev,tok) / L2 (prev,tok,next) (M-CEIL: L3 adds +0.001). REDUCE: sum (CFs are additive). Store .cache/reader_db.npz
(certifier llama-teacher; never mixed with answer-key facts). Per-token DB rows (backoff L0>L1>L2, "others >= 3"), scaled by
1/sqrt(global spread), written as fp16 memmaps .cache/readerdb_{SPLIT}.npy (rows, T_ALG, 256), pads zero:
  TRAIN (formpm35cBP): LEAVE-ONE-OUT (LS - z_self) / (N - 1) with N - 1 >= 3 -- the lookup never sees its own target;
  EVAL (wildholdBP, test23BP): plain lookup with N >= 3.
Reads: R^2 per level and backoff on each split (wild = the W1/W2 bars). Writes .cache/reader_db_build.txt."""
import os, sys, time

import numpy as np

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("ALG2", "1")
from phase1_algebra_head import tokenize, STATES_NPZ, STATES_NPY, T_ALG

P, MINN, CH = 256, 3, 256
TRAIN = (".cache/form_mix_pm35c.jsonl", "formpm35cBP")
EVALS = ((".cache/wild_admitted_holdout.jsonl", "wildholdBP"), (".cache/algebra_nl_test.jsonl", "test23BP"))
DB, OUT = ".cache/reader_db.npz", ".cache/reader_db_build.txt"
DBNPY = ".cache/readerdb_{split}.npy"
LEVELS = ("L0", "L1", "L2")
M64 = np.uint64(0x9E3779B97F4A7C15)


def mix(a, b):
    with np.errstate(over="ignore"):
        return (a * M64) ^ (b.astype(np.uint64) + np.uint64(0x632BE59BD9B4E019))


def row_keys(ids, n):
    t = ids[:n].astype(np.uint64); pad = np.uint64(1 << 20)
    prev = np.r_[pad, t[:-1]]; nxt = np.r_[t[1:], pad]
    k1 = mix(t, prev); return np.stack([t, k1, mix(k1, nxt)])


def load_store(path=DB):
    """-> (mu, U, scale, {level: (keys, N, LS, SS)}) from the saved DB"""
    z = np.load(path); return z["mu"], z["U"], float(z["scale"]), {lv: tuple(z[f"{lv}_{n}"] for n in ("keys", "N", "LS", "SS")) for lv in LEVELS}


def lookup(store, k, z=None, kappa=None):
    """one row's DB prediction [n, P] (unscaled). kappa None = as built (the deepest node with N >= MINN); kappa = SHRINKAGE (each
    level's mean blended with its parent's by n / (n + kappa), the root prior 0 = the corpus mean). z given = leave-one-out."""
    pred = np.zeros((k.shape[1], P), np.float32)
    for li, lv in enumerate(l for l in LEVELS if l in store):     # a truncated store = a shallower tree
        uk, N, LS, _ = store[lv]; pos = np.clip(np.searchsorted(uk, k[li]), 0, len(uk) - 1); hit = uk[pos] == k[li]
        n = np.where(hit, N[pos] - (1 if z is not None else 0), 0.0)
        S = np.where(hit[:, None], LS[pos] - (z if z is not None else 0), 0.0)
        if kappa is None:
            ok = n >= MINN; pred[ok] = S[ok] / n[ok, None]
        else:
            pred = ((S + kappa * pred) / (n + kappa)[:, None]).astype(np.float32)
    return pred


def load(jsonl, split):
    _, ids, mask, _ = tokenize(jsonl)
    tm = np.load(STATES_NPZ.format(split=split))["tokmask"]; A = np.load(STATES_NPY.format(split=split), mmap_mode="r")
    assert tm.shape == mask.shape and (tm.astype(bool) == mask.astype(bool)).all(), f"{split}: tokmask != tokenizer mask"
    return ids, mask.sum(1).astype(int), A


def project(A, ntok, ids, mu, U):
    """-> per row (z [n,P] float32, resid [n] = ||x-mu||^2 - ||z||^2, keys [3,n])"""
    for s0 in range(0, len(ntok), CH):
        s1 = min(s0 + CH, len(ntok)); blk = np.asarray(A[s0:s1, :int(ntok[s0:s1].max())], np.float32)
        for j, i in enumerate(range(s0, s1)):
            x = blk[j, :ntok[i]] - mu; z = x @ U
            yield i, z, (x * x).sum(1) - (z * z).sum(1), row_keys(ids[i], ntok[i])


if __name__ == "__main__":
    for f in [DB] + [DBNPY.format(split=s) for _, s in (TRAIN,) + EVALS]:
        assert not os.path.exists(f), f"refusing to overwrite {f}"
    t0 = time.time(); rng = np.random.default_rng(0); L = []
    ids, ntok, A = load(*TRAIN)
    sr = rng.choice(len(ntok), 1500, replace=False)
    S = np.concatenate([np.asarray(A[i, :ntok[i]], np.float64) for i in sorted(sr)])
    mu = S.mean(0); ev, U = np.linalg.eigh(np.cov(S - mu, rowvar=False))
    U = U[:, ::-1][:, :P].astype(np.float32); mu = mu.astype(np.float32); pca_frac = ev[::-1][:P].sum() / ev.sum(); del S
    L.append(f"PCA-{P}: {pca_frac:.4f} of the teacher's variance")

    # MAP: one pass over the teacher; keep z (fp16) for the reduce and the LOO write
    Z, K, R, off = [], [], [], np.zeros(len(ntok) + 1, np.int64)
    for i, z, r, k in project(A, ntok, ids, mu, U):
        Z.append(z.astype(np.float16)); K.append(k); R.append(r); off[i + 1] = off[i] + len(z)
    Z = np.concatenate(Z); K = np.concatenate(K, 1); R = np.concatenate(R).astype(np.float64)
    g_spread = float((Z.astype(np.float64) ** 2).sum(1).mean()); scale = 1 / np.sqrt(g_spread)
    print(f"[dbb] map: {len(Z)} tokens ({time.time() - t0:.0f}s)", flush=True)

    # REDUCE: CF = (N, LS, SS) per key; nodes with N >= MINN are stored
    sq = (Z.astype(np.float32) ** 2).sum(1); store = {}
    for li, lv in enumerate(LEVELS):
        uk, inv = np.unique(K[li], return_inverse=True)
        N = np.bincount(inv, minlength=len(uk)).astype(np.float64); keep = N >= MINN
        LS = np.stack([np.bincount(inv, weights=Z[:, d].astype(np.float64), minlength=len(uk))[keep] for d in range(P)], 1)
        SS = np.bincount(inv, weights=sq, minlength=len(uk))[keep]
        store[lv] = (uk[keep], N[keep], LS.astype(np.float32), SS)
        print(f"[dbb] reduce {lv}: {len(uk)} keys, {int(keep.sum())} stored ({time.time() - t0:.0f}s)", flush=True)
    np.savez(DB, mu=mu, U=U, scale=scale, pca_frac=pca_frac, certifier="llama-teacher", teacher=TRAIN[1],
             **{f"{lv}_{n}": a for lv in LEVELS for n, a in zip(("keys", "N", "LS", "SS"), store[lv])})

    def rows_db(z, k, loo):
        """backoff L0 > L1 > L2 -> (pred [n,P], depth [n])"""
        pred = np.zeros_like(z, dtype=np.float32); depth = np.full(len(z), -1); per = {}
        for li, lv in enumerate(LEVELS):
            uk, N, LS, _ = store[lv]; pos = np.clip(np.searchsorted(uk, k[li]), 0, len(uk) - 1); hit = uk[pos] == k[li]
            n = N[pos] - (1 if loo else 0); ok = hit & (n >= MINN)
            m = (LS[pos] - (z if loo else 0)) / np.maximum(n, 1)[:, None]
            per[lv] = np.where(ok[:, None], m, 0.0)
            pred[ok] = m[ok]; depth[ok] = li
        return pred, depth, per

    def write(split, ntok, gen, label=None):
        st = np.lib.format.open_memmap(DBNPY.format(split=split), mode="w+", dtype=np.float16, shape=(len(ntok), T_ALG, P))
        sse = {k: 0.0 for k in LEVELS + ("BACKOFF",)}; sst = 0.0; dep = np.zeros(len(LEVELS) + 1)
        for i, z, r, k, loo in gen:
            pred, depth, per = rows_db(z, k, loo); row = np.zeros((T_ALG, P), np.float16); row[:len(z)] = pred * scale
            st[i] = row; sst += r.sum() + (z.astype(np.float64) ** 2).sum()
            for lv in LEVELS:
                pl = per[lv]; pl = np.where((pl == 0).all(1, keepdims=True), per["L0"], pl)   # a level's miss falls back to L0
                sse[lv] += r.sum() + ((z - pl) ** 2).sum()
            sse["BACKOFF"] += r.sum() + ((z - pred) ** 2).sum(); dep += np.bincount(depth + 1, minlength=len(LEVELS) + 1)
        st.flush(); r2 = {k: 1 - v / sst for k, v in sse.items()}
        L.append(f"{label or split}: " + " ".join(f"{k} R2 {v:.4f}" for k, v in r2.items()) + " | backoff depth shares "
                 + " ".join(f"{n}:{c / dep.sum():.3f}" for n, c in zip(("miss",) + LEVELS, dep)))
        print(f"[dbb] {L[-1]} ({time.time() - t0:.0f}s)", flush=True); return r2

    def train_gen():
        for i in range(len(ntok)):
            yield i, Z[off[i]:off[i + 1]].astype(np.float32), R[off[i]:off[i + 1]], K[:, off[i]:off[i + 1]], True
    write(TRAIN[1], ntok, train_gen(), label=TRAIN[1] + " (LOO)")
    del Z, K, R
    res = {}
    for jsonl, split in EVALS:
        e_ids, e_n, e_A = load(jsonl, split)
        res[split] = write(split, e_n, ((i, z, r, k, False) for i, z, r, k in project(e_A, e_n, e_ids, mu, U)))
    w = res["wildholdBP"]; W1 = w["BACKOFF"] >= 0.55; W2 = w["BACKOFF"] - w["L0"] >= 0.05
    L += [f"W1 wild backoff R2 {w['BACKOFF']:.4f} (bar >= 0.55; PCA ceiling {pca_frac:.4f}) -> {'PASS' if W1 else 'FAIL'}",
          f"W2 wild backoff - token-only {w['BACKOFF'] - w['L0']:+.4f} (bar +0.05) -> {'PASS' if W2 else 'FAIL'}",
          f"db rows scaled by 1/sqrt(global spread {g_spread:.3f}); ({time.time() - t0:.0f}s)"]
    print("\n".join(L)); open(OUT, "w").write("\n".join(L) + "\n")
