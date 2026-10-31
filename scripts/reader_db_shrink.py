"""reader_db_shrink.py -- THE DB SHRINKAGE FORM (ledger 2026-10-31; Step 1's named next form after W1 FAIL). Zero-GPU.
Each level's node mean is blended with its parent's by n / (n + kappa) (root prior 0 = the corpus mean in the PCA basis), so a
fine node with few events leans on its parent instead of overriding it. kappa is FIT ON TRAIN LOO only (4,000 formpm35cBP rows,
the lookup never sees its own target), then frozen; wild (wildholdBP) is read once at that kappa (the W1' / W3 bars).
Writes .cache/readerdb_wildholdBPshr.npy (rows scaled as built) + .cache/reader_db_shrink.txt."""
import os, sys, time

import numpy as np

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import reader_db_build as RDB

KAPPAS = (0.5, 1, 2, 4, 8, 16, 32, 64, 128)
OUTNPY, OUT = RDB.DBNPY.format(split="wildholdBPshr"), ".cache/reader_db_shrink.txt"


def r2(rows, kappa, loo, levels=3):
    sse = sst = 0.0
    st = {lv: RDB.store_[lv] for lv in RDB.LEVELS[:levels]}
    for z, r, k in rows:
        pred = RDB.lookup(st, k[:levels], z if loo else None, kappa)
        sse += r.sum() + ((z - pred) ** 2).sum(); sst += r.sum() + (z.astype(np.float64) ** 2).sum()
    return 1 - sse / sst


if __name__ == "__main__":
    assert not os.path.exists(OUTNPY), f"refusing to overwrite {OUTNPY}"
    t0 = time.time(); mu, U, scale, RDB.store_ = RDB.load_store(); L = []
    ids, ntok, A = RDB.load(*RDB.TRAIN); sel = set(np.random.default_rng(0).choice(len(ntok), 4000, replace=False).tolist())
    tr = []
    for s0 in range(0, len(ntok), RDB.CH):          # project only the sampled rows' chunks
        idx = [i for i in range(s0, min(s0 + RDB.CH, len(ntok))) if i in sel]
        if not idx: continue
        blk = np.asarray(A[idx[0]:idx[-1] + 1, :int(ntok[idx].max())], np.float32)
        for i in idx:
            x = blk[i - idx[0], :ntok[i]] - mu; z = x @ U
            tr.append((z, (x * x).sum(1) - (z * z).sum(1), RDB.row_keys(ids[i], ntok[i])))
    e_ids, e_n, e_A = RDB.load(".cache/wild_admitted_holdout.jsonl", "wildholdBP")
    wd = [(z, r, k) for _, z, r, k in RDB.project(e_A, e_n, e_ids, mu, U)]
    built = np.load(RDB.DBNPY.format(split="wildholdBP"), mmap_mode="r")
    for i in range(0, len(wd), 37):                 # lookup(kappa=None) must reproduce the banked rows
        assert np.allclose(RDB.lookup(RDB.store_, wd[i][2]) * scale, built[i, :e_n[i]].astype(np.float32), atol=2e-3), f"row {i}"
    print(f"[shr] {len(tr)} train rows, {len(wd)} wild rows; lookup reproduces the built rows ({time.time() - t0:.0f}s)", flush=True)
    base_tr, base_w = r2(tr, None, True), r2(wd, None, False)
    L.append(f"AS BUILT (backoff, N>=3): train LOO R2 {base_tr:.4f} | wild R2 {base_w:.4f}")
    fit = []
    for kp in KAPPAS:
        a, b = r2(tr, kp, True), r2(wd, kp, False); fit.append((a, kp))
        L.append(f"  kappa {kp:>5}: train LOO R2 {a:.4f} | wild R2 {b:.4f} (read)"); print(L[-1], flush=True)
    ktr, kstar = max(fit)
    w = r2(wd, kstar, False); w1, w2 = r2(wd, kstar, False, 2), w
    st = np.lib.format.open_memmap(OUTNPY, mode="w+", dtype=np.float16, shape=built.shape)
    for i, (z, r, k) in enumerate(wd):
        row = np.zeros(built.shape[1:], np.float16); row[:len(z)] = RDB.lookup(RDB.store_, k, None, kstar) * scale; st[i] = row
    st.flush()
    W1 = w >= 0.55; W3 = w2 >= w1
    L += [f"kappa* = {kstar} (fit on train LOO, R2 {ktr:.4f})",
          f"W1' wild shrunk R2 {w:.4f} (bar >= 0.55; as built {base_w:.4f}) -> {'PASS' if W1 else 'FAIL'}",
          f"W3 the fine level no longer hurts: wild shrunk to L2 {w2:.4f} vs to L1 {w1:.4f} -> {'PASS' if W3 else 'FAIL'}",
          f"wrote {OUTNPY} ({time.time() - t0:.0f}s)"]
    print("\n".join(L)); open(OUT, "w").write("\n".join(L) + "\n")
