"""RB8 (ledger 2026-11-03): build the per-level Tree rows for the TRAIN split once -> .cache/readerdb3_formpm35cBP.npy
[rows, T_ALG, 3 x 256] fp16, LEAVE-ONE-OUT from each row's own teacher state (reader_db_build.level_rows). Custody check: the backoff
collapse of these rows must reproduce the banked readerdb_formpm35cBP.npy rows (sampled)."""
import os, sys, time
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "scripts"); os.environ.setdefault("ALG2", "1")
from phase1_algebra_head import tokenize, STATES_NPY, T_ALG
import reader_db_build as RDB

OUT = ".cache/readerdb3_formpm35cBP.npy"
assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
mu, U, scale, store = RDB.load_store()
_, ids, mask, _ = tokenize(RDB.TRAIN[0])
teach = np.load(STATES_NPY.format(split=RDB.TRAIN[1]), mmap_mode="r"); ref = np.load(RDB.DBNPY.format(split=RDB.TRAIN[1]), mmap_mode="r")
st = np.lib.format.open_memmap(OUT + ".part", mode="w+", dtype=np.float16, shape=(len(ids), T_ALG, 3 * RDB.P))
t0 = time.time()
for s0 in range(0, len(ids), 256):
    sl = slice(s0, min(s0 + 256, len(ids)))
    st[sl] = RDB.level_rows(store, mu, U, scale, ids[sl], mask[sl], teach[sl], T_ALG)
    if s0 % 10240 == 0: print(f"[rb8rows] {s0}/{len(ids)} ({time.time() - t0:.0f}s)", flush=True)
st.flush()
idx = np.sort(np.random.default_rng(0).choice(len(ids), 64, replace=False))
lv = np.asarray(st[idx], np.float32).reshape(len(idx), T_ALG, 3, RDB.P); back = np.zeros((len(idx), T_ALG, RDB.P), np.float32)
for l in range(3):
    ok = np.abs(lv[:, :, l]).sum(-1) > 0; back[ok] = lv[:, :, l][ok]
m = mask[idx] > 0; err = float(np.abs(back - np.asarray(ref[idx], np.float32))[m].max())
assert err < 1e-3, f"backoff collapse differs from the banked rows: {err}"
os.rename(OUT + ".part", OUT)
print(f"[rb8rows] wrote {OUT} {st.shape}; custody: backoff collapse vs banked max |d| {err:.2e} (64 rows) ({time.time() - t0:.0f}s)", flush=True)
