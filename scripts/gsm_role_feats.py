"""gsm_role_feats.py -- THE TREE AS A ROAD (ledger 2026-10-31): per numeral word of every bench text, the Live Oak's two memories of its
answer-key-certified ROLE (gsm_role_tree.roles): [0:7) the exact-key tree's backoff distribution (L2 > L1 > L0, N >= 2, else the
prior) and [7:14) the KNN vote on the student's states (gsm_role_knn's form, k 25), each normalized to sum 1. TRAIN-text rows (TRAIN
and ORIG share texts) are LEAVE-ONE-PROBLEM-OUT in both memories; CF and TEST rows read the full tree. Laid out as the states
memmap (one row per word of each distinct text; zeros on non-numerals). Writes .cache/gsmb_role_feats.npy. Zero-GPU."""
import collections, json, os, sys, zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
import gsm_bench_data as GD
from gsm_role_tree import roles, keys

OUT, KNN_K, MINN = GD.V1.C(os.environ.get("RF_OUT", "gsmb_role_feats_t9ns.npy")), 25, 2   # RF_OUT / RF_STATES (ledger 11-02 CY): another reader's KNN half
LABS = ["*", "+", "-A", "-B", "/A", "/B", "none"]


if __name__ == "__main__":
    assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
    D = json.load(open(GD.OUT)); idx = json.load(open(GD.V1.C("gsmb_trunk_index.json")))
    S = np.load(GD.V1.C(os.environ.get("RF_STATES", "gsmb_t9ns_states.npy")), mmap_mode="r"); li = {l: j for j, l in enumerate(LABS)}
    rk = lambda sp, it, i: keys(it, i, np.random.default_rng(zlib.crc32(f"{sp}|{it['gid']}|{i}".encode())))
    # the exact-key tree and the KNN bank, both from TRAIN
    C = {lv: collections.defaultdict(lambda: np.zeros(len(LABS))) for lv in ("L0", "L1", "L2")}; prior = np.zeros(len(LABS))
    Xtr, ytr, ptr, own = [], [], [], {}
    for n, (it, s0) in enumerate(zip(D["TRAIN"], idx["TRAIN"])):
        own[s0] = n
        for i, r in roles(it).items():
            k = rk("TRAIN", it, i); prior[li[r]] += 1
            for lv in C: C[lv][k[lv]][li[r]] += 1
            Xtr.append(np.asarray(S[s0 + i], np.float32)); ytr.append(li[r]); ptr.append(n)
    Xtr = np.stack(Xtr); MU = Xtr.mean(0); Xtr -= MU; Xtr /= np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-6
    Ytr = np.eye(len(LABS))[ytr]; ptr = np.array(ptr); pri = prior / prior.sum()
    F = np.lib.format.open_memmap(OUT, mode="w+", dtype=np.float32, shape=(len(S), 2 * len(LABS)))
    done = set(); nrow = 0
    for sp in D:
        for it, s0 in zip(D[sp], idx[sp]):
            if s0 in done: continue
            done.add(s0); R = roles(it); pid = own.get(s0, -1)                 # a TRAIN text -> leave its problem out
            ownc = collections.Counter()
            if pid >= 0:
                for i, r in R.items():
                    k = rk("TRAIN", it, i)
                    for lv in C: ownc[(lv, k[lv], r)] += 1
            for i, r in R.items():
                k = rk("TRAIN" if pid >= 0 else sp, it, i); d = pri
                for lv in ("L0", "L1", "L2"):
                    c = C[lv].get(k[lv])
                    if c is None: continue
                    c = c.copy()
                    if pid >= 0:
                        for j, l in enumerate(LABS): c[j] -= ownc[(lv, k[lv], l)]
                    if c.sum() >= MINN: d = c / c.sum()
                x = np.asarray(S[s0 + i], np.float32) - MU; x /= np.linalg.norm(x) + 1e-6
                sim = Xtr @ x
                if pid >= 0: sim[ptr == pid] = -9
                nb = np.argpartition(-sim, KNN_K)[:KNN_K]; v = np.maximum(sim[nb], 0) @ Ytr[nb]
                F[s0 + i] = np.concatenate([d, v / max(v.sum(), 1e-6)]); nrow += 1
    F.flush(); print(f"[rolefeats] {len(done)} texts, {nrow} numeral rows -> {OUT}", flush=True)
