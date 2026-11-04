"""AG3 part 3, THE VIEW-1 ROLE FEATS (ledger 2026-11-04): zero-GPU. The BT4M8 models read per-numeral role feats (the Live Oak's two
memories, gsm_role_feats.py). For view 1 the memories must be THE MODEL'S OWN (built from view-0 TRAIN: the exact-key tree counts and the KNN
bank on view-0 mbr8 states) and the QUERIES come from the view: keys from the view tokens, KNN on the view-1 states. ORIG rows are leave-
one-problem-out by gid (the problem's own view-0 TRAIN contributions removed; as gsm_role_feats does); TEST rows plain. Same formula, same
seeds (crc32 of sp|gid|i), same LABS / MINN / KNN_K as gsm_role_feats.py (imported, not copied).
Writes ONLY the view file; rows laid out as the view states memmap.
env: V_DATA (.cache/gsmb_data_v1.json) V_IDX (.cache/gsmb_trunk_index_v1.json) V_STATES (.cache/gsmb_mbr8_states_v1.npy)
     V_OUT (.cache/gsmb_role_feats_mbr8_v1.npy)
"""
import collections, json, os, sys, zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
import gsm_bench_data as GD
import gsm_role_feats as RF
from gsm_role_tree import roles, keys

LABS, KNN_K, MINN = RF.LABS, RF.KNN_K, RF.MINN
E = os.environ.get
OUT = E("V_OUT", ".cache/gsmb_role_feats_mbr8_v1.npy")
assert OUT.endswith("_v1.npy") and os.path.abspath(OUT) != os.path.abspath(".cache/gsmb_role_feats_mbr8.npy") and not os.path.exists(OUT)
D0 = json.load(open(GD.OUT)); idx0 = json.load(open(GD.V1.C("gsmb_trunk_index.json"))); S0 = np.load(GD.V1.C("gsmb_mbr8_states.npy"), mmap_mode="r")
D1 = json.load(open(E("V_DATA", ".cache/gsmb_data_v1.json"))); idx1 = json.load(open(E("V_IDX", ".cache/gsmb_trunk_index_v1.json")))
S1 = np.load(E("V_STATES", ".cache/gsmb_mbr8_states_v1.npy"), mmap_mode="r"); li = {l: j for j, l in enumerate(LABS)}
rk = lambda sp, it, i: keys(it, i, np.random.default_rng(zlib.crc32(f"{sp}|{it['gid']}|{i}".encode())))
C = {lv: collections.defaultdict(lambda: np.zeros(len(LABS))) for lv in ("L0", "L1", "L2")}; prior = np.zeros(len(LABS))
Xtr, ytr, ptr, own = [], [], [], {}
for n, (it, s0) in enumerate(zip(D0["TRAIN"], idx0["TRAIN"])):
    own[it["gid"]] = n
    for i, r in roles(it).items():
        k = rk("TRAIN", it, i); prior[li[r]] += 1
        for lv in C: C[lv][k[lv]][li[r]] += 1
        Xtr.append(np.asarray(S0[s0 + i], np.float32)); ytr.append(li[r]); ptr.append(n)
Xtr = np.stack(Xtr); MU = Xtr.mean(0); Xtr -= MU; Xtr /= np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-6
Ytr = np.eye(len(LABS))[ytr]; ptr = np.array(ptr); pri = prior / prior.sum()
F = np.lib.format.open_memmap(OUT, mode="w+", dtype=np.float32, shape=(len(S1), 2 * len(LABS)))
done, nrow = set(), 0
for sp in D1:
    for it, s1 in zip(D1[sp][:len(idx1[sp])], idx1[sp]):
        if s1 in done: continue
        done.add(s1); R = roles(it); pid = own.get(it["gid"], -1) if sp == "ORIG" else -1     # ORIG = TRAIN problems: leave the problem out
        ownc = collections.Counter()
        if pid >= 0:
            for i, r in roles(D0["TRAIN"][pid]).items():
                k = rk("TRAIN", D0["TRAIN"][pid], i)
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
            x = np.asarray(S1[s1 + i], np.float32) - MU; x /= np.linalg.norm(x) + 1e-6
            sim = Xtr @ x
            if pid >= 0: sim[ptr == pid] = -9
            nb = np.argpartition(-sim, KNN_K)[:KNN_K]; v = np.maximum(sim[nb], 0) @ Ytr[nb]
            F[s1 + i] = np.concatenate([d, v / max(v.sum(), 1e-6)]); nrow += 1
F.flush(); print(f"[rolefeats_v1] {len(done)} texts, {nrow} numeral rows -> {OUT}", flush=True)
