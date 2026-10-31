"""gsm_role_knn.py -- G-TREE3 (ledger 2026-10-31): the Live Oak's KNN. The KEY is a numeral word's state from the reader (default the
MLP-free student, .cache/gsmb_student_states.npy; GK_STATES=gsmb_trunk_states.npy reads the BANKED Llama states as an instrument);
the VALUE is the answer-key-certified role (gsm_role_tree.roles). Cosine KNN over TRAIN numeral words, k 25, similarity-weighted vote;
TRAIN read leave-one-problem-out. Zero-GPU. Writes .cache/gsm_role_knn_{STATES}.txt."""
import collections, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
import gsm_bench_data as GD
from gsm_role_tree import roles

K = 25
ST = os.environ.get("GK_STATES", "gsmb_student_states.npy")


def bank(D, idx, S, sp):
    X, y, pid = [], [], []
    for n, (it, s0) in enumerate(zip(D[sp], idx[sp])):
        for i, r in roles(it).items():
            X.append(np.asarray(S[s0 + i], np.float32)); y.append(r); pid.append(n)
    X = np.stack(X); X -= MU; X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-6
    return X, np.array(y), np.array(pid)


if __name__ == "__main__":
    D = json.load(open(GD.OUT)); idx = json.load(open(GD.V1.C("gsmb_trunk_index.json"))); S = np.load(GD.V1.C(ST), mmap_mode="r")
    MU = 0
    Xtr, ytr, ptr = bank(D, idx, S, "TRAIN"); MU = Xtr.mean(0)      # center on the TRAIN numeral mean, then re-bank
    Xtr, ytr, ptr = bank(D, idx, S, "TRAIN")
    labs = sorted(set(ytr)); li = {l: j for j, l in enumerate(labs)}; Ytr = np.eye(len(labs))[[li[l] for l in ytr]]
    prior = collections.Counter(ytr).most_common(1)[0][0]
    L = [f"G-TREE3 KNN (keys = {ST}; k {K}; TRAIN numeral words {len(ytr)})"]; res = {}
    for sp in ("TRAIN", "TEST", "CFNUM", "CFNAME"):
        X, y, p = (Xtr, ytr, ptr) if sp == "TRAIN" else bank(D, idx, S, sp)
        hit = []
        for s0 in range(0, len(X), 1024):
            sim = X[s0:s0 + 1024] @ Xtr.T
            if sp == "TRAIN": sim[p[s0:s0 + 1024, None] == ptr[None]] = -9      # leave the problem out
            nb = np.argpartition(-sim, K, 1)[:, :K]; w = np.maximum(np.take_along_axis(sim, nb, 1), 0)
            vote = np.einsum("nk,nkc->nc", w, Ytr[nb]); hit += list(np.array(labs)[vote.argmax(1)] == y[s0:s0 + 1024])
        res[sp] = (float(np.mean(hit)), float(np.mean(y == prior)))
        L.append(f"  {sp:6s} KNN {res[sp][0]:.3f} | prior {res[sp][1]:.3f}" + (" (leave-one-problem-out)" if sp == "TRAIN" else ""))
    t = res["TEST"]
    L += [f"T1 THE TREE KNOWS ROLES: TEST KNN {t[0]:.3f} - prior {t[1]:.3f} = {t[0] - t[1]:+.3f} (bar +0.10) -> {'PASS' if t[0] - t[1] >= 0.10 else 'FAIL'}",
          f"T2 SOFT KEYS BEAT EXACT KEYS: TEST KNN - G-TREE L0 (0.518) = {t[0] - 0.518:+.3f} (bar +0.03) -> {'PASS' if t[0] - 0.518 >= 0.03 else 'FAIL'}"]
    print("\n".join(L)); open(f".cache/gsm_role_knn_{ST.split('.')[0]}.txt", "w").write("\n".join(L) + "\n")
