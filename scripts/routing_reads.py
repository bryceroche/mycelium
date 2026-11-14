"""routing_reads.py -- the KNN census (ledger 2026-11-14 routing queue, zero-GPU) on saved GKNN checkpoints (.cache/rel_dsl_{TAG}_P.npz),
the census batch (256 DEEP items, rng 777), env = the run's config: per breath, the share of each real token's k chosen neighbours
(self excluded) that are TRUE factor-graph neighbours, vs chance (true degree / (real tokens - 1)). -> .cache/spin_reads/{TAG}_knn.json"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import rel_dsl_tg as T
from tinygrad import Tensor

_, _, _, deep = T.RO.load_depth(); os.makedirs(".cache/spin_reads", exist_ok=True)
for tag in sys.argv[1:]:
    z = np.load(f".cache/rel_dsl_{tag}_P.npz"); P = {k: Tensor(z[k]).contiguous().realize() for k in z.files}
    rng = np.random.default_rng(777); its = [deep[i % len(deep)] for i in range(256)]
    arrs, ids = T.make_batch(its, rng); bt, idt = T.tens(arrs, ids); r = T.forward(P, bt, idt, T.UNIT, ex=True)
    nv = arrs[0].shape[1]; B, NVp, NFp = arrs[0].shape[0], arrs[0].shape[1], arrs[2].shape[1]; L = NVp + NFp
    ig = ((arrs[4] + arrs[5] + arrs[6]) > 0).transpose(0, 2, 1)
    nbr = np.zeros((B, L, L), bool); nbr[:, :nv, nv:] = ig; nbr[:, nv:, :nv] = ig.transpose(0, 2, 1)
    val = np.concatenate([arrs[1], arrs[3]], 1) > 0; eye = np.eye(L, dtype=bool)[None]
    nreal = val.sum(1)[:, None].repeat(L, 1)
    chance = float((nbr.sum(-1)[val] / (nreal[val] - 1)).mean())
    prec = []
    for s in r["knn"]:
        s = s.numpy() & ~eye & val[:, None, :]; hit = (s & nbr).sum(-1); n = s.sum(-1)
        prec.append(float((hit[val] / np.maximum(n[val], 1)).mean()))
    out = dict(tag=tag, precision=prec, chance=chance, ratio_last=prec[-1] / chance)
    print(f"[knn] {tag}: true-neighbour share of the k picks per breath " + " ".join(f"{p:.3f}" for p in prec) + f" | chance {chance:.3f} | last / chance {prec[-1] / chance:.2f}", flush=True)
    json.dump(out, open(f".cache/spin_reads/{tag}_knn.json", "w"))
