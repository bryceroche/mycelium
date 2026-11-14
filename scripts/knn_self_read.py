"""knn_self_read.py -- (ledger 2026-11-14 14:28, zero-GPU) the GK autopsy: routed-head attention mass on self / vars / factors per breath for
derived vars on a GKNN checkpoint (64 DEEP items, rng 777). Env = the run config."""
import os, sys
sys.path.insert(0, "scripts")
import numpy as np, rel_dsl_tg as T
from tinygrad import Tensor
ATT = []; _m = T.mha
def rec(P, y, mask):
    B, L, _ = y.shape; qkv = y @ P["blocks.0.att.in_proj_weight"].T + P["blocks.0.att.in_proj_bias"]
    q, k, v = (qkv[..., i*T.D:(i+1)*T.D].reshape(B, L, T.NH, T.DH).transpose(1, 2) for i in range(3))
    ATT.append(((q @ k.transpose(-2, -1)) * (1/np.sqrt(T.DH)) + mask).softmax(-1).numpy()); return _m(P, y, mask)
T.mha = rec
_, _, _, deep = T.RO.load_depth()
for tag in sys.argv[1:]:
    ATT.clear(); z = np.load(f".cache/rel_dsl_{tag}_P.npz"); P = {k: Tensor(z[k]).contiguous().realize() for k in z.files}
    rng = np.random.default_rng(777); its = [deep[i % len(deep)] for i in range(64)]
    arrs, ids = T.make_batch(its, rng); bt, idt = T.tens(arrs, ids); T.forward(P, bt, idt, T.UNIT, ex=True)
    val = np.concatenate([arrs[1], arrs[3]], 1) > 0; L = val.shape[1]
    selfm = [float(np.diagonal(a.mean(1), axis1=1, axis2=2)[val].mean()) for a in ATT]
    print(tag, "self mass per breath (head mean)", " ".join(f"{s:.3f}" for s in selfm))
    nv = arrs[0].shape[1]; dv = arrs[9].astype(bool)[:, :nv]; a = ATT[-1]
    print(tag, "derived-var mass, routed head mean: self %.3f, vars %.3f, factors %.3f" % (float(np.diagonal(a.mean(1), axis1=1, axis2=2)[:, :nv][dv].mean()), float(a.mean(1)[:, :nv, :nv].sum(-1)[dv].mean()), float(a.mean(1)[:, :nv, nv:].sum(-1)[dv].mean())))
