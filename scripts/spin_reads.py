"""spin_reads.py -- ANGLE + COMM (ledger 2026-11-14 09:00, zero-GPU) on saved checkpoints (.cache/rel_dsl_{TAG}_P.npz), the census batch
(256 DEEP items, rng 777), env = the run's config. Per derived var: the last breath's pick correct (a gold target) or not; ANGLE = AUROC
of minus the breath 9->10 turn angle (LayerNorm-centered) for a correct pick; COMM = the attention mass (head mean, last breath) the var
puts on its own incident factors, wrong-pick mean minus right-pick mean. Self-check for dial runs: the radius AUROC equals the banked
dial_census. -> .cache/spin_reads/{TAG}.json"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import rel_dsl_tg as T
from tinygrad import Tensor

ATT = []
_mha = T.mha


def mha_rec(P, y, mask):
    B, L, _ = y.shape
    qkv = y @ P["blocks.0.att.in_proj_weight"].T + P["blocks.0.att.in_proj_bias"]
    q, k, v = (qkv[..., i * T.D:(i + 1) * T.D].reshape(B, L, T.NH, T.DH).transpose(1, 2) for i in range(3))
    ATT.append(((q @ k.transpose(-2, -1)) * (1 / np.sqrt(T.DH)) + mask).softmax(-1))
    return _mha(P, y, mask)


def auroc(sc, ok):
    rk = np.argsort(np.argsort(sc)) + 1; p, n = int(ok.sum()), int((~ok).sum())
    return float((rk[ok].sum() - p * (p + 1) / 2) / (p * n)) if p and n else float("nan")


T.mha = mha_rec
_, _, _, deep = T.RO.load_depth(); os.makedirs(".cache/spin_reads", exist_ok=True)
for tag in sys.argv[1:]:
    z = np.load(f".cache/rel_dsl_{tag}_P.npz"); P = {k: Tensor(z[k]).contiguous().realize() for k in z.files}
    rng = np.random.default_rng(777); its = [deep[i % len(deep)] for i in range(256)]
    arrs, ids = T.make_batch(its, rng); bt, idt = T.tens(arrs, ids); ATT.clear(); r = T.forward(P, bt, idt, T.UNIT, ex=True)
    nv = arrs[0].shape[1]; tg, dv = arrs[8].astype(bool), arrs[9].astype(bool)[:, :nv]
    pick = r["lg"].numpy()[:, -1].argmax(-1)[:, :nv]; ok = np.take_along_axis(tg[:, :nv], pick[..., None], -1)[..., 0][dv]
    a, c = (x.numpy().astype(np.float64) for x in r["xs"][-2:]); a, c = a - a.mean(-1, keepdims=True), c - c.mean(-1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip((a * c).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(c, axis=-1), 1e-12), -1, 1)))[:, :nv][dv]
    inc = ((arrs[4] + arrs[5] + arrs[6]) > 0).transpose(0, 2, 1)                 # B x V x F
    att = ATT[-1].numpy().mean(1)[:, :nv, nv:]                                       # B x V x F (head mean, last breath)
    mass = (att * inc).sum(-1)[dv]
    nvalid = (np.concatenate([arrs[1], arrs[3]], 1) > 0).sum(1)                        # real tokens per item (beside: the uniform-attention baseline)
    uni = (inc.sum(-1) / nvalid[:, None])[dv]
    lr = np.log(r["nrm"][-1].numpy() / r["nrm"][0].numpy())[:, :nv][dv]
    res = dict(tag=tag, n_pos=int(ok.sum()), n_neg=int((~ok).sum()), angle_auroc=auroc(-ang, ok), angle_med_right=float(np.median(ang[ok])),
               angle_med_wrong=float(np.median(ang[~ok])) if (~ok).any() else None, comm_right=float(mass[ok].mean()),
               comm_wrong=float(mass[~ok].mean()) if (~ok).any() else None, radius_auroc=auroc(lr, ok), comm_uniform=float(uni.mean()), att_self_var=float(np.mean([
               (ATT[-1].numpy().mean(1)[:, :nv, :nv] * (arrs[1][:, None, :] > 0)).sum(-1)[dv].mean()])))
    res["comm_diff"] = None if res["comm_wrong"] is None else res["comm_wrong"] - res["comm_right"]
    json.dump(res, open(f".cache/spin_reads/{tag}.json", "w"))
    print(f"[spin] {tag}: right {res['n_pos']} wrong {res['n_neg']} | ANGLE AUROC {res['angle_auroc']:.3f} (median turn right {res['angle_med_right']:.2f} wrong "
          f"{res['angle_med_wrong']} deg) | COMM own-factor mass right {res['comm_right']:.3f} wrong {res['comm_wrong']} diff {res['comm_diff']} | radius AUROC {res['radius_auroc']:.3f} | uniform baseline {res['comm_uniform']:.3f} | mass on vars {res['att_self_var']:.3f}", flush=True)
