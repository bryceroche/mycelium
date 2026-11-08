"""gsm_btc_autopsy.py -- THE BTC AUTOPSY (ledger 2026-11-07; zero GPU: DEV=CPU on the banked BTC_s{S} weights + atoms).

The run's scored type accuracy swung +-0.2 between evals while the loss fell smoothly. Two reads separate the suspects:
  TF   teacher-forced type accuracy: slots < k committed with their GOLD types (radius 1), slot k read at breath floor(k NCYC / K)
       (the reader per-pointer protocol) -- the classifier without exposure to its own mistakes
  TF*  the same with the atoms REBUILT: the mean silhouette per gold type over TRAIN[:N] teacher-forced slots under the final weights
       (a settled codebook) -- if TF* >> TF, the Welford atoms lag the silhouettes (the drift suspect)
  TFP  teacher-forced pointer accuracy (type = gold): both operands valid
Scored type_ok comes from the run's FINAL json. TF << 1 with TF* ~ TF: the classifier itself; TF high but scored low: exposure bias.
Env: BTC_S (seed, default 0), AU_N (items per split, default 600). -> .cache/gsm_btc_autopsy_s{S}.txt
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gsm_btc as G                      # (imports gsm_breath_reader first, the 11-04 rule)
import numpy as np
from tinygrad import Tensor
from tinygrad.nn.state import safe_load

BR = G.BR; K, NCYC, NT = G.K, G.NCYC, G.NT
S = int(os.environ.get("BTC_S", "0")); N = int(os.environ.get("AU_N", "600"))
Dt, states = BR.TR.load(); tree = BR.TR.load_tree(); BR.RF = np.load(BR.GD.V1.C(BR.rolef_name()), mmap_mode="r")
t0_, E0 = BR.batch(Dt["TRAIN"][:2], states, tree, np.random.default_rng(0)); P0 = G.port_batch(t0_, E0, Dt["TRAIN"][:2], 64)
bt = G.BTM.BreathingTransformer(G.config(P0["x"].shape[-1], P0["aux"].shape[-1]))
bt.load(safe_load(f".cache/gsmb_btc_BTC_s{S}.safetensors"))
A0 = np.load(f".cache/gsmb_btc_BTC_s{S}_atoms.npy")


def tf_read(its, atoms, collect=False):
    """-> (type hits, pointer hits, [(z, gold type)] if collect) over every gold slot k < m, teacher-forced"""
    th, ph, zs = [], [], []
    for s0 in range(0, len(its), 64):
        part = its[s0:s0 + 64]
        t, E = BR.batch(part, states, tree, np.random.default_rng(s0)); L = t["pad"].shape[1]; B = len(part)
        for b, it in enumerate(part): E[b]["vals"] = it["vals"]
        progs, gt = BR.gold(E); m = np.array([e["m"] for e in E])
        Lb = max(64, int(np.ceil((L - K) / G.BUCKET) * G.BUCKET)); P = G.port_batch(t, E, part, Lb); V = G.valid_masks(P, B)
        ta, tp = bt.layout.angles(P["tf"], (B, Lb)); T = lambda a: Tensor(np.ascontiguousarray(a))
        va, vb = t["va"].numpy(), t["vb"].numpy()
        for k in range(int(m.max())):
            com = np.zeros((B, K), bool); com[:, :k] = True; c = min(k * NCYC // K, NCYC - 1)
            nb = BR.notebook(E, L, progs, com, gt, np.ones((B, K), np.float32)); H_ = G.host_io(bt, P, nb, np.full(B, c))
            lt, z, lp, _, _ = G.forward(bt, Tensor(atoms), T(P["x"]), T(P["aux"]), T(ta), T(tp), T(H_["sa"]), T(H_["sp"]), T(H_["mt"]),
                                        T(H_["ms"]), T(H_["com"]), T(H_["atom"].astype(np.int32)), T(H_["rad"]), T(H_["cert"]),
                                        T(V["token_self"]), T(V["slot_self"]), T(V["cross"]), T(V["pt"]), T(V["ps"]), T(V["pc"]),
                                        bt.c.tau(c), cond=Tensor(gt.astype(np.int32)))
            J = G.joint(lp, Tensor(G._KM[gt]), [None, T(V["ps"]).any(-1), None])
            pa, pb = (np.concatenate([J[r][0].numpy()[..., :P["Lq"]], np.full((B, K, L - P["Lq"]), -1e4), J[r][1].numpy(), J[r][2].numpy()], -1)
                      for r in "ab")
            ty = lt.numpy().argmax(-1); zz = z.numpy()
            for b in range(B):
                if k >= m[b]: continue
                th.append(ty[b, k] == gt[b, k])
                ph.append(bool(va[b, k, pa[b, k].argmax()] and vb[b, k, pb[b, k].argmax()]))
                if collect: zs.append((zz[b, k], gt[b, k]))
    return np.mean(th), np.mean(ph), zs


L_ = []
_, _, Z = tf_read(Dt["TRAIN"][:N], A0, collect=True)
A1 = A0.copy()
for j in range(NT):
    zj = [z for z, g in Z if g == j]
    if zj: A1[j] = np.mean(zj, 0)
R = json.load(open(f".cache/gsmb_btc_BTC_s{S}.json"))
for sp in ("ORIG", "TEST"):
    tf, tfp, _ = tf_read(Dt[sp][:N], A0); tfs, _, _ = tf_read(Dt[sp][:N], A1)
    L_.append(f"{sp}[:{N}] s{S}: TF type {tf:.4f} | TF* (rebuilt atoms) {tfs:.4f} | TFP pointer {tfp:.4f} | scored type_ok {R[f'{sp}_type_ok']:.4f} "
              f"step_ptr {R[f'{sp}_step_ptr']:.4f} (all items)")
L_.append(f"  atoms rebuilt from {len(Z)} TRAIN slots; mean |A1 - A0| {np.abs(A1 - A0).mean():.4f} (|A0| {np.abs(A0).mean():.4f})")
print("\n".join(L_)); open(f".cache/gsm_btc_autopsy_s{S}.txt", "w").write("\n".join(L_) + "\n")
