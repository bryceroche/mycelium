"""rj_coupling_read.py -- THE ROMEO-AND-JULIET READ (ledger 2026-11-14 17:05, zero-GPU, weights only). Bryce 11-14: the coupled systems as
3b1b's Romeo and Juliet (dR/dt = J, dJ/dt = -R: an ANTISYMMETRIC coupling, e^{At} a rotation; a SYMMETRIC one is heat flow). In the
breath, a token's write from another is attention weight (>= 0, the same both ways when routed along an undirected edge) times the head's
OV map M_h = W_O,h W_V,h (D x D), so the var <-> factor pair coupling [[0, aM], [bM, 0]] is antisymmetric only as M is. Per head: the
antisymmetric fraction ||M - M^T||^2 / (||M - M^T||^2 + ||M + M^T||^2) (0.5 = a random matrix, 1 = pure rotation, 0 = pure diffusion)
and the spectral one sum |Im lambda| / sum |lambda|. Arms: G8, UG8, G8P (placebo: same sparsity, wrong edges), U, G4 (routed 0-3 / free
4-7), additive s0, and torch init. Bars: ROTATION iff G8 head mean >= 0.6 both seeds AND G8 - G8P >= +0.05 both seeds; DIFFUSION iff
G8 <= 0.4 both seeds; else MIXED. -> .cache/rj_coupling_read.txt"""
import os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
D, NH = 256, 8; DH = D // NH


def heads(z):
    W = z["blocks.0.att.in_proj_weight"]; Wv = W[2 * D:3 * D]; Wo = z["blocks.0.att.out_proj.weight"]; out = []
    for h in range(NH):
        M = Wo[:, h * DH:(h + 1) * DH] @ Wv[h * DH:(h + 1) * DH]; A, S = (M - M.T) / 2, (M + M.T) / 2
        lam = np.linalg.eigvals(M); out.append(((A ** 2).sum() / ((A ** 2).sum() + (S ** 2).sum()), np.abs(lam.imag).sum() / np.abs(lam).sum()))
    return np.array(out)


C = ".cache/rel_dsl_TGNOMLP{}_depth2c0.25_s{}_P.npz"
arms = {"G8": [C.format("G8", s) for s in (0, 1)], "UG8": [C.format("UG8", s) for s in (0, 1)], "G8P": [C.format("G8P", s) for s in (0, 1)],
        "G4": [C.format("G4", s) for s in (0, 1)], "U": [C.format("U", "0_step"), C.format("U", "1_rs")], "additive": [C.format("", "0_step")]}
res, out = {}, []
import torch, rel_dsl_organs as RO
torch.manual_seed(0); init = {k: v.numpy() for k, v in RO.make_net().state_dict().items()}
hi = heads(init); out.append(f"  init (torch seed 0)  antisym per head " + " ".join(f"{a:.2f}" for a, _ in hi) + f" | mean {hi[:, 0].mean():.3f} | spectral-imag mean {hi[:, 1].mean():.3f}")
for k, fs in arms.items():
    res[k] = []
    for i, f in enumerate(fs):
        h = heads(np.load(f)); res[k].append(h)
        extra = f" | routed 0-3 {h[:4, 0].mean():.3f} free 4-7 {h[4:, 0].mean():.3f}" if k == "G4" else ""
        out.append(f"  {k:8s} s{i} antisym per head " + " ".join(f"{a:.2f}" for a, _ in h) + f" | mean {h[:, 0].mean():.3f} | spectral-imag mean {h[:, 1].mean():.3f}{extra}")
g8 = [h[:, 0].mean() for h in res["G8"]]; dp = [a[:, 0].mean() - b[:, 0].mean() for a, b in zip(res["G8"], res["G8P"])]
v = "ROTATION" if min(g8) >= 0.6 and min(dp) >= 0.05 else "DIFFUSION" if max(g8) <= 0.4 else "MIXED"
out.append(f"R-J: G8 head mean {g8[0]:.3f} / {g8[1]:.3f}; G8 - G8P {dp[0]:+.3f} / {dp[1]:+.3f} -> {v}")
txt = "\n".join(out); print(txt); open(".cache/rj_coupling_read.txt", "w").write(txt + "\n")
