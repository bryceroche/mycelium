"""bindbus_codes_z6.py — THE Z6 RE-MINT (2026-09-18, word given: the Z6 genesis arm). The bind bus's
codebook (CB: 32 codes x 256 planes x (cos, sin); four role offsets theta_* per plane) re-minted with every
phase and offset on Z6 = multiples of pi/3, magnitudes kept from the incumbent: binding (rotate the code by
the role offset) and unbinding become exact integer addition mod 6 per plane (the sextet unification,
ledger 2026-08-29/30: crosstalk statistically identical to continuous; arithmetic snap 89.5% exact under
20-degree noise). Prints both probes for the new file beside the incumbent. usage: bindbus_codes_z6.py [in] [out]"""
import sys, numpy as np
src = sys.argv[1] if len(sys.argv) > 1 else ".cache/bindbus_codes512.npz"; dst = sys.argv[2] if len(sys.argv) > 2 else ".cache/bindbus_codes512_z6.npz"
z = np.load(src); CB = z["CB"]; C, D = CB.shape; P = D // 2; Q = np.pi / 3
def quant(a): return np.round(a / Q) * Q
pairs = CB.reshape(C, P, 2); mag = np.linalg.norm(pairs, axis=-1); ang = np.arctan2(pairs[..., 1], pairs[..., 0]); angq = quant(ang)
CBq = np.stack([mag * np.cos(angq), mag * np.sin(angq)], -1).reshape(C, D); CBq = CBq / np.linalg.norm(CBq, axis=1, keepdims=True)
out = {"CB": CBq.astype(CB.dtype)}
for k in z.files:
    if k.startswith("theta"): out[k] = (quant(z[k]) % (2 * np.pi)).astype(z[k].dtype)
np.savez(dst, **out)
def crosstalk(M):
    G = (M / np.linalg.norm(M, axis=1, keepdims=True)) @ (M / np.linalg.norm(M, axis=1, keepdims=True)).T; off = G[~np.eye(len(M), dtype=bool)]; return float(np.mean(np.abs(off))), float(np.max(np.abs(off)))
def snap(M, th, noise_deg=20.0, trials=2000, seed=0):
    """bind code c with role offset th, add per-plane phase noise, unbind (rotate back), round to Z6: the share of planes exactly recovered"""
    rng = np.random.default_rng(seed); pr = M.reshape(len(M), P, 2); a = np.arctan2(pr[..., 1], pr[..., 0]); exact = 0; tot = 0
    for _ in range(trials):
        c = rng.integers(len(M)); bound = a[c] + th + rng.normal(0, np.deg2rad(noise_deg), P); rec = quant(bound - th); exact += int(np.sum(np.isclose(((rec - a[c]) + np.pi) % (2 * np.pi) - np.pi, 0, atol=1e-6))); tot += P
    return exact / tot
for name, M, th in (("incumbent", CB, z["theta_res"]), ("Z6", CBq, out["theta_res"])):
    m, mx = crosstalk(M); print(f"[z6] {name:<9} crosstalk mean |cos| {m:.4f} max {mx:.3f} | arithmetic snap under 20-deg noise: {snap(M, th):.3f} of planes exact" + ("" if name == "Z6" else " (continuous: the snap rounds a continuous phase — quoted for the contrast)"))
print(f"[z6] {dst}: CB {CBq.shape}, offsets on Z6 ({', '.join(k for k in out if k != 'CB')}); max |angle - Z6| after re-mint {float(np.max(np.abs(((np.arctan2(CBq.reshape(C,P,2)[...,1], CBq.reshape(C,P,2)[...,0]) / Q) - np.round(np.arctan2(CBq.reshape(C,P,2)[...,1], CBq.reshape(C,P,2)[...,0]) / Q))))):.2e} (in units of pi/3)")
