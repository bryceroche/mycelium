"""reg_precompute.py — per-row REGISTER signal for form8 (2026-08-26;
register-aware commitment's input feature). The mouth's exact read
replicated: pooled trunk states (memmap, chunked) -> top-8 kNN cosine
distance vs the deployed 2000-row bank -> length-corrected
(knn - (coef0 + coef1/L)). INPUT feature, not gold; the mouth stays
diagnostic (Goodhart fence) — this is the pawl being handed a copy of
the map, not the mouth entering any loss. CPU-only (trunk states are
input-space: no never-mix concern — the mouth's own jurisdiction).
Output: .cache/reg_form8.npy (n,) float32, aligned to form8 row order.
"""
import numpy as np

z = np.load('.cache/phase1_alg_states_form8.npz')
tk = z['tokmask'].astype(np.float32)
st = np.load('.cache/phase1_alg_states_form8_states.npy', mmap_mode='r')
n = len(tk)
assert len(st) == n, f"desync: {len(st)} states vs {n} tokmask"
zr = np.load('.cache/recognition_mouth.npz')
bank = zr['bank'].astype(np.float32)
zl = np.load('.cache/mouth_length_correction.npz')
coef = zl['coef'].astype(np.float32)
out = np.zeros(n, np.float32)
CH = 512
for s0 in range(0, n, CH):
    s1 = min(s0 + CH, n)
    stc = np.asarray(st[s0:s1]).astype(np.float32)
    m = tk[s0:s1][:, :, None]
    v = (stc * m).sum(1) / np.maximum(m.sum(1), 1)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    d = 1.0 - v @ bank.T
    knn = np.sort(d, axis=1)[:, :8].mean(1)
    L = np.maximum(tk[s0:s1].sum(1), 1)
    out[s0:s1] = knn - (coef[0] + coef[1] / L)
    if s0 % 8192 == 0:
        print(f"[reg] {s0}/{n}", flush=True)
np.save('.cache/reg_form8.npy', out)
print(f"[reg] saved .cache/reg_form8.npy: n={n} mean {out.mean():.4f} "
      f"std {out.std():.4f} p5 {np.percentile(out,5):.4f} "
      f"p95 {np.percentile(out,95):.4f} (mouth thr 0.0072)", flush=True)
