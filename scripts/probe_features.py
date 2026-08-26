"""probe_features.py — sufficient statistics for the three-arm token-grain
count probe (2026-08-26, word given): per row, MEAN-pooled trunk state +
token count L. Arms A (mean=intensive), B (sum=mean*L=extensive), C
(angle-winding of the same sum) all derive from (P, L). CPU on the
banked memmap; the mechanism test needs no GPU."""
import numpy as np
z = np.load('.cache/phase1_alg_states_form8.npz')
tk = z['tokmask'].astype(np.float32)
st = np.load('.cache/phase1_alg_states_form8_states.npy', mmap_mode='r')
n = len(tk)
P = np.zeros((n, st.shape[2]), np.float32)
L = tk.sum(1).astype(np.float32)
for s0 in range(0, n, 512):
    s1 = min(s0 + 512, n)
    stc = np.asarray(st[s0:s1]).astype(np.float32)
    m = tk[s0:s1][:, :, None]
    P[s0:s1] = (stc * m).sum(1) / np.maximum(m.sum(1), 1)
    if s0 % 8192 == 0: print(f"[pf] {s0}/{n}", flush=True)
np.save('.cache/probe_P_form8.npy', P)
np.save('.cache/probe_L_form8.npy', L)
print(f"[pf] saved: P {P.shape}, L mean {L.mean():.1f}", flush=True)
