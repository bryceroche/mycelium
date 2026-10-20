"""complex_unbind_codes.py -- THE CODE-OVERLAP READ (2026-10-20; Bryce's word on Opus's 'learned writing beat the oracle').
Zero-training read on saved PH_learn checkpoints. QUESTION: did PH_learn beat PH_oracle (+0.067 held-out hops 2-3) because
training SPREADS the entity codes further apart than random unit phasors (less crosstalk), or does the gain live
elsewhere (lookup, snap)? For each seed: the learned codes vs their own init (PH_learn starts from PH_oracle's random
phasors, generator 2000+seed):
  mean / max off-diagonal |<e_i, e_j>| / (|e_i| |e_j|)  (Hermitian, over the 40 entities)
  per-plane modulus spread (std/mean of |e|; 0 = unit phasors)
  the ANALYTIC CEILING (complex_unbind_ceiling.py; exact lookup, exact unbind, cosine snap) with the learned codes vs the
  init codes, held-out hops 2-3, read at K=4 facts per record (at K=3 the 96-plane ceiling sits at 0.99: no room).
BARS (pinned 10-20, ledger): CODES SPREAD iff (seed-mean) learned mean|cos| <= 0.8 x init's AND learned ceiling >= init
ceiling + 0.01; NOT SPREAD iff learned mean|cos| >= 0.95 x init's; else PARTIAL.
Usage: .venv/bin/python3 scripts/toys/complex_unbind_codes.py <ckpt prefix, e.g. .cache/complex_unbind_machine_depth>
"""
import os, re, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
pre = sys.argv[1]; MP = 96; out = open(f"{pre}_codes.txt", "x")


def say(s):
    print(s, flush=True); out.write(s + "\n"); out.flush()


def geom(z):
    e = z[:, :MP] + 1j * z[:, MP:]; n = np.linalg.norm(e, axis=1)
    c = np.abs(np.conj(e) @ e.T) / np.outer(n, n); off = c[~np.eye(len(e), dtype=bool)]
    m = np.abs(e); return off.mean(), off.max(), float((m.std(1) / m.mean(1)).mean())


def ceiling(z, tag):
    path = f"{pre}_codes_{tag}.npz"; np.savez(path, ent=z)
    r = subprocess.run([sys.executable, "scripts/toys/complex_unbind_ceiling.py"], capture_output=True, text=True, check=True,
                       env={**os.environ, "CEIL_CODES": path, "OMP_NUM_THREADS": "1", "UT_K": "4"}).stdout
    h = dict(re.findall(r"(h\d) ([\d.]+)", r.splitlines()[1])); return (float(h["h2"]) + float(h["h3"])) / 2


rows = []
for seed in (0, 1, 2):
    g = torch.Generator().manual_seed(2000 + seed); te = torch.rand(40, MP, generator=g) * 2 * np.pi
    init = torch.cat([torch.cos(te), torch.sin(te)], -1).numpy()
    learned = torch.load(f"{pre}_PH_learn_s{seed}.pt")["ent"].numpy()
    gi, gl = geom(init), geom(learned); ci, cl = ceiling(init, f"init{seed}"), ceiling(learned, f"learn{seed}")
    rows.append((gi[0], gl[0], ci, cl))
    say(f"seed {seed}: mean|cos| init {gi[0]:.4f} learned {gl[0]:.4f} | max|cos| init {gi[1]:.4f} learned {gl[1]:.4f} | "
        f"modulus spread init {gi[2]:.3f} learned {gl[2]:.3f} | ceiling held-out h2-3 init {ci:.4f} learned {cl:.4f}")
mi, ml, ci, cl = np.mean(rows, 0)
v = ("CODES SPREAD" if ml <= 0.8 * mi and cl >= ci + 0.01 else "NOT SPREAD" if ml >= 0.95 * mi else "PARTIAL")
say(f"MEAN: mean|cos| init {mi:.4f} learned {ml:.4f} (ratio {ml / mi:.3f}) | ceiling init {ci:.4f} learned {cl:.4f} "
    f"({cl - ci:+.4f}) -> VERDICT (pinned 10-20): {v}")
