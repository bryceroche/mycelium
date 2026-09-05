"""atlas_collapse_check.py — THE COLLAPSE CHECK (2026-09-05, Gemini's
safeguard, adopted measure-first). If per-breath attention is diffuse
or stopword-locked, the seven NL pages degenerate to near-identical
vectors and the trajectory is an illusion of pooling. This check runs
on the mined npz (CPU, zero GPU): per class, PAGE SPREAD = mean
pairwise cosine distance among its 7 pages, for BOTH charts. PINNED
WARNING BAR (ledger 2026-09-05): NL spread < 10% of the same class's
MATH spread => "NL-COLLAPSE WARNING" printed loudly (diagnostic only —
no gate, no filter; if warnings fire, the stopword-masking design
question opens WITH data in hand). Env: ACC_ATLAS (default
.cache/step_atlas_current.npz).
"""
import os
import numpy as np

z = np.load(os.environ.get("ACC_ATLAS", ".cache/step_atlas_current.npz"),
            allow_pickle=False)
classes = [str(c) for c in z["classes"]]


def spread(bank, counts, ci):
    live = counts[:, ci] > 0
    P = bank[live, ci]                      # (k_live, D)
    if len(P) < 2:
        return None
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
    sim = Pn @ Pn.T
    iu = np.triu_indices(len(P), 1)
    return float((1.0 - sim[iu]).mean())


print(f"[collapse-check] {os.environ.get('ACC_ATLAS', '.cache/step_atlas_current.npz')}"
      f" stamp={z['era_stamp']}")
warn = 0
for ci, c in enumerate(classes):
    ms = spread(z["means"], z["counts"], ci)
    ns = (spread(z["nl_means"], z["nl_counts"], ci)
          if "nl_means" in z.files else None)
    if ms is None:
        continue
    line = f"  {c:12s} math-spread {ms:.4f}"
    if ns is not None:
        ratio = ns / max(ms, 1e-9)
        line += f"  nl-spread {ns:.4f}  ratio {ratio:.2f}"
        if ratio < 0.10:
            line += "  <<< NL-COLLAPSE WARNING (pinned bar 0.10)"
            warn += 1
    print(line)
print(f"[collapse-check] {'CLEAN — the trajectory is real'
      if warn == 0 else f'{warn} class(es) warned — the stopword question opens, with data'}")
