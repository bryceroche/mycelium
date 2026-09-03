"""step_atlas.py — THE PER-STEP ATLAS store (2026-09-03, word given).
CPU plumbing for the registered design (ledger 2026-09-02): the math-
operation atlas as SEVEN per-breath_step centroid banks, current era
only, keyed (breath_step_id, centroid_id). The era survives as a
STAMP asserted against the manifest at load — a stale atlas in fresh
coordinates is the never-mix-generations law's doorway, so the door
is loud. Centroids are maintained with Welford (mean + M2 + count);
the consult helper returns nearest pages for CONDITIONING only —
never a loss target (Goodhart fence).
The miner (GPU, fires after the current repair verdicts name the
incumbent whose coordinates this atlas anchors to) imports
StepWelford and save_atlas; the read engine imports load_atlas and
consult.
"""
import json
import os

import numpy as np

K_STEPS = 7          # breath_steps per cycle (step 0 = intake)
ATLAS_PATH = ".cache/step_atlas_current.npz"
MANIFEST = ".cache/GENERATION.json"


class StepWelford:
    """Welford accumulator for one (step, class) cell."""

    def __init__(self, dim):
        self.n = 0
        self.mean = np.zeros(dim, np.float64)
        self.M2 = np.zeros(dim, np.float64)

    def add(self, x):
        x = np.asarray(x, np.float64)
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.M2 += d * (x - self.mean)

    def var(self):
        return self.M2 / max(self.n - 1, 1)


def _era_stamp(manifest_path=MANIFEST):
    m = json.load(open(manifest_path))
    return os.path.basename(m["parser_ckpt"])


def save_atlas(cells, dim, path=ATLAS_PATH, manifest_path=MANIFEST,
               class_names=None):
    """cells: {(step_id, class_id): StepWelford}. Writes the runtime
    npz: per-step centroid banks + counts + vars + the era stamp."""
    classes = sorted({c for (_, c) in cells})
    cid = {c: i for i, c in enumerate(classes)}
    C = len(classes)
    means = np.zeros((K_STEPS, C, dim), np.float32)
    varis = np.zeros((K_STEPS, C, dim), np.float32)
    counts = np.zeros((K_STEPS, C), np.int64)
    for (s, c), w in cells.items():
        means[s, cid[c]] = w.mean
        varis[s, cid[c]] = w.var()
        counts[s, cid[c]] = w.n
    np.savez(path, means=means, vars=varis, counts=counts,
             classes=np.array([str(c) for c in classes]),
             class_names=np.array([str((class_names or {}).get(c, c))
                                   for c in classes]),
             era_stamp=np.array(_era_stamp(manifest_path)),
             k_steps=np.array(K_STEPS))
    return path


def load_atlas(path=ATLAS_PATH, manifest_path=MANIFEST):
    """THE LOUD DOOR: refuses an atlas whose era stamp does not match
    the manifest's deployed parser — no silent cross-era reads, ever."""
    z = np.load(path, allow_pickle=False)
    stamp = str(z["era_stamp"])
    cur = _era_stamp(manifest_path)
    if stamp != cur:
        raise RuntimeError(
            f"step_atlas: era stamp {stamp!r} != deployed parser {cur!r} "
            f"— stale atlas in fresh coordinates (never-mix-generations "
            f"law). Re-mine before reading; refusing to serve.")
    return {"means": z["means"], "vars": z["vars"],
            "counts": z["counts"], "classes": list(z["classes"]),
            "stamp": stamp}


def consult(atlas, step_id, states, k=3):
    """Nearest atlas pages for conditioning. states: (B, D) numpy for
    breath_step step_id. Returns (idx (B,k), dist (B,k), centroids
    (B,k,D)). Cosine on L2-normalized vectors; empty cells (count 0)
    excluded. CONDITIONING ONLY — never a supervised target."""
    M = atlas["means"][step_id]
    live = atlas["counts"][step_id] > 0
    Ml = M[live]
    ln = Ml / (np.linalg.norm(Ml, axis=1, keepdims=True) + 1e-9)
    sn = states / (np.linalg.norm(states, axis=1, keepdims=True) + 1e-9)
    sim = sn @ ln.T
    idx = np.argsort(-sim, axis=1)[:, :k]
    lidx = np.where(live)[0]
    return (lidx[idx],
            np.take_along_axis(1.0 - sim, idx, 1),
            Ml[idx])


if __name__ == "__main__":
    # CPU self-test on synthetic data: two classes, drifting per step
    rng = np.random.default_rng(7)
    cells = {}
    D = 16
    for s in range(K_STEPS):
        for c in ("mul_chain", "add_ladder"):
            w = StepWelford(D)
            base = (1.0 if c == "mul_chain" else -1.0) * (s + 1)
            for _ in range(50):
                w.add(base + rng.standard_normal(D) * 0.1)
            cells[(s, c)] = w
    import tempfile, json as _j
    mtmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    _j.dump({"parser_ckpt": ".cache/test_era.safetensors"}, mtmp)
    mtmp.close()
    p = save_atlas(cells, D, path=tempfile.mktemp(suffix=".npz"),
                   manifest_path=mtmp.name)
    a = load_atlas(p, manifest_path=mtmp.name)
    q = np.full((2, D), 3.0)          # near mul_chain at step 2
    idx, dist, cents = consult(a, 2, q, k=1)
    assert a["classes"][idx[0, 0]] == "mul_chain", "consult miss"
    # the loud door: wrong-era manifest must refuse
    m2 = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    _j.dump({"parser_ckpt": ".cache/OTHER_era.safetensors"}, m2)
    m2.close()
    try:
        load_atlas(p, manifest_path=m2.name)
        raise AssertionError("stale atlas served — door failed")
    except RuntimeError:
        pass
    print("[step_atlas] self-test PASS: Welford banks, stamped save/"
          "load, loud door refuses cross-era, consult finds the kind")
