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
THE PAIRED ATLAS (2026-09-05, registered): the same npz may carry a
SECOND chart — nl_means/nl_vars/nl_counts, the per-breath
attention-pooled waisted token-state pages (the READING) beside the
slot-state pages (the COMMITMENT). Same (K_STEPS, C, D) shape, same
classes index by construction, same stamp, same loud door.
cross_prior matches a breath-0 NL state to its kind and returns the
kind's math trajectory (retrieval, not oracle labels);
leadlag_needle is the alternation instrument (positive lag =
reading leads commitment). Both are CONDITIONING/DIAGNOSTIC only —
never a supervised target (Goodhart fence).
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


def atlas_class(gen):
    """THE ATLAS CLASS LABELER (2026-09-05, mask-head round 2) — the
    SINGLE SOURCE for a row's atlas class (meter-divergence law: the
    miner and every feed-time consumer call THIS function, never a
    reimplementation). gen = row.get("gen") ({} / None safe).
      gen.ladder present      -> "ladder2-4" / "ladder5-8" / "ladder9+"
                                 (depth buckets; teeth costume IGNORED)
      gen.src == "gsm8k"      -> "wild_gsm8k"
      gen.src == "r7"         -> "wild_r7"
      anything else           -> "mint_form8"
    """
    g = gen if isinstance(gen, dict) else {}   # some rows carry gen as a string tag
    if "ladder" in g:
        d = int(g["ladder"])
        return ("ladder2-4" if d <= 4
                else "ladder5-8" if d <= 8 else "ladder9+")
    if g.get("src") == "gsm8k":
        return "wild_gsm8k"
    if g.get("src") == "r7":
        return "wild_r7"
    return "mint_form8"


def _era_stamp(manifest_path=MANIFEST):
    m = json.load(open(manifest_path))
    return os.path.basename(m["parser_ckpt"])


def save_atlas(cells, dim, path=ATLAS_PATH, manifest_path=MANIFEST,
               class_names=None, nl_cells=None):
    """cells: {(step_id, class_id): StepWelford}. Writes the runtime
    npz: per-step centroid banks + counts + vars + the era stamp.
    nl_cells (optional, same key scheme): the SECOND chart — one
    atlas file, two charts, one classes index (the union; a class
    absent from one bank gets zero pages + zero counts there)."""
    classes = sorted({c for (_, c) in cells}
                     | {c for (_, c) in (nl_cells or {})})
    cid = {c: i for i, c in enumerate(classes)}
    C = len(classes)
    means = np.zeros((K_STEPS, C, dim), np.float32)
    varis = np.zeros((K_STEPS, C, dim), np.float32)
    counts = np.zeros((K_STEPS, C), np.int64)
    for (s, c), w in cells.items():
        means[s, cid[c]] = w.mean
        varis[s, cid[c]] = w.var()
        counts[s, cid[c]] = w.n
    extra = {}
    if nl_cells is not None:
        # THE SECOND CHART: same shape, same class index, same stamp
        nlm = np.zeros((K_STEPS, C, dim), np.float32)
        nlv = np.zeros((K_STEPS, C, dim), np.float32)
        nlc = np.zeros((K_STEPS, C), np.int64)
        for (st, c), w in nl_cells.items():
            nlm[st, cid[c]] = w.mean
            nlv[st, cid[c]] = w.var()
            nlc[st, cid[c]] = w.n
        extra = {'nl_means': nlm, 'nl_vars': nlv, 'nl_counts': nlc}
    np.savez(path, means=means, vars=varis, counts=counts,
             classes=np.array([str(c) for c in classes]),
             class_names=np.array([str((class_names or {}).get(c, c))
                                   for c in classes]),
             era_stamp=np.array(_era_stamp(manifest_path)),
             k_steps=np.array(K_STEPS), **extra)
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
            "stamp": stamp,
            # the paired chart (None on a single-chart atlas —
            # consumers that NEED it must check loudly)
            "nl_means": z["nl_means"] if "nl_means" in z.files else None,
            "nl_vars": z["nl_vars"] if "nl_vars" in z.files else None,
            "nl_counts": (z["nl_counts"] if "nl_counts" in z.files
                          else None)}


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


def cross_prior(atlas, nl_state_b0, return_traj=True):
    """THE CROSS-ATLAS PRIOR (2026-09-05, registered): match a
    breath-0 NL state against the NL chart's page 0 (cosine, live
    cells only — consult's metric) and return the matched class's
    MATH trajectory. The two charts share the class index by
    construction; transport = the parse. Retrieval instead of
    oracle labels (gen-label feed = oracle upper bound, training
    scaffolding; THIS is the deployable path).
    nl_state_b0: (D,) one item -> (class_idx, traj (K_STEPS, D));
                 (B, D) batch  -> (idx (B,), traj (B, K_STEPS, D)).
    return_traj=False skips materializing trajectories (corpus-
    scale callers index their own page table by class_idx).
    CONDITIONING ONLY — never a supervised target (Goodhart
    fence)."""
    nlm, nlc = atlas.get("nl_means"), atlas.get("nl_counts")
    if nlm is None or nlc is None:
        raise RuntimeError(
            "cross_prior: this atlas has no NL chart — re-mine with"
            " the paired miner (single-chart atlases cannot serve"
            " the prior; refusing loudly, never a silent zero page).")
    live = nlc[0] > 0
    if not live.any():
        raise RuntimeError("cross_prior: NL page 0 is empty — the"
                           " breath-0 chart was never mined.")
    x = np.asarray(nl_state_b0, np.float64)
    single = (x.ndim == 1)
    X = x[None] if single else x
    Ml = nlm[0][live].astype(np.float64)
    ln = Ml / (np.linalg.norm(Ml, axis=1, keepdims=True) + 1e-9)
    xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    lidx = np.where(live)[0]
    ci = lidx[(xn @ ln.T).argmax(1)]
    traj = None
    if return_traj:
        traj = atlas["means"][:, ci, :].transpose(1, 0, 2)
    if single:
        return int(ci[0]), (traj[0] if traj is not None else None)
    return ci, traj


def leadlag_needle(nl_diffs, math_diffs, max_lag=3):
    """THE LEAD/LAG NEEDLE (2026-09-05, registered): Pearson
    cross-correlation of paired per-breath DIFFERENCE curves
    (first differences of the contraction curves — events, not
    levels; levels co-contract trivially) at lags -max_lag..+max_lag.
    nl_diffs, math_diffs: (N_items, M) aligned arrays. At lag L the
    pairs are (nl[k], math[k+L]) pooled over items and valid k —
    POSITIVE lag = the reading's drop precedes the commitment's
    (reading LEADS commitment). Returns (lags (2*max_lag+1,),
    corrs (same,)); degenerate lags (no variance / too few pairs)
    are NaN, never faked. DIAGNOSTIC REGISTER ONLY — this number
    must never enter a loss (Goodhart fence: a supervised needle
    teaches the clock to lie)."""
    nl = np.asarray(nl_diffs, np.float64)
    mt = np.asarray(math_diffs, np.float64)
    assert nl.shape == mt.shape and nl.ndim == 2, (nl.shape, mt.shape)
    M = nl.shape[1]
    lags = np.arange(-max_lag, max_lag + 1)
    corrs = np.full(len(lags), np.nan)
    for li, lag in enumerate(lags):
        lag = int(lag)
        if abs(lag) >= M:
            continue
        if lag >= 0:
            a = nl[:, :M - lag].ravel()
            b = mt[:, lag:].ravel()
        else:
            a = nl[:, -lag:].ravel()
            b = mt[:, :M + lag].ravel()
        if len(a) >= 2 and a.std() > 1e-12 and b.std() > 1e-12:
            corrs[li] = float(np.corrcoef(a, b)[0, 1])
    return lags, corrs


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
    # PAIRED: NL cells — same classes, own drift, separable pages
    cells_nl = {}
    for s in range(K_STEPS):
        for c in ("mul_chain", "add_ladder"):
            w = StepWelford(D)
            base = (1.0 if c == "mul_chain" else -1.0) * (s + 2)
            for _ in range(50):
                w.add(base + rng.standard_normal(D) * 0.1)
            cells_nl[(s, c)] = w
    import tempfile, json as _j
    mtmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    _j.dump({"parser_ckpt": ".cache/test_era.safetensors"}, mtmp)
    mtmp.close()
    p = save_atlas(cells, D, path=tempfile.mktemp(suffix=".npz"),
                   manifest_path=mtmp.name, nl_cells=cells_nl)
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
    finally:
        for _f in (mtmp.name, p, m2.name):     # no temp litter per run
            try: os.remove(_f)
            except OSError: pass
    # the labeler: one organ, every caller (miner + feeds)
    assert atlas_class({"ladder": 3}) == "ladder2-4"
    assert atlas_class({"ladder": 5, "teeth": 0.4}) == "ladder5-8"
    assert atlas_class({"ladder": 12}) == "ladder9+"
    assert atlas_class({"src": "gsm8k", "src_idx": 7}) == "wild_gsm8k"
    assert atlas_class({"src": "r7"}) == "wild_r7"
    assert atlas_class(None) == "mint_form8"
    assert atlas_class({"shape": "x"}) == "mint_form8"
    # THE SECOND CHART: shape/count round trip
    assert a["nl_means"] is not None and \
        a["nl_means"].shape == a["means"].shape
    assert (a["nl_counts"] == a["counts"]).all()
    # THE CROSS-ATLAS PRIOR: a noisy add_ladder breath-0 NL state
    # must retrieve add_ladder's MATH trajectory (round trip)
    _ai = a["classes"].index("add_ladder")
    _q0 = a["nl_means"][0, _ai] + rng.standard_normal(D) * 0.05
    _ci, _tr = cross_prior(a, _q0)
    assert a["classes"][_ci] == "add_ladder", "cross_prior miss"
    assert _tr.shape == (K_STEPS, D)
    assert np.allclose(_tr, a["means"][:, _ci, :]), \
        "cross_prior returned the wrong chart's trajectory"
    _cib, _trb = cross_prior(a, np.stack([_q0, a["nl_means"][0, 1 - _ai]]))
    assert _cib.shape == (2,) and _trb.shape == (2, K_STEPS, D)
    _cin, _trn = cross_prior(a, _q0, return_traj=False)
    assert _cin == _ci and _trn is None
    try:
        cross_prior({"means": a["means"], "nl_means": None,
                     "nl_counts": None}, _q0)
        raise AssertionError("single-chart atlas served the prior")
    except RuntimeError:
        pass
    # THE NEEDLE on synthetic drifting data: math events are the NL
    # events shifted one breath LATER -> the needle must peak at +1
    _ev = rng.standard_normal((64, K_STEPS - 1))
    _nd = _ev.copy()
    _md = np.concatenate(
        [rng.standard_normal((64, 1)) * 0.05, _ev[:, :-1]], axis=1)
    _lags, _corrs = leadlag_needle(_nd, _md, max_lag=3)
    assert int(_lags[np.nanargmax(_corrs)]) == 1, \
        (f"needle peak at {int(_lags[np.nanargmax(_corrs)])}, "
         f"designed +1")
    assert _corrs[list(_lags).index(1)] > 0.9
    assert abs(_corrs[list(_lags).index(0)]) < 0.3
    print("[step_atlas] self-test PASS: Welford banks, stamped save/"
          "load, loud door refuses cross-era, consult finds the kind, "
          "atlas_class buckets agree, SECOND CHART round-trips, "
          "cross_prior retrieves the paired math trajectory, needle "
          "hears the designed +1 lead")
