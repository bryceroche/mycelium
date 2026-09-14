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
ONE MAP, TWO CHARTS (2026-09-14, word given; blog "One map, two
charts"): the atlas is ONE object — Atlas — with a chart per MEDIUM
("slot" = the commitment, the math loop's medium; "token" = the
reading, the NL loop's medium) under ONE class index and ONE era
stamp. A row is located on whichever chart the caller stands on and
the other chart is read at the same coordinate: Atlas.transport.
cross_prior is transport("token" -> "slot") at breath 0; consult is
the slot chart's nearest pages; the happy-family z-radius is
Atlas.radius on either chart. The npz layout is unchanged (means/
vars/counts = the slot chart; nl_means/nl_vars/nl_counts = the token
chart) so every mined atlas still loads, and the loaded Atlas keeps
the dict protocol (atlas["means"], atlas.get("nl_means")) so every
existing reader is untouched. AtlasBanks is the miner's accumulator:
one Welford bank per (medium, breath_step, class), one save.
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


MEDIA = ("slot", "token")     # the two mediums: the commitment and the reading
_KEYS = {"slot": ("means", "vars", "counts"),
         "token": ("nl_means", "nl_vars", "nl_counts")}   # the npz layout, unchanged


def _unit(x):
    x = np.asarray(x, np.float64)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


class Chart:
    """One medium's pages: means/vars (K_STEPS, C, D), counts (K_STEPS, C).
    A cell is LIVE when its count > 0; every read excludes dead cells."""

    def __init__(self, means, varis, counts):
        self.means = np.asarray(means); self.vars = np.asarray(varis)
        self.counts = np.asarray(counts)
        assert self.means.shape == self.vars.shape and \
            self.counts.shape == self.means.shape[:2], \
            (self.means.shape, self.vars.shape, self.counts.shape)

    @property
    def k_steps(self): return int(self.means.shape[0])

    def live(self, step):
        return self.counts[step] > 0

    def locate(self, step, states, k=1):
        """Nearest live cells at breath_step `step` by cosine (consult's
        metric). states (B, D) -> (class_idx (B, k), dist (B, k))."""
        live = self.live(step)
        if not live.any():
            raise RuntimeError(f"Chart.locate: page {step} is empty — never mined.")
        lidx = np.where(live)[0]
        sim = _unit(states) @ _unit(self.means[step][live]).T
        idx = np.argsort(-sim, axis=1)[:, :k]
        return lidx[idx], np.take_along_axis(1.0 - sim, idx, 1)

    def trajectory(self, class_idx):
        """The class's pages across breaths: int -> (K_STEPS, D);
        (B,) -> (B, K_STEPS, D)."""
        ci = np.asarray(class_idx)
        return (self.means[:, int(ci), :] if ci.ndim == 0
                else self.means[:, ci, :].transpose(1, 0, 2))

    def radius(self, step, states, class_idx=None):
        """THE HAPPY-FAMILY z-RADIUS (scripts/happy_family_read.py's
        formula, one call): ||s - mu|| / sqrt(mean var) to the nearest
        live cell (or to class_idx when given). Returns (z (B,), ci (B,))."""
        S = np.atleast_2d(np.asarray(states, np.float64))
        ci = (self.locate(step, S, 1)[0][:, 0] if class_idx is None
              else np.broadcast_to(np.asarray(class_idx), (S.shape[0],)))
        mu = self.means[step][ci].astype(np.float64)
        sd = np.sqrt(self.vars[step][ci].astype(np.float64).mean(-1)) + 1e-9
        return np.linalg.norm(S - mu, axis=1) / sd, ci


class Atlas:
    """ONE MAP, TWO CHARTS. charts: {"slot": Chart, "token": Chart}
    (a single-chart atlas has only "slot"); one class index; one era
    stamp. Keeps the dict protocol of the legacy loader so every
    existing reader (atlas["means"], atlas.get("nl_means")) reads
    the same arrays it always did."""

    def __init__(self, classes, stamp, charts, class_names=None):
        self.classes = [str(c) for c in classes]
        self.stamp = str(stamp)
        self.charts = dict(charts)
        self.class_names = list(class_names) if class_names is not None else list(self.classes)
        C = len(self.classes)
        for m, ch in self.charts.items():
            assert m in MEDIA, m
            assert ch.means.shape[1] == C, (m, ch.means.shape, C)

    # --- the charts ------------------------------------------------------
    def has(self, medium):
        return medium in self.charts

    def chart(self, medium):
        """THE LOUD DOOR for a missing chart: never a silent zero page."""
        if medium not in self.charts:
            raise RuntimeError(
                f"Atlas: no {medium!r} chart in this atlas (charts: "
                f"{sorted(self.charts)}) — re-mine with the paired miner; "
                f"refusing loudly, never a silent zero page.")
        return self.charts[medium]

    def class_index(self, name):
        return self.classes.index(name)

    # --- reads (CONDITIONING / DIAGNOSTIC only — Goodhart fence) ---------
    def locate(self, medium, step, states, k=1):
        return self.chart(medium).locate(step, states, k)

    def consult(self, medium, step, states, k=3):
        """Nearest pages for conditioning: (idx (B,k), dist (B,k),
        centroids (B,k,D)) — the legacy consult() on the named chart."""
        idx, dist = self.locate(medium, step, states, k)
        return idx, dist, self.chart(medium).means[step][idx]

    def transport(self, src, dst, step, states, return_traj=True):
        """THE CROSSING: locate `states` on the `src` chart at breath_step
        `step`, read the `dst` chart's trajectory at that coordinate.
        (D,) -> (class_idx, traj (K_STEPS, D)); (B, D) -> ((B,), (B, K, D)).
        transport("token", "slot", 0, x) is the cross-atlas prior."""
        x = np.asarray(states, np.float64)
        single = (x.ndim == 1)
        ci = self.locate(src, step, x[None] if single else x, 1)[0][:, 0]
        traj = self.chart(dst).trajectory(ci) if return_traj else None
        if single:
            return int(ci[0]), (traj[0] if traj is not None else None)
        return ci, traj

    def radius(self, medium, step, states, class_idx=None):
        return self.chart(medium).radius(step, states, class_idx)

    # --- the legacy dict protocol ----------------------------------------
    def _legacy(self):
        d = {"classes": list(self.classes), "stamp": self.stamp,
             "class_names": list(self.class_names)}
        for m in MEDIA:
            ch = self.charts.get(m)
            for key, arr in zip(_KEYS[m], (("means", "vars", "counts") if ch else (None,) * 3)):
                d[key] = getattr(ch, arr) if ch is not None else None
        return d

    def __getitem__(self, k): return self._legacy()[k]
    def get(self, k, default=None): return self._legacy().get(k, default)
    def keys(self): return self._legacy().keys()
    def __contains__(self, k): return k in self._legacy()

    # --- the file ----------------------------------------------------------
    @classmethod
    def load(cls, path=ATLAS_PATH, manifest_path=MANIFEST):
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
        charts = {}
        for m in MEDIA:
            km, kv, kc = _KEYS[m]
            if km in z.files:
                charts[m] = Chart(z[km], z[kv], z[kc])
        return cls(list(z["classes"]), stamp, charts,
                   class_names=(list(z["class_names"]) if "class_names" in z.files else None))


class AtlasBanks:
    """THE MINER'S ACCUMULATOR: one StepWelford per (medium, breath_step,
    class); one save writing one file with one class index (the union
    over both charts) and one stamp. The paired-count law lives here:
    a class mined on one chart must be mined on the other."""

    def __init__(self, dim, media=MEDIA):
        self.dim = int(dim); self.media = tuple(media)
        self.cells = {m: {} for m in self.media}

    def add(self, medium, step, cls, x):
        bank = self.cells[medium]
        key = (int(step), cls)
        if key not in bank:
            bank[key] = StepWelford(self.dim)
        bank[key].add(x)

    def count(self, medium, step, cls):
        w = self.cells[medium].get((int(step), cls))
        return w.n if w is not None else 0

    def classes(self):
        return sorted({c for m in self.media for (_, c) in self.cells[m]})

    def save(self, path=ATLAS_PATH, manifest_path=MANIFEST, class_names=None,
             paired=True):
        classes = self.classes()
        cid = {c: i for i, c in enumerate(classes)}
        C = len(classes)
        out = {}
        for m in self.media:
            if m != "slot" and not self.cells[m]:
                continue            # a single-chart atlas: no token keys written
            km, kv, kc = _KEYS[m]
            means = np.zeros((K_STEPS, C, self.dim), np.float32)
            varis = np.zeros((K_STEPS, C, self.dim), np.float32)
            counts = np.zeros((K_STEPS, C), np.int64)
            for (s, c), w in self.cells[m].items():
                means[s, cid[c]] = w.mean; varis[s, cid[c]] = w.var(); counts[s, cid[c]] = w.n
            out[km], out[kv], out[kc] = means, varis, counts
        if paired and "nl_counts" in out:
            assert (out["nl_counts"] == out["counts"]).all(), \
                "paired charts disagree on counts — a row was banked on one chart only"
        np.savez(path, classes=np.array([str(c) for c in classes]),
                 class_names=np.array([str((class_names or {}).get(c, c)) for c in classes]),
                 era_stamp=np.array(_era_stamp(manifest_path)),
                 k_steps=np.array(K_STEPS), **out)
        return path


def _era_stamp(manifest_path=MANIFEST):
    m = json.load(open(manifest_path))
    return os.path.basename(m["parser_ckpt"])


def save_atlas(cells, dim, path=ATLAS_PATH, manifest_path=MANIFEST,
               class_names=None, nl_cells=None):
    """LEGACY ENTRY (the miner's old signature): cells / nl_cells are
    {(step_id, class_id): StepWelford} for the slot / token charts.
    Writes through AtlasBanks — one file, two charts, one class index
    (the union), one stamp; layout unchanged."""
    banks = AtlasBanks(dim)
    banks.cells["slot"] = dict(cells)
    if nl_cells is not None:
        banks.cells["token"] = dict(nl_cells)
    return banks.save(path, manifest_path, class_names, paired=False)


def load_atlas(path=ATLAS_PATH, manifest_path=MANIFEST):
    """THE LOUD DOOR (Atlas.load): refuses an atlas whose era stamp does
    not match the manifest's deployed parser. Returns the Atlas, which
    keeps the legacy dict protocol (atlas["means"], atlas.get("nl_means"),
    None for a chart the file lacks)."""
    return Atlas.load(path, manifest_path)


def _as_atlas(atlas):
    """Legacy callers may hand a plain dict (tests, monkeypatches)."""
    if isinstance(atlas, Atlas):
        return atlas
    charts = {}
    for m in MEDIA:
        km, kv, kc = _KEYS[m]
        if atlas.get(km) is not None and atlas.get(kc) is not None:
            charts[m] = Chart(atlas[km], atlas.get(kv, np.zeros_like(atlas[km])), atlas[kc])
    return Atlas(atlas.get("classes", [str(i) for i in range(atlas["means"].shape[1])]),
                 atlas.get("stamp", ""), charts)


def consult(atlas, step_id, states, k=3):
    """Nearest SLOT-chart pages for conditioning: (idx (B,k), dist (B,k),
    centroids (B,k,D)); cosine on L2-normalized vectors; dead cells
    excluded. CONDITIONING ONLY — never a supervised target."""
    return _as_atlas(atlas).consult("slot", step_id, states, k)


def cross_prior(atlas, nl_state_b0, return_traj=True):
    """THE CROSS-ATLAS PRIOR (2026-09-05, registered) = Atlas.transport
    ("token" -> "slot") at breath 0: match a breath-0 NL state against
    the token chart's page 0 and return the matched class's MATH
    trajectory (retrieval, not oracle labels). (D,) -> (class_idx, traj
    (K_STEPS, D)); (B, D) -> ((B,), (B, K_STEPS, D)); return_traj=False
    skips the trajectories. CONDITIONING ONLY (Goodhart fence)."""
    a = _as_atlas(atlas)
    if not a.has("token"):
        raise RuntimeError(
            "cross_prior: this atlas has no NL chart — re-mine with"
            " the paired miner (single-chart atlases cannot serve"
            " the prior; refusing loudly, never a silent zero page).")
    if not a.chart("token").live(0).any():
        raise RuntimeError("cross_prior: NL page 0 is empty — the"
                           " breath-0 chart was never mined.")
    return a.transport("token", "slot", 0, nl_state_b0, return_traj)


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
    # ONE MAP, TWO CHARTS: the object's own API against the legacy reads
    assert isinstance(a, Atlas) and sorted(a.charts) == ["slot", "token"]
    assert a.chart("slot").means is a["means"] and a.chart("token").means is a["nl_means"]
    assert list(a.keys()) >= ["classes"] and "nl_counts" in a and a.get("nope") is None
    _i2, _d2, _c2 = a.consult("slot", 2, q, k=1)
    assert (_i2 == idx).all() and np.allclose(_d2, dist) and np.allclose(_c2, cents)
    _ct, _tt = a.transport("token", "slot", 0, _q0)
    assert _ct == _ci and np.allclose(_tt, _tr), "transport != cross_prior"
    _cs, _ts = a.transport("slot", "token", 2, q[0])       # the other way across
    assert a.classes[_cs] == "mul_chain" and np.allclose(_ts, a["nl_means"][:, _cs, :])
    # the radius: happy_family_read's formula, on either chart
    _z, _zc = a.radius("slot", 2, q)
    _mu = a["means"][2][_zc[0]]; _ref = np.linalg.norm(q[0] - _mu) / (np.sqrt(a["vars"][2][_zc[0]].mean()) + 1e-9)
    assert np.allclose(_z[0], _ref) and a.classes[_zc[0]] == "mul_chain"
    _zt, _ = a.radius("token", 0, _q0[None])
    assert _zt.shape == (1,) and _zt[0] < 4.0      # noise 0.05 over 16 dims at sd 0.1 -> z ~ 2
    # a single-chart atlas: the token chart's door is loud, the slot reads work
    _b1 = AtlasBanks(D); _b1.cells["slot"] = dict(cells)
    _mt2 = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    _j.dump({"parser_ckpt": ".cache/test_era.safetensors"}, _mt2); _mt2.close()
    _p1 = _b1.save(tempfile.mktemp(suffix=".npz"), _mt2.name)
    _a1 = load_atlas(_p1, manifest_path=_mt2.name)
    assert not _a1.has("token") and _a1["nl_means"] is None and _a1.get("nl_counts") is None
    try:
        _a1.chart("token"); raise AssertionError("missing chart served silently")
    except RuntimeError:
        pass
    assert _a1.consult("slot", 2, q, k=1)[0][0, 0] == idx[0, 0]
    # AtlasBanks: the paired-count law and the legacy file layout
    _bk = AtlasBanks(D)
    for (st, c), w in cells.items():
        for _ in range(3): _bk.add("slot", st, c, w.mean); _bk.add("token", st, c, w.mean + 1)
    _p2 = _bk.save(tempfile.mktemp(suffix=".npz"), _mt2.name)
    _z2 = np.load(_p2)
    assert set(_z2.files) == {"means", "vars", "counts", "nl_means", "nl_vars", "nl_counts",
                              "classes", "class_names", "era_stamp", "k_steps"}, _z2.files
    _bk.add("token", 0, "mul_chain", np.zeros(D))
    try:
        _bk.save(tempfile.mktemp(suffix=".npz"), _mt2.name); raise AssertionError("unpaired counts saved")
    except AssertionError as e:
        assert "paired" in str(e)
    for _f in (_mt2.name, _p1, _p2):
        try: os.remove(_f)
        except OSError: pass
    print("[step_atlas] self-test PASS: Welford banks, stamped save/"
          "load, loud door refuses cross-era, consult finds the kind, "
          "atlas_class buckets agree, SECOND CHART round-trips, "
          "cross_prior retrieves the paired math trajectory, needle "
          "hears the designed +1 lead; ONE MAP TWO CHARTS: Atlas API == "
          "legacy reads, transport both ways, radius = happy-family, "
          "single-chart door loud, AtlasBanks paired-count law + layout")
