"""apply_paired_atlas.py — THE PAIRED ATLAS, chart-side (2026-09-05,
ledger "REGISTERED + WORD GIVEN: THE PAIRED ATLAS"). STAGED patch to
mycelium/step_atlas.py (the running mask-head round-2b chain's pending
loop_val legs import this module — running-invocation law: the chain we
stage applies this after round-2b exits; nothing is touched live).

WHAT IT ADDS (all additive; every existing caller byte-compatible):

1. THE SECOND CHART. save_atlas gains nl_cells=None — a second Welford
   bank written into the SAME npz as nl_means/nl_vars/nl_counts, same
   (K_STEPS, C, D) shape, same classes array BY CONSTRUCTION (the class
   index is computed once, over the union of both banks' keys), same
   era stamp, same loud door. One atlas file, two charts: the math
   chart (slot-state pages — what the organ COMMITS) and the NL chart
   (attention-pooled waisted token-state pages — what the organ READS).
   nl_cells=None (every existing caller) writes exactly today's npz.

2. THE CROSS-ATLAS PRIOR. cross_prior(atlas, nl_state_b0): match a
   breath-0 NL state against nl_means[0] (cosine, live cells only —
   consult's own metric) and return the matched class's MATH
   trajectory (K_STEPS, D). The two charts share the kind index by
   construction; transport = the parse. Retrieval instead of oracle
   labels — the honest deployable path (gen-label feed is the oracle
   upper bound, training scaffolding). CONDITIONING ONLY — never a
   loss target (Goodhart fence, same clause as consult).

3. THE LEAD/LAG NEEDLE's math. leadlag_needle(nl_diffs, math_diffs):
   Pearson cross-correlation of paired per-breath difference-curves at
   lags -3..+3; POSITIVE lag = the reading's contraction PRECEDES the
   commitment's (reading leads commitment). Pure numpy, self-tested on
   synthetic drifting data; the GPU reader (scripts/leadlag_read.py)
   imports THIS function (meter-divergence law: one needle, every
   caller).

4. Self-test extended: second-bank round trip, cross_prior retrieval,
   needle peak on a designed +1-lag drift.

--check: builds the would-be result, ast-parses it, writes NOTHING.
"""
import ast
import sys

CHECK = '--check' in sys.argv
ANCHORS = []


def note(desc):
    ANCHORS.append(desc)


def sub(s, old, new, n=1, desc=""):
    assert s.count(old) == n, \
        f"anchor MISSING/NOT-UNIQUE (want {n}, have {s.count(old)}): {desc}"
    note(f"{desc} (x{n})")
    return s.replace(old, new, n)


fn = 'mycelium/step_atlas.py'
s = open(fn).read()
n_lines0 = s.count('\n')

assert 'nl_means' not in s and 'cross_prior' not in s, \
    "paired-atlas chart already present — refuse (idempotence guard)"

# --- 1. docstring: the paired-atlas clause ---------------------------------
s = sub(s,
        'The miner (GPU, fires after the current repair verdicts name the\n'
        'incumbent whose coordinates this atlas anchors to) imports\n'
        'StepWelford and save_atlas; the read engine imports load_atlas and\n'
        'consult.\n'
        '"""\n',
        'The miner (GPU, fires after the current repair verdicts name the\n'
        'incumbent whose coordinates this atlas anchors to) imports\n'
        'StepWelford and save_atlas; the read engine imports load_atlas and\n'
        'consult.\n'
        'THE PAIRED ATLAS (2026-09-05, registered): the same npz may carry a\n'
        'SECOND chart — nl_means/nl_vars/nl_counts, the per-breath\n'
        'attention-pooled waisted token-state pages (the READING) beside the\n'
        'slot-state pages (the COMMITMENT). Same (K_STEPS, C, D) shape, same\n'
        'classes index by construction, same stamp, same loud door.\n'
        'cross_prior matches a breath-0 NL state to its kind and returns the\n'
        'kind\'s math trajectory (retrieval, not oracle labels);\n'
        'leadlag_needle is the alternation instrument (positive lag =\n'
        'reading leads commitment). Both are CONDITIONING/DIAGNOSTIC only —\n'
        'never a supervised target (Goodhart fence).\n'
        '"""\n',
        1, "docstring gains the paired-atlas clause")

# --- 2. save_atlas: the second bank ----------------------------------------
s = sub(s,
        'def save_atlas(cells, dim, path=ATLAS_PATH, manifest_path=MANIFEST,\n'
        '               class_names=None):\n'
        '    """cells: {(step_id, class_id): StepWelford}. Writes the runtime\n'
        '    npz: per-step centroid banks + counts + vars + the era stamp."""\n'
        '    classes = sorted({c for (_, c) in cells})\n',
        'def save_atlas(cells, dim, path=ATLAS_PATH, manifest_path=MANIFEST,\n'
        '               class_names=None, nl_cells=None):\n'
        '    """cells: {(step_id, class_id): StepWelford}. Writes the runtime\n'
        '    npz: per-step centroid banks + counts + vars + the era stamp.\n'
        '    nl_cells (optional, same key scheme): the SECOND chart — one\n'
        '    atlas file, two charts, one classes index (the union; a class\n'
        '    absent from one bank gets zero pages + zero counts there)."""\n'
        '    classes = sorted({c for (_, c) in cells}\n'
        '                     | {c for (_, c) in (nl_cells or {})})\n',
        1, "save_atlas signature + union class index")

s = sub(s,
        "    np.savez(path, means=means, vars=varis, counts=counts,\n"
        "             classes=np.array([str(c) for c in classes]),\n"
        "             class_names=np.array([str((class_names or {}).get(c, c))\n"
        "                                   for c in classes]),\n"
        "             era_stamp=np.array(_era_stamp(manifest_path)),\n"
        "             k_steps=np.array(K_STEPS))\n"
        "    return path\n",
        "    extra = {}\n"
        "    if nl_cells is not None:\n"
        "        # THE SECOND CHART: same shape, same class index, same stamp\n"
        "        nlm = np.zeros((K_STEPS, C, dim), np.float32)\n"
        "        nlv = np.zeros((K_STEPS, C, dim), np.float32)\n"
        "        nlc = np.zeros((K_STEPS, C), np.int64)\n"
        "        for (st, c), w in nl_cells.items():\n"
        "            nlm[st, cid[c]] = w.mean\n"
        "            nlv[st, cid[c]] = w.var()\n"
        "            nlc[st, cid[c]] = w.n\n"
        "        extra = {'nl_means': nlm, 'nl_vars': nlv, 'nl_counts': nlc}\n"
        "    np.savez(path, means=means, vars=varis, counts=counts,\n"
        "             classes=np.array([str(c) for c in classes]),\n"
        "             class_names=np.array([str((class_names or {}).get(c, c))\n"
        "                                   for c in classes]),\n"
        "             era_stamp=np.array(_era_stamp(manifest_path)),\n"
        "             k_steps=np.array(K_STEPS), **extra)\n"
        "    return path\n",
        1, "save_atlas writes nl bank when given")

# --- 3. load_atlas: hand back the second chart when present ----------------
s = sub(s,
        '    return {"means": z["means"], "vars": z["vars"],\n'
        '            "counts": z["counts"], "classes": list(z["classes"]),\n'
        '            "stamp": stamp}\n',
        '    return {"means": z["means"], "vars": z["vars"],\n'
        '            "counts": z["counts"], "classes": list(z["classes"]),\n'
        '            "stamp": stamp,\n'
        '            # the paired chart (None on a single-chart atlas —\n'
        '            # consumers that NEED it must check loudly)\n'
        '            "nl_means": z["nl_means"] if "nl_means" in z.files else None,\n'
        '            "nl_vars": z["nl_vars"] if "nl_vars" in z.files else None,\n'
        '            "nl_counts": (z["nl_counts"] if "nl_counts" in z.files\n'
        '                          else None)}\n',
        1, "load_atlas returns nl chart (None-safe for old files)")

# --- 4. cross_prior + leadlag_needle ---------------------------------------
s = sub(s,
        'if __name__ == "__main__":\n'
        '    # CPU self-test on synthetic data: two classes, drifting per step\n',
        'def cross_prior(atlas, nl_state_b0, return_traj=True):\n'
        '    """THE CROSS-ATLAS PRIOR (2026-09-05, registered): match a\n'
        '    breath-0 NL state against the NL chart\'s page 0 (cosine, live\n'
        '    cells only — consult\'s metric) and return the matched class\'s\n'
        '    MATH trajectory. The two charts share the class index by\n'
        '    construction; transport = the parse. Retrieval instead of\n'
        '    oracle labels (gen-label feed = oracle upper bound, training\n'
        '    scaffolding; THIS is the deployable path).\n'
        '    nl_state_b0: (D,) one item -> (class_idx, traj (K_STEPS, D));\n'
        '                 (B, D) batch  -> (idx (B,), traj (B, K_STEPS, D)).\n'
        '    return_traj=False skips materializing trajectories (corpus-\n'
        '    scale callers index their own page table by class_idx).\n'
        '    CONDITIONING ONLY — never a supervised target (Goodhart\n'
        '    fence)."""\n'
        '    nlm, nlc = atlas.get("nl_means"), atlas.get("nl_counts")\n'
        '    if nlm is None or nlc is None:\n'
        '        raise RuntimeError(\n'
        '            "cross_prior: this atlas has no NL chart — re-mine with"\n'
        '            " the paired miner (single-chart atlases cannot serve"\n'
        '            " the prior; refusing loudly, never a silent zero page).")\n'
        '    live = nlc[0] > 0\n'
        '    if not live.any():\n'
        '        raise RuntimeError("cross_prior: NL page 0 is empty — the"\n'
        '                           " breath-0 chart was never mined.")\n'
        '    x = np.asarray(nl_state_b0, np.float64)\n'
        '    single = (x.ndim == 1)\n'
        '    X = x[None] if single else x\n'
        '    Ml = nlm[0][live].astype(np.float64)\n'
        '    ln = Ml / (np.linalg.norm(Ml, axis=1, keepdims=True) + 1e-9)\n'
        '    xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)\n'
        '    lidx = np.where(live)[0]\n'
        '    ci = lidx[(xn @ ln.T).argmax(1)]\n'
        '    traj = None\n'
        '    if return_traj:\n'
        '        traj = atlas["means"][:, ci, :].transpose(1, 0, 2)\n'
        '    if single:\n'
        '        return int(ci[0]), (traj[0] if traj is not None else None)\n'
        '    return ci, traj\n'
        '\n'
        '\n'
        'def leadlag_needle(nl_diffs, math_diffs, max_lag=3):\n'
        '    """THE LEAD/LAG NEEDLE (2026-09-05, registered): Pearson\n'
        '    cross-correlation of paired per-breath DIFFERENCE curves\n'
        '    (first differences of the contraction curves — events, not\n'
        '    levels; levels co-contract trivially) at lags -max_lag..+max_lag.\n'
        '    nl_diffs, math_diffs: (N_items, M) aligned arrays. At lag L the\n'
        '    pairs are (nl[k], math[k+L]) pooled over items and valid k —\n'
        '    POSITIVE lag = the reading\'s drop precedes the commitment\'s\n'
        '    (reading LEADS commitment). Returns (lags (2*max_lag+1,),\n'
        '    corrs (same,)); degenerate lags (no variance / too few pairs)\n'
        '    are NaN, never faked. DIAGNOSTIC REGISTER ONLY — this number\n'
        '    must never enter a loss (Goodhart fence: a supervised needle\n'
        '    teaches the clock to lie)."""\n'
        '    nl = np.asarray(nl_diffs, np.float64)\n'
        '    mt = np.asarray(math_diffs, np.float64)\n'
        '    assert nl.shape == mt.shape and nl.ndim == 2, (nl.shape, mt.shape)\n'
        '    M = nl.shape[1]\n'
        '    lags = np.arange(-max_lag, max_lag + 1)\n'
        '    corrs = np.full(len(lags), np.nan)\n'
        '    for li, lag in enumerate(lags):\n'
        '        lag = int(lag)\n'
        '        if abs(lag) >= M:\n'
        '            continue\n'
        '        if lag >= 0:\n'
        '            a = nl[:, :M - lag].ravel()\n'
        '            b = mt[:, lag:].ravel()\n'
        '        else:\n'
        '            a = nl[:, -lag:].ravel()\n'
        '            b = mt[:, :M + lag].ravel()\n'
        '        if len(a) >= 2 and a.std() > 1e-12 and b.std() > 1e-12:\n'
        '            corrs[li] = float(np.corrcoef(a, b)[0, 1])\n'
        '    return lags, corrs\n'
        '\n'
        '\n'
        'if __name__ == "__main__":\n'
        '    # CPU self-test on synthetic data: two classes, drifting per step\n',
        1, "cross_prior + leadlag_needle installed before the self-test")

# --- 5. self-test: build an NL bank too ------------------------------------
s = sub(s,
        '    import tempfile, json as _j\n'
        '    mtmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)\n',
        '    # PAIRED: NL cells — same classes, own drift, separable pages\n'
        '    cells_nl = {}\n'
        '    for s in range(K_STEPS):\n'
        '        for c in ("mul_chain", "add_ladder"):\n'
        '            w = StepWelford(D)\n'
        '            base = (1.0 if c == "mul_chain" else -1.0) * (s + 2)\n'
        '            for _ in range(50):\n'
        '                w.add(base + rng.standard_normal(D) * 0.1)\n'
        '            cells_nl[(s, c)] = w\n'
        '    import tempfile, json as _j\n'
        '    mtmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)\n',
        1, "self-test: synthetic NL cells")

s = sub(s,
        '    p = save_atlas(cells, D, path=tempfile.mktemp(suffix=".npz"),\n'
        '                   manifest_path=mtmp.name)\n',
        '    p = save_atlas(cells, D, path=tempfile.mktemp(suffix=".npz"),\n'
        '                   manifest_path=mtmp.name, nl_cells=cells_nl)\n',
        1, "self-test: save both charts")

s = sub(s,
        '    print("[step_atlas] self-test PASS: Welford banks, stamped save/"\n'
        '          "load, loud door refuses cross-era, consult finds the kind, "\n'
        '          "atlas_class buckets agree")\n',
        '    # THE SECOND CHART: shape/count round trip\n'
        '    assert a["nl_means"] is not None and \\\n'
        '        a["nl_means"].shape == a["means"].shape\n'
        '    assert (a["nl_counts"] == a["counts"]).all()\n'
        '    # THE CROSS-ATLAS PRIOR: a noisy add_ladder breath-0 NL state\n'
        '    # must retrieve add_ladder\'s MATH trajectory (round trip)\n'
        '    _ai = a["classes"].index("add_ladder")\n'
        '    _q0 = a["nl_means"][0, _ai] + rng.standard_normal(D) * 0.05\n'
        '    _ci, _tr = cross_prior(a, _q0)\n'
        '    assert a["classes"][_ci] == "add_ladder", "cross_prior miss"\n'
        '    assert _tr.shape == (K_STEPS, D)\n'
        '    assert np.allclose(_tr, a["means"][:, _ci, :]), \\\n'
        '        "cross_prior returned the wrong chart\'s trajectory"\n'
        '    _cib, _trb = cross_prior(a, np.stack([_q0, a["nl_means"][0, 1 - _ai]]))\n'
        '    assert _cib.shape == (2,) and _trb.shape == (2, K_STEPS, D)\n'
        '    _cin, _trn = cross_prior(a, _q0, return_traj=False)\n'
        '    assert _cin == _ci and _trn is None\n'
        '    try:\n'
        '        cross_prior({"means": a["means"], "nl_means": None,\n'
        '                     "nl_counts": None}, _q0)\n'
        '        raise AssertionError("single-chart atlas served the prior")\n'
        '    except RuntimeError:\n'
        '        pass\n'
        '    # THE NEEDLE on synthetic drifting data: math events are the NL\n'
        '    # events shifted one breath LATER -> the needle must peak at +1\n'
        '    _ev = rng.standard_normal((64, K_STEPS - 1))\n'
        '    _nd = _ev.copy()\n'
        '    _md = np.concatenate(\n'
        '        [rng.standard_normal((64, 1)) * 0.05, _ev[:, :-1]], axis=1)\n'
        '    _lags, _corrs = leadlag_needle(_nd, _md, max_lag=3)\n'
        '    assert int(_lags[np.nanargmax(_corrs)]) == 1, \\\n'
        '        (f"needle peak at {int(_lags[np.nanargmax(_corrs)])}, "\n'
        '         f"designed +1")\n'
        '    assert _corrs[list(_lags).index(1)] > 0.9\n'
        '    assert abs(_corrs[list(_lags).index(0)]) < 0.3\n'
        '    print("[step_atlas] self-test PASS: Welford banks, stamped save/"\n'
        '          "load, loud door refuses cross-era, consult finds the kind, "\n'
        '          "atlas_class buckets agree, SECOND CHART round-trips, "\n'
        '          "cross_prior retrieves the paired math trajectory, needle "\n'
        '          "hears the designed +1 lead")\n',
        1, "self-test: second bank + cross_prior + needle")

ast.parse(s)
assert s.count('nl_means') >= 6 and s.count('leadlag_needle') >= 2

print(f"[paired-atlas chart] {len(ANCHORS)} anchors OK "
      f"(step_atlas +{s.count(chr(10)) - n_lines0} lines):")
for i, desc in enumerate(ANCHORS, 1):
    print(f"  {i:2d}. {desc}")
if CHECK:
    print("[paired-atlas chart] --check: ast OK on the would-be result; "
          "NOTHING written")
else:
    open(fn, 'w').write(s)
    print("[paired-atlas chart] APPLIED (mycelium/step_atlas.py); run its "
          "self-test (.venv/bin/python3 mycelium/step_atlas.py) before "
          "trusting")
