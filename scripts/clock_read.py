"""clock_read.py — RUNG 1 OF THE LADDER: THE CLOCK READ (2026-09-07).

ZERO-TRAINING, ZERO-GRADIENT READ of the breath clock (the temporal
rotor, CLAUDE.md S4 / docs/rotational_bus.md S0) on a champion
checkpoint, over a fixture of test rows. Two measurements:

  A. THE BREATH PROBE — is breath identity DECODABLE from the pooled
     per-breath slot state? Closed-form ridge (numpy, lambda=1e-2 on
     standardized features) from state -> one-hot breath id, fit on a
     70% item split, scored on the held-out 30%. PINNED BAR: >= 0.95.
     CONTROL, printed beside it: the same classification using ONLY
     the state's norm ||x|| (1-D nearest class-mean). The pair
     separates "the clock is written in the GEOMETRY" (probe high,
     norm-only low) from "breath id is just the shrinking radius"
     (both high — consolidation, not a clock; the radius-is-a-
     consolidation-clock law says radius alone is a legitimate
     confound and must be reported, never assumed away).

  B. THE SEXTET SIGNATURE — does the state ROTATE ~60 degrees per
     breath? Orthogonal Procrustes R_k minimizing ||X_{k+1} - X_k R_k||
     over items (per-breath centered), fit inside a COMMON PCA
     subspace (rank r = min(CR_RANK, N-1, D); the full 512-d map is
     rank-deficient at any practical N and its null space would print
     as spurious 0-degree mass). Eigenvalues of R_k -> rotation angles
     in degrees; histogram in 10-degree bins, WEIGHTED by the share of
     X_k's variance living in each eigen-plane. Also: the fraction of
     ||X_{k+1}||^2 the fitted rotation explains (R^2).
     EXPECTATION (descriptive only, printed with the table): a wired
     sextet clock peaks near 60 degrees; a consolidation-only state
     (shrink, no turn) puts its mass near 0 degrees.
     THE CLOCK'S OWN SPEC (mycelium/rotor_clock.py): breath 0 is
     UNCLOCKED (outside time); loop breaths 1..6 carry phase
     (k-1)*60 degrees. So consecutive-pair angles are reported for
     LOOP pairs only (1->2 ... 5->6), with 0->1 printed SEPARATELY
     as the entry step (an unclocked -> clocked hop; no 60-degree
     expectation attaches to it).

THE TAP: ALG_MINE_BREATHS=1 -> forward() returns out["breaths_all"],
the K per-breath raw slot states (B, L_FAC, H_W). No patch to the
head; nothing new runs in the graph. CR_KEY selects the tap: the
default 'breaths_all' is the historical read (r*u / the old
coordinates), and under the POLAR WAIST (ALG_POLAR=1) 'breaths_u'
reads the unit-direction channel beside it — the same probe with the
radius confound removed. The key is stamped into the npz and the
[clock] line: never-mix-coordinates applies between these two taps as
much as between generations. POOLING: mean over the 24 factor
slots per item per breath — the miner's own choice
(scripts/mine_step_atlas.py), so these coordinates are the ATLAS's
coordinates. PASS STRUCTURE: the deployable two-pass cycle
(loop_val's shape) verbatim from the miner — pass-1 unmasked parse ->
build_slot_masks + alt2_fact_buf (live facts) -> masked pass-2 with
fact_buf; breaths_all comes from pass-2.

GOODHART FENCE (mycelium/diagnostic_register.py): every number here is
DIAGNOSTIC. None of it may enter a loss, a training signal, or a
data-selection criterion — a supervised clock learns to look like a
clock. Read it; never train toward it.

Envs:
  CR_N     rows to read (default 256; first N of the fixture)
  CR_CKPT  checkpoint (default .cache/sharp_fedon242.safetensors)
  CR_TEST  'mint' (.cache/algebra_nl_test.jsonl / test23) |
           'wild' (.cache/wild_admitted_holdout.jsonl / wildhold);
           UNSET/'both' = run BOTH, each in its own child process
           (ALG_TEST is read at head IMPORT time — one process can
           hold exactly one fixture)
  CR_KEY   which per-breath tap to read (default 'breaths_all' — the
           r*u / old coordinates, so every existing invocation and every
           banked clock_read_*.npz is UNCHANGED). 'breaths_u' reads the
           POLAR WAIST's direction channel (needs ALG_POLAR=1 in the
           env; docs/polar_waist_spec.md S4) — the same probe with the
           radius confound removed, which is the spec S6 instrument.
           The key travels into the npz and the [clock] summary line, so
           a read can never be mistaken for the other coordinate.
  OUT      .cache/clock_read_<fixture>.npz for the DEFAULT dials, and
           .cache/clock_read_<fixture>_<CR_KEY>_<CR_PLANES>.npz for any
           non-default read — so a polar read can never overwrite the
           historical one, or another dial's.
  CR_PLANES  which DIMS the Procrustes fit (B) may use — ONE selection
           or a COMMA-SEPARATED LIST of them (2026-09-08; e.g.
           'breath_hand,parity,content'). A list collects the states
           ONCE, reads section A ONCE, and then runs section B per
           selection: one [clock] line and one npz each, same fields,
           same naming. A single value behaves exactly as before.
           Selections: 'all'
           (default, unchanged behaviour), 'clocked' (every clocked
           plane), or ONE WHEEL — 'breath_hand' (32 planes, designed
           60 deg/breath), 'parity' (16, designed 120 deg), 'pass' (8,
           static this era), 'content' (200, never clocked). Dims come
           from .cache/polar_bands.json (2p, 2p+1 per plane p); the
           CONTENT planes are fitted separately as a standing control
           for every clocked selection. A wheel selection also reports
           designed_mass = the mass at THAT wheel's DESIGNED angle on
           THAT wheel's own planes ([50,80) breath hand, [110,130)
           parity, [0,20) pass and content) — the bar's quantity.
           WHY PER-WHEEL: 'clocked' mixes three designed angles (60 deg
           on 32 planes, 120 on 16, static on 8), so its [50,80) bin
           has a hard ceiling near the breath hand's share — a diluted
           meter, one level up from the 'all' dilution. WHY (2026-09-07, the birth read):
           the polar waist clocks a designed subset of the 256 planes (read from the bands file) and leaves
           200 at 0 deg, so a variance-weighted histogram over all 512
           dims has a CEILING on mass[50,80) near the clocked planes'
           share of the energy — the birth read's 0.057 is that
           ceiling, not the absence of a clock. Restricting the fit
           asks the question the bar meant to ask. THE DENOMINATOR
           CHANGES with this dial: masses are shares of the FITTED
           subspace's energy, so an 'all' number and a 'clocked' number
           are different meters and must never be compared (never-mix-
           coordinates, applied to instruments). The dial is stamped
           into the npz and the [clock] line for exactly that reason.
  CR_RANK  Procrustes subspace rank (default 64; capped by the fitted
           dim count, so a restricted read cannot ask for more rank
           than it has dims)
  CR_SEED  split/permutation seed (default 242)
  DEV      setdefault PCI+AMD (GPU); test with DEV=CPU CR_N=6

Out: .cache/clock_read_<fixture>.npz + a final "[clock] " line.
GPU script when DEV=PCI+AMD; `import clock_read` is CPU-safe (all
work under __main__/main()).
"""
import os
import subprocess
import sys

# repo root — every path in this stack is relative to it (the
# cd-omission lesson: absolute anchor, then chdir once)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

_FIXTURES = {
    "mint": (".cache/algebra_nl_test.jsonl", "test23"),
    "wild": (".cache/wild_admitted_holdout.jsonl", "wildhold"),
}

CKPT = os.environ.get("CR_CKPT", ".cache/sharp_fedon242.safetensors")
CR_N = int(os.environ.get("CR_N", "256"))
CR_RANK = int(os.environ.get("CR_RANK", "64"))
CR_SEED = int(os.environ.get("CR_SEED", "242"))
CR_TEST = os.environ.get("CR_TEST", "both").strip().lower()
CR_KEY = os.environ.get("CR_KEY", "breaths_all").strip()
CR_PLANES = os.environ.get("CR_PLANES", "all").strip().lower()
# FAIL FAST: a typo'd dial must not cost a full read before it errors
# (the selection is only consumed after collect_states, which is the
# expensive part). Validated here, at import, in one second.
_CR_PLANE_SELS = ("all", "clocked", "breath_hand", "parity", "pass", "content")
# MULTI-WHEEL (2026-09-08): CR_PLANES is a COMMA-SEPARATED LIST. Every
# entry is validated here, at import, before a single state is collected;
# duplicates are refused because two identical selections would write the
# same npz twice (one meter, one file — the naming law of 2026-09-07).
# A single value is a list of one and behaves exactly as before.
CR_PLANE_LIST = [x.strip() for x in CR_PLANES.split(",") if x.strip()]
assert CR_PLANE_LIST, "CR_PLANES is empty (want all|clocked|<wheel>[,...])"
for _sel in CR_PLANE_LIST:
    assert _sel in _CR_PLANE_SELS, (
        f"CR_PLANES entry {_sel!r} unknown "
        f"(want {'|'.join(_CR_PLANE_SELS)}, comma-separated for several)")
assert len(set(CR_PLANE_LIST)) == len(CR_PLANE_LIST), (
    f"CR_PLANES={CR_PLANES!r} repeats a selection — the second read would "
    f"overwrite the first's npz")
POLAR_BANDS = os.environ.get("ALG_POLAR_BANDS", ".cache/polar_bands.json")
LAM = 1e-2
BATCH = 8


def _install_envs(fixture):
    """The champion (fed-on) env stack — VERBATIM from
    scripts/stamp_amplitude_read.py's ENV, MINUS ALG_PC_MIX (no seal
    dose: this read watches the OPEN champion) and with SC_EVAL left
    exactly as the caller left it (never set, never popped here).
    setdefault so a chain's explicit envs WIN (the wave_field_check
    lesson); the FIXTURE envs are set hard, because CR_TEST is this
    script's declared door."""
    os.environ.setdefault("DEV", "PCI+AMD")
    env = {"ALG2": "1", "ALG_FTYPES": "9", "ALG_DUP": "1", "ALG_HW": "512",
           "ALG_WIDE": "1", "ALG_BREATH": "7", "ALG_NOTEBOOK": "1",
           "ALG_SIXWAVE": "1", "NB_PERSLOT": "1", "ALG_BINDBUS": "7",
           "ALG_BIND_D": "512",
           "BIND_CODES": ".cache/bindbus_codes512.npz",
           "ALG_BUSGARAGE": "2", "ALG_SHELF_CIRCLE": "2",
           "ALG_ALTMASK": "1", "ALG_ALT21": "1", "ALG_ALT2": "1",
           "ALG_MASKHEAD": "1", "ALG_FED": "1",
           # the tap: breaths_all (read-only, zero graph change)
           "ALG_MINE_BREATHS": "1"}
    for k, v in env.items():
        os.environ.setdefault(k, v)
    path, name = _FIXTURES[fixture]
    os.environ["ALG_TEST"] = path
    os.environ["ALG_TEST_NAME"] = name
    return path, name


# ------------------------------------------------------------------ math
def ridge_probe(X, seed):
    """X: (K, N, D) per-breath pooled states. Ridge (closed form,
    standardized features, lambda=LAM) state -> one-hot breath id.
    Split is by ITEM (70/30, fixed seed) so no item straddles it."""
    import numpy as np
    K, N, D = X.shape
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_tr = max(1, int(round(0.7 * N)))
    n_tr = min(n_tr, N - 1) if N > 1 else N
    tr_i, te_i = perm[:n_tr], perm[n_tr:]

    def flatten(idx):
        xs = np.concatenate([X[k][idx] for k in range(K)], axis=0)
        ys = np.concatenate([np.full(len(idx), k) for k in range(K)])
        return xs, ys

    Xtr, ytr = flatten(tr_i)
    Xte, yte = flatten(te_i)
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Ztr = np.concatenate([(Xtr - mu) / sd, np.ones((len(Xtr), 1))], 1)
    Zte = np.concatenate([(Xte - mu) / sd, np.ones((len(Xte), 1))], 1)
    Y = np.eye(K)[ytr]
    A = Ztr.T @ Ztr + LAM * len(Ztr) * np.eye(Ztr.shape[1])
    W = np.linalg.solve(A, Ztr.T @ Y)
    pred = np.argmax(Zte @ W, 1)
    acc = float(np.mean(pred == yte)) if len(yte) else float("nan")
    conf = np.zeros((K, K), np.int64)
    for t, q in zip(yte, pred):
        conf[t, q] += 1

    # NORM-ONLY CONTROL: 1-D nearest class-mean on ||x||
    ntr = np.linalg.norm(Xtr, axis=1)
    nte = np.linalg.norm(Xte, axis=1)
    cmean = np.array([ntr[ytr == k].mean() if np.any(ytr == k) else np.inf
                      for k in range(K)])
    npred = np.argmin(np.abs(nte[:, None] - cmean[None, :]), 1)
    nacc = float(np.mean(npred == yte)) if len(yte) else float("nan")
    per_breath_norm = np.array([np.linalg.norm(X[k], axis=1).mean()
                                for k in range(K)])
    return dict(acc=acc, conf=conf, norm_acc=nacc, n_train=len(tr_i),
                n_test=len(te_i), class_norm_mean=cmean,
                per_breath_norm=per_breath_norm)


def common_basis(X, ks, rank):
    """Top-`rank` right-singular directions of the centered states
    stacked over the breaths in `ks` — one shared frame so R_k is a
    map between comparable coordinates, not a rank-deficient 512x512
    with a null space that would print as fake 0-degree mass."""
    import numpy as np
    C = np.concatenate([X[k] - X[k].mean(0) for k in ks], 0)
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    r = int(min(rank, Vt.shape[0], C.shape[0] - 1))
    r = max(r, 2)
    var_kept = float((S[:r] ** 2).sum() / max((S ** 2).sum(), 1e-30))
    return Vt[:r].T, r, var_kept


# THE DESIGNED ANGLE per selectable dim-set: the window the sextet's own
# design says that wheel's planes must turn into, in 10-degree histogram
# bins. Reading the mass at a wheel's DESIGNED angle ON THAT WHEEL'S OWN
# PLANES is the only comparison that is not diluted by the other wheels:
# the clocked set mixes 60 deg (32 planes), 120 deg (16) and 0 deg (8),
# so its [50,80) bin has a hard ceiling near the breath hand's share.
DESIGNED = {"breath_hand": (5, 8, "[50,80)"),      # 60 deg/breath
            "parity":      (11, 13, "[110,130)"),  # 120 deg/breath
            "pass":        (0, 2, "[0,20)"),       # static this era
            "content":     (0, 2, "[0,20)")}       # never clocked
_WHEEL_ALIAS = {"pass": "pass_wheel", "pass_wheel": "pass"}


def band_sets(D):
    """Every selectable dim-set for a D-dimensional state, from the SAME
    allocation file the head reads (ALG_POLAR_BANDS, default
    .cache/polar_bands.json — one band table, one reader idiom: the
    bands note's lesson is that a table nobody reads governs nothing).
    Returns {name: dim indices} with a key per wheel ('breath_hand',
    'parity', 'pass'), plus 'clocked' (their union) and 'content' (the
    complement). Plane p occupies dims (2p, 2p+1), the interleaved-real
    convention of rotor_clock / the head's _rot2."""
    import json
    import numpy as np
    path = os.path.join(ROOT, POLAR_BANDS) if not os.path.isabs(POLAR_BANDS) \
        else POLAR_BANDS
    assert os.path.exists(path), (
        f"CR_PLANES={CR_PLANES} needs the band allocation {path} (the file "
        f"the head itself reads); run with CR_PLANES=all otherwise")
    with open(path) as f:
        bj = json.load(f)
    P = D // 2
    assert int(bj["n_planes"]) == P, (
        f"{path} declares {bj['n_planes']} planes but the states are "
        f"D={D} ({P} planes) — wrong allocation for this read")

    def dims_of(planes):
        pl = sorted({int(x) for x in planes})
        assert pl and pl[-1] < P
        return np.concatenate([[2 * p, 2 * p + 1] for p in pl]).astype(np.int64)

    sets, seen = {}, set()
    for w in bj["wheels"]:
        nm = str(w["name"])
        nm = _WHEEL_ALIAS.get(nm, nm) if nm not in DESIGNED else nm
        assert nm in DESIGNED, (
            f"{path}: wheel '{w['name']}' has no designed angle here "
            f"(known: {sorted(DESIGNED)}) — the file and this reader "
            f"disagree about what the clock is")
        sets[nm] = dims_of(w["planes"])
        seen |= set(int(x) for x in w["planes"])
    sets["clocked"] = dims_of(sorted(seen))
    sets["content"] = np.array([d for d in range(D)
                                if d not in set(sets["clocked"].tolist())],
                               np.int64)
    return sets


def band_dims(D):
    """(clocked_dims, content_dims) — the two-set view, from band_sets
    (one source of truth for the allocation)."""
    st = band_sets(D)
    return st["clocked"], st["content"]


def loop_sextet(X, pairs, rank, dims=None):
    """THE SEXTET ORGAN — the Procrustes sweep over consecutive loop
    pairs, factored so the PRIMARY read and the CONTENT CONTROL call
    the SAME meter (the meter-divergence law: a check must call its
    organ). dims=None fits the full state (the historical path, bit-
    for-bit); an index array restricts the shared basis AND the fit to
    those dims. Returns everything the printer and the npz need."""
    import numpy as np
    Xs = X if dims is None else X[:, :, dims]
    K = Xs.shape[0]
    basis, rank_, var_kept = common_basis(Xs, list(range(K)), rank)
    hist_loop = np.zeros(18)
    rows, all_ang, all_w = [], [], []
    for (a, b) in pairs:
        ang, w, r2 = procrustes_pair(Xs[a], Xs[b], basis)
        hist_loop += hist18(ang, w)
        wm = float((ang * w).sum()) if w.sum() > 0 else float("nan")
        pk = int(np.argmax(hist18(ang, w)))
        rows.append((f"{a}->{b}", r2, wm, pk))
        all_ang.append(ang)
        all_w.append(w)
    if len(pairs):
        hist_loop = hist_loop / len(pairs)
    ang01, w01, r2_01 = procrustes_pair(Xs[0], Xs[1], basis)
    return dict(basis=basis, rank=rank_, var_kept=var_kept,
                hist_loop=hist_loop, rows=rows, all_ang=all_ang,
                all_w=all_w, ang01=ang01, w01=w01, r2_01=r2_01,
                n_dims=Xs.shape[2],
                mass60=float(hist_loop[5:8].sum()),
                mass0=float(hist_loop[0:2].sum()))


def procrustes_pair(Xa, Xb, B):
    """Orthogonal Procrustes in the shared subspace B: R = U V^T from
    SVD of Qa^T Qb. Returns angles (deg, one per eigen-plane / real
    eigenvector), their variance weights (share of Qa's energy in that
    plane), and R^2 = 1 - ||Qb - Qa R||^2 / ||Qb||^2."""
    import numpy as np
    Qa = (Xa - Xa.mean(0)) @ B
    Qb = (Xb - Xb.mean(0)) @ B
    M = Qa.T @ Qb
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt
    resid = Qb - Qa @ R
    denom = float((Qb ** 2).sum())
    r2 = float(1.0 - (resid ** 2).sum() / denom) if denom > 1e-30 \
        else float("nan")
    w, v = np.linalg.eig(R)
    tot = float((Qa ** 2).sum()) + 1e-30
    angles, weights = [], []
    used = np.zeros(len(w), bool)
    for i in range(len(w)):
        if used[i]:
            continue
        used[i] = True
        ang = abs(np.degrees(np.arctan2(w[i].imag, w[i].real)))
        if abs(w[i].imag) < 1e-9:                      # real: 0 or 180 deg
            d = np.real(v[:, i])
            nd = np.linalg.norm(d)
            if nd < 1e-12:
                continue
            wt = float(((Qa @ (d / nd)) ** 2).sum() / tot)
        else:                                          # complex pair
            for j in range(i + 1, len(w)):
                if not used[j] and abs(w[j] - np.conj(w[i])) < 1e-9:
                    used[j] = True
                    break
            a, b = np.real(v[:, i]), np.imag(v[:, i])
            na = np.linalg.norm(a)
            if na < 1e-12:
                continue
            a = a / na
            b = b - (b @ a) * a
            nb = np.linalg.norm(b)
            P = np.stack([a, b / nb], 1) if nb > 1e-12 else a[:, None]
            wt = float(((Qa @ P) ** 2).sum() / tot)
        angles.append(float(ang))
        weights.append(wt)
    a = np.array(angles)
    wgt = np.array(weights)
    if wgt.sum() > 0:
        wgt = wgt / wgt.sum()
    return a, wgt, r2


def _pad(rows):
    """Stack ragged per-pair vectors into a NaN-padded rectangle (npz
    refuses object arrays without pickle)."""
    import numpy as np
    if not rows:
        return np.zeros((0, 0))
    w = max(len(r) for r in rows)
    out = np.full((len(rows), w), np.nan)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


def hist18(angles, weights):
    import numpy as np
    h, _ = np.histogram(np.clip(angles, 0, 179.999), bins=18,
                        range=(0.0, 180.0), weights=weights)
    return h


# ------------------------------------------------------------------ read
def collect_states(fixture):
    """Run the miner's two-pass cycle over the fixture; return
    X (K, N, D) pooled per-breath states + row count info."""
    import numpy as np
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    from phase1_algebra_head import (build_params, forward, load_alg,
                                     build_slot_masks, alt2_fact_buf,
                                     K_VARS)
    # THE JIT'D READ FORWARD (door ALG_JIT_READ; mycelium/jit_read.py).
    # Door unset -> _rf(forward, ...) IS forward(...), `keys` dropped.
    from mycelium.jit_read import read_forward as _rf
    _JK_OPEN = ("fat", "args", "res", "pres", "ftype", "op", "dig", "dup")

    samples, states, tokmask, gold, sent = load_alg("test")
    n_all = len(samples)
    n_take = min(CR_N, n_all) if CR_N > 0 else n_all
    take = np.arange(n_take)
    print(f"[clock-read] fixture={fixture} "
          f"({os.environ['ALG_TEST_NAME']}) rows={n_take}/{n_all} "
          f"ckpt={CKPT} dev={os.environ.get('DEV')} key={CR_KEY} "
          f"polar={os.environ.get('ALG_POLAR', '0')}", flush=True)

    p = build_params(0)
    sd = safe_load(CKPT)
    assert set(sd.keys()) == set(p.keys()), \
        (sorted(set(sd) - set(p))[:4], sorted(set(p) - set(sd))[:4])
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

    pages = None      # list over breaths of list over items
    for s0 in range(0, len(take), BATCH):
        sl = take[s0:s0 + BATCH]
        pad = BATCH - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        ts = Tensor(states[sl_p].astype(np.float32), dtype=dtypes.float)
        tk = Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float)
        se = Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int)
        # pass 1: unmasked parse -> masks + live facts (loop_val's cycle)
        o0 = _rf(forward, p, ts, tk, se, keys=_JK_OPEN)
        onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
        mk = build_slot_masks(onp0, sent[sl_p].astype(np.int32))
        _ka = (("pres", "ftype", "op", "dig")
               + (("dup",) if "dup" in o0 else ()))
        _oa = {**onp0, **{k: o0[k].realize().numpy() for k in _ka}}
        _nv = np.array([samples[int(i)].get("n_vars", K_VARS)
                        for i in sl_p])
        _ma = np.array([samples[int(i)].get("m", 0) for i in sl_p])
        fb = alt2_fact_buf(_oa, sent[sl_p], _nv, _ma)
        # pass 2: the breathing walk; the tap hands back the raw
        # per-breath slot states
        o = _rf(forward, p, ts, tk, se, keys=(CR_KEY,),
                slot_mask=Tensor(mk, dtype=dtypes.float),
                fact_buf=Tensor(fb, dtype=dtypes.float))
        assert CR_KEY in o, (
            f"forward() emitted no '{CR_KEY}' (keys: "
            f"{sorted(k for k in o if k.startswith('breaths'))}). "
            f"'breaths_u' needs ALG_POLAR=1 in the env (the polar waist's "
            f"direction tap); 'breaths_all' needs ALG_MINE_BREATHS=1.")
        br = [b.realize().numpy() for b in o[CR_KEY]]
        assert br[0].ndim == 3, (
            f"CR_KEY='{CR_KEY}' gives {br[0].ndim}-d pages; this read pools "
            f"(B, L_FAC, D) slot-major states (breaths_all / breaths_u)")
        if pages is None:
            pages = [[] for _ in br]
            print(f"[clock-read] {CR_KEY} K={len(br)} "
                  f"page={br[0].shape} (K=7 expected: breath-0 + six "
                  f"loop breaths)", flush=True)
        elif len(br) != len(pages):
            print(f"[clock-read] WARNING: breaths_all changed count "
                  f"{len(pages)} -> {len(br)}; batch skipped", flush=True)
            continue
        for bi in range(len(sl)):          # pads (bi >= len(sl)) skipped
            for s_id in range(len(br)):
                # POOLING: mean over the 24 factor slots (the miner's
                # own idiom -> atlas coordinates)
                pages[s_id].append(br[s_id][bi].mean(0).astype(np.float64))
        if (s0 // BATCH) % 8 == 0:
            print(f"[clock-read] {min(s0 + BATCH, len(take))}/{len(take)}",
                  flush=True)
    assert pages is not None and len(pages[0]) > 0, "no rows read"
    X = np.stack([np.stack(pg) for pg in pages])       # (K, N, D)
    return X



def sextet_section(X, K, N, D, pr, bar, pairs, fixture, sel):
    """SECTION B for ONE plane selection — the whole of the old
    section B, verbatim, with the module-level CR_PLANES read replaced
    by the `sel` argument. MULTI-WHEEL (2026-09-08): CR_PLANES takes a
    comma-separated LIST, the states are collected ONCE per fixture and
    section A is read ONCE, and this organ runs per listed selection —
    one [clock] line and one npz each, under the existing naming
    convention. A single-value CR_PLANES therefore calls this exactly
    once and prints exactly what it printed before (the chain that read
    three wheels per fixture went from six state collections to two;
    collect_states is the expensive half)."""
    import numpy as np
    fit_dims = None                        # None = the full state (historical)
    ctl = None
    dw = DESIGNED.get(sel)                 # the selection's designed window
    if sel != "all":
        # THE PLANE RESTRICTION (2026-09-07, the birth read): the polar
        # waist clocks a designed subset of the 256 planes (bands file) and pins the rest at 0 deg,
        # so a variance-weighted histogram over all 512 dims has a
        # CEILING on mass[50,80) near the clocked planes' energy share.
        # PER-WHEEL (the second refinement): even 'clocked' mixes three
        # designed angles — 60 deg on 32 planes, 120 deg on 16, static
        # on 8 — so its [50,80) bin cannot exceed the breath hand's
        # share either. A wheel's own planes read at that wheel's own
        # DESIGNED angle is the undiluted question.
        _sets = band_sets(D)
        assert sel in _sets, (
            f"CR_PLANES={sel!r} is a known selection but "
            f"{POLAR_BANDS} declares no such set (has {sorted(_sets)}) — "
            f"the allocation file and this reader disagree")
        fit_dims = _sets[sel]
        if sel != "content":
            # the content planes are the standing control for every
            # clocked selection (their mass belongs at 0 deg — that is
            # what "unclocked" means). When content IS the selection it
            # is its own primary; no second copy.
            ctl = loop_sextet(X, pairs, CR_RANK, _sets["content"])
    sx = loop_sextet(X, pairs, CR_RANK, fit_dims)
    basis, rank, var_kept = sx["basis"], sx["rank"], sx["var_kept"]
    rows, all_ang, all_w = sx["rows"], sx["all_ang"], sx["all_w"]
    hist_loop = sx["hist_loop"]
    ang01, w01, r2_01 = sx["ang01"], sx["w01"], sx["r2_01"]
    print()
    print("== B. THE SEXTET SIGNATURE (orthogonal Procrustes per "
          "consecutive breath pair) ==")
    if fit_dims is not None:
        print(f"   PLANES={sel}: fit restricted to {sx['n_dims']} of "
              f"{D} dims ({len(fit_dims) // 2} planes from {POLAR_BANDS})"
              + ("" if ctl is None else
                 f"; {ctl['n_dims'] // 2} content planes fitted separately "
                 f"below as the control")
              + ". THE DENOMINATOR IS THE RESTRICTED SUBSPACE — these "
                "masses are NOT comparable to a CR_PLANES=all read.")
        if dw is not None:
            print(f"   DESIGNED ANGLE for '{sel}' = {dw[2]} deg "
                  f"(rotor_clock: breath hand 60/breath, parity 120/breath, "
                  f"pass wheel static this era) — designed_mass below is "
                  f"the bar's quantity: mass at THIS wheel's designed "
                  f"angle on THIS wheel's own planes.")
        else:
            print(f"   NO SINGLE DESIGNED ANGLE for '{sel}' (it mixes "
                  f"wheels: 60 deg on 32 planes, 120 on 16, static on 8) — "
                  f"its [50,80) bin is CEILINGED by the breath hand's "
                  f"share; select a wheel for the undiluted read.")
    print(f"   shared subspace rank r={rank} "
          f"(captures {var_kept:.4f} of centered variance); angles are "
          f"eigen-plane angles of R_k, WEIGHTED by that plane's share "
          f"of X_k's energy")
    print("   EXPECTATION (descriptive): wired sextet clock -> mass "
          "near 60 deg; consolidation-only -> mass near 0 deg")
    if N < 4 * rank:
        print(f"   WARNING: N={N} items vs r={rank} dims — the "
              f"Procrustes fit is UNDER-DETERMINED (R^2 near 1 is "
              f"overfit, angles are noise). Want N >> r; raise CR_N "
              f"or lower CR_RANK.")
    print("   pair        R^2      wmean_ang  peak_bin   top-3 "
          "(angle:weight)")
    for ((a, b), (_nm, r2, wm, pk), ang, w) in zip(pairs, rows,
                                                   all_ang, all_w):
        order = np.argsort(-w)[:3]
        top = " ".join(f"{ang[i]:.1f}:{w[i]:.3f}" for i in order)
        print(f"   b{a}->b{b}   {r2:>7.4f}  {wm:>9.2f}  "
              f"[{pk * 10:>3d},{pk * 10 + 10:>3d})   {top}")
    h01 = hist18(ang01, w01)
    wm01 = float((ang01 * w01).sum()) if w01.sum() > 0 else float("nan")
    print(f"   b0->b1 (ENTRY STEP, unclocked -> clocked; reported "
          f"SEPARATELY, no 60-deg expectation)")
    print(f"          R^2={r2_01:.4f}  wmean_ang={wm01:.2f}  "
          f"peak_bin=[{int(np.argmax(h01)) * 10},"
          f"{int(np.argmax(h01)) * 10 + 10})")
    print()
    print("   LOOP-PAIR ANGLE HISTOGRAM (variance-weighted, mean over "
          f"{len(pairs)} loop pairs):")
    hmax = max(float(hist_loop.max()), 1e-12)
    for i in range(18):
        bar_s = "#" * int(round(40 * hist_loop[i] / hmax))
        star = "  <- 60 deg (sextet quantum)" if i == 6 else ""
        print(f"     [{i * 10:>3d},{i * 10 + 10:>3d})  "
              f"{hist_loop[i]:.4f}  {bar_s}{star}")
    peak = int(np.argmax(hist_loop))
    mass60 = float(hist_loop[5:8].sum())      # 50-80 deg
    mass0 = float(hist_loop[0:2].sum())       # 0-20 deg
    print(f"   peak bin = [{peak * 10},{peak * 10 + 10})   "
          f"mass[50,80) = {mass60:.4f}   mass[0,20) = {mass0:.4f}")
    dmass = None
    if dw is not None:
        dmass = float(hist_loop[dw[0]:dw[1]].sum())
        print(f"   DESIGNED MASS ({sel} @ {dw[2]} deg) = "
              f"{dmass:.4f}   <- THE BAR'S QUANTITY")
    if ctl is not None:
        print()
        print(f"   CONTENT-PLANE CONTROL ({ctl['n_dims'] // 2} unclocked "
              f"planes, {ctl['n_dims']} dims, r={ctl['rank']}, "
              f"var_kept={ctl['var_kept']:.4f}):")
        print(f"      peak bin = [{int(np.argmax(ctl['hist_loop'])) * 10},"
              f"{int(np.argmax(ctl['hist_loop'])) * 10 + 10})   "
              f"mass[50,80) = {ctl['mass60']:.4f}   "
              f"mass[0,20) = {ctl['mass0']:.4f}   "
              f"designed[0,20) = {ctl['mass0']:.4f}")
        print(f"      EXPECTATION (descriptive): the clock never touches "
              f"these planes, so their mass belongs near 0 deg. Mass at "
              f"60 deg HERE would mean the restriction is not doing what "
              f"it says — either the allocation file disagrees with the "
              f"weights, or the state is turning for another reason.")

    # THE OUTPUT NAME CARRIES THE DIAL (2026-09-07). Historically the
    # name held only the fixture, so a breaths_u read and a breaths_all
    # read of the same fixture — or a clocked and an all read — SILENTLY
    # OVERWROTE each other; this was not hypothetical, a CPU n=6 smoke
    # clobbered the GPU birth read's mint npz while the dial was being
    # built. DEFAULTS KEEP THE HISTORICAL PATH (nothing downstream moves);
    # any non-default read gets its own file, so two meters can never
    # land in one filename (never-mix-coordinates, at the filesystem).
    _sfx = "" if (CR_KEY == "breaths_all" and sel == "all") \
        else f"_{CR_KEY}_{sel}"
    out = os.path.join(ROOT, ".cache", f"clock_read_{fixture}{_sfx}.npz")
    np.savez(out,
             fixture=fixture, ckpt=CKPT, key=CR_KEY, planes=sel,
             n_dims_fit=sx["n_dims"], K=K, N=N, D=D, rank=rank,
             **({} if dw is None else dict(
                 designed_lo_deg=dw[0] * 10, designed_hi_deg=dw[1] * 10,
                 designed_mass=dmass)),
             var_kept=var_kept, seed=CR_SEED,
             **({} if ctl is None else dict(
                 content_n_dims=ctl["n_dims"], content_rank=ctl["rank"],
                 content_var_kept=ctl["var_kept"],
                 content_hist_loop=ctl["hist_loop"],
                 content_mass60=ctl["mass60"], content_mass0=ctl["mass0"])),
             probe_acc=pr["acc"], norm_only_acc=pr["norm_acc"],
             confusion=pr["conf"], per_breath_norm=pr["per_breath_norm"],
             class_norm_mean=pr["class_norm_mean"],
             pair_names=np.array([r[0] for r in rows]),
             pair_r2=np.array([r[1] for r in rows]),
             pair_wmean_angle=np.array([r[2] for r in rows]),
             pair_peak_bin=np.array([r[3] for r in rows]),
             # ragged-safe: eigen-plane counts can differ per pair
             # (a conjugate pair can degenerate to two real roots), so
             # pad to a rectangle with NaN rather than pickling
             angles_loop=_pad(all_ang), weights_loop=_pad(all_w),
             hist_loop=hist_loop, hist_01=h01,
             angles_01=ang01, weights_01=w01, r2_01=r2_01)
    print(f"[clock-read] saved {out}")
    print(f"[clock] fixture={fixture} key={CR_KEY} planes={sel}"
          f"({sx['n_dims']}/{D}d) n={N} K={K} "
          f"probe_acc={pr['acc']:.4f} (bar>=0.95 {bar}) "
          f"norm_only_acc={pr['norm_acc']:.4f} "
          f"sextet_peak_bin=[{peak * 10},{peak * 10 + 10}) "
          f"mass[50,80)={mass60:.4f} mass[0,20)={mass0:.4f} "
          + ("" if dw is None else
             f"designed_mass={dmass:.4f}(@{dw[2]}) ")
          + ("" if ctl is None else
             f"content_mass[50,80)={ctl['mass60']:.4f} "
             f"content_mass[0,20)={ctl['mass0']:.4f} ")
          + f"entry_b0b1_wmean={wm01:.2f}", flush=True)


def run(fixture):
    # (numpy moved with section B into sextet_section — run() itself no
    # longer touches it)
    _install_envs(fixture)
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    X = collect_states(fixture)
    K, N, D = X.shape
    print(f"[clock-read] states X = (K={K}, N={N}, D={D})", flush=True)

    # ---------------------------------------------- A: THE BREATH PROBE
    pr = ridge_probe(X, CR_SEED)
    bar = "PASS" if pr["acc"] >= 0.95 else "FAIL"
    print()
    print("== A. THE BREATH PROBE (state -> breath id; ridge, "
          "70/30 by item) ==")
    print(f"   items train/test = {pr['n_train']}/{pr['n_test']}   "
          f"rows/class = {pr['n_test']}")
    print(f"   probe accuracy      = {pr['acc']:.4f}   "
          f"[PINNED BAR >= 0.95: {bar}]")
    print(f"   norm-only accuracy  = {pr['norm_acc']:.4f}   "
          f"(1-D ||x|| nearest class-mean; the consolidation control)")
    print("   mean ||x|| per breath: " +
          " ".join(f"b{k}:{v:.4g}" for k, v in
                   enumerate(pr["per_breath_norm"])))
    print("   confusion (rows=true breath, cols=predicted):")
    print("        " + " ".join(f"{k:>4d}" for k in range(K)))
    for k in range(K):
        print(f"     b{k}  " + " ".join(f"{c:>4d}" for c in pr["conf"][k]))

    # ------------------------------------------ B: THE SEXTET SIGNATURE
    # MULTI-WHEEL (2026-09-08): the pairs are a property of K, not of
    # the plane selection — built once, handed to every selection.
    loop_ks = list(range(1, K))            # breath 0 is outside time
    pairs = [(k, k + 1) for k in loop_ks[:-1]]
    for _sel in CR_PLANE_LIST:
        sextet_section(X, K, N, D, pr, bar, pairs, fixture, _sel)


def main():
    if CR_TEST in _FIXTURES:
        run(CR_TEST)
        return 0
    if CR_TEST not in ("both", ""):
        print(f"[clock-read] unknown CR_TEST={CR_TEST!r} "
              f"(want mint|wild|both)")
        return 2
    # BOTH: one fixture per process — ALG_TEST/ALG_TEST_NAME are read
    # at phase1_algebra_head IMPORT time (module-level STATES_NPZ /
    # TEST_NAME), so a single process can hold exactly one fixture.
    rc = 0
    for fx in ("mint", "wild"):
        print(f"\n########## CR_TEST={fx} ##########", flush=True)
        env = dict(os.environ, CR_TEST=fx)
        r = subprocess.run([sys.executable, os.path.abspath(__file__)],
                           env=env, cwd=ROOT)
        rc = rc or r.returncode
    return rc


if __name__ == "__main__":
    sys.exit(main())
