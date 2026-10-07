"""scripts/welford_atlas.py -- THE WELFORD ATLAS (2026-10-07, Bryce: "the centroids need some
love, and Welford stats; collect telemetry of every problem solved and build a clean library of
centroids over time"). Zero-GPU: one build, two reads.

THE RULING (ledger 2026-10-05 "THE ATLAS RADIUS READ", 2026-10-06 "NL ATLAS v0 READ" / "THE META
READ" / "THE LEARNED PERCEIVER v1 READ"; CLAUDE.md S4 "never mix generations' coordinates" + the
two-channel law -- angle=identity, radius=consolidation): the Welford atlas is a MONITOR and a
perceiver INPUT, never a decoder. It learns ONLY from gold-graded rows (the annotated diet, below)
-- never from read-time "wins" (the courtroom's own judge is 9%-precise, not a label source) and
never from the wild holdout, which this script reads EXACTLY ONCE, below, as a measurement.

THE THREE ACTS:

  build (slow, CPU forward pass; ~45 min on the 775-row diet slice) -- streams
    .cache/form_pm35c_slice1024_valid2.jsonl (the annotated validation slice held out from
    PMS8_241's own training; chosen per the task's own fallback clause: the already-collected
    telemetry npz (.cache/perceiver_telemetry_PMS8_241_pm35cslicevalid2.npz) holds FEATURES
    (ent/cos_given/cos_rel/...), not raw per-slot states -- confirmed by inspection, no state
    array in its keys -- so a fresh CPU forward is required either way, on the SAME 775-row slice
    perceiver_collect.py used, under the SURF8 family env for PMS8_241 (adaptive_stop.FAMILY_ENVS,
    imported, not re-typed). Builds TWO Welford libraries, each keyed by kind (given / rel:add /
    rel:mul / rel:sub / rel:div -- atlas_radius_read.py's gold_kind, copied verbatim, 4 lines) AND
    by coarse knot (meta_read.coarse_knot, imported):
      content -- the head's own FINAL-BREATH slot state, content dims only (_hier_band_dims(),
        384 of 512; the 128 clock dims excluded, same door atlas_radius_read.py uses).
      retina  -- the frozen trunk's clause embedding (jury_features.py's span pool over
        stamp_arg_mentions.clause_of's window, exactly nl_atlas_clause.py's own call pattern,
        reused via import), PROSE rows only (gsm8k/svamp/asdiv; nl_atlas_clause.py's own
        convention -- mint's templated text is not a clause-meaning exemplar).
    Writes .cache/welford_atlas_PMS8_241_content.npz, .cache/welford_atlas_PMS8_241_retina.npz.

  read (zero GPU, ONCE on wild) -- per gold slot of the 311-row wild holdout
    (.cache/wild_admitted_holdout.jsonl), in both spaces: a Mahalanobis-lite distance (diagonal
    variance) and a cosine to its OWN KIND's mean and to the NEAREST KNOT's mean (library cells
    with n >= MIN_KNOT_N, a density/typicality read, not a claim that slot's own knot was seen).
    AUROC of each distance/cosine vs per-slot correctness (.cache/ps_legal_wild_PMS8_241.npz) and
    vs the membrane-entropy baseline (.cache/perceiver_telemetry_PMS8_241_wildhold.npz's per-slot
    `ent`, final breath -- the ledger's "AUROC 0.708" quantity, recomputed here for parity on the
    SAME 2051-slot denominator). Per-row means vs row correctness (.cache/courtroom_PMS8_241.pkl's
    top1, the 14/311 of record). THE BAR (pinned): AUROC >= entropy + 0.03 in at least one space.

  drift (zero GPU, a read of banked states, allowed) -- the clock-band probe's states for
    PMS8_241 / HS_241 / HSd_241 (+ EY_241, bonus; RK_241 has no banked clock_band_states file --
    flagged below, not fabricated) on wild: per-kind mean direction per body, the pairwise cosine
    between bodies (the rotation CLAUDE.md S4 warns about), and the Welford radius per body.

usage:  DEV=CPU .venv/bin/python3 scripts/welford_atlas.py [build|read|drift|all]
        WA_BUILD_LIMIT=16 DEV=CPU .venv/bin/python3 scripts/welford_atlas.py build   # CPU smoke
"""
import os
import sys
import json
import time
import pickle
import collections

os.environ["DEV"] = "CPU"   # the one line that must win every setdefault below -- zero-GPU, always
os.environ.setdefault("ALG_TEST", ".cache/form_pm35c_slice1024_valid2.jsonl")
os.environ.setdefault("ALG_TEST_NAME", "pm35cslicevalid2")

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/picker")

import numpy as np

import adaptive_stop as AS          # FAMILY_ENVS["PMS8_241"] -- the SURF8 recipe, imported not retyped
import meta_read as MR              # coarse_knot() -- the value-abstracted WL digest, imported
from mycelium.custody_gold import row_gold
from mycelium.welford import Welford, WelfordLibrary, js_divergence

AS._build_family_env("PMS8_241", "")
assert os.environ["DEV"] == "CPU", "welford_atlas: zero-GPU, always"

DIET_SLICE = ".cache/form_pm35c_slice1024_valid2.jsonl"
WILD_PATH = ".cache/wild_admitted_holdout.jsonl"
CKPT = ".cache/sharp_PMS8_241.safetensors"
CONTENT_LIB_PATH = ".cache/welford_atlas_PMS8_241_content.npz"
RETINA_LIB_PATH = ".cache/welford_atlas_PMS8_241_retina.npz"
CONTENT_BREATHS_LIB_PATH = ".cache/welford_atlas_PMS8_241_content_breaths.npz"   # THE PER-BREATH LIBRARY (2026-10-07 build)
DIET_STATES_BREATHS_PATH = ".cache/welford_atlas_PMS8_241_diet_states_breaths.npz"  # raw per-breath content states, diet, cached so the feature step never re-forwards
PERCEIVER_V1B_DIET_FEAT = ".cache/perceiver_v1b_features_PMS8_241_pm35cslicevalid2.npz"
PERCEIVER_V1B_WILD_FEAT = ".cache/perceiver_v1b_features_PMS8_241_wildhold.npz"
OUT_TXT = ".cache/welford_atlas_PMS8_241.txt"
DRIFT_TAGS = ["PMS8_241", "HS_241", "HSd_241", "EY_241"]   # RK_241: no clock_band_states file (checked, absent)
MIN_N_KIND = 40       # atlas_radius_read.py's own floor for a kept kind-cell
MIN_N_KNOT = 3        # a coarse knot needs >= 3 slot members before it is even considered
MIN_ROWS_KNOT = 2     # AND >= 2 DISTINCT ROWS (a single row's 3-8 factors clear MIN_N_KNOT alone)
BATCH = 16

LOG = []


def P(s=""):
    print(s, flush=True)
    LOG.append(s)


# ======================================================================================
# THE GOLD KIND -- atlas_radius_read.py's gold_kind, copied verbatim (4 lines; the one identity
# door every per-slot read in this file shares): forward relation (own result) -> add/mul;
# inverse relation (consumed as an argument elsewhere) -> sub/div; everything else -> given.
# ======================================================================================
def gold_kind(fac, j):
    ft = fac["ftype"]
    if ft == "given":
        return "given"
    if ft == "rel":
        op = fac["op"]
        # atlas_radius_read.py's own form assumed every rel's literal op is add/mul, with sub/div
        # read OFF the forward/inverse form (result==j). The diet (unlike the wild fixture it was
        # built from) also carries LITERAL sub/div op annotations -- kept as themselves, no
        # forward/inverse remap (there is nothing to invert: a literal "sub" IS rel:sub).
        if op in ("add", "mul"):
            return "rel:" + (op if fac["result"] == j else {"add": "sub", "mul": "div"}[op])
        assert op in ("sub", "div"), op
        return "rel:" + op
    # the diet (unlike wild, per THE META READ: "wild carries no sel/mod/pct/fdiv/macro gold")
    # carries the other 6 of the 8-way ftype palette (CLAUDE.md S1) -- kept as their own kind key
    # in the library (unused by the wild read's KINDS loop, which only needs the 5 wild carries).
    assert ft in ("mod", "sel", "pct", "fdiv", "macro", "frac"), ft
    return ft


KINDS = ["given", "rel:add", "rel:mul", "rel:sub", "rel:div"]


def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


def auroc(y, score):
    """Mann-Whitney AUROC, no sklearn dependency assumed at import time (kept local -- every other
    script in this codebase that needs one either has sklearn or computes it by rank; this avoids
    adding a new hard dependency to a zero-GPU read)."""
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    keep = np.isfinite(score)
    y, score = y[keep], score[keep]
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan"), n1, n0
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    sorted_scores = score[order]
    i = 0
    r = 1
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (r + (r + (j - i))) / 2.0
        ranks[order[i:j + 1]] = avg_rank
        r += (j - i + 1)
        i = j + 1
    auc = (ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return float(auc), n1, n0


# ======================================================================================
# ACT 1: BUILD -- the diet's gold rows, two spaces, two Welford libraries
# ======================================================================================
def build():
    t0 = time.time()
    import phase1_algebra_head as H
    from phase1_algebra_head import build_params, forward, load_alg, build_slot_masks, alt2_fact_buf, K_VARS, L_FAC
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load

    bands, clock_dims = H._hier_band_dims()
    CONTENT = np.sort(np.concatenate(bands))
    assert len(np.intersect1d(CONTENT, clock_dims)) == 0
    C = len(CONTENT)
    K_B = int(os.environ["ALG_BREATH"])

    vs, vst, vtk, vg, vse = load_alg("test")
    n_full = len(vs)
    limit = int(os.environ.get("WA_BUILD_LIMIT", "0")) or n_full
    n = min(n_full, limit)
    P(f"[build] diet slice {DIET_SLICE}: {n}/{n_full} rows (WA_BUILD_LIMIT={os.environ.get('WA_BUILD_LIMIT', '0')})")
    # THE CUSTODY DOOR, per-row (soft gate): most rows pass row_gold() (mint's solution field or a
    # pen row's harvest-keyed answer); a known wrinkle in this fixture (asdiv/eq_sets_extract rows
    # admitted without src_idx AND without a populated solution vector -- neither cleanly pen nor
    # mint by custody_gold.is_pen_row's own discriminator) fails it. custody_gold.py is an existing,
    # authoritative module (never edited here) -- the correct response to an ungradable row is
    # EXCLUSION, logged loudly, never a crash and never a silent include.
    admissible = np.ones(n, dtype=bool)
    for i in range(n):
        try:
            row_gold(vs[i])
        except Exception as e:
            admissible[i] = False
    n_bad = int((~admissible).sum())
    P(f"[build] custody door: {n_bad}/{n} rows failed row_gold() (excluded from both libraries, "
      f"not silently included) -- {n - n_bad} admitted")

    p = build_params(0)
    sd = safe_load(CKPT)
    assert set(sd) == set(p), (sorted(set(sd) - set(p))[:4], sorted(set(p) - set(sd))[:4])
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    P(f"[build] ckpt loaded ({time.time()-t0:.0f}s); C={C} content dims / 512; K_B={K_B}")

    final_kb = K_B - 1
    # THE PER-BREATH TAP (2026-10-07 extension): the SAME forward pass already stamps "state" (the
    # content entering EVERY breath kb=0..K_B-1, H._CENSUS's own convention -- clock_band_probe.py's
    # "the state the band probe would see", verified below against the pre-existing final-only
    # library) -- captured once here so the per-breath library and the original final-breath library
    # cost exactly one forward pass between them, not two.
    states_all = np.zeros((n, K_B, L_FAC, C), np.float16)
    for s0 in range(0, n, BATCH):
        sl = np.arange(s0, min(s0 + BATCH, n))
        pad = BATCH - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        ts = Tensor(np.ascontiguousarray(vst[sl_p]), dtype=dtypes.half)
        tk = Tensor(vtk[sl_p].astype(np.float32), dtype=dtypes.float)
        se = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)
        o0 = forward(p, ts, tk, se)
        onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
        mk = build_slot_masks(onp0, vse[sl_p].astype(np.int32))
        _ka = ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())
        _oa = {**onp0, **{k: o0[k].realize().numpy() for k in _ka}}
        _nv = np.array([vs[int(i)].get("n_vars", K_VARS) for i in sl_p])
        _ma = np.array([vs[int(i)].get("m", 0) for i in sl_p])
        fb = alt2_fact_buf(_oa, vse[sl_p].astype(np.int32), _nv, _ma)
        H._CENSUS = []
        o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float), fact_buf=Tensor(fb, dtype=dtypes.float))
        o["fat"].realize()
        got = {kb: arr for (kb, tag, arr) in H._CENSUS if tag == "state"}
        H._CENSUS = None
        assert final_kb in got, sorted(got)
        assert set(range(K_B)) <= set(got), sorted(got)
        for kb in range(K_B):
            states_all[sl, kb] = got[kb][:len(sl), :L_FAC, :][:, :, CONTENT].astype(np.float16)
        if s0 % (BATCH * 8) == 0:
            P(f"[build] content forward {s0 + len(sl)}/{n} ({time.time()-t0:.0f}s)")
    P(f"[build] content forward passes done ({time.time()-t0:.0f}s)")
    states_final = states_all[:, final_kb].astype(np.float32)

    # ---- CONTENT-SPACE LIBRARY: keyed by kind and by coarse knot ----
    # A knot's SLOT count (cell.n) is not the same question as how many DISTINCT ROWS fed it -- one
    # row can contribute 3-8 slots to its own knot, so a slot-count floor alone would call a
    # singleton row "legible". knot_rows tracks distinct-row membership per knot key, saved as a
    # sidecar JSON next to the npz; nearest_knot_cos()/the read below require BOTH floors.
    content_lib = WelfordLibrary()
    row_knot = {}
    knot_rows = collections.defaultdict(set)
    n_slots = 0
    for i in range(n):
        if not admissible[i]:
            continue
        row = vs[i]
        facs = row["factors"]
        knot = row_knot.get(i)
        if knot is None:
            knot = MR.coarse_knot(row)
            row_knot[i] = knot
        for j in range(min(L_FAC, len(facs))):
            if vg["presence"][i, j] <= 0.5:
                continue
            k = gold_kind(facs[j], j)
            x = states_final[i, j]
            content_lib.get_or_create(f"kind:{k}", dim=C).update(x)
            content_lib.get_or_create(f"knot:{knot}", dim=C).update(x)
            knot_rows[f"knot:{knot}"].add(i)
            n_slots += 1
    content_lib.save(CONTENT_LIB_PATH, extra_meta=json.dumps(dict(tag="PMS8_241", space="content",
                      source=DIET_SLICE, n_rows=n, n_slots=n_slots, content_dims=C,
                      generated=time.strftime("%Y-%m-%d %H:%M:%S"))))
    json.dump({k: len(v) for k, v in knot_rows.items()}, open(CONTENT_LIB_PATH + ".knot_rows.json", "w"))
    P(f"[build] content library: {n_slots} slots over {n} rows -> {CONTENT_LIB_PATH}")
    kind_n = {k: content_lib[f"kind:{k}"].n for k in KINDS if f"kind:{k}" in content_lib}
    knot_cells = [k for k in content_lib if k.startswith("knot:")]
    knot_multi_row = [k for k in knot_cells if len(knot_rows[k]) >= MIN_ROWS_KNOT]
    P(f"[build] content kind counts: {kind_n}")
    P(f"[build] content coarse knots: {len(knot_cells)} distinct over {n} rows; "
      f"{sum(1 for k in knot_cells if content_lib[k].n >= MIN_N_KNOT)} with n>={MIN_N_KNOT} slots, "
      f"{len(knot_multi_row)} fed by >={MIN_ROWS_KNOT} DISTINCT ROWS (the honest repetition floor -- "
      f"a single row's own 3-8 factors otherwise clears a slot-count floor trivially)")

    # ---- CONTENT-SPACE PER-BREATH LIBRARY: keyed by (kind, breath) and (coarse knot, breath) ----
    # (Bryce 2026-10-07: "a composite key (centroid, breath) in the atlas = the expected flight path
    # per breath") -- reuses states_all (the SAME forward pass above, zero extra GPU/CPU cost) and the
    # SAME admissibility/knot assignment as the final-breath library, so the kb=K_B-1 slice of this
    # library is, by construction, the IDENTICAL data the final-only library above was built from
    # (verified below, not assumed).
    content_breaths_lib = WelfordLibrary()
    n_slots_b = 0
    for i in range(n):
        if not admissible[i]:
            continue
        row = vs[i]
        facs = row["factors"]
        knot = row_knot[i]
        for j in range(min(L_FAC, len(facs))):
            if vg["presence"][i, j] <= 0.5:
                continue
            k = gold_kind(facs[j], j)
            for kb in range(K_B):
                x = states_all[i, kb, j].astype(np.float32)
                content_breaths_lib.get_or_create(f"kind:{k}@{kb}", dim=C).update(x)
                content_breaths_lib.get_or_create(f"knot:{knot}@{kb}", dim=C).update(x)
            n_slots_b += 1
    content_breaths_lib.save(CONTENT_BREATHS_LIB_PATH, extra_meta=json.dumps(dict(
        tag="PMS8_241", space="content_breaths", source=DIET_SLICE, n_rows=n, n_slots=n_slots_b,
        content_dims=C, K_B=K_B, generated=time.strftime("%Y-%m-%d %H:%M:%S"))))
    # the knot's distinct-row floor is BREATH-INDEPENDENT (the same rows feed every kb of a given
    # knot) -- reuse the final-breath library's own knot_rows tally (computed above) rather than
    # re-tracking row membership seven times over.
    knot_rows_b = {f"knot:{knot_key}@{kb}": len(rows_) for knot_key, rows_ in
                   ((k[5:], knot_rows[k]) for k in knot_cells) for kb in range(K_B)}
    json.dump(knot_rows_b, open(CONTENT_BREATHS_LIB_PATH + ".knot_rows.json", "w"))
    np.savez_compressed(DIET_STATES_BREATHS_PATH, states_all=states_all, admissible=admissible,
                         n=n, K_B=K_B, C=C,
                         meta=json.dumps(dict(source=DIET_SLICE, generated=time.strftime("%Y-%m-%d %H:%M:%S"))))
    P(f"[build] per-breath content library: {n_slots_b} slots x {K_B} breaths -> {CONTENT_BREATHS_LIB_PATH}")
    P(f"[build] raw diet per-breath states cached -> {DIET_STATES_BREATHS_PATH} "
      f"(states_all {states_all.shape} float16, admissible {int(admissible.sum())}/{n})")
    # ---- VERIFICATION: the per-breath library's kb=K_B-1 cell == the final-only library's cell ----
    P(f"[build] VERIFY: per-breath library kb={final_kb} vs the final-only library (cosine, expect ~1.0000):")
    for k in KINDS:
        kk, kb_key = f"kind:{k}", f"kind:{k}@{final_kb}"
        if kk in content_lib and kb_key in content_breaths_lib:
            ma, mb = content_lib[kk].mean, content_breaths_lib[kb_key].mean
            c = float((ma @ mb) / (np.linalg.norm(ma) * np.linalg.norm(mb) + 1e-12))
            na, nb = content_lib[kk].n, content_breaths_lib[kb_key].n
            P(f"    {k:10} cos={c:.6f}  n(final-only)={na:.0f} n(per-breath@{final_kb})={nb:.0f} "
              f"{'MATCH' if na == nb and c > 0.9999 else 'MISMATCH -- investigate'}")

    # ---- RETINA-SPACE LIBRARY: clause embeddings, prose rows only, nl_atlas_clause's own machinery ----
    import nl_atlas_clause as NAC   # row_factor_records, embed_and_pool, l2norm, gen_src/is_prose (DEV=CPU asserted inside)
    import jury_features as JF

    prose_idx = [i for i in range(n) if NAC.is_prose(vs[i]) and admissible[i]]
    P(f"[build] retina space: {len(prose_idx)}/{n} diet-slice rows are prose (gsm8k/svamp/asdiv)")
    records = []
    for i in prose_idx:
        row = vs[i]
        knot = row_knot.get(i) or MR.coarse_knot(row)
        recs, text = NAC.row_factor_records(row)
        for idx, fac, windows, payload, roles in recs:
            if windows is None:
                continue
            k = gold_kind(fac, idx)
            records.append(dict(text=text, windows=windows, kind=k, knot=knot, row=i))
    all_texts = list({r["text"] for r in records})
    P(f"[build] retina space: {len(records)} resolved-clause factors over {len(all_texts)} unique texts; embedding (frozen trunk, CPU)...")
    trunk_cache = JF.embed_texts(all_texts, batch_size=32, log=P)
    retina_lib = WelfordLibrary()
    retina_knot_rows = collections.defaultdict(set)
    D_RET = 2048
    for r in records:
        entry = trunk_cache[r["text"]]
        vec = JF.pool_states(entry, r["windows"])
        retina_lib.get_or_create(f"kind:{r['kind']}", dim=D_RET).update(vec)
        retina_lib.get_or_create(f"knot:{r['knot']}", dim=D_RET).update(vec)
        retina_knot_rows[f"knot:{r['knot']}"].add(r["row"])
    retina_lib.save(RETINA_LIB_PATH, extra_meta=json.dumps(dict(tag="PMS8_241", space="retina",
                     source=DIET_SLICE, n_rows=len(prose_idx), n_factors=len(records), dim=D_RET,
                     generated=time.strftime("%Y-%m-%d %H:%M:%S"))))
    json.dump({k: len(v) for k, v in retina_knot_rows.items()}, open(RETINA_LIB_PATH + ".knot_rows.json", "w"))
    rkind_n = {k: retina_lib[f"kind:{k}"].n for k in KINDS if f"kind:{k}" in retina_lib}
    knot_cells_r = [k for k in retina_lib if k.startswith("knot:")]
    knot_multi_row_r = [k for k in knot_cells_r if len(retina_knot_rows[k]) >= MIN_ROWS_KNOT]
    P(f"[build] retina library: {len(records)} factors over {len(prose_idx)} prose rows -> {RETINA_LIB_PATH}")
    P(f"[build] retina kind counts: {rkind_n}")
    P(f"[build] retina coarse knots: {len(knot_cells_r)} distinct; {len(knot_multi_row_r)} fed by >={MIN_ROWS_KNOT} distinct rows")
    P(f"[build] done ({time.time()-t0:.0f}s)")


# ======================================================================================
# per-key distances/cosines against a loaded library (shared by read() and drift())
# ======================================================================================
def maha_lite(mean, var, x, eps=1e-6):
    d = x - mean[None, :]
    return np.sqrt((d * d / (var[None, :] + eps)).sum(-1))


def cos_to(mean, x):
    num = x @ mean
    den = np.linalg.norm(x, axis=-1) * (np.linalg.norm(mean) + 1e-12) + 1e-12
    return num / den


def nearest_knot_cos(lib, x, lib_path, min_n=MIN_N_KNOT, min_rows=MIN_ROWS_KNOT):
    rows_path = lib_path + ".knot_rows.json"
    knot_rows = json.load(open(rows_path)) if os.path.exists(rows_path) else {}
    keys = [k for k in lib if k.startswith("knot:") and lib[k].n >= min_n
            and knot_rows.get(k, 0) >= min_rows]
    if not keys:
        return np.full(len(x), np.nan), 0
    means = np.stack([lib[k].mean for k in keys], axis=0)
    means_u = means / (np.linalg.norm(means, axis=-1, keepdims=True) + 1e-12)
    xu = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)
    sims = xu @ means_u.T
    return sims.max(1), len(keys)


def per_slot_features(lib, kind, X, lib_path):
    maha = np.full(len(X), np.nan)
    cosk = np.full(len(X), np.nan)
    for k in KINDS:
        key = f"kind:{k}"
        if key not in lib:
            continue
        sel = kind == k
        if not sel.any():
            continue
        cell = lib[key]
        maha[sel] = maha_lite(cell.mean, cell.variance, X[sel])
        cosk[sel] = cos_to(cell.mean, X[sel])
    cosn, n_knots = nearest_knot_cos(lib, X, lib_path)
    return maha, cosk, cosn, n_knots


# ======================================================================================
# ACT 2: READ -- the wild holdout, once
# ======================================================================================
def read_wild():
    import phase1_algebra_head as H
    bands, clock_dims = H._hier_band_dims()
    CONTENT = np.sort(np.concatenate(bands))

    rows = load_jsonl(WILD_PATH)
    assert len(rows) == 311, len(rows)
    for r in rows:
        assert isinstance(row_gold(r), int)

    # ---- enumerate every gold slot once (kind + knot), independent of any one body's dump ----
    ri, ji, kind, knot = [], [], [], []
    row_knot = {}
    for r_idx, row in enumerate(rows):
        facs = row["factors"]
        k_ = row_knot.get(r_idx)
        if k_ is None:
            k_ = MR.coarse_knot(row)
            row_knot[r_idx] = k_
        for j, fac in enumerate(facs):
            ri.append(r_idx); ji.append(j); kind.append(gold_kind(fac, j)); knot.append(k_)
    ri, ji, kind, knot = np.array(ri), np.array(ji), np.array(kind), np.array(knot)
    P(f"[read] wild gold slots enumerated: {len(ri)} (expect 2051)")

    ps = np.load(".cache/ps_legal_wild_PMS8_241.npz")
    okmap = {(int(r), int(j)): bool(o) for r, j, o in zip(ps["rows"], ps["slots"], ps["ok"])}
    ok = np.array([okmap[(r, j)] for r, j in zip(ri, ji)])
    assert len(ok) == len(ps["rows"]), (len(ok), len(ps["rows"]))

    tel = np.load(".cache/perceiver_telemetry_PMS8_241_wildhold.npz", allow_pickle=True)
    ent_final = tel["ent"][:, 6, :]               # K_B=7 breaths, index 6 = final
    right_final = tel["right_final"]              # (311,24) in {-1,0,1}
    pres_tel = tel["pres_gold"] > 0
    assert int(pres_tel.sum()) == len(ri), (int(pres_tel.sum()), len(ri))
    agree = (right_final[ri, ji] == np.where(ok, 1, 0)).mean()
    P(f"[read] ps_legal_wild vs perceiver_telemetry right_final agreement on gold slots: {agree:.4f}")
    ent = ent_final[ri, ji]

    court = pickle.load(open(".cache/courtroom_PMS8_241.pkl", "rb"))
    row_correct = np.zeros(311, dtype=bool)
    for res in court["final"]["results"]:
        row_correct[res["i"]] = bool(res["top1"]["correct"])
    P(f"[read] courtroom row verdicts: {int(row_correct.sum())}/311 correct (expect 14)")

    # ---- CONTENT SPACE ----
    z = np.load(".cache/clock_band_states_PMS8_241.npz")
    S = z["states"]                 # (311, 6, L_FAC, 512), breath index 0..5 = loop breaths 1..6
    X_content = S[ri, :, ji, :][:, 5, CONTENT].astype(np.float32)   # final loop breath, content dims
    lib_c = WelfordLibrary.load(CONTENT_LIB_PATH)
    maha_c, cosk_c, cosn_c, nk_c = per_slot_features(lib_c, kind, X_content, CONTENT_LIB_PATH)
    P(f"[read] content space: {nk_c} legible knot centroids (n>={MIN_N_KNOT}) out of the diet library")

    # ---- RETINA SPACE ----
    import nl_atlas_clause as NAC
    import jury_features as JF
    rec_by_rowslot = {}
    for r_idx, row in enumerate(rows):
        recs, text = NAC.row_factor_records(row)
        for idx, fac, windows, payload, roles in recs:
            rec_by_rowslot[(r_idx, idx)] = (text, windows)
    all_texts = list({rec_by_rowslot[(r, j)][0] for r, j in zip(ri, ji)})
    P(f"[read] retina space: embedding {len(all_texts)} unique wild texts (frozen trunk, CPU)...")
    trunk_cache = JF.embed_texts(all_texts, batch_size=32, log=P)
    X_retina = np.zeros((len(ri), 2048), np.float32)
    for n_i, (r, j) in enumerate(zip(ri, ji)):
        text, windows = rec_by_rowslot[(r, j)]
        entry = trunk_cache[text]
        X_retina[n_i] = JF.pool_states(entry, windows)
    lib_r = WelfordLibrary.load(RETINA_LIB_PATH)
    maha_r, cosk_r, cosn_r, nk_r = per_slot_features(lib_r, kind, X_retina, RETINA_LIB_PATH)
    P(f"[read] retina space: {nk_r} legible knot centroids (n>={MIN_N_KNOT}) out of the diet library")

    # ---- AUROC table, per-slot ----
    P("\n" + "=" * 100)
    P("THE WELFORD ATLAS READ -- wild, per-slot AUROC vs correctness (n=%d gold slots; %d right)"
      % (len(ok), int(ok.sum())))
    P("=" * 100)
    a_ent, n1, n0 = auroc(ok, -ent)
    P(f"  {'feature':38} {'AUROC':>8} {'n_pos':>7} {'n_neg':>7}")
    P(f"  {'entropy baseline (membrane, final breath)':38} {a_ent:8.4f} {n1:7d} {n0:7d}")
    results = {}
    for space, (maha, cosk, cosn) in (("content", (maha_c, cosk_c, cosn_c)), ("retina", (maha_r, cosk_r, cosn_r))):
        a_maha, _, _ = auroc(ok, -maha)
        a_cosk, _, _ = auroc(ok, cosk)
        a_cosn, _, _ = auroc(ok, cosn)
        results[space] = dict(maha=a_maha, cosk=a_cosk, cosn=a_cosn)
        P(f"  {space+' maha-lite (own kind)':38} {a_maha:8.4f} {n1:7d} {n0:7d}")
        P(f"  {space+' cos to own kind mean':38} {a_cosk:8.4f} {n1:7d} {n0:7d}")
        P(f"  {space+' cos to nearest knot mean':38} {a_cosn:8.4f} {n1:7d} {n0:7d}")

    bar_met = any((v["maha"] - a_ent >= 0.03) or (v["cosk"] - a_ent >= 0.03) or (v["cosn"] - a_ent >= 0.03)
                   for v in results.values())
    best = max([("entropy", a_ent)] + [(f"{s}/{f}", v[f]) for s, v in results.items() for f in v], key=lambda t: t[1])
    P(f"\n  THE BAR (AUROC >= entropy + 0.03 in at least one space): {'MET' if bar_met else 'MISSED'} "
      f"(best = {best[0]} {best[1]:.4f} vs entropy {a_ent:.4f}, diff {best[1]-a_ent:+.4f})")

    # ---- per-row means vs row correctness ----
    P("\n" + "-" * 100)
    P("PER-ROW (mean distance/cosine over the row's gold slots) vs row correctness (courtroom top1)")
    P("-" * 100)
    row_feats = {}
    for space, (maha, cosk, cosn) in (("content", (maha_c, cosk_c, cosn_c)), ("retina", (maha_r, cosk_r, cosn_r))):
        for name, arr in (("maha", maha), ("cosk", cosk), ("cosn", cosn)):
            per_row = np.full(311, np.nan)
            for r_idx in range(311):
                sel = ri == r_idx
                if sel.any():
                    per_row[r_idx] = np.nanmean(arr[sel])
            row_feats[f"{space}/{name}"] = per_row
    row_ent = np.full(311, np.nan)
    for r_idx in range(311):
        sel = ri == r_idx
        if sel.any():
            row_ent[r_idx] = np.nanmean(ent[sel])
    a_row_ent, rn1, rn0 = auroc(row_correct, -row_ent)
    P(f"  {'feature':38} {'AUROC':>8} {'n_correct':>10} {'n_wrong/refused':>16}")
    P(f"  {'entropy (row mean)':38} {a_row_ent:8.4f} {rn1:10d} {rn0:16d}")
    for name, arr in row_feats.items():
        sign = -1.0 if "maha" in name else 1.0
        a, _, _ = auroc(row_correct, sign * arr)
        P(f"  {name:38} {a:8.4f} {rn1:10d} {rn0:16d}")

    return dict(a_ent=a_ent, results=results, bar_met=bar_met, best=best)


# ======================================================================================
# ACT 3: DRIFT -- per-kind mean direction + Welford radius, per body, pairwise cosines
# ======================================================================================
def drift_monitor():
    import phase1_algebra_head as H
    bands, clock_dims = H._hier_band_dims()
    CONTENT = np.sort(np.concatenate(bands))

    rows = load_jsonl(WILD_PATH)
    ri, ji, kind = [], [], []
    for r_idx, row in enumerate(rows):
        for j, fac in enumerate(row["factors"]):
            ri.append(r_idx); ji.append(j); kind.append(gold_kind(fac, j))
    ri, ji, kind = np.array(ri), np.array(ji), np.array(kind)

    P("\n" + "=" * 100)
    P("THE DRIFT MONITOR -- per-kind mean direction + Welford radius, per body (wild, final breath)")
    P("=" * 100)
    available = [t for t in DRIFT_TAGS if os.path.exists(f".cache/clock_band_states_{t}.npz")]
    missing = [t for t in DRIFT_TAGS if t not in available]
    if missing:
        P(f"  MISSING (no banked clock_band_states_<tag>.npz, not fabricated): {missing}")
        if "RK_241" not in missing:
            P("  NOTE: RK_241 was requested by the ask but is not among the missing list check -- investigate.")

    cells = {}   # tag -> {kind -> Welford}
    for tag in available:
        z = np.load(f".cache/clock_band_states_{tag}.npz")
        S = z["states"]
        assert S.shape[0] == 311 and S.shape[1] == 6, (tag, S.shape)
        X = S[ri, :, ji, :][:, 5, CONTENT].astype(np.float32)
        lib = WelfordLibrary()
        for k in KINDS:
            sel = kind == k
            if sel.sum() < MIN_N_KIND:
                continue
            lib.get_or_create(f"kind:{k}", dim=len(CONTENT)).update(X[sel])
        cells[tag] = lib

    kinds_present = sorted({k[5:] for lib in cells.values() for k in lib if k.startswith("kind:")})
    P(f"\n  bodies read: {available}; kinds with n>={MIN_N_KIND} slots on every body: "
      f"{[k for k in kinds_present if all(f'kind:{k}' in cells[t] for t in available)]}")

    P(f"\n  {'tag':12} {'kind':10} {'n':>6} {'radius(cos2mu)':>16}")
    for tag in available:
        for k in KINDS:
            key = f"kind:{k}"
            if key not in cells[tag]:
                continue
            cell = cells[tag][key]
            # radius = mean cosine of members to the mean -- recomputed directly off this body's
            # own X (not the running Welford mean mid-stream), consistent with the task's own
            # definition ("the mean cosine of members to the mean").
            sel = kind == k
            z = np.load(f".cache/clock_band_states_{tag}.npz")
            X = z["states"][ri, :, ji, :][:, 5, CONTENT].astype(np.float32)[sel]
            rad = float(cos_to(cell.mean, X).mean())
            P(f"  {tag:12} {k:10} {int(cell.n):6d} {rad:16.4f}")

    P(f"\n  PAIRWISE COSINE between bodies' per-kind mean directions (the rotation CLAUDE.md S4 warns about):")
    for k in KINDS:
        tags_k = [t for t in available if f"kind:{k}" in cells[t]]
        if len(tags_k) < 2:
            continue
        P(f"    kind={k}:")
        for a_i in range(len(tags_k)):
            for b_i in range(a_i + 1, len(tags_k)):
                ta, tb = tags_k[a_i], tags_k[b_i]
                ma, mb = cells[ta][f"kind:{k}"].mean, cells[tb][f"kind:{k}"].mean
                c = float((ma @ mb) / (np.linalg.norm(ma) * np.linalg.norm(mb) + 1e-12))
                P(f"      {ta:12} vs {tb:12}  cos={c:.4f}")
    return dict(available=available, missing=missing)


# ======================================================================================
# ACT 4: THE CODEBOOK DISTRIBUTION + JS (2026-10-07, Bryce: "a composite key (centroid, breath) in
# the atlas = the expected flight path per breath"; "Jensen-Shannon divergence on the DISTRIBUTION
# OVER CENTROIDS ... never over the 512 dims"). Per gold slot, per breath: p_b = softmax(cos(state_b,
# mean[kind,b]) / tau) over the KINDS present at that breath (the per-breath library above); features
# built from p_b: the cosine to the slot's OWN kind (the flight path itself), the entropy of p_b (how
# undecided the reading is), the JS to the PREVIOUS breath's p (the movement) and the JS to the
# kind's own EXPECTED p at that breath -- a diet-only average over admissible diet slots of that
# kind, never wild, never a read-time "win" (the deviation from the flight path). tau is tuned ONCE
# (tune_tau, diet only, before any wild touch) and then reused unchanged for both the diet's own
# feature file and the wild read.
# ======================================================================================
TAU_GRID = [0.01, 0.02, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]


def _kind_means_by_breath(lib, K_B, kinds=KINDS, min_n=MIN_N_KIND):
    out = {}
    for kb in range(K_B):
        present = [k for k in kinds if f"kind:{k}@{kb}" in lib and lib[f"kind:{k}@{kb}"].n >= min_n]
        means = (np.stack([lib[f"kind:{k}@{kb}"].mean for k in present], axis=0).astype(np.float32)
                 if present else None)
        out[kb] = (present, means)
    return out


def _softmax_p(X, means, tau):
    """X: (...,C), means: (K,C) -> (p (...,K) softmax(cos/tau), cos (...,K)) -- cosine (both sides
    L2-normalized), never a raw dot product (states are not unit-norm)."""
    Xu = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-12)
    mu = means / (np.linalg.norm(means, axis=-1, keepdims=True) + 1e-12)
    cos = Xu @ mu.T
    z = cos / tau
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return (e / (e.sum(axis=-1, keepdims=True) + 1e-12)).astype(np.float32), cos.astype(np.float32)


def _diet_rows_and_states():
    from phase1_algebra_head import load_alg
    os.environ.setdefault("ALG_TEST", DIET_SLICE)
    os.environ.setdefault("ALG_TEST_NAME", "pm35cslicevalid2")
    vs, vst, vtk, vg, vse = load_alg("test")
    z = np.load(DIET_STATES_BREATHS_PATH)
    states_all = z["states_all"].astype(np.float32)   # (n, K_B, L_FAC, C)
    admissible = z["admissible"]
    assert states_all.shape[0] == len(vs), (states_all.shape, len(vs))
    return vs, vg, states_all, admissible


def tune_tau(content_breaths_lib, K_B):
    """THE ONE TAU-TUNING PASS (diet only, before any wild touch): own-kind softmax probability at
    the FINAL breath vs right_final, AUROC-maximising over TAU_GRID -- the raw cosine's own AUROC is
    tau-invariant (a monotone rescaling of one number), so the tau-SENSITIVE quantity this rule tunes
    is the softmax PROBABILITY MASS on the slot's own gold kind (which depends on every kind's
    cosine, not just the own one, once more than two kinds compete for the normalization). Falls back
    to 0.1 if the sweep's AUROC spread is < 0.01 (flat -- no informative tau to pick)."""
    vs, vg, states_all, _ = _diet_rows_and_states()
    n = len(vs)
    final_kb = K_B - 1
    present, means = _kind_means_by_breath(content_breaths_lib, K_B)[final_kb]
    assert means is not None, "tune_tau: no kind means at the final breath"
    tel = np.load(".cache/perceiver_telemetry_PMS8_241_pm35cslicevalid2.npz", allow_pickle=True)
    right_final = tel["right_final"]
    assert right_final.shape[0] == n, (right_final.shape, n)

    ri, ji, kind = [], [], []
    for i in range(n):
        facs = vs[i]["factors"]
        for j in range(min(states_all.shape[2], len(facs))):
            if vg["presence"][i, j] <= 0.5:
                continue
            ri.append(i); ji.append(j); kind.append(gold_kind(facs[j], j))
    ri, ji, kind = np.array(ri), np.array(ji), np.array(kind)
    y = right_final[ri, ji]
    keep = y >= 0
    ri, ji, kind, y = ri[keep], ji[keep], kind[keep], y[keep].astype(bool)
    X = states_all[ri, final_kb, ji, :]
    kidx = {k: ii for ii, k in enumerate(present)}
    own = np.array([kidx.get(k, -1) for k in kind])
    has_own = own >= 0

    results = []
    for tau in TAU_GRID:
        p, _ = _softmax_p(X, means, tau)
        p_own = np.where(has_own, p[np.arange(len(p)), np.clip(own, 0, None)], np.nan)
        a, _, _ = auroc(y[has_own], p_own[has_own])
        results.append((tau, a))
    aurocs = [a for _, a in results if np.isfinite(a)]
    spread = (max(aurocs) - min(aurocs)) if aurocs else 0.0
    if spread < 0.01 or not aurocs:
        tau_star, rule = 0.1, f"sweep flat (spread {spread:.4f} < 0.01) -> fixed 0.1"
    else:
        tau_star = max(results, key=lambda t: (t[1] if np.isfinite(t[1]) else -1.0))[0]
        rule = f"argmax over the sweep (spread {spread:.4f})"
    return tau_star, results, rule


def build_q_table(content_breaths_lib, K_B, tau):
    """q[kb] = (present_kinds, {kind: mean p_b vector}) -- the diet's OWN expected codebook reading
    per kind per breath, admissible diet rows only (the SAME membership the library's means were
    built from); THE reference point THE JS-deviation feature reads against. Diet-only by
    construction -- never touches wild."""
    vs, vg, states_all, admissible = _diet_rows_and_states()
    n = len(vs)
    kmb = _kind_means_by_breath(content_breaths_lib, K_B)
    q = {}
    for kb in range(K_B):
        present, means = kmb[kb]
        if means is None:
            q[kb] = (present, {})
            continue
        acc = {k: [] for k in present}
        X_kb = states_all[:, kb]
        for i in range(n):
            if not admissible[i]:
                continue
            facs = vs[i]["factors"]
            for j in range(min(X_kb.shape[1], len(facs))):
                if vg["presence"][i, j] <= 0.5:
                    continue
                k = gold_kind(facs[j], j)
                if k not in acc:
                    continue
                p, _ = _softmax_p(X_kb[i, j][None, :], means, tau)
                acc[k].append(p[0])
        q[kb] = (present, {k: (np.mean(v, axis=0) if v else None) for k, v in acc.items()})
    return q


def build_atlas_features(side, tau, content_breaths_lib, q_table, K_B):
    """side in {'diet','wild'} -> (n, K_B, L_FAC) atlas_cos / atlas_ent / atlas_js_move /
    atlas_js_dev, written to PERCEIVER_V1B_{DIET,WILD}_FEAT. wild's per-breath states come from the
    ALREADY-BANKED clock_band_states_PMS8_241.npz (loop breaths 1..6 only, zero-GPU read of an
    existing artifact -- kb=0 is honestly NaN for wild, never fabricated); diet's come from the
    DIET_STATES_BREATHS_PATH cache this same run's build() wrote (all 7 breaths)."""
    kmb = _kind_means_by_breath(content_breaths_lib, K_B)
    if side == "diet":
        vs, vg, states_all, _ = _diet_rows_and_states()
        rows = vs
        n = len(rows)
        presence = vg["presence"] > 0.5
        out_path = PERCEIVER_V1B_DIET_FEAT
    else:
        import phase1_algebra_head as H
        bands, clock_dims = H._hier_band_dims()
        CONTENT = np.sort(np.concatenate(bands))
        rows = load_jsonl(WILD_PATH)
        n = len(rows)
        zc = np.load(".cache/clock_band_states_PMS8_241.npz")
        S = zc["states"].astype(np.float32)   # (n, 6, L_FAC, 512), loop breaths 1..6
        assert S.shape[0] == n, (S.shape, n)
        L_FAC_w = S.shape[2]
        states_all = np.full((n, K_B, L_FAC_w, len(CONTENT)), np.nan, np.float32)
        for bb in range(S.shape[1]):
            kb = bb + 1
            if kb < K_B:
                states_all[:, kb] = S[:, bb][:, :, CONTENT]
        tel = np.load(".cache/perceiver_telemetry_PMS8_241_wildhold.npz", allow_pickle=True)
        presence = tel["pres_gold"] > 0
        out_path = PERCEIVER_V1B_WILD_FEAT

    L_FAC = states_all.shape[2]
    C = states_all.shape[-1]
    kind_of = np.full((n, L_FAC), "", dtype=object)
    for i in range(n):
        facs = rows[i]["factors"]
        for j in range(min(L_FAC, len(facs))):
            if presence[i, j]:
                kind_of[i, j] = gold_kind(facs[j], j)

    atlas_cos = np.full((n, K_B, L_FAC), np.nan, np.float32)
    atlas_ent = np.full((n, K_B, L_FAC), np.nan, np.float32)
    atlas_js_move = np.full((n, K_B, L_FAC), np.nan, np.float32)
    atlas_js_dev = np.full((n, K_B, L_FAC), np.nan, np.float32)
    p_prev = None
    present_prev = None
    mism_move = 0
    for kb in range(K_B):
        present, means = kmb[kb]
        if means is None:
            p_prev, present_prev = None, None
            continue
        X = states_all[:, kb].reshape(-1, C)
        valid = np.isfinite(X).all(-1)
        p = np.full((X.shape[0], len(present)), np.nan, np.float32)
        cosv = np.full((X.shape[0], len(present)), np.nan, np.float32)
        if valid.any():
            pv, cv = _softmax_p(X[valid], means, tau)
            p[valid], cosv[valid] = pv, cv
        p = p.reshape(n, L_FAC, len(present))
        cosv = cosv.reshape(n, L_FAC, len(present))
        kidx = {k: ii for ii, k in enumerate(present)}
        q_present, q_map = q_table[kb]
        for i in range(n):
            for j in range(L_FAC):
                k = kind_of[i, j]
                if not k or np.isnan(p[i, j]).any():
                    continue
                atlas_ent[i, kb, j] = float(-(p[i, j] * np.log(p[i, j] + 1e-12)).sum())
                if k in kidx:
                    atlas_cos[i, kb, j] = cosv[i, j, kidx[k]]
                qv = q_map.get(k) if k in (q_present or []) else None
                if qv is not None and len(qv) == len(present):
                    atlas_js_dev[i, kb, j] = js_divergence(p[i, j], qv)
                if p_prev is not None and present_prev == present and not np.isnan(p_prev[i, j]).any():
                    atlas_js_move[i, kb, j] = js_divergence(p[i, j], p_prev[i, j])
        if p_prev is not None and present_prev != present:
            mism_move += 1
        p_prev, present_prev = p, present
    if mism_move:
        P(f"[jsfeat] {side}: {mism_move} breath-pairs had a DIFFERENT present-kind set -- js_move left NaN there")

    np.savez(out_path, atlas_cos=atlas_cos, atlas_entropy=atlas_ent, atlas_js_move=atlas_js_move,
             atlas_js_dev=atlas_js_dev, tau=np.float32(tau), K_B=K_B, n=n,
             meta=json.dumps(dict(side=side, tau=float(tau), generated=time.strftime("%Y-%m-%d %H:%M:%S"))))
    P(f"[jsfeat] {side}: wrote {out_path} (atlas_cos/entropy/js_move/js_dev, shape ({n},{K_B},{L_FAC}))")
    return out_path


def jsfeat():
    if not os.path.exists(CONTENT_BREATHS_LIB_PATH):
        raise SystemExit(f"welford_atlas jsfeat: {CONTENT_BREATHS_LIB_PATH} missing -- run `build` first")
    import phase1_algebra_head as H
    K_B = int(os.environ.get("ALG_BREATH", "7"))
    lib = WelfordLibrary.load(CONTENT_BREATHS_LIB_PATH)
    tau_star, results, rule = tune_tau(lib, K_B)
    P("\n" + "=" * 100)
    P("THE CODEBOOK DISTRIBUTION + JS -- tau tuning (diet only)")
    P("=" * 100)
    for tau, a in results:
        P(f"  tau={tau:<6} own-kind-probability AUROC (final breath, diet right_final) = {a:.4f}")
    P(f"  TAU* = {tau_star} ({rule})")
    q_table = build_q_table(lib, K_B, tau_star)
    build_atlas_features("diet", tau_star, lib, q_table, K_B)
    build_atlas_features("wild", tau_star, lib, q_table, K_B)
    return tau_star


# ======================================================================================
def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    assert mode in ("build", "read", "drift", "jsfeat", "all"), mode
    P(f"THE WELFORD ATLAS -- PMS8_241 -- mode={mode} -- {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if mode in ("build", "all"):
        build()
    if mode in ("read", "all"):
        if not (os.path.exists(CONTENT_LIB_PATH) and os.path.exists(RETINA_LIB_PATH)):
            raise SystemExit(f"welford_atlas read: libraries not built yet -- run `build` first "
                              f"({CONTENT_LIB_PATH}, {RETINA_LIB_PATH})")
        read_wild()
    if mode in ("drift", "all"):
        drift_monitor()
    if mode in ("jsfeat", "all"):
        jsfeat()
    if mode == "all":
        P("\n" + "=" * 100)
        P("THE SIX-LINE READING")
        P("=" * 100)
        P("(filled in by hand after the read -- see the ledger entry this run's output feeds.)")
    open(OUT_TXT, "a" if mode != "all" else "w").write("\n".join(LOG) + "\n")
    print(f"[welford-atlas] wrote {OUT_TXT}")


if __name__ == "__main__":
    main()
