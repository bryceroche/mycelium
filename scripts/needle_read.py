"""needle_read.py — RUNG 2 OF THE LADDER: THE DISPERSION NEEDLE
(2026-09-07). Zero-training, zero-gradient GPU READ. The registered
question (Tolstoy / happy families): do CORRECT readings of a problem
stay close to their kind's per-breath road in the atlas, while WRONG
readings DISPERSE?

The atlas (mycelium/step_atlas.py) already banks both halves of the
answer: a Welford MEAN (the road) and a Welford VAR (its width), per
(breath_step, class), on both charts — the math chart (mean-pooled
factor-slot states, the COMMITMENT) and the NL chart (attention-pooled
waisted token states, the READING). This reader measures, per item and
per breath step s, three distances from the item's OWN class cell:

  (a) z-radius (math)  sqrt(mean_d(((x - mean)/sqrt(var + 1e-6))^2))
      — diagonal Mahalanobis: distance in units of the kind's own
      per-coordinate spread (this is what the VAR bank is FOR).
  (b) cosine distance (math) to the class mean — consult()'s own
      metric, the variance-blind control for (a).
  (c) z-radius (NL) against nl_means/nl_vars — the same needle held
      against the READING instead of the COMMITMENT.

...and asks whether each separates WRONG parses from CORRECT ones:
rank-based AUC with WRONG as the POSITIVE class, so AUC > 0.5 means
"further from the road predicts a wrong parse".

PINNED BARS (registered BEFORE measurement, printed beside every
number): AUC >= 0.70 at the metric's best step PASSES; AUC < 0.60 at
the best step is the KILL — "the atlas is not a dispersion meter at
this granularity". 0.60 <= AUC < 0.70 = INCONCLUSIVE (neither bar).

CORRECTNESS = loop_val.py's fac-exact criterion, imported from
step_engine_read.score_batch (meter-divergence law: one criterion, one
organ, never a reimplementation), computed on the SAME pass whose
states are pooled — the miner's masked pass 2. An item is CORRECT iff
every present gold factor is realized exactly (n_ok == n_tot).

THE PASS = mine_step_atlas.py's cycle, replicated exactly: pass 1
unmasked parse -> build_slot_masks + alt2_fact_buf (live facts) ->
masked pass 2 with fact_buf; ALG_MINE_BREATHS=1 hands back
out["breaths_all"] (K=7 raw per-breath slot states) and out["nl_all"].
POOLING = the miner's: br[s][bi].mean(0) over the 24 factor slots for
the math side; nl[s][bi] as-is for the NL side (the tap pooled it in
graph). Atlas coordinates therefore match what was mined.

ERA (the never-mix-generations law): the atlas is loaded through
step_atlas.load_atlas's LOUD DOOR against the RESEARCH manifest —
.cache/step_atlas_fed.npz is stamped to sharp_fedon242.safetensors, so
NR_CKPT must be that checkpoint or the door refuses. The door is never
bypassed here.

THE SEAL FENCE (found at build time, CPU N=6, and the reason SC_EVAL is
set here): .cache/step_atlas_fed.npz was mined by fed_mind_chain.sh
step 10 with SC_EVAL=0 — the OPEN shelf-circle. With ALG_SHELF_CIRCLE=2
and SC_EVAL UNSET, forward()'s _SEV data buffer defaults to 1.0 and the
residual is SEVERED at breath SC_KB=4, so breaths 4-6 walk a different
road entirely: measured on 6 wildhold rows, z-radius jumps 0.9-1.8 ->
67-101 and cosine distance exceeds 1.0 (anti-aligned) at steps 4-6,
versus a smooth contracting curve (cos 0.088 -> 0.074) under SC_EVAL=0.
Reading a sealed pass against an open atlas is the never-mix-coordinates
law's doorway, so this reader declares SC_EVAL=0 (the miner's own env)
via setdefault and says so loudly at startup; a caller who overrides it
gets a printed warning, never a silent cross-regime read.

GOODHART FENCE: every number this script prints is DIAGNOSTIC. Not a
loss, not a training signal, not a selection criterion for training
data — a supervised dispersion needle teaches the parse to hug the
road while staying wrong. Read it; never train toward it.

Envs: NR_TEST = wild (default, .cache/wild_admitted_holdout.jsonl /
wildhold) | mint (.cache/algebra_nl_test.jsonl / test23);
NR_N (default 0 = the whole fixture; >0 = first N rows, deterministic);
NR_CKPT (default .cache/sharp_fedon242.safetensors); NR_ATLAS /
NR_ATLAS_MANIFEST (default the fed pair); NR_OUT (default
.cache/needle_read_<fixture>.npz). Champion env stack via setdefault —
a caller's/chain's explicit envs WIN (the wave_field_check lesson).
GPU script; `import needle_read` is CPU-safe (all work under main()).
"""
import os
import sys

_FIXTURES = {
    "wild": (".cache/wild_admitted_holdout.jsonl", "wildhold"),
    "mint": (".cache/algebra_nl_test.jsonl", "test23"),
}
NR_TEST = os.environ.get("NR_TEST", "wild")
_fx_path, _fx_name = _FIXTURES.get(NR_TEST,
                                   (NR_TEST, os.path.basename(NR_TEST)))

# THE CHAMPION ENV STACK — verbatim from stamp_amplitude_read.py's ENV
# dict (the fed-era read stack), MINUS ALG_PC_MIX (this is not a
# pressure-cooker read: no seal, no _PCV buffer) and WITHOUT touching
# SC_EVAL (neither set nor popped — whatever the caller declares
# stands). Applied via setdefault so a chain's explicit envs win.
_CHAMPION_ENVS = {
    "DEV": "PCI+AMD",
    "ALG2": "1", "ALG_FTYPES": "9", "ALG_DUP": "1", "ALG_HW": "512",
    "ALG_WIDE": "1", "ALG_BREATH": "7", "ALG_NOTEBOOK": "1",
    "ALG_SIXWAVE": "1", "NB_PERSLOT": "1", "ALG_BINDBUS": "7",
    "ALG_BIND_D": "512", "BIND_CODES": ".cache/bindbus_codes512.npz",
    "ALG_BUSGARAGE": "2", "ALG_SHELF_CIRCLE": "2", "ALG_ALTMASK": "1",
    "ALG_ALT21": "1", "ALG_ALT2": "1", "ALG_MASKHEAD": "1",
    "ALG_FED": "1",
    # SC_EVAL=0 — the ATLAS'S OWN mining env (fed_mind_chain.sh step
    # 10), NOT a copy of stamp_amplitude_read's pop (that pop exists
    # only because ALG_PC_MIX forbids the var, and this is not a
    # pressure-cooker read). See THE SEAL FENCE in the docstring:
    # unset here would seal the shelf circle at breath 4 and put
    # breaths 4-6 in coordinates the atlas never mined.
    "SC_EVAL": "0",
    "ALG_TEST": _fx_path, "ALG_TEST_NAME": _fx_name,
    # the taps: breaths_all (math chart) + nl_all (the reading)
    "ALG_MINE_BREATHS": "1",
}
for _k, _v in _CHAMPION_ENVS.items():
    os.environ.setdefault(_k, _v)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

CKPT = os.environ.get("NR_CKPT", ".cache/sharp_fedon242.safetensors")
ATLAS = os.environ.get("NR_ATLAS", ".cache/step_atlas_fed.npz")
MANIFEST = os.environ.get("NR_ATLAS_MANIFEST",
                          ".cache/RESEARCH_MANIFEST_FED.json")
NR_N = int(os.environ.get("NR_N", "0"))
OUT = os.environ.get("NR_OUT", f".cache/needle_read_{NR_TEST}.npz")
BATCH = 8

# THE PINNED BARS (registered before measurement; never bent after)
BAR_PASS = 0.70
BAR_KILL = 0.60

METRICS = (("z_math", "z-radius (math)"),
           ("cos_math", "cos-dist (math)"),
           ("z_nl", "z-radius (NL)"))


def rank_auc(scores, is_wrong):
    """Rank-based AUC (Mann-Whitney U, average ranks on ties), POSITIVE
    class = WRONG. AUC > 0.5 <=> larger score predicts a wrong parse.
    Non-finite scores are dropped (dead cells are never zero-filled).
    Returns (auc, n_wrong_used, n_correct_used)."""
    import numpy as np
    s = np.asarray(scores, np.float64)
    y = np.asarray(is_wrong, bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    n_w = int(y.sum())
    n_c = int((~y).sum())
    if n_w == 0 or n_c == 0:
        return float("nan"), n_w, n_c
    order = np.argsort(s, kind="mergesort")
    sr = s[order]
    r = np.empty(len(sr), np.float64)
    i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        r[i:j + 1] = (i + j) / 2.0 + 1.0        # 1-based average rank
        i = j + 1
    ranks = np.empty(len(sr), np.float64)
    ranks[order] = r
    auc = (ranks[y].sum() - n_w * (n_w + 1) / 2.0) / (n_w * n_c)
    return float(auc), n_w, n_c


def verdict(auc):
    if auc != auc:                                    # NaN
        return "NO-READ"
    if auc >= BAR_PASS:
        return "PASS"
    if auc < BAR_KILL:
        return "KILL"
    return "inconc"


def main():
    import numpy as np
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    from phase1_algebra_head import (build_params, forward, load_alg,
                                     build_slot_masks, alt2_fact_buf,
                                     K_VARS)
    from mycelium.step_atlas import load_atlas, atlas_class, K_STEPS
    from step_engine_read import score_batch      # loop_val's criterion

    # THE LOUD DOOR first (era stamp vs the research manifest): a
    # refusal here must be reported verbatim, never bypassed.
    atlas = load_atlas(ATLAS, manifest_path=MANIFEST)
    assert atlas["nl_means"] is not None and atlas["nl_vars"] is not None \
        and atlas["nl_counts"] is not None, \
        ("needle_read needs the PAIRED atlas (nl chart) for metric (c) "
         "— re-mine with the paired miner; single-chart atlas refused")
    acls = {c: i for i, c in enumerate(atlas["classes"])}

    vs, vst, vtk, vg, vse = load_alg("test")
    n_all = len(vs)
    n_take = min(NR_N, n_all) if NR_N > 0 else n_all
    take = np.arange(n_take)
    print(f"[needle] fixture={NR_TEST}({_fx_name}) rows={n_take}/{n_all} "
          f"ckpt={CKPT}", flush=True)
    print(f"[needle] atlas={ATLAS} era={atlas['stamp']} "
          f"manifest={MANIFEST} classes={atlas['classes']}", flush=True)
    _sce = os.environ.get("SC_EVAL", "")
    print(f"[needle] shelf circle: ALG_SHELF_CIRCLE="
          f"{os.environ.get('ALG_SHELF_CIRCLE', '')} SC_EVAL={_sce!r} "
          f"(the atlas was mined at SC_EVAL='0', OPEN)", flush=True)
    if _sce != "0":
        print("[needle] WARNING: SC_EVAL != '0' — the shelf circle "
              "seals at breath SC_KB and breaths >= SC_KB leave the "
              "atlas's mined coordinates (the seal fence). Distances "
              "past the seal are NOT comparable to these pages.",
              flush=True)
    print(f"[needle] BARS (pinned before measurement): best-step AUC >= "
          f"{BAR_PASS:.2f} PASSES; < {BAR_KILL:.2f} KILLS "
          f"('the atlas is not a dispersion meter at this granularity')",
          flush=True)

    p = build_params(0)
    sd = safe_load(CKPT)
    assert set(sd.keys()) == set(p.keys()), \
        (sorted(set(sd) - set(p))[:4], sorted(set(p) - set(sd))[:4])
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

    z_math = np.full((n_take, K_STEPS), np.nan)
    c_math = np.full((n_take, K_STEPS), np.nan)
    z_nl = np.full((n_take, K_STEPS), np.nan)
    correct = np.zeros(n_take, np.int8)
    cls_idx = np.full(n_take, -1, np.int64)
    usable = np.zeros(n_take, bool)          # class on atlas, cells live
    n_offatlas = n_deadcell = n_nogold = 0

    for s0 in range(0, len(take), BATCH):
        sl = take[s0:s0 + BATCH]
        pad = BATCH - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        ts = Tensor(vst[sl_p].astype(np.float32), dtype=dtypes.float)
        tk = Tensor(vtk[sl_p].astype(np.float32), dtype=dtypes.float)
        se = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)
        # ---- the miner's cycle, verbatim: pass 1 open -> masks+facts
        o0 = forward(p, ts, tk, se)
        onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
        mk = build_slot_masks(onp0, vse[sl_p].astype(np.int32))
        _ka = (("pres", "ftype", "op", "dig")
               + (("dup",) if "dup" in o0 else ()))
        _oa = {**onp0, **{k: o0[k].realize().numpy() for k in _ka}}
        _nv = np.array([vs[int(i)].get("n_vars", K_VARS) for i in sl_p])
        _ma = np.array([vs[int(i)].get("m", 0) for i in sl_p])
        fb = alt2_fact_buf(_oa, vse[sl_p], _nv, _ma)
        # ---- pass 2: the breathing walk (states AND correctness)
        o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float),
                    fact_buf=Tensor(fb, dtype=dtypes.float))
        _ks = ("pres", "ftype", "op", "dig", "args", "res") \
            + (("dup",) if "dup" in o else ())
        onp = {k: o[k].realize().numpy() for k in _ks}
        br = [b.realize().numpy() for b in o["breaths_all"]]
        nl = [t.realize().numpy() for t in o["nl_all"]]
        assert len(br) == K_STEPS, \
            f"breaths_all has {len(br)} steps, atlas wants {K_STEPS}"
        assert len(nl) == K_STEPS, \
            f"nl_all has {len(nl)} steps, atlas wants {K_STEPS}"

        for bi, i in enumerate(sl):        # pads (bi >= len(sl)) skipped
            i = int(i)
            # correctness from THIS pass, via loop_val's own criterion:
            # one-row slice so score_batch's bi maps to this item
            n_ok, n_tot = score_batch(
                {k: v[bi:bi + 1] for k, v in onp.items()}, vg, [i])
            if n_tot == 0:
                n_nogold += 1
                continue
            correct[i] = int(n_ok == n_tot)
            cls = atlas_class(vs[i].get("gen"))
            ci = acls.get(cls)
            if ci is None:
                n_offatlas += 1
                continue
            cls_idx[i] = ci
            live = ((atlas["counts"][:, ci] > 0)
                    & (atlas["nl_counts"][:, ci] > 0))
            if not live.all():
                n_deadcell += 1                # some page has no road
            if not live.any():
                continue
            usable[i] = True
            for s in range(K_STEPS):
                if not live[s]:
                    continue                   # dead cell stays NaN
                x = br[s][bi].mean(0).astype(np.float64)
                m = atlas["means"][s, ci].astype(np.float64)
                v = atlas["vars"][s, ci].astype(np.float64)
                z_math[i, s] = np.sqrt(
                    np.mean(((x - m) / np.sqrt(v + 1e-6)) ** 2))
                c_math[i, s] = 1.0 - float(
                    np.dot(x, m) / ((np.linalg.norm(x) + 1e-9)
                                    * (np.linalg.norm(m) + 1e-9)))
                xn = nl[s][bi].astype(np.float64)
                mn = atlas["nl_means"][s, ci].astype(np.float64)
                vn = atlas["nl_vars"][s, ci].astype(np.float64)
                z_nl[i, s] = np.sqrt(
                    np.mean(((xn - mn) / np.sqrt(vn + 1e-6)) ** 2))
        if (s0 // BATCH) % 8 == 0:
            print(f"[needle] {s0 + len(sl)}/{len(take)}", flush=True)

    # ------------------------------------------------------------------
    sel = usable
    n_use = int(sel.sum())
    is_wrong = (correct[sel] == 0)
    print(f"[needle] items: {n_use} usable / {n_take} read  "
          f"(off-atlas class {n_offatlas}, no live cell / partial "
          f"{n_deadcell}, no gold factors {n_nogold})")
    if n_use == 0:
        print("[needle] NO USABLE ITEMS — every row's class was "
              "off-atlas or had no live cell (loud, not a silent table)")
        return
    print(f"[needle] correct={int((~is_wrong).sum())} "
          f"wrong={int(is_wrong.sum())} "
          f"(fac-exact, loop_val criterion, masked pass 2)")

    cols = {"z_math": z_math[sel], "cos_math": c_math[sel],
            "z_nl": z_nl[sel]}
    aucs = {k: np.full(K_STEPS, np.nan) for k in cols}
    ns = {k: np.zeros((K_STEPS, 2), np.int64) for k in cols}
    for k, arr in cols.items():
        for s in range(K_STEPS):
            a, nw, nc = rank_auc(arr[:, s], is_wrong)
            aucs[k][s] = a
            ns[k][s] = (nc, nw)

    print(f"[needle] AUC (WRONG = positive class; >0.5 = farther from "
          f"the road predicts a wrong parse)")
    print(f"[needle] {'step':>4}  {'z_math':>8} {'cos_math':>9} "
          f"{'z_nl':>8}   {'n_corr':>6} {'n_wrong':>7}")
    for s in range(K_STEPS):
        nc, nw = ns["z_math"][s]
        cells = []
        for k, w in (("z_math", 8), ("cos_math", 9), ("z_nl", 8)):
            a = aucs[k][s]
            cells.append(f"{a:>{w}.4f}" if a == a else f"{'nan':>{w}}")
        print(f"[needle] {s:>4}  " + " ".join(cells)
              + f"   {nc:>6} {nw:>7}")

    best = {}
    for k, label in METRICS:
        a = aucs[k]
        if np.all(np.isnan(a)):
            best[k] = (None, float("nan"))
            print(f"[needle] {label:18s} best step: NO READ (all NaN)")
            continue
        s_best = int(np.nanargmax(a))
        best[k] = (s_best, float(a[s_best]))
        vd = verdict(a[s_best])
        arr = cols[k][:, s_best]
        med_c = float(np.nanmedian(arr[~is_wrong]))
        med_w = float(np.nanmedian(arr[is_wrong]))
        print(f"[needle] {label:18s} best step {s_best}  "
              f"AUC={a[s_best]:.4f}  [{vd}]  "
              f"(bars: >={BAR_PASS:.2f} pass / <{BAR_KILL:.2f} kill)  "
              f"median correct={med_c:.4f} wrong={med_w:.4f}")

    np.savez(OUT, z_math=z_math, cos_math=c_math, z_nl=z_nl,
             correct=correct, usable=usable, cls_idx=cls_idx,
             row_idx=np.arange(n_take),
             classes=np.array([str(c) for c in atlas["classes"]]),
             auc_z_math=aucs["z_math"], auc_cos_math=aucs["cos_math"],
             auc_z_nl=aucs["z_nl"],
             era_stamp=np.array(atlas["stamp"]),
             ckpt=np.array(CKPT), fixture=np.array(NR_TEST))
    print(f"[needle] per-item arrays -> {OUT}")

    hdr = "  ".join(
        f"{k}=best s{best[k][0] if best[k][0] is not None else '-'} "
        f"AUC {best[k][1]:.4f} [{verdict(best[k][1])}]"
        for k, _ in METRICS)
    overall = ("PASS" if any(verdict(best[k][1]) == "PASS" for k, _ in METRICS)
               else "KILL" if all(verdict(best[k][1]) == "KILL"
                                  for k, _ in METRICS)
               else "INCONCLUSIVE")
    print(f"[needle] {NR_TEST}/{_fx_name} n={n_use} "
          f"corr={int((~is_wrong).sum())} wrong={int(is_wrong.sum())} | "
          f"{hdr} | VERDICT {overall}")


if __name__ == "__main__":
    main()
