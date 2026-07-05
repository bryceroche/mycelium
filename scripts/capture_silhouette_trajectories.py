"""capture_silhouette_trajectories.py — the FIRST unpooled silhouette capture + the
LINEARITY CHECK + the first rendered picture of the dancer (2026-07-05).

WHAT (CLAUDE.md §8.8 Brick-B protocol, steps 1–2, + the render): capture per-breath,
per-cell, full-1024d readout-point residual trajectories — (K, S, H) per instance, NO
pooling (the earlier dart capture pooled away BOTH axes at capture time, which is why it
was a dead end for segmentation) — from the HEALTHY KenKen ckpt (fg_kenken_k16_reg),
under FOUR factor-membership variants of the SAME puzzles:

    full   : rows + cols + arithmetic cages + given cages   (the natural puzzle)
    rowcol : rows + cols + given cages                      (arithmetic cages DROPPED)
    cage   : arithmetic cages + given cages                 (rows/cols DROPPED)
    base   : given cages only                               (both dropped)

Given-cages and input_cells (the givens evidence) are IDENTICAL across variants, so the
varying constituents are exactly {row/col factors} and {arithmetic cage factors}.
Dropping arithmetic cages happens at the RECORD level (encode_puzzle tolerates uncovered
cells: cell_cage_id stays -1); dropping rows/cols happens by zeroing their membership
rows (the mask builder's SELF-EDGE FIX makes those heads degrade to self-only attention,
NOT uniform attention — the clean "constituent removed" semantics). Solo variants are
OUT-OF-DISTRIBUTION stimuli for the trained model — this probes the dynamical system's
response to constituents, not natural operation. Say so when reading results.

THE LINEARITY CHECK (decides the segmentation branch — do NOT assume superposition):
    f(full) - f(base)  ?≈?  [f(rowcol) - f(base)] + [f(cage) - f(base)]
per breath, over valid cells. Near-linear -> BirdNET-style synthetic mixes + linear
matched filters are justified. Grossly nonlinear -> compose PROBLEMS, not silhouettes,
and the matched-filter story needs a nonlinear front-end.

SCHEMA (maximally raw — capture once, plot forever): fp16 reps (n_inst, K, S, H) per
variant; per-instance metadata (band, N, n_givens, deduction_depth); per-cell arrays
(gold, input_cells, cell_valid, cell_cage_id); per-cage cage_op; per-variant final
argmax + cell_acc; meta json (ckpt, K, capture point = engine fg_resid_capture sink,
i.e. the PRE-readout-LN per-breath residual).

REUSE (parity-validated, not re-implemented): build_kenken_spec /
build_kenken_deducer_model / the fg_resid_capture eager-sink pattern from
scripts/diag_kenken_granularity_probe.py; KenKenLoader encoding verbatim.

USAGE:
  CPU selftest (no GPU, no ckpt):
    .venv/bin/python3 scripts/capture_silhouette_trajectories.py --selftest
  Capture (GPU, eager — ~minutes):
    DEV=AMD .venv/bin/python3 scripts/capture_silhouette_trajectories.py \
        --ckpt .cache/fg_ckpts/fg_kenken_k16_reg/fg_kenken_k16_reg_final.safetensors \
        --test .cache/kenken_test_curriculum.jsonl --per-band 6 --out .cache/silhouette_traj_kenken_reg.npz
  Analyze + render (CPU, reads the npz):
    .venv/bin/python3 scripts/capture_silhouette_trajectories.py \
        --analyze .cache/silhouette_traj_kenken_reg.npz --render-dir .cache/silhouette_render
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_THIS_FILE = os.path.abspath(__file__)
_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(_THIS_FILE))

import numpy as np  # noqa: E402

VARIANTS = ("full", "rowcol", "cage", "base")
# variant -> (keep_arith_cages, keep_rowcol)
VARIANT_DEF = {
    "full":   (True,  True),
    "rowcol": (False, True),
    "cage":   (True,  False),
    "base":   (False, False),
}


# ===========================================================================
# RECORD-LEVEL variant: drop arithmetic cages, keep given cages (evidence const)
# ===========================================================================

def variant_record(rec: dict, keep_arith: bool) -> dict:
    """Return a shallow-copied record with arithmetic cages optionally dropped.

    Given cages (clue op == 'given') are ALWAYS kept — they carry the observed
    cells, which must be identical across variants for the linearity check to be
    about the FACTOR constituents and nothing else.
    """
    if keep_arith:
        return rec
    cages, clues = [], []
    for cage, clue in zip(rec["cages"], rec["clues"]):
        if clue[0] == "given":
            cages.append(cage)
            clues.append(clue)
    r2 = dict(rec)
    r2["cages"] = cages
    r2["clues"] = clues
    return r2


# ===========================================================================
# FORWARD + capture (eager; engine fg_resid_capture sink; membership edit)
# ===========================================================================

def run_capture_forward(model, kb, spec, K: int, keep_rowcol: bool):
    """One eager K-breath forward; returns (reps (K,B,S,H) fp32, pred (B,S) int,
    the per-breath value-logits final tensor argmax convention: value = argmax+1)."""
    from tinygrad import Tensor, dtypes
    from mycelium.factor_graph_engine import (
        make_kenken_factor_batch, factor_breathing_forward,
    )
    from mycelium.kenken import build_verification_inlet
    from mycelium.kenken_data import N_MAX

    Tensor.training = False
    inlet = build_verification_inlet(
        model, kb.cage_op, kb.cage_target, kb.cage_size, kb.cell_cage_id).realize()
    batch = make_kenken_factor_batch(kb, spec, prebuilt_inlet=inlet)

    if not keep_rowcol:
        # Zero the 2*N_MAX row/col latent rows -> those heads become self-only
        # (mask builder's SELF-EDGE FIX), the clean "no coupling" degradation.
        L = int(batch.membership.shape[1])
        keep = np.ones((1, L, 1), dtype=np.float32)
        keep[:, : 2 * N_MAX, :] = 0.0
        batch.membership = (batch.membership
                            * Tensor(keep, dtype=dtypes.float)).contiguous().realize()

    model.fg_resid_capture = []
    try:
        logits_hist, _calib = factor_breathing_forward(model, batch, spec, K)
        final_logits = logits_hist[-1].realize()
        caps = list(model.fg_resid_capture)
    finally:
        model.fg_resid_capture = None  # disarm -> byte-identical no-op state

    reps = np.stack(caps, axis=0).astype(np.float32)          # (K, B, S, H)
    assert reps.shape[0] == K, f"expected {K} captures, got {reps.shape[0]}"
    pred = final_logits.argmax(-1).realize().numpy().astype(np.int32) + 1  # (B,S) 1..N
    return reps, pred


# ===========================================================================
# CAPTURE main
# ===========================================================================

def do_capture(args) -> None:
    from mycelium.kenken_data import KenKenLoader
    from diag_kenken_granularity_probe import (
        build_kenken_spec, build_kenken_deducer_model,
    )

    loader = KenKenLoader(args.test, batch_size=args.batch)
    # Stratify: per_band records from each band, in stable file order.
    by_band: dict[str, list] = {}
    for r in loader.records:
        by_band.setdefault(r.get("band", "all"), []).append(r)
    subset = []
    for band in sorted(by_band, key=lambda b: {"g40": 0, "g30": 1, "g20": 2, "g10": 3}.get(b, 9)):
        subset.extend(by_band[band][: args.per_band])
    n_inst = len(subset)
    print(f"[capture] {n_inst} instances "
          f"({ {b: min(len(v), args.per_band) for b, v in by_band.items()} })", flush=True)

    spec = build_kenken_spec(args.K)
    model = build_kenken_deducer_model(spec, args.ckpt, seed=0)

    out: dict[str, np.ndarray] = {}
    gold_all = valid_all = None
    for vname in VARIANTS:
        keep_arith, keep_rowcol = VARIANT_DEF[vname]
        loader.records = [variant_record(r, keep_arith) for r in subset]
        reps_l, pred_l, meta_rows = [], [], []
        for kb in loader.iter_eval(args.batch):
            reps, pred = run_capture_forward(model, kb, spec, args.K, keep_rowcol)
            reps_l.append(reps.transpose(1, 0, 2, 3))         # (B, K, S, H)
            pred_l.append(pred)
            if vname == "full":
                meta_rows.append(kb)
        reps_v = np.concatenate(reps_l, axis=0)[:n_inst]      # (n, K, S, H)
        pred_v = np.concatenate(pred_l, axis=0)[:n_inst]      # (n, S)
        out[f"reps_{vname}"] = reps_v.astype(np.float16)
        out[f"pred_{vname}"] = pred_v
        if vname == "full":
            gold_l, valid_l, cid_l, inp_l, cop_l = [], [], [], [], []
            band_l, N_l, ngiv_l, depth_l = [], [], [], []
            for kb in meta_rows:
                gold_l.append(kb.gold.numpy());     valid_l.append(kb.cell_valid.numpy())
                cid_l.append(kb.cell_cage_id.numpy()); inp_l.append(kb.input_cells.numpy())
                cop_l.append(kb.cage_op.numpy())
                band_l += kb.band; N_l += kb.N; ngiv_l += kb.n_givens
                depth_l += kb.deduction_depth
            gold_all = np.concatenate(gold_l)[:n_inst].astype(np.int32)
            valid_all = np.concatenate(valid_l)[:n_inst] > 0.5
            out["gold"] = gold_all
            out["cell_valid"] = valid_all
            out["cell_cage_id"] = np.concatenate(cid_l)[:n_inst].astype(np.int32)
            out["input_cells"] = np.concatenate(inp_l)[:n_inst].astype(np.int32)
            out["cage_op"] = np.concatenate(cop_l)[:n_inst].astype(np.int32)
            out["band"] = np.array(band_l[:n_inst])
            out["N"] = np.array(N_l[:n_inst], dtype=np.int32)
            out["n_givens"] = np.array(ngiv_l[:n_inst], dtype=np.int32)
            out["deduction_depth"] = np.array(depth_l[:n_inst], dtype=np.int32)
        acc = float((((pred_v == gold_all) & valid_all).sum()) / valid_all.sum())
        out[f"cell_acc_{vname}"] = np.array(
            [float(((pred_v[i] == gold_all[i]) & valid_all[i]).sum() / valid_all[i].sum())
             for i in range(n_inst)], dtype=np.float32)
        print(f"[capture] variant={vname:6s} reps={reps_v.shape} pooled cell_acc={acc:.3f}",
              flush=True)

    out["meta"] = np.array(json.dumps({
        "ckpt": args.ckpt, "test": args.test, "K": args.K, "date": "2026-07-05",
        "capture_point": "engine fg_resid_capture sink (per-breath PRE-readout-LN residual)",
        "variants": {v: {"keep_arith_cages": VARIANT_DEF[v][0],
                          "keep_rowcol": VARIANT_DEF[v][1]} for v in VARIANTS},
        "note": "given-cages + input_cells identical across variants; solo variants are OOD stimuli",
    }))
    np.savez_compressed(args.out, **out)
    sz = os.path.getsize(args.out) / 1e6
    print(f"[capture] wrote {args.out} ({sz:.1f} MB)", flush=True)


# ===========================================================================
# ANALYSIS — linearity check + renders (pure numpy/matplotlib, CPU)
# ===========================================================================

def cell_kind_labels(cell_cage_id, cage_op, input_cells, cell_valid):
    """Per-cell kind: 0=given, 1=add, 2=sub, 3=mul, 4=div, -1=invalid/uncaged.
    (OP_VOCAB = given,add,sub,mul,div -> ids 0..4.)"""
    n, S = cell_cage_id.shape
    kind = np.full((n, S), -1, dtype=np.int32)
    for i in range(n):
        for s in range(S):
            if not cell_valid[i, s]:
                continue
            if input_cells[i, s] > 0:
                kind[i, s] = 0
            elif cell_cage_id[i, s] >= 0:
                kind[i, s] = int(cage_op[i, cell_cage_id[i, s]])
    return kind


def linearity_table(z, valid):
    """z: dict variant -> (n,K,S,H) fp32. Returns per-breath (ratio, cos) arrays.
    lhs = full-base, rhs = (rowcol-base)+(cage-base), over valid cells only."""
    lhs = z["full"] - z["base"]                                # (n,K,S,H)
    rhs = (z["rowcol"] - z["base"]) + (z["cage"] - z["base"])
    K = lhs.shape[1]
    m = valid[:, None, :, None]                                # (n,1,S,1)
    lhs = lhs * m
    rhs = rhs * m
    ratios, coss = [], []
    for k in range(K):
        a, b = lhs[:, k].ravel(), rhs[:, k].ravel()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        ratios.append(float(np.linalg.norm(a - b) / max(na, 1e-9)))
        coss.append(float(a @ b / max(na * nb, 1e-9)))
    return np.array(ratios), np.array(coss)


def settle_breaths(reps, valid, frac=0.05):
    """Per-cell settle breath from residual deltas: first k after which every
    per-breath delta stays below frac*max_delta(cell). reps (n,K,S,H)."""
    d = np.linalg.norm(np.diff(reps, axis=1), axis=-1)         # (n,K-1,S)
    dmax = d.max(axis=1, keepdims=True)                        # (n,1,S)
    below = d < (frac * np.maximum(dmax, 1e-9))                # (n,K-1,S)
    n, Km1, S = below.shape
    settle = np.zeros((n, S), dtype=np.int32)
    for i in range(n):
        for s in range(S):
            if not valid[i, s]:
                settle[i, s] = -1
                continue
            k = Km1
            for j in range(Km1 - 1, -1, -1):
                if not below[i, j, s]:
                    k = j + 1
                    break
            else:
                k = 0
            settle[i, s] = k
    return settle                                              # in delta-index units


def do_analyze(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(args.analyze, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    valid = z["cell_valid"]
    reps = {v: z[f"reps_{v}"].astype(np.float32) for v in VARIANTS}
    n, K, S, H = reps["full"].shape
    os.makedirs(args.render_dir, exist_ok=True)
    print(f"[analyze] {n} instances, K={K}, S={S}, H={H}; ckpt={meta['ckpt']}")
    for v in VARIANTS:
        print(f"  cell_acc[{v:6s}] mean={float(z[f'cell_acc_{v}'].mean()):.3f}")

    # ---- (1) THE LINEARITY CHECK ----
    ratios, coss = linearity_table(reps, valid)
    print("\n[linearity] lhs=full-base  rhs=(rowcol-base)+(cage-base)  (valid cells)")
    print("  k : residual_ratio  cosine")
    for k in range(K):
        print(f"  {k:2d}:      {ratios[k]:.3f}       {coss[k]:.3f}")
    print(f"  POOLED (all breaths): ratio={float(np.mean(ratios)):.3f} "
          f"cos={float(np.mean(coss)):.3f}")
    verdict = ("NEAR-LINEAR (ratio<=0.3): synthetic mixes + linear matched filters justified"
               if float(np.mean(ratios)) <= 0.3 else
               "MODERATELY NONLINEAR (0.3<ratio<=0.7): compose PROBLEMS, not silhouettes"
               if float(np.mean(ratios)) <= 0.7 else
               "GROSSLY NONLINEAR (ratio>0.7): matched-filter story needs a nonlinear front-end")
    print(f"  VERDICT: {verdict}")

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(ratios, marker="o"); ax[0].set_title("linearity residual ratio / breath")
    ax[0].set_xlabel("breath k"); ax[0].axhline(0.3, ls="--", c="gray"); ax[0].axhline(0.7, ls="--", c="gray")
    ax[1].plot(coss, marker="o", color="tab:green"); ax[1].set_title("cos(lhs, rhs) / breath")
    ax[1].set_xlabel("breath k"); ax[1].set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(os.path.join(args.render_dir, "linearity.png"), dpi=120)
    plt.close(fig)

    # ---- (2) the spectrogram-style render: prototype basis from SOLO common modes ----
    def common_mode(v):
        m = valid[:, None, :, None]
        return (reps[v] * m).sum(axis=2) / np.maximum(valid.sum(axis=1)[:, None, None], 1)  # (n,K,H)

    cm = {v: common_mode(v) for v in VARIANTS}
    solo = np.concatenate([(cm["rowcol"] - cm["base"]).reshape(-1, H),
                           (cm["cage"] - cm["base"]).reshape(-1, H)], axis=0)
    solo = solo - solo.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(solo, full_matrices=False)
    P = Vt[:24]                                                # prototype basis (24,H)

    bands = z["band"]
    picks = []
    for b in ("g40", "g30", "g20", "g10"):
        idx = np.where(bands == b)[0]
        if len(idx):
            picks.append(int(idx[0]))
    for i in picks:
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), sharey=True)
        for j, v in enumerate(VARIANTS):
            img = (cm[v][i] - cm["base"][i]) @ P.T if v != "base" else cm[v][i] @ P.T
            axes[j].imshow(img.T, aspect="auto", cmap="RdBu_r",
                           vmin=-np.abs(img).max(), vmax=np.abs(img).max())
            axes[j].set_title(f"{v} (inst {i}, {bands[i]})", fontsize=9)
            axes[j].set_xlabel("breath k")
        axes[0].set_ylabel("solo-prototype component")
        fig.tight_layout()
        fig.savefig(os.path.join(args.render_dir, f"spectro_inst{i}.png"), dpi=120)
        plt.close(fig)

    # ---- (3) temporal structure: per-cell delta-norm heatmap sorted by cell kind ----
    kind = cell_kind_labels(z["cell_cage_id"], z["cage_op"], z["input_cells"], valid)
    kind_names = {0: "given", 1: "add", 2: "sub", 3: "mul", 4: "div"}
    for i in picks:
        d = np.linalg.norm(np.diff(reps["full"][i], axis=0), axis=-1)  # (K-1,S)
        cells = [s for s in range(S) if valid[i, s]]
        cells.sort(key=lambda s: (kind[i, s], s))
        img = d[:, cells]
        fig, ax = plt.subplots(figsize=(10, 3.4))
        im = ax.imshow(img, aspect="auto", cmap="magma")
        bounds, labels, prev = [], [], None
        for x, s in enumerate(cells):
            if kind[i, s] != prev:
                bounds.append(x); labels.append(kind_names.get(kind[i, s], "?")); prev = kind[i, s]
        for x in bounds[1:]:
            ax.axvline(x - 0.5, color="w", lw=0.8)
        ax.set_xticks(bounds); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("breath k -> k+1"); ax.set_title(f"per-cell |Δresidual| (full), inst {i} ({bands[i]})")
        fig.colorbar(im, shrink=0.85)
        fig.tight_layout()
        fig.savefig(os.path.join(args.render_dir, f"deltas_inst{i}.png"), dpi=120)
        plt.close(fig)

    # ---- (4) settle-breath by cell kind (the "which species calls when" first read) ----
    settle = settle_breaths(reps["full"], valid)
    print("\n[settle] per-cell settle breath (residual-delta based) by cell kind:")
    rows = []
    for kd in (0, 1, 2, 3, 4):
        vals = settle[(kind == kd) & (settle >= 0)]
        if len(vals):
            rows.append((kind_names[kd], len(vals), float(vals.mean()), float(np.median(vals))))
            print(f"  {kind_names[kd]:6s}: n={len(vals):4d} mean={vals.mean():.2f} "
                  f"median={np.median(vals):.1f}")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    data = [settle[(kind == kd) & (settle >= 0)] for kd in (0, 1, 2, 3, 4)]
    ax.boxplot([d if len(d) else [0] for d in data],
               tick_labels=[kind_names[kd] for kd in (0, 1, 2, 3, 4)])
    ax.set_ylabel("settle breath"); ax.set_title("settle breath by cell kind (full variant)")
    fig.tight_layout(); fig.savefig(os.path.join(args.render_dir, "settle_by_kind.png"), dpi=120)
    plt.close(fig)
    print(f"\n[analyze] renders in {args.render_dir}/")


# ===========================================================================
# SELFTEST (CPU, no GPU, no npz)
# ===========================================================================

def selftest() -> None:
    # variant_record: given cages kept, arithmetic dropped
    rec = {"N": 3, "cages": [[[0, 0]], [[0, 1], [0, 2]]],
           "clues": [["given", 2], ["add", 5]], "solution": [[2, 1, 3], [0]*3, [0]*3]}
    r2 = variant_record(rec, keep_arith=False)
    assert len(r2["cages"]) == 1 and r2["clues"][0][0] == "given"
    assert len(rec["cages"]) == 2, "input record must be untouched"
    # linearity_table: perfectly linear synthetic -> ratio 0, cos 1
    n, K, S, H = 2, 3, 4, 5
    rng = np.random.RandomState(0)
    base = rng.randn(n, K, S, H).astype(np.float32)
    a, b = rng.randn(n, K, S, H).astype(np.float32), rng.randn(n, K, S, H).astype(np.float32)
    z = {"base": base, "rowcol": base + a, "cage": base + b, "full": base + a + b}
    valid = np.ones((n, S), dtype=bool)
    ratios, coss = linearity_table(z, valid)
    assert ratios.max() < 1e-5 and coss.min() > 1 - 1e-5, (ratios, coss)
    # and a nonlinear synthetic -> ratio far from 0
    z["full"] = base + 3.0 * a * b
    ratios2, _ = linearity_table(z, valid)
    assert ratios2.min() > 0.5, ratios2
    # settle_breaths: a cell that stops moving at k=1 settles at 1
    reps = np.zeros((1, 4, 2, 3), dtype=np.float32)
    reps[0, 0, 0] = 10.0  # big delta k0->k1, then still
    reps[0, :, 1] = np.arange(4)[:, None]  # keeps moving -> settles at last delta
    s = settle_breaths(reps, np.ones((1, 2), dtype=bool))
    assert s[0, 0] == 1 and s[0, 1] == 3, s
    print("[selftest] OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt", default=".cache/fg_ckpts/fg_kenken_k16_reg/fg_kenken_k16_reg_final.safetensors")
    ap.add_argument("--test", default=".cache/kenken_test_curriculum.jsonl")
    ap.add_argument("--per-band", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--out", default=".cache/silhouette_traj_kenken_reg.npz")
    ap.add_argument("--analyze", default="", help="path to a captured npz -> run analysis instead")
    ap.add_argument("--render-dir", default=".cache/silhouette_render")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return
    if args.analyze:
        do_analyze(args)
        return
    do_capture(args)


if __name__ == "__main__":
    main()
