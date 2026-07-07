"""silhouette_library.py — the SILHOUETTE LIBRARY + Brick-0's analytic arm + the
first parse-side render (spec §13, the Nazaré funnel — registered before build).

(a) PARSE-SIDE LIBRARY (KenKen parse waist, banked memmap + gold spans):
    per-factor-KIND token centroids in the waist space, built on TRAIN, evaluated on
    TEST as per-token SEGMENT-AND-CLASSIFY: cosine-to-centroid argmax vs gold span
    labels, at width 512 AND the 128-d narrow waist (the canyon check). Kinds:
    preamble(row/col), cage-arith, given, distractor/none. Plus the first
    token x kind-similarity render — the parse-side dancer, finally pictured.

(b) DEDUCE-SIDE ANALYTIC ARM of Brick-0 (banked 4-variant trajectory capture):
    prototype matching survives nonlinear composition? Per-variant prototypes from
    HALF the instances' common-mode trajectories; classify the OTHER half's variants
    by cosine. 4-way accuracy vs 0.25 chance. (Linearity is refuted — this measures
    whether matched filters still CLASSIFY even where amplitudes don't add.)

(c) THE ARTIFACT: .cache/silhouette_library_v0.npz — the centroid bank on disk
    (parse-side kind centroids at both widths + deduce-side variant prototypes),
    the registry's learned twin, matchable by anything downstream.

USAGE:  DEV=CPU .venv/bin/python3 scripts/silhouette_library.py
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import numpy as np

KINDS = ["none", "rowcol", "cage", "given"]   # per-token gold kinds (KenKen parse)
LIB_OUT = ".cache/silhouette_library_v0.npz"
RENDER_DIR = ".cache/silhouette_render"


def token_kind_labels(samples, offsets_like_span, gold, sent, tokmask):
    """Per-token kind labels from gold factor spans (already tokenized in gold:
    fspan is (n, L, T) token masks; type is (n, L)). none=0 covers preamble-less
    tokens + distractors."""
    n, L, T = gold["span"].shape
    lab = np.zeros((n, T), np.int8)
    for i in range(n):
        for j in range(L):
            if gold["presence"][i, j] < 0.5:
                continue
            t = int(gold["type"][i, j])          # 0=row 1=col 2=cage
            mask = gold["span"][i, j] > 0
            if t in (0, 1):
                lab[i][mask] = 1                  # rowcol (preamble)
            else:
                is_given = (gold["is_cage"][i, j] > 0.5 and
                            gold["op"][i, j] == 0)
                lab[i][mask] = 3 if is_given else 2
        lab[i][tokmask[i] == 0] = 0
    return lab


def parse_side_library():
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    from phase1_delta_head import (
        build_head_params, head_forward, load_split, CKPT_PATH, H_WAIST)

    p = build_head_params(0)
    sd = safe_load(CKPT_PATH)
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

    def waists_and_labels(split, cap):
        samples, states, tokmask, gold, sent = load_split(split)
        n = min(len(samples), cap)
        lab = token_kind_labels(samples[:n], None,
                                {k: v[:n] for k, v in gold.items()},
                                sent[:n], tokmask[:n])
        W = np.zeros((n, tokmask.shape[1], H_WAIST), np.float16)
        for s0 in range(0, n, 8):
            sl = np.arange(s0, min(s0 + 8, n))
            pad = 8 - len(sl)
            sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
            out = head_forward(
                p, Tensor(np.asarray(states[sl_p], dtype=np.float32), dtype=dtypes.float),
                Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float),
                Tensor(np.ones((1, 1, H_WAIST), np.float32), dtype=dtypes.float),
                Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int))
            w = out["waist"].realize().numpy()
            W[sl] = w[:len(sl)].astype(np.float16)
        return W, lab, tokmask[:n]

    print("[library:parse] computing TRAIN waists (centroid source, n=400)...",
          flush=True)
    Wtr, Ltr, Mtr = waists_and_labels("train", 400)
    print("[library:parse] computing TEST waists (evaluation)...", flush=True)
    Wte, Lte, Mte = waists_and_labels("test", 10 ** 9)

    results, centroids_out = {}, {}
    for width in (512, 128):
        # centroids per kind, in the (possibly narrowed) waist
        cents = np.zeros((len(KINDS), width), np.float32)
        for kd in range(len(KINDS)):
            m = (Ltr == kd) & (Mtr > 0)
            v = Wtr[..., :width][m].astype(np.float32)
            c = v.mean(0)
            cents[kd] = c / (np.linalg.norm(c) + 1e-9)
        centroids_out[width] = cents
        # per-token classification on TEST: cosine argmax
        X = Wte[..., :width].astype(np.float32)
        Xn = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-9)
        sim = Xn @ cents.T                                # (n, T, K)
        pred = sim.argmax(-1)
        valid = Mte > 0
        acc = float((pred[valid] == Lte[valid]).mean())
        maj = max(float((Lte[valid] == kd).mean()) for kd in range(len(KINDS)))
        per_kind = {KINDS[kd]: float((pred[valid & (Lte == kd)] == kd).mean())
                    for kd in range(len(KINDS)) if (valid & (Lte == kd)).any()}
        results[width] = (acc, maj, per_kind)
        print(f"[library:parse] width {width}: per-token kind acc = {acc:.3f} "
              f"(majority floor {maj:.3f}) | per-kind {per_kind}", flush=True)

    # the render: first parse-side silhouette picture (one test instance)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(RENDER_DIR, exist_ok=True)
        i = 0
        T_i = int(Mte[i].sum())
        X = Wte[i, :T_i, :].astype(np.float32)
        Xn = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-9)
        sim = Xn @ centroids_out[512].T
        fig, ax = plt.subplots(2, 1, figsize=(11, 4.6), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1]})
        im = ax[0].imshow(sim.T, aspect="auto", cmap="RdBu_r",
                          vmin=-abs(sim).max(), vmax=abs(sim).max())
        ax[0].set_yticks(range(len(KINDS)))
        ax[0].set_yticklabels(KINDS)
        ax[0].set_title("parse-side silhouette: token x kind-centroid similarity "
                        "(test inst 0)")
        ax[1].imshow(Lte[i, :T_i][None], aspect="auto", cmap="tab10", vmin=0, vmax=9)
        ax[1].set_yticks([])
        ax[1].set_ylabel("gold")
        ax[1].set_xlabel("token position")
        fig.colorbar(im, ax=ax[0], shrink=0.8)
        fig.tight_layout()
        fig.savefig(os.path.join(RENDER_DIR, "parse_silhouette_inst0.png"), dpi=120)
        plt.close(fig)
        print(f"[library:parse] render -> {RENDER_DIR}/parse_silhouette_inst0.png",
              flush=True)
    except Exception as e:
        print(f"[library:parse] render skipped: {e}")
    return centroids_out, results


def deduce_side_brick0():
    """Analytic Brick-0 arm: variant classification by prototype cosine, despite
    refuted linearity. Split-half over the 24 banked instances."""
    z = np.load(".cache/silhouette_traj_kenken_reg.npz")
    variants = ["full", "rowcol", "cage", "base"]
    valid = z["cell_valid"]
    cms = {}
    for v in variants:
        reps = z[f"reps_{v}"].astype(np.float32)          # (n, K, S, H)
        m = valid[:, None, :, None]
        cm = (reps * m).sum(2) / np.maximum(valid.sum(1)[:, None, None], 1)
        cms[v] = cm.reshape(cm.shape[0], -1)              # (n, K*H)
    n = cms["full"].shape[0]
    tr = np.arange(0, n, 2)
    te = np.arange(1, n, 2)
    protos = []
    for v in variants:
        c = cms[v][tr].mean(0)
        protos.append(c / (np.linalg.norm(c) + 1e-9))
    protos = np.stack(protos)
    correct = total = 0
    for vi, v in enumerate(variants):
        X = cms[v][te]
        Xn = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-9)
        pred = (Xn @ protos.T).argmax(-1)
        correct += int((pred == vi).sum())
        total += len(te)
    acc = correct / total
    print(f"[brick-0:analytic] deduce-side variant classification (split-half, "
          f"4-way): {correct}/{total} = {acc:.3f} vs 0.25 chance", flush=True)
    print(f"[brick-0:analytic] read: matched filters {'SURVIVE' if acc > 0.6 else 'STRUGGLE WITH'}"
          f" nonlinear composition at the classification level", flush=True)
    return protos, acc




# ===========================================================================
# --eval-only: saved centroids vs a (hardened) slice — the teeth-robustness check
# ===========================================================================

def eval_only():
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    from phase1_delta_head import (
        build_head_params, head_forward, load_split, CKPT_PATH, H_WAIST)

    lib = np.load(LIB_OUT, allow_pickle=True)
    p = build_head_params(0)
    sd = safe_load(CKPT_PATH)
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    samples, states, tokmask, gold, sent = load_split("test")
    n = len(samples)
    lab = token_kind_labels(samples, None, gold, sent, tokmask)
    W = np.zeros((n, tokmask.shape[1], H_WAIST), np.float16)
    for s0 in range(0, n, 8):
        sl = np.arange(s0, min(s0 + 8, n))
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        out = head_forward(
            p, Tensor(np.asarray(states[sl_p], dtype=np.float32), dtype=dtypes.float),
            Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float),
            Tensor(np.ones((1, 1, H_WAIST), np.float32), dtype=dtypes.float),
            Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int))
        W[sl] = out["waist"].realize().numpy()[:len(sl)].astype(np.float16)
    for width in (512, 128):
        cents = lib[f"parse_centroids_{width}"]
        X = W[..., :width].astype(np.float32)
        Xn = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-9)
        pred = (Xn @ cents.T).argmax(-1)
        valid = tokmask > 0
        acc = float((pred[valid] == lab[valid]).mean())
        maj = max(float((lab[valid] == kd).mean()) for kd in range(len(KINDS)))
        print(f"[teeth-check] width {width}: per-token kind acc = {acc:.3f} "
              f"(majority floor {maj:.3f}) on THIS slice", flush=True)


# ===========================================================================
# --crosscheck: head vs library disagreement as a gold-free suspect signal
# ===========================================================================

def crosscheck():
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    from phase1_delta_head import (
        L_SLOTS, H_WAIST, build_head_params, head_forward, load_split,
        CKPT_PATH, TYPES)
    from phase1_brick_a import NACK_NPZ, wrong_slot_mask
    from phase1_brick_c import slot_confidence

    lib = np.load(LIB_OUT, allow_pickle=True)
    cents = lib["parse_centroids_512"]              # (K_kinds, 512)
    p = build_head_params(0)
    sd = safe_load(CKPT_PATH)
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    samples, states, tokmask, gold, sent = load_split("test")
    z = np.load(NACK_NPZ.format(split="test"))
    fail_i = [i for i in range(len(samples)) if z["has_fail"][i]]

    disagree, confs, wrongs = [], [], []
    for s0 in range(0, len(fail_i), 8):
        sl = np.array(fail_i[s0:s0 + 8])
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        out = head_forward(
            p, Tensor(np.asarray(states[sl_p], dtype=np.float32), dtype=dtypes.float),
            Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float),
            Tensor(np.ones((1, 1, H_WAIST), np.float32), dtype=dtypes.float),
            Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int))
        o = {k: out[k].realize().numpy() for k in
             ("pres", "type", "op", "dig", "mem", "attn_mean", "waist")}
        for bi, i in enumerate(sl):
            i = int(i)
            wrong = wrong_slot_mask(
                {k: o[k][bi][None] for k in
                 ("pres", "type", "op", "dig", "mem")}, gold, i, 0)
            X = o["waist"][bi].astype(np.float32)
            Xn = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-9)
            tok_kind_sim = Xn @ cents.T                        # (T, K_kinds)
            for j in range(L_SLOTS):
                if gold["presence"][i, j] < 0.5 or o["pres"][bi, j] <= 0:
                    continue
                # HEAD's kind for this slot
                ty = int(o["type"][bi, j].argmax())
                if ty in (0, 1):
                    head_kind = 1                              # rowcol
                else:
                    head_kind = 3 if int(o["op"][bi, j].argmax()) == 0 else 2
                # LIBRARY's kind: slot-attention-weighted token similarity
                att = o["attn_mean"][bi, j]
                att = att / (att.sum() + 1e-9)
                slot_sim = att @ tok_kind_sim                  # (K_kinds,)
                lib_kind = int(slot_sim[1:].argmax()) + 1      # exclude 'none'
                disagree.append(float(head_kind != lib_kind))
                confs.append(slot_confidence(
                    {k: o[k][bi] for k in ("pres", "type", "op", "dig", "mem")}, j))
                wrongs.append(bool(wrong[j]))

    disagree = np.array(disagree)
    confs = np.array(confs)
    wrongs = np.array(wrongs)

    def auc(scores, labels):
        # MIDRANK ties (binary/discrete scores would otherwise get arbitrary
        # tie-order and an artifact AUC — caught 2026-07-08)
        pos, neg = scores[labels], scores[~labels]
        if not len(pos) or not len(neg):
            return float("nan")
        allv = np.concatenate([pos, neg])
        order = np.argsort(allv, kind="mergesort")
        r = np.empty(len(allv))
        i = 0
        while i < len(allv):
            j = i
            while j + 1 < len(allv) and allv[order[j + 1]] == allv[order[i]]:
                j += 1
            r[order[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return (r[:len(pos)].mean() - (len(pos) - 1) / 2) / len(neg)

    a_dis = auc(disagree, wrongs)
    a_conf = auc(-confs, wrongs)
    corr = float(np.corrcoef(disagree, -confs)[0, 1])
    combined = -confs + disagree                    # crude sum (mis-weighted: rare
    a_comb = auc(combined, wrongs)                  # binary swamped by conf noise)
    lex = disagree * 10.0 - confs                   # LEXICOGRAPHIC: disagreements
    a_lex = auc(lex, wrongs)                        # first, confidence breaks ties
    print(f"[crosscheck] slots={len(wrongs)} wrong={int(wrongs.sum())} "
          f"disagree-rate={disagree.mean():.3f}")
    print(f"  AUC(disagreement -> wrong)      = {a_dis:.3f}")
    print(f"  AUC(low-confidence -> wrong)    = {a_conf:.3f}   (tier-0 baseline)")
    print(f"  corr(disagree, low-conf)        = {corr:+.3f}   (decorrelation check)")
    print(f"  AUC(crude-sum combiner)         = {a_comb:.3f}   (mis-weighted, kept for honesty)")
    print(f"  AUC(LEXICOGRAPHIC combiner)     = {a_lex:.3f}   (vs 0.613 withholding-order baseline)")


def main():
    cents, parse_results = parse_side_library()
    protos, ded_acc = deduce_side_brick0()
    np.savez_compressed(
        LIB_OUT,
        parse_kinds=np.array(KINDS),
        parse_centroids_512=cents[512], parse_centroids_128=cents[128],
        deduce_variant_protos=protos,
        meta=np.array(str({
            "built": "2026-07-07 night", "spec": "§13",
            "parse_acc": {w: parse_results[w][0] for w in parse_results},
            "deduce_4way": ded_acc})))
    print(f"\n[library] artifact -> {LIB_OUT}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--crosscheck", action="store_true")
    a = ap.parse_args()
    if a.eval_only:
        eval_only()
    elif a.crosscheck:
        crosscheck()
    else:
        main()
