"""waist_abstention_probe.py — the parse-side waist probe, both halves (spec
registration, 2026-07-09): INTERPOLATION COHERENCE (the corrected 50/50) +
ABSTENTION SEPARATION (the paying customer).

THE SPACE: fst — the algebra head's slot-vector bank (L_FAC x 512), the one
place an interpolated vector flows forward into actual emissions (the
problem-agnostic field heads ftype/islit/op/dig are linear maps from it). NOT
the tap (no decoder — the category error corrected 2026-07-09); NOT
cross-problem token grids (ill-defined). Pointer heads (args/res/query) are
problem-relative and excluded by scope.

HALF 1 — INTERPOLATION (registered 50/50; the coordinate-swap evidence does NOT
bear on convex combination): same-kind slot-vector pairs across problems
(givens matched by decoded digit-length; rels by op), alpha=0.5 midpoints
decoded through the linear heads. PINNED "coherent": midpoint digit/op
sharpness >= 0.8x endpoint sharpness AND midpoint decodes ONE ENDPOINT's
value/op >= 0.5 of pairs. Coherent -> the space is smooth within kind (KL
machinery buys little). Garbage -> measured deficiency for the (per-kind-prior)
VAE arm.

HALF 2 — ABSTENTION (the customer: 226 committed-wrong vs 1051 correct from
the deployment-honest audit): per-kind centroids of emitted slot vectors from
the TRAIN split (deployable labels — the head's own claims); score = min over
emitted slots of cosine(slot vec, claimed-kind centroid); midrank AUC + rare-
flag operating points. REGISTERED: dense AUC 0.55-0.65 (misbindings look
locally normal — weak as a ranker); USABLE-FLAG BAR: precision at top-10%
flag rate >= 2x base rate (base 226/1277 = 0.177, bar 0.354). Per the
portfolio rule the signal is CLASSIFIED ON ARRIVAL: dense ranker vs
rare-precise flag vs dead.

USAGE (after deployment_honest_audit.py has written its npz):
  DEV=AMD ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest \
      .venv/bin/python3 scripts/waist_abstention_probe.py
"""
from __future__ import annotations

import math
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import numpy as np

from phase1_algebra_head import (  # noqa: E402
    T_ALG, L_FAC, N_DIG, H_W, N_HEADS, build_params, load_alg, ALG_CKPT,
)
from survivor_multiplicity import midrank_auc  # noqa: E402


def compute_fst(p, states, tokmask, sent, idx):
    """Replicates forward()'s factor bank up to fst (B, L_FAC, H_W)."""
    from tinygrad import Tensor, dtypes
    out = np.zeros((len(idx), L_FAC, H_W), np.float32)
    for s0 in range(0, len(idx), 8):
        sl = np.array(idx[s0:s0 + 8])
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        trunk = Tensor(states[sl_p].astype(np.float32), dtype=dtypes.float)
        tm = Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float)
        st_ = Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int)
        B = trunk.shape[0]
        waist = (trunk @ p["waist_w"] + p["waist_b"]).gelu() + p["sent_emb"][st_]
        q = p["fq"] @ p["attn_wq"] + p["attn_wq_b"]
        k = waist @ p["attn_wk"] + p["attn_wk_b"]
        v = waist @ p["attn_wv"] + p["attn_wv_b"]
        hd = H_W // N_HEADS
        qh = q.reshape(L_FAC, N_HEADS, hd).transpose(0, 1)
        kh = k.reshape(B, -1, N_HEADS, hd).permute(0, 2, 1, 3)
        vh = v.reshape(B, -1, N_HEADS, hd).permute(0, 2, 1, 3)
        sc = (qh.unsqueeze(0) @ kh.transpose(-2, -1)) / math.sqrt(hd)
        sc = sc.clip(-1e4, 1e4) + (1.0 - tm.reshape(B, 1, 1, -1)) * -1e4
        at = sc.softmax(-1)
        stv = (at @ vh).permute(0, 2, 1, 3).reshape(B, L_FAC, H_W)
        stv = stv @ p["attn_wo"] + p["attn_wo_b"] + p["fq"].unsqueeze(0)
        stv = stv + ((stv @ p["ffn_w1"] + p["ffn_b1"]).gelu()
                     @ p["ffn_w2"] + p["ffn_b2"])
        out[s0:s0 + len(sl)] = stv.realize().numpy()[:len(sl)]
    return out


def np_heads(p):
    """Field heads as numpy mats."""
    g = lambda k: p[k].detach().numpy()
    return {k: (g(f"h_{k}"), g(f"h_{k}_b"))
            for k in ("pres", "ftype", "op", "islit", "dig")}


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def slot_kind(hd, vec):
    """Deployable kind label from the head's own claims."""
    if vec @ hd["pres"][0][:, 0] + hd["pres"][1][0] <= 0:
        return None
    islit = vec @ hd["islit"][0][:, 0] + hd["islit"][1][0] > 0
    if islit:
        return "given"
    op = (vec @ hd["op"][0] + hd["op"][1]).argmax()
    return f"rel_{'add' if op == 0 else 'mul'}"


def decode_value(hd, vec):
    dig = softmax((vec @ hd["dig"][0] + hd["dig"][1]).reshape(N_DIG, 10))
    val = int("".join(str(d) for d in dig.argmax(-1)))
    sharp = float(dig.max(-1).mean())
    return val, sharp


def main():
    from tinygrad.nn.state import safe_load

    p = build_params(0)
    sd = safe_load(ALG_CKPT)
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    hd = np_heads(p)

    # ---- centroids + interpolation pairs from the TRAIN split ----
    tr_s, tr_states, tr_tok, _tg, tr_sent = load_alg("train")
    fst_tr = compute_fst(p, tr_states, tr_tok, tr_sent, list(range(len(tr_s))))
    by_kind, given_pool, rel_pool = {}, [], []
    for i in range(len(tr_s)):
        for j in range(L_FAC):
            vec = fst_tr[i, j]
            kind = slot_kind(hd, vec)
            if kind is None:
                continue
            by_kind.setdefault(kind, []).append(vec)
            if kind == "given":
                val, sharp = decode_value(hd, vec)
                given_pool.append((vec, val, sharp))
            else:
                rel_pool.append((vec, kind))
    cent = {k: np.mean(v, axis=0) for k, v in by_kind.items()}
    cent = {k: c / np.linalg.norm(c) for k, c in cent.items()}
    print(f"[waist] train slots per kind: " +
          " ".join(f"{k}={len(v)}" for k, v in sorted(by_kind.items())))

    # ---- HALF 1: interpolation coherence (registered 50/50) ----
    rng = np.random.RandomState(0)
    n_pairs, match, sharp_ratio, ep_sharp, mid_sharp = 0, 0, [], [], []
    by_len = {}
    for vec, val, sharp in given_pool:
        by_len.setdefault(len(str(val)), []).append((vec, val, sharp))
    for L, pool in by_len.items():
        if len(pool) < 2:
            continue
        for _ in range(min(300, len(pool))):
            a, b = rng.randint(0, len(pool), 2)
            if a == b or pool[a][1] == pool[b][1]:
                continue
            va, vb = pool[a], pool[b]
            mid = 0.5 * (va[0] + vb[0])
            mval, msharp = decode_value(hd, mid)
            es = 0.5 * (va[2] + vb[2])
            n_pairs += 1
            match += int(mval in (va[1], vb[1]))
            sharp_ratio.append(msharp / max(es, 1e-9))
            ep_sharp.append(es)
            mid_sharp.append(msharp)
    # islit consistency at midpoints
    print(f"\n  HALF 1 — interpolation (given slots, digit head), "
          f"n={n_pairs} same-length pairs")
    print(f"  endpoint sharpness {np.mean(ep_sharp):.3f} | midpoint "
          f"{np.mean(mid_sharp):.3f} | ratio {np.mean(sharp_ratio):.3f} "
          f"(bar 0.80)")
    print(f"  midpoint-decodes-an-endpoint: {match / max(n_pairs, 1):.3f} "
          f"(bar 0.50)")
    coherent = (np.mean(sharp_ratio) >= 0.80 and
                match / max(n_pairs, 1) >= 0.50)
    print(f"  VERDICT: {'COHERENT — smooth within kind; KL machinery buys '
          'little' if coherent else 'NOT coherent — measured deficiency for '
          'the per-kind-prior arm'}")

    # ---- HALF 2: abstention separation (the paying customer) ----
    aud = np.load(".cache/deploy_audit_bigtest.npz")
    te_s, te_states, te_tok, _g2, te_sent = load_alg("test")
    idx = [int(i) for i in aud["idx"]]
    fst_te = compute_fst(p, te_states, te_tok, te_sent, idx)
    correct = {int(i): int(c) for i, c in zip(aud["idx"], aud["correct"])}
    scores, labels = [], []
    for r, i in enumerate(idx):
        worst = 1.0
        for j in range(L_FAC):
            vec = fst_te[r, j]
            kind = slot_kind(hd, vec)
            if kind is None or kind not in cent:
                continue
            c = float((vec / max(np.linalg.norm(vec), 1e-9)) @ cent[kind])
            worst = min(worst, c)
        scores.append(1.0 - worst)          # higher = more anomalous
        labels.append(1 - correct[i])       # 1 = committed-wrong
    scores, labels = np.array(scores), np.array(labels)
    auc = midrank_auc(scores[labels == 1], scores[labels == 0])
    base = labels.mean()
    print(f"\n  HALF 2 — abstention: answered={len(idx)} wrong={labels.sum()} "
          f"(base {base:.3f})")
    print(f"  dense AUC (midrank) = {auc:.3f}  (registered prior 0.55-0.65)")
    order = np.argsort(-scores)
    for fr in (0.05, 0.10, 0.20):
        k = int(len(idx) * fr)
        fl = labels[order[:k]]
        print(f"  flag top-{int(fr * 100)}%: precision {fl.mean():.3f} | "
              f"recall {fl.sum() / labels.sum():.3f}  "
              f"{'(>= 2x base — USABLE FLAG)' if fl.mean() >= 2 * base else ''}")
    print(f"\n  PORTFOLIO CLASSIFICATION on arrival: AUC>=0.70 dense ranker; "
          f"precision@10% >= {2 * base:.3f} rare-flag; neither = dead.")


if __name__ == "__main__":
    main()
