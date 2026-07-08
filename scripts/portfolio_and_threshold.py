"""portfolio_and_threshold.py — the two cheap reads TTA opened (spec
registration, 2026-07-09 night).

READ 1 — THE PORTFOLIO COMPOSITION (the board's most valuable unmeasured
number): agreement (0.840, behavioral) x waist-distance (0.728, geometric) —
two anomaly signals with plausibly independent failure modes. REGISTERED:
Spearman correlation < 0.4 = effectively independent; combined (rank-sum) AUC
> 0.86 (beats agreement alone) = the portfolio pays; the combined abstention
dial reported at fixed coverage points. Waist scores PERSISTED this time
(.cache/waist_scores_bigtest.npz — the unsaved-scores lesson, third sighting).

READ 2 — THE VOTE-THRESHOLD SWEEP (free from the persisted plurality):
precision/recall at t in {3,4,5} of 5. REGISTERED (relay): unanimity 5/5 is
the 0.99+ certification channel. K>5 extension priced at ~7 min GPU per view;
deferred until this curve says more K pays.

USAGE: DEV=AMD ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest \
           .venv/bin/python3 scripts/portfolio_and_threshold.py
"""
from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import numpy as np

from phase1_algebra_head import L_FAC, build_params, load_alg, ALG_CKPT  # noqa: E402
from waist_abstention_probe import compute_fst, np_heads, slot_kind  # noqa: E402
from survivor_multiplicity import midrank_auc  # noqa: E402

SENT = -10**9


def spearman(a, b):
    def ranks(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x))
        r[order] = np.arange(len(x))
        return r
    ra, rb = ranks(np.asarray(a)), ranks(np.asarray(b))
    ra = (ra - ra.mean()) / (ra.std() + 1e-12)
    rb = (rb - rb.mean()) / (rb.std() + 1e-12)
    return float((ra * rb).mean())


def main():
    from tinygrad.nn.state import safe_load

    samples, states, tokmask, gold, sent = load_alg("test")
    n = len(samples)
    gold_ans = [s["solution"][s["query_var"]] for s in samples]
    aud = np.load(".cache/deploy_audit_bigtest.npz")
    correct = {int(i): int(c) for i, c in zip(aud["idx"], aud["correct"])}
    D = np.load(".cache/tta_arm_D_bigtest.npz")
    va, agree = D["vote_ans"], D["agree"]

    # ---- waist scores, recomputed once and PERSISTED ----
    if os.path.exists(".cache/waist_scores_bigtest.npz"):
        wsc = np.load(".cache/waist_scores_bigtest.npz")["score"]
    else:
        p = build_params(0)
        sd = safe_load(ALG_CKPT)
        for k in p:
            p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
        hd = np_heads(p)
        tr_s, tr_states, tr_tok, _tg, tr_sent = load_alg("train")
        fst_tr = compute_fst(p, tr_states, tr_tok, tr_sent,
                             list(range(len(tr_s))))
        by_kind = {}
        for i in range(len(tr_s)):
            for j in range(L_FAC):
                kind = slot_kind(hd, fst_tr[i, j])
                if kind:
                    by_kind.setdefault(kind, []).append(fst_tr[i, j])
        cent = {k: (lambda c: c / np.linalg.norm(c))(np.mean(v, axis=0))
                for k, v in by_kind.items()}
        fst_te = compute_fst(p, states, tokmask, sent, list(range(n)))
        wsc = np.zeros(n)
        for i in range(n):
            w = 1.0
            for j in range(L_FAC):
                kind = slot_kind(hd, fst_te[i, j])
                if kind is None or kind not in cent:
                    continue
                v = fst_te[i, j]
                w = min(w, float((v / max(np.linalg.norm(v), 1e-9))
                                 @ cent[kind]))
            wsc[i] = 1.0 - w
        np.savez(".cache/waist_scores_bigtest.npz", score=wsc)
        print("[saved] .cache/waist_scores_bigtest.npz")

    # ---- READ 1: portfolio composition on the audit's answered set ----
    ans_idx = sorted(correct)
    dis = np.array([1.0 - agree[i] for i in ans_idx])   # behavioral
    geo = np.array([wsc[i] for i in ans_idx])            # geometric
    lab = np.array([1 - correct[i] for i in ans_idx])    # 1 = committed-wrong
    rho = spearman(dis, geo)
    auc_d = midrank_auc(dis[lab == 1], dis[lab == 0])
    auc_g = midrank_auc(geo[lab == 1], geo[lab == 0])

    def rank_norm(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x))
        r[order] = np.arange(len(x)) / (len(x) - 1)
        return r
    combo = rank_norm(dis) + rank_norm(geo)
    auc_c = midrank_auc(combo[lab == 1], combo[lab == 0])
    print(f"\n=== READ 1: PORTFOLIO (answered n={len(ans_idx)}, "
          f"wrong={int(lab.sum())}) ===")
    print(f"  Spearman(disagreement, waist) = {rho:.3f} (bar <0.4 = "
          f"independent)")
    print(f"  AUC: disagreement {auc_d:.3f} | waist {auc_g:.3f} | "
          f"RANK-SUM COMBO {auc_c:.3f} (bar >0.86)")
    order = np.argsort(-combo)
    base = lab.mean()
    for fr in (0.05, 0.10, 0.20):
        k = int(len(ans_idx) * fr)
        fl = lab[order[:k]]
        kept = np.delete(np.arange(len(ans_idx)), order[:k])
        prec_kept = 1 - lab[kept].mean()
        print(f"  abstain top-{int(fr * 100)}%: flag-precision {fl.mean():.3f}"
              f" | recall {fl.sum() / lab.sum():.3f} | kept "
              f"answered-precision {prec_kept:.3f} (floor 0.823)")

    # ---- READ 2: vote-threshold sweep (free from persisted plurality) ----
    print(f"\n=== READ 2: VOTE THRESHOLD (K=5) ===")
    for t in (3, 4, 5):
        r = w = 0
        for i in range(n):
            if va[i] == SENT or agree[i] * 5 < t - 1e-9:
                continue
            if int(va[i]) == gold_ans[i]:
                r += 1
            else:
                w += 1
        print(f"  t={t}/5: accepted {r + w:4d} | right {r:4d} | precision "
              f"{r / max(r + w, 1):.4f} | coverage {(r + w) / n:.3f}")
    print(f"  (relay bar: unanimity = 0.99+ certification channel; K>5 "
          f"extension ~7 min GPU/view, deferred until this curve says it "
          f"pays)")


if __name__ == "__main__":
    main()
