"""tta_views.py — TEST-TIME AUGMENTATION over solution-preserving views (spec
registration, 2026-07-09 — the MC-pi darts, aimed at the routing wall).

THE MECHANISM CLAIM: the 396's pointer misbinds SPECIFIC SURFACE FORMS,
deterministically. Change the surface, and surface-keyed errors land
elsewhere — the Anna Karenina vote: happy parses agree (the correct graph is
the same under every rendering), unhappy parses disagree in different ways.
TTA sidesteps the pointer instead of steering it — the only mechanism class
with any right to touch the routing wall (the beacon tremor's cousin, done
with symmetry instead of perturbation).

DEPLOYMENT-HONESTY (the flaw named before firing): re-rendering needs GOLD
factors — oracle machinery at MATH-500 time (the graph is what parsing is FOR;
re-rendering the parser's own graph correlates with its errors). Two arms:
  ARM O (oracle ceiling): K=4 gold re-renders — shuffled letters, fresh
    templates/surfaces/order. The mechanism's potential.
  ARM D (deployable):     K=4 sentence permutations — pure text transform,
    graph-free (first/last sentence pinned, middle shuffled; the parser's
    sentence-index features genuinely shift).

THE MC-PI GATE (per arm, measured FIRST): on originals whose forced answer is
WRONG, the rate of views forcing the SAME wrong answer must be < 0.30 —
correlated darts estimate nothing; fail = the pointer keys deeper than
surface and the arm dies honestly in one table.

REGISTERED (before firing):
  - Relay: ARM O voting recovers a NONZERO slice of the routing-wall
    population (the 460 / the 90 invisibles); agreement-AUC on
    committed-wrongs lands near/above 0.728 (combinable under the portfolio
    rule — behavioral stability vs representation geometry).
  - Mine: ARM O decorrelates (<0.30 same-wrong); ARM D WEAKER (0.30-0.60 —
    misbinding is plausibly local to sentence content, which permutation
    preserves); voting net-positive in O, ~flat in D.
  - Imposter discipline (standard): vote-accepted right/wrong split reported;
    votes use majority >=3 of 5 (original + 4 views), forced answers only.
  - Views render teeth-easy (oblique/distractor OFF, letters+order shuffled):
    solution-preserving by construction; easier views are in-family (the
    training dial was randomized per-sample).

USAGE: DEV=AMD ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest \
           .venv/bin/python3 scripts/tta_views.py
"""
from __future__ import annotations

import copy
import os
import random
import re
import sys
from collections import Counter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import numpy as np

from phase1_algebra_head import (  # noqa: E402
    T_ALG, build_params, forward, load_alg, decode, sent_indices,
    ALG_CKPT, TOKENIZER_JSON,
)
from algebra_nl_gen import render  # noqa: E402
from beacon_closing_arm import recompute_states  # noqa: E402
from repair_replace_swap import solve_forced  # noqa: E402
from survivor_multiplicity import midrank_auc  # noqa: E402

K_VIEWS = 4


def oracle_view(smp, seed):
    rng = random.Random(seed)
    facs = copy.deepcopy(smp["factors"])
    for f in facs:
        f.pop("spans", None)
        if f["ftype"] == "rel":
            f["surface"] = (rng.choice(["add", "sub"]) if f["op"] == "add"
                            else "mul")
    text, _gf, _m = render(rng, smp["n_vars"], facs, smp["query_var"],
                           shuffle=True, shuffle_letters=True,
                           oblique_prob=0.0, distractor_prob=0.0)
    return text


def permuted_view(text, seed):
    rng = random.Random(seed)
    parts = re.split(r"(?<=\.)\s+", text.strip())
    if len(parts) <= 3:
        return text
    mid = parts[1:-1]
    rng.shuffle(mid)
    return " ".join([parts[0]] + mid + [parts[-1]])


def main():
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(TOKENIZER_JSON)
    samples, states, tokmask, gold, sent = load_alg("test")
    n = len(samples)
    prof = np.load(".cache/survivor_profile_bigtest.npz")
    surv = set(int(i) for i, s in zip(prof["idx"], prof["status"]) if s == 2)
    aud = np.load(".cache/deploy_audit_bigtest.npz")
    answered = {int(i): int(c) for i, c in zip(aud["idx"], aud["correct"])}
    stage0 = {int(i) for i, s in zip(aud["idx"], aud["stage"]) if s == 0}

    p = build_params(0)
    sd = safe_load(ALG_CKPT)
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

    def parse_answers(sts, msk, snt):
        ans = {}
        for s0 in range(0, n, 8):
            sl = np.arange(s0, min(s0 + 8, n))
            pad = 8 - len(sl)
            sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
            out = forward(p, Tensor(sts[sl_p].astype(np.float32),
                                    dtype=dtypes.float),
                          Tensor(msk[sl_p].astype(np.float32),
                                 dtype=dtypes.float),
                          Tensor(snt[sl_p].astype(np.int32),
                                 dtype=dtypes.int))
            o = {k: out[k].realize().numpy() for k in
                 ("pres", "ftype", "op", "islit", "dig", "args", "res",
                  "query")}
            for bi, i in enumerate(sl):
                facs, q = decode({k: o[k][bi] for k in o})
                ans[int(i)] = solve_forced(facs, q, samples[int(i)])
        return ans

    def views_answers(texts):
        ids = np.zeros((n, T_ALG), np.int32)
        msk = np.zeros((n, T_ALG), np.float32)
        snt = np.zeros((n, T_ALG), np.int32)
        skipped = 0
        for i, t in enumerate(texts):
            e = tok.encode(t)
            if len(e.ids) > T_ALG:
                skipped += 1
                ids[i] = 0
                continue
            ids[i, :len(e.ids)] = e.ids
            msk[i, :len(e.ids)] = 1.0
            snt[i] = sent_indices(t, list(e.offsets), msk[i])
        sts = recompute_states(ids)
        a = parse_answers(sts, msk, snt)
        return a, skipped

    print(f"[tta] view 0 (original) ...")
    ans0 = parse_answers(states, tokmask, sent)
    gold_ans = {i: samples[i]["solution"][samples[i]["query_var"]]
                for i in range(n)}

    for arm, mk in (("O (oracle re-render)",
                     lambda i, k: oracle_view(samples[i], 1000 * k + i)),
                    ("D (sentence permutation)",
                     lambda i, k: permuted_view(samples[i]["text"],
                                                1000 * k + i))):
        view_ans = [ans0]
        for k in range(1, K_VIEWS + 1):
            texts = [mk(i, k) for i in range(n)]
            a, skipped = views_answers(texts)
            view_ans.append(a)
            print(f"  arm {arm[:1]} view {k}: forced "
                  f"{sum(1 for v in a.values() if v is not None)}/{n}"
                  f"{f' (skipped {skipped})' if skipped else ''}")

        # MC-PI GATE: decorrelation on wrong-forced originals
        wrong0 = [i for i in range(n)
                  if ans0[i] is not None and ans0[i] != gold_ans[i]]
        same = diff = right = unforced = 0
        for i in wrong0:
            for k in range(1, K_VIEWS + 1):
                a = view_ans[k][i]
                if a is None:
                    unforced += 1
                elif a == ans0[i]:
                    same += 1
                elif a == gold_ans[i]:
                    right += 1
                else:
                    diff += 1
        tot = len(wrong0) * K_VIEWS
        print(f"\n=== ARM {arm} ===")
        print(f"  MC-PI GATE (n={len(wrong0)} wrong-forced originals x "
              f"{K_VIEWS} views): same-wrong {same / tot:.3f} (gate <0.30) | "
              f"diff-wrong {diff / tot:.3f} | right {right / tot:.3f} | "
              f"unforced {unforced / tot:.3f}")
        gate = same / tot < 0.30

        # VOTING: majority >=3 of 5 forced answers
        vote_right = vote_wrong = abstain = 0
        rec_surv = rec_inv = 0
        agree_score = {}
        for i in range(n):
            votes = [view_ans[k][i] for k in range(K_VIEWS + 1)
                     if view_ans[k][i] is not None]
            if votes:
                top, cnt = Counter(votes).most_common(1)[0]
                agree_score[i] = cnt / (K_VIEWS + 1)
            else:
                agree_score[i] = 0.0
            if not votes or Counter(votes).most_common(1)[0][1] < 3:
                abstain += 1
                continue
            top = Counter(votes).most_common(1)[0][0]
            if top == gold_ans[i]:
                vote_right += 1
                if i in surv:
                    rec_surv += 1
                if i in stage0 and not answered.get(i, 1):
                    rec_inv += 1
            else:
                vote_wrong += 1
        print(f"  VOTING (>=3/5): right {vote_right} | wrong {vote_wrong} | "
              f"abstain {abstain} -> end-to-end {vote_right}/1500 = "
              f"{vote_right / n:.3f} (floor 0.701) | answered-precision "
              f"{vote_right / max(vote_right + vote_wrong, 1):.3f} "
              f"(floor 0.823)")
        print(f"  routing-wall recoveries: {rec_surv}/460 survivors | "
              f"{rec_inv}/90 invisibles (relay bar: nonzero)")

        # AGREEMENT-AS-ABSTENTION on the audit's answered set
        s_w = np.array([1 - agree_score[i] for i in answered
                        if not answered[i]])
        s_c = np.array([1 - agree_score[i] for i in answered if answered[i]])
        print(f"  agreement-AUC on committed-wrongs: "
              f"{midrank_auc(s_w, s_c):.3f} (waist monitor 0.728) | "
              f"gate {'PASSED' if gate else 'FAILED — voting void, '
              'agreement columns informational only'}")

        # persist per-sample outcomes for composition analysis (zero-GPU cuts)
        tag = "O" if arm.startswith("O") else "D"
        vote_ans = np.full(n, -10**9, np.int64)
        for i in range(n):
            votes = [view_ans[k][i] for k in range(K_VIEWS + 1)
                     if view_ans[k][i] is not None]
            if votes:
                top, cnt = Counter(votes).most_common(1)[0]
                if cnt >= 3:
                    vote_ans[i] = top
        np.savez(f".cache/tta_arm_{tag}_bigtest.npz",
                 vote_ans=vote_ans,
                 agree=np.array([agree_score[i] for i in range(n)]),
                 view_forced=np.array([[view_ans[k][i] is not None
                                        for k in range(K_VIEWS + 1)]
                                       for i in range(n)]))
        print(f"  [saved] .cache/tta_arm_{tag}_bigtest.npz")


if __name__ == "__main__":
    main()
