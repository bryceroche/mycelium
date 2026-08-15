"""engineered_views.py — THE PHASE-ENGINEERED VIEW SET (scope pinned
2026-08-15 pre-fire): tests ENGINEERED DECORRELATION BEATS CONVENIENCE at
the vote grain. Does NOT test the six-wave frame's unique claims (pinned).

Stage A (recruit, means-vs-overlaps at the view grain): 8 candidate
graph-free permutation views (antiphase pairs by construction) measured
for per-item failure vectors on EVEN-index rows of the 233 + solved
controls; greedy-select 4 minimizing max pairwise agreement-on-failure.
Stage B (the bars, on untouched rows): (1) all-views-fail union on
ODD-index 233 rows: engineered < baseline (original + 4 convenience
shuffles); (2) row 183 (the four-door constant) reported per-view;
(3) vote accuracy on a fixed 300-row slice (seed 41): engineered NOT
below baseline. MC-PI same-wrong discipline reported per selected view.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import numpy as np

from phase1_algebra_head import (  # noqa: E402
    T_ALG, build_params, forward, load_alg, decode, sent_indices,
    ALG_CKPT, TOKENIZER_JSON,
)
from beacon_closing_arm import recompute_states  # noqa: E402
from repair_replace_swap import solve_forced  # noqa: E402


def _parts(text):
    return re.split(r"(?<=\.)\s+", text.strip())


def _join(first, mid, last):
    return " ".join([first] + mid + [last])


def _mk(fn):
    def view(text, seed):
        p = _parts(text)
        if len(p) <= 3:
            return text
        return _join(p[0], fn(p[1:-1], seed), p[-1])
    return view


CANDIDATES = [
    ("rev",      _mk(lambda m, s: m[::-1])),
    ("rot+",     _mk(lambda m, s: m[len(m) // 2:] + m[:len(m) // 2])),
    ("rot-",     _mk(lambda m, s: m[-(len(m) // 2):] + m[:-(len(m) // 2)]
                     if len(m) // 2 else m)),
    ("swap2",    _mk(lambda m, s: [m[i ^ 1] if (i ^ 1) < len(m) else m[i]
                                   for i in range(len(m))])),
    ("len_asc",  _mk(lambda m, s: sorted(m, key=len))),
    ("len_desc", _mk(lambda m, s: sorted(m, key=len, reverse=True))),
    ("shufA",    _mk(lambda m, s: random.Random(7000 + s).sample(m, len(m)))),
    ("shufB",    _mk(lambda m, s: random.Random(9000 + s).sample(m, len(m)))),
]
BASELINE = [("conv%d" % k,
             _mk(lambda m, s, _k=k:
                 random.Random(1000 * _k + s).sample(m, len(m))))
            for k in (1, 2, 3, 4)]


def main():
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(TOKENIZER_JSON)
    samples, states, tokmask, gold, sent = load_alg("test")
    n = len(samples)
    gold_ans = {i: samples[i]["solution"][samples[i]["query_var"]]
                for i in range(n)}

    p = build_params(0)
    sd = safe_load(ALG_CKPT)
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

    def parse_rows(sts, msk, snt, rows):
        ans = {}
        m = len(rows)
        for s0 in range(0, m, 8):
            sl = np.arange(s0, min(s0 + 8, m))
            pad = 8 - len(sl)
            slp = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
            out = forward(p, Tensor(sts[slp].astype(np.float32),
                                    dtype=dtypes.float),
                          Tensor(msk[slp].astype(np.float32),
                                 dtype=dtypes.float),
                          Tensor(snt[slp].astype(np.int32),
                                 dtype=dtypes.int))
            o = {k: out[k].realize().numpy() for k in
                 ("pres", "ftype", "op", "islit", "dig", "args", "res",
                  "query")}
            for bi, li in enumerate(sl):
                ri = rows[int(li)]
                facs, q = decode({k: o[k][bi] for k in o})
                ans[ri] = solve_forced(facs, q, samples[ri])
        return ans

    def view_answers(viewfn, rows):
        m = len(rows)
        ids = np.zeros((m, T_ALG), np.int32)
        msk = np.zeros((m, T_ALG), np.float32)
        snt = np.zeros((m, T_ALG), np.int32)
        for li, ri in enumerate(rows):
            t = viewfn(samples[ri]["text"], ri)
            e = tok.encode(t)
            if len(e.ids) > T_ALG:
                continue
            ids[li, :len(e.ids)] = e.ids
            msk[li, :len(e.ids)] = 1.0
            snt[li] = sent_indices(t, list(e.offsets), msk[li])
        sts = recompute_states(ids)
        return parse_rows(sts, msk, snt, rows)

    census = json.load(open(".cache/miss_census_gen41.json"))
    miss = sorted(int(i) for i in census["miss_idx"])
    even233 = [i for i in miss if i % 2 == 0]
    odd233 = [i for i in miss if i % 2 == 1]
    solved = [i for i in range(n) if i not in set(miss)]
    ctrl = sorted(random.Random(41).sample(solved, 100))
    vslice = sorted(random.Random(41).sample(range(n), 300))
    print(f"[ev] 233={len(miss)} even={len(even233)} odd={len(odd233)} "
          f"ctrl=100 vote-slice=300  (183 in odd: {183 in odd233})",
          flush=True)

    # original answers on all rows we touch (banked states — no recompute)
    all_rows = sorted(set(even233 + odd233 + ctrl + vslice))
    r2li = {r: i for i, r in enumerate(all_rows)}
    a0 = parse_rows(states[all_rows], tokmask[all_rows], sent[all_rows],
                    all_rows)

    # ---- Stage A: recruit on even-233 + controls ----
    sel_rows = even233 + ctrl
    fail_vecs, cand_ans = {}, {}
    for name, fn in CANDIDATES:
        a = view_answers(fn, sel_rows)
        cand_ans[name] = a
        fail_vecs[name] = np.array(
            [0 if a.get(r) == gold_ans[r] else 1 for r in sel_rows], np.int32)
        conv = sum(1 for r in even233 if a.get(r) == gold_ans[r])
        held = sum(1 for r in ctrl if a.get(r) == gold_ans[r])
        print(f"[cand {name:9s}] converts(even233) {conv}/{len(even233)}  "
              f"holds(ctrl) {held}/100", flush=True)

    names = [c[0] for c in CANDIDATES]

    def agree(x, y):  # agreement-on-failure among rows either fails
        both = ((fail_vecs[x] == 1) & (fail_vecs[y] == 1)).sum()
        either = ((fail_vecs[x] == 1) | (fail_vecs[y] == 1)).sum()
        return both / max(either, 1)

    pairs = [(agree(x, y), x, y) for xi, x in enumerate(names)
             for y in names[xi + 1:]]
    pairs.sort()
    chosen = [pairs[0][1], pairs[0][2]]
    while len(chosen) < 4:
        best = min((c for c in names if c not in chosen),
                   key=lambda c: max(agree(c, ch) for ch in chosen))
        chosen.append(best)
    print(f"[recruit] chosen {chosen}  "
          f"(most-antiphase pair {pairs[0][1]}/{pairs[0][2]} "
          f"agree {pairs[0][0]:.3f})", flush=True)

    # ---- Stage B: the bars on untouched rows ----
    eng = [(c, dict(CANDIDATES)[c]) for c in chosen]
    b_rows = sorted(set(odd233 + vslice))

    def read_set(tag, viewset):
        va = [{r: a0[r] for r in b_rows}]
        for name, fn in viewset:
            va.append(view_answers(fn, b_rows))
            print(f"  [{tag}] view {name} done", flush=True)
        union = [r for r in odd233
                 if not any(v.get(r) == gold_ans[r] for v in va)]
        vr = va_n = 0
        for r in vslice:
            votes = [v.get(r) for v in va if v.get(r) is not None]
            if votes:
                top, cnt = max(((x, votes.count(x)) for x in set(votes)),
                               key=lambda t: t[1])
                if cnt >= 3:
                    va_n += 1
                    vr += int(top == gold_ans[r])
        w0 = [r for r in odd233 if a0.get(r) is not None]
        sw = [np.mean([1 if v.get(r) == a0[r] else 0 for r in w0])
              for v in va[1:]]
        print(f"[{tag}] UNION(all-fail, odd233) {len(union)}/{len(odd233)}  "
              f"vote(300): right {vr} voted {va_n}  "
              f"same-wrong {['%.2f' % x for x in sw]}", flush=True)
        print(f"[{tag}] row 183: orig={a0.get(183)} " +
              " ".join(f"{name}={v.get(183)}"
                       for (name, _), v in zip(viewset, va[1:])) +
              f"  gold={gold_ans[183]}", flush=True)
        return union, vr

    ub, vb = read_set("baseline", BASELINE)
    ue, ve = read_set("engineered", eng)
    print(f"\n=== BARS === union {len(ue)} vs {len(ub)} "
          f"({'SHRINKS' if len(ue) < len(ub) else 'NO'})  |  "
          f"183 {'CONVERTS' if 183 not in ue else 'CONSTANT'} "
          f"(baseline: {'converts' if 183 not in ub else 'constant'})  |  "
          f"vote {ve} vs {vb} "
          f"({'HELD' if ve >= vb else 'DEGRADED'})", flush=True)
    json.dump({"chosen": chosen, "union_base": ub, "union_eng": ue,
               "vote_base": vb, "vote_eng": ve},
              open(".cache/engineered_views.json", "w"))


if __name__ == "__main__":
    main()
