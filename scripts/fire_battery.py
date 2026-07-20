"""fire_battery.py — THE FIRE'S BATTERY (2026-07-20). Reads, per the
charter's pinned bars, for all four arms:
  (1) bigtest ANSWER (no-regression core; gen-14 floor 1149)
  (2) MACRO ACCEPTANCE: the 9 wild crowns' macro dialects, 5-view vote
      (decode emits ftype-6 macros; solve2 expands at its doorstep)
  (3) THE DIVIDEND READ, honest shapes: [100] and [73] re-annotated at
      macro floor. The crown absorbs mul-add plumbing (sheds 2-3 vars);
      it does NOT absorb fdivs — so the read tests whether crown
      compression alone crosses back under the fdiv-mass wall. Either
      sentence pre-written: dividend or honest miss.
Verdict table at the end; gen15_verdict (the manifest writer) runs only
after this read is graded by the pre-pinned frames.
"""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from collections import Counter
os.environ.setdefault("ALG2", "1")
os.environ["ALG_FTYPES"] = "7"
os.environ.setdefault("ALG_DUP", "1")
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_views import permuted_view
from tta_alg2_dials import solve2
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

ARMS = {"A": ".cache/fire_armA.safetensors",
        "B": ".cache/fire_armB.safetensors",
        "C1": ".cache/fire_armC1.safetensors",
        "C2": ".cache/fire_armC2.safetensors"}
tok = Tokenizer.from_file(TOKENIZER_JSON)

# the 9 wild crowns' macro dialects (from the banked book-4 rows)
b4 = [json.loads(l) for l in open(".cache/book4_prose_pairs.jsonl")]
wild = [(r["gen"]["dialect"], r["factors"], r["query_var"],
         solve2(r["factors"], r["query_var"], {"n_vars": 24, "m": 300}))
        for r in b4 if r["gen"]["floor"] == "macro"]
assert len(wild) == 9 and all(a is not None for *_, a in wild)

# THE DIVIDEND SHAPES (macro-floor re-annotations of the wall's markers;
# fdivs remain — the crown sheds the mul-add vars only; honest test)
D = "Consider the numbers "
DIVIDEND = [
 ("[73]", 19, D + "a, b, c, d, e, f, g, h. a is 56. When a is divided by 7, "
  "the quotient is b and the remainder is c. When a is divided by 4, the "
  "quotient is d and the remainder is e. 3 times b plus d equals f. When f "
  "is divided by 2, the quotient is g and the remainder is h. What is g?"),
 ("[100]", 11, D + "a, b, c, d, e, f, g, h, i, j. a is 3. When a is divided "
  "by 2, the quotient is b and the remainder is c. d is 9. When d is divided "
  "by 4, the quotient is e and the remainder is f. g is 81. When g is divided "
  "by 16, the quotient is h and the remainder is i. b plus e equals j. "
  "j plus h plus 3 equals k. What is k?"),
]
# [100] note: '+3' folds the three ceil +1 adjustments; k via crown-ish sum —
# rendered as prime chain below if the crown phrasing misparses, the miss is
# honest: the fdiv-mass wall standing through crown compression.

results = {}
for arm, ckpt in ARMS.items():
    p = build_params(0)
    sd = safe_load(ckpt)
    assert set(sd.keys()) == set(p.keys()), f"key mismatch {ckpt}"
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

    def parse_batch(texts):
        n = len(texts)
        N = ((n + 7) // 8) * 8
        ids = np.zeros((N, T_ALG), np.int32)
        msk = np.zeros((N, T_ALG), np.float32)
        snt = np.zeros((N, T_ALG), np.int32)
        for i, t in enumerate(texts):
            e = tok.encode(t)
            Ln = min(len(e.ids), T_ALG)
            ids[i, :Ln] = e.ids[:Ln]; msk[i, :Ln] = 1.0
            snt[i] = sent_indices(t, list(e.offsets), msk[i])
        st = recompute_states(ids)
        res = []
        for s0 in range(0, N, 8):
            out = forward(p, Tensor(st[s0:s0+8].astype(np.float32), dtype=dtypes.float),
                          Tensor(msk[s0:s0+8].astype(np.float32), dtype=dtypes.float),
                          Tensor(snt[s0:s0+8].astype(np.int32), dtype=dtypes.int))
            keys = [k for k in ("pres", "ftype", "op", "islit", "dig", "dig2",
                                "args", "res", "query", "sel", "dup", "y")
                    if k in out]
            o = {k: out[k].realize().numpy() for k in keys}
            for bi in range(8):
                if s0 + bi < n:
                    res.append(decode({k: o[k][bi] for k in o}))
        return res

    def vote5(text, m, answer, seed0):
        texts = [text] + [permuted_view(text, seed0 + k) for k in range(1, 5)]
        votes = []
        for facs, q in parse_batch(texts):
            a = solve2(facs, q, {"n_vars": 24, "m": m})
            if a is not None:
                votes.append(a)
        top, cnt = (Counter(votes).most_common(1)[0] if votes else (None, 0))
        return (cnt >= 3 and top == answer), votes

    # (1) bigtest ANSWER (banked states — fast path)
    z = np.load(".cache/phase1_alg_states_bigtest.npz")
    rows = [json.loads(l) for l in open(".cache/algebra_nl_bigtest.jsonl")]
    st, tk, se = z["states"], z["tokmask"], z["sent"]
    n_ans = 0
    for s0 in range(0, len(rows), 8):
        sl = np.arange(s0, min(s0 + 8, len(rows)))
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        out = forward(p, Tensor(st[sl_p].astype(np.float32), dtype=dtypes.float),
                      Tensor(tk[sl_p].astype(np.float32), dtype=dtypes.float),
                      Tensor(se[sl_p].astype(np.int32), dtype=dtypes.int))
        keys = [k for k in ("pres", "ftype", "op", "islit", "dig", "dig2",
                            "args", "res", "query", "sel", "dup", "y") if k in out]
        o = {k: out[k].realize().numpy() for k in keys}
        for bi, i in enumerate(sl):
            i = int(i)
            facs, q = decode({k: o[k][bi] for k in o})
            a = solve2(facs, q, {"n_vars": 24, "m": rows[i].get("m", 60)})
            n_ans += (a == rows[i]["solution"][rows[i]["query_var"]])
    # (2) macro acceptance
    acc = sum(vote5(d_, 300, a_, 300000 + 10 * wi)[0]
              for wi, (d_, _, _, a_) in enumerate(wild))
    # (3) dividend
    div = {}
    for name, gold_a, dia in DIVIDEND:
        ok, votes = vote5(dia, 300, gold_a, 310000)
        div[name] = (ok, votes)
    results[arm] = {"bigtest": n_ans, "macro_acc": acc, "dividend": div}
    print(f"[battery] arm {arm}: bigtest {n_ans}/1500 | wild crowns {acc}/9 | "
          f"dividend { {k: v[0] for k, v in div.items()} } "
          f"votes { {k: v[1][:5] for k, v in div.items()} }", flush=True)

print("\n=== THE FIRE'S TABLE (floor: bigtest >=1149; charter frames rule) ===")
for arm, r in results.items():
    d = r["dividend"]
    print(f"  {arm:3s} bigtest {r['bigtest']:4d}  crowns {r['macro_acc']}/9  "
          f"[73] {'DIVIDEND' if d['[73]'][0] else 'miss'}  "
          f"[100] {'DIVIDEND' if d['[100]'][0] else 'miss'}")
json.dump({a: {"bigtest": r["bigtest"], "macro_acc": r["macro_acc"],
               "dividend": {k: {"ok": v[0], "votes": v[1]}
                            for k, v in r["dividend"].items()}}
           for a, r in results.items()},
          open(".cache/fire_battery.json", "w"))
print("[battery] banked -> .cache/fire_battery.json")
