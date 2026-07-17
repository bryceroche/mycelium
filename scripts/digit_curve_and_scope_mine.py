"""digit_curve_and_scope_mine.py — GUTS #29+#30 READS (b1)+(b2)
(2026-07-17, fired on the word).

(b1) THE EMISSION DIGIT CURVE — the chained-fdiv autopsy's FORMAL
OPENING READ (specimens 1+2 already filed: [26] wild-caught + its
mul-inverse cure). One forward pass over banked bigtest states;
per-slot digit-bank emissions vs gold (slot-aligned by construction —
the training layout), stratified by PATH (given value vs mod/fdiv/pct
parameter) x MAGNITUDE (1/2/3 significant digits) x POSITION
(MSD/mid/LSD). PINNED PREDICTION ([26]'s mechanism generalized): the
parameter path is weaker than the given path at matched magnitude,
with the deficit concentrated off the LSD (the specimen dropped the
tens digit of 27). Kill-only.

(b2) THE SCOPE-PAIR MINE — panama hats at the emission skin's sibling:
same-words-different-knots. Minimal pairs in near-wild phrasing
('the difference of the squares' vs 'the square of the difference'),
instantiated over value pairs, 5-view gate: per pair, outcomes classify
DISCRIMINATED (both parse to their own correct answers) / COLLAPSED
(both parse to the SAME answer — the compound ate the scope) /
REGISTER-WALL (no parse). Kill-only census; no bar — the mine counts.
"""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from collections import Counter, defaultdict
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "6")
os.environ.setdefault("ALG_DUP", "1")
from phase1_algebra_head import T_ALG, L_FAC, N_DIG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_views import permuted_view
from tta_alg2_dials import solve2
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

p = build_params(0)
CKPT = os.environ.get("GATE_CKPT", ".cache/phase1_gen14_head.safetensors")
sd = safe_load(CKPT)
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
print(f"[probe] gate = {CKPT}")

# ---------------- (b1) the emission digit curve ----------------
# B1_NPZ env: bigtest turned out GIVEN-ONLY (no param-path digit slots);
# the param curve lives in the alg2+/alg4 corpora (mod/fdiv/pct).
z = np.load(os.environ.get("B1_NPZ", ".cache/phase1_alg_states_bigtest.npz"))
states, tokmask, sent = z["states"], z["tokmask"], z["sent"]
g_pres, g_ft, g_dig = z["g_presence"], z["g_ftype"], z["g_digits"]
n = len(states)
FT_GIVEN, FT_MOD, FT_PCT, FT_FDIV = 1, 2, 4, 5   # per head layout (rel=0, sel=3)

acc = defaultdict(lambda: np.zeros(2))   # (path, mag, pos) -> [right, total]
slot_acc = defaultdict(lambda: np.zeros(2))
for s0 in range(0, n, 8):
    sl = np.arange(s0, min(s0 + 8, n))
    pad = 8 - len(sl)
    sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
    out = forward(p, Tensor(states[sl_p].astype(np.float32), dtype=dtypes.float),
                  Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float),
                  Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int))
    dig = out["dig"].realize().numpy()
    for bi, i in enumerate(sl):
        i = int(i)
        for j in range(L_FAC):
            if g_pres[i, j] <= 0:
                continue
            ft = int(g_ft[i, j])
            if ft == FT_GIVEN:
                path = "given"
            elif ft in (FT_MOD, FT_FDIV, FT_PCT):
                path = "param"
            else:
                continue                      # rel/sel slots carry no digits
            gold_d = g_dig[i, j]
            val = int(sum(d * 10 ** (N_DIG - 1 - k_) for k_, d in enumerate(gold_d)))
            mag = len(str(max(val, 0))) if val > 0 else 1
            pred = dig[bi, j].argmax(-1)
            for pos, name in enumerate(("MSD", "mid", "LSD")):
                acc[(path, mag, name)] += (int(pred[pos] == gold_d[pos]), 1)
            slot_acc[(path, mag)] += (int((pred == gold_d).all()), 1)

print("\n=== (b1) EMISSION DIGIT CURVE (bigtest, slot-aligned vs gold) ===")
print(f"  {'path':6s} {'mag':>3s} {'n':>6s}  {'all-3':>7s}   MSD    mid    LSD")
for path in ("given", "param"):
    for mag in (1, 2, 3):
        t = slot_acc[(path, mag)]
        if t[1] == 0:
            continue
        row = [acc[(path, mag, pos)] for pos in ("MSD", "mid", "LSD")]
        print(f"  {path:6s} {mag:3d} {int(t[1]):6d}  {t[0]/t[1]:7.3f}  " +
              "  ".join(f"{r[0]/r[1]:.3f}" for r in row))
deficits = {}
for mag in (2, 3):
    tg, tp = slot_acc[("given", mag)], slot_acc[("param", mag)]
    if tg[1] and tp[1]:
        deficits[mag] = tg[0]/tg[1] - tp[0]/tp[1]
        print(f"  magnitude-{mag} given-vs-param deficit: {deficits[mag]:+.3f}")
pred_holds = all(d > 0 for d in deficits.values()) if deficits else None
print(f"  PINNED PREDICTION (param weaker at matched magnitude): "
      f"{'HOLDS' if pred_holds else 'FAILS' if pred_holds is not None else 'N/A'}")

# ---------------- (b2) the scope-pair mine ----------------
tok = Tokenizer.from_file(TOKENIZER_JSON)


def parse5(text, m, seed0):
    texts = [text] + [permuted_view(text, seed0 + k) for k in range(1, 5)]
    N = 8
    ids = np.zeros((N, T_ALG), np.int32); msk = np.zeros((N, T_ALG), np.float32)
    snt = np.zeros((N, T_ALG), np.int32)
    for i, t in enumerate(texts):
        e = tok.encode(t)
        Ln = min(len(e.ids), T_ALG)
        ids[i, :Ln] = e.ids[:Ln]; msk[i, :Ln] = 1.0
        snt[i] = sent_indices(t, list(e.offsets), msk[i])
    st = recompute_states(ids)
    out = forward(p, Tensor(st.astype(np.float32), dtype=dtypes.float),
                  Tensor(msk.astype(np.float32), dtype=dtypes.float),
                  Tensor(snt.astype(np.int32), dtype=dtypes.int))
    keys = [k for k in ("pres", "ftype", "op", "islit", "dig", "args", "res",
                        "query", "sel", "dup") if k in out]
    o = {k: out[k].realize().numpy() for k in keys}
    votes = []
    for bi in range(5):
        facs, q = decode({k: o[k][bi] for k in o})
        a = solve2(facs, q, {"n_vars": 24, "m": m})
        if a is not None:
            votes.append(a)
    t_, c = (Counter(votes).most_common(1)[0] if votes else (None, 0))
    return t_ if c >= 3 else None


VALS = [(7, 4), (9, 5), (8, 3), (11, 6), (12, 7), (13, 4), (10, 3), (15, 8),
        (14, 5), (9, 2)]
D = "Consider the numbers "
disc = coll = wall = 0
print(f"\n=== (b2) SCOPE-PAIR MINE ({len(VALS)} pairs, near-wild phrasing) ===")
for a_, b_ in VALS:
    dsq = a_ * a_ - b_ * b_          # difference of squares
    sqd = (a_ - b_) ** 2             # square of the difference
    t1 = (D + f"a, b, c. a is {a_}. b is {b_}. The difference of the squares "
          f"of a and b equals c. What is c?")
    t2 = (D + f"a, b, c. a is {a_}. b is {b_}. The square of the difference "
          f"of a and b equals c. What is c?")
    v1 = parse5(t1, 300, 160000 + a_ * 100 + b_)
    v2 = parse5(t2, 300, 170000 + a_ * 100 + b_)
    if v1 == dsq and v2 == sqd:
        disc += 1; tag = "DISCRIMINATED"
    elif v1 is not None and v1 == v2:
        coll += 1; tag = f"COLLAPSED (both -> {v1})"
    elif v1 is None and v2 is None:
        wall += 1; tag = "REGISTER-WALL"
    else:
        tag = f"mixed (v1={v1} v2={v2}; gold {dsq}/{sqd})"
    print(f"  ({a_:2d},{b_:2d}) gold {dsq:3d}/{sqd:3d}: {tag}")
print(f"\n  census: discriminated {disc} | collapsed {coll} | "
      f"register-wall {wall} | mixed {len(VALS)-disc-coll-wall} of {len(VALS)}")

json.dump({"digit_curve": {f"{p_}|{m_}": {"n": int(slot_acc[(p_, m_)][1]),
                                          "acc": float(slot_acc[(p_, m_)][0] /
                                                       max(slot_acc[(p_, m_)][1], 1))}
                           for p_ in ("given", "param") for m_ in (1, 2, 3)},
           "deficits": {str(k): float(v) for k, v in deficits.items()},
           "prediction_holds": bool(pred_holds) if pred_holds is not None else None,
           "scope": {"discriminated": disc, "collapsed": coll,
                     "register_wall": wall, "n": len(VALS)}},
          open(".cache/digit_curve_scope_mine.json", "w"))
print("[probe] banked -> .cache/digit_curve_scope_mine.json")
