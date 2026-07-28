"""attestation_check.py — measure the attestation fence against its pinned
bars (ledger 2026-07-25, pinned BLIND before this script existed).

Decodes bigtest 5-view under the gate (same seeds as the lattice member),
HARD-ASSERTS the reproduced votes match the banked lattice byte-for-byte
(any drift voids the read — no silent fallback), then runs the fence on
every quorum item.

Prints the two bar numbers:
  false flags on quorum-right rows   (bar: ZERO)
  catches on quorum-lie rows         (expected: exactly 1 — item 228)

Env: CKPT (default manifest parser_ckpt), LATTICE (default gen21_H),
     LIMIT (smoke: first N items only, skips the full-lattice assert).
"""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_HW", "512"); os.environ.setdefault("ALG_DUP", "1")
from collections import Counter
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_views import permuted_view
from tta_alg2_dials import solve2
from mycelium.attestation import attest_quorum
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

MANIFEST = json.load(open(".cache/GENERATION.json"))
CKPT = os.environ.get("CKPT", MANIFEST["parser_ckpt"])
LATTICE = os.environ.get("LATTICE", ".cache/lattice_gen21_H.json")
LIMIT = int(os.environ.get("LIMIT", "0"))

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
sd = safe_load(CKPT)
assert set(sd.keys()) == set(p.keys()), f"key mismatch loading {CKPT}"
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
print(f"[attest] gate {CKPT} (HW={os.environ['ALG_HW']}, DUP={os.environ['ALG_DUP']})", flush=True)

def parse_batch(texts):
    n = len(texts)
    N = ((n + 7) // 8) * 8
    ids = np.zeros((N, T_ALG), np.int32); msk = np.zeros((N, T_ALG), np.float32)
    snt = np.zeros((N, T_ALG), np.int32)
    for i, t in enumerate(texts):
        e = tok.encode(t)
        L = min(len(e.ids), T_ALG)
        ids[i, :L] = e.ids[:L]; msk[i, :L] = 1.0
        snt[i] = sent_indices(t, list(e.offsets), msk[i])
    st = recompute_states(ids)
    res = []
    for s0 in range(0, N, 8):
        out = forward(p, Tensor(st[s0:s0+8].astype(np.float32), dtype=dtypes.float),
                      Tensor(msk[s0:s0+8].astype(np.float32), dtype=dtypes.float),
                      Tensor(snt[s0:s0+8].astype(np.int32), dtype=dtypes.int))
        keys = ("pres","ftype","op","islit","dig","args","res","query") + \
            (("sel",) if "sel" in out else ()) + (("dup",) if "dup" in out else ())
        o = {k: out[k].realize().numpy() for k in keys}
        for bi in range(8):
            if s0 + bi < n:
                res.append(decode({k: o[k][bi] for k in o}))
    return res

from mycelium.custody_gold import row_gold   # solution is pen-side scratch, never custody-side gold (law 2026-07-28)
rows = [json.loads(l) for l in open(".cache/algebra_nl_bigtest.jsonl")]
gold = [row_gold(r) for r in rows]
banked = json.load(open(LATTICE))["bigtest"]
if LIMIT:
    rows = rows[:LIMIT]
    print(f"[attest] SMOKE: first {LIMIT} items", flush=True)

vote_mismatch = 0
res = {"quorum_right": 0, "quorum_lie": 0, "false_flags": [], "catches": [],
       "abstain": 0, "ckpt": CKPT, "lattice": LATTICE}
for i, r in enumerate(rows):
    texts = [r["text"]] + [permuted_view(r["text"], 40000 + 10 * i + k)
                           for k in range(1, 5)]
    parses = parse_batch(texts)
    m = r.get("m", 60)
    views = [(f, q, solve2(f, q, {"n_vars": 24, "m": m})) for f, q in parses]
    votes = [a for _, _, a in views]
    if votes != banked[i]:
        vote_mismatch += 1
        print(f"  [DRIFT] item {i}: reproduced {votes} != banked {banked[i]}", flush=True)
    nn = [a for a in votes if a is not None]
    c = Counter(nn).most_common(1)
    plur, cnt = c[0] if c else (None, 0)
    if cnt < 3:
        res["abstain"] += 1
        continue
    attested, n_win, n_att = attest_quorum(views, plur, r["text"], m, solve2)
    if plur == gold[i]:
        res["quorum_right"] += 1
        if not attested:
            res["false_flags"].append({"item": i, "answer": plur, "n_win": n_win})
            print(f"  [FALSE-FLAG] item {i}: correct answer {plur} unattested ({n_att}/{n_win} views)", flush=True)
    else:
        res["quorum_lie"] += 1
        det = {"item": i, "answer": plur, "gold": gold[i],
               "flagged": not attested, "n_win": n_win, "n_att": n_att}
        if not attested:
            res["catches"].append(det)
        else:
            res.setdefault("missed", []).append(det)
        print(f"  [LIE] item {i}: {plur} vs gold {gold[i]} -> {'FLAGGED abstain-ungrounded' if not attested else 'NOT flagged (attested?!)'}", flush=True)
    if (i + 1) % 300 == 0:
        print(f"  ...{i+1}/{len(rows)} | right {res['quorum_right']} lie {res['quorum_lie']} ff {len(res['false_flags'])}", flush=True)

assert vote_mismatch == 0, f"{vote_mismatch} vote drift(s) vs banked lattice — READ VOID"
print("\n=== ATTESTATION VERDICT (bars pinned blind, ledger 2026-07-25) ===")
print(f"  quorum-right rows: {res['quorum_right']} | FALSE FLAGS: {len(res['false_flags'])}  (bar: ZERO)")
print(f"  quorum-lie rows:   {res['quorum_lie']} | CATCHES: {len(res['catches'])}  (expected: exactly 1, item 228)")
if res.get("missed"):
    print(f"  MISSED LIES (attested but wrong): {res['missed']}")
bar_i = len(res["false_flags"]) == 0
print(f"  BAR (i) zero-false-flags: {'PASS' if bar_i else 'FAIL'}")
print(f"  SEAT VERDICT: {'the fence SEATS in the battery' if bar_i else 'the fence does NOT seat; false flags are provenance-gap bugs — audit them'}")
json.dump(res, open(".cache/attestation_read_gen21.json", "w"), indent=1)
print("[attest] banked -> .cache/attestation_read_gen21.json")
