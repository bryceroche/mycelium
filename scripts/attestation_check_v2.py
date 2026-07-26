"""attestation_check_v2.py — fence v2 measured against its pinned bars
(ledger 2026-07-25, pinned BLIND before this pass fires), with the
graph-BANKING design: every decoded view graph is banked so all future
fence iterations are zero-GPU replays.

v2 = v1 strip-and-reforce + E13 self-loop form law (seated verifier) +
multiplicity clause (formal register — this fixture IS the formal
register; the flag is carried by the caller per the dialect's scoping
rule, not inferred).

Bars (Bryce's word, overfit hazard named — v2 was assembled looking at
its one customer):
  (i)   ZERO false flags across all 1227 banked-correct rows
  (ii)  item 228 flags on ALL THREE winning views -> quorum unattested
        -> abstain-ungrounded -> gen-21 at zero lies on the fixture
  (iii) NO other quorum item changes disposition vs the v1 read; any
        additional flip sends scan and fence BOTH under audit

Also prints the full-verifier census (all codes incl. E13) over the
entire 7,500-view graph population — the honest re-read of the stale
0/3800 double-zero (whose substrate was never banked; population here
is the complete view set, stated as such).

Banks: .cache/attest_graphs_gen21.jsonl (one line per item: 5 views'
factors+answers) and .cache/attestation_v2_read_gen21.json (verdict).
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
from mycelium.attestation import attest_quorum, attest_quorum_v2
from mycelium.arith3_verifier import verify
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
print(f"[attest-v2] gate {CKPT} (HW={os.environ['ALG_HW']}, DUP={os.environ['ALG_DUP']})", flush=True)

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

rows = [json.loads(l) for l in open(".cache/algebra_nl_bigtest.jsonl")]
gold = [r["solution"][r["query_var"]] for r in rows]
banked = json.load(open(LATTICE))["bigtest"]
if LIMIT:
    rows = rows[:LIMIT]
    print(f"[attest-v2] SMOKE: first {LIMIT} items", flush=True)

vote_mismatch = 0
vcensus = Counter()          # verifier codes over ALL view graphs
vcensus_by_outcome = {}      # code counts split by view outcome
res = {"quorum_right": 0, "quorum_lie": 0,
       "false_flags_v2": [], "catches_v2": [], "missed_v2": [],
       "disposition_flips": [],  # items where v1 and v2 disagree
       "abstain": 0, "ckpt": CKPT, "lattice": LATTICE}
gbank = open(".cache/attest_graphs_gen21.jsonl", "w")
for i, r in enumerate(rows):
    texts = [r["text"]] + [permuted_view(r["text"], 40000 + 10 * i + k)
                           for k in range(1, 5)]
    parses = parse_batch(texts)
    m = r.get("m", 60)
    views = [(f, q, solve2(f, q, {"n_vars": 24, "m": m})) for f, q in parses]
    votes = [a for _, _, a in views]
    gbank.write(json.dumps({"item": i, "views": [
        {"factors": f, "query": int(q), "answer": a} for f, q, a in views]},
        default=int) + "\n")
    if votes != banked[i]:
        vote_mismatch += 1
        print(f"  [DRIFT] item {i}: reproduced {votes} != banked {banked[i]}", flush=True)
    # full-verifier census over every view graph, split by view outcome
    for f, q, a in views:
        outc = "abstain" if a is None else ("right" if a == gold[i] else "wrong")
        for code, _, _ in verify(f):
            vcensus[code] += 1
            vcensus_by_outcome.setdefault(outc, Counter())[code] += 1
    nn = [a for a in votes if a is not None]
    c = Counter(nn).most_common(1)
    plur, cnt = c[0] if c else (None, 0)
    if cnt < 3:
        res["abstain"] += 1
        continue
    a1, n_win, n_att1 = attest_quorum(views, plur, r["text"], m, solve2)
    a2, _, n_att2 = attest_quorum_v2(views, plur, r["text"], m, solve2)
    if a1 != a2:
        res["disposition_flips"].append(
            {"item": i, "answer": plur, "gold": gold[i],
             "v1_attested": a1, "v2_attested": a2,
             "n_win": n_win, "n_att_v1": n_att1, "n_att_v2": n_att2})
    if plur == gold[i]:
        res["quorum_right"] += 1
        if not a2:
            res["false_flags_v2"].append({"item": i, "answer": plur, "n_win": n_win})
            print(f"  [FALSE-FLAG-v2] item {i}: correct {plur} unattested", flush=True)
    else:
        res["quorum_lie"] += 1
        det = {"item": i, "answer": plur, "gold": gold[i],
               "n_win": n_win, "n_att_v2": n_att2}
        (res["catches_v2"] if not a2 else res["missed_v2"]).append(det)
        print(f"  [LIE] item {i}: {plur} vs gold {gold[i]} -> "
              f"{'FLAGGED abstain-ungrounded (v2)' if not a2 else 'MISSED by v2'}", flush=True)
    if (i + 1) % 300 == 0:
        print(f"  ...{i+1}/{len(rows)} | right {res['quorum_right']} lie {res['quorum_lie']} "
              f"ffv2 {len(res['false_flags_v2'])} flips {len(res['disposition_flips'])}", flush=True)
gbank.close()

assert vote_mismatch == 0, f"{vote_mismatch} vote drift(s) vs banked lattice — READ VOID"
print("\n=== FENCE v2 VERDICT (bars pinned blind, ledger 2026-07-25) ===")
print(f"  quorum-right rows: {res['quorum_right']} | FALSE FLAGS (v2): {len(res['false_flags_v2'])}  (bar i: ZERO)")
c228 = [d for d in res["catches_v2"] if d["item"] == 228]
print(f"  item 228: {'CAUGHT — ' + str(c228[0]) if c228 else 'NOT caught'}  (bar ii: all three winning views refused, n_att_v2==0)")
others = [d for d in res["disposition_flips"] if d["item"] != 228]
print(f"  disposition flips beyond 228: {len(others)}  (bar iii: ZERO)  {others if others else ''}")
print(f"  quorum lies: {res['quorum_lie']} | caught by v2: {len(res['catches_v2'])} | missed: {len(res['missed_v2'])}")
bar_i = len(res["false_flags_v2"]) == 0
bar_ii = bool(c228) and c228[0]["n_att_v2"] == 0
bar_iii = len(others) == 0
print(f"  BAR (i) {'PASS' if bar_i else 'FAIL'} | BAR (ii) {'PASS' if bar_ii else 'FAIL'} | BAR (iii) {'PASS' if bar_iii else 'FAIL'}")
print(f"  VERDICT: {'v2 SEATS — gen-21 stands at ZERO LIES on the fixture' if bar_i and bar_ii and bar_iii else 'v2 does NOT seat; audit per the pinned frame'}")
print("\n=== FORM-VERIFIER CENSUS over all view graphs (the stale double-zero's honest re-read; population = complete view set) ===")
print(f"  total view graphs: {5 * len(rows)}")
print(f"  codes: {dict(vcensus) if vcensus else 'NONE — form-clean across the entire view population'}")
for outc, cc in sorted(vcensus_by_outcome.items()):
    print(f"    by outcome [{outc}]: {dict(cc)}")
res["verifier_census"] = {k: int(v) for k, v in vcensus.items()}
res["verifier_census_by_outcome"] = {o: {k: int(v) for k, v in cc.items()}
                                     for o, cc in vcensus_by_outcome.items()}
json.dump(res, open(".cache/attestation_v2_read_gen21.json", "w"), indent=1)
print("[attest-v2] verdict -> .cache/attestation_v2_read_gen21.json | graphs -> .cache/attest_graphs_gen21.jsonl")
