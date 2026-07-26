"""roundtrip_read.py — THE ROUND-TRIP LOOP, budget 1 (2026-07-26; rulings
at 0f652ec). For each g21 deployed abstention: strip each view's graph to
its LAWFUL CORE (fence: only core-derived facts may inject), extract
forced facts (uniqueness-tested), inject as dialect text ('It is known
that L is v.'), re-parse 5 views, re-solve, re-vote, attest. Exits:
attested-answer / same-graph / new-flags. BARS: zero new lies; lift over
the 0.688 blind floor; budget held at one."""
import sys, os, json, re
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_HW", "512"); os.environ.setdefault("ALG_DUP", "1")
from collections import Counter
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_views import permuted_view
from tta_alg2_dials import solve2
from mycelium.attestation import lawful_core, attest_quorum_v3
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

rows = [json.loads(l) for l in open(".cache/algebra_nl_bigtest.jsonl")]
gold = [r["solution"][r["query_var"]] for r in rows]
lat = json.load(open(".cache/lattice_gen21_H.json"))["bigtest"]
bank = {}
for l in open(".cache/attest_graphs_gen21.jsonl"):
    d = json.loads(l); bank[d["item"]] = d["views"]

def letters_of(text):
    m = re.match(r'^(Consider the numbers|Let|The following facts hold about)\s+(.+?)(?:\s+be whole numbers)?\.', text)
    toks = re.split(r',\s*|\s+and\s+', m.group(2)) if m else []
    return [t.strip() for t in toks if re.fullmatch(r'[a-z]', t.strip())]

def forced_facts(facs, text, m, letters, cap=3):
    """Facts forced by the LAWFUL CORE (fence-compliant by construction)."""
    core = lawful_core(facs, text)
    from mycelium.csp_domains import problem_from_algebra3
    from mycelium.csp_core import solve_symbolic
    gv = {f["var"]: f["value"] for f in core if f["ftype"] == "given"}
    try:
        nv = max([24] + [v+1 for f in core for v in
                 (list(f.get("args", [])) + [f[k] for k in ("result","var") if k in f])
                 if isinstance(v, int)])
        prob = problem_from_algebra3(nv, core, gv, m)
        res = solve_symbolic(prob, budget=100_000, seed=0)
        if res["status"] != "solved": return []
        out = []
        for v in range(min(nv, len(letters))):
            if v in gv: continue
            val = int(res["assignment"][v])
            p2 = problem_from_algebra3(nv, core, gv, m)
            p2.domains0[v].discard(val)
            if p2.domains0[v] and solve_symbolic(p2, budget=50_000, seed=0)["status"] == "solved":
                continue
            out.append((letters[v], val))
            if len(out) >= cap: break
        return out
    except Exception:
        return []

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
sd = safe_load(".cache/g21.safetensors")
assert set(sd.keys()) == set(p.keys())
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

def parse_batch(texts):
    n = len(texts); N = ((n+7)//8)*8
    ids = np.zeros((N, T_ALG), np.int32); msk = np.zeros((N, T_ALG), np.float32)
    snt = np.zeros((N, T_ALG), np.int32)
    for i, t in enumerate(texts):
        e = tok.encode(t); L = min(len(e.ids), T_ALG)
        ids[i, :L] = e.ids[:L]; msk[i, :L] = 1.0
        snt[i] = sent_indices(t, list(e.offsets), msk[i])
    st = recompute_states(ids)
    out_r = []
    for s0 in range(0, N, 8):
        out = forward(p, Tensor(st[s0:s0+8].astype(np.float32), dtype=dtypes.float),
                      Tensor(msk[s0:s0+8].astype(np.float32), dtype=dtypes.float),
                      Tensor(snt[s0:s0+8].astype(np.int32), dtype=dtypes.int))
        keys = ("pres","ftype","op","islit","dig","args","res","query") + \
            (("sel",) if "sel" in out else ()) + (("dup",) if "dup" in out else ())
        o = {k: out[k].realize().numpy() for k in keys}
        for bi in range(8):
            if s0+bi < n: out_r.append(decode({k: o[k][bi] for k in o}))
    return out_r

LIMIT = int(os.environ.get("LIMIT", "0"))
abst = []
for i, v in enumerate(lat):
    nn = [a for a in v if a is not None]
    c = Counter(nn).most_common(1); plur, cnt = c[0] if c else (None, 0)
    if cnt < 3: abst.append(i)
if LIMIT: abst = abst[:LIMIT]
print(f"[roundtrip] {len(abst)} abstentions", flush=True)

res = {"recovered": 0, "new_lie": [], "still_abstain": 0, "no_facts": 0, "n": len(abst)}
for idx, i in enumerate(abst):
    r = rows[i]; m = r.get("m", 60); letters = letters_of(r["text"])
    facts = {}
    for v in bank[i]:
        for L, val in forced_facts(v["factors"], r["text"], m, letters):
            facts[L] = val
    if not facts:
        res["no_facts"] += 1; res["still_abstain"] += 1; continue
    ann = r["text"] + " " + " ".join(f"It is known that {L} is {val}." for L, val in sorted(facts.items()))
    texts = [ann] + [permuted_view(ann, 40000 + 10*i + k) for k in range(1, 5)]
    views = [(f, q, solve2(f, q, {"n_vars": 24, "m": m})) for f, q in parse_batch(texts)]
    votes = [a for _, _, a in views]
    nn = [a for a in votes if a is not None]
    c = Counter(nn).most_common(1); plur, cnt = c[0] if c else (None, 0)
    if cnt >= 3:
        att, _, _ = attest_quorum_v3(views, plur, ann, m, solve2)
        if att:
            if plur == gold[i]: res["recovered"] += 1
            else: res["new_lie"].append({"item": i, "answer": plur, "gold": gold[i]})
        else: res["still_abstain"] += 1
    else: res["still_abstain"] += 1
    if (idx+1) % 25 == 0:
        print(f"  {idx+1}/{len(abst)} | rec {res['recovered']} lie {len(res['new_lie'])} abst {res['still_abstain']}", flush=True)

print(f"\n=== ROUND-TRIP VERDICT (budget 1) ===", flush=True)
rr = res["recovered"] / max(1, res["n"])
print(f"  recovered {res['recovered']}/{res['n']} = {rr:.3f} | blind floor ceiling 0.688 | NEW LIES: {len(res['new_lie'])} (bar: ZERO) {res['new_lie'][:3]}")
print(f"  no-facts (diagnosis empty): {res['no_facts']} | still-abstain: {res['still_abstain']}")
print(f"  BAR (i) zero-new-lies: {'PASS' if not res['new_lie'] else 'FAIL'}")
json.dump(res, open(".cache/roundtrip_read.json", "w"), indent=1)
