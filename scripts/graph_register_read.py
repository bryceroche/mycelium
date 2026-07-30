"""graph_register_read.py — THE GRAPH-REGISTER FIRST READ (2026-07-30,
gut #103's word: retroactive-first, digest-and-compare, no build).

Question: would a graph-shape mouth (winning parse's digest vs the
certified family's digest bank) have caught the wild lies — idx 34
especially, the specimen that defeats both existing planes (honest
values, foreign shape)?

PINNED BEFORE LOOKING (the two riders):
  (1) BOTH numbers print: lie-outside-family AND leave-one-out
      false-foreign on the 266 certified rows. A register that rejects
      its own family is a mouth, not a fence.
  (2) THREE GRAINS swept; the GRAIN IS PICKED BY THE LOO RATE, never
      the lie-catch (fitting the fence to its one specimen is the
      named failure). Fence-grade: LOO false-foreign <= 5%.
      Mouth-grade (graded signal only): 5-30%. Refuted at that
      grain: > 30%.
GRAINS:
  G1 native   — hash_audit_iso.canon verbatim (WL, VALUES INCLUDED —
                the dedup grain; predicted far too strict for a shape
                register, swept for honesty not hope)
  G2 shape    — same WL canon, values abstracted (kinds/ops/sel kept,
                given-values, k, p stripped): pure topology + typing
  G3 profile  — coarse: multiset of (ftype, op|sel) + n_givens +
                n_factors (no wiring at all)
Family = the 266 certified book-8 graphs (annotated factors, the banked
artifact). Wild side = the forty's winning STRESS parses under g21 (the
gate that told the lies) — GPU, so it WAITS for battery-g22 to release
the device; the CPU family read prints first."""
import sys, os, json, glob, hashlib, subprocess, time
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from collections import Counter, defaultdict
from hash_audit_iso import canon, level0, fdesc

def fdesc_shape(f):
    """G2: fdesc with values abstracted — topology + typing only."""
    ft = f["ftype"]
    if ft == "rel":
        return (("rel", f["op"]), (("a", f["args"][0]), ("a", f["args"][1]), ("r", f["result"])))
    if ft == "given":
        return (("giv",), (("v", f["var"]),))
    if ft == "mod":
        return (("mod",), (("s", f["var"]), ("r", f["result"])))
    if ft == "fdiv":
        return (("fdv",), (("s", f["var"]), ("r", f["result"])))
    if ft == "pct":
        return (("pct",), (("p", f["args"][0]), ("b", f["args"][1])))
    if ft == "sel":
        return (("sel", f["sel"]), (("a", f["args"][0]), ("a", f["args"][1]), ("r", f["result"])))
    raise ValueError(ft)

def wl_digest(row, desc_fn):
    facs_l0, nv = level0(row)
    facs = [desc_fn(f) for f in facs_l0]
    q = row["query_var"]
    col = {v: ("Q" if v == q else ".") for v in range(nv)}
    for _ in range(6):
        fcols = []
        for kind, mem in facs:
            aa = tuple(sorted(col[m] for r, m in mem if r == "a"))
            rest = tuple((r, col[m]) for r, m in mem if r != "a")
            fcols.append((kind, aa, rest))
        inc = defaultdict(list)
        for (kind, mem), fc in zip(facs, fcols):
            for r, m in mem:
                inc[m].append((fc, r))
        col = {v: (col[v], tuple(sorted(map(repr, inc[v])))) for v in range(nv)}
        ranks = {c: i for i, c in enumerate(sorted(set(map(repr, col.values()))))}
        col = {v: ranks[repr(c)] for v, c in col.items()}
    sig = sorted(repr((kind,
                       tuple(sorted(col[m] for r, m in mem if r == "a")),
                       tuple((r, col[m]) for r, m in mem if r != "a")))
                 for kind, mem in facs)
    return hashlib.sha256(("|".join(sig) + f"#q{col[q]}").encode()).hexdigest()

def g3_profile(row):
    facs_l0, _ = level0(row)
    kinds = sorted((f["ftype"], f.get("op", f.get("sel", ""))) for f in facs_l0)
    n_giv = sum(1 for f in facs_l0 if f["ftype"] == "given")
    return hashlib.sha256(repr((tuple(kinds), n_giv, len(facs_l0))).encode()).hexdigest()

GRAINS = {"G1-native": lambda r: canon(r)[0],
          "G2-shape": lambda r: wl_digest(r, fdesc_shape),
          "G3-profile": g3_profile}

# ---- family: the 266 certified graphs (CPU, prints first) ----
fam_rows = []
for draft, certf in [(d, d.replace("_prose_pairs_draft.jsonl", "_certification.json")
                      .replace("book8_prose", "book8_certification").replace("_certification.json", "_certification.json"))
                     for d in sorted(glob.glob(".cache/book8_*prose_pairs_draft.jsonl"))]:
    certf = draft.replace("prose_pairs_draft.jsonl", "certification.json")
    if not os.path.exists(certf):
        certf = ".cache/book8_certification.json"
    rows = [json.loads(l) for l in open(draft)]
    cert = json.load(open(certf))["certified"]
    fam_rows += [rows[e["i"]] for e in cert]
print(f"[family] {len(fam_rows)} certified graphs")
results = {}
for gname, fn in GRAINS.items():
    digs = [fn(r) for r in fam_rows]
    cnt = Counter(digs)
    loo_foreign = sum(1 for d in digs if cnt[d] == 1)
    rate = loo_foreign / len(digs)
    grade = "FENCE-grade" if rate <= 0.05 else ("MOUTH-grade" if rate <= 0.30 else "REFUTED at this grain")
    results[gname] = {"family_digests": digs, "uniques": len(cnt),
                      "loo_false_foreign": loo_foreign, "loo_rate": rate, "grade": grade}
    print(f"[{gname}] family uniques {len(cnt)}/266 | LOO false-foreign "
          f"{loo_foreign}/266 = {rate:.1%} -> {grade}")

# ---- wild side: the forty's winning STRESS parses under g21 (GPU; waits) ----
print("[wild] waiting for battery-g22 to release the GPU...", flush=True)
while subprocess.run(["systemctl", "--user", "is-active", "battery-g22.service"],
                     capture_output=True, text=True).stdout.strip() == "active":
    time.sleep(30)
print("[wild] GPU free — parsing the forty under g21 (the gate that told the lies)", flush=True)
os.environ.update(ALG2="1", ALG_FTYPES="8", ALG_HW="512", ALG_DUP="1")
import re
import numpy as np
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_views import permuted_view
from tta_alg2_dials import solve2
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
sd = safe_load(".cache/g21.safetensors")
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
def parse_batch(texts):
    n = len(texts); N = ((n+7)//8)*8
    ids = np.zeros((N, T_ALG), np.int32); msk = np.zeros((N, T_ALG), np.float32); snt = np.zeros((N, T_ALG), np.int32)
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
        keys = ("pres","ftype","op","islit","dig","args","res","query") + (("sel",) if "sel" in out else ()) + (("dup",) if "dup" in out else ())
        o = {k: out[k].realize().numpy() for k in keys}
        for bi in range(8):
            if s0+bi < n: out_r.append(decode({k: o[k][bi] for k in o}))
    return out_r
def int_answer(a):
    s = str(a).strip().replace("$", "").replace(",", "")
    return int(s) if re.fullmatch(r"-?\d+", s) else None
h = [json.loads(l) for l in open(".cache/math_harvest_v0.jsonl")]
cands = [x for x in h if x["level"] == "Level 4" and len(x["problem"]) < 300
         and "asy]" not in x["problem"]
         and all(int(n) <= 300 for n in re.findall(r"\d+", x["problem"]))
         and int_answer(x["answer"]) is not None
         and 0 <= int_answer(x["answer"]) <= 300][:40]
LIE_IDXS = {1, 8, 22, 34}
wild = {}
for j, x in enumerate(cands):
    text = x["problem"]; gold = int_answer(x["answer"])
    vt = [text] + [permuted_view(text, 88000 + 10*j + k) for k in range(1, 5)]
    views = [(f, q, solve2(f, q, {"n_vars": 24, "m": 300})) for f, q in parse_batch(vt)]
    nn = [a for _, _, a in views if a is not None]
    c = Counter(nn).most_common(1); plur, cnt = c[0] if c else (None, 0)
    if cnt < 3: continue
    f0, q0, _ = next((v for v in views if v[2] == plur))
    row = {"factors": f0, "query_var": q0, "n_vars": 24}
    wild[j] = {"is_lie": j in LIE_IDXS and plur != gold, "plur": plur, "gold": gold,
               "digests": {g: fn(row) for g, fn in GRAINS.items()}}
print(f"[wild] {len(wild)} quorum graphs digested (of 40)")
out = {"family_loo": {g: {k: v for k, v in r.items() if k != "family_digests"}
                      for g, r in results.items()}, "wild": {}}
for gname, r in results.items():
    fam_set = set(r["family_digests"])
    n_out = sum(1 for w in wild.values() if w["digests"][gname] not in fam_set)
    lies_out = [j for j, w in wild.items() if w["is_lie"] and w["digests"][gname] not in fam_set]
    lies_in = [j for j, w in wild.items() if w["is_lie"] and w["digests"][gname] in fam_set]
    out["wild"][gname] = {"quorum_outside": n_out, "quorum_total": len(wild),
                          "lies_outside": lies_out, "lies_inside": lies_in}
    print(f"[{gname}] wild quorum outside family {n_out}/{len(wild)} | "
          f"LIES outside {lies_out} inside {lies_in}")
json.dump(out, open(".cache/graph_register_read.json", "w"), indent=1)
print("[done] .cache/graph_register_read.json")
