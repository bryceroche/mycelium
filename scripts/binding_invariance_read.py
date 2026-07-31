"""binding_invariance_read.py — THE PRICING READ FOR GUT #110
(2026-07-31, registered; bars pinned in the ledger). Not
agreement-as-confidence (#38's kill stands): a BINDING test on the
input axis. Each of the wild ledger's answered items re-renders under a
semantic-preserving transform (variable renaming where variables
exist; phrase-swap lexicon otherwise), re-parses through the same
5-view quorum, and the transformed plurality compares to the original.
Priced against gold: does invariance separate the 78 correct from the
46 wrong? SUCCESS: invariant-precision >= 0.75 AND flipped <= 0.40
(support >= 20 each). FAIL: gap < 15 pts -> #38's kill extends to the
input axis. Untransformable items excluded and counted."""
import sys, os, json, re, subprocess, time
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_HW", "512"); os.environ.setdefault("ALG_DUP", "1")

# queue behind the pct read for the device
while subprocess.run(["systemctl", "--user", "is-active", "pct-read.service"],
                     capture_output=True, text=True).stdout.strip() == "active":
    time.sleep(20)

import numpy as np
from collections import Counter
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_views import permuted_view
from tta_alg2_dials import solve2
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

MAN = json.load(open(".cache/GENERATION.json"))
CKPT = MAN["parser_ckpt"]
tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
sd = safe_load(CKPT)
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
print(f"[binding] gate from manifest: {CKPT}")

recs = [json.loads(l) for l in open(".cache/wild_ledger_v1.jsonl")]
ans = [r for r in recs if r["tier"] == "answered"]
h = [json.loads(l) for l in open(".cache/math_harvest_v0.jsonl")]
print(f"[binding] answered items: {len(ans)} "
      f"(correct {sum(r['correct'] for r in ans)}, wrong {sum(not r['correct'] for r in ans)})")

VMAP = {"x": "t", "y": "u", "z": "w", "n": "m", "a": "p", "b": "q", "c": "r",
        "t": "s", "m": "k", "k": "j", "r": "v", "s": "d", "p": "g", "q": "h"}
PHRASES = [("What is the value of", "Determine the value of"),
           ("What is", "Determine"), ("How many", "Determine how many"),
           ("Find the", "Determine the"), ("Find", "Determine")]

def transform(text):
    """Variable renaming where single-letter math vars exist; else phrase
    swap. Returns (new_text, kind) or (None, None) if untransformable."""
    vars_ = set(re.findall(r"\$([a-z])\$", text)) | \
            set(re.findall(r"\$([a-z])[\^_ =+\-]", text))
    vars_ = {v for v in vars_ if v in VMAP}
    if vars_:
        out = text
        for v in sorted(vars_):
            out = re.sub(rf"(?<=\$){v}(?=[\$\^_ =+\-])", VMAP[v], out)
        if out != text:
            return out, "var-rename"
    for a, b in PHRASES:
        if a in text:
            return text.replace(a, b, 1), "phrase-swap"
    return None, None

def parse_batch(texts):
    n = len(texts); N = ((n+7)//8)*8
    ids = np.zeros((N, T_ALG), np.int32); msk = np.zeros((N, T_ALG), np.float32); snt = np.zeros((N, T_ALG), np.int32)
    for i, t in enumerate(texts):
        e = tok.encode(t); Ln = min(len(e.ids), T_ALG)
        ids[i, :Ln] = e.ids[:Ln]; msk[i, :Ln] = 1.0
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

results = []
skipped = Counter()
for j, r in enumerate(ans):
    text = h[r["harvest_idx"]]["problem"]
    nt, kind = transform(text)
    if nt is None:
        skipped["untransformable"] += 1; continue
    vt = [nt] + [permuted_view(nt, 96000 + 10*j + k) for k in range(1, 5)]
    views = [solve2(f, q, {"n_vars": 24, "m": 300}) for f, q in parse_batch(vt)]
    nn = [a for a in views if a is not None]
    c = Counter(nn).most_common(1); plur2, cnt2 = c[0] if c else (None, 0)
    invariant = (cnt2 >= 3 and plur2 == r["plur"])
    results.append({"harvest_idx": r["harvest_idx"], "kind": kind,
                    "correct": bool(r["correct"]), "invariant": bool(invariant),
                    "plur": r["plur"], "plur2": plur2, "q2": cnt2})
    if (j+1) % 25 == 0: print(f"[binding] {j+1}/{len(ans)}", flush=True)

inv = [r for r in results if r["invariant"]]
flp = [r for r in results if not r["invariant"]]
def prec(s): return sum(r["correct"] for r in s) / max(len(s), 1)
print(f"\n=== BINDING INVARIANCE (n={len(results)} transformed; skipped {dict(skipped)}) ===")
print(f"  INVARIANT: n={len(inv):3d}  precision {prec(inv):.3f}")
print(f"  FLIPPED:   n={len(flp):3d}  precision {prec(flp):.3f}")
for kind in ("var-rename", "phrase-swap"):
    s = [r for r in results if r["kind"] == kind]
    si = [r for r in s if r["invariant"]]
    print(f"  [{kind}] n={len(s)}  invariant {len(si)} (prec {prec(si):.2f}) "
          f"flipped {len(s)-len(si)} (prec {prec([r for r in s if not r['invariant']]):.2f})")
gap = (prec(inv) - prec(flp)) * 100
support = min(len(inv), len(flp))
if prec(inv) >= 0.75 and prec(flp) <= 0.40 and support >= 20:
    verdict = "SUCCESS — the first grading signal; the tier ladder revives"
elif gap < 15:
    verdict = "FAIL — #38's kill EXTENDS to the input axis (errors are invariant)"
else:
    verdict = "MIXED — per-transform map only; no signal claim"
print(f"=== gap {gap:+.0f} pts, min-support {support} -> VERDICT (pinned): {verdict} ===")
json.dump({"results": results, "skipped": dict(skipped),
           "prec_invariant": prec(inv), "prec_flipped": prec(flp),
           "gap_pts": gap, "verdict": verdict},
          open(".cache/binding_invariance.json", "w"), indent=1)
print("[saved] .cache/binding_invariance.json")
