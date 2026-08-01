"""fdiv_surface_check.py — THE DIET-HISTORY SCOPE CHECK (2026-07-31,
registered). Does gen-22's fdiv cure survive a surface it never saw?
Per the rotation law's second clause: SURFACE varied, CONFIGURATION
retained (fdiv on a 1-step-derived operand — the cured d1 cell).
The 2×2 with in-session anchors: {g21 pre-cure, g22 cured} ×
{original surface, varied surface}. Anchor cells recomputed in-session
(draw order now matches the reference instrument; the first run used a
swapped order — fairly sampled, not bit-identical). VARIED SURFACE: "The product of a and
b is c. Dividing c by K gives d." — phrasings absent from diet corpus
v3. SURVIVES = varied-under-g22 within 2 probes of original-under-g22;
FAILS = varied-under-g22 at or near the g21 anchor. Queues behind the
neighbor reads."""
import sys, os, json, subprocess, time
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_HW", "512"); os.environ.setdefault("ALG_DUP", "1")

while subprocess.run(["systemctl", "--user", "is-active", "dup-verdict.service"],
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
tok = Tokenizer.from_file(TOKENIZER_JSON)

def make_d1(rng, varied):
    # deep clean 2026-08-01: K drawn FIRST (the reference instrument's own
    # order — diet_after_read.make_row); the original A,B,K order consumed
    # the stream differently, so "exact reproduction" was false (the
    # internal 2x2 was still apples-to-apples; the keystone stands).
    K = int(rng.choice([2, 3, 4, 5, 6, 7]))
    A = int(rng.randint(2, 12)); B = int(rng.randint(2, 12))
    C = A * B
    if C % K or C > 300: return None
    if varied:
        text = (f"Consider the numbers a, b, c, d. a is {A}. b is {B}. "
                f"The product of a and b is c. Dividing c by {K} gives d. What is d?")
    else:
        text = (f"Consider the numbers a, b, c, d. a is {A}. b is {B}. "
                f"a times b equals c. When c is divided by {K}, the quotient is d. What is d?")
    facs = [{"ftype":"given","var":0,"value":A},{"ftype":"given","var":1,"value":B},
            {"ftype":"rel","op":"mul","args":[0,1],"result":2},
            {"ftype":"fdiv","var":2,"k":K,"result":3}]
    return text, facs, 3, C // K

def parse_batch(p, texts):
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

def quorum_pass(p, varied, seed0):
    rng = np.random.RandomState(4001)          # the d1 fixture's own seed
    n_ok = made = tries = 0
    while made < 8 and tries < 200:
        tries += 1
        pr = make_d1(rng, varied)
        if pr is None: continue
        text, facs, q, gold = pr
        assert solve2(facs, q, {"n_vars": 24, "m": 300}) == gold
        vt = [text] + [permuted_view(text, seed0 + 20*made + k) for k in range(1, 5)]
        views = [solve2(f, qq, {"n_vars": 24, "m": 300}) for f, qq in parse_batch(p, vt)]
        nn = [a for a in views if a is not None]
        c = Counter(nn).most_common(1); plur, cnt = c[0] if c else (None, 0)
        n_ok += (cnt >= 3 and plur == gold)
        made += 1
    return n_ok

results = {}
for name, ckpt in (("g21_precure", ".cache/g21.safetensors"),
                   ("g22_cured", ".cache/g22.safetensors")):
    p = build_params(0)
    sd = safe_load(ckpt)
    for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    orig = quorum_pass(p, False, 5000)
    var = quorum_pass(p, True, 6000)
    results[name] = {"original": orig, "varied": var}
    print(f"[{name}] d1 original-surface {orig}/8   varied-surface {var}/8", flush=True)

g22o, g22v = results["g22_cured"]["original"], results["g22_cured"]["varied"]
g21v = results["g21_precure"]["varied"]
if g22v >= g22o - 2:
    verdict = f"SURVIVES — varied {g22v}/8 within 2 of original {g22o}/8: the fdiv corpus's spanned axes bought real generality; the diet history stands"
elif g22v <= g21v + 1:
    verdict = f"FAILS — varied {g22v}/8 at the pre-cure anchor ({g21v}/8): the fdiv cure is also surface-local; the diet history gains a scope line; AUGMENTATION IS PREREQUISITE"
else:
    verdict = f"PARTIAL — varied {g22v}/8 between anchors (g21 {g21v}/8, g22-orig {g22o}/8): generalizes with attenuation; the attenuation is the split"
print(f"=== VERDICT (pinned): {verdict} ===")
json.dump({"results": results, "verdict": verdict},
          open(".cache/fdiv_surface_check.json", "w"), indent=1)
print("[saved] .cache/fdiv_surface_check.json")
