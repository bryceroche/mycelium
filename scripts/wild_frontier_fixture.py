"""wild_frontier_fixture.py — THE STANDING REHEARSAL (2026-07-30, gut
#101's yield): outside air on a schedule, not by occasion.

The fixture: the first 40 wild L4 harvest problems (deterministic
filter — <300 chars, no asy, integer answers 0..300, values <=300),
raw text, deployment mode, no annotation. Two reads, every generation
and every diet:

  DEPLOYED  — mouth first (length-corrected kNN, thr from
              mouth_length_correction.npz), then vote/trace/attest on
              survivors. This is the chain as shipped.
  STRESS    — mouth bypassed: vote -> trace layer (7 markers + value
              consistency) -> attest. This reads the INNER fences, which
              the mouth's current total refusal would otherwise hide;
              the lie exposure this read measures is the exposure that
              goes live the day the mouth's frontier widens.

Read out: certified / correct / lies per tier. The g21 baseline
(2026-07-30): DEPLOYED 0 certified (mouth refuses 40/40); STRESS
1 certified / 0 correct / 1 lie (trace layer holds 3 of the 4
value-fabrication lies; survivor is the coordinate-geometry register
case). A generation that raises STRESS lies has widened a wound; a
generation that certifies-correct under DEPLOYED has genuinely moved
the frontier. Registered ancestry: the wild rehearsal + the four
autopsies (ledger 2026-07-30). Never trained on; MATH-500 untouched
(this is the MATH *training* split's wild slice)."""
import sys, os, json, re
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_HW", "512"); os.environ.setdefault("ALG_DUP", "1")
import numpy as np
from collections import Counter
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_views import permuted_view
from tta_alg2_dials import solve2
from mycelium.attestation import attest_quorum_v3
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

CKPT = os.environ.get("WFF_CKPT", ".cache/g21.safetensors")
OUT = os.environ.get("WFF_OUT", ".cache/wild_frontier_fixture.json")
tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
sd = safe_load(CKPT)
assert set(sd.keys()) == set(p.keys())
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

def int_answer(a):
    s = str(a).strip().replace("$", "").replace(",", "")
    return int(s) if re.fullmatch(r"-?\d+", s) else None

h = [json.loads(l) for l in open(".cache/math_harvest_v0.jsonl")]
cands = [x for x in h if x["level"] == "Level 4" and len(x["problem"]) < 300
         and "asy]" not in x["problem"]
         and all(int(n) <= 300 for n in re.findall(r"\d+", x["problem"]))
         and int_answer(x["answer"]) is not None
         and 0 <= int_answer(x["answer"]) <= 300][:40]

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

# --- the mouth (deployed tier's first link) ---
zc = np.load(".cache/mouth_length_correction.npz")
coef, mthr = zc["coef"], float(zc["thr"])
bank = np.load(".cache/recognition_mouth_gen9b.npz")["bank"]
texts = [x["problem"] for x in cands]
V, Ls = [], []
for s0 in range(0, len(texts), 8):
    chunk = texts[s0:s0+8]
    ids = np.zeros((8, T_ALG), np.int32); msk = np.zeros((8, T_ALG), np.float32)
    for i, t in enumerate(chunk):
        e = tok.encode(t); L = min(len(e.ids), T_ALG)
        ids[i, :L] = e.ids[:L]; msk[i, :L] = 1.0
    s = recompute_states(ids).astype(np.float32)
    v = (s * msk[:, :, None]).sum(1) / np.maximum(msk.sum(1)[:, None], 1)
    V.append(v[:len(chunk)]); Ls.append(msk.sum(1)[:len(chunk)])
V = np.concatenate(V); Ls = np.concatenate(Ls)
V /= np.linalg.norm(V, axis=1, keepdims=True)
mouth_res = np.sort(1.0 - V @ bank.T, axis=1)[:, :8].mean(1) - (coef[0] + coef[1] / Ls)
mouth_foreign = mouth_res > mthr

# --- the trace layer: the ONE shared fence (deep-clean unification) ---
from mycelium.trace_layer import trace_trips

def inner_chain(j, text, gold):
    """vote -> trace -> attest on raw text; returns (decision, is_lie)."""
    vt = [text] + [permuted_view(text, 88000 + 10*j + k) for k in range(1, 5)]
    views = [(f, q, solve2(f, q, {"n_vars": 24, "m": 300})) for f, q in parse_batch(vt)]
    nn = [a for _, _, a in views if a is not None]
    c = Counter(nn).most_common(1); plur, cnt = c[0] if c else (None, 0)
    if cnt < 3: return "abstain", False
    win = [f_ for (f_, q_, a_) in views if a_ == plur]
    if trace_trips(text, win): return "tripped", False
    att, _, _ = attest_quorum_v3(views, plur, text, 300, solve2)
    if not att: return "attest_block", False
    return ("certified_correct", False) if plur == gold else ("certified_lie", True)

tiers = {"deployed": Counter(), "stress": Counter()}
lies = {"deployed": [], "stress": []}
for j, x in enumerate(cands):
    gold = int_answer(x["answer"])
    d, lie = inner_chain(j, x["problem"], gold)
    tiers["stress"][d] += 1
    if lie: lies["stress"].append(j)
    if mouth_foreign[j]:
        tiers["deployed"]["mouth_refused"] += 1
    else:
        tiers["deployed"][d] += 1
        if lie: lies["deployed"].append(j)
print(f"[wild frontier fixture] ckpt {CKPT}  n=40")
for tier in ("deployed", "stress"):
    t = tiers[tier]
    print(f"  {tier.upper():9s}: " + "  ".join(f"{k} {v}" for k, v in sorted(t.items()))
          + (f"  LIE IDXS {lies[tier]}" if lies[tier] else ""))
json.dump({"ckpt": CKPT, "tiers": {k: dict(v) for k, v in tiers.items()},
           "lie_idxs": lies, "mouth_refused": int(mouth_foreign.sum())},
          open(OUT, "w"), indent=1)
print(f"[saved] {OUT}")
