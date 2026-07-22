"""collapse_probe.py — GUT #43's FIRE (2026-07-21): THE COLLAPSE-CROSSOVER
PROBE — where does the bottleneck live in the stack? Substrate: the
fluency held-out mints re-rendered — SAME KNOT in two keys (pairs) vs
DIFFERENT KNOTS in the same key (controls). Distances read at two
depths: pooled TRUNK states (the pretrained prior) and pooled FST rows
(the head's binding layer, crown_reader_v2). PINNED (kill-only): trunk
reads key < knot (frames bind); head reads knot < key (the compressor
inverts) — no inversion = the 27/27 rides on something other than
collapse, its own finding.
"""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
os.environ.setdefault("ALG2", "1"); os.environ["ALG_FTYPES"] = "8"
os.environ.setdefault("ALG_DUP", "1")
from phase1_algebra_head import T_ALG, build_params, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from waist_abstention_probe import compute_fst
from fluency_mint import render_macro, KEYS
from tokenizers import Tokenizer
from tinygrad.nn.state import safe_load
import random

held = [json.loads(l) for l in open(".cache/fluency_mint_held.jsonl")][:80]
rng = random.Random(9)
tok = Tokenizer.from_file(TOKENIZER_JSON)

# each held knot rendered in ITS key and one OTHER key -> same-knot pair
texts, meta = [], []
for i, r in enumerate(held):
    k1 = r["key"]
    k2 = rng.choice([k for k in KEYS if k != k1])
    t1 = r["macro"]["text"]
    t2 = render_macro(r["macro"]["factors"], r["query_var"], k2, rng)
    texts += [t1, t2]
    meta.append((i, k1, k2))

ids = np.zeros((len(texts), T_ALG), np.int32)
msk = np.zeros((len(texts), T_ALG), np.float32)
snt = np.zeros((len(texts), T_ALG), np.int32)
for i, t in enumerate(texts):
    e = tok.encode(t)
    L = min(len(e.ids), T_ALG)
    ids[i, :L] = e.ids[:L]; msk[i, :L] = 1.0
    snt[i] = sent_indices(t, list(e.offsets), msk[i])
st = recompute_states(ids)

# depth 1: pooled trunk
V = (st.astype(np.float32) * msk[:, :, None]).sum(1) / \
    np.maximum(msk.sum(1)[:, None], 1)
V /= np.linalg.norm(V, axis=1, keepdims=True)

# depth 2: pooled fst under crown_reader_v2
p = build_params(0)
sd = safe_load(os.environ.get("COLLAPSE_CKPT", ".cache/crown_reader_v2.safetensors"))
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
fst = compute_fst(p, st, msk.astype(np.uint8), snt, list(range(len(texts))))
F = fst.mean(1)
F /= np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-9)


def dists(M):
    same_knot = [1 - float(M[2*i] @ M[2*i+1]) for i in range(len(held))]
    # controls: same key, different knots
    by_key = {}
    for i, r in enumerate(held):
        by_key.setdefault(r["key"], []).append(2*i)
    same_key = []
    for k, idxs in by_key.items():
        for _ in range(min(len(idxs), 40)):
            a, b = rng.sample(idxs, 2)
            same_key.append(1 - float(M[a] @ M[b]))
    return np.mean(same_knot), np.mean(same_key)


kt, ky = dists(V)
print(f"[collapse] TRUNK:  d(same-knot,diff-key) {kt:.4f}  "
      f"d(same-key,diff-knot) {ky:.4f}  -> "
      f"{'key binds tighter (frames rule the prior)' if ky < kt else 'knot binds tighter ALREADY'}")
kt2, ky2 = dists(F)
print(f"[collapse] HEAD:   d(same-knot,diff-key) {kt2:.4f}  "
      f"d(same-key,diff-knot) {ky2:.4f}  -> "
      f"{'KNOT BINDS TIGHTER — THE INVERSION PRINTS' if kt2 < ky2 else 'no inversion'}")
inv_trunk = ky < kt
inv_head = kt2 < ky2
verdict = ("THE BOTTLENECK LIVES IN THE HEAD — inversion measured "
           "(trunk: frames rule; head: knots rule)" if inv_trunk and inv_head
           else "PARTIAL/NO INVERSION — the 27/27 rides on something else"
           if not inv_head else "knots rule at BOTH depths — collapse begins in the trunk")
print(f"[collapse] VERDICT: {verdict}")
json.dump({"trunk": [kt, ky], "head": [kt2, ky2], "verdict": verdict},
          open(".cache/collapse_probe.json", "w"))
print("[collapse] banked -> .cache/collapse_probe.json")
