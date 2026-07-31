"""dup_isolation_rung.py — THE ISOLATION RUNG (2026-07-31, registered).
Surface changed, ORDERING RETAINED: distractors-first (the re-engagement
law's unstable configuration — the causal candidate) with the held-out's
surface conventions ("The sum of X and X is Y", 5-6 distractors).
g22 HIGH here = the configuration is the species, the fire's cure real;
g22 LOW = the difficulty was surface-narrow; arm read alongside."""
import sys, os, json
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_HW", "512"); os.environ.setdefault("ALG_DUP", "1")
import numpy as np
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_alg2_dials import solve2
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
tok = Tokenizer.from_file(TOKENIZER_JSON)
L = "abcdefghij"

def mint_iso(n_target, seed):
    rng = np.random.RandomState(seed)
    rows, tries = [], 0
    while len(rows) < n_target and tries < n_target * 15:
        tries += 1
        op = "add" if rng.rand() < 0.5 else "mul"
        x = int(rng.randint(2, 60)) if op == "add" else int(rng.randint(2, 13))
        n_dist = int(rng.randint(5, 7))                 # held-out surface: 5-6
        gv = [int(rng.randint(2, 90)) for _ in range(n_dist)]
        gold = x + x if op == "add" else x * x
        if gold > 300: continue
        dv = n_dist; res = n_dist + 1                   # ORDERING RETAINED: distractors FIRST
        facs = [{"ftype": "given", "var": i, "value": gv[i]} for i in range(n_dist)]
        facs.append({"ftype": "given", "var": dv, "value": x})
        facs.append({"ftype": "rel", "op": op, "args": [dv, dv], "result": res})
        word = "sum" if op == "add" else "product"      # held-out surface phrasing
        order = list(range(n_dist)); rng.shuffle(order)
        sents = [f"{L[i]} is {gv[i]}." for i in order] + \
                [f"{L[dv]} is {x}.", f"The {word} of {L[dv]} and {L[dv]} is {L[res]}."]
        letters = ", ".join(L[:res+1])
        text = f"Consider the numbers {letters}. " + " ".join(sents) + f" What is {L[res]}?"
        if solve2(facs, res, {"n_vars": 24, "m": 300}) != gold: continue
        rows.append({"text": text, "dv": dv, "op": op})
    return rows

rows = mint_iso(100, 71000)
print(f"[iso] {len(rows)} rows: distractors-FIRST + held-out surface")

def load_gate(ckpt):
    p = build_params(0)
    sd = safe_load(ckpt)
    for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    return p

def misbind(p, rows):
    mis = 0
    for r in rows:
        ids = np.zeros((8, T_ALG), np.int32); msk = np.zeros((8, T_ALG), np.float32); snt = np.zeros((8, T_ALG), np.int32)
        e = tok.encode(r["text"]); Ln = min(len(e.ids), T_ALG)
        ids[0, :Ln] = e.ids[:Ln]; msk[0, :Ln] = 1.0
        snt[0] = sent_indices(r["text"], list(e.offsets), msk[0])
        out = forward(p, Tensor(recompute_states(ids).astype(np.float32), dtype=dtypes.float),
                      Tensor(msk, dtype=dtypes.float), Tensor(snt, dtype=dtypes.int))
        keys = ("pres","ftype","op","islit","dig","args","res","query") + (("sel",) if "sel" in out else ()) + (("dup",) if "dup" in out else ())
        o = {k: out[k].realize().numpy() for k in keys}
        facs, _ = decode({k: o[k][0] for k in o})
        ok = any(f.get("ftype") == "rel" and f.get("args") == [r["dv"], r["dv"]]
                 and f.get("op") == r["op"] for f in facs)
        mis += (not ok)
    return mis

for name, ckpt in (("g22", ".cache/g22.safetensors"),
                   ("arm_dry_d02", ".cache/g23_dry_d02.safetensors")):
    p = load_gate(ckpt)
    m = misbind(p, rows)
    print(f"[{name}] ISO misbound {m}/{len(rows)} = {m/len(rows):.0%}", flush=True)
