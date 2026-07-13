"""book1_prose_pairs.py — THE FIRST REAL-PROSE INGEST, v0 (gen-10,
2026-07-12; informational arm, n=14).

For each faithful book-1 entry: parse the DIALECT under the gate ckpt
(single unpermuted view), verify the parse solves to the banked answer,
and emit the RAW PROSE as a training row whose gold graph is that
verified parse — factors WITHOUT spans, mentions empty (raw prose has
no letter anchors; span losses auto-mask). The head gets its first
gradient from stranger prose; the census and raw-prose acceptance read
the needle before/after. n=14 moves nothing by itself — the arm exists
to build and measure the machinery honestly (Brick discipline).
"""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "6")
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_alg2_dials import solve2
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

entries = [json.loads(l) for l in open(".cache/book1.jsonl")]
entries = [e for e in entries if not e.get("residual")]

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
CKPT = os.environ.get("GATE_CKPT", ".cache/phase1_gen9b_head.safetensors")
sd = safe_load(CKPT)
assert set(sd.keys()) == set(p.keys())
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

def parse_one(text):
    ids = np.zeros((8, T_ALG), np.int32); msk = np.zeros((8, T_ALG), np.float32)
    snt = np.zeros((8, T_ALG), np.int32)
    e = tok.encode(text)
    L = min(len(e.ids), T_ALG)
    ids[0, :L] = e.ids[:L]; msk[0, :L] = 1.0
    snt[0] = sent_indices(text, list(e.offsets), msk[0])
    st = recompute_states(ids)
    out = forward(p, Tensor(st.astype(np.float32), dtype=dtypes.float),
                  Tensor(msk, dtype=dtypes.float),
                  Tensor(snt, dtype=dtypes.int))
    keys = ("pres","ftype","op","islit","dig","args","res","query") + \
        (("sel",) if "sel" in out else ()) + (("dup",) if "dup" in out else ())
    o = {k: out[k].realize().numpy()[0] for k in keys}
    return decode(o)

rows, skipped = [], 0
for e in entries:
    facs, q = parse_one(e["dialect"])
    a = solve2(facs, q, {"n_vars": 24, "m": 300})
    if a != e["answer"]:
        skipped += 1
        print(f"  [skip] idx {e['idx']}: dialect parse solves to {a} != {e['answer']}")
        continue
    rows.append({"n_vars": 24, "m": 300, "text": e["raw"],
                 "factors": [dict(f, spans=[]) for f in facs],
                 "mentions": {}, "query_var": q,
                 "solution": [0] * 24, "decisions": 0,
                 "gen": {"shape": "prose-v0", "generation": 10,
                         "src_idx": e["idx"]}})
with open(".cache/book1_prose_pairs.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"[prose-v0] {len(rows)} raw-prose training pairs "
      f"({skipped} skipped on verification) -> .cache/book1_prose_pairs.jsonl")
