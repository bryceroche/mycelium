"""audition_read_one.py — one candidate's audition read (subprocess:
candidate envs must be set BEFORE import — notebook/breath arms gate
module-level structure). Env: CAND_ID. Reads the fixture set (wild-val
20 + held-out 20 + census-100) through the candidate's ckpt with its
own rank/span/proj/ropeoff/nb envs; emits .cache/audition_{ID}.json:
per-row answers (null = refused). The recruiter consumes these.
"""
import os, sys, json, glob

CID = os.environ["CAND_ID"]
cfg = [r for r in json.load(open('docs/reader_pool.json')) if r["id"] == CID][0]
os.environ.update({"DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "8",
                   "ALG_DUP": "1", "ALG_HW": "512", "ALG_WIDE": "1",
                   "ALG_TEST": ".cache/algebra_nl_bigtest.jsonl",
                   "ALG_TEST_NAME": "bigtest", "ALG_TRUNK_LORA": "1",
                   "ALG_LORA_R": str(cfg["rank"]),
                   "ALG_LORA_SPAN": str(cfg["span"]),
                   "ALG_LORA_PROJ": cfg["proj"],
                   "ALG_ROPE_OFF": str(cfg["ropeoff"])})
if cfg.get("nb"):
    os.environ["ALG_NOTEBOOK"] = "1"; os.environ["ALG_BREATH"] = "3"
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
from phase1_algebra_head import (build_params, forward, decode, T_ALG,
                                 TOKENIZER_JSON, sent_indices, load_alg)
from repair_replace_swap import solve_forced
from beacon_closing_arm import recompute_states, _trunk_host
from mycelium.llama_loader import _rms_norm
from mycelium.clause_grains import canonicalize
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

_ = load_alg("test")
tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
sd = safe_load(f".cache/pool_{CID}.safetensors")
assert set(sd.keys()) == set(p.keys()), f"{CID} key mismatch"
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
host = _trunk_host()
RO = int(cfg["ropeoff"])
RC = host.llama_rope_cos[RO:] if RO else host.llama_rope_cos
RS = host.llama_rope_sin[RO:] if RO else host.llama_rope_sin
LD = [{f"{nm}_{ab}": (p[f"lora{li}_{nm}_{ab}"] * (8.0 if ab == "B" else 1.0))
       for nm in ("wq", "wo", "wdown") for ab in ("A", "B")
       if f"lora{li}_{nm}_A" in p} or None for li in range(4)]
K = ("pres", "ftype", "op", "islit", "dig", "args", "res", "query")
CANON = int(cfg.get("incanon", 0)) == 1

def read(rows):
    out = []
    for s0 in range(0, len(rows), 8):
        sl = rows[s0:s0 + 8]
        ids = np.zeros((8, T_ALG), np.int32); msk = np.zeros((8, T_ALG), np.float32)
        snt = np.zeros((8, T_ALG), np.int32)
        for li, r in enumerate(sl):
            t = canonicalize(r["original"]) if CANON else r["original"]
            e = tok.encode(t)
            if len(e.ids) > T_ALG: continue
            ids[li, :len(e.ids)] = e.ids; msk[li, :len(e.ids)] = 1.0
            snt[li] = sent_indices(t, list(e.offsets), msk[li])
        x = host.llama_embed[Tensor(ids, dtype=dtypes.int)]
        for li2, layer in enumerate(host.llama_layers):
            x = layer(x, RC, RS, lora=LD[li2])
        x = _rms_norm(x, host.llama_layers[-1].ffn_norm, host.llama_cfg.rms_norm_eps)
        sts = x.cast(dtypes.float).realize().numpy()
        o = forward(p, Tensor(sts.astype(np.float32), dtype=dtypes.float),
                    Tensor(msk.astype(np.float32), dtype=dtypes.float),
                    Tensor(snt.astype(np.int32), dtype=dtypes.int))
        ex = tuple(k2 for k2 in ("sel", "dup", "sgn") if k2 in o)
        onp = {k2: o[k2].realize().numpy() for k2 in K + ex}
        for bi in range(len(sl)):
            facs, q = decode({k2: onp[k2][bi] for k2 in onp})
            try:
                a = solve_forced(facs, q, {"n_vars": 24, "m": 300})
            except Exception:
                a = None
            out.append(a)
    return out

def fixtures():
    wv = [json.loads(l) for l in open('.cache/g55_wildval.jsonl')]
    rows = [{"original": r["original"], "answer": r["answer"], "tag": "wv"}
            for r in wv]
    _all = [json.loads(l) for f in sorted(glob.glob('.cache/book*_t*_batch*.jsonl'))
            for l in open(f) if l.strip()]
    byid = {r["src_idx"]: r for r in _all}
    for l in open('.cache/book12_anchor_batch1.jsonl'):
        r = json.loads(l); byid[r["src_idx"]] = r
    sk = set(json.load(open('.cache/book12_anchor_skips.json')))
    dd = [json.loads(l) for l in open('.cache/base_t7self_deeds.jsonl')]
    h = [json.loads(l) for l in open('.cache/math_harvest_v0.jsonl')]
    drafted = set(byid) | sk | set(r["src_idx"] for r in dd)
    for seed in (99, 299):
        rg = np.random.default_rng(seed)
        rows += [{"original": h[i]["problem"],
                  "answer": int(str(h[i]["answer"]).strip()), "tag": "held"}
                 for i in rg.permutation(len(h)) if i not in drafted
                 and str(h[i]["answer"]).strip().isdigit()][:10]
    cen = json.load(open('.cache/frontier_census_t7.json'))["rows"]
    take = [r for r in cen if r["src_idx"] not in drafted][:100]
    rows += [{"original": h[r["src_idx"]]["problem"],
              "answer": int(str(h[r["src_idx"]]["answer"]).strip()), "tag": "cen"}
             for r in take
             if str(h[r["src_idx"]]["answer"]).strip().isdigit()]
    return rows

rows = fixtures()
ans = read(rows)
json.dump({"id": CID,
           "rows": [{"tag": r["tag"], "key": r["answer"], "got": a}
                    for r, a in zip(rows, ans)]},
          open(f'.cache/audition_{CID}.json', 'w'))
n_r = sum(1 for r, a in zip(rows, ans) if a == r["answer"])
print(f"[audition {CID}] rows {len(rows)} right {n_r} "
      f"refused {sum(1 for a in ans if a is None)}", flush=True)
