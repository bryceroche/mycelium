"""retrieval_a1.py — THE TWO-WAVE MATCHER (2026-08-23, word given;
surface-vs-subsurface gut). Surface wave: pooled pure-trunk kNN finds
the FAMILY (A0-receipted, cos .9+). Subsurface wave: the head's own
parsed OP-SKELETON — the multiset of (ftype, op) — picks the TWIN
among surface-tied neighbors (the binding theorem operationalized:
structure lives below the text embedding). Rebind winner (A0 machinery)
-> solve_forced -> grade once. Baseline: A0 = 0/40 right, 18 solved.
"""
import os, sys, json, glob
os.environ.update({"DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "8",
                   "ALG_DUP": "1", "ALG_HW": "512", "ALG_WIDE": "1",
                   "ALG_TEST": ".cache/algebra_nl_bigtest.jsonl",
                   "ALG_TEST_NAME": "bigtest", "ALG_TRUNK_LORA": "1"})
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
from collections import Counter
from phase1_algebra_head import (build_params, forward, decode, T_ALG,
                                 TOKENIZER_JSON, sent_indices, load_alg)
from repair_replace_swap import solve_forced
from beacon_closing_arm import recompute_states, _trunk_host
from mycelium.llama_loader import _rms_norm
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
from retrieval_bridge import pooled, rebind, num_seq

_ = load_alg("test")
tok = Tokenizer.from_file(TOKENIZER_JSON)
READER = os.environ.get("A1_READER", ".cache/g65_bridge.safetensors")
p = build_params(0); sd = safe_load(READER)
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
host = _trunk_host()
LD = [{f"{nm}_{ab}": (p[f"lora{li}_{nm}_{ab}"] * (8.0 if ab == "B" else 1.0))
       for nm in ("wq", "wo", "wdown") for ab in ("A", "B")} for li in range(4)]
K = ("pres", "ftype", "op", "islit", "dig", "args", "res", "query")

def skeleton(facs):
    return Counter((f["ftype"], f.get("op", f.get("name", ""))) for f in facs
                   if f["ftype"] != "given")

def head_skeletons(texts):
    out = []
    for s0 in range(0, len(texts), 8):
        sl = texts[s0:s0 + 8]
        ids = np.zeros((8, T_ALG), np.int32); msk = np.zeros((8, T_ALG), np.float32)
        snt = np.zeros((8, T_ALG), np.int32)
        for li, t in enumerate(sl):
            e = tok.encode(t)
            if len(e.ids) > T_ALG: continue
            ids[li, :len(e.ids)] = e.ids; msk[li, :len(e.ids)] = 1.0
            snt[li] = sent_indices(t, list(e.offsets), msk[li])
        x = host.llama_embed[Tensor(ids, dtype=dtypes.int)]
        for li2, layer in enumerate(host.llama_layers):
            x = layer(x, host.llama_rope_cos, host.llama_rope_sin, lora=LD[li2])
        x = _rms_norm(x, host.llama_layers[-1].ffn_norm, host.llama_cfg.rms_norm_eps)
        sts = x.cast(dtypes.float).realize().numpy()
        o = forward(p, Tensor(sts.astype(np.float32), dtype=dtypes.float),
                    Tensor(msk.astype(np.float32), dtype=dtypes.float),
                    Tensor(snt.astype(np.int32), dtype=dtypes.int))
        ex = tuple(k2 for k2 in ("sel", "dup", "sgn") if k2 in o)
        onp = {k2: o[k2].realize().numpy() for k2 in K + ex}
        for bi in range(len(sl)):
            facs, q = decode({k2: onp[k2][bi] for k2 in onp})
            out.append(skeleton(facs))
    return out

def agree(a, b):
    inter = sum((a & b).values())
    return inter / max(sum(a.values()), sum(b.values()), 1)

def main():
    _all = [json.loads(l) for f in sorted(glob.glob('.cache/book*_t*_batch*.jsonl'))
            for l in open(f) if l.strip()]
    byid = {r["src_idx"]: r for r in _all}
    for l in open('.cache/book12_anchor_batch1.jsonl'):
        r = json.loads(l); byid[r["src_idx"]] = r
    sk = set(json.load(open('.cache/book12_anchor_skips.json')))
    wvs = set(json.loads(l)["src_idx"] for l in open('.cache/g55_wildval.jsonl'))
    index_rows = [v for k, v in sorted(byid.items()) if k not in sk and k not in wvs]
    wv = [json.loads(l) for l in open('.cache/g55_wildval.jsonl')]
    fixtures = [{"original": r["original"], "answer": r["answer"]} for r in wv]
    dd = [json.loads(l) for l in open('.cache/base_t7self_deeds.jsonl')]
    h = [json.loads(l) for l in open('.cache/math_harvest_v0.jsonl')]
    drafted = set(byid) | sk | set(r["src_idx"] for r in dd)
    for seed in (99, 299):
        rg = np.random.default_rng(seed)
        fixtures += [{"original": h[i]["problem"],
                      "answer": int(str(h[i]["answer"]).strip())}
                     for i in rg.permutation(len(h)) if i not in drafted
                     and str(h[i]["answer"]).strip().isdigit()][:10]
    E_idx = pooled([r["original"] for r in index_rows])
    E_q = pooled([r["original"] for r in fixtures])
    if os.environ.get("A1_MODE", "head") == "textops":
        # A2: the subsurface probe made SYMBOLIC — operator characters
        # inside math runs, reader-free (twins differ by visible chars)
        import re as _re
        _MR = _re.compile(r"\$[^$]+\$")
        def _textops(t):
            ops = Counter()
            for m in _MR.finditer(t):
                for ch in m.group(0):
                    if ch in "+-*/^%=<>": ops[ch] += 1
            return ops
        q_skel = [_textops(r["original"]) for r in fixtures]
        n_skel = [_textops(r["original"]) for r in index_rows]
    else:
        q_skel = head_skeletons([r["original"] for r in fixtures])
        n_skel = [skeleton(r["factors"]) for r in index_rows]
    sims = E_q @ E_idx.T
    right = 0; attempted = 0; flips = 0
    for qi, q in enumerate(fixtures):
        top = np.argsort(-sims[qi])[:8]
        # subsurface re-rank: skeleton agreement first, surface cos second
        ranked = sorted(top, key=lambda ni: (-agree(q_skel[qi], n_skel[int(ni)]),
                                             -sims[qi][int(ni)]))
        if int(ranked[0]) != int(top[0]): flips += 1
        got = None
        for ni in ranked:
            facs = rebind(index_rows[int(ni)], q["original"])
            if facs is None: continue
            try:
                a = solve_forced(facs, index_rows[int(ni)]["query"],
                                 {"n_vars": 24, "m": 300})
            except Exception:
                continue
            if a is not None:
                got = a; break
        if got is not None:
            attempted += 1
            if got == q["answer"]: right += 1
    print(f"[rb A1] fixtures 40: SOLVED {attempted} RIGHT {right} "
          f"(subsurface flipped ranking on {flips}/40; A0 baseline 18/0)",
          flush=True)

if __name__ == "__main__":
    main()
