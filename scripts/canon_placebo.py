"""canonical_residue.py — THE CANONICAL-REORDER DOOR (registered
2026-08-15, bars pinned pre-fire): len_asc as a single view on the full
233 + all solved bigtest rows, gate checkpoint, zero training.
Bars: conversion >=55% of the 233; solved hold >=0.95; residue banked
as a frozen text-keyed fixture; enrichment profile (weakening clause
pinned). See the ledger entry for full forms.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import numpy as np

from phase1_algebra_head import (  # noqa: E402
    T_ALG, build_params, forward, load_alg, decode, sent_indices,
    ALG_CKPT, TOKENIZER_JSON,
)
from beacon_closing_arm import recompute_states  # noqa: E402
from repair_replace_swap import solve_forced  # noqa: E402

ORD = re.compile(r"(first|second|third|fourth|fifth) number")


def len_asc(text):
    import random as _r
    p = re.split(r"(?<=\.)\s+", text.strip())
    if len(p) <= 3:
        return text
    m = p[1:-1]
    _r.Random(4242 + len(text)).shuffle(m)
    return " ".join([p[0]] + m + [p[-1]])


def main():
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(TOKENIZER_JSON)
    samples, states, tokmask, gold, _sent = load_alg("test")
    n = len(samples)
    gold_ans = {i: samples[i]["solution"][samples[i]["query_var"]]
                for i in range(n)}

    p = build_params(0)
    sd = safe_load(ALG_CKPT)
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

    def parse_rows(sts, msk, snt, rows):
        ans = {}
        m = len(rows)
        for s0 in range(0, m, 8):
            sl = np.arange(s0, min(s0 + 8, m))
            pad = 8 - len(sl)
            slp = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
            out = forward(p, Tensor(sts[slp].astype(np.float32),
                                    dtype=dtypes.float),
                          Tensor(msk[slp].astype(np.float32),
                                 dtype=dtypes.float),
                          Tensor(snt[slp].astype(np.int32),
                                 dtype=dtypes.int))
            o = {k: out[k].realize().numpy() for k in
                 ("pres", "ftype", "op", "islit", "dig", "args", "res",
                  "query")}
            for bi, li in enumerate(sl):
                ri = rows[int(li)]
                facs, q = decode({k: o[k][bi] for k in o})
                ans[ri] = solve_forced(facs, q, samples[ri])
        return ans

    census = json.load(open(".cache/miss_census_gen41.json"))
    miss = sorted(int(i) for i in census["miss_idx"])
    solved = [i for i in range(n) if i not in set(miss)]
    print(f"[canon] 233={len(miss)} solved={len(solved)}", flush=True)

    # len_asc view over ALL rows, chunked to bound memory
    va = {}
    CH = 256
    for c0 in range(0, n, CH):
        rows = list(range(c0, min(c0 + CH, n)))
        ids = np.zeros((len(rows), T_ALG), np.int32)
        msk = np.zeros((len(rows), T_ALG), np.float32)
        snt = np.zeros((len(rows), T_ALG), np.int32)
        for li, ri in enumerate(rows):
            t = len_asc(samples[ri]["text"])
            e = tok.encode(t)
            if len(e.ids) > T_ALG:
                continue
            ids[li, :len(e.ids)] = e.ids
            msk[li, :len(e.ids)] = 1.0
            snt[li] = sent_indices(t, list(e.offsets), msk[li])
        sts = recompute_states(ids)
        va.update(parse_rows(sts, msk, snt, rows))
        print(f"[canon] rows {c0}-{rows[-1]} done", flush=True)

    conv = [i for i in miss if va.get(i) == gold_ans[i]]
    residue = [i for i in miss if va.get(i) != gold_ans[i]]
    held = sum(1 for i in solved if va.get(i) == gold_ans[i])
    print(f"[canon] CONVERSION {len(conv)}/{len(miss)} = "
          f"{len(conv) / len(miss):.3f}  (bar >=0.55)", flush=True)
    print(f"[canon] SOLVED HOLD {held}/{len(solved)} = "
          f"{held / len(solved):.4f}  (bar >=0.95)", flush=True)

    def profile(rows, tag):
        if not rows:
            print(f"[prof {tag}] empty", flush=True)
            return {}
        o = np.mean([1.0 if ORD.search(samples[i]["text"]) else 0.0
                     for i in rows])
        dup = np.mean([1.0 if any(f["ftype"] == "rel"
                                  and len(f.get("args", []))
                                  != len(set(f.get("args", [])))
                                  for f in samples[i]["factors"]) else 0.0
                       for i in rows])
        nsent = np.mean([len(re.split(r"(?<=\.)\s+",
                                      samples[i]["text"].strip()))
                         for i in rows])
        nfac = np.mean([len(samples[i]["factors"]) for i in rows])
        unf = np.mean([1.0 if va.get(i) is None else 0.0 for i in rows])
        print(f"[prof {tag}] n={len(rows)} ordinal {o:.3f}  dup-args "
              f"{dup:.3f}  sents {nsent:.1f}  factors {nfac:.1f}  "
              f"unforced-under-view {unf:.3f}", flush=True)
        return {"ordinal": float(o), "dup": float(dup),
                "nsent": float(nsent), "nfac": float(nfac)}

    pc = profile(conv, "converted")
    pr = profile(residue, "residue")
    if pc and pr:
        print(f"[enrich] ordinal x{pr['ordinal'] / max(pc['ordinal'], 1e-9):.2f}"
              f"  dup x{pr['dup'] / max(pc['dup'], 1e-9):.2f}"
              f"  sents x{pr['nsent'] / max(pc['nsent'], 1e-9):.2f}"
              f"  factors x{pr['nfac'] / max(pc['nfac'], 1e-9):.2f}",
              flush=True)

    try:
        eng = json.load(open(".cache/engineered_views.json"))
        u17 = set(eng["union_eng"])
        print(f"[overlap] residue ∩ odd-union17: "
              f"{len(u17 & set(residue))}/{len(u17)}   trio in residue: "
              f"{sorted(set([183, 332, 825]) & set(residue))}", flush=True)
    except Exception as e:
        print(f"[overlap] skipped ({e})", flush=True)

    fixture = [{"idx": i,
                "sha": hashlib.sha256(
                    samples[i]["text"].encode()).hexdigest()[:16],
                "mode": "unforced" if va.get(i) is None else "wrong"}
               for i in residue]
    json.dump({"ckpt": os.path.basename(ALG_CKPT), "view": "random_reorder_placebo",
               "n_residue": len(residue), "rows": fixture},
              open(".cache/residue_census_placebo.json", "w"), indent=1)
    print(f"[canon] RESIDUE BANKED: {len(residue)} rows -> "
          f".cache/residue_census_placebo.json (text-sha keyed)", flush=True)
    print("== CANON COMPLETE ==", flush=True)


if __name__ == "__main__":
    main()
