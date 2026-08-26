"""raw_solve_read.py — ATLAS-FREE substrate health read (2026-08-25):
single-shot two-pass decode + solve_forced on the standard 3 fixtures.
No loop, no chains, no revoke input — pure parse health of RAW_CKPT.
Compares rings vs real on identical rows; the v2.5 salvage's step 1.
"""
import os, sys, json, glob
os.environ.setdefault("ALG_BREATH", "7")
os.environ.setdefault("ALG_NOTEBOOK", "1")
os.environ.setdefault("ALG_SIXWAVE", "1")
os.environ.update({"DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "9",
                   "ALG_DUP": "1", "ALG_HW": "512", "ALG_WIDE": "1",
                   "ALG_TEST": ".cache/algebra_nl_bigtest.jsonl",
                   "ALG_TEST_NAME": "bigtest"})
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
from phase1_algebra_head import (build_params, forward, decode, T_ALG,
                                 TOKENIZER_JSON, sent_indices, load_alg,
                                 build_slot_masks, L_FAC)
from repair_replace_swap import solve_forced
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
from beacon_closing_arm import recompute_states

_ = load_alg("test")
tok = Tokenizer.from_file(TOKENIZER_JSON)
K = ("pres", "ftype", "op", "islit", "dig", "args", "res", "query")

def main():
    ck = os.environ.get("RAW_CKPT", "gsb227_rings")
    p = build_params(0)
    sd = safe_load(f'.cache/{ck}.safetensors')
    assert set(sd.keys()) == set(p.keys()), \
        f"{ck} key mismatch: {len(set(sd)-set(p))}/{len(set(p)-set(sd))}"
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    byid = {}
    for f in sorted(glob.glob('.cache/book*_t*_batch*.jsonl')):
        for l in open(f): r = json.loads(l); byid[r["src_idx"]] = r
    for l in open('.cache/book12_anchor_batch1.jsonl'):
        r = json.loads(l); byid[r["src_idx"]] = r
    sk = set(json.load(open('.cache/book12_anchor_skips.json')))
    gold = [{"original": v["original"], "answer": v["answer"], "tag": "gold"}
            for k, v in sorted(byid.items()) if k not in sk]
    wv = [json.loads(l) for l in open('.cache/g55_wildval.jsonl')]
    never = [{"original": r["original"], "answer": r["answer"], "tag": "wv"}
             for r in wv]
    dd = [json.loads(l) for l in open('.cache/base_t7self_deeds.jsonl')]
    h = [json.loads(l) for l in open('.cache/math_harvest_v0.jsonl')]
    drafted = set(byid) | sk | set(r["src_idx"] for r in dd)
    for seed in (99, 299):
        rg = np.random.default_rng(seed)
        never += [{"original": h[i]["problem"],
                   "answer": int(str(h[i]["answer"]).strip()), "tag": "held"}
                  for i in rg.permutation(len(h)) if i not in drafted
                  and str(h[i]["answer"]).strip().isdigit()][:10]
    rows = gold + never
    T = {t: [0, 0, 0] for t in ("gold", "wv", "held")}
    for s0 in range(0, len(rows), 8):
        sl = rows[s0:s0 + 8]
        ids = np.zeros((8, T_ALG), np.int32); msk = np.zeros((8, T_ALG), np.float32)
        snt = np.zeros((8, T_ALG), np.int32)
        for li, r in enumerate(sl):
            e = tok.encode(r["original"])
            if len(e.ids) > T_ALG: continue
            ids[li, :len(e.ids)] = e.ids; msk[li, :len(e.ids)] = 1.0
            snt[li] = sent_indices(r["original"], list(e.offsets), msk[li])
        sts = np.asarray(recompute_states(ids)).astype(np.float32)
        ts = Tensor(sts, dtype=dtypes.float)
        tk = Tensor(msk, dtype=dtypes.float)
        se = Tensor(snt.astype(np.int32), dtype=dtypes.int)
        o0 = forward(p, ts, tk, se)
        onp0 = {k2: o0[k2].realize().numpy() for k2 in ("fat", "args", "res")}
        mk = build_slot_masks(onp0, snt)
        o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float))
        ex = tuple(k2 for k2 in ("sel", "dup", "sgn") if k2 in o)
        onp = {k2: o[k2].realize().numpy() for k2 in K + ex}
        for li, r in enumerate(sl):
            facs, q = decode({k2: onp[k2][li] for k2 in onp})
            try:
                a = solve_forced(facs, q, {"n_vars": 24, "m": 300})
            except Exception:
                a = None
            t = T[r["tag"]]
            if a is not None:
                t[0] += 1
                if a == r["answer"]: t[1] += 1
                else: t[2] += 1
    for tag, t in T.items():
        n = {"gold": 143, "wv": 20, "held": 20}[tag]
        print(f"[raw {ck} {tag}] forced {t[0]}/{n} right {t[1]} lies {t[2]} "
              f"(net {t[1]-t[2]})", flush=True)

if __name__ == "__main__":
    main()
