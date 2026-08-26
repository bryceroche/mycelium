"""mask_ladder.py — M1 + M2: THE MASKS START EARNING THEIR KEEP
(2026-08-26, word given). The inventory that provoked this: the deployed
head uses ONE mask — symmetric, binary, built from breath-0's rough
decode, frozen across all seven cycles — while the position-channel kill
proved wiring is RELATIONAL and the mask is the only between-slots
object in the head (the pointer law names masks as a binding remedy).
Arms, all eval-only on gsb227_sharp10k (clean substrate, no rings):
  RAW    — the incumbent two-pass (in-harness control).
  M1     — MASK REFRESH: rebuild build_slot_masks from the FINAL
           decode's sharpened outputs, re-breathe (reconstruction, not
           amputation — the first ADDITION test past the fixed point).
  M2     — DIRECTED SELF-WIRING: the decoded graph's own producer->
           consumer edges as an ASYMMETRIC mask (consumer attends its
           producers + self; sc2 masking is already asymmetry-capable —
           never used until now).
BARS (pinned pre-read): per arm, PASS = gold net STRICTLY > the
in-harness RAW arm's; never-seen lies reported; KILL per arm if net
degrades. M2b (enum-derived wirings for unsolvable rows) registered,
not in this read.
"""
import os, sys, json, glob
os.environ.setdefault("NB_PERSLOT", "1")
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

def var_of(f):
    return f.get("result", f.get("var", -1))

def directed_mask(onp, li):
    """consumer -> producer edges from the decoded graph's own pointers
    (+ self). Asymmetric on purpose: wiring has direction."""
    m = np.eye(L_FAC, dtype=np.float32)
    facs, _q = decode({k2: onp[k2][li] for k2 in onp})
    present = [j for j in range(L_FAC) if onp["pres"][li, j] > 0.5]
    slot_of = {fi: present[fi] for fi in range(len(facs))
               if fi < len(present)}
    prod = {}
    for fi, f in enumerate(facs):
        v = var_of(f)
        if fi in slot_of and v >= 0:
            prod.setdefault(v, []).append(slot_of[fi])
    for fi, f in enumerate(facs):
        if fi not in slot_of: continue
        j = slot_of[fi]
        for a in f.get("args", []):
            for k in prod.get(a, []):
                m[j, k] = 1.0
    return m

def main():
    p = build_params(0)
    ck = os.environ.get("ML_CKPT", "gsb227_sharp10k")
    sd = safe_load(f'.cache/{ck}.safetensors')
    assert set(sd.keys()) == set(p.keys())
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
    T = {t: {a: [0, 0, 0] for a in ("raw", "m1", "m2")}
         for t in ("gold", "wv", "held")}
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
        M0 = build_slot_masks(onp0, snt)
        o1 = forward(p, ts, tk, se, slot_mask=Tensor(M0, dtype=dtypes.float))
        ex = tuple(k2 for k2 in ("sel", "dup", "sgn") if k2 in o1)
        onp1 = {k2: o1[k2].realize().numpy() for k2 in K + ex + ("fat",)}
        onp1_ptr = {k2: onp1[k2] for k2 in ("fat", "args", "res")}
        M1 = build_slot_masks(onp1_ptr, snt)            # refresh from sharp
        M2 = np.stack([directed_mask(onp1, li) for li in range(8)])
        o2a = forward(p, ts, tk, se, slot_mask=Tensor(M1, dtype=dtypes.float))
        o2b = forward(p, ts, tk, se, slot_mask=Tensor(M2, dtype=dtypes.float))
        onp2a = {k2: o2a[k2].realize().numpy() for k2 in K + ex}
        onp2b = {k2: o2b[k2].realize().numpy() for k2 in K + ex}
        for li, r in enumerate(sl):
            for arm, onp in (("raw", onp1), ("m1", onp2a), ("m2", onp2b)):
                facs, q = decode({k2: onp[k2][li] for k2 in onp})
                try:
                    a = solve_forced(facs, q, {"n_vars": 24, "m": 300})
                except Exception:
                    a = None
                t = T[r["tag"]][arm]
                if a is not None:
                    t[0] += 1
                    if a == r["answer"]: t[1] += 1
                    else: t[2] += 1
    for tag in ("gold", "wv", "held"):
        n = {"gold": 143, "wv": 20, "held": 20}[tag]
        parts = []
        for arm in ("raw", "m1", "m2"):
            t = T[tag][arm]
            parts.append(f"{arm.upper()} f{t[0]} r{t[1]} l{t[2]} "
                         f"(net {t[1]-t[2]})")
        print(f"[mask {tag}/{n}] " + "  |  ".join(parts), flush=True)

if __name__ == "__main__":
    main()
