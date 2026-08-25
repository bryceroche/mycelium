"""iter_a0.py — THE CLOSED LOOP v0: ITERATED A0 (2026-08-25, word
given). The vision's missing sixth component at minimum viable size:
decode -> chain-consistency check -> amputation becomes a SLOT MASK ->
re-breathe the full K=7 -> decode again -> repeat to fixpoint (cap 5).
Feedback between full breaths (v1 = inside the breath loop; v2 = the
CSP solver as the checker). Amputated slots lose their mask COLUMN
(no survivor attends to their evidence) and their factors are dropped
at decode.
BARS (pinned pre-read): net (rights - lies) on gold STRICTLY better
than single-shot A0 (4 - 26 = -22, reproduced in-harness); KILL if
fixpoint net <= single-shot net. Banked either way: iteration
distribution + mask-effect meter (did re-breathing change surviving
slots' decode at all).
"""
import os, sys, json, glob
os.environ.setdefault("ALG_MINE_BREATHS", "1")
os.environ.setdefault("ALG_BREATH", "7")
os.environ.setdefault("ALG_NOTEBOOK", "1")
os.environ.setdefault("ALG_SIXWAVE", "1")
os.environ.update({"DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "9",
                   "ALG_DUP": "1", "ALG_HW": "512", "ALG_WIDE": "1",
                   "ALG_TEST": ".cache/algebra_nl_bigtest.jsonl",
                   "ALG_TEST_NAME": "bigtest"})
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
from collections import Counter
from phase1_algebra_head import (build_params, forward, decode, T_ALG,
                                 TOKENIZER_JSON, sent_indices, load_alg,
                                 build_slot_masks, L_FAC)
from repair_replace_swap import solve_forced
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
from beacon_closing_arm import recompute_states
from chain_decode import load_atlas

_ = load_alg("test")
tok = Tokenizer.from_file(TOKENIZER_JSON)
K = ("pres", "ftype", "op", "islit", "dig", "args", "res", "query")
MAX_IT = int(os.environ.get("ITER_MAX", "5"))

def chain_labels(Bst, cents, ckinds, trans, cycles, li, pres_row):
    """per present slot: greedy transition-guided chain -> endpoint label
    (A0's exact recipe)."""
    lab = {}
    for j in range(L_FAC):
        if pres_row[j] <= 0.5: continue
        chain = []
        for ci, cyc in enumerate(cycles[:len(Bst)]):
            v = Bst[min(ci, len(Bst) - 1)][li, j]
            v = v / (np.linalg.norm(v) + 1e-9)
            bank = cents[cyc]; bids = list(bank.keys())
            sims = np.array([float(v @ bank[b]) for b in bids])
            order = np.argsort(-sims)[:3]
            if chain:
                tr = trans.get(cyc, {}).get(chain[-1], {})
                cand = [(sims[oi] + 0.2 * np.log1p(tr.get(bids[oi], 0)), oi)
                        for oi in order]
                _, oi = max(cand)
            else:
                oi = order[0]
            chain.append(bids[oi])
        for cid in reversed(chain):
            if cid in ckinds and ckinds[cid]:
                lab[j] = max(ckinds[cid], key=ckinds[cid].get)
                break
    return lab

def main():
    cents, ckinds, trans = load_atlas()
    cycles = sorted(cents)
    p = build_params(0)
    sd = safe_load('.cache/gsb227_real.safetensors')
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
    tallies = {t: {"single": [0, 0, 0], "iter": [0, 0, 0]}
               for t in ("gold", "wv", "held")}   # [forced, right, lies]
    it_hist = Counter(); changed_meter = 0; rows_iterated = 0
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
        M = build_slot_masks(onp0, snt)                    # (8, L, L)
        amp = [set() for _ in range(8)]                    # amputated slots
        frozen = [False] * 8
        # per-row records of the CURRENT decode + amputation-filtered facs
        rec = [None] * 8
        single = [None] * 8
        prev_sig = [None] * 8
        for it in range(MAX_IT):
            o = forward(p, ts, tk, se, slot_mask=Tensor(M, dtype=dtypes.float))
            ex = tuple(k2 for k2 in ("sel", "dup", "sgn") if k2 in o)
            onp = {k2: o[k2].realize().numpy() for k2 in K + ex}
            Bst = [b.realize().numpy() for b in o["breaths_all"]]
            new_any = False
            for li, r in enumerate(sl):
                if frozen[li]: continue
                lab = chain_labels(Bst, cents, ckinds, trans, cycles, li,
                                   onp["pres"][li])
                facs, q = decode({k2: onp[k2][li] for k2 in onp})
                present = [j for j in range(L_FAC)
                           if onp["pres"][li, j] > 0.5]
                keep = []; new_amp = set()
                for fi, f in enumerate(facs):
                    j = present[fi] if fi < len(present) else None
                    cl = lab.get(j)
                    if cl is None or cl == f["ftype"]:
                        keep.append(f)
                    elif j is not None:
                        new_amp.add(j)
                sig = json.dumps([sorted(new_amp),
                                  [(f["ftype"], f.get("op")) for f in keep]])
                rec[li] = (keep, q)
                if it == 0:
                    single[li] = (keep, q)                 # single-shot A0
                fresh = new_amp - amp[li]
                if it > 0 and sig != prev_sig[li]:
                    changed_meter += 1
                prev_sig[li] = sig
                if not fresh:
                    frozen[li] = True
                    it_hist[it] += 1
                    continue
                amp[li] |= fresh
                for j in amp[li]:
                    M[li, :, j] = 0.0                      # no one attends
                    M[li, j, :] = 0.0                      # to the dead slot
                    M[li, j, j] = 1.0                      # (self-loop kept)
                new_any = True
            if not new_any:
                break
        for li in range(8):
            if not frozen[li]:
                it_hist[MAX_IT] += 1
        for li, r in enumerate(sl):
            if rec[li] is None: continue
            if len(amp[li]) > 0: rows_iterated += 1
            T = tallies[r["tag"]]
            for name, rc in (("single", single[li]), ("iter", rec[li])):
                keep, q = rc
                try:
                    a = solve_forced(keep, q, {"n_vars": 24, "m": 300})
                except Exception:
                    a = None
                if a is not None:
                    T[name][0] += 1
                    if a == r["answer"]: T[name][1] += 1
                    else: T[name][2] += 1
    for t, T in tallies.items():
        n = {"gold": 143, "wv": 20, "held": 20}[t]
        s, i = T["single"], T["iter"]
        print(f"[it {t}] SINGLE forced {s[0]}/{n} right {s[1]} lies {s[2]} "
              f"(net {s[1]-s[2]})  |  ITER forced {i[0]} right {i[1]} "
              f"lies {i[2]} (net {i[1]-i[2]})", flush=True)
    print(f"[it] fixpoint-at-iteration histogram: "
          f"{dict(sorted(it_hist.items()))}  rows-with-amputation: "
          f"{rows_iterated}  decode-changed-on-rebreathe: {changed_meter}",
          flush=True)

if __name__ == "__main__":
    main()
