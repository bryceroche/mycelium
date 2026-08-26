"""iter_v25.py — THE CLOSED LOOP v2.5: THE REVOKE PORT AWAKE
(2026-08-25, word given). Feedback is INJECTION, not subtraction: sworn
slots (both-witness criteria, unchanged from v2a2) are fed to the
head's organ-2 revoke input — their committed mass DUMPS mid-breathing
(XARM=dump) and the trained release dynamics let the slot RE-BIND.
No mask cuts, no factor drops: the head answers the contradiction with
a new emission, and the new decode is judged whole. Revokes accumulate
across iterations until accept/fixpoint (cap 6). Checkpoint:
gsb227_rings (organ-2 trained via the XOUT_TR two-pass). The answer key
NEVER enters the loop.
BARS (pinned pre-fire): substrate precondition val within 0.02 of
0.5864; PRIMARY gold net > -20; GUARD wv+held lies <= 12; rights
reported explicitly (the minting question).
"""
import os, sys, json, glob
os.environ.setdefault("ALG_MINE_BREATHS", "1")
os.environ.setdefault("ALG_BREATH", "7")
os.environ.setdefault("ALG_NOTEBOOK", "1")
os.environ.setdefault("ALG_SIXWAVE", "1")
os.environ.setdefault("ALG_RINGS", "1")
os.environ.setdefault("ALG_XOUT", "1")
os.environ.setdefault("ALG_XARM", "dump")
os.environ.setdefault("NB_PERSLOT", "1")
os.environ.setdefault("ALG_CMT_REG", "1")
os.environ.setdefault("ATLAS_TABLE", "waist_patterns_sharp")
os.environ.setdefault("ATLAS_TRANS", "sharp_transitions")
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
from iter_a0 import chain_labels

_ = load_alg("test")
tok = Tokenizer.from_file(TOKENIZER_JSON)
K = ("pres", "ftype", "op", "islit", "dig", "args", "res", "query")
_zr = np.load('.cache/recognition_mouth.npz')
_MBANK = _zr['bank'].astype(np.float32)
_MCOEF = np.load('.cache/mouth_length_correction.npz')['coef'].astype(np.float32)

def mouth_reg(sts, msk):
    m = msk[:, :, None]
    v = (sts * m).sum(1) / np.maximum(m.sum(1), 1)
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    d = 1.0 - v @ _MBANK.T
    knn = np.sort(d, axis=1)[:, :8].mean(1)
    L = np.maximum(msk.sum(1), 1)
    return (knn - (_MCOEF[0] + _MCOEF[1] / L)).astype(np.float32)
MAX_IT = int(os.environ.get("ITER_MAX", "6"))
SMP = {"n_vars": 24, "m": 300}

def try_solve(facs, q):
    try:
        return solve_forced(facs, q, SMP)
    except Exception:
        return None

def main():
    cents, ckinds, trans = load_atlas()
    cycles = sorted(cents)
    p = build_params(0)
    sd = safe_load(f".cache/{os.environ.get('V25_CKPT', 'gsb227_sharpreg')}.safetensors")
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
    tallies = {t: [0, 0, 0] for t in ("gold", "wv", "held")}  # forced/right/lies
    it_hist = Counter(); repair_used = 0; repair_emitted = 0
    accept_clean = 0; refuse_guard = 0
    dis_stats = []          # validity tripwire: it-0 dissent fraction/row
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
        rg = Tensor(mouth_reg(sts, msk), dtype=dtypes.float)
        o0 = forward(p, ts, tk, se, reg=rg)
        onp0 = {k2: o0[k2].realize().numpy() for k2 in ("fat", "args", "res")}
        M = build_slot_masks(onp0, snt)
        RV = np.zeros((8, L_FAC), np.float32)          # the revoke vector
        amp = [set() for _ in range(8)]                # revoked (bookkeeping)
        n_solver_amp = [0] * 8
        frozen = [False] * 8
        rec = [None] * 8                     # final (facs, q)
        used_repair = [False] * 8
        for it in range(MAX_IT):
            o = forward(p, ts, tk, se, slot_mask=Tensor(M, dtype=dtypes.float),
                        revoke=Tensor(RV, dtype=dtypes.float), reg=rg)
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
                # INJECTION, not subtraction: revoked slots stay LIVE —
                # the head re-binds them under released mass; the decode
                # is judged whole
                live = [(f, present[fi] if fi < len(present) else None)
                        for fi, f in enumerate(facs)]
                lf = [f for f, _ in live]
                def _opg(f):
                    # SAME-LANGUAGE (the triad, enforced where it was
                    # violated): the sharp atlas speaks OP-grain; map the
                    # head's claim into it before comparing (chain_decode's
                    # gold mapping verbatim)
                    if f["ftype"] == "rel":
                        if f.get("op") == "mul" and len(set(f.get("args", []))) == 1:
                            return "sq"
                        return f.get("op", "rel")
                    if f["ftype"] == "macro":
                        return "opa" if f.get("name") == "OP_APPLY" else "fr"
                    if f["ftype"] == "frac":
                        return "fr"
                    return f["ftype"]
                dis = {j for f, j in live
                       if j is not None and lab.get(j) is not None
                       and lab[j] != _opg(f)}
                if it == 0 and live:
                    dis_stats.append(len(dis) / max(len(live), 1))
                rec[li] = (lf, q)
                a = try_solve(lf, q)
                to_amp = set()
                if a is not None:
                    if not dis:
                        frozen[li] = True; it_hist[it] += 1
                        accept_clean += 1
                        continue
                    to_amp = dis                       # v0 road
                else:
                    # solver-guided repair: LOO culprit localization
                    if n_solver_amp[li] >= 2:
                        frozen[li] = True; it_hist[it] += 1
                        refuse_guard += 1
                        continue
                    cands = []
                    for fi, (f, j) in enumerate(live):
                        if j is None: continue
                        # attempt 2 (V2_BOTH, default on): the culprit must
                        # be sworn by BOTH witnesses — solver (consistency:
                        # LOO restores forcedness) AND chain (correspondence:
                        # text-side dissent). Consistency alone was the lie
                        # machine (attempt 1: +10 gold lies, 21 repair-emits
                        # nearly all wrong).
                        if int(os.environ.get("V2_BOTH", "1")) and j not in dis:
                            continue
                        if try_solve([g for gi, (g, _) in enumerate(live)
                                      if gi != fi], q) is not None:
                            cands.append((j in dis, -onp["pres"][li, j], j))
                    if cands:
                        cands.sort(reverse=True)
                        to_amp = {cands[0][2]}
                        n_solver_amp[li] += 1
                        used_repair[li] = True
                    elif dis:
                        to_amp = dis                   # fall back to v0 road
                    else:
                        frozen[li] = True; it_hist[it] += 1
                        continue                        # refuse: no repair
                fresh = to_amp - amp[li]
                if not fresh:
                    frozen[li] = True; it_hist[it] += 1
                    continue
                amp[li] |= fresh
                for j in amp[li]:
                    RV[li, j] = 1.0                    # named contradiction
                new_any = True
            if not new_any:
                break
        for li in range(8):
            if not frozen[li] and rec[li] is not None:
                it_hist[MAX_IT] += 1
        for li, r in enumerate(sl):
            if rec[li] is None: continue
            if used_repair[li]: repair_used += 1
            lf, q = rec[li]
            a = try_solve(lf, q)
            T = tallies[r["tag"]]
            if a is not None:
                T[0] += 1
                if used_repair[li]: repair_emitted += 1
                if a == r["answer"]: T[1] += 1
                else: T[2] += 1
    for t, T in tallies.items():
        n = {"gold": 143, "wv": 20, "held": 20}[t]
        print(f"[v25 {t}] forced {T[0]}/{n} right {T[1]} lies {T[2]} "
              f"(net {T[1]-T[2]})", flush=True)
    print(f"[v25] fixpoint hist {dict(sorted(it_hist.items()))}  "
          f"repair-used {repair_used} repair-emitted {repair_emitted} "
          f"accept-clean {accept_clean} refuse-at-guard {refuse_guard}",
          flush=True)
    ds = np.array(dis_stats)
    print(f"[v25] TRIPWIRE it-0 dissent/present: median {np.median(ds):.2f} "
          f"mean {ds.mean():.2f} (VOID if median > 0.5)", flush=True)
    print("[v25] BARS: gold net > -20 (v0)  |  guard: wv+held lies <= 12",
          flush=True)

if __name__ == "__main__":
    main()
