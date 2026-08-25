"""iter_v2p.py — THE CLOSED LOOP v2 ATTEMPT 3: PAROLE (2026-08-25;
the covenant's third try). Amputation-only feedback can REMOVE a bad
slot but never FIX one (rights pinned 4-5 across v0/v2a1/v2a2). Parole:
an offending slot is quarantined for ONE re-breathe (workspace cleaned,
others re-read), then its mask column is RESTORED — one chance to
re-offer its evidence in the corrected context; re-offend on parole ->
permanent cut. Offense criteria = attempt 2 (both-witness). Bars unbent:
gold net > -20; wv+held lies <= 12. Base design (v2): Checker hierarchy per
decode: (a) forced solve + no chain dissent -> ACCEPT; (b) forced solve
+ chain dissent -> v0 amputation (the solver is blind to wrong-but-
solvable graphs; the chain organ is not); (c) NO forced solve -> the
solver LOCALIZES the culprit by leave-one-out (the factor whose removal
restores a forced solve), preference: chain-dissenting > lowest
presence. Amputations cut mask columns; re-breathe K=7; fixpoint/cap 5.
THE LIE-MACHINE GUARD (pinned): max 2 solver-amputations per row —
amputating to ANY forced solve births confident wrongness; past 2 the
row REFUSES. The answer key NEVER enters the loop (grades at exit).
BARS (pinned pre-read, vs banked v0 on identical fixtures): PRIMARY
gold net > -20; GUARD never-seen lies (wv+held) <= 12 (v0: 5+7).
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
from iter_a0 import chain_labels

_ = load_alg("test")
tok = Tokenizer.from_file(TOKENIZER_JSON)
K = ("pres", "ftype", "op", "islit", "dig", "args", "res", "query")
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
    tallies = {t: [0, 0, 0] for t in ("gold", "wv", "held")}  # forced/right/lies
    it_hist = Counter(); repair_used = 0; repair_emitted = 0
    accept_clean = 0; refuse_guard = 0
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
        M = M0.copy()
        quar = [set() for _ in range(8)]
        paroled = [set() for _ in range(8)]
        perm = [set() for _ in range(8)]
        n_solver_amp = [0] * 8
        frozen = [False] * 8
        rec = [None] * 8                     # final (facs, q)
        used_repair = [False] * 8
        for it in range(MAX_IT):
            for li in range(8):
                if frozen[li]: continue
                M[li] = M0[li].copy()
                for j in (quar[li] | perm[li]):
                    M[li, :, j] = 0.0; M[li, j, :] = 0.0; M[li, j, j] = 1.0
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
                # factors surviving prior amputations, with slot ids
                cuts = quar[li] | perm[li]
                live = [(f, present[fi] if fi < len(present) else None)
                        for fi, f in enumerate(facs)
                        if (fi >= len(present) or present[fi] not in cuts)]
                lf = [f for f, _ in live]
                dis = {j for f, j in live
                       if j is not None and lab.get(j) is not None
                       and lab[j] != f["ftype"]}
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
                # parole bookkeeping: offenders on parole -> permanent;
                # first offenders -> one-iteration quarantine; current
                # quarantine is RELEASED to parole (restored next breathe)
                new_quar = set(); promoted = set()
                for j in to_amp:
                    if j in paroled[li]: promoted.add(j)
                    else: new_quar.add(j)
                perm[li] |= promoted
                if not to_amp and not quar[li]:
                    frozen[li] = True; it_hist[it] += 1
                    continue
                paroled[li] |= quar[li]
                quar[li] = new_quar - perm[li]
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
        print(f"[v2 {t}] forced {T[0]}/{n} right {T[1]} lies {T[2]} "
              f"(net {T[1]-T[2]})", flush=True)
    print(f"[v2] fixpoint hist {dict(sorted(it_hist.items()))}  "
          f"repair-used {repair_used} repair-emitted {repair_emitted} "
          f"accept-clean {accept_clean} refuse-at-guard {refuse_guard}",
          flush=True)
    print("[v2] BARS: gold net > -20 (v0)  |  guard: wv+held lies <= 12",
          flush=True)

if __name__ == "__main__":
    main()
