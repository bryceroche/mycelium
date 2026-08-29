"""bind_amp_audit.py — THE AMPLITUDE AUDIT (2026-08-29, word given;
pre-registered in docs/rotational_bus.md S3): does the soft-pointer mixture's
modulus A_r = ||sum_k p_r(k) c_k|| carry information about correctness BEYOND
softmax entropy H(p_r)? The hypothesis: amplitude is GEOMETRY-AWARE
confidence (mass split across nearby codes interferes less than across
distant codes) where entropy is geometry-blind.

PASS RULE (pinned before measurement): conditional MI I(correct; A_r | H)
beats a within-H-decile permutation null at p < 0.05 on >= 2 of 4 roles on
mint. Wild reported alongside. Selective-prediction comparison (accuracy at
50% coverage, rank by A vs rank by -H) reported as the practical payoff.
GOODHART FENCE: this is an OBSERVATIONAL read on a banked artifact
(sharp_bind5r); amplitude never enters any loss.
"""
import os, sys, json, glob
os.environ.update({"DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "9",
                   "ALG_DUP": "1", "ALG_HW": "512", "ALG_WIDE": "1",
                   "ALG_BREATH": "7", "ALG_NOTEBOOK": "1", "ALG_SIXWAVE": "1",
                   "NB_PERSLOT": "1", "ALG_BINDBUS": "5", "ALG_BIND_D": "256",
                   "BIND_CODES": os.environ.get("BIND_CODES",
                                                ".cache/bindbus_codes256.npz"),
                   "ALG_TEST": ".cache/algebra_nl_bigtest.jsonl",
                   "ALG_TEST_NAME": "bigtest"})
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
from phase1_algebra_head import (build_params, forward, T_ALG, TOKENIZER_JSON,
                                 sent_indices, load_alg, build_slot_masks,
                                 L_FAC, build_gold)
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

_ = load_alg("test")
tok = Tokenizer.from_file(TOKENIZER_JSON)
bz = np.load(os.environ["BIND_CODES"])
CB = bz["CB"].astype(np.float32)
ROLES = ("arg1", "arg2", "res", "op")


def rows_and_truth():
    mint = [json.loads(l) for l in open('.cache/algebra_nl_test.jsonl')][:200]
    for r in mint:
        r['tag'] = 'mint'; r['original'] = r.get('text') or r.get('original')
    # custody law: src_idx is BOOK-LOCAL — key by TEXT identity; skips
    # apply book12-locally (the fixture fix, audit 2026-08-30)
    byid = {}
    for f in sorted(glob.glob('.cache/book*_t*_batch*.jsonl')):
        for l in open(f):
            r = json.loads(l); byid[r["original"].strip()] = r
    sk = set(json.load(open('.cache/book12_anchor_skips.json')))
    for l in open('.cache/book12_anchor_batch1.jsonl'):
        r = json.loads(l)
        if r["src_idx"] in sk: continue
        byid[r["original"].strip()] = r
    golds = [dict(v, tag='gold') for k, v in sorted(byid.items())]
    for r in mint + golds:
        r.setdefault('n_vars', 24); r.setdefault('m', 300)
        r.setdefault('query_var', r.get('query', 0))
        r.setdefault('decisions', 0); r.setdefault('mentions', {})
        r.setdefault('text', r['original'])
    return mint + golds


def cond_mi_and_null(A, C, H, n_perm=1000, seed=0):
    """MI(correct; A>median | H-decile) in bits, plus permutation null
    (A shuffled WITHIN deciles), returning (mi, p_value)."""
    rng = np.random.default_rng(seed)
    dec = np.searchsorted(np.quantile(H, np.linspace(0, 1, 11)[1:-1]), H)

    def mi_of(a):
        tot = 0.0
        for d in np.unique(dec):
            m = dec == d
            if m.sum() < 8:
                continue
            ab = a[m] > np.median(a[m])
            for av in (0, 1):
                for cv in (0, 1):
                    pj = ((ab == av) & (C[m] == cv)).mean()
                    if pj > 0:
                        tot += m.mean() * pj * np.log2(
                            pj / (((ab == av).mean()) * ((C[m] == cv).mean()) + 1e-12))
        return tot

    mi = mi_of(A)
    null = np.empty(n_perm)
    Ap = A.copy()
    for i in range(n_perm):
        for d in np.unique(dec):
            m = dec == d
            Ap[m] = rng.permutation(Ap[m])
        null[i] = mi_of(Ap)
    return mi, float((null >= mi).mean())


def selective(acc_key, score, C, coverage=0.5):
    k = int(len(C) * coverage)
    idx = np.argsort(-score)[:k]
    return C[idx].mean()


def main():
    p = build_params(0)
    sd = safe_load(os.environ.get('BR_CKPT', '.cache/sharp_bind5r.safetensors'))
    assert set(sd.keys()) == set(p.keys())
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    rows = rows_and_truth()
    data = {t: {r: {"A": [], "H": [], "C": []} for r in ROLES}
            for t in ("mint", "gold")}
    for s0 in range(0, len(rows), 8):
        sl = rows[s0:s0 + 8]
        ids = np.zeros((8, T_ALG), np.int32)
        msk = np.zeros((8, T_ALG), np.float32)
        snt = np.zeros((8, T_ALG), np.int32)
        offs = []
        for i, r in enumerate(sl):
            e = tok.encode(r['original'])
            Ln = min(len(e.ids), T_ALG)
            ids[i, :Ln] = e.ids[:Ln]; msk[i, :Ln] = 1.0
            snt[i] = sent_indices(r['original'], list(e.offsets), msk[i])
            offs.append(list(e.offsets))
        g = build_gold(sl, offs)
        sts = np.asarray(recompute_states(ids)).astype(np.float32)
        o0 = forward(p, Tensor(sts, dtype=dtypes.float),
                     Tensor(msk, dtype=dtypes.float),
                     Tensor(snt.astype(np.int32), dtype=dtypes.int))
        onp0 = {k2: o0[k2].realize().numpy() for k2 in ("fat", "args", "res")}
        mk = build_slot_masks(onp0, snt)
        o = forward(p, Tensor(sts, dtype=dtypes.float),
                    Tensor(msk, dtype=dtypes.float),
                    Tensor(snt.astype(np.int32), dtype=dtypes.int),
                    slot_mask=Tensor(mk, dtype=dtypes.float))
        LG = o["bind_lg"].realize().numpy()          # (8, L_FAC, 4, 32)
        for i, r in enumerate(sl):
            for j in range(L_FAC):
                if g["presence"][i, j] <= 0:
                    continue
                aidx = np.where(g["args"][i, j] > 0)[0]
                if len(aidx) == 0:
                    a1 = a2 = int(g["res"][i, j])
                elif len(aidx) == 1:
                    a1 = a2 = int(aidx[0])
                else:
                    a1, a2 = int(aidx[0]), int(aidx[1])
                truth = (a1, a2, int(g["res"][i, j]),
                         24 + min(int(g["ftype"][i, j]), 7))
                for ri, rn in enumerate(ROLES):
                    lg = LG[i, j, ri]
                    pr = np.exp(lg - lg.max()); pr /= pr.sum()
                    v = pr @ CB
                    d = data[r['tag']][rn]
                    d["A"].append(float(np.linalg.norm(v)))
                    d["H"].append(float(-(pr * np.log2(pr + 1e-12)).sum()))
                    d["C"].append(int(int(pr.argmax()) == truth[ri]))

    print("role  tag   n     acc    MI(C;A|H)  perm-p   sel@50: A     -H")
    passes = 0
    for rn in ROLES:
        for t in ("mint", "gold"):
            d = data[t][rn]
            A = np.array(d["A"]); H = np.array(d["H"])
            C = np.array(d["C"])
            mi, pv = cond_mi_and_null(A, C, H)
            sA = selective("A", A, C)
            sH = selective("H", -H, C)
            flag = ""
            if t == "mint" and pv < 0.05:
                passes += 1; flag = "  *"
            print(f"{rn:5s} {t:5s} {len(C):5d}  {C.mean():.3f}   "
                  f"{mi:+.4f}    {pv:.3f}   {sA:.3f}  {sH:.3f}{flag}")
    print(f"[verdict] mint roles passing perm test: {passes}/4 "
          f"(pinned bar: >=2) -> {'PASS' if passes >= 2 else 'FAIL'}")


if __name__ == "__main__":
    main()
