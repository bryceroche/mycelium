"""lowering_probe.py — THE LOWERING PROBE (2026-08-30, word given).
Is the breath loop an MLIR-style progressive lowering pipeline?

Per-breath states (ALG_MINE_BREATHS) on mint-val + the lawful wild golds;
ridge probes per breath at three grains:
  COARSE — ftype (8-way, present slots)
  MID    — res var (24-way, present slots)
  FINE   — leading digit (10-way, given slots)
Probes FIT on mint rows [0:150], EVAL on mint [150:200] + wild golds
(transfer). PINNED PREDICTIONS: (1) peak-breath ordering monotone in grain
(coarse saturates early, fine climbs late); (2) wild transfer decays DOWN
the ladder (coarse ports, fine doesn't — the two-tap law as IR
portability). Observational; nothing here enters any loss.
"""
import os, sys, json, glob
os.environ.update({"DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "9",
                   "ALG_DUP": "1", "ALG_HW": "512", "ALG_WIDE": "1",
                   "ALG_BREATH": "7", "ALG_NOTEBOOK": "1", "ALG_SIXWAVE": "1",
                   "NB_PERSLOT": "1", "ALG_BINDBUS": "7", "ALG_BIND_D": "512",
                   "BIND_CODES": os.environ.get("BIND_CODES",
                                                ".cache/bindbus_codes512.npz"),
                   "ALG_MINE_BREATHS": "1",
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
K_B = 7


def rows():
    mint = [json.loads(l) for l in open('.cache/algebra_nl_test.jsonl')][:200]
    for r in mint:
        r['tag'] = 'mint'; r['original'] = r.get('text') or r.get('original')
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
    return mint, golds


def mine(p, rws):
    """-> per-breath slot-state matrix + labels, flattened over rows/slots."""
    S = {k: [] for k in range(K_B)}
    LB = {"ftype": [], "res": [], "digit": [], "isgiven": [], "pres": []}
    for s0 in range(0, len(rws), 8):
        sl = rws[s0:s0 + 8]
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
        st = np.asarray(recompute_states(ids)).astype(np.float32)
        ts = Tensor(st, dtype=dtypes.float)
        tk = Tensor(msk, dtype=dtypes.float)
        se = Tensor(snt.astype(np.int32), dtype=dtypes.int)
        o0 = forward(p, ts, tk, se)
        onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
        mk = build_slot_masks(onp0, snt)
        o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float))
        br = [b.realize().numpy() for b in o["breaths_all"]]
        for i, r in enumerate(sl):
            for j in range(L_FAC):
                if g["presence"][i, j] <= 0: continue
                for k in range(min(K_B, len(br))):
                    S[k].append(br[k][i, j])
                ft = min(int(g["ftype"][i, j]), 7)
                LB["ftype"].append(ft)
                LB["res"].append(int(g["res"][i, j]))
                LB["isgiven"].append(1 if ft == 1 else 0)
                LB["digit"].append(int(g["digits"][i, j][0])
                                   if "digits" in g else 0)
    return {k: np.stack(v) for k, v in S.items()}, \
           {k: np.array(v) for k, v in LB.items()}


def ridge_fit_eval(Xtr, ytr, Xev, yev, ncls, lam=10.0):
    Y = np.eye(ncls, dtype=np.float32)[ytr]
    Xt = np.concatenate([Xtr, np.ones((len(Xtr), 1), np.float32)], 1)
    Xe = np.concatenate([Xev, np.ones((len(Xev), 1), np.float32)], 1)
    W = np.linalg.solve(Xt.T @ Xt + lam * np.eye(Xt.shape[1], dtype=np.float32),
                        Xt.T @ Y)
    return float((np.argmax(Xe @ W, -1) == yev).mean())


def main():
    p = build_params(0)
    sd = safe_load(os.environ.get("LP_CKPT", ".cache/sharp_bind10n.safetensors"))
    assert set(sd.keys()) == set(p.keys())
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    mint, golds = rows()
    Smt, Lmt = mine(p, mint)
    Swd, Lwd = mine(p, golds)
    # split mint slots by originating row: first 150 rows' slots = fit
    # (approximate by slot order — rows are processed in order)
    ntr = int(len(Lmt["ftype"]) * 0.75)
    print(f"[probe] mint slots {len(Lmt['ftype'])} (fit {ntr}), "
          f"wild slots {len(Lwd['ftype'])}")
    GRAINS = (("coarse/ftype", "ftype", None, 8),
              ("mid/res", "res", None, 24),
              ("fine/digit", "digit", "isgiven", 10))
    print(f"{'grain':13s} " + " ".join(f"b{k}" for k in range(K_B))
          + "   | wild: " + " ".join(f"b{k}" for k in range(K_B)))
    peaks = {}
    for name, key, cond, ncls in GRAINS:
        mt_mask = np.ones(len(Lmt[key]), bool) if cond is None \
            else Lmt[cond] == 1
        wd_mask = np.ones(len(Lwd[key]), bool) if cond is None \
            else Lwd[cond] == 1
        accs, waccs = [], []
        for k in range(K_B):
            Xa, ya = Smt[k][mt_mask], Lmt[key][mt_mask]
            cut = int(len(ya) * 0.75)
            acc = ridge_fit_eval(Xa[:cut], ya[:cut], Xa[cut:], ya[cut:], ncls)
            wacc = ridge_fit_eval(Xa[:cut], ya[:cut],
                                  Swd[k][wd_mask], Lwd[key][wd_mask], ncls)
            accs.append(acc); waccs.append(wacc)
        peaks[name] = int(np.argmax(accs))
        print(f"{name:13s} " + " ".join(f"{a:.2f}" for a in accs)
              + "   |       " + " ".join(f"{a:.2f}" for a in waccs)
              + f"   peak b{peaks[name]}", flush=True)
    order = [peaks[n] for n, *_ in GRAINS]
    mono = order[0] <= order[1] <= order[2]
    print(f"[verdict] peak-breath ordering coarse<=mid<=fine: {order} -> "
          f"{'MONOTONE (lowering CONFIRMED)' if mono else 'NOT monotone (refuted at this grain)'}")


if __name__ == "__main__":
    main()
