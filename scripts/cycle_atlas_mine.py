"""cycle_atlas_mine.py — THE CYCLE-RESOLVED ATLAS (2026-09-01, word given).
Keyed (register, cycle, class): Welford centroids of SLOT states per
breath cycle per ftype-kind, mined from a banked stack artifact
(regime-tagged). The NL/trunk atlas stays cycle=NULL BY PHYSICS (input-
space, breath-invariant — outside time, like breath-0); the op/slot
atlas is the breathing one.
FREE INSTRUMENT: per-cycle mint<->wild centroid alignment — THE DIALECT
CREEP BY FLOOR. PINNED: divergence GROWS with cycle (the two-tap law
rendered as a curve).
Env: CA_CKPT (default sharp_bind14a).
"""
import os, sys, json, glob, sqlite3
os.environ.update({"DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "9",
                   "ALG_DUP": "1", "ALG_HW": "512", "ALG_WIDE": "1",
                   "ALG_BREATH": "7", "ALG_NOTEBOOK": "1", "ALG_SIXWAVE": "1",
                   "NB_PERSLOT": "1", "ALG_BINDBUS": "7", "ALG_BIND_D": "512",
                   "BIND_CODES": ".cache/bindbus_codes512.npz",
                   "ALG_BUSGARAGE": "2", "ALG_MINE_BREATHS": "1",
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
CKPT = os.environ.get("CA_CKPT", ".cache/sharp_bind14a.safetensors")
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
    return mint + golds


class W:
    def __init__(s, d): s.n = 0; s.mean = np.zeros(d, np.float64); s.m2 = np.zeros(d, np.float64)
    def add(s, x):
        s.n += 1; d = x - s.mean; s.mean += d / s.n; s.m2 += d * (x - s.mean)


def main():
    p = build_params(0)
    sd = safe_load(CKPT)
    for k in p:
        if k in sd:
            p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    acc = {}
    for s0 in range(0, len(rows_all), 8):
        sl = rows_all[s0:s0 + 8]
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
                cls = int(min(g["ftype"][i, j], 7))
                for c in range(min(K_B, len(br))):
                    key = (r['tag'], c, cls)
                    if key not in acc: acc[key] = W(br[c].shape[-1])
                    acc[key].add(br[c][i, j].astype(np.float64))
    con = sqlite3.connect('.cache/campaign.db')
    con.execute("""CREATE TABLE IF NOT EXISTS cycle_atlas
        (register TEXT, cycle INT, class INT, count INT,
         mean BLOB, m2 BLOB, ckpt TEXT,
         PRIMARY KEY (register, cycle, class, ckpt))""")
    for (tag, c, cls), w in acc.items():
        con.execute("INSERT OR REPLACE INTO cycle_atlas VALUES (?,?,?,?,?,?,?)",
                    (tag, c, cls, w.n,
                     w.mean.astype(np.float32).tobytes(),
                     w.m2.astype(np.float32).tobytes(), CKPT))
    con.commit()
    print(f"[cycle-atlas] {len(acc)} (register,cycle,class) centroids -> "
          f"campaign.db (ckpt={CKPT.split('/')[-1]})")
    # THE DIALECT CREEP BY FLOOR: mint<->wild alignment per cycle
    print("cycle  mean cos(mint, wild) over shared classes")
    for c in range(K_B):
        cos = []
        for cls in range(8):
            a = acc.get(('mint', c, cls)); b = acc.get(('gold', c, cls))
            if a is None or b is None or a.n < 20 or b.n < 20: continue
            va, vb = a.mean, b.mean
            cos.append(float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9)))
        if cos:
            print(f"  b{c}:  {np.mean(cos):.4f}  (n_classes={len(cos)})")


rows_all = rows()
main()
