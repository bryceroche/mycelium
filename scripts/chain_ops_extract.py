"""chain_ops_extract.py — extract per-row ATLAS-CHAIN op multisets for
the 180-row fixture (the cross-mechanism gate's chain wave). Runs on
the CURRENT op atlas (fire after the fuller mine). Output:
.cache/chain_ops.json [{tag, key, ops}].
"""
import os, sys, json, glob
os.environ.setdefault("ALG_MINE_BREATHS", "1")
os.environ.setdefault("ALG_BREATH", "7")
os.environ.setdefault("ALG_NOTEBOOK", "1")
os.environ.setdefault("ALG_SIXWAVE", "1")
os.environ.setdefault("ATLAS_TABLE", "waist_patterns_op")
os.environ.setdefault("ATLAS_TRANS", "op_transitions")
os.environ.update({"DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "9",
                   "ALG_DUP": "1", "ALG_HW": "512", "ALG_WIDE": "1",
                   "ALG_TEST": ".cache/algebra_nl_bigtest.jsonl",
                   "ALG_TEST_NAME": "bigtest"})
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
from phase1_algebra_head import (build_params, forward, T_ALG,
                                 TOKENIZER_JSON, sent_indices, load_alg,
                                 build_slot_masks, L_FAC)
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
from beacon_closing_arm import recompute_states
from chain_decode import load_atlas

_ = load_alg("test")
tok = Tokenizer.from_file(TOKENIZER_JSON)

def main():
    cents, ckinds, trans = load_atlas()
    cycles = sorted(cents)
    p = build_params(0)
    sd = safe_load('.cache/gsb227_real.safetensors')
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    aro = open('scripts/audition_read_one.py').read()
    ns = {"json": json, "glob": glob, "np": np}
    exec(aro[aro.index("def fixtures():"):aro.index("rows = fixtures()")], ns)
    rows = ns["fixtures"]()
    out = []
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
        onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
        mk = build_slot_masks(onp0, snt)
        o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float))
        Bst = [b.realize().numpy() for b in o["breaths_all"]]
        pres = o["pres"].realize().numpy()
        fat = o["fat"].realize().numpy()
        import re as _re
        _NUM = _re.compile(r"(?<![\d.])(\d+)(?![\d.])")
        for li, r in enumerate(sl):
            e2 = tok.encode(r["original"])
            offs2 = list(e2.offsets)
            numtoks = []
            for m in _NUM.finditer(r["original"]):
                numtoks.append([ti for ti, (a2, b2) in enumerate(offs2)
                                if a2 < m.end() and b2 > m.start()])
            labs = []; slotinfo = []
            for j in range(L_FAC):
                if pres[li, j] <= 0.5: continue
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
                lab = None
                for cid in reversed(chain):
                    if cid in ckinds and ckinds[cid]:
                        lab = max(ckinds[cid], key=ckinds[cid].get); break
                if lab:
                    labs.append(lab)
                    fv = fat[li, j]
                    aff = [float(fv[ts2].sum()) if ts2 else 0.0
                           for ts2 in numtoks]
                    slotinfo.append({"op": lab, "aff": aff,
                                     "fv": [round(float(x), 4)
                                            for x in fv[:len(offs2)]]})
            # op-op overlaps (flow adjacency)
            ovl = []
            for a2 in range(len(slotinfo)):
                row_o = []
                for b2 in range(len(slotinfo)):
                    fa = np.array(slotinfo[a2]["fv"]); fb = np.array(slotinfo[b2]["fv"])
                    d = float(np.minimum(fa, fb).sum())
                    row_o.append(round(d, 4))
                ovl.append(row_o)
            out.append({"tag": r["tag"], "key": r["answer"],
                        "ops": sorted(l for l in labs if l != "given"),
                        "slots": [{"op": si["op"], "aff": si["aff"]}
                                  for si in slotinfo],
                        "ovl": ovl})
    json.dump(out, open('.cache/chain_ops.json', 'w'))
    print(f"[cx] chain multisets banked: {len(out)} rows", flush=True)

if __name__ == "__main__":
    main()
