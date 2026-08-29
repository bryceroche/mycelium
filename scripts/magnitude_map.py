"""magnitude_map.py — THE MAGNITUDE MAP (2026-08-30; registered w/ pinned
predictions). Read r where only theta has been read: per-slot wire L2 norm
on the incumbent monolith line, against (a) per-role recovery correctness,
(b) register (mint/wild), (c) slot kind. PINNED: ||wire|| correlates with
res-recovery correctness beyond chance; mint louder than wild
(consolidation displaces magnitude). Observational; Goodhart-fenced.
Env: MM_CKPT, BR_V, BR_D, BIND_CODES (as bind_read).
"""
import os, sys, json, glob
os.environ.update({"DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "9",
                   "ALG_DUP": "1", "ALG_HW": "512", "ALG_WIDE": "1",
                   "ALG_BREATH": "7", "ALG_NOTEBOOK": "1", "ALG_SIXWAVE": "1",
                   "NB_PERSLOT": "1",
                   "ALG_BINDBUS": os.environ.get("BR_V", "3"),
                   "ALG_BIND_D": os.environ.get("BR_D", "256"),
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
bz = np.load(os.environ.get('BIND_CODES', '.cache/bindbus_codes.npz'))
CB = bz['CB']; P = CB.shape[1] // 2
CBc = (CB.reshape(32, P, 2)[..., 0] + 1j * CB.reshape(32, P, 2)[..., 1]).astype(np.complex64)
ROLE = {r: np.exp(-1j * bz[f'theta_{r}']).astype(np.complex64)
        for r in ('arg1', 'arg2', 'res', 'op')}


def cleanup(v, role):
    z = (v.reshape(P, 2)[:, 0] + 1j * v.reshape(P, 2)[:, 1]) * ROLE[role]
    z = z / (np.sqrt((np.abs(z) ** 2).sum()) + 1e-9)
    return int((z @ np.conj(CBc).T).real.argmax())


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


def pbr(x, y):
    """point-biserial corr of continuous x vs binary y."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if y.std() == 0 or x.std() == 0: return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main():
    p = build_params(0)
    sd = safe_load(os.environ.get('MM_CKPT', '.cache/sharp_bind7c.safetensors'))
    assert set(sd.keys()) == set(p.keys())
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    data = {t: {"nrm": [], "cres": [], "ft": []} for t in ("mint", "gold")}
    rws = rows()
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
        Bv = o["bind"].realize().numpy()
        for i, r in enumerate(sl):
            for j in range(L_FAC):
                if g["presence"][i, j] <= 0: continue
                d = data[r['tag']]
                d["nrm"].append(float(np.linalg.norm(Bv[i, j])))
                d["cres"].append(int(cleanup(Bv[i, j], 'res')
                                     == int(g["res"][i, j])))
                d["ft"].append(min(int(g["ftype"][i, j]), 7))
    print(f"[magnitude map] ckpt={os.environ.get('MM_CKPT')}")
    for t in ("mint", "gold"):
        d = data[t]
        nrm = np.array(d["nrm"]); cr = np.array(d["cres"])
        print(f"  {t:5s} n={len(nrm)}  ||w|| mean {nrm.mean():.3f} sd "
              f"{nrm.std():.3f}  corr(||w||, res-correct) {pbr(nrm, cr):+.3f}"
              f"  (mean ||w|| right {nrm[cr == 1].mean():.3f} vs wrong "
              f"{nrm[cr == 0].mean():.3f})", flush=True)
    mm = np.array(data['mint']['nrm']).mean()
    gm = np.array(data['gold']['nrm']).mean()
    print(f"  register displacement: mint {mm:.3f} vs wild {gm:.3f} "
          f"({'mint louder' if mm > gm else 'WILD louder'}, "
          f"ratio {mm / gm:.3f})", flush=True)
    # the selective curve: res-recovery precision at loudness coverage
    for t in ("mint", "gold"):
        nrm = np.array(data[t]["nrm"]); cr = np.array(data[t]["cres"])
        base = cr.mean()
        line = " ".join(
            f"@{int(c*100)}%:{cr[np.argsort(-nrm)[:max(int(len(cr)*c),1)]].mean():.3f}"
            for c in (0.75, 0.5, 0.25, 0.1))
        print(f"  selective res precision {t:5s} base {base:.3f}  {line}",
              flush=True)


if __name__ == "__main__":
    main()
