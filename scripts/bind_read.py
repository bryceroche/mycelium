"""bind_read.py — THE BUS'S FIRST READ (2026-08-28): unbind each slot's
emitted vector role-by-role (counter-rotation), cleanup against the
codebook, recover (arg1, arg2, res, op); grade vs gold wiring on
mint-val rows AND the 143 wild golds; baseline = the pointer heads'
args/res on the same rows. BARS (pinned): mint-val role recovery >= 90%
(the head can emit bindings); wild recovery reported vs pointers (parity
= a second, mechanism-decorrelated wiring witness for the door).
"""
import os, sys, json, glob
os.environ.update({"DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "9",
                   "ALG_DUP": "1", "ALG_HW": "512", "ALG_WIDE": "1",
                   "ALG_BREATH": "7", "ALG_NOTEBOOK": "1", "ALG_SIXWAVE": "1",
                   "NB_PERSLOT": "1", "ALG_BINDBUS": os.environ.get("BR_V", "1"),
                   "ALG_BIND_D": os.environ.get("BR_D", "128"),
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
# THE BUS'S FORMAL LANGUAGE IS COMPLEX (2026-08-28, word given):
# C^64 is the truth, R^128 the implementation. Interleaved-real (P,2)
# lifts to complex64; unbinding = multiplication by the conjugate role
# phasor e^{-i theta}; cleanup similarity = Re<z, c> (the exact
# isomorph of the R^128 cosine numerator).
bz = np.load(os.environ.get('BIND_CODES', '.cache/bindbus_codes.npz'))
CB = bz['CB']; P = CB.shape[1] // 2
def lift(v):
    v2 = v.reshape(*v.shape[:-1], P, 2)
    return (v2[..., 0] + 1j * v2[..., 1]).astype(np.complex64)
CBc = lift(CB)
ROLE = {r: np.exp(-1j * bz[f'theta_{r}']).astype(np.complex64)
        for r in ('arg1', 'arg2', 'res', 'op')}
def unbind(v, role):
    return lift(v) * ROLE[role]
def cleanup_c(zc):
    zn = zc / (np.sqrt((np.abs(zc) ** 2).sum(-1, keepdims=True)) + 1e-9)
    return (zn @ np.conj(CBc).T).real.argmax(-1)
# isomorphism self-check: complex path == real path on random probes
_r = np.random.default_rng(0)
_v = _r.standard_normal(CB.shape[1]).astype(np.float32)
for _role in ROLE:
    _th = bz[f'theta_{_role}']
    _v2 = _v.reshape(P, 2); _c, _s = np.cos(-_th), np.sin(-_th)
    _real = np.stack([_c * _v2[:, 0] - _s * _v2[:, 1],
                      _s * _v2[:, 0] + _c * _v2[:, 1]], -1).reshape(-1)
    _cplx = unbind(_v, _role)
    assert np.allclose(lift(_real), _cplx, atol=1e-4), "C/R isomorphism broken"

def main():
    p = build_params(0)
    sd = safe_load(os.environ.get('BR_CKPT', '.cache/sharp_bind.safetensors'))
    assert set(sd.keys()) == set(p.keys())
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    # rows: mint-val (200 from test23) + the 143 golds
    mint = [json.loads(l) for l in open('.cache/algebra_nl_test.jsonl')][:200]
    for r in mint: r['tag'] = 'mint'; r['original'] = r.get('text') or r.get('original')
    # custody law: src_idx is BOOK-LOCAL — the pool keys by TEXT identity
    # (audit 2026-08-30: bare-src_idx merge overwrote 74 tranche rows and
    # the skip list hit 34 wrong ones); skips apply book12-locally.
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
    rows = mint + golds
    from tokenizers import Tokenizer as _T
    stats = {t: {r2: [0, 0] for r2 in ('arg1', 'arg2', 'res', 'op')}
             for t in ('mint', 'gold')}
    ptr = {t: [0, 0, 0, 0] for t in ('mint', 'gold')}   # a_ok, a_n, r_ok, r_n
    for s0 in range(0, len(rows), 8):
        sl = rows[s0:s0 + 8]
        ids = np.zeros((8, T_ALG), np.int32); msk = np.zeros((8, T_ALG), np.float32)
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
        ts = Tensor(sts, dtype=dtypes.float)
        tk = Tensor(msk, dtype=dtypes.float)
        se = Tensor(snt.astype(np.int32), dtype=dtypes.int)
        o0 = forward(p, ts, tk, se)
        onp0 = {k2: o0[k2].realize().numpy() for k2 in ("fat", "args", "res")}
        mk = build_slot_masks(onp0, snt)
        o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float))
        Bv = o["bind"].realize().numpy()
        Ap = o["args"].realize().numpy(); Rp = o["res"].realize().numpy()
        Dp = o["dup"].realize().numpy() if "dup" in o else None
        for i, r in enumerate(sl):
            for j in range(L_FAC):
                if g["presence"][i, j] <= 0: continue
                aidx = np.where(g["args"][i, j] > 0)[0]
                if len(aidx) == 0: a1 = a2 = int(g["res"][i, j])
                elif len(aidx) == 1: a1 = a2 = int(aidx[0])
                else: a1, a2 = int(aidx[0]), int(aidx[1])
                truth = {'arg1': a1, 'arg2': a2, 'res': int(g["res"][i, j]),
                         'op': 24 + min(int(g["ftype"][i, j]), 7)}
                for role in truth:
                    rec = int(cleanup_c(unbind(Bv[i, j], role)))
                    st = stats[r['tag']][role]
                    st[1] += 1
                    if rec == truth[role]: st[0] += 1
                # pointer-args baseline, dup-fair AND slot-scoped (audit
                # 2026-08-30 + scope fix): scored ONLY on arg-bearing
                # slots (given slots carry args:=res by bus convention;
                # the pointer head never answers them — including them
                # craters the baseline unfairly). Genuine-dup slots use
                # decode's own rule (dup>0 -> (argmax, argmax)); two-arg
                # slots use sorted-top-2 multiset equality. NOTE: bus arg
                # stats still include given slots (its design covers them).
                if len(aidx) >= 1:
                    if len(aidx) == 1:
                        _ok = (Dp is not None and float(Dp[i, j]) > 0
                               and int(np.argmax(Ap[i, j])) == a1)
                    else:
                        dec = tuple(sorted(int(x) for x in np.argsort(-Ap[i, j])[:2]))
                        _ok = dec == tuple(sorted((a1, a2)))
                    ptr[r['tag']][1] += 1
                    if _ok: ptr[r['tag']][0] += 1
                ptr[r['tag']][3] += 1
                if int(Rp[i, j].argmax()) == truth['res']: ptr[r['tag']][2] += 1
    for t in ('mint', 'gold'):
        line = " ".join(f"{ro}:{stats[t][ro][0]/max(stats[t][ro][1],1):.3f}"
                        for ro in ('arg1', 'arg2', 'res', 'op'))
        a_ok, a_n, r_ok, r_n = ptr[t]
        print(f"[bind {t}] BUS recovery {line}  (n={stats[t]['res'][1]})"
              f"  |  POINTERS args {a_ok/max(a_n,1):.3f} res {r_ok/max(r_n,1):.3f}",
              flush=True)

if __name__ == "__main__":
    main()
