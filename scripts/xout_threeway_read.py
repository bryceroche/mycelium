"""xout_threeway_read.py — THE FIRE'S CENTER (2026-08-05; word given;
bars pinned in ledger). Per arm: pass1 (revoke=None) finds wrong
bindings; pass2 forces revocation of exactly those slots under the
arm's dynamics; each revoked slot classifies SAME-WRONG / NEW-RIGHT /
NEW-WRONG (decode delta = ftype+res argmax pair). RIDERS: ctl gets
forced revocation under dump dynamics = the UNTRAINED re-bind baseline
(arms demonstrate learning only if they beat it); rates report SPLIT BY
FILLER GAP (clean gap=2 vs filler gap>=3) BEFORE the aggregate."""
import os, sys, json
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_HW", "512"); os.environ.setdefault("ALG_DUP", "1")
os.environ.setdefault("ALG_WIDE", "1")
os.environ["ALG_BREATH"] = "3"; os.environ["ALG_RINGS"] = "1"
os.environ.setdefault("ALG_TEST", ".cache/algebra_nl_bigtest.jsonl")
os.environ.setdefault("ALG_TEST_NAME", "bigtest")
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from phase1_algebra_head import (build_params, forward, load_alg, L_FAC,
                                 build_slot_masks)
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

samples, states, tokmask, gold, sent = load_alg("test")
n = states.shape[0]

def nsent(t):
    c, i = 1, t.find(". ")
    while i != -1:
        c += 1; i = t.find(". ", i + 1)
    return c

gaps = np.array([nsent(s["text"]) - len(s["factors"]) for s in samples])

def load(ck):
    p = build_params(0); sd = safe_load(ck)
    assert set(sd.keys()) == set(p.keys()), f"key mismatch {ck}"
    for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    return p

ARMS = [("ctl", ".cache/g25_xout_ctl.safetensors", "dump"),
        ("dump", ".cache/g25_xout_dump.safetensors", "dump"),
        ("graded", ".cache/g25_xout_graded.safetensors", "graded"),
        ("elastic", ".cache/g25_xout_elastic.safetensors", "elastic")]
res = {}
for name, ck, dyn in ARMS:
    os.environ["ALG_XOUT"] = "1"; os.environ["ALG_XARM"] = dyn
    p = load(ck)
    cnt = {c: {"same_wrong": 0, "new_right": 0, "new_wrong": 0}
           for c in ("clean", "filler")}
    for s0 in range(0, n, 8):
        sl = np.arange(s0, min(s0 + 8, n))
        pad = 8 - len(sl)
        slp = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        tr = Tensor(states[slp].astype(np.float32), dtype=dtypes.float)
        tk = Tensor(tokmask[slp].astype(np.float32), dtype=dtypes.float)
        se = Tensor(sent[slp].astype(np.int32), dtype=dtypes.int)
        o0 = forward(p, tr, tk, se)
        onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
        mk = Tensor(build_slot_masks(onp0, sent[slp]), dtype=dtypes.float)
        o1 = forward(p, tr, tk, se, slot_mask=mk)
        ft1 = o1["ftype"].realize().numpy().argmax(-1)
        rs1 = o1["res"].realize().numpy().argmax(-1)
        gft = gold["ftype"][slp]; grs = gold["res"][slp]
        prs = gold["presence"][slp]
        ok1 = (ft1 == gft) & (rs1 == grs)
        rv = (prs * (1.0 - ok1.astype(np.float32))).astype(np.float32)
        o2 = forward(p, tr, tk, se, slot_mask=mk,
                     revoke=Tensor(rv, dtype=dtypes.float))
        ft2 = o2["ftype"].realize().numpy().argmax(-1)
        rs2 = o2["res"].realize().numpy().argmax(-1)
        ok2 = (ft2 == gft) & (rs2 == grs)
        for bi, ri in enumerate(sl):
            cat = "clean" if gaps[ri] == 2 else "filler"
            for j in range(L_FAC):
                if rv[bi, j] <= 0: continue
                if ft2[bi, j] == ft1[bi, j] and rs2[bi, j] == rs1[bi, j]:
                    cnt[cat]["same_wrong"] += 1
                elif ok2[bi, j]:
                    cnt[cat]["new_right"] += 1
                else:
                    cnt[cat]["new_wrong"] += 1
        if s0 % 400 == 0: print(f"[{name}] {s0}/{n}", flush=True)
    res[name] = cnt
    for cat in ("filler", "clean"):     # filler FIRST per the rider
        c = cnt[cat]; tot = sum(c.values()) or 1
        print(f"[{name}][{cat}] n={tot} same-wrong {c['same_wrong']/tot:.3f} "
              f"new-right {c['new_right']/tot:.3f} new-wrong "
              f"{c['new_wrong']/tot:.3f}", flush=True)
json.dump(res, open(".cache/xout_threeway.json", "w"), indent=1)
print("[saved] .cache/xout_threeway.json — verdict reads against pinned bars")
