"""parse_dump.py — dump the decoded parse (givens / relations, with the
variables each references) beside the gold for rows matching a phrase
(default: the multipliers), on a checkpoint. A read. Env: family env +
PD_CKPT, ALG_TEST(_NAME), PD_PHRASES (regex)."""
import os, sys, re


def _main():
    import numpy as np, json
    sys.path.insert(0, "."); sys.path.insert(0, "scripts")
    from phase1_algebra_head import build_params, forward, load_alg, build_slot_masks, alt2_fact_buf, _decode_slots, L_FAC, K_VARS
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    pat = re.compile(os.environ.get("PD_PHRASES", r"\b(twice|double|half|triple|a pair of|a couple of|dozen)\b"), re.I)
    vs, vst, vtk, vg, vse = load_alg("test")
    idx = np.array([i for i, r in enumerate(vs) if pat.search(r["text"])])
    p = build_params(0); sd = safe_load(os.environ["PD_CKPT"]); assert set(sd) == set(p)
    for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    KEYS = ("pres", "ftype", "op", "dig", "args", "res") + (("dup",) if "h_dup" in p else ())
    print(f"[parse-dump] {len(idx)} rows match {pat.pattern}")
    for s0 in range(0, len(idx), 8):
        sl = idx[s0:s0 + 8]; pad = 8 - len(sl); sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        nv = np.array([vs[int(i)].get("n_vars", K_VARS) for i in sl_p]); ma = np.array([vs[int(i)].get("m", 0) for i in sl_p])
        ts = Tensor(np.ascontiguousarray(vst[sl_p]), dtype=dtypes.half); tk = Tensor(vtk[sl_p].astype(np.float32)); se = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)
        o0 = forward(p, ts, tk, se); onp0 = {k: o0[k].numpy() for k in ("fat", "args", "res")}
        mk = build_slot_masks(onp0, se.numpy()); _oa = {**onp0, **{k: o0[k].numpy() for k in ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())}}
        o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float), fact_buf=Tensor(alt2_fact_buf(_oa, se.numpy(), nv, ma), dtype=dtypes.float))
        onp = {k: o[k].numpy() for k in KEYS}
        for bi, i in enumerate(sl):
            i = int(i); r = vs[i]; parse = _decode_slots({k: onp[k][bi] for k in KEYS})
            m = pat.search(r["text"]); ctx = r["text"][max(0, m.start() - 40): m.end() + 40].replace("\n", " ")
            giv = {f["var"]: f["value"] for f in parse if f["ftype"] == "given"}; rels = [(f["op"], f["args"], f["result"]) for f in parse if f["ftype"] == "rel"]
            refd = {a for _, args, res in rels for a in list(args) + [res]}
            gg = {f["var"]: f["value"] for f in r["factors"] if f["ftype"] == "given"}; gr = [(f["op"], f["args"], f["result"]) for f in r["factors"] if f["ftype"] == "rel"]
            print(f"row {i} ...{ctx}...\n   decoded givens {giv} | rels {rels[:5]} | vars referenced by rels but given by none: {sorted(refd - set(giv))}\n   gold    givens {gg} | rels {gr[:5]}")


if __name__ == "__main__":
    _main()
