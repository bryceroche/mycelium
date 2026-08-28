"""emission_bus_witness.py — THE BUS AT THE DOOR, STAGE 1: OBSERVE
(2026-08-29, word given). Audit-before-diet: before the bus gets any veto
power at the emission door, MEASURE whether bus-lane concordance separates
the door's rights from its lies (the fingerpost law: a weaker reader must
prove discrimination before it votes).

For each emission dumped by scored_emission.py (EMIT_DUMP), read the raw
text through the BUS INCUMBENT (sharp_bindj) and solve two lanes:
  lane B — bindj's pointer parse as decoded (cross-lineage second reader);
  lane A — same parse with WIRING OVERRIDDEN by bus unbinding (arg1/arg2/res
           recovered from the wire per present slot; rel + given slots only;
           an op-id in a var role leaves the original pointer untouched).
Concordance = emitted top ∈ {ansA, ansB}.

PINNED BAR (before measurement): the veto 'refuse unless concordant' is
PROMOTION-ELIGIBLE only if, applied to the dumped emissions, it yields
net (rights_kept - lies_kept) >= +3 AND rights_kept >= 6 on gold.
Otherwise the read banks as observational and the door is untouched.
"""
import os, sys, json
os.environ.update({"DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "9",
                   "ALG_DUP": "1", "ALG_HW": "512", "ALG_WIDE": "1",
                   "ALG_BREATH": "7", "ALG_NOTEBOOK": "1", "ALG_SIXWAVE": "1",
                   "NB_PERSLOT": "1", "ALG_BINDBUS": "3", "ALG_BIND_D": "256",
                   "BIND_CODES": ".cache/bindbus_codes256.npz",
                   "ALG_TEST": ".cache/algebra_nl_bigtest.jsonl",
                   "ALG_TEST_NAME": "bigtest"})
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
from phase1_algebra_head import (build_params, forward, decode, T_ALG,
                                 TOKENIZER_JSON, sent_indices, load_alg,
                                 build_slot_masks, L_FAC)
from beacon_closing_arm import recompute_states
from tta_alg2_dials import solve2
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

_ = load_alg("test")
tok = Tokenizer.from_file(TOKENIZER_JSON)
bz = np.load(".cache/bindbus_codes256.npz")
CB = bz["CB"].astype(np.float32); P = CB.shape[1] // 2
CBc = (CB.reshape(32, P, 2)[..., 0] + 1j * CB.reshape(32, P, 2)[..., 1]).astype(np.complex64)
ROLE = {r: np.exp(-1j * bz[f"theta_{r}"]).astype(np.complex64)
        for r in ("arg1", "arg2", "res", "op")}


def unbind_id(wire, role):
    z = (wire.reshape(P, 2)[:, 0] + 1j * wire.reshape(P, 2)[:, 1]) * ROLE[role]
    z = z / (np.sqrt((np.abs(z) ** 2).sum()) + 1e-9)
    return int((z @ np.conj(CBc).T).real.argmax())


def bus_facs(facs, pres_j, Bv):
    """Wiring-override lane: substitute bus-recovered var ids into rel/given
    slots (i-th fac <-> i-th present slot, decode's order). Op-role ids
    (>=24) landing in a var role leave the pointer value untouched."""
    out = []
    for f, j in zip(facs, pres_j):
        f = dict(f)
        a1, a2, rs = (unbind_id(Bv[j], r) for r in ("arg1", "arg2", "res"))
        if f["ftype"] == "rel":
            if a1 < 24 and a2 < 24:
                f["args"] = sorted((a1, a2))
            if rs < 24:
                f["result"] = rs
        elif f["ftype"] == "given" and rs < 24:
            f["var"] = rs
        out.append(f)
    return out


def main():
    p = build_params(0)
    sd = safe_load(os.environ.get("BW_CKPT", ".cache/sharp_bindj.safetensors"))
    assert set(sd.keys()) == set(p.keys())
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    ems = json.load(open(os.environ.get("EMIT_DUMP", ".cache/door_emissions.json")))
    print(f"[bus-door] {len(ems)} emissions to witness", flush=True)
    K = ("pres", "ftype", "op", "islit", "dig", "args", "res", "query")
    stats = {}
    for s0 in range(0, len(ems), 8):
        sl = ems[s0:s0 + 8]
        ids = np.zeros((8, T_ALG), np.int32)
        msk = np.zeros((8, T_ALG), np.float32)
        snt = np.zeros((8, T_ALG), np.int32)
        for i, r in enumerate(sl):
            e = tok.encode(r["original"])
            Ln = min(len(e.ids), T_ALG)
            ids[i, :Ln] = e.ids[:Ln]; msk[i, :Ln] = 1.0
            snt[i] = sent_indices(r["original"], list(e.offsets), msk[i])
        st = np.asarray(recompute_states(ids)).astype(np.float32)
        o0 = forward(p, Tensor(st, dtype=dtypes.float),
                     Tensor(msk, dtype=dtypes.float),
                     Tensor(snt.astype(np.int32), dtype=dtypes.int))
        onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
        mk = build_slot_masks(onp0, snt)
        o = forward(p, Tensor(st, dtype=dtypes.float),
                    Tensor(msk, dtype=dtypes.float),
                    Tensor(snt.astype(np.int32), dtype=dtypes.int),
                    slot_mask=Tensor(mk, dtype=dtypes.float))
        keys = K + tuple(k for k in ("sel", "dup", "sgn") if k in o)
        onp = {k: o[k].realize().numpy() for k in keys}
        Bv = o["bind"].realize().numpy()
        for i, r in enumerate(sl):
            ob = {k: onp[k][i] for k in onp}
            facs, q = decode(ob)
            pres_j = [j for j in range(L_FAC) if ob["pres"][j] > 0]
            ans = {}
            for lane, ff in (("B", facs),
                             ("A", bus_facs(facs, pres_j, Bv[i]))):
                try:
                    ans[lane] = solve2(ff, q, {"n_vars": 24, "m": 300})
                except Exception:
                    ans[lane] = None
            concord = r["top"] in {v for v in ans.values() if v is not None}
            # STAGE 1b — the WIRING grain: Jaccard between the door's
            # winning-view edge set and the bus-recovered edge set, both in
            # the shared consecutive-letter var space
            jac = None
            if r.get("facs"):
                eg = edges_of(r["facs"])
                eb = set()
                for f, j in zip(facs, pres_j):
                    a1, a2, rs = (unbind_id(Bv[i][j], ro)
                                  for ro in ("arg1", "arg2", "res"))
                    if f["ftype"] == "rel" and a1 < 24 and a2 < 24 and rs < 24:
                        for a in sorted((a1, a2)):
                            eb.add(("rel", f.get("op"), a, rs))
                    elif f["ftype"] == "given" and rs < 24:
                        eb.add(("given", rs))
                jac = len(eg & eb) / max(len(eg | eb), 1)
            key = (r["tag"], r["right"])
            stats.setdefault(key, []).append(
                (concord, ans["A"] == r["top"], ans["B"] == r["top"], jac))
    print("tag   class  n   concord  laneA  laneB   J-mean  J-list")
    kept = {"r": 0, "l": 0}
    for (tag, right), xs in sorted(stats.items()):
        n = len(xs)
        c = sum(1 for x in xs if x[0])
        a = sum(1 for x in xs if x[1]); b = sum(1 for x in xs if x[2])
        js = [x[3] for x in xs if x[3] is not None]
        jm = sum(js) / max(len(js), 1)
        cls = "RIGHT" if right else "LIE"
        print(f"{tag:5s} {cls:5s} {n:3d}   {c}/{n}     {a}/{n}   {b}/{n}   "
              f"{jm:.3f}  {[round(j, 2) for j in js]}", flush=True)
        if tag == "gold":
            kept["r" if right else "l"] = c
    net = kept["r"] - kept["l"]
    ok = net >= 3 and kept["r"] >= 6
    print(f"[verdict answer-grain] veto-applied gold: rights kept {kept['r']}, "
          f"lies kept {kept['l']}, net {net:+d} (bar: net >= +3 AND rights >= 6) "
          f"-> {'PROMOTION-ELIGIBLE' if ok else 'OBSERVATIONAL ONLY'}", flush=True)
    gj = {right: [x[3] for x in stats.get(("gold", right), []) if x[3] is not None]
          for right in (0, 1)}
    if gj[0] and gj[1]:
        mr, ml = sum(gj[1]) / len(gj[1]), sum(gj[0]) / len(gj[0])
        print(f"[verdict wiring-grain] gold J: rights mean {mr:.3f} vs lies "
              f"mean {ml:.3f} (separation {mr - ml:+.3f}) — OBSERVATIONAL "
              f"(promotion needs pinned tau on FRESH emissions)", flush=True)


def edges_of(facs):
    eg = set()
    for f in facs:
        if f["ftype"] == "rel":
            for a in sorted(f.get("args", [])):
                eg.add(("rel", f.get("op"), a, f.get("result")))
        elif f["ftype"] == "given":
            eg.add(("given", f.get("var")))
    return eg


if __name__ == "__main__":
    main()
