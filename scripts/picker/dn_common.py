"""dn_common.py -- shared gold-scoring helpers for the three denoising-breath zero-GPU reads
(READ A/B/C, ledger 2026-10-12 12:35, "THE DENOISING BREATH, AMENDED BY OPUS'S CRITIQUE").

Reuses loop_val.py's own per-slot "ok" convention VERBATIM (scripts/loop_val.py:508-550) rather
than reimplementing a different notion of correctness -- same field semantics, same gset/top2/dup
logic for rel slots, same digit-array-equality for non-rel slots. Never touches loop_val.py itself
(new file, scripts/picker/ per the mission brief).

Gold comes from the custody door's g_* arrays (.cache/phase1_alg_states_<split>.npz), never a
fixture's own 'factors' field (the custody-gold law).
"""
import numpy as np


def load_gold(states_npz):
    z = np.load(states_npz)
    return {k[2:]: z[k] for k in z.files if k.startswith("g_")}


def score_row_slots(headrec, vg, i, l_fac=24):
    """headrec: dict with full-row arrays pres(L,) ftype(L,C) op(L,2) dig(L,7,10) args(L,L) res(L,L)
    [dup(L,)]. Returns {j: dict(ok,f_pres,f_ftype,f_res,f_op,f_args,f_dig,is_rel)} for gold-present j."""
    out = {}
    pres = np.asarray(headrec["pres"]); ftype = np.asarray(headrec["ftype"])
    op = np.asarray(headrec["op"]); dig = np.asarray(headrec["dig"])
    args = np.asarray(headrec["args"]); res = np.asarray(headrec["res"])
    dup = np.asarray(headrec["dup"]) if "dup" in headrec and headrec["dup"] is not None else None
    for j in range(l_fac):
        if vg["presence"][i, j] < 0.5:
            continue
        f_pres = bool(pres[j] > 0)
        f_ftype = int(ftype[j].argmax()) == int(vg["ftype"][i, j])
        f_res = int(res[j].argmax()) == int(vg["res"][i, j])
        ok = f_pres and f_ftype and f_res
        is_rel = int(vg["ftype"][i, j]) == 0
        f_op = f_args = f_dig = None
        if is_rel:
            gset = set(np.where(vg["args"][i, j] > .5)[0].tolist())
            f_op = int(op[j].argmax()) == int(vg["op"][i, j])
            if len(gset) == 1 and dup is not None:
                f_args = bool(dup[j] > 0) and int(np.argmax(args[j])) in gset
            else:
                top2 = set(np.argsort(-args[j])[:2].tolist())
                f_args = top2 == gset
            ok = ok and f_op and f_args
        else:
            f_dig = bool((dig[j].argmax(-1) == vg["digits"][i, j]).all())
            ok = ok and f_dig
        out[j] = dict(ok=ok, f_pres=f_pres, f_ftype=f_ftype, f_res=f_res,
                      f_op=f_op, f_args=f_args, f_dig=f_dig, is_rel=is_rel)
    return out


def index_breaths(rawbreaths_pkl_dict):
    """{kb: [rowrec,...]} -> {kb: {i: rowrec}}"""
    return {kb: {int(r["i"]): r for r in recs} for kb, recs in rawbreaths_pkl_dict.items()}
