"""AG3 part 4, THE VIEW-1 BEAM DUMP (ledger 2026-11-04): gsm_beam_dump.py's read on the VIEW items. Zero-training, CPU.

gsm_trunk_reader.load is REPLACED (before gsm_beam_read is imported) by a loader of the view data / view index / view states
(env V_DATA, V_IDX, V_STATES -- the reader's RD_STATES must name the view states; BT_ROLEF the view role feats), so the banked BT scores
the view texts through the unmodified BEAM machinery (GB.gate_batch / GB.beam). rho0 and lam are NOT refit: the view-1 dump REUSES VIEW 0's
fit from .cache/gsmb_agdump_{TAG}.json (the same model; the selector's TRAIN fit is a property of the model, not of the view; fitting on
view-1 TRAIN would need TRAIN views and would let the two views' selectors differ). Items stay in view-data order (= view 0's order).
-> .cache/gsmb_agdump_{TAG}_v1.json (same schema as the view-0 dump; tag = {TAG}_v1)   BEAM_SMOKE=n: first n per split, '_smoke' suffix.
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
import numpy as np
import gsm_breath_reader as _BR   # FIRST: it sets RD_ARM=TICK, which gsm_trunk_reader reads at import (11-04 bug: TR imported
TR = _BR.TR                         # first ran the VANILLA arm: no step tick, no notebook features; the first AG3 read is VOID)
assert TR.ARM == "TICK" and TR.TICK and TR.NOTE, f"trunk reader arm {TR.ARM}: the BT's encoder features are missing"

E = os.environ.get


def load():
    Dt = json.load(open(E("V_DATA", ".cache/gsmb_data_v1.json"))); idx = json.load(open(E("V_IDX", ".cache/gsmb_trunk_index_v1.json")))
    for sp, its in Dt.items():
        its[:] = its[:len(idx[sp])]
        for it, s in zip(its, idx[sp]):
            it["st0"] = s
            nl = {n.lower(): n for n in it["names"]}; it["surf"] = [nl.get(x, x) if f else x for x, f in zip(it["toks"], it["isn"])]
    st = E("RD_STATES", "")
    assert st.endswith("_v1.npy") or E("AG3_SMOKE_OK") == "1", f"RD_STATES={st!r} is not a view-1 states file"
    return Dt, np.load(TR.GD.V1.C(st), mmap_mode="r")


TR.load = load
import gsm_beam_read as GB
BR, Dt = GB.BR, GB.Dt


def dump(its, rho):
    """as gsm_beam_dump.dump: per item greedy value, every finished beam program (log p, final value, right?)"""
    res = []
    for s0 in range(0, len(its), 128):
        part = its[s0:s0 + 128]
        t, Ee, xq, L, progs, cands = GB.gate_batch(part, rho, s0)
        for b, it in enumerate(part):
            vals = it["vals"] + [None] * (L - len(it["vals"]))
            prog = []
            for st in progs[b]:
                if st[0] == ".": break
                prog.append(st)
            g = BR.GD.execute(prog, vals)
            res.append(dict(g=None if g is None else float(g), g_ok=bool(g is not None and BR.GD.V1.near(g, it["final"])),
                            beam=[(float(lp), float(v), bool(BR.GD.V1.near(v, it["final"]))) for lp, _, v in GB.beam(cands[b], vals)]))
    return res


SM = GB.SMOKE or None
d0 = json.load(open(f".cache/gsmb_agdump_{GB.TAG}.json"))                  # view 0's dump: the same model's rho0 / lam
assert d0["load"] == BR.LOAD, f"view-0 dump was made by {d0['load']}, this is {BR.LOAD}"
rho, lam = d0["rho0"], tuple(d0["lam"])
out = dict(tag=GB.TAG + "_v1", load=BR.LOAD, bw=GB.BW, rho0=rho, lam=lam, view=1)
print(f"[agdump_v1] {out['tag']} rho0 {rho} lam {lam} (reused from view 0)", flush=True)
for split in ("ORIG", "TEST"):
    out[split] = dump(Dt[split][:SM], rho)
    assert len(out[split]) == len(d0[split]) or SM, f"{split}: {len(out[split])} items vs view 0's {len(d0[split])}"
    print(f"[agdump_v1] {out['tag']} {split} n {len(out[split])} greedy {np.mean([r['g_ok'] for r in out[split]]):.4f} (view 0: {np.mean([r['g_ok'] for r in d0[split]]):.4f})", flush=True)
json.dump(out, open(f".cache/gsmb_agdump_{out['tag']}{'_smoke' if SM else ''}.json", "w"))
