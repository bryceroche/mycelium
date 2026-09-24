"""chain_acc.py — ROW-LEVEL CHAIN ACCURACY (2026-09-18, the headline wild read): for every row of the test
split, the machine's decoded graph (optionally with THE NUMERAL MASK on the givens' digits) goes to the June
solver; the row is CORRECT when the solver assigns the machine's query variable the row's key. Order-free
(the solver reads a set), mask-aware, and what MATH-500 measures. Prints correct / refused (no solve) / wrong.
Env: family env + CA_CKPT, ALG_TEST(_NAME), CA_MASK=1 (the numeral mask), CA_WALL (s per row, default 2)."""
import os, sys, re, json, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "scripts")

def _solve_task(t):
    """one row's solve in a worker: (i, q, key, parse, gv, nvv) -> (i, status, value); the ladder capped at 10^4
    (a wrong decoded graph is refused fast on a small domain; the 10^5 rung is the gate's for certified graphs)"""
    import os as _os, sys as _sys; _sys.path.insert(0, "."); _sys.path.insert(0, "scripts")
    from admit_annotation import solve_walled
    from alternator_bridge import problem_from_algebra3
    i, q, key, parse, gv, nvv = t; wall = float(_os.environ.get("CA_WALL", "3"))
    gmax = max([int(v) for v in gv.values()] + [1]); m0 = int(min(10001, max(300, 2 * gmax, 2 * key)))
    for m in ([m0] + ([10000] if m0 < 10000 else [])):
        try: res = solve_walled(problem_from_algebra3(nvv, parse, gv, m), budget=5000, wall=wall)
        except Exception: return i, "unbuildable", None
        if res.get("status") == "solved": return i, "solved", int(res["assignment"][q])
    return i, res.get("status", "?"), None


def main():
    from phase1_algebra_head import build_params, forward, load_alg, build_slot_masks, alt2_fact_buf, _decode_slots, K_VARS
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    from alternator_bridge import problem_from_algebra3
    from mycelium.custody_gold import row_gold
    vs, vst, vtk, vg, vse = load_alg("test"); n = len(vs)
    p = build_params(0); sd = safe_load(os.environ["CA_CKPT"]); assert set(sd) == set(p)
    for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    mask = bool(int(os.environ.get("CA_MASK", "0"))); wall = float(os.environ.get("CA_WALL", "3"))
    KEYS = ("pres", "ftype", "op", "dig", "args", "res") + (("dup",) if "h_dup" in p else ())
    from mycelium.rulebook import legal_digit_logits
    _HUD_ON = int(os.environ.get("ALG_HUD", "0")) != 0   # THE TOKEN HUD, read-time road (2026-09-21)
    _BUSREG_ON = float(os.environ.get("ALG_BUSREG", "0")) != 0 or float(os.environ.get("ALG_IDKEY", "0")) != 0   # THE BUS REGISTER's token-id port (2026-09-22)
    if _HUD_ON or _BUSREG_ON:
        import phase1_algebra_head as _HH
    correct = refused = wrong = 0; nd = None; tasks = []; keys = {}; rawdump = [] if os.environ.get("CA_RAWDUMP") else None
    for s0 in range(0, n, 8):
        sl = np.arange(s0, min(s0 + 8, n)); pad = 8 - len(sl); sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        nv = np.array([vs[int(i)].get("n_vars", K_VARS) for i in sl_p]); ma = np.array([vs[int(i)].get("m", 0) for i in sl_p])
        ts = Tensor(np.ascontiguousarray(vst[sl_p]), dtype=dtypes.half); tk = Tensor(vtk[sl_p].astype(np.float32)); se = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)
        hud_t = None
        if _HUD_ON:
            hud_t = Tensor(np.stack([_HH.hud_row_features(
                vs[int(i)]["text"], vtk[i], vse[i], _HH.T_ALG)
                for i in sl_p]).astype(np.int32), dtype=dtypes.int)
        ident_t = None
        if _BUSREG_ON:   # THE BUS REGISTER's port: the row's token ids, host-side per batch
            ident_t = Tensor(np.stack([_HH.ident_row_ids(vs[int(i)]["text"], _HH.T_ALG) for i in sl_p]).astype(np.int32), dtype=dtypes.int)
        o0 = forward(p, ts, tk, se, hud=hud_t, ident=ident_t); onp0 = {k: o0[k].numpy() for k in ("fat", "args", "res")}
        mk = build_slot_masks(onp0, se.numpy()); _oa = {**onp0, **{k: o0[k].numpy() for k in ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())}}
        _fb_t = Tensor(alt2_fact_buf(_oa, se.numpy(), nv, ma), dtype=dtypes.float)
        _valreg_on = bool(float(os.environ.get("ALG_VALREG", "0")) or float(os.environ.get("ALG_VALREG_ADDR", "0")))
        _alt3 = int(os.environ.get("ALG_ALT3", "0")) != 0
        f3_t = f5_t = None
        if _alt3:   # THE THREE CONSULTS at read (2026-09-24): the trainer's three curbs, eager here
            _ck3 = ("pres", "ftype", "op", "dig", "args", "res") + (("dup",) if "dup" in o0 else ())
            _mk3 = Tensor(mk, dtype=dtypes.float)
            _oa3 = forward(p, ts, tk, se, slot_mask=_mk3, hud=hud_t, ident=ident_t, stop_after=2)
            f3_t = Tensor(alt2_fact_buf({k: _oa3[k].numpy() for k in _ck3}, se.numpy(), nv, ma), dtype=dtypes.float)
            _ob3 = forward(p, ts, tk, se, slot_mask=_mk3, hud=hud_t, ident=ident_t, stop_after=4, facts3=f3_t)
            f5_t = Tensor(alt2_fact_buf({k: _ob3[k].numpy() for k in _ck3}, se.numpy(), nv, ma), dtype=dtypes.float)
        o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float), fact_buf=(None if _alt3 else _fb_t), hud=hud_t, ident=ident_t, valfact=(_fb_t if _valreg_on else None), facts3=f3_t, facts5=f5_t)   # THE VALUE STREAM: the live pass-1 facts; THE THREE CONSULTS' facts
        onp = {k: o[k].numpy() for k in KEYS}; qv = o["query"].numpy().argmax(-1); qlog = o["query"].numpy()
        if rawdump is not None:   # CA_RAWDUMP=path: every slot's raw heads per row, for the annealed decode (2026-09-18)
            for bi, i in enumerate(sl):
                rawdump.append({"i": int(i), "text": vs[int(i)]["text"], "q": qlog[bi].astype(np.float32), **{k: onp[k][bi].astype(np.float32) for k in KEYS}})
        for bi, i in enumerate(sl):
            i = int(i); row = {k: onp[k][bi].copy() for k in KEYS}
            if mask:   # THE NUMERAL MASK — one rulebook, two doors
                for j in range(row["ftype"].shape[0]):
                    if int(row["ftype"][j].argmax()) == 0: continue
                    fake = legal_digit_logits(row["dig"][j], vs[i]["text"])
                    if fake is not None: row["dig"][j] = fake
            parse = _decode_slots(row); q = int(qv[bi])
            try: key = int(row_gold(vs[i]))
            except Exception: key = vs[i].get("key"); key = int(key) if key is not None else None
            used = [f.get("var") for f in parse if f["ftype"] == "given"] + [a for f in parse if f["ftype"] == "rel" for a in list(f["args"]) + [f["result"]]]
            nvv = max([q + 1] + [v + 1 for v in used if v is not None]); gv = {f["var"]: f["value"] for f in parse if f["ftype"] == "given"}
            if key is None or not parse: refused += 1; continue
            tasks.append((i, q, key, parse, gv, nvv)); keys[i] = key
    # THE SOLVES ACROSS CORES (2026-09-18): the rows are independent; a hang-proof unordered map with a wall per result
    import multiprocessing as mp
    workers = int(os.environ.get("CA_WORKERS", "6")); out = {}
    with mp.get_context("spawn").Pool(workers) as pool:
        it = pool.imap_unordered(_solve_task, tasks, chunksize=1)
        try:
            for _ in range(len(tasks)):
                i, st, val = it.next(timeout=wall * 3 + 30); out[i] = (st, val)
        except mp.TimeoutError:
            print(f"[chain-acc] {len(tasks) - len(out)} rows never returned — counted as refused", flush=True)
    per_row = {}
    for (i, q, key, parse, gv, nvv) in tasks:
        st, val = out.get(i, ("hung", None))
        if st != "solved": refused += 1; per_row[i] = "refused"
        elif val == key: correct += 1; per_row[i] = "correct"
        else: wrong += 1; per_row[i] = "wrong"
    if os.environ.get("CA_ROWS"):   # per-row labels for the certifier's bench (2026-09-18)
        json.dump({str(k): v for k, v in per_row.items()}, open(os.environ["CA_ROWS"], "w"))
    if rawdump is not None:
        import pickle
        for r in rawdump:
            try: r["key"] = int(row_gold(vs[r["i"]]))
            except Exception: r["key"] = vs[r["i"]].get("key")
        pickle.dump(rawdump, open(os.environ["CA_RAWDUMP"], "wb")); print(f"[chain-acc] raw slots for {len(rawdump)} rows -> {os.environ['CA_RAWDUMP']}", flush=True)
    print(f"[chain-acc] {os.path.basename(os.environ['CA_CKPT'])} on {os.environ.get('ALG_TEST_NAME')} mask={int(mask)}: rows {n} | CORRECT {correct} ({correct/n:.3f}) | refused {refused} ({refused/n:.3f}) | wrong {wrong} ({wrong/n:.3f})", flush=True)

if __name__ == "__main__":
    main()
