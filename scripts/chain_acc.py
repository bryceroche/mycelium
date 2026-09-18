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
    correct = refused = wrong = 0; nd = None; tasks = []; keys = {}
    for s0 in range(0, n, 8):
        sl = np.arange(s0, min(s0 + 8, n)); pad = 8 - len(sl); sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        nv = np.array([vs[int(i)].get("n_vars", K_VARS) for i in sl_p]); ma = np.array([vs[int(i)].get("m", 0) for i in sl_p])
        ts = Tensor(np.ascontiguousarray(vst[sl_p]), dtype=dtypes.half); tk = Tensor(vtk[sl_p].astype(np.float32)); se = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)
        o0 = forward(p, ts, tk, se); onp0 = {k: o0[k].numpy() for k in ("fat", "args", "res")}
        mk = build_slot_masks(onp0, se.numpy()); _oa = {**onp0, **{k: o0[k].numpy() for k in ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())}}
        o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float), fact_buf=Tensor(alt2_fact_buf(_oa, se.numpy(), nv, ma), dtype=dtypes.float))
        onp = {k: o[k].numpy() for k in KEYS}; qv = o["query"].numpy().argmax(-1)
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
    for (i, q, key, parse, gv, nvv) in tasks:
        st, val = out.get(i, ("hung", None))
        if st != "solved": refused += 1
        elif val == key: correct += 1
        else: wrong += 1
    print(f"[chain-acc] {os.path.basename(os.environ['CA_CKPT'])} on {os.environ.get('ALG_TEST_NAME')} mask={int(mask)}: rows {n} | CORRECT {correct} ({correct/n:.3f}) | refused {refused} ({refused/n:.3f}) | wrong {wrong} ({wrong/n:.3f})", flush=True)

if __name__ == "__main__":
    main()
