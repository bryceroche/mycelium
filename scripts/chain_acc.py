"""chain_acc.py — ROW-LEVEL CHAIN ACCURACY (2026-09-18, the headline wild read): for every row of the test
split, the machine's decoded graph (optionally with THE NUMERAL MASK on the givens' digits) goes to the June
solver; the row is CORRECT when the solver assigns the machine's query variable the row's key. Order-free
(the solver reads a set), mask-aware, and what MATH-500 measures. Prints correct / refused (no solve) / wrong.
Env: family env + CA_CKPT, ALG_TEST(_NAME), CA_MASK=1 (the numeral mask), CA_WALL (s per row, default 2)."""
import os, sys, re, json, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "scripts")

def main():
    from phase1_algebra_head import build_params, forward, load_alg, build_slot_masks, alt2_fact_buf, _decode_slots, K_VARS
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    from admit_annotation import solve_walled
    from alternator_bridge import problem_from_algebra3
    from mycelium.custody_gold import row_gold
    from mycelium import lexicon as L
    vs, vst, vtk, vg, vse = load_alg("test"); n = len(vs)
    p = build_params(0); sd = safe_load(os.environ["CA_CKPT"]); assert set(sd) == set(p)
    for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    mask = bool(int(os.environ.get("CA_MASK", "0"))); wall = float(os.environ.get("CA_WALL", "2"))
    KEYS = ("pres", "ftype", "op", "dig", "args", "res") + (("dup",) if "h_dup" in p else ())
    def lsm(x): x = x - x.max(-1, keepdims=True); return x - np.log(np.exp(x).sum(-1, keepdims=True))
    def legal(text):
        vals = {1}
        for m in re.findall(r"\d[\d,]*", text):
            try: v = int(m.replace(",", ""))
            except ValueError: continue
            if 0 <= v < 10 ** 7: vals.add(v)
        for _, _, v in L.constants(text):
            if 0 <= int(v) < 10 ** 7: vals.add(int(v))
        return sorted(vals)
    correct = refused = wrong = 0; nd = None
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
            if mask:
                vals = legal(vs[i]["text"]); dg = lsm(row["dig"]); nd = dg.shape[1]
                for j in range(row["ftype"].shape[0]):
                    if int(row["ftype"][j].argmax()) == 0: continue
                    best = max(vals, key=lambda v: sum(dg[j, d, (v // 10 ** (nd - 1 - d)) % 10] for d in range(nd)))
                    fake = np.full_like(row["dig"][j], -1e9)
                    for d in range(nd): fake[d, (best // 10 ** (nd - 1 - d)) % 10] = 0.0
                    row["dig"][j] = fake
            parse = _decode_slots(row); q = int(qv[bi])
            try: key = int(row_gold(vs[i]))
            except Exception: key = vs[i].get("key"); key = int(key) if key is not None else None
            used = [f.get("var") for f in parse if f["ftype"] == "given"] + [a for f in parse if f["ftype"] == "rel" for a in list(f["args"]) + [f["result"]]]
            nvv = max([q + 1] + [v + 1 for v in used if v is not None]); gv = {f["var"]: f["value"] for f in parse if f["ftype"] == "given"}
            if key is None or not parse: refused += 1; continue
            try:
                res = solve_walled(problem_from_algebra3(nvv, parse, gv, 10000), budget=5000, wall=wall)
            except Exception: res = {"status": "unbuildable"}
            if res.get("status") != "solved": refused += 1; continue
            if int(res["assignment"][q]) == key: correct += 1
            else: wrong += 1
    print(f"[chain-acc] {os.path.basename(os.environ['CA_CKPT'])} on {os.environ.get('ALG_TEST_NAME')} mask={int(mask)}: rows {n} | CORRECT {correct} ({correct/n:.3f}) | refused {refused} ({refused/n:.3f}) | wrong {wrong} ({wrong/n:.3f})", flush=True)

if __name__ == "__main__":
    main()
