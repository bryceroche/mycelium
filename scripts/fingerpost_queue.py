"""fingerpost_queue.py — THE THIRD PASS of the annotation queue (2026-09-15):
the fingerpost's views on the tranche. Per row: the original parse and 4
sentence-permuted views (the trunk recomputed per view); per present slot
of the original its coarse key's AGREEMENT across views; the row's
INSTABILITY = the fraction of its slots with agreement <= 1 (94% wrong
on the holdout). Writes the queue back with the instability. A read.
Env: family env + FQ_CKPT, ALG_TEST(_NAME) (the tranche split), FQ_IN/FQ_OUT."""
import os, sys


def _main():
    import json, numpy as np
    sys.path.insert(0, "."); sys.path.insert(0, "scripts")
    from phase1_algebra_head import build_params, forward, load_alg, build_slot_masks, alt2_fact_buf, decode, sent_indices, L_FAC, K_VARS, T_ALG, TOKENIZER_JSON
    from beacon_closing_arm import recompute_states
    from tta_views import permuted_view
    from tokenizers import Tokenizer
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    vs, vst, vtk, vg, vse = load_alg("test"); N = len(vs)
    p = build_params(0); sd = safe_load(os.environ["FQ_CKPT"]); assert set(sd) == set(p)
    for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    tok = Tokenizer.from_file(TOKENIZER_JSON); KEYS = ("pres", "ftype", "op", "dig", "args", "res") + (("dup",) if "h_dup" in p else ())

    def two_pass(ts, tk, se, nv, ma):
        o0 = forward(p, ts, tk, se); onp0 = {k: o0[k].numpy() for k in ("fat", "args", "res")}
        mk = build_slot_masks(onp0, se.numpy()); _oa = {**onp0, **{k: o0[k].numpy() for k in ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())}}
        o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float), fact_buf=Tensor(alt2_fact_buf(_oa, se.numpy(), nv, ma), dtype=dtypes.float))
        return {k: o[k].numpy() for k in KEYS}

    def slot_keys(onp, b):
        row = {k: onp[k][b] for k in onp}; row["query"] = np.zeros(K_VARS, np.float32); keys = {}
        for j in range(L_FAC):
            if row["pres"][j] <= 0: continue
            rj = dict(row); pr = np.full_like(row["pres"], -1.0); pr[j] = row["pres"][j]; rj["pres"] = pr
            try: facs, _ = decode(rj)
            except Exception: facs = []
            keys[j] = tuple(sorted((f["ftype"], f.get("op", ""), f.get("value", "")) for f in facs))
        return keys

    inst = {}
    for s0 in range(0, N, 8):
        sl = np.arange(s0, min(s0 + 8, N)); pad = 8 - len(sl); sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        nv = np.array([K_VARS] * 8); ma = np.array([0] * 8)
        ts = Tensor(np.ascontiguousarray(vst[sl_p]), dtype=dtypes.half); tk = Tensor(vtk[sl_p].astype(np.float32)); se = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)
        keys0 = [slot_keys(two_pass(ts, tk, se, nv, ma), b) for b in range(8)]; agree = [{j: 0 for j in keys0[b]} for b in range(8)]
        for v in range(1, 5):
            texts = [permuted_view(vs[int(i)]["text"], 40000 + 10 * int(i) + v) for i in sl_p]
            ids = np.zeros((8, T_ALG), np.int32); msk = np.zeros((8, T_ALG), np.float32); snt = np.zeros((8, T_ALG), np.int32)
            for b, t in enumerate(texts):
                e = tok.encode(t); Ln = min(len(e.ids), T_ALG); ids[b, :Ln] = e.ids[:Ln]; msk[b, :Ln] = 1.0; snt[b] = sent_indices(t, list(e.offsets), msk[b])
            ov = two_pass(Tensor(recompute_states(ids), dtype=dtypes.half), Tensor(msk), Tensor(snt, dtype=dtypes.int), nv, ma)
            for b in range(8):
                kv = list(slot_keys(ov, b).values())
                for j, kj in keys0[b].items():
                    if kj in kv: agree[b][j] += 1; kv.remove(kj)
        for b, i in enumerate(sl):
            a = list(agree[b].values()); inst[int(i)] = {"n_slots": len(a), "unstable": (sum(x <= 1 for x in a) / len(a)) if a else None}
        if s0 % 80 == 0: print(f"[fingerpost-queue] {s0 + len(sl)}/{N}", flush=True)
    rows = [json.loads(l) for l in open(os.environ["FQ_IN"])]
    for r in rows:
        r["fingerpost"] = inst.get(int(r["rank"]) - 1)
    rows.sort(key=lambda r: (-(r["fingerpost"]["unstable"] or 0) if r.get("fingerpost") and r["fingerpost"]["unstable"] is not None else 0, r["rank"]))
    with open(os.environ["FQ_OUT"], "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    u = np.array([r["fingerpost"]["unstable"] for r in rows if r.get("fingerpost") and r["fingerpost"]["unstable"] is not None])
    print(f"[fingerpost-queue] wrote {os.environ['FQ_OUT']}: {len(rows)} rows; unstable fraction quartiles {np.percentile(u, [25, 50, 75]).round(2)}; rows with >= 1/3 unstable slots {np.mean(u >= 1 / 3):.2f}")


if __name__ == "__main__":
    _main()
