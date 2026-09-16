"""locus_precompute.py — THE NL CERTIFICATE FOR TRAINING (2026-09-15, word
given for parity if warranted): the fingerpost's per-slot instability over
the diet's PEN rows (the wild register in the training mix), computed ONCE
with a FROZEN TEACHER (a fixed checkpoint) and 4 sentence-permuted views —
a slot of the teacher's parse is UNSTABLE when its coarse key is absent in
>= LP_K of the views. Output: npz with row indices (into the train split)
and an (n, L_FAC) unstable mask. Declared use (mycelium/diagnostic_register:
view instability is loss-never): the mask enters training as a per-slot
LOSS WEIGHT from a frozen teacher — curricular, no gradient reaches the
meter, the model cannot game it. Env: family env + LP_CKPT, ALG_TRAIN(_NAME),
LP_N (0 = all pen rows), LP_K (2), LP_OUT."""
import os, sys


def _main():
    import json, numpy as np, time
    sys.path.insert(0, "."); sys.path.insert(0, "scripts")
    from phase1_algebra_head import build_params, forward, load_alg, build_slot_masks, alt2_fact_buf, decode, sent_indices, L_FAC, K_VARS, T_ALG, TOKENIZER_JSON
    from beacon_closing_arm import recompute_states
    from tta_views import permuted_view
    from tokenizers import Tokenizer
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    vs, vst, vtk, vg, vse = load_alg("train")
    pen = np.array([i for i, r in enumerate(vs) if isinstance(r.get("gen"), dict) and r["gen"].get("src") == "gsm8k"])
    n_cap = int(os.environ.get("LP_N", "0") or 0); idx = pen[:n_cap] if n_cap else pen; LP_K = int(os.environ.get("LP_K", "2"))
    p = build_params(0); sd = safe_load(os.environ["LP_CKPT"]); assert set(sd) == set(p)
    for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    tok = Tokenizer.from_file(TOKENIZER_JSON); KEYS = ("pres", "ftype", "op", "dig", "args", "res") + (("dup",) if "h_dup" in p else ())
    print(f"[locus] {len(pen)} pen rows in the diet; computing {len(idx)}; teacher {os.path.basename(os.environ['LP_CKPT'])}; unstable = absent in >= {LP_K}/4 views", flush=True)

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

    unstable = np.zeros((len(idx), L_FAC), np.float32); present = np.zeros((len(idx), L_FAC), np.float32); t0 = time.time()
    for s0 in range(0, len(idx), 8):
        sl = idx[s0:s0 + 8]; pad = 8 - len(sl); sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        nv = np.array([vs[int(i)].get("n_vars", K_VARS) for i in sl_p]); ma = np.array([vs[int(i)].get("m", 0) for i in sl_p])
        ts = Tensor(np.ascontiguousarray(vst[sl_p]), dtype=dtypes.half); tk = Tensor(vtk[sl_p].astype(np.float32)); se = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)
        keys0 = [slot_keys(two_pass(ts, tk, se, nv, ma), b) for b in range(8)]; miss = np.zeros((8, L_FAC))
        for v in range(1, 5):
            texts = [permuted_view(vs[int(i)]["text"], 40000 + 10 * int(i) + v) for i in sl_p]
            ids = np.zeros((8, T_ALG), np.int32); msk = np.zeros((8, T_ALG), np.float32); snt = np.zeros((8, T_ALG), np.int32)
            for b, t in enumerate(texts):
                e = tok.encode(t); Ln = min(len(e.ids), T_ALG); ids[b, :Ln] = e.ids[:Ln]; msk[b, :Ln] = 1.0; snt[b] = sent_indices(t, list(e.offsets), msk[b])
            ov = two_pass(Tensor(recompute_states(ids), dtype=dtypes.half), Tensor(msk), Tensor(snt, dtype=dtypes.int), nv, ma)
            for b in range(8):
                kv = list(slot_keys(ov, b).values())
                for j, kj in keys0[b].items():
                    if kj in kv: kv.remove(kj)
                    else: miss[b, j] += 1
        for b in range(len(sl)):
            for j in keys0[b]: present[s0 + b, j] = 1.0
            unstable[s0 + b] = (miss[b] >= LP_K).astype(np.float32) * present[s0 + b]
        if s0 % 800 == 0: print(f"[locus] {s0 + len(sl)}/{len(idx)} ({(time.time() - t0) / 60:.0f} min)", flush=True)
    out = os.environ.get("LP_OUT", ".cache/locus_pen_ds242.npz")
    np.savez(out, rows=idx, unstable=unstable, present=present, teacher=np.array(os.path.basename(os.environ["LP_CKPT"])), k=np.array(LP_K))
    print(f"[locus] wrote {out}: {len(idx)} rows; unstable slots {unstable.sum() / max(present.sum(), 1):.3f} of present; rows with any {np.mean(unstable.sum(1) > 0):.3f}")


if __name__ == "__main__":
    _main()
