"""perceiver_queue.py — THE ANNOTATION QUEUE, first pass (2026-09-15, wild is
the priority): TRAINING-side wild rows (the pen rows inside the diet; never
the holdout, a measurement fixture) ranked by what sees wild failure — the
row's wrong slots (gold-scored: these rows carry gold) and the least
confident relation slot's args margin (the perceiver's compass, 0.76) — with
the row's nearest kind on both atlas charts. Output: a JSONL queue with the
row's text, its wrong slots, the margins, the kind, and the annotation the
books' L3 lane needs (spans / mentions for quantity phrases and relations).
Views (the fingerpost) are the second pass on the top tranche. A read.
Env: family env + ALG_MINE_BREATHS=1 (set), PQ_CKPT, PQ_N (2000), PQ_OUT."""
import os, sys


def _main():
    import json, numpy as np
    os.environ.setdefault("ALG_MINE_BREATHS", "1")
    sys.path.insert(0, "."); sys.path.insert(0, "scripts")
    from phase1_algebra_head import build_params, forward, load_alg, build_slot_masks, alt2_fact_buf, _slot_margins, L_FAC, K_VARS
    from mycelium.step_atlas import load_atlas
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    vs, vst, vtk, vg, vse = load_alg("train")
    pen = np.array([i for i, r in enumerate(vs) if isinstance(r.get("gen"), dict) and r["gen"].get("src") == "gsm8k"])
    N = min(int(os.environ.get("PQ_N", "2000")), len(pen)); rng = np.random.RandomState(0); idx = np.sort(rng.choice(pen, N, replace=False))
    print(f"[queue] {len(pen)} pen rows in the diet; reading {N}", flush=True)
    p = build_params(0); sd = safe_load(os.environ["PQ_CKPT"]); assert set(sd) == set(p)
    for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    atlas = load_atlas(os.environ.get("PV_ATLAS", ".cache/step_atlas_ds242.npz"), manifest_path=os.environ.get("MH_ATLAS_MANIFEST", ".cache/RESEARCH_MANIFEST.json"))
    assert atlas.stamp == os.path.basename(os.environ["PQ_CKPT"]), "atlas era != ckpt"
    out = []
    for s0 in range(0, N, 8):
        sl = idx[s0:s0 + 8]; pad = 8 - len(sl); sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        ts = Tensor(np.ascontiguousarray(vst[sl_p]), dtype=dtypes.half); tk = Tensor(vtk[sl_p].astype(np.float32)); se = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)
        o0 = forward(p, ts, tk, se); onp0 = {k: o0[k].numpy() for k in ("fat", "args", "res")}
        mk = build_slot_masks(onp0, vse[sl_p].astype(np.int32)); _oa = {**onp0, **{k: o0[k].numpy() for k in ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())}}
        _nv = np.array([vs[int(i)].get("n_vars", K_VARS) for i in sl_p]); _ma = np.array([vs[int(i)].get("m", 0) for i in sl_p])
        o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float), fact_buf=Tensor(alt2_fact_buf(_oa, vse[sl_p].astype(np.int32), _nv, _ma), dtype=dtypes.float))
        hd = {k: v.numpy() for k, v in o["heads_all"][-1].items() if k in ("pres", "ftype", "op", "dig", "args", "res", "dup")}
        S = o["breaths_all"][-1].numpy().mean(1); NL = o["nl_all"][-1].numpy()
        _, cs = atlas.radius("slot", 6, S); _, ct = atlas.radius("token", 6, NL)
        for bi, i in enumerate(sl):
            i = int(i); r = vs[i]; row = {k: hd[k][bi] for k in hd}; wrong = []; n_p = 0; min_args = None
            for j in range(L_FAC):
                if vg["presence"][i, j] < 0.5: continue
                n_p += 1; rel = vg["ftype"][i, j] == 0
                ok = (row["pres"][j] > 0) and int(row["ftype"][j].argmax()) == vg["ftype"][i, j] and int(row["res"][j].argmax()) == vg["res"][i, j]
                if rel:
                    ok = ok and int(row["op"][j].argmax()) == vg["op"][i, j]; gset = set(np.where(vg["args"][i, j] > .5)[0].tolist())
                    ok = ok and ((bool(row["dup"][j] > 0) and int(np.argmax(row["args"][j])) in gset) if (len(gset) == 1 and "dup" in row) else set(np.argsort(-row["args"][j])[:2].tolist()) == gset)
                    if row["pres"][j] > 0:
                        for f, m, _ in _slot_margins(row, j):
                            if f == "args": min_args = m if min_args is None else min(min_args, m)
                else:
                    ok = ok and bool((row["dig"][j].argmax(-1) == vg["digits"][i, j]).all())
                if not ok: wrong.append({"slot": j, "ftype": "rel" if rel else "given", "gold": {k: v for k, v in r["factors"][j].items() if k != "spans"} if j < len(r["factors"]) else None})
            out.append({"row": i, "src_idx": r["gen"].get("src_idx"), "text": r["text"], "n_present": n_p, "n_wrong": len(wrong), "frac_wrong": len(wrong) / max(n_p, 1),
                        "min_args_margin": None if min_args is None else float(min_args), "kind_slot": atlas.classes[int(cs[bi])], "kind_token": atlas.classes[int(ct[bi])], "wrong": wrong,
                        "has_spans": any(f.get("spans") for f in r["factors"])})
        if s0 % 400 == 0: print(f"[queue] {s0 + len(sl)}/{N}", flush=True)
    out.sort(key=lambda q: (-q["frac_wrong"], q["min_args_margin"] if q["min_args_margin"] is not None else 1e9))
    path = os.environ.get("PQ_OUT", ".cache/annotation_queue_v1.jsonl")
    with open(path, "w") as f:
        for q in out: f.write(json.dumps(q) + "\n")
    fw = np.array([q["frac_wrong"] for q in out]); print(f"[queue] wrote {path}: {len(out)} rows; rows with any wrong slot {np.mean(fw > 0):.3f}; >= half wrong {np.mean(fw >= 0.5):.3f}; rows with spans {np.mean([q['has_spans'] for q in out]):.3f}")
    import collections; print("[queue] kinds (slot chart) among >= half wrong:", collections.Counter(q["kind_slot"] for q in out if q["frac_wrong"] >= 0.5).most_common(6))
    print("[queue] wrong slots by type among all wrong:", collections.Counter(w["ftype"] for q in out for w in q["wrong"]).most_common())


if __name__ == "__main__":
    _main()
