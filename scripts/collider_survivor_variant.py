"""collider_survivor_variant.py — EXPLORATORY (2026-09-07, after the collider
KILL): a different rule, not the pinned one. When the ORIGINAL reading is
UNDERDETERMINED and EXACTLY ONE single-slot rival makes the query UNIQUE,
is that rival's slot correct? Reuses collider_read's forward, decode,
collide_item and grader verbatim; grades only afterwards. Read-only."""
import os, sys
os.environ.setdefault("DEV", "PCI+AMD")
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
import collider_read as C

def main():
    from phase1_algebra_head import build_params, forward, load_alg
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    vs, vst, vtk, vg, vse = load_alg("test")
    n = len(vs)
    p = build_params(0); sd = safe_load(C.CKPT)
    for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    tot = ok = was_ok = nogold = 0; by_kind = {}
    for s0 in range(0, n, C.CL_B):
        sl = np.arange(s0, min(s0 + C.CL_B, n)); pad = C.CL_B - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        o = forward(p, Tensor(vst[sl_p].astype(np.float32), dtype=dtypes.float),
                    Tensor(vtk[sl_p].astype(np.float32), dtype=dtypes.float),
                    Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int))
        kset = ["pres", "ftype", "op", "args", "res", "dig"] + [k for k in ("dup", "sgn", "dargs") if k in o]
        onp = {k: o[k].realize().numpy() for k in kset}; has_dup = "dup" in onp
        D = C.decode_batch(onp, C.THETA)
        for bi, i in enumerate(sl):
            i = int(i); row = vs[i]
            per, _ = C.grade_item(onp, bi, vg, i, has_dup)
            r = C.collide_item(D, bi, int(row.get("n_vars", C.K_VARS)), int(row.get("m", 0)),
                               int(row["query_var"]), C.THETA, C.MARGIN, C.MAXALT)
            if r["orig_label"] == "UNIQUE": continue
            uniq = [a for a in r["alts"] if a.get("label") == "UNIQUE"]
            if len(uniq) != 1: continue
            a = uniq[0]; j = a["slot"]
            w, now = C.grade_flip(per, C.slot_rec(onp, bi, j, has_dup), a["ov"], vg, i, j, has_dup)
            tot += 1
            if w is None: nogold += 1; continue
            ok += int(now); was_ok += int(w)
            by_kind.setdefault(a["kind"], [0, 0]); by_kind[a["kind"]][0] += int(now); by_kind[a["kind"]][1] += 1
    print(f"[survivor-variant] {C._FNAME}: single-survivor rivals={tot} (no-gold-slot {nogold}); "
          f"rival slot correct {ok}/{tot-nogold} = {ok/max(tot-nogold,1):.3f}; the ORIGINAL slot was already correct in {was_ok}/{tot-nogold}; "
          f"by kind {{k: correct/n}} = {by_kind}", flush=True)

if __name__ == "__main__":
    main()
