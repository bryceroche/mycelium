"""loop_val.py — THE LOOP-ENGAGED VAL (2026-08-31). _quick_val never runs
the breath loop (no slot_mask — the loop-free-val finding); this reader
computes the SAME fac-exact criterion on the masked two-pass forward, so
organs are measured OPERATING, not just via weight-shaping. Env: LV_CKPT;
mode via ALG_* envs of the caller (SC_EVAL/ALG_SHELF_CIRCLE for seal).
"""
import os, sys
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
from phase1_algebra_head import (build_params, forward, load_alg,
                                 build_slot_masks, L_FAC)
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

# THE JIT'D READ FORWARD (2026-09-08; door ALG_JIT_READ, the module
# mycelium/jit_read.py). With the door UNSET, `_rf(forward, ...)` IS
# `forward(...)` — the same call with the same arguments, `keys`
# dropped on the floor — so this file is byte-inert until the door
# opens. With it set, each pass runs from a captured graph keyed by
# (which ports are fed, which outputs are asked for, the batch shape,
# the value of EVERY env name the head's source reads — SC_EVAL among
# them, per THE UNLIT STOVE — and the identity of the param dict).
# Weights swap in place (p[k].assign), so a checkpoint swap re-uses
# the same graph; a mode change never can.
from mycelium.jit_read import read_forward as _rf

# REFACTOR (2026-09-08, scripts/read_batch.py): the module body moved
# into atlas_tables()/read()/main() with ZERO change to any computation
# — same statements, same order, same arithmetic, same print. Standalone
# stdout is byte-identical (proven on CPU before this patch was offered).
# WHY: every read was a fresh process (trunk + kernel compile per read);
# read() takes an already-built param dict so N checkpoints share ONE
# process via the loop_val idiom (assign-in-place, then re-run).
_XPV = int(os.environ.get("ALG_MH_XPRIOR", "0"))
assert not _XPV or int(os.environ.get("ALG_MH_ATLAS", "0")), \
    "ALG_MH_XPRIOR requires ALG_MH_ATLAS=1 (loud, never dark)"
_ATL_CACHE = None


def atlas_tables(vs):
    """The module-level atlas block, VERBATIM, cached per process:
    the tables depend only on the fixture + env, never on the
    checkpoint, so N checkpoints in one process share them.
    Returns (_ATAB, _AIDX, _atl, _acls)."""
    global _ATL_CACHE
    if _ATL_CACHE is not None:
        return _ATL_CACHE
    # MASK HEAD round 2 (apply_mass_thread.py, 2026-09-05): the read
    # legs light the same two ports the trainer lit — toggled by the
    # SAME envs the chain sets per arm (trained-env law). Atlas loads
    # via the research-manifest loud door; env set + file missing =
    # hard error (no silent dark ports).
    _ATAB = _AIDX = _atl = _acls = None
    if int(os.environ.get("ALG_MH_ATLAS", "0")):
        from mycelium.step_atlas import load_atlas, atlas_class
        _atl = load_atlas(
            os.environ.get("MH_ATLAS", ".cache/step_atlas_current.npz"),
            manifest_path=os.environ.get("MH_ATLAS_MANIFEST",
                                         ".cache/RESEARCH_MANIFEST.json"))
        _acls = {c: i for i, c in enumerate(_atl["classes"])}
        _tab = np.ascontiguousarray(
            _atl["means"].transpose(1, 0, 2)).astype(np.float32)
        _ATAB = np.concatenate(
            [_tab, np.zeros((1,) + _tab.shape[1:], np.float32)])
        _AIDX = np.array(
            [_acls.get(atlas_class(s.get("gen")), len(_acls))
             for s in vs], np.int64)
        if _XPV:
            from mycelium.step_atlas import cross_prior
            assert _atl.get("nl_means") is not None, \
                ("ALG_MH_XPRIOR needs the PAIRED atlas (nl chart) — "
                 "re-mine with the paired miner")
    _ATL_CACHE = (_ATAB, _AIDX, _atl, _acls)
    return _ATL_CACHE


def read(ckpt, data=None, p=None):
    """loop_val's read, VERBATIM. data = load_alg("test") tuple and
    p = build_params(0) may be handed in already built (read_batch);
    None means build them here, exactly as the standalone did.
    Returns (n_ok, n_tot) — main() does the printing."""
    vs, vst, vtk, vg, vse = load_alg("test") if data is None else data
    if p is None:
        p = build_params(0)
    sd = safe_load(ckpt)
    assert set(sd.keys()) == set(p.keys()), \
        (sorted(set(sd) - set(p))[:4], sorted(set(p) - set(sd))[:4])
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    _ATAB, _AIDX, _atl, _acls = atlas_tables(vs)
    # THE JIT'D READ's output declarations (door ALG_JIT_READ): a
    # captured graph's return value is fixed at capture, so the keys a
    # pass consumes must be named before the graph exists. These two
    # tuples are built from the SAME branch conditions the eager
    # realize sets below use — ALT2/LV_NOFACT for the open pass's fact
    # block, _XPV for nl0, "h_dup" in p for dup — so the door never
    # asks forward() for an output the eager path would not have
    # realized. With the door shut they are unused.
    _jk_open = (("fat", "args", "res")
                + (("pres", "ftype", "op", "dig", "dup")
                   if int(os.environ.get("ALG_ALT2", "0"))
                   and not int(os.environ.get("LV_NOFACT", "0")) else ())
                + (("nl0",) if _XPV else ()))
    _jk_masked = (("pres", "ftype", "op", "islit", "dig", "args", "res")
                  + (("dup",) if "h_dup" in p else ()))
    n_ok = n_tot = 0
    _PS = [] if os.environ.get("LV_PER_SLOT") else None   # THE PAIRED READ (2026-09-12): per-slot outcomes to an npz
    for s0 in range(0, len(vs), 8):
        sl = np.arange(s0, min(s0 + 8, len(vs)))
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        ts = Tensor(np.ascontiguousarray(vst[sl_p]), dtype=dtypes.half)   # perf audit #4: half feed, upcast in-graph
        tk = Tensor(vtk[sl_p].astype(np.float32), dtype=dtypes.float)
        se = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)
        o0 = _rf(forward, p, ts, tk, se, keys=_jk_open)
        onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
        mk = build_slot_masks(onp0, vse[sl_p].astype(np.int32))
        fact_t = mass_t = None
        if int(os.environ.get("ALG_ALT2", "0")) \
                and not int(os.environ.get("LV_NOFACT", "0")):
            # ALTERNATOR V2 fact-fed read (2026-09-01): live facts from this
            # checkpoint's own pass-1 parse — same convention as _quick_val
            from phase1_algebra_head import alt2_fact_buf, K_VARS
            _ka = ("pres", "ftype", "op", "dig") + \
                (("dup",) if "dup" in o0 else ())
            _oa = {**onp0, **{k: o0[k].realize().numpy() for k in _ka}}
            _nv = np.array([vs[int(i)].get("n_vars", K_VARS) for i in sl_p])
            _ma = np.array([vs[int(i)].get("m", 0) for i in sl_p])
            _mo = (np.zeros((len(sl_p), K_VARS), np.float32)
                   if int(os.environ.get("ALG_MH_MASS", "0")) else None)
            fb = alt2_fact_buf(_oa, vse[sl_p].astype(np.int32), _nv, _ma,
                               mass_out=_mo)
            fact_t = Tensor(fb, dtype=dtypes.float)
            if _mo is not None:
                mass_t = Tensor(np.clip(_mo / 301.0, 0.0, 1.0)
                                [:, :, None].astype(np.float32),
                                dtype=dtypes.float)
        _lvai = _AIDX[sl_p].copy() if _AIDX is not None else None
        if _XPV and _lvai is not None:
            # THE CROSS-ATLAS PRIOR (apply_cross_prior.py): retrieval
            # off the pass-1 breath-0 NL state (the tap) — mode 1 =
            # unknown rows only, mode 2 = every row (deployable)
            from mycelium.step_atlas import cross_prior
            _xlv, _ = cross_prior(_atl, o0["nl0"].realize().numpy(),
                                  return_traj=False)
            _rlv = ((_lvai == len(_acls)) if _XPV == 1
                    else np.ones(len(_lvai), bool))
            _lvai = np.where(_rlv, _xlv, _lvai)
        _mha_t = (Tensor(_ATAB[_lvai], dtype=dtypes.float)
                  if _ATAB is not None else None)
        o = _rf(forward, p, ts, tk, se, keys=_jk_masked,
                slot_mask=Tensor(mk, dtype=dtypes.float),
                fact_buf=fact_t, mh_mass=mass_t, mh_atlas_traj=_mha_t)
        onp = {k: o[k].realize().numpy() for k in
               (("pres", "ftype", "op", "islit", "dig", "args", "res")
                + (("dup",) if "h_dup" in p else ()))}
        for bi, i in enumerate(sl):
            i = int(i)
            for j in range(L_FAC):
                if vg["presence"][i, j] < 0.5:
                    continue
                n_tot += 1
                ok = (onp["pres"][bi, j] > 0)
                ok &= int(onp["ftype"][bi, j].argmax()) == vg["ftype"][i, j]
                ok &= int(onp["res"][bi, j].argmax()) == vg["res"][i, j]
                if vg["ftype"][i, j] == 0:
                    ok &= int(onp["op"][bi, j].argmax()) == vg["op"][i, j]
                    gset = set(np.where(vg["args"][i, j] > .5)[0].tolist())
                    if len(gset) == 1 and "dup" in onp:
                        ok &= bool(onp["dup"][bi, j] > 0)
                        ok &= int(np.argmax(onp["args"][bi, j])) in gset
                    else:
                        top2 = set(np.argsort(-onp["args"][bi, j])[:2].tolist())
                        ok &= top2 == gset
                else:
                    ok &= bool((onp["dig"][bi, j].argmax(-1) ==
                                vg["digits"][i, j]).all())
                n_ok += ok
                if _PS is not None:
                    _PS.append((i, j, bool(ok)))
    if _PS is not None:
        np.savez(os.environ["LV_PER_SLOT"], rows=np.array([r for r, _, _ in _PS]), slots=np.array([c for _, c, _ in _PS]), ok=np.array([o for _, _, o in _PS]))
    return n_ok, n_tot


def main(ckpt=None, data=None, p=None):
    """Unchanged standalone behaviour: LV_CKPT, one read, one
    printed line. The keyword arguments only let read_batch hand
    in an already-built fixture/param dict and the checkpoint of
    the moment, so the PRINT stays in the organ (one f-string,
    one place) instead of being copied into the caller."""
    ckpt = os.environ["LV_CKPT"] if ckpt is None else ckpt
    n_ok, n_tot = read(ckpt, data=data, p=p)
    print(f"[loop-val] {ckpt} mode="
          f"SC={os.environ.get('ALG_SHELF_CIRCLE','0')}/EVAL={os.environ.get('SC_EVAL','-')} "
          f"fac-exact={n_ok / max(n_tot, 1):.4f} (n={n_tot})")
    return n_ok, n_tot


if __name__ == "__main__":
    main()
