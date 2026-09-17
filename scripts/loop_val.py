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


_LEX = int(os.environ.get("LV_LEXICON", "0"))
_LEX_STATS = {"spans": 0, "taken": 0, "rows": 0}


def _lex_apply(onp, fat, sl, sl_p, vs):
    import phase1_algebra_head as _H
    from mycelium.lexicon import match, span_tokens
    from mycelium.loop_bridge import Bridge
    from tokenizers import Tokenizer
    global _LEX_TOK
    try:
        _LEX_TOK
    except NameError:
        _LEX_TOK = Tokenizer.from_file(_H.TOKENIZER_JSON)
    B = len(sl_p); T = fat.shape[-1]
    br = Bridge(fat, np.zeros((B, T), np.int32))          # the correspondence only (source tokens); no sentences needed
    for bi in range(len(sl)):
        text = vs[int(sl[bi])]["text"]; ms = match(text)
        if not ms:
            continue
        _LEX_STATS["rows"] += 1
        _enc = _LEX_TOK.encode(text)
        spans = span_tokens(ms, list(_enc.offsets), _H.T_ALG)
        vspans = [(set(toks), val) for toks, role, val in spans if role == "value"]
        # the numerals compete on equal footing (mode 2): a digit token is a candidate whose
        # win leaves the head's own decode alone — the lexicon never overrides a numeral read
        numtoks = [({ti}, None) for ti, tid in enumerate(_enc.ids[:_H.T_ALG]) if _LEX_TOK.decode([tid]).strip().isdigit()]
        numvals = {int(x) for x in __import__("re").findall(r"\d+", text)}
        _LEX_STATS["spans"] += len(vspans)
        if not vspans:
            continue
        row = {k: onp[k][bi] for k in onp}
        for f in _H._decode_slots(row):
            if f.get("ftype") != "given":
                continue
            j = f["_slot"]; chosen = None
            # THE NUMERAL GUARD (mode 2, 2026-09-15): a decode that already matches a numeral
            # present in the text is a numeral READ — the lexicon never overrides it
            if _LEX >= 2 and f.get("value") is not None and int(f["value"]) in numvals:
                continue
            if _LEX == 1:                       # mode 1: the source token (argmax) inside a span
                t0 = br.source_token(bi, j)
                chosen = next((val for toks, val in vspans if t0 in toks), None)
            else:                               # mode 2: THE BRIDGE's mass — the span this slot reads most
                cands = vspans + numtoks
                mass = [fat[bi, j, sorted(toks)].sum() for toks, _ in cands]
                k = int(np.argmax(mass))
                if mass[k] > 0:
                    chosen = cands[k][1]        # None when a numeral token wins: the decode stands
            if chosen is not None and float(chosen).is_integer() and 0 <= int(chosen) < 10 ** onp["dig"].shape[-2]:
                digs = [int(c) for c in str(int(chosen)).zfill(onp["dig"].shape[-2])]
                onp["dig"][bi, j] = 0.0
                for di, d in enumerate(digs):
                    onp["dig"][bi, j, di, d] = 20.0
                _LEX_STATS["taken"] += 1


_NUMTAB = None
_NUMSPOT_BETA = float(os.environ.get("LV_NUMSPOT", "0") or 0)
_NS_STATS = {"slots": 0}


def _ns_setup():
    import phase1_algebra_head as _H
    from tokenizers import Tokenizer
    global _LEX_TOK
    try:
        _LEX_TOK
    except NameError:
        _LEX_TOK = Tokenizer.from_file(_H.TOKENIZER_JSON)


def _lex_substitute(st_np, tk_np, sl, sl_p, vs):
    import phase1_algebra_head as _H
    from mycelium.lexicon import match, span_tokens
    from tokenizers import Tokenizer
    global _LEX_TOK, _NUMTAB
    try:
        _LEX_TOK
    except NameError:
        _LEX_TOK = Tokenizer.from_file(_H.TOKENIZER_JSON)
    if _NUMTAB is None:
        _z = np.load(".cache/numeral_table.npz"); _NUMTAB = (_z["mean"], _z["count"])
    mean, cnt = _NUMTAB
    for bi in range(len(sl)):
        text = vs[int(sl[bi])]["text"]; ms = match(text)
        if not ms:
            continue
        _LEX_STATS["rows"] += 1
        enc = _LEX_TOK.encode(text)
        # THE CENSUS RULE (2026-09-15): a row that writes its quantities as digits does not write one
        # of them as a bare small cardinal ("one of them", "three times a week") — wild's givens are
        # numerals 99%; bare cardinals < 10 are substituted only in rows with NO digit numeral.
        _has_digits = bool(__import__("re").search(r"\d", text))
        for toks, role, val in span_tokens(ms, list(enc.offsets), _H.T_ALG):
            if role != "value" or not float(val).is_integer() or not (0 <= int(val) < len(cnt)) or cnt[int(val)] < 5:
                continue
            if _has_digits and int(val) < 10 and len(toks) == 1:
                continue                                   # a bare small cardinal in a digit row: prose, not a quantity
            if any(_LEX_TOK.decode([enc.ids[t]]).strip().isdigit() for t in toks):
                continue                                   # a digit token: never touched
            toks = sorted(toks); _LEX_STATS["spans"] += 1
            st_np[bi, toks[0]] = mean[int(val)]
            for t in toks[1:]:
                tk_np[bi, t] = 0.0
            _LEX_STATS["taken"] += 1


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
    _PS = [] if os.environ.get("LV_PER_SLOT") else None
    import collections as _c; _FIELDS = _c.defaultdict(lambda: [0, 0]) if os.environ.get("LV_FIELDS") else None   # THE PAIRED READ (2026-09-12): per-slot outcomes to an npz
    for s0 in range(0, len(vs), 8):
        sl = np.arange(s0, min(s0 + 8, len(vs)))
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        _st_np = np.ascontiguousarray(vst[sl_p]); _tk_np = vtk[sl_p].astype(np.float32)
        if _LEX == 3:
            # THE SUBSTITUTION ROAD (2026-09-15): a matched WORD span's first token takes the mean
            # trunk state of its numeral (.cache/numeral_table.npz) and the span's other tokens are
            # masked out — the head reads "a dozen" the way it reads "12" and allocates the slot
            # itself. Digit tokens are never touched. Zero parameters; the same road at train time.
            _st_np = _st_np.copy(); _tk_np = _tk_np.copy()
            _lex_substitute(_st_np, _tk_np, sl, sl_p, vs)
        ts = Tensor(_st_np, dtype=dtypes.half)   # perf audit #4: half feed, upcast in-graph
        tk = Tensor(_tk_np, dtype=dtypes.float)
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
        if _NUMSPOT_BETA:
            # THE NUMERAL SPOTLIGHT (2026-09-15): digit tokens (the census's class) get +beta on the
            # token scores of the slots pass 1 typed as GIVEN — a token-born certificate through the
            # wheel's own road. Constant across breaths; read time only.
            import phase1_algebra_head as _H
            _ns_setup(); _B = len(sl_p); _T = vse.shape[1]
            _nsb = np.zeros((_B, 1, _H.L_TOT, _T), np.float32)
            _gid = 1   # decode()'s ftype id for "given" (asserted from the source by the apply)
            for bi in range(_B):
                enc = _LEX_TOK.encode(vs[int(sl_p[bi])]["text"]); dig = np.zeros(_T, bool)
                for t_, tid in enumerate(enc.ids[:_T]):
                    if _LEX_TOK.decode([tid]).strip().isdigit(): dig[t_] = True
                for j in range(L_FAC):
                    if int(_oa["ftype"][bi, j].argmax()) == _gid and _oa["pres"][bi, j] > 0:
                        _nsb[bi, 0, j, dig] = _NUMSPOT_BETA
            _H._NUMSPOT = _nsb; _NS_STATS["slots"] += int((_nsb[:, 0, :, :].max(-1) > 0).sum())
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
        if _NUMSPOT_BETA:
            import phase1_algebra_head as _H2; _H2._NUMSPOT = None
        if _LEX in (1, 2):
            # THE LEXICON ROAD (2026-09-15, word given; mycelium/lexicon.py, the
            # symbolic convolution): a GIVEN slot whose source token (the bridge's
            # argmax read at intake) lies inside a matched value span takes the
            # entry's digits. Parameter-free; neural proposes, the lexicon disposes.
            _lex_apply(onp, onp0["fat"], sl, sl_p, vs)
        for bi, i in enumerate(sl):
            i = int(i)
            for j in range(L_FAC):
                if vg["presence"][i, j] < 0.5:
                    continue
                n_tot += 1
                f_pres = bool(onp["pres"][bi, j] > 0)
                f_ftype = int(onp["ftype"][bi, j].argmax()) == vg["ftype"][i, j]
                f_res = int(onp["res"][bi, j].argmax()) == vg["res"][i, j]
                ok = f_pres and f_ftype and f_res
                f_op = f_args = f_dig = None
                if vg["ftype"][i, j] == 0:
                    f_op = int(onp["op"][bi, j].argmax()) == vg["op"][i, j]
                    gset = set(np.where(vg["args"][i, j] > .5)[0].tolist())
                    if len(gset) == 1 and "dup" in onp:
                        f_args = bool(onp["dup"][bi, j] > 0) and int(np.argmax(onp["args"][bi, j])) in gset
                    else:
                        top2 = set(np.argsort(-onp["args"][bi, j])[:2].tolist())
                        f_args = top2 == gset
                    ok = ok and f_op and f_args
                else:
                    f_dig = bool((onp["dig"][bi, j].argmax(-1) ==
                                  vg["digits"][i, j]).all())
                    ok = ok and f_dig
                n_ok += ok
                if _FIELDS is not None:   # LV_FIELDS=1: per-field and per-slot-position tallies (the fit-read instrument, 2026-09-16)
                    for k, v in (("pres", f_pres), ("ftype", f_ftype), ("res", f_res), ("op", f_op), ("args", f_args), ("dig", f_dig), ("exact", ok)):
                        if v is not None: _FIELDS[k][0] += int(v); _FIELDS[k][1] += 1
                    _FIELDS["slot%02d" % j][0] += int(ok); _FIELDS["slot%02d" % j][1] += 1
                if _PS is not None:
                    _PS.append((i, j, bool(ok)))
    if _FIELDS is not None:
        print("[fields] " + " ".join(f"{k}={v[0]/max(v[1],1):.3f}({v[1]})" for k, v in _FIELDS.items() if not k.startswith("slot")), flush=True)
        print("[slots]  " + " ".join(f"{k[4:]}:{v[0]/max(v[1],1):.2f}({v[1]})" for k, v in sorted(_FIELDS.items()) if k.startswith("slot")), flush=True)
    if _PS is not None:
        if _NUMSPOT_BETA: print(f"[numspot] beta {_NUMSPOT_BETA}: given slots spotlit {_NS_STATS['slots']}", flush=True)
        if _LEX: print(f"[lexicon] rows with matches {_LEX_STATS['rows']}, value spans {_LEX_STATS['spans']}, given slots taken {_LEX_STATS['taken']}", flush=True)
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
