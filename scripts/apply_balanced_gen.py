"""apply_balanced_gen.py — THE BALANCED GENERATION (2026-09-09, the word).
Four doors, all env-gated, ALL unset = the forward is untouched (eq gate):
  ALG_PRUNE=pforms,s4,fednl0,lane2  build_params DROPS the keys of the
      organs the severance ladder found dead (4.9M); the forward already
      guards three of them by key; station 4 gets a key guard here.
  ALG_NB2=1   the second notebook lane REBORN AS A ROAD: own ink nb2_sil
      (ZERO-INIT: silent birth, the station-3 idiom), own query nb2_nq,
      shelf rows 8..15, NO GAIN (the mandatory-road law; lane 2 of the fed
      package died behind fed_nb_g = 0.0004).
  ALG_ALT5=1  STATION 5: a third slots<-tokens bank attention in station
      3's form (the wild organ: 5.4 points per 1.1M), ZERO-INIT output.
  ALG_BAL_COOK=<share>  THE BALANCED COOKER: on a stable index-hash share
      of rows the OLD roads are sealed — lane 1's read x0, the main bank's
      and station 3's slot->token attention flattened — so lane 2 and
      station 5 are the ONLY memory and grounding on those rows (the
      pressure-cooker form). Val/reads push BC_EVAL (open). Does not
      compose with ALG_TOK_COOK (asserted).
Sever doors gain "s5" and "nb2" (the ladder reads on the new organs).
Idempotent: refuses a target that already carries the doors."""
import sys
P = "scripts/phase1_algebra_head.py"
s = open(P).read()
if "_PRUNE_ORGANS" in s:
    print("apply_balanced_gen: target already carries the doors — refusing"); sys.exit(2)
assert "_SEVER_ORGANS" in s, "apply_sever_doors.py must be applied first"
n0 = len(s)

def rep(old, new, count=1):
    global s
    c = s.count(old)
    assert c == count, (c, old[:90])
    s = s.replace(old, new)

# B0 — the dials, beside the sever registry
rep('_SEVER_ORGANS = frozenset(("notebook", "garage", "s3", "s4", "mixer",\n'
    '                           "fedtwin", "altv0", "ffn", "pforms"))\n',
    '_SEVER_ORGANS = frozenset(("notebook", "garage", "s3", "s4", "mixer",\n'
    '                           "fedtwin", "altv0", "ffn", "pforms",\n'
    '                           "s5", "nb2"))   # the balanced generation\n')
rep('assert _SEVER <= _SEVER_ORGANS, \\\n'
    '    f"ALG_SEVER unknown organ(s) {sorted(_SEVER - _SEVER_ORGANS)}; " \\\n'
    '    f"known: {sorted(_SEVER_ORGANS)}"\n',
    'assert _SEVER <= _SEVER_ORGANS, \\\n'
    '    f"ALG_SEVER unknown organ(s) {sorted(_SEVER - _SEVER_ORGANS)}; " \\\n'
    '    f"known: {sorted(_SEVER_ORGANS)}"\n'
    '# THE BALANCED GENERATION (apply_balanced_gen.py, 2026-09-09): prune\n'
    '# the dead-by-severance organs at birth; regrow where the removal cost\n'
    '# per parameter is highest (lane 2 as a road; station 5); the balanced\n'
    '# cooker makes the new organs mandatory on a share of rows.\n'
    '_PRUNE_ORGANS = frozenset(("pforms", "s4", "fednl0", "lane2"))\n'
    '_PRUNE = frozenset(x for x in os.environ.get("ALG_PRUNE", "").split(",") if x)\n'
    'assert _PRUNE <= _PRUNE_ORGANS, \\\n'
    '    f"ALG_PRUNE unknown organ(s) {sorted(_PRUNE - _PRUNE_ORGANS)}"\n'
    'ALG_NB2 = int(os.environ.get("ALG_NB2", "0"))\n'
    'ALG_ALT5 = int(os.environ.get("ALG_ALT5", "0"))\n'
    'ALG_BAL_COOK = float(os.environ.get("ALG_BAL_COOK", "0"))\n'
    '\n'
    '\n'
    'def _bal_cook_v():\n'
    '    """THE BALANCED COOKER\'s per-row seal value: None (nothing sealed —\n'
    '    today\'s path), or the (B,1,1,1) `_BCV` buffer do_train arms (the\n'
    '    _PCV/_MCV/_TCV idiom). BC_EVAL non-empty forces OPEN (val/read)."""\n'
    '    if os.environ.get("BC_EVAL", "") or ALG_BAL_COOK <= 0.0:\n'
    '        return None\n'
    '    _v = globals().get("_BCV")\n'
    '    return None if _v is None else _v.reshape(-1, 1, 1, 1)\n')

# B1 — build_params tail: prune, then grow
rep('              % (POLAR_D, _dtxt, POLAR_EM, _etxt, H_W - _nc, _nc),\n'
    '              flush=True)\n'
    '    return p\n',
    '              % (POLAR_D, _dtxt, POLAR_EM, _etxt, H_W - _nc, _nc),\n'
    '              flush=True)\n'
    '    # THE BALANCED GENERATION (apply_balanced_gen.py, 2026-09-09)\n'
    '    if _PRUNE:\n'
    '        _drop = [k for k in p if\n'
    '                 ("pforms" in _PRUNE and k.startswith("fed_pf_"))\n'
    '                 or ("s4" in _PRUNE and k.startswith("alt21_W_b"))\n'
    '                 or ("fednl0" in _PRUNE and k == "fed_nl0_w")\n'
    '                 or ("lane2" in _PRUNE and k in ("fed_sil2", "fed_nb_g"))]\n'
    '        _np_ = 0\n'
    '        for k in _drop:\n'
    '            _np_ += int(np.prod(p[k].shape)); del p[k]\n'
    '        print(f"[prune] ALG_PRUNE={sorted(_PRUNE)}: dropped {len(_drop)} "\n'
    '              f"keys / {_np_} params (dead by severance, ledger 2026-09-09)",\n'
    '              flush=True)\n'
    '    if ALG_NB2:\n'
    '        assert ALG_NOTEBOOK and NB_PERSLOT and "fed_sil2" not in p, (\n'
    '            "ALG_NB2 needs the per-slot notebook and lane 2 of the fed "\n'
    '            "package PRUNED (ALG_PRUNE=lane2): one second lane, as a road")\n'
    '        assert NB_ROWS >= 16, "ALG_NB2 stamps rows 8..15 (NB_ROWS=16)"\n'
    '        p["nb2_sil"] = t(np.zeros((H_W, H_W)))   # ZERO INK: silent birth\n'
    '        p["nb2_nq"] = t(rng.randn(H_W, H_W) / math.sqrt(H_W))\n'
    '    if ALG_ALT5:\n'
    '        assert "alt21_attn_wo" in p, "ALG_ALT5 rides the ALT21 flow (station 3)"\n'
    '        p["alt5_attn_wq"], p["alt5_attn_wq_b"] = lin(H_W, H_W)\n'
    '        p["alt5_attn_wk"], p["alt5_attn_wk_b"] = lin(H_W, H_W)\n'
    '        p["alt5_attn_wv"], p["alt5_attn_wv_b"] = lin(H_W, H_W)\n'
    '        p["alt5_attn_wo"] = t(np.zeros((H_W, H_W)))   # ZERO: silent birth\n'
    '        p["alt5_attn_wo_b"] = t(np.zeros(H_W))\n'
    '    return p\n')

# B2 — lane 2 birth at kb == 1
rep('        _nb = [(cur @ p["W_sil"]) if NB_PERSLOT\n'
    '               else (_fed_core(cur).mean(1) @ p["W_sil"])]   # sharp vs blurred\n',
    '        _nb = [(cur @ p["W_sil"]) if NB_PERSLOT\n'
    '               else (_fed_core(cur).mean(1) @ p["W_sil"])]   # sharp vs blurred\n'
    '        if ALG_NB2 and "nb2_sil" in p:\n'
    '            state["nbb"] = [cur @ p["nb2_sil"]]   # lane 2 as a road\n')

# B3 — the balanced seal value, computed before the notebook read
rep('    q_extra = cur + p["breath_emb"][kb].reshape(1, 1, -1)\n'
    '    if _CENSUS is not None:\n'
    '        _CENSUS.append((kb, "state", cur.realize().numpy()))\n',
    '    _bal_v = _bal_cook_v()      # THE BALANCED COOKER (None = open)\n'
    '    q_extra = cur + p["breath_emb"][kb].reshape(1, 1, -1)\n'
    '    if _CENSUS is not None:\n'
    '        _CENSUS.append((kb, "state", cur.realize().numpy()))\n')

# B4 — lane 1 sealed on cooked rows; lane 2 read (per-slot branch)
rep('            if "notebook" in _SEVER:\n'
    '                _rd = _rd * 0.0\n'
    '            q_extra = q_extra + _rd           # (B, L, H) — no blur\n',
    '            if "notebook" in _SEVER:\n'
    '                _rd = _rd * 0.0\n'
    '            if _bal_v is not None:            # THE BALANCED COOKER:\n'
    '                _rd = _rd * (1.0 - _bal_v.reshape(B, 1, 1))   # lane 1 sealed\n'
    '            q_extra = q_extra + _rd           # (B, L, H) — no blur\n'
    '            if ALG_NB2 and "nb2_sil" in p:\n'
    '                # LANE 2 AS A ROAD (the balanced generation): own ink,\n'
    '                # own query, rows 8..15 of the stamp alphabet, NO GAIN.\n'
    '                _nbb = state["nbb"]\n'
    '                _qb = cur @ p["nb2_nq"]\n'
    '                _scb = (_qb @ _nb_st[8:8 + len(_nbb)].transpose(1, 0)) \\\n'
    '                    / math.sqrt(H_W)\n'
    '                if NB_FOCAL > 0:\n'
    '                    _scb = _scb * NB_FOCAL\n'
    '                _atb = _scb.softmax(-1)\n'
    '                _rdb = sum(_atb[:, :, _jb:_jb + 1] * _nbb[_jb]\n'
    '                           for _jb in range(len(_nbb)))\n'
    '                if "nb2" in _SEVER:\n'
    '                    _rdb = _rdb * 0.0\n'
    '                q_extra = q_extra + _rdb\n'
    '                if _CENSUS is not None:\n'
    '                    _CENSUS.append((kb, "nb2", _rdb.realize().numpy()))\n')
rep('            _rd = sum(_at[:, j:j + 1] * _nb[j] for j in range(len(_nb)))\n'
    '            if "notebook" in _SEVER:\n'
    '                _rd = _rd * 0.0\n'
    '            q_extra = q_extra + _rd.reshape(B, 1, -1)\n',
    '            _rd = sum(_at[:, j:j + 1] * _nb[j] for j in range(len(_nb)))\n'
    '            if "notebook" in _SEVER:\n'
    '                _rd = _rd * 0.0\n'
    '            assert _bal_v is None and not ALG_NB2, \\\n'
    '                "the balanced generation is per-slot (NB_PERSLOT=1) only"\n'
    '            q_extra = q_extra + _rd.reshape(B, 1, -1)\n')

# B5 — the grounding blend: the balanced cooker rides the token cooker's road
rep('    _tcv = _tok_cook_v()\n'
    '    _tgt = None\n'
    '    if _tcv is not None:\n',
    '    _tcv = _tok_cook_v()\n'
    '    _tgt = None\n'
    '    if _bal_v is not None:\n'
    '        # THE BALANCED COOKER on the grounding: sealed rows read the\n'
    '        # tokens FLAT through the main bank and station 3 (the token\n'
    '        # seal\'s field, per row, at the weights\' source — the cooker\n'
    '        # blend the token cooker built); station 5 is exempt: on those\n'
    '        # rows it is the only aimed grounding. Does not compose with\n'
    '        # the token cooker (one aim per road).\n'
    '        assert _tcv is None, "ALG_BAL_COOK and ALG_TOK_COOK do not compose"\n'
    '        _tgt = _tok_flat(tokmask, B).reshape(B, 1, -1) \\\n'
    '            .expand(B, L_TOT, tokmask.shape[-1]).contiguous()\n'
    '        _tcv = _bal_v\n'
    '    elif _tcv is not None:\n')

# B6 — station 5 (before station 4), station 4 key-guarded
rep('    if int(os.environ.get("ALG_ALT21", "0")) and "alt21_W_bo" in p:\n'
    '        # ALTERNATOR v2.1 STATIONS 3-4 (2026-09-02): the INTEGRATE\n',
    '    if int(os.environ.get("ALG_ALT21", "0")) and "alt21_attn_wo" in p:\n'
    '        # ALTERNATOR v2.1 STATIONS 3-4 (2026-09-02): the INTEGRATE\n')
rep('        _s21 = _s21 + _d21a              # exact zero at birth\n'
    '        # STATION 4: second slot-mixer over the SAME breathed mask\n',
    '        _s21 = _s21 + _d21a              # exact zero at birth\n'
    '        _d21c = None\n'
    '        if ALG_ALT5 and "alt5_attn_wo" in p:\n'
    '            # STATION 5 (the balanced generation, 2026-09-09): a THIRD\n'
    '            # slots<-tokens bank attention in station 3\'s form — the\n'
    '            # ladder read station 3 as the wild organ (0.054 per 1.1M).\n'
    '            # ZERO-INIT output (silent birth). Under the READ-TIME token\n'
    '            # seal it flattens with the other grounding roads; under the\n'
    '            # BALANCED cooker it is the exempt, mandatory road.\n'
    '            _qx5 = p["fq"].unsqueeze(0) + _s21 + (q_extra - cur)\n'
    '            _q5 = _qx5 @ p["alt5_attn_wq"] + p["alt5_attn_wq_b"]\n'
    '            _k5 = waist @ p["alt5_attn_wk"] + p["alt5_attn_wk_b"]\n'
    '            _v5 = waist @ p["alt5_attn_wv"] + p["alt5_attn_wv_b"]\n'
    '            _qh5 = _q5.reshape(B, L_TOT, N_HEADS, _hd21).permute(0, 2, 1, 3)\n'
    '            _kh5 = _k5.reshape(B, -1, N_HEADS, _hd21).permute(0, 2, 1, 3)\n'
    '            _vh5 = _v5.reshape(B, -1, N_HEADS, _hd21).permute(0, 2, 1, 3)\n'
    '            _sa5 = (_qh5 @ _kh5.transpose(-2, -1)) / math.sqrt(_hd21)\n'
    '            if _sync is not None:\n'
    '                _sa5 = _sa5 + _sync[0](kb)\n'
    '            if _rb7 is not None:\n'
    '                _sa5 = _sa5 + _rb7.unsqueeze(1) * p["r_gain"].reshape(1, 1, 1, 1)\n'
    '            _sa5 = _sa5.clip(-1e4, 1e4) + (1.0 - tokmask.reshape(B, 1, 1, -1)) * -1e4\n'
    '            _a5 = _sa5.softmax(-1)\n'
    '            if _tok_seal_on(kb) and int(os.environ.get("ALG_TOK_SEAL_S3", "1")):\n'
    '                _a5 = _a5 * 0.0 + _tok_flat(tokmask, B)   # the read-time seal\n'
    '            if _tgt is not None and _bal_v is None \\\n'
    '                    and int(os.environ.get("ALG_TOK_COOK_S3", "1")):\n'
    '                _a5 = _a5 * (1.0 - _tcv) + _tgt.reshape(B, 1, L_TOT, -1) * _tcv\n'
    '            _st5 = (_a5 @ _vh5).permute(0, 2, 1, 3).reshape(B, L_TOT, H_W)\n'
    '            _d21c = _st5 @ p["alt5_attn_wo"] + p["alt5_attn_wo_b"]\n'
    '            if "s5" in _SEVER:\n'
    '                _d21c = _d21c * 0.0\n'
    '            _s21 = _s21 + _d21c\n'
    '            if _CENSUS is not None:\n'
    '                _CENSUS.append((kb, "alt21_s5", _d21c.realize().numpy()))\n'
    '        # STATION 4: second slot-mixer over the SAME breathed mask\n')
# station 4 body -> key-guarded (re-indent)
i0 = s.index('        # STATION 4: second slot-mixer over the SAME breathed mask\n')
end_marker = ('        _d21b = (_sm21.softmax(-1) @ _bv21) @ p["alt21_W_bo"] \\\n'
              '            + p["alt21_W_bo_b"]\n')
i1 = s.index(end_marker, i0) + len(end_marker)
body = s[i0:i1]
assert body.count('\n') < 40, body.count('\n')
ind = ''.join(('    ' + ln if ln.strip() else ln) + '\n' for ln in body.rstrip('\n').split('\n'))
s = s[:i0] + '        if "alt21_W_bo" in p:   # station 4 (pruned = absent)\n' + ind + \
    '        else:\n            _d21b = None\n' + s[i1:]
rep('        if "s4" in _SEVER:\n'
    '            _d21b = _d21b * 0.0\n'
    '        h_slot = h_slot + _d21a + _d21b  # additive; zeros at birth\n',
    '        if "s4" in _SEVER and _d21b is not None:\n'
    '            _d21b = _d21b * 0.0\n'
    '        if _d21b is not None:\n'
    '            h_slot = h_slot + _d21a + _d21b  # additive; zeros at birth\n'
    '        else:\n'
    '            h_slot = h_slot + _d21a\n'
    '        if _d21c is not None:\n'
    '            h_slot = h_slot + _d21c          # station 5, additive\n')
rep('            _CENSUS.append((kb, "alt21_s4", _d21b.realize().numpy()))\n',
    '            if _d21b is not None:\n'
    '                _CENSUS.append((kb, "alt21_s4", _d21b.realize().numpy()))\n')

# B7 — lane 2 append per breath
rep('    if ALG_NOTEBOOK:\n'
    '        _nb.append((cur @ p["W_sil"]) if NB_PERSLOT\n'
    '                   else (_fed_core(cur).mean(1) @ p["W_sil"]))\n',
    '    if ALG_NOTEBOOK:\n'
    '        _nb.append((cur @ p["W_sil"]) if NB_PERSLOT\n'
    '                   else (_fed_core(cur).mean(1) @ p["W_sil"]))\n'
    '        if ALG_NB2 and "nb2_sil" in p and state.get("nbb") is not None:\n'
    '            state["nbb"].append(cur @ p["nb2_sil"])\n')

# B8 — do_train: arm the buffer (after the token cooker's arming block)
rep('    t0 = time.time()\n    for s in range(steps):\n',
    '    _bc_assign = None\n'
    '    if ALG_BAL_COOK > 0.0:\n'
    '        # THE BALANCED COOKER (2026-09-09): the per-row seal of the OLD\n'
    '        # memory and grounding roads; armed like the three cookers before\n'
    '        # it (a (B,1,1) buffer BEFORE the first capture; a FOURTH Knuth\n'
    '        # multiplier and addend, stable index-hash, never re-rolled).\n'
    '        assert ALG_NB2 and ALG_ALT5 and "nb2_sil" in p and "alt5_attn_wo" in p, (\n'
    '            "ALG_BAL_COOK seals lane 1 / bank / station 3 — it needs the "\n'
    '            "organs that carry the sealed rows: ALG_NB2=1 ALG_ALT5=1")\n'
    '        assert float(os.environ.get("ALG_TOK_COOK", "0")) <= 0.0, \\\n'
    '            "ALG_BAL_COOK and ALG_TOK_COOK do not compose"\n'
    '        assert _tok_seal_mode() == "0" and not os.environ.get("BC_EVAL", ""), \\\n'
    '            "ALG_TOK_SEAL / BC_EVAL are read-time doors; unset them for training"\n'
    '        _bc_h = ((np.arange(n, dtype=np.uint64) * np.uint64(668265263)\n'
    '                  + np.uint64(1013904223)) % np.uint64(4294967296)\n'
    '                 ).astype(np.float64) / 4294967296.0\n'
    '        _bc_assign = (_bc_h < ALG_BAL_COOK).astype(np.float32)\n'
    '        globals()["_BCV"] = Tensor(\n'
    '            np.zeros((batch, 1, 1), np.float32)).contiguous().realize()\n'
    '        print(f"[balcook] armed: share={ALG_BAL_COOK} -> "\n'
    '              f"{int(_bc_assign.sum())}/{n} rows sealed (stable index-hash, "\n'
    '              f"multiplier 668265263); on those rows lane 1 reads x0 and the "\n'
    '              f"main bank + station 3 read the tokens FLAT at loop breaths — "\n'
    '              f"lane 2 and station 5 are the only memory and grounding",\n'
    '              flush=True)\n'
    '    t0 = time.time()\n    for s in range(steps):\n')
rep('        if _tc_assign is not None:\n'
    '            globals()["_TCV"].assign(Tensor(\n'
    '                _tc_assign[idx].reshape(-1, 1, 1),\n'
    '                dtype=globals()["_TCV"].dtype)).realize()\n',
    '        if _tc_assign is not None:\n'
    '            globals()["_TCV"].assign(Tensor(\n'
    '                _tc_assign[idx].reshape(-1, 1, 1),\n'
    '                dtype=globals()["_TCV"].dtype)).realize()\n'
    '        if _bc_assign is not None:\n'
    '            globals()["_BCV"].assign(Tensor(\n'
    '                _bc_assign[idx].reshape(-1, 1, 1),\n'
    '                dtype=globals()["_BCV"].dtype)).realize()\n')
# B9 — val hygiene
rep('            os.environ["TC_EVAL"] = "0"       # ... and the OPEN reading\n'
    '            fv = _quick_val()\n'
    '            os.environ.pop("TC_EVAL", None)\n',
    '            os.environ["TC_EVAL"] = "0"       # ... and the OPEN reading\n'
    '            os.environ["BC_EVAL"] = "0"       # ... and the balanced cooker OPEN\n'
    '            fv = _quick_val()\n'
    '            os.environ.pop("BC_EVAL", None)\n'
    '            os.environ.pop("TC_EVAL", None)\n')
open(P, "w").write(s)
print(f"apply_balanced_gen: applied ({n0} -> {len(s)} bytes)")
