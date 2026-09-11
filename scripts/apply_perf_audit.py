"""apply_perf_audit.py — the 2026-09-11 tinygrad perf audit's head-side items,
all bit-identical by construction (the eq gate proves it):
  #4 half feed: the trunk states memmap is float16; feed it as half and
     upcast IN-GRAPH (exact) — the trainer's b_tr buffer, the feed line,
     the maskprep pass, _quick_val; forward() upcasts any non-float trunk.
  #5 one realize per step for the whole feed (was ~30 dispatches).
  #7 per-breath host constants cached as realized Tensors (_ct): the
     wheel tables, the stamps, the mask cooker's column mask.
  #9 _quick_val through the JIT reader (door ALG_JIT_VAL=1).
  #3 the facts pass across CPU cores (door ALG_FACTS_POOL=1; scripts/
     facts_pool.py)."""
import sys
P = "scripts/phase1_algebra_head.py"; s = open(P).read()
if "_CONST_T" in s:
    print("apply_perf_audit: already applied — refusing"); sys.exit(2)
def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90]); s = s.replace(old, new)
# #7 the constant cache
rep('_GTAP = None      # THE GRADIENT TAP (apply_grad_tap.py): read-only probe leaves per breath\n',
    '_GTAP = None      # THE GRADIENT TAP (apply_grad_tap.py): read-only probe leaves per breath\n'
    '_CONST_T = {}\n'
    '\n'
    '\n'
    'def _ct(key, arr):\n'
    '    """A realized constant Tensor cached by key (perf audit #7): per-breath wheel\n'
    '    tables and stamps were re-uploaded as graph nodes on every step."""\n'
    '    t = _CONST_T.get(key)\n'
    '    if t is None:\n'
    '        from tinygrad import Tensor as _Tc, dtypes as _dc\n'
    '        t = _Tc(np.ascontiguousarray(arr, dtype=np.float32), dtype=_dc.float).contiguous().realize()\n'
    '        _CONST_T[key] = t\n'
    '    return t\n')
rep('    return rot2(x, _Tf(_fac[k - 1], dtype=_df.float),\n'
    '                _Tf(float(sign) * _fas[k - 1], dtype=_df.float))\n',
    '    return rot2(x, _ct(("fac", k), _fac[k - 1]),\n'
    '                _ct(("fas", k, int(sign)), float(sign) * _fas[k - 1]))\n')
rep('        _nb_st = _T2(NB_STAMPS, dtype=_dt2.float)\n', '        _nb_st = _ct("nb_st", NB_STAMPS)\n')
rep('        _bq2 = _rot2(bq,\n'
    '                     _Tm(_mac[kb - 1], dtype=_dm.float),\n'
    '                     _Tm(_mas[kb - 1], dtype=_dm.float))\n',
    '        _bq2 = _rot2(bq, _ct(("mac", kb), _mac[kb - 1]), _ct(("mas", kb), _mas[kb - 1]))\n')
rep('            _mcc = Tensor(np.concatenate(\n'
    '                [np.ones(L_FAC, np.float32),\n'
    '                 np.zeros(L_TOT - L_FAC, np.float32)])).reshape(1, 1, -1)\n',
    '            _mcc = _ct("mcc", np.concatenate(\n'
    '                [np.ones(L_FAC, np.float32),\n'
    '                 np.zeros(L_TOT - L_FAC, np.float32)])).reshape(1, 1, -1)\n')
rep('            _rcq = _Tq(_qac[kb - 1].reshape(MX_HEADS, _mx_hd // 2),\n'
    '                       dtype=_dq.float).reshape(1, MX_HEADS, 1, -1)\n'
    '            _rsq = _Tq(_qas[kb - 1].reshape(MX_HEADS, _mx_hd // 2),\n'
    '                       dtype=_dq.float).reshape(1, MX_HEADS, 1, -1)\n',
    '            _rcq = _ct(("qac", kb), _qac[kb - 1].reshape(MX_HEADS, _mx_hd // 2)).reshape(1, MX_HEADS, 1, -1)\n'
    '            _rsq = _ct(("qas", kb), _qas[kb - 1].reshape(MX_HEADS, _mx_hd // 2)).reshape(1, MX_HEADS, 1, -1)\n')
rep('            _pol_u = _rot2(_pol_u,\n'
    '                           _Tp(_pdc[kb - 1], dtype=_dp.float),\n'
    '                           _Tp(_pds[kb - 1], dtype=_dp.float))\n',
    '            _pol_u = _rot2(_pol_u, _ct(("pdc", kb), _pdc[kb - 1]), _ct(("pds", kb), _pds[kb - 1]))\n')
# #4 forward upcasts any non-float trunk (exact)
rep('    B = trunk.shape[0]\n'
    '    waist = (trunk @ p["waist_w"] + p["waist_b"]).gelu() + p["sent_emb"][sent]\n',
    '    B = trunk.shape[0]\n'
    '    if trunk.dtype != dtypes.float:\n'
    '        trunk = trunk.cast(dtypes.float)   # perf audit #4: half feeds upcast in-graph (exact)\n'
    '    waist = (trunk @ p["waist_w"] + p["waist_b"]).gelu() + p["sent_emb"][sent]\n')
# the trainer's buffer + graph entry
rep('    b_tr = fix(np.zeros((batch, T_ALG, H_TRUNK), np.float32), dtypes.float)\n',
    '    b_tr = fix(np.zeros((batch, T_ALG, H_TRUNK), np.float16), dtypes.half)   # perf audit #4\n')
rep('        else:\n            s_tr = b_tr\n', '        else:\n            s_tr = b_tr.cast(dtypes.float)   # perf audit #4: exact upcast in-graph\n')
# #5 + #4 the feed: one realize
rep('        if not TRUNK_LORA:   # audit #15: b_tr is dead under the in-graph trunk\n'
    '            b_tr.assign(Tensor(states[idx].astype(np.float32), dtype=dtypes.float).contiguous()).realize()\n'
    '        _rl = [b_tk.assign(Tensor(tokmask[idx].astype(np.float32), dtype=dtypes.float).contiguous()),\n'
    '               b_se.assign(Tensor(sent[idx].astype(np.int32), dtype=dtypes.int).contiguous())]\n'
    '        if TRUNK_LORA:\n'
    '            _rl.append(b_ids.assign(Tensor(IDS_ALL[idx].astype(np.int32), dtype=dtypes.int).contiguous()))\n'
    '        Tensor.realize(*_rl)   # perf audit #2: one combined schedule, not N dispatches\n',
    '        _rl = []\n'
    '        if not TRUNK_LORA:   # audit #15: b_tr is dead under the in-graph trunk\n'
    '            _rl.append(b_tr.assign(Tensor(np.ascontiguousarray(states[idx]), dtype=dtypes.half).contiguous()))   # perf audit #4: half feed\n'
    '        _rl += [b_tk.assign(Tensor(tokmask[idx].astype(np.float32), dtype=dtypes.float).contiguous()),\n'
    '                b_se.assign(Tensor(sent[idx].astype(np.int32), dtype=dtypes.int).contiguous())]\n'
    '        if TRUNK_LORA:\n'
    '            _rl.append(b_ids.assign(Tensor(IDS_ALL[idx].astype(np.int32), dtype=dtypes.int).contiguous()))\n')
for old_line, new_line in (
    ('            bg["parents"].assign(Tensor(PARENTS[idx], dtype=dtypes.float).contiguous()).realize()\n', '            _rl.append(bg["parents"].assign(Tensor(PARENTS[idx], dtype=dtypes.float).contiguous()))\n'),
    ('            bg["claimed"].assign(Tensor(CLAIMED[idx], dtype=dtypes.float).contiguous()).realize()\n', '            _rl.append(bg["claimed"].assign(Tensor(CLAIMED[idx], dtype=dtypes.float).contiguous()))\n'),
    ('            b_ls.assign(Tensor(gold["lsent"][idx].astype(np.float32), dtype=dtypes.float).contiguous()).realize()\n', '            _rl.append(b_ls.assign(Tensor(gold["lsent"][idx].astype(np.float32), dtype=dtypes.float).contiguous()))\n'),
    ('            b_mask.assign(Tensor(_mfeed, dtype=dtypes.float).contiguous()).realize()\n', '            _rl.append(b_mask.assign(Tensor(_mfeed, dtype=dtypes.float).contiguous()))\n'),
    ('            b_fact.assign(Tensor(FACTS[idx], dtype=dtypes.float).contiguous()).realize()\n', '            _rl.append(b_fact.assign(Tensor(FACTS[idx], dtype=dtypes.float).contiguous()))\n'),
    ('            b_mhm.assign(Tensor(MASSB[idx][:, :, None],\n                                dtype=dtypes.float).contiguous()).realize()\n', '            _rl.append(b_mhm.assign(Tensor(MASSB[idx][:, :, None],\n                                dtype=dtypes.float).contiguous()))\n'),
    ('            b_mha.assign(Tensor(ATLAS_TAB[ATLAS_IDX[idx]],\n                                dtype=dtypes.float).contiguous()).realize()\n', '            _rl.append(b_mha.assign(Tensor(ATLAS_TAB[ATLAS_IDX[idx]],\n                                dtype=dtypes.float).contiguous()))\n'),
    ('            b_tail.assign(Tensor(TAILS[idx].astype(np.float32), dtype=dtypes.float).contiguous()).realize()\n', '            _rl.append(b_tail.assign(Tensor(TAILS[idx].astype(np.float32), dtype=dtypes.float).contiguous()))\n'),
    ('            b_reg.assign(Tensor(REG[idx].astype(np.float32), dtype=dtypes.float).contiguous()).realize()\n', '            _rl.append(b_reg.assign(Tensor(REG[idx].astype(np.float32), dtype=dtypes.float).contiguous()))\n'),
    ('            b_drop.assign(Tensor(np.array(\n                [1.0 if rng.rand() >= float(os.environ["BREATH_DROPOUT"]) else 0.0],\n                np.float32), dtype=dtypes.float).contiguous()).realize()\n', '            _rl.append(b_drop.assign(Tensor(np.array(\n                [1.0 if rng.rand() >= float(os.environ["BREATH_DROPOUT"]) else 0.0],\n                np.float32), dtype=dtypes.float).contiguous()))\n'),
    ('            bg[k].assign(Tensor(v.astype(npdt), dtype=bg[k].dtype).contiguous()).realize()\n', '            _rl.append(bg[k].assign(Tensor(v.astype(npdt), dtype=bg[k].dtype).contiguous()))\n        Tensor.realize(*_rl)   # perf audit #5: the whole feed in ONE schedule\n'),
):
    rep(old_line, new_line)
# #4 the maskprep pass + _quick_val feed half
rep('            _mp_args = (p, Tensor(states[sl_p].astype(np.float32), dtype=dtypes.float),\n',
    '            _mp_args = (p, Tensor(np.ascontiguousarray(states[sl_p]), dtype=dtypes.half),\n')
rep('        t_tr = Tensor(states[sl_p].astype(np.float32), dtype=dtypes.float)\n',
    '        t_tr = Tensor(np.ascontiguousarray(states[sl_p]), dtype=dtypes.half)   # perf audit #4\n')
# #9 _quick_val through the JIT reader
rep('        out = forward(p, t_tr, t_tk, t_se, tail=_tl)\n'
    '        if int(os.environ.get("ALG_BREATH", "1")) > 1 and "W_bo" in p \\\n'
    '                and not int(os.environ.get("BREATH_SILENT", "0")):\n'
    '            o0 = {k: out[k].realize().numpy() for k in ("fat", "args", "res")}\n'
    '            mk = build_slot_masks(o0, sent[sl_p])\n'
    '            out = forward(p, t_tr, t_tk, t_se, tail=_tl,\n'
    '                          slot_mask=Tensor(mk, dtype=dtypes.float))\n',
    '        _jv = int(os.environ.get("ALG_JIT_VAL", "0"))      # perf audit #9\n'
    '        if _jv:\n'
    '            from mycelium.jit_read import read_forward as _rfv\n'
    '            _jk_v = (("pres", "ftype", "op", "islit", "dig", "args", "res", "query")\n'
    '                     + (("sel",) if "h_sel" in p else ()) + (("dup",) if "h_dup" in p else ())\n'
    '                     + (("dargs",) if "W_dargs" in p else ()))\n'
    '            _prev_jr = os.environ.get("ALG_JIT_READ"); os.environ["ALG_JIT_READ"] = "1"\n'
    '            out = _rfv(forward, p, t_tr, t_tk, t_se, keys=("fat", "args", "res"), tail=_tl)\n'
    '        else:\n'
    '            out = forward(p, t_tr, t_tk, t_se, tail=_tl)\n'
    '        if int(os.environ.get("ALG_BREATH", "1")) > 1 and "W_bo" in p \\\n'
    '                and not int(os.environ.get("BREATH_SILENT", "0")):\n'
    '            o0 = {k: out[k].realize().numpy() for k in ("fat", "args", "res")}\n'
    '            mk = build_slot_masks(o0, sent[sl_p])\n'
    '            if _jv:\n'
    '                out = _rfv(forward, p, t_tr, t_tk, t_se, keys=_jk_v, tail=_tl,\n'
    '                           slot_mask=Tensor(mk, dtype=dtypes.float))\n'
    '            else:\n'
    '                out = forward(p, t_tr, t_tk, t_se, tail=_tl,\n'
    '                              slot_mask=Tensor(mk, dtype=dtypes.float))\n'
    '        if _jv:\n'
    '            if _prev_jr is None: os.environ.pop("ALG_JIT_READ", None)\n'
    '            else: os.environ["ALG_JIT_READ"] = _prev_jr\n')
# #3 the facts pass across cores
rep('                FACTS[sl] = alt2_fact_buf(_oa2, sent[sl_p], _nv2,\n',
    '                if int(os.environ.get("ALG_FACTS_POOL", "0")):     # perf audit #3\n'
    '                    from facts_pool import run as _fp_run\n'
    '                    FACTS[sl] = _fp_run(_oa2, sent[sl_p], _nv2, _ma2, mass_out=_mo2)[:len(sl)]\n'
    '                else:\n'
    '                  FACTS[sl] = alt2_fact_buf(_oa2, sent[sl_p], _nv2,\n')
open(P, "w").write(s)
# loop_val: half feed
L = "scripts/loop_val.py"; t = open(L).read()
old = '        ts = Tensor(vst[sl_p].astype(np.float32), dtype=dtypes.float)\n'
assert t.count(old) == 1; open(L, "w").write(t.replace(old, '        ts = Tensor(np.ascontiguousarray(vst[sl_p]), dtype=dtypes.half)   # perf audit #4: half feed, upcast in-graph\n'))
print("apply_perf_audit: head (#3 #4 #5 #7 #9) + loop_val (#4) applied")
