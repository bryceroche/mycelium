"""Patch (ledger 2026-11-02 CY): the T6/T9 student's GSM8K export builds sentence indices with the training split's own segmenter
(phase1_algebra_head.sent_indices; checked 400/400 rows equal to formpm35cBP's stored `sent`). Applied after lfq5's last launch."""
p = "scripts/distill_reader.py"; s = open(p).read()
R = [('    assert not T6, "the T6 student\'s GSM8K export needs GSM8K sentence indices (not built yet)"\n',
      '    from phase1_algebra_head import sent_indices   # (ledger 11-02 CY) the training split\'s own segmenter, so T6 forms export too\n'),
     ('        xin = (Tensor(EMB[ids], dtype=dtypes.half), addmask(m), Tensor(db, dtype=dtypes.half) if DBIN else None)\n',
      '        xin = (Tensor(EMB[ids], dtype=dtypes.half), addmask(m), Tensor(db, dtype=dtypes.half) if DBIN else None)\n'
      '        rc = tuple(Tensor(a, dtype=dtypes.float) for a in t6_rot(np.stack([sent_indices(uniq[i], enc[i].offsets, m[r]).astype(np.int8)\n'
      '                                                                    for r, i in enumerate(sl)]), m)) if T6 else None\n'),
     ('            o = (_rms_norm(body(p, *xin), p["fnorm"], EPS) @ p["w_nat"] + p["b_nat"]).realize().numpy()\n',
      '            o = (_rms_norm(body(p, *xin, rc), p["fnorm"], EPS) @ p["w_nat"] + p["b_nat"]).realize().numpy()\n'),
     ('            o = forward(p, *xin).realize().numpy()\n        assert np.isfinite(o).all()\n        for r, i in enumerate(sl): st[start[i]',
      '            o = forward(p, *xin, rc).realize().numpy()\n        assert np.isfinite(o).all()\n        for r, i in enumerate(sl): st[start[i]')]
for a, b in R:
    assert s.count(a) == 1, a[:60]; s = s.replace(a, b)
open(p, "w").write(s); print("applied t6_gsm_export")
