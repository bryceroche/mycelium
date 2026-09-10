"""apply_maskprep_jit.py — THE MASK-PREP REBUILD (2026-09-10, the word:
"drop the mask half; keep the facts; route the pass through the JIT
reader"). Two doors, both unset = untouched:
  ALG_SLOT_ALL=1      build_slot_masks returns ALL-ONES (all-to-all among
      the slots — the severance verdict: the frozen mask is not load-
      bearing). One door, one place: the mask-prep pass, _quick_val,
      loop_val, the collider read and every other consumer inherit it.
      The E&B coupling rides the same lanes (row-normalized ones).
  ALG_MASKPREP_JIT=1  the mask-prep pass calls mycelium.jit_read
      .read_forward (captured graph, bit-identical to eager — the JIT
      read's own proof) at batch ALG_MASKPREP_B (default 32) instead of
      the eager forward at batch 8. The cache's verify-on-hit re-runs the
      verification batches THROUGH the new path and asserts them against
      the eager-built bucket: the bit-identity proof is the cache's own.
      The captured graph is dropped after the pass (jit_read.reset()).
Idempotent: refuses a target that already carries the doors."""
import sys
P = "scripts/phase1_algebra_head.py"
s = open(P).read()
if "ALG_MASKPREP_JIT" in s:
    print("apply_maskprep_jit: target already carries the doors — refusing"); sys.exit(2)
n0 = len(s)

def rep(old, new, count=1):
    global s
    c = s.count(old)
    assert c == count, (c, old[:90])
    s = s.replace(old, new)

# M1 — the all-to-all door in build_slot_masks
rep('    B = o_np["fat"].shape[0]\n'
    '    masks = np.zeros((B, L_FAC, L_FAC), np.float32)\n'
    '    for bi in range(B):\n'
    '        tok_star = o_np["fat"][bi].argmax(-1)              # (L,)\n',
    '    B = o_np["fat"].shape[0]\n'
    '    if int(os.environ.get("ALG_SLOT_ALL", "0")):\n'
    '        # THE MASK HALF DROPPED (apply_maskprep_jit.py, 2026-09-10): the\n'
    '        # severance ladder read the frozen slot mask as not load-bearing\n'
    '        # (all-to-all 0.2521 vs 0.2511); the family runs all-to-all.\n'
    '        return np.ones((B, L_FAC, L_FAC), np.float32)\n'
    '    masks = np.zeros((B, L_FAC, L_FAC), np.float32)\n'
    '    for bi in range(B):\n'
    '        tok_star = o_np["fat"][bi].argmax(-1)              # (L,)\n')

# M2 — the pass: batch + JIT
rep('        _mp_key, _mp_fp, _mp_cached = _maskprep_lookup(p, n, seed)\n'
    '        _mp_starts = (range(0, n, 8) if _mp_cached is None\n'
    '                      else _maskprep_ver_starts(n))\n'
    '        for s0 in _mp_starts:\n'
    '            sl = np.arange(s0, min(s0 + 8, n))\n'
    '            pad = 8 - len(sl)\n'
    '            sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl\n'
    '            out0 = forward(p, Tensor(states[sl_p].astype(np.float32), dtype=dtypes.float),\n'
    '                           Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float),\n'
    '                           Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int),\n'
    '                           lsent=(Tensor(gold["lsent"][sl_p].astype(np.float32),\n'
    '                                         dtype=dtypes.float)\n'
    '                                  if ALG_LSENT and "lsent" in gold else None))\n',
    '        _mp_key, _mp_fp, _mp_cached = _maskprep_lookup(p, n, seed)\n'
    '        # THE MASK-PREP REBUILD (apply_maskprep_jit.py, 2026-09-10): the\n'
    '        # pass through the JIT reader at a larger batch. Verification\n'
    '        # batches (8-row granularity) are a subset of any batch >= 8, so\n'
    '        # the hit path still re-runs and asserts exactly those rows.\n'
    '        _mp_jit = int(os.environ.get("ALG_MASKPREP_JIT", "0"))\n'
    '        _mp_B = int(os.environ.get("ALG_MASKPREP_B", "32")) if _mp_jit else 8\n'
    '        assert _mp_B >= 8 and _mp_B % 8 == 0, _mp_B\n'
    '        if _mp_jit:\n'
    '            from mycelium import jit_read as _mp_jr\n'
    '            _mp_prev = os.environ.get("ALG_JIT_READ")\n'
    '            os.environ["ALG_JIT_READ"] = "1"\n'
    '            _mp_keys = (("fat", "args", "res")\n'
    '                        + (("pres", "ftype", "op", "dig", "dup") if FACTS is not None else ())\n'
    '                        + (("nl0",) if (ATLAS_TAB is not None and NL0 is not None) else ()))\n'
    '            print(f"[maskprep] JIT pass: batch {_mp_B}, keys {_mp_keys}", flush=True)\n'
    '        _mp_starts = (range(0, n, _mp_B) if _mp_cached is None\n'
    '                      else _maskprep_ver_starts(n))\n'
    '        for s0 in _mp_starts:\n'
    '            sl = np.arange(s0, min(s0 + _mp_B, n))\n'
    '            pad = _mp_B - len(sl)\n'
    '            sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl\n'
    '            _mp_args = (p, Tensor(states[sl_p].astype(np.float32), dtype=dtypes.float),\n'
    '                        Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float),\n'
    '                        Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int))\n'
    '            _mp_ls = (Tensor(gold["lsent"][sl_p].astype(np.float32), dtype=dtypes.float)\n'
    '                      if ALG_LSENT and "lsent" in gold else None)\n'
    '            out0 = (_mp_jr.read_forward(forward, *_mp_args, keys=_mp_keys, lsent=_mp_ls)\n'
    '                    if _mp_jit else forward(*_mp_args, lsent=_mp_ls))\n')
rep('        print(f"[breath] masks ready (mean degree "\n'
    '              f"{MASKS.sum(-1).mean():.1f}/{L_FAC})", flush=True)\n',
    '        if _mp_jit:\n'
    '            _mp_jr.reset()            # drop the captured read graph\n'
    '            if _mp_prev is None:\n'
    '                os.environ.pop("ALG_JIT_READ", None)\n'
    '            else:\n'
    '                os.environ["ALG_JIT_READ"] = _mp_prev\n'
    '        print(f"[breath] masks ready (mean degree "\n'
    '              f"{MASKS.sum(-1).mean():.1f}/{L_FAC})", flush=True)\n')
open(P, "w").write(s)
print(f"apply_maskprep_jit: 2 doors, 4 anchors applied ({n0} -> {len(s)} bytes)")
