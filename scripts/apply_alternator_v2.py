"""apply_alternator_v2.py — the staged ALTERNATOR V2 patch (2026-09-01).
Cycle-level neural/symbolic ping-pong: pass-1 parse commits its confident
slots, alternator_bridge.ping propagates (GAC only, no search), and the
forced facts re-enter pass-2 as var-slot conditioning (fact_buf).

RUN ONLY AFTER the running chain exits (the running-chain law — a live
systemd unit has this module imported). --check mode loads the file,
asserts every anchor, ast-parses the WOULD-BE result and writes NOTHING.

Equivalence contract (checked by design, verify with an eq read):
  ALG_ALT2 unset            -> zero behavior change (every path guarded)
  ALG_ALT2=1, fact_buf=None -> forward bit-identical (injection skipped)
JIT discipline: when ALG_ALT2=1 the jitted step ALWAYS receives the fixed
b_fact buffer (zeros when no facts) — stable signature, assign-in-place,
no dtypes.float32 literals (memory/reference_tinygrad_am_quirks.md).
"""
import ast
import sys

fn = 'scripts/phase1_algebra_head.py'
CHECK = '--check' in sys.argv
s = open(fn).read()
n_lines0 = s.count('\n')

PATCHES = []


def patch(num, desc, old, new):
    PATCHES.append((num, desc, old, new))


# 1. build_params: W_fact + alt2_g beside the other env-gated organ params
#    (anchored on the detwave block's tail). W_fact small-random (the W_det
#    idiom — it is the ONLY path for facts, zero-init would starve it);
#    alt2_g ajar at 0.02 (the bus_g/det_g idiom).
patch(1, "build_params: W_fact + alt2_g (env ALG_ALT2)", '''        p["W_det"] = t(rng.randn(3, H_W) / math.sqrt(3))
        p["det_g"] = t(np.full(1, 0.02))''', '''        p["W_det"] = t(rng.randn(3, H_W) / math.sqrt(3))
        p["det_g"] = t(np.full(1, 0.02))
    if int(os.environ.get("ALG_ALT2", "0")):
        # ALTERNATOR V2 (2026-09-01, word given): cycle-level ping-pong —
        # pass-1 commits confident slots, the bridge propagates
        # (alternator_bridge.ping; meter-divergence law: the check calls
        # its organ), forced facts re-enter pass-2 as var-slot
        # conditioning. W_fact is the ONLY path for facts: small-random
        # init (the W_det idiom, NEVER zero); gate ajar (0.02, the law).
        p["W_fact"] = t(rng.randn(4, H_W) / math.sqrt(4))
        p["alt2_g"] = t(np.full(1, 0.02))''')

# 2. the commit adapter + cycle driver, module-level right after
#    build_slot_masks (train and val share it — two-terminal discipline)
patch(2, "module helper alt2_fact_buf after build_slot_masks", '''        shared = (M.astype(np.int32) @ M.astype(np.int32).T) > 0
        masks[bi] = (same | shared | np.eye(L_FAC, dtype=bool)).astype(np.float32)
    return masks''', '''        shared = (M.astype(np.int32) @ M.astype(np.int32).T) > 0
        masks[bi] = (same | shared | np.eye(L_FAC, dtype=bool)).astype(np.float32)
    return masks


def alt2_fact_buf(onp, se_np, n_vars_arr, m_arr, theta=0.9):
    """ALTERNATOR V2 commit adapter + cycle driver (2026-09-01). Consumes
    the realized pass-1 output dict (decode()'s key conventions: pres/
    ftype/op/dig/args/res logits + optional dup), discretizes ONLY
    confident slots (presence sigmoid > theta; ftype/res softmax top-prob
    > theta; args sigmoid > theta — args is BCE-trained 2-hot, a softmax
    read is wrong there), commits given/rel factors in the mint grammar,
    calls the symbolic half (alternator_bridge.ping: GAC propagation
    only), and packs the forced facts into (B, 24, 4) float32:
    [1.0, h/9, t/9, o/9] per known var, zeros elsewhere. Contradiction
    (mass None) or ANY per-item exception -> zeros for that item —
    silence, never a crash (the bridge contract). Numpy in, numpy out:
    detached by construction (the dual-terminal law). se_np rides for
    signature symmetry with build_slot_masks (unused)."""
    from alternator_bridge import ping   # lazy — scripts/ is on sys.path
    B = onp["pres"].shape[0]
    buf = np.zeros((B, K_VARS, 4), np.float32)

    def _sig(x):
        return 1.0 / (1.0 + np.exp(-x))

    def _smax(x):
        e = np.exp(x - x.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)

    for bi in range(B):
        try:
            ftp = _smax(onp["ftype"][bi])          # (L, nft)
            rsp = _smax(onp["res"][bi])            # (L, K_VARS)
            agp = _sig(onp["args"][bi])            # (L, K_VARS) 2-hot BCE
            facs = []
            for j in range(L_FAC):
                if _sig(onp["pres"][bi, j]) <= theta:
                    continue
                ft = int(ftp[j].argmax())
                if ftp[j, ft] <= theta:
                    continue
                res = int(rsp[j].argmax())
                if rsp[j, res] <= theta:
                    continue
                if ft == 1:                        # given: digits carry value
                    digs = onp["dig"][bi, j].argmax(-1)
                    v = int(sum(d * 10 ** (N_DIG - 1 - i2)
                                for i2, d in enumerate(digs)))
                    facs.append({"ftype": "given", "var": res, "value": v})
                elif ft == 0:                      # rel (decode conventions)
                    op = "add" if onp["op"][bi, j].argmax() == 0 else "mul"
                    if "dup" in onp and onp["dup"][bi, j] > 0:
                        a0 = int(np.argmax(onp["args"][bi, j]))
                        if agp[j, a0] <= theta:
                            continue
                        args = [a0, a0]
                    else:
                        top2 = np.argsort(-onp["args"][bi, j])[:2]
                        if float(agp[j, top2].min()) <= theta:
                            continue
                        args = sorted(int(a) for a in top2)
                    facs.append({"ftype": "rel", "op": op,
                                 "args": args, "result": res})
                # other ftypes: never committed (bridge grammar: given/rel)
            if not facs:
                continue
            nv = max([int(n_vars_arr[bi])]        # do_eval's nv convention
                     + [v + 1 for f in facs for v in
                        ([f["var"]] if f["ftype"] == "given"
                         else list(f["args"]) + [f["result"]])])
            facts, mass, _r = ping(nv, facs, int(m_arr[bi]))
            if mass is None:                       # contradiction: silence
                continue
            for v, val in facts.items():
                if 0 <= v < K_VARS and 0 <= val <= 999:
                    buf[bi, v] = (1.0, (val // 100) / 9.0,
                                  (val // 10 % 10) / 9.0, (val % 10) / 9.0)
        except Exception:
            buf[bi] = 0.0                          # per-item silence
    return buf''')

# 3. forward signature: the fact_buf input port (default None = inert)
patch(3, "forward(): fact_buf=None kwarg", '''def forward(p, trunk, tokmask, sent, slot_mask=None, revoke=None, tail=None, drop=None, anchor=None, amask=None, gmod=None, pmask=None, lsent=None, reg=None):''', '''def forward(p, trunk, tokmask, sent, slot_mask=None, revoke=None, tail=None, drop=None, anchor=None, amask=None, gmod=None, pmask=None, lsent=None, reg=None, fact_buf=None):''')

# 4. the injection: var-slot states, exactly once, before the breath loop.
#    vst is the per-var-slot representation every pointer head reads
#    (args/res/y/query all score against vst) and the breath loop's bank
#    re-reads — facts condition the whole cycle.
patch(4, "forward(): fact_buf injection into vst", '''    vst, vat = bank(p["vq"], K_VARS, pbias=_lb)''', '''    vst, vat = bank(p["vq"], K_VARS, pbias=_lb)
    if int(os.environ.get("ALG_ALT2", "0")) and fact_buf is not None:
        # ALTERNATOR V2 injection: symbolic facts ((B, 24, 4): known flag
        # + MSD digits/9) condition the var-slot states that args/res/y/
        # query pointers and every breath read against. BOTH guards are
        # load-bearing: env unset -> byte-identical baseline; fact_buf
        # None -> byte-identical too (the injection is skipped entirely).
        vst = vst + (fact_buf @ p["W_fact"]) * p["alt2_g"].reshape(1, 1, 1)''')

# 5. do_train: the banked FACTS array beside MASKS
patch(5, "do_train: FACTS bank declaration", '''    MASKS = None
    if K_B > 1:''', '''    MASKS = None
    ALT2 = int(os.environ.get("ALG_ALT2", "0"))
    FACTS = np.zeros((n, K_VARS, 4), np.float32) if ALT2 else None
    if K_B > 1:''')

# 6. mask-prep pass: after the pass-1 realize (where masks are built),
#    also commit + ping + bank facts (frozen like MASKS, same protocol)
patch(6, "do_train mask-prep: pass-1 commit -> FACTS", '''            o0 = {k: out0[k].realize().numpy() for k in ("fat", "args", "res")}
            MASKS[sl] = build_slot_masks(o0, sent[sl_p])[:len(sl)]''', '''            o0 = {k: out0[k].realize().numpy() for k in ("fat", "args", "res")}
            MASKS[sl] = build_slot_masks(o0, sent[sl_p])[:len(sl)]
            if FACTS is not None:
                # ALTERNATOR V2 pass-1 commit: the same realized parse the
                # masks come from; facts banked like MASKS (frozen for
                # training efficiency, rebuilt with the head at each prep)
                _ka2 = (("pres", "ftype", "op", "dig")
                        + (("dup",) if "dup" in out0 else ()))
                _oa2 = {**o0, **{k: out0[k].realize().numpy() for k in _ka2}}
                _nv2 = np.array([samples[int(i)].get("n_vars", K_VARS)
                                 for i in sl_p])
                _ma2 = np.array([samples[int(i)].get("m", 0)
                                 for i in sl_p])
                FACTS[sl] = alt2_fact_buf(_oa2, sent[sl_p], _nv2,
                                          _ma2)[:len(sl)]''')

# 7. the fixed input buffer (assign-in-place protocol, like b_mask/b_tail)
patch(7, "do_train: b_fact fixed buffer", '''    b_mask = fix(np.zeros((batch, L_FAC, L_FAC), np.float32), dtypes.float) \\
        if K_B > 1 else None''', '''    b_mask = fix(np.zeros((batch, L_FAC, L_FAC), np.float32), dtypes.float) \\
        if K_B > 1 else None
    b_fact = fix(np.zeros((batch, K_VARS, 4), np.float32), dtypes.float) \\
        if ALT2 else None   # ALT2: fixed shape, ALWAYS fed (zeros when no
                            # facts) — the jitted step's signature is stable''')

# 8-10. every pass-2 forward inside the jitted step carries fact_buf —
#       b_fact is None when ALG_ALT2=0 (identical to the default), and
#       wiring all three branches keeps W_fact/alt2_g in the graph under
#       every training mode (no None-grad at the optimizer).
patch(8, "step() XOUT branch pass-2: fact_buf", '''            o = forward(p, s_tr, b_tk, b_se, slot_mask=b_mask, revoke=rv,
                        tail=b_tail, reg=b_reg)''', '''            o = forward(p, s_tr, b_tk, b_se, slot_mask=b_mask, revoke=rv,
                        tail=b_tail, reg=b_reg, fact_buf=b_fact)''')

patch(9, "step() NAZ branch pass-2: fact_buf", '''            o = forward(p, s_tr, b_tk, b_se, slot_mask=b_mask, tail=b_tail,
                        gmod=_gm)''', '''            o = forward(p, s_tr, b_tk, b_se, slot_mask=b_mask, tail=b_tail,
                        gmod=_gm, fact_buf=b_fact)''')

patch(10, "step() main branch: fact_buf", '''            o = forward(p, s_tr, b_tk, b_se, slot_mask=b_mask, tail=b_tail,
                        drop=(b_drop if _bd else None), lsent=b_ls, reg=b_reg)''', '''            o = forward(p, s_tr, b_tk, b_se, slot_mask=b_mask, tail=b_tail,
                        drop=(b_drop if _bd else None), lsent=b_ls, reg=b_reg,
                        fact_buf=b_fact)''')

# 11. the per-step feed (assign-in-place, one buffer, every step)
patch(11, "do_train loop: b_fact per-step assign", '''        if b_tail is not None:
            b_tail.assign(Tensor(TAILS[idx].astype(np.float32), dtype=dtypes.float).contiguous()).realize()''', '''        if b_fact is not None:
            b_fact.assign(Tensor(FACTS[idx], dtype=dtypes.float).contiguous()).realize()
        if b_tail is not None:
            b_tail.assign(Tensor(TAILS[idx].astype(np.float32), dtype=dtypes.float).contiguous()).realize()''')

# 12. _quick_val: the same guarded two-pass so val measures what train
#     trains (loop discipline — only under ALG_ALT2=1; otherwise the
#     restructured call is behavior-identical to the incumbent)
patch(12, "_quick_val: guarded two-pass + fact_buf", '''            o = forward(p, Tensor(vst[sl_p].astype(np.float32), dtype=dtypes.float),
                        Tensor(vtk[sl_p].astype(np.float32), dtype=dtypes.float),
                        Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int))''', '''            _t1 = Tensor(vst[sl_p].astype(np.float32), dtype=dtypes.float)
            _t2 = Tensor(vtk[sl_p].astype(np.float32), dtype=dtypes.float)
            _t3 = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)
            o = forward(p, _t1, _t2, _t3)
            if int(os.environ.get("ALG_ALT2", "0")):
                # ALTERNATOR V2 val two-pass: masked pass-2 + LIVE facts
                # (recomputed from this checkpoint's own pass-1, not the
                # banked FACTS — val measures the deployable cycle)
                _kv = (("pres", "ftype", "op", "dig", "fat", "args", "res")
                       + (("dup",) if "dup" in o else ()))
                _ov = {k: o[k].realize().numpy() for k in _kv}
                _mkv = build_slot_masks(_ov, vse[sl_p].astype(np.int32))
                _nvv = np.array([vs[int(i)].get("n_vars", K_VARS)
                                 for i in sl_p])
                _mav = np.array([vs[int(i)].get("m", 0) for i in sl_p])
                _fbv = alt2_fact_buf(_ov, vse[sl_p], _nvv, _mav)
                o = forward(p, _t1, _t2, _t3,
                            slot_mask=Tensor(_mkv, dtype=dtypes.float),
                            fact_buf=Tensor(_fbv, dtype=dtypes.float))''')


for num, desc, old, new in PATCHES:
    assert old in s, f"anchor {num} MISSING ({desc}) — read the file, adjust"
    assert s.count(old) == 1, f"anchor {num} NOT UNIQUE ({desc})"
    s = s.replace(old, new, 1)

tree = ast.parse(s)                       # the would-be result must parse

# structural asserts on the would-be module (cheap, no import, no GPU)
_defs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
assert "alt2_fact_buf" in _defs, "helper missing from patched tree"
assert any(a.arg == "fact_buf" for a in _defs["forward"].args.args), \
    "forward() lost the fact_buf kwarg"

print(f"[alternator-v2] {len(PATCHES)} anchors OK "
      f"(+{s.count(chr(10)) - n_lines0} lines):")
for num, desc, _o, _n in PATCHES:
    print(f"  {num:2d}. {desc}")
if CHECK:
    print("[alternator-v2] --check: ast OK on the would-be result; "
          "NOTHING written")
else:
    open(fn, 'w').write(s)
    print("[alternator-v2] APPLIED; ast OK — run the ALG_ALT2-unset eq "
          "read before trusting (equivalence contract)")
