"""apply_wheel_read.py — THE STEERING WHEEL AT READ TIME (2026-09-10, the
word: "give the middle breaths a job; every breath commits a tentative
parse and receives a core"). A read-only hook, the _GTAP/_CENSUS idiom:
`_WHEEL` (module global, None everywhere except under scripts/
wheel_read.py) carries the batch's n_vars / m, the spotlight strength
beta and the mode. After each loop breath k < K_B-1, forward commits the
breath's parse (the emission heads on the breath state, decode() per
slot), solves it completely, and on a certified refusal takes the
minimal unsatisfiable core; the core's slots get a SPOTLIGHT for the
next breath: +beta on the token scores of their source sentences (mode
'own': each core slot re-reads its own sentence; 'union': every core
slot re-reads the union of the core's sentences — the conflict's whole
evidence). The spotlight enters BOTH grounding roads (the main bank's
pbias and station 3's scores) — one road, consistently. Open-only: a
positive bias, never a mask. Unset = untouched (eq gate)."""
import sys
P = "scripts/phase1_algebra_head.py"; s = open(P).read()
if "_wheel_turn" in s:
    print("apply_wheel_read: already applied — refusing"); sys.exit(2)
def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80]); s = s.replace(old, new)
# W0 — the hook + the turn
rep('_GTAP = None      # THE GRADIENT TAP (apply_grad_tap.py): read-only probe leaves per breath\n',
    '_GTAP = None      # THE GRADIENT TAP (apply_grad_tap.py): read-only probe leaves per breath\n'
    '_WHEEL = None     # THE STEERING WHEEL at read time (apply_wheel_read.py): None = no wheel\n'
    '\n'
    '\n'
    'def _wheel_turn(p, state, kb, fat, sent, vst, B):\n'
    '    """Commit this breath\'s parse to the solver; on a certified refusal,\n'
    '    the core\'s slots get a spotlight on their source sentences for the\n'
    '    next breath. Returns a (B, 1, L_TOT, T) token-score bias or None."""\n'
    '    import numpy as _np\n'
    '    from tinygrad import Tensor as _Tw, dtypes as _dw\n'
    '    sys.path.insert(0, "scripts")\n'
    '    from alternator_bridge import refuse_and_core\n'
    '    o = _heads_of(p, state["cur"], vst, B)\n'
    '    onp = {k: v.realize().numpy() for k, v in o.items()}\n'
    '    fat_np = fat.realize().numpy() if hasattr(fat, "realize") else _np.asarray(fat)\n'
    '    sent_np = sent.numpy() if hasattr(sent, "numpy") else _np.asarray(sent)\n'
    '    T = sent_np.shape[1]\n'
    '    bias = _np.zeros((B, 1, L_TOT, T), _np.float32)\n'
    '    beta = float(_WHEEL.get("beta", 3.0)); mode = _WHEEL.get("mode", "union")\n'
    '    turned = 0\n'
    '    for b in range(B):\n'
    '        row = {k: onp[k][b] for k in onp}\n'
    '        row["query"] = _np.zeros(K_VARS, _np.float32)      # decode returns (facs, query)\n'
    '        parse = []\n'
    '        for j in range(L_FAC):\n'
    '            if row["pres"][j] <= 0:\n'
    '                continue\n'
    '            rj = dict(row); pr = _np.full_like(row["pres"], -1.0); pr[j] = row["pres"][j]; rj["pres"] = pr\n'
    '            try:\n'
    '                _facs, _ = decode(rj)\n'
    '            except Exception:\n'
    '                _facs = []\n'
    '            for f in _facs:\n'
    '                f["_slot"] = j; parse.append(f)\n'
    '        r = refuse_and_core(int(_WHEEL["n_vars"][b]), parse, int(_WHEEL["m"][b]))\n'
    '        _WHEEL.setdefault("stats", []).append((kb, r["status"], len(r["core"])))\n'
    '        if r["status"] != "unsat" or not r["core"]:\n'
    '            continue\n'
    '        turned += 1\n'
    '        core_slots = [parse[k]["_slot"] for k in r["core"]]\n'
    '        sents = {}\n'
    '        for j in core_slots:\n'
    '            tok = int(fat_np[b, j].argmax()); sents[j] = int(sent_np[b, min(tok, T - 1)])\n'
    '        for j in core_slots:\n'
    '            want = set(sents.values()) if mode == "union" else {sents[j]}\n'
    '            m = _np.isin(sent_np[b], list(want))\n'
    '            bias[b, 0, j, :] = _np.where(m, beta, 0.0).astype(_np.float32)\n'
    '    _WHEEL.setdefault("turned", []).append((kb, turned, B))\n'
    '    return _Tw(bias, dtype=_dw.float) if turned else None\n')
# W1 — the main bank: the spotlight joins the sync bias
rep('    h_tok, fat_cur = bank(p["fq"], L_TOT, extra=q_extra,\n'
    '                          pbias=(_sync[0](kb) if _sync is not None\n'
    '                                 else None),\n',
    '    _pb_kb = (_sync[0](kb) if _sync is not None else None)\n'
    '    _wb = state.get("wheel_bias")\n'
    '    if _wb is not None:                    # THE STEERING WHEEL\'s spotlight\n'
    '        _pb_kb = _wb if _pb_kb is None else _pb_kb + _wb\n'
    '    h_tok, fat_cur = bank(p["fq"], L_TOT, extra=q_extra,\n'
    '                          pbias=_pb_kb,\n')
# W2 — station 3: the same spotlight
rep('        if _sync is not None:            # the same breath rotation\n'
    '            _sa21 = _sa21 + _sync[0](kb)\n',
    '        if _sync is not None:            # the same breath rotation\n'
    '            _sa21 = _sa21 + _sync[0](kb)\n'
    '        if state.get("wheel_bias") is not None:   # the wheel\'s spotlight, second road\n'
    '            _sa21 = _sa21 + state["wheel_bias"]\n')
# W3 — forward's loop: turn the wheel after each breath (but the last)
rep('            for kb in range(1, K_B):\n'
    '                breath_step(p, _bs_state, kb, _bs_ctx)\n'
    '            cur = _bs_state["cur"]\n',
    '            for kb in range(1, K_B):\n'
    '                breath_step(p, _bs_state, kb, _bs_ctx)\n'
    '                if _WHEEL is not None and kb < K_B - 1:\n'
    '                    _bs_state["wheel_bias"] = _wheel_turn(p, _bs_state, kb, fat, sent, vst, B)\n'
    '            cur = _bs_state["cur"]\n')
open(P, "w").write(s); print("apply_wheel_read: hook + turn + 3 anchors applied")
