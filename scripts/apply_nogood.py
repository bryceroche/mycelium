"""THE NOGOOD (2026-09-13, Bryce: "the second read might be the same as the
first"): two additions to the melt, both from the certificate. (1) melt the
LEAST-CONFIDENT member of the core, not the whole core (a minimal core says
"one of these is wrong"; the machine's margins say which); (2) a NOGOOD on that
slot's previous binding: its least-confident field's previous choice is masked
out at the next breath's decode (and at the final read), so the second read is
different by construction. ALG_WHEEL_NOGOOD=1 (unset = bit-identical).
Pure-numpy helpers (_nogood_pick / _nogood_apply) so the logic is testable
without the model. Idempotent; HEAD_PATH / WR_PATH env for copies."""
import os, sys
hp = os.environ.get("HEAD_PATH", "scripts/phase1_algebra_head.py"); s = open(hp).read()
wp = os.environ.get("WR_PATH", "scripts/wheel_read.py"); w = open(wp).read()
def rep(src, old, new):
    assert src.count(old) == 1, (src.count(old), old[:80]); return src.replace(old, new)
if "_WHEEL_NOGOOD" not in s:
    s = rep(s, '''def _melt(cur, m):
''', '''_WHEEL_NOGOOD = int(os.environ.get("ALG_WHEEL_NOGOOD", "0"))


def _slot_margins(row, j):
    """(field, margin, prev_choice) per decision field of slot j from a row's
    head logits: res (top1-top2), op (rel slots), args (2nd-of-top2 vs 3rd)."""
    import numpy as _np
    out = []
    r = _np.sort(row["res"][j])[::-1]; out.append(("res", float(r[0] - r[1]), int(row["res"][j].argmax())))
    if int(row["ftype"][j].argmax()) == 0 and "op" in row:
        o = _np.sort(row["op"][j])[::-1]; out.append(("op", float(o[0] - o[1]), int(row["op"][j].argmax())))
        a = _np.argsort(-row["args"][j]); av = row["args"][j][a]
        out.append(("args", float(av[1] - av[2]), int(a[1])))     # the weaker of the top-2 pointers
    return out


def _nogood_pick(row, core_slots):
    """The least-confident (slot, field, prev_choice) among the core's slots."""
    best = None
    for j in core_slots:
        for f, m, c in _slot_margins(row, j):
            if best is None or m < best[1]:
                best = (j, m, f, c)
    return (best[0], best[2], best[3]) if best else None


def _nogood_apply(onp, nogoods):
    """Mask the previous choices out of the head logits, in place. nogoods: list of (b, j, field, idx)."""
    for b, j, f, idx in nogoods:
        onp[f][b, j, idx] = -1e9
    return onp


def _melt(cur, m):
''')
    s = rep(s, '''    o = _heads_of(p, state["cur"], vst, B)
    onp = {k: v.realize().numpy() for k, v in o.items()}
    fat_np = fat.realize().numpy() if hasattr(fat, "realize") else _np.asarray(fat)
''', '''    o = _heads_of(p, state["cur"], vst, B)
    onp = {k: v.realize().numpy() for k, v in o.items()}
    if _WHEEL_NOGOOD and _WHEEL.get("nogood"):
        _nogood_apply(onp, _WHEEL["nogood"])          # THE NOGOOD: refuted choices are not re-committed
    fat_np = fat.realize().numpy() if hasattr(fat, "realize") else _np.asarray(fat)
''')
    s = rep(s, '''        turned += 1
        core_slots = [parse[k]["_slot"] for k in _core]
        for j in core_slots:
            melt[b, j] = 1.0
        sents = {}
''', '''        turned += 1
        core_slots = [parse[k]["_slot"] for k in _core]
        if _WHEEL_NOGOOD:
            _pk = _nogood_pick(row, sorted(set(core_slots)))
            if _pk is not None:
                _j, _f, _c = _pk
                core_slots = [_j]                       # the least-confident member only
                _WHEEL.setdefault("nogood", []).append((b, _j, _f, _c))
        for j in core_slots:
            melt[b, j] = 1.0
        sents = {}
''')
    open(hp, "w").write(s); print("[apply] the nogood applied to", hp)
else:
    print("[apply] head already applied")
if "nogood" not in w:
    w = rep(w, '''    onp = {k: o[k].realize().numpy() for k in (("pres", "ftype", "op", "islit", "dig", "args", "res") + (("dup",) if "h_dup" in p else ()))}
    ALLST.extend(H._WHEEL["stats"]); ALLTU.extend(H._WHEEL["turned"]); H._WHEEL = None
''', '''    onp = {k: o[k].realize().numpy() for k in (("pres", "ftype", "op", "islit", "dig", "args", "res") + (("dup",) if "h_dup" in p else ()))}
    if getattr(H, "_WHEEL_NOGOOD", 0) and H._WHEEL.get("nogood"):
        H._nogood_apply(onp, H._WHEEL["nogood"])        # THE NOGOOD reaches the final read: the ranking filtered by the parse's own refutations
    ALLST.extend(H._WHEEL["stats"]); ALLTU.extend(H._WHEEL["turned"]); H._WHEEL = None
''')
    open(wp, "w").write(w); print("[apply] the nogood applied to", wp)
else:
    print("[apply] wheel_read already applied")
