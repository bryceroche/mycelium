"""apply_router_graded.py — THE ROUTER GRADED-INPUT REPAIR (2026-09-03,
word given). The organ capacity audit's headline: the router conditions
inside the breath loop on the HARD one-hot snap (`_oh4`, argmax over the
codebook logits) while the GRADED logits `_lg4` sit one line earlier —
a boolean-where-graded-signal-exists, the same representability
pathology that starved dynamic masking (docs/organ_capacity_audit.md).

THE FIX (env ALG_ROUTER_GRADED=1): build a parallel GRADED snap tuple —
softmax over `_lg4` instead of argmax — detached (dual-terminal: the
snap is a fact; the router's own params still earn gradient downstream),
and feed the router's snap-conditioning `_sf7` from it. The CANONICAL
SHELF deposit stays HARD and untouched (its detached one-hot IS the fact
the solver reads; only the router's input is graded). Env unset =
bit-identical. Staged patch, apply_*.py style: numbered anchors,
s.replace(...,1), ast.parse, --check dry-run.
"""
import ast
import sys

FN = "scripts/phase1_algebra_head.py"


def patch(s):
    n = 0

    # 1. parallel graded-snap list, beside _snaps
    old1 = "    _snaps = []"
    new1 = ("    _snaps = []\n"
            "    _snaps_g = []   # router graded-input repair: softmax "
            "snap tuple (ALG_ROUTER_GRADED)")
    assert old1 in s, "anchor 1 (_snaps init)"
    s = s.replace(old1, new1, 1); n += 1

    # 2. graded collection inside the per-role snap loop, right after the
    #    hard one-hot is sliced. Anchor on the op-role slice (last one).
    old2 = "                        elif _rn4 == \"op\":\n" \
           "                            _snap_g5 = _oh4[..., 25]   # ftype 'given' code"
    new2 = ("                        elif _rn4 == \"op\":\n"
            "                            _snap_g5 = _oh4[..., 25]   # ftype 'given' code\n"
            "                        if int(os.environ.get(\"ALG_ROUTER_GRADED\", \"0\")):\n"
            "                            _sg4 = _lg4.softmax(-1)   # GRADED: the\n"
            "                            # confidence distribution the argmax throws away\n"
            "                            if _rn4 == \"arg1\": _grad_a5 = _sg4[..., :24]\n"
            "                            elif _rn4 == \"arg2\": _grad_b5 = _sg4[..., :24]\n"
            "                            elif _rn4 == \"res\": _grad_r5 = _sg4[..., :24]\n"
            "                            elif _rn4 == \"op\": _grad_g5 = _sg4[..., 25]")
    assert old2 in s, "anchor 2 (per-role graded slice)"
    s = s.replace(old2, new2, 1); n += 1

    # 3. append the graded tuple alongside the hard one (same guard set)
    old3 = "                        _snaps.append((_snap_a5.detach(), _snap_b5.detach(),\n" \
           "                                       _snap_r5.detach(), _snap_g5.detach()))"
    new3 = ("                        _snaps.append((_snap_a5.detach(), _snap_b5.detach(),\n"
            "                                       _snap_r5.detach(), _snap_g5.detach()))\n"
            "                        if int(os.environ.get(\"ALG_ROUTER_GRADED\", \"0\")):\n"
            "                            _snaps_g.append((_grad_a5.detach(),\n"
            "                                _grad_b5.detach(), _grad_r5.detach(),\n"
            "                                _grad_g5.detach()))")
    assert old3 in s, "anchor 3 (snap append)"
    s = s.replace(old3, new3, 1); n += 1

    # 4. router reads the graded tuple when armed (dual: _snaps_g is
    #    populated only when the env is set AND a snap exists)
    old4 = "                if _snaps:\n" \
           "                    _sf7 = Tensor.cat(_snaps[-1][0], _snaps[-1][1],\n" \
           "                                      _snaps[-1][2],\n" \
           "                                      _snaps[-1][3].unsqueeze(-1), dim=-1)\n" \
           "                    _cq7 = cur + _sf7 @ p[\"W_rs\"]"
    new4 = "                if _snaps:\n" \
           "                    _src7 = (_snaps_g[-1] if (_snaps_g and\n" \
           "                             int(os.environ.get(\"ALG_ROUTER_GRADED\", \"0\")))\n" \
           "                             else _snaps[-1])\n" \
           "                    _sf7 = Tensor.cat(_src7[0], _src7[1],\n" \
           "                                      _src7[2],\n" \
           "                                      _src7[3].unsqueeze(-1), dim=-1)\n" \
           "                    _cq7 = cur + _sf7 @ p[\"W_rs\"]"
    assert old4 in s, "anchor 4 (router _sf7 source)"
    s = s.replace(old4, new4, 1); n += 1

    return s, n


def main():
    s = open(FN).read()
    if "_snaps_g" in s and "--check" not in sys.argv:
        print("[router-graded] already applied (_snaps_g present); refusing")
        return
    s2, n = patch(s)
    ast.parse(s2)
    if "--check" in sys.argv:
        print(f"[router-graded] --check: {n} anchors OK; ast OK on the "
              f"would-be result; NOTHING written")
        return
    open(FN, "w").write(s2)
    print(f"[router-graded] APPLIED ({n} anchors); ast OK — run eq_check "
          f"(ALG_ROUTER_GRADED unset must be bit-identical)")


if __name__ == "__main__":
    main()
