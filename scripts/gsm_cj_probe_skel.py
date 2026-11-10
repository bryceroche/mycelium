"""the CJ probe's skeleton (the gold (op, kind a, kind b) sequence), importable without the model stack (zero-GPU reads)."""
K = 12


def first_kind(S):
    return "t" if S["t"] else ("s" if S["s"] else "c")


def skeleton(it):
    return tuple((o, first_kind(A), first_kind(B)) for o, A, B in it["prog"][:K])


def seen(train):
    return {skeleton(it) for it in train}
