"""macros.py — THE MACRO REGISTRY (grammar mg1; first admission 2026-07-16).

The hierarchical library's second floor, one entry tall. Constitution
(the recursion charter, ledger 2026-07-14/15 + the ladder constitution
2026-07-16): macros are ANNOTATION-side vocabulary — the parse target one
level up. Every macro expands DETERMINISTICALLY to primitive factors
before the solver sees anything; the answer key grades in primitives,
always; certification is level-invariant because the ground floor never
moves. The solver imports nothing from this module — ever.

Registry protocol: entries are ADMITTED (rank-never-admit — the miner
proposes by frequency, the review prices, admission is a ledger event),
carry the grammar version and their crown's WL digest (pinned by
scripts/macro_admission_review.py, root-marked, values abstracted), and
never change semantics under one grammar version — a semantic change is
a NEW version, because banked macro-annotated rows must re-expand
byte-identically forever.

Admitted entries:
  OP_APPLY (mg1, 2026-07-16) — r = k1*x <op> k2*y, op in {add, sub}.
    Evidence: harvest 4.9% of items vs train 0.26% (~19x over-represented
    in real prose); crown digest 916f019f77831ce0; 4 primitive factors
    absorbed per instance. Specimens: custom-operator problems
    ("a S b = 3a+5b"), coupled linear systems (pens-and-pencils,
    legs-and-heads). k=1 legs are legal and drop their given+mul on
    expansion (the affine form x + k*y is the same macro, not a second
    entry).
"""

MACRO_GRAMMAR_VERSION = "mg2"   # mg1 entries FROZEN; mg2 adds FRAC_OF

# Crown digests pinned by the admission review (root-marked WL canon of
# {given k1, mul, given k2, mul, root-op}, values abstracted).
OP_APPLY_CROWN_DIGEST = "916f019f77831ce0"


def expand_op_apply(f, next_var):
    """Expand one OP_APPLY macro factor into primitive factors.

    f = {"ftype": "macro", "name": "OP_APPLY", "op": "add"|"sub",
         "k1": int>=1, "x": var, "k2": int>=1, "y": var, "result": var}

    Returns (primitive_factors, next_var). Pure and deterministic:
    identical input -> byte-identical output. Temp vars are allocated
    consecutively from next_var; a k==1 leg contributes the bare operand
    (no given, no mul).
    """
    assert f["ftype"] == "macro" and f["name"] == "OP_APPLY"
    assert f["op"] in ("add", "sub"), f["op"]
    out = []
    legs = []
    for k_key, v_key in (("k1", "x"), ("k2", "y")):
        k, v = f[k_key], f[v_key]
        assert isinstance(k, int) and k >= 1, (k_key, k)
        if k == 1:
            legs.append(v)
            continue
        kv = next_var
        out.append({"ftype": "given", "var": kv, "value": k, "spans": []})
        mv = next_var + 1
        next_var += 2
        out.append({"ftype": "rel", "op": "mul", "args": [kv, v],
                    "result": mv, "spans": []})
        legs.append(mv)
    out.append({"ftype": "rel", "op": f["op"], "args": legs,
                "result": f["result"], "spans": []})
    return out, next_var


def expand_frac_of(f, next_var):
    """Expand one FRAC_OF macro: r = (a * x) // k  — THE FRACTION-OF BEND
    (mg2, admitted 2026-07-21 under the four-clause mandate: fdiv-absorbing,
    sliver-bounding, hexagonal, canyon-damping).

    f = {"ftype": "macro", "name": "FRAC_OF", "a": int>=1, "k": int>=2,
         "x": var, "result": var}
    a == 1 contributes the bare operand (no given, no mul) — pure fdiv
    absorbs as the crown's own edge case, mirroring OP_APPLY's k=1 leg.
    Deterministic; solution-preserving; the solver sees primes only.
    """
    assert f["ftype"] == "macro" and f["name"] == "FRAC_OF"
    a, k = f["a"], f["k"]
    assert isinstance(a, int) and a >= 1 and isinstance(k, int) and k >= 2
    out = []
    if a == 1:
        src_var = f["x"]
    else:
        av = next_var
        out.append({"ftype": "given", "var": av, "value": a, "spans": []})
        mv = next_var + 1
        next_var += 2
        out.append({"ftype": "rel", "op": "mul", "args": [av, f["x"]],
                    "result": mv, "spans": []})
        src_var = mv
    out.append({"ftype": "fdiv", "var": src_var, "k": k,
                "result": f["result"], "spans": []})
    return out, next_var


EXPANDERS = {"OP_APPLY": expand_op_apply, "FRAC_OF": expand_frac_of}


def expand_graph(factors, n_vars):
    """Expand every macro factor in a graph, in order, deterministically.

    Returns (primitive_factors, new_n_vars). Asserts the output is pure
    primitives — the solver-facing invariant, checked mechanically.
    """
    out, nv = [], n_vars
    for f in factors:
        if f.get("ftype") == "macro":
            exp, nv = EXPANDERS[f["name"]](f, nv)
            out.extend(exp)
        else:
            out.append(f)
    assert all(g.get("ftype") != "macro" for g in out)
    return out, nv
