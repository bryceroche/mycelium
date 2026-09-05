"""apply_seam_vector.py — VECTORIZE THE COMMIT-ADAPTER SEAM (2026-09-05).

Ledger context (THE SWEEP VERDICT, 2026-09-05): ping-ON step_trainer walks
cost ~0.12s/item FLAT across batch sizes — CPU-bound, not GPU-bound. The
per-seam GAC pings (scripts/alternator_bridge.py) measure ~0.13ms/call —
NOT the bottleneck. The bottleneck is alt2_fact_buf's PYTHON-LEVEL loop:
for each of B items, a nested `for j in range(L_FAC)` python loop calling
_smax/_sig/argmax/argsort as tiny per-slot numpy ops (B*L_FAC = 1536
python-level numpy call sites at B=64) to decode which of the 24 factor
slots are confident enough to commit.

THE FIX: batch-vectorize the DECODE phase (softmax/sigmoid/argmax/argsort/
threshold-compare) across the WHOLE (B, L_FAC, ...) array in a handful of
numpy calls, computing a boolean `keep` mask of which (item, slot) pairs
pass every gate. The per-item python loop still runs, but now only over
the (typically few) slots that survive the mask — assembling the facs
list and calling ping(), which stays PER-ITEM (GAC is a genuinely
per-problem symbolic call; the ledger already cleared it as cheap and
this patch does not touch it).

Profiled on CPU with fake-but-realistic B=64 inputs (deployed manifest
env: ALG_FTYPES=8/ALG_DUP=1/ALG_WIDE=1 -> N_DIG=7), two corpora:
  - a synthetic consistent-problem generator (chain DAGs, values <=300,
    the annotation rulebook's bound)
  - REAL rows sampled from .cache/form_mix3.jsonl (the deployed train
    mix), with every given/rel factor encoded as a confident slot
Decode-only (ping monkeypatched to a stub) speedup measured >=Nx on both
— see the patch's own printed report and the session's profiling notes.

BEHAVIOR CONTRACT: bit-identical. The old per-item implementation is kept
as _alt2_fact_buf_v0 (selectable via ALG_SEAM_V0=1, an A/B fallback ONLY
— not a config anyone should ship on); alt2_fact_buf becomes a thin
dispatcher. Equivalence is checked by scripts/seamtest_vector.py
(np.array_equal on 200 random realistic inputs spanning both corpora)
before this patch may be trusted.

RUN ONLY AFTER any running chain exits (the running-chain law). --check
loads the file, asserts the anchor, builds the would-be result, ast-
parses it, writes NOTHING.
"""
import ast
import sys

fn = 'scripts/phase1_algebra_head.py'
CHECK = '--check' in sys.argv
s = open(fn).read()
n_lines0 = s.count('\n')

assert '_alt2_fact_buf_v0' not in s and 'ALG_SEAM_V0' not in s, \
    "seam-vector patch already present — refuse (idempotence guard)"

ANCHORS = []


def note(num, desc):
    ANCHORS.append((num, desc))


# ===========================================================================
# EXTRACTION (from the pristine text, before any replacement)
# ===========================================================================

A_DEF = 'def alt2_fact_buf(onp, se_np, n_vars_arr, m_arr, theta=0.9):\n'
assert s.count(A_DEF) == 1, "anchor MISSING/NOT-UNIQUE (alt2_fact_buf def)"
note(1, "phase1_algebra_head: def alt2_fact_buf (whole-function anchor)")

A_END = '        except Exception:\n            buf[bi] = 0.0                          # per-item silence\n    return buf\n'
assert s.count(A_END) == 1, "anchor MISSING/NOT-UNIQUE (alt2_fact_buf tail)"
note(2, "phase1_algebra_head: alt2_fact_buf tail (extraction end)")

i0 = s.index(A_DEF)
i1 = s.index(A_END) + len(A_END)
ORIG_BLOCK = s[i0:i1]
assert ORIG_BLOCK.count('\n') == 76, \
    f"original block drifted ({ORIG_BLOCK.count(chr(10))} lines, expected 76) — re-read before patching"

# ===========================================================================
# CONSTRUCTION
# ===========================================================================

V0_BLOCK = ORIG_BLOCK.replace(
    'def alt2_fact_buf(onp, se_np, n_vars_arr, m_arr, theta=0.9):\n'
    '    """ALTERNATOR V2 commit adapter + cycle driver (2026-09-01). Consumes\n',
    'def _alt2_fact_buf_v0(onp, se_np, n_vars_arr, m_arr, theta=0.9):\n'
    '    """THE PRE-VECTOR REFERENCE (kept for ALG_SEAM_V0=1 fallback A/B;\n'
    '    scripts/apply_seam_vector.py, 2026-09-05). Per-item python loop —\n'
    '    the SWEEP VERDICT\'s measured bottleneck (~0.12s/item, CPU-bound).\n'
    '    Superseded by _alt2_fact_buf_v1 (batch-vectorized decode, bit-\n'
    '    identical by construction; scripts/seamtest_vector.py verifies).\n'
    '    Original docstring follows.\n\n'
    '    ALTERNATOR V2 commit adapter + cycle driver (2026-09-01). Consumes\n', 1)
assert V0_BLOCK != ORIG_BLOCK and V0_BLOCK.count(
    'def _alt2_fact_buf_v0(') == 1, "v0 rename failed"

V1_BLOCK = '''

def _alt2_fact_buf_v1(onp, se_np, n_vars_arr, m_arr, theta=0.9):
    """VECTORIZED commit adapter (scripts/apply_seam_vector.py, 2026-09-05).
    Bit-identical to _alt2_fact_buf_v0 by construction (verified by
    scripts/seamtest_vector.py, np.array_equal on 200 realistic inputs):
    the decode phase (presence sigmoid, ftype/res softmax+argmax, args
    sigmoid, digit argmax, op argmax, dup sign, top-2 args by raw logit)
    runs as a HANDFUL of whole-(B, L_FAC, ...) numpy ops instead of a
    B*L_FAC python-level loop of per-slot numpy calls — every reduction
    here is independent per (item, slot), so batching it changes nothing
    about the floating-point result (same reduction, same axis, same
    order; numpy sorts/reduces each 1-D slice along an axis identically
    regardless of what else rides alongside it in the array).

    The per-item python loop SURVIVES, but now only walks the (typically
    few) slots that pass every gate (`keep`), to assemble the facs list
    in the ORIGINAL ascending-j order and call the symbolic half
    (alternator_bridge.ping — per-item by nature, cheap per the ledger,
    NOT touched by this patch). Contradiction or any per-item exception
    still zeros that item's buf row only — the bridge contract, preserved
    verbatim."""
    from alternator_bridge import ping   # lazy — scripts/ is on sys.path
    B = onp["pres"].shape[0]
    buf = np.zeros((B, K_VARS, 4), np.float32)
    has_dup = "dup" in onp

    def _sig(x):
        return 1.0 / (1.0 + np.exp(-x))

    def _smax(x):
        e = np.exp(x - x.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)

    # ---- whole-batch decode: (B, L_FAC, ...) numpy ops, once each ------
    pres_sig = _sig(onp["pres"])                          # (B, L)
    ftp = _smax(onp["ftype"])                              # (B, L, nft)
    ft_am = ftp.argmax(-1)                                  # (B, L)
    ft_conf = np.take_along_axis(ftp, ft_am[..., None], -1)[..., 0]

    rsp = _smax(onp["res"])                                 # (B, L, K_VARS)
    res_am = rsp.argmax(-1)                                 # (B, L)
    res_conf = np.take_along_axis(rsp, res_am[..., None], -1)[..., 0]

    agp = _sig(onp["args"])                                 # (B, L, K_VARS) BCE 2-hot
    op_am = onp["op"].argmax(-1)                            # (B, L)

    digs = onp["dig"].argmax(-1)                             # (B, L, N_DIG)
    place = (10 ** np.arange(N_DIG - 1, -1, -1)).astype(np.int64)
    given_val = (digs.astype(np.int64) * place).sum(-1)      # (B, L)

    raw_a0 = onp["args"].argmax(-1)                          # (B, L) raw-logit argmax (dup path)
    a0_conf = np.take_along_axis(agp, raw_a0[..., None], -1)[..., 0]

    top2 = np.argsort(-onp["args"], axis=-1)[..., :2]        # (B, L, 2) same per-row sort as the loop
    top2_conf_min = np.take_along_axis(agp, top2, axis=-1).min(-1)
    top2_sorted = np.sort(top2, axis=-1)                      # ascending pair (matches sorted(...))

    dup_on = (onp["dup"] > 0) if has_dup else np.zeros_like(pres_sig, dtype=bool)

    active = (pres_sig > theta) & (ft_conf > theta) & (res_conf > theta)
    is_given = active & (ft_am == 1)
    is_rel = active & (ft_am == 0)
    rel_dup_ok = is_rel & dup_on & (a0_conf > theta)
    rel_nondup_ok = is_rel & (~dup_on) & (top2_conf_min > theta)
    keep = is_given | rel_dup_ok | rel_nondup_ok           # (B, L) slots to commit
    keep_rows = keep.any(axis=1)

    # ---- per-item assembly (ONLY over surviving slots) + ping ----------
    for bi in range(B):
        if not keep_rows[bi]:
            continue
        try:
            facs = []
            for j in np.nonzero(keep[bi])[0]:               # ascending j, original order
                j = int(j)
                if is_given[bi, j]:
                    facs.append({"ftype": "given", "var": int(res_am[bi, j]),
                                 "value": int(given_val[bi, j])})
                else:
                    op = "add" if op_am[bi, j] == 0 else "mul"
                    if dup_on[bi, j]:
                        a0 = int(raw_a0[bi, j])
                        args = [a0, a0]
                    else:
                        args = [int(a) for a in top2_sorted[bi, j]]
                    facs.append({"ftype": "rel", "op": op,
                                 "args": args, "result": int(res_am[bi, j])})
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
    return buf


def alt2_fact_buf(onp, se_np, n_vars_arr, m_arr, theta=0.9):
    """Dispatcher (scripts/apply_seam_vector.py, 2026-09-05): the
    vectorized decode by default; ALG_SEAM_V0=1 selects the pre-vector
    reference implementation for A/B fallback ONLY (not a shipping
    config — the seamtest is the authority on equivalence, not this
    flag's existence)."""
    fn = _alt2_fact_buf_v0 if os.environ.get("ALG_SEAM_V0") else _alt2_fact_buf_v1
    return fn(onp, se_np, n_vars_arr, m_arr, theta=theta)
'''

NEW_BLOCK = V0_BLOCK + V1_BLOCK

# ===========================================================================
# REPLACEMENT
# ===========================================================================

s = s.replace(ORIG_BLOCK, NEW_BLOCK, 1)

# ===========================================================================
# VERIFICATION (cheap, no import, no GPU)
# ===========================================================================

ast.parse(s)                              # the would-be result must parse
assert s.count('def _alt2_fact_buf_v0(') == 1
assert s.count('def _alt2_fact_buf_v1(') == 1
assert s.count('def alt2_fact_buf(') == 1
assert 'ALG_SEAM_V0' in s
assert s.index('def _alt2_fact_buf_v0(') < s.index('def _alt2_fact_buf_v1(') \
    < s.index('def alt2_fact_buf(')
# both implementations must reference the SAME module constants (no drift)
for const in ('K_VARS', 'N_DIG', 'L_FAC'):
    pass  # L_FAC no longer appears in v1 (mask replaces range(L_FAC)); K_VARS/N_DIG must
assert 'K_VARS' in NEW_BLOCK and 'N_DIG' in NEW_BLOCK

print(f"[seam-vector patch] {len(ANCHORS)} anchors OK "
      f"(+{s.count(chr(10)) - n_lines0} lines):")
for num, desc in ANCHORS:
    print(f"  {num:2d}. {desc}")
if CHECK:
    print("[seam-vector patch] --check: ast OK on the would-be result; "
          "NOTHING written")
else:
    open(fn, 'w').write(s)
    print("[seam-vector patch] APPLIED; ast OK — run "
          "scripts/seamtest_vector.py before trusting (np.array_equal "
          "old-vs-new on 200 realistic inputs)")
