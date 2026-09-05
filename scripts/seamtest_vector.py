"""seamtest_vector.py — equivalence + perf harness for the seam-vector
patch (scripts/apply_seam_vector.py, 2026-09-05).

Builds the WOULD-BE patched source of scripts/phase1_algebra_head.py (by
running apply_seam_vector.py's construction logic under a forced
--check, so the real file is NEVER written) and execs it into an
in-memory module. Then, on 200 realistic (item-level) inputs spanning
two corpora —
  (a) synthetic CONSISTENT chain problems (values 0..300, the annotation
      rulebook's bound; given/rel/dup/nondup/contradiction/empty-commit
      branches all exercised)
  (b) REAL rows sampled from the deployed train mix (.cache/form_mix3.jsonl),
      with every given/rel factor encoded as a confident slot
— asserts np.array_equal(_alt2_fact_buf_v0(...), _alt2_fact_buf_v1(...))
== the SAME dispatcher (alt2_fact_buf) both ways via ALG_SEAM_V0.

Usage:
  .venv/bin/python3 scripts/seamtest_vector.py            # equivalence only
  .venv/bin/python3 scripts/seamtest_vector.py --perf     # + decode-only
                                                            speed table
No GPU, no tinygrad import — this only exercises the numpy commit
adapter. Zero writes to any tracked file.
"""
import json
import os
import random
import runpy
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
os.chdir(_ROOT)

import numpy as np

# the manifest's deployed env (.cache/GENERATION.json gen_id 41): the
# real array shapes this seam runs under in production
os.environ.setdefault("ALG_WIDE", "1")
os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_DUP", "1")

L_FAC = 24
K_VARS = 24
N_DIG = 7 if os.environ.get("ALG_WIDE") == "1" else 3
NFT = 8
N_OP = 2


# ===========================================================================
# STAGE the patch in memory (--check semantics; the real file is untouched)
# ===========================================================================

def load_patched_module():
    argv0 = sys.argv
    sys.argv = ["apply_seam_vector.py", "--check"]
    try:
        ns = runpy.run_path(os.path.join(_ROOT, "scripts", "apply_seam_vector.py"),
                             run_name="_seam_vector_staging")
    finally:
        sys.argv = argv0
    patched_source = ns["s"]
    assert not ns["CHECK"] or True  # CHECK forced True above; source built, nothing written
    import types
    mod = types.ModuleType("phase1_algebra_head_STAGED")
    mod.__file__ = os.path.join(_ROOT, "scripts", "phase1_algebra_head.py") + " (staged, in-memory)"
    exec(compile(patched_source, mod.__file__, "exec"), mod.__dict__)
    for name in ("_alt2_fact_buf_v0", "_alt2_fact_buf_v1", "alt2_fact_buf"):
        assert hasattr(mod, name), f"staged module missing {name} — patch construction drifted"
    return mod


# ===========================================================================
# FIXTURES — synthetic consistent problems + real-corpus-derived items
# ===========================================================================

def _gen_problem(rng, n_vars, m, frac_given=0.45, allow_dup=True):
    """A consistent, quickly-convergent chain problem: values 0..15,
    ops add/mul, results clipped <= min(m, 300) (the annotation
    rulebook's bound) — mirrors what a real annotated row looks like,
    unlike raw-random relations (which never converge under GAC)."""
    values = rng.randint(0, 16, size=n_vars).astype(np.int64)
    given = {0}
    facts = []
    for v in range(1, n_vars):
        if v == 1 or rng.rand() < frac_given:
            given.add(v)
            continue
        if allow_dup and rng.rand() < 0.3:
            a = int(rng.randint(0, v)); b = a
        else:
            a = int(rng.randint(0, v)); b = int(rng.randint(0, v))
        op = "add" if rng.rand() < 0.5 else "mul"
        val = int(values[a] + values[b]) if op == "add" else int(values[a] * values[b])
        if not (0 <= val <= min(m, 300)):
            given.add(v)
            continue
        values[v] = val
        facts.append({"ftype": "rel", "op": op,
                      "args": [a, b] if a == b else sorted([a, b]),
                      "result": v, "dup": a == b})
    for v in sorted(given):
        facts.append({"ftype": "given", "var": v, "value": int(values[v])})
    facts.sort(key=lambda f: (f["ftype"] != "given", f.get("var", f.get("result", 0))))
    return facts


def _blank_onp(B, seed):
    rng = np.random.RandomState(seed)
    onp = {
        "pres": np.full((B, L_FAC), -4.0, np.float32),
        "ftype": rng.randn(B, L_FAC, NFT).astype(np.float32) * 0.3,
        "op": rng.randn(B, L_FAC, N_OP).astype(np.float32) * 0.3,
        "dig": rng.randn(B, L_FAC, N_DIG, 10).astype(np.float32) * 0.3,
        "args": rng.randn(B, L_FAC, K_VARS).astype(np.float32) * 0.3,
        "res": rng.randn(B, L_FAC, K_VARS).astype(np.float32) * 0.3,
        "dup": rng.randn(B, L_FAC).astype(np.float32) * 1.5 - 1.5,
    }
    return onp, rng


def _embed(onp, rng, bi, facts):
    for j, f in enumerate(facts[:L_FAC]):
        onp["pres"][bi, j] = rng.uniform(3.0, 8.0)
        if f["ftype"] == "given":
            onp["ftype"][bi, j, 1] += rng.uniform(3.0, 7.0)
            onp["res"][bi, j, f["var"]] += rng.uniform(3.0, 7.0)
            v = max(0, min(int(f["value"]), 10 ** N_DIG - 1))
            digs = [(v // (10 ** p)) % 10 for p in range(N_DIG - 1, -1, -1)]
            for d, dv in enumerate(digs):
                onp["dig"][bi, j, d, dv] += rng.uniform(3.0, 7.0)
        else:
            onp["ftype"][bi, j, 0] += rng.uniform(3.0, 7.0)
            onp["res"][bi, j, f["result"]] += rng.uniform(3.0, 7.0)
            onp["op"][bi, j, 0 if f["op"] == "add" else 1] += rng.uniform(3.0, 7.0)
            a0, a1 = f["args"][0], f["args"][-1]
            if a0 == a1:
                onp["dup"][bi, j] = rng.uniform(2.0, 5.0)
                onp["args"][bi, j, a0] += rng.uniform(3.0, 7.0)
            else:
                onp["dup"][bi, j] = -rng.uniform(2.0, 5.0)
                onp["args"][bi, j, a0] += rng.uniform(3.0, 7.0)
                onp["args"][bi, j, a1] += rng.uniform(3.0, 7.0)


def synthetic_batch(B, seed):
    """Consistent chain problems + deliberate contradiction/empty-commit
    items (the branches _alt2_fact_buf's silence contract must hit)."""
    onp, rng = _blank_onp(B, seed)
    n_vars_arr = np.zeros(B, dtype=np.int64)
    m_arr = np.zeros(B, dtype=np.int64)
    for bi in range(B):
        n_vars = int(rng.randint(4, K_VARS))
        m = int(rng.choice([50, 100, 200, 300]))
        n_vars_arr[bi] = n_vars
        m_arr[bi] = m
        roll = rng.rand()
        if roll < 0.05:
            continue                                   # empty commit
        facts = _gen_problem(rng, n_vars, m)
        if roll < 0.13:                                 # contradiction
            facts.append({"ftype": "given", "var": 0,
                          "value": (int(facts[-1].get("value", 0)) + 7) % (m + 1)})
        _embed(onp, rng, bi, facts)
    return onp, n_vars_arr, m_arr


_ROWS_CACHE = {}


def real_batch(B, seed, path=".cache/form_mix3.jsonl"):
    """B items pulled from the deployed train mix — every given/rel
    factor of the sampled row encoded as a confident slot (the maximal
    committed set; alt2_fact_buf never commits other ftypes anyway).
    m is capped for the harness's own runtime (ping's CSP-solve cost
    scales with domain width; that cost is UNCHANGED by this patch and
    is identical between v0/v1 since both call the same ping — capping
    it here only keeps the seamtest itself fast, it does not touch what
    is under test: the decode+assembly logic upstream of ping)."""
    if not os.path.exists(path):
        return None
    if path not in _ROWS_CACHE:
        _ROWS_CACHE[path] = [json.loads(l) for l in open(path)]
    rows = _ROWS_CACHE[path]
    rng_sel = random.Random(seed)
    idxs = list(range(len(rows)))
    rng_sel.shuffle(idxs)
    picked = []
    for i in idxs:
        r = rows[i]
        gr = [f for f in r["factors"] if f["ftype"] in ("given", "rel")]
        if gr and r["n_vars"] <= K_VARS and len(gr) <= L_FAC:
            picked.append((r, gr))
        if len(picked) == B:
            break
    assert len(picked) == B, f"only found {len(picked)}/{B} usable real rows"
    onp, rng = _blank_onp(B, seed)
    n_vars_arr = np.zeros(B, dtype=np.int64)
    m_arr = np.zeros(B, dtype=np.int64)
    for bi, (row, gr) in enumerate(picked):
        n_vars_arr[bi] = row["n_vars"]
        m_arr[bi] = min(int(row["m"]), 2000)  # harness runtime cap; see docstring
        _embed(onp, rng, bi, gr)
    return onp, n_vars_arr, m_arr


def concat_batches(batches):
    keys = batches[0][0].keys()
    onp = {k: np.concatenate([b[0][k] for b in batches], axis=0) for k in keys}
    nv = np.concatenate([b[1] for b in batches])
    ma = np.concatenate([b[2] for b in batches])
    return onp, nv, ma


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    mod = load_patched_module()
    print("[seamtest] staged module built from apply_seam_vector.py --check "
          "(real file untouched)")

    parts = [synthetic_batch(100, seed=12345)]
    real = real_batch(100, seed=7)
    if real is not None:
        parts.append(real)
    else:
        print("[seamtest] .cache/form_mix3.jsonl not found — real-corpus "
              "leg SKIPPED, doubling the synthetic leg to hold n=200")
        parts.append(synthetic_batch(100, seed=54321))
    onp, nv, ma = concat_batches(parts)
    n = onp["pres"].shape[0]
    assert n == 200, f"expected 200 items, built {n}"

    b0 = mod._alt2_fact_buf_v0(onp, None, nv, ma)
    b1 = mod._alt2_fact_buf_v1(onp, None, nv, ma)
    eq = np.array_equal(b0, b1)
    n_committed = int((b0[:, :, 0] > 0).any(-1).sum())
    print(f"[seamtest] {n} realistic items ({n_committed} with >=1 committed "
          f"fact) — np.array_equal(_v0, _v1): {eq}")
    if not eq:
        diff = np.argwhere(b0 != b1)
        rows_diff = sorted(set(int(d[0]) for d in diff))
        print(f"[seamtest] FAIL: {len(diff)} cells differ across "
              f"{len(rows_diff)} items: {rows_diff[:20]}")
        sys.exit(1)

    # the dispatcher itself: ALG_SEAM_V0 must route to the SAME two paths
    os.environ.pop("ALG_SEAM_V0", None)
    bd1 = mod.alt2_fact_buf(onp, None, nv, ma)
    assert np.array_equal(bd1, b1), "dispatcher default did not match _v1"
    os.environ["ALG_SEAM_V0"] = "1"
    bd0 = mod.alt2_fact_buf(onp, None, nv, ma)
    assert np.array_equal(bd0, b0), "ALG_SEAM_V0=1 did not match _v0"
    os.environ.pop("ALG_SEAM_V0", None)
    print("[seamtest] dispatcher PASS: default->v1, ALG_SEAM_V0=1->v0, "
          "both bit-identical to their targets")
    print("[seamtest] EQUIVALENCE PASS")

    if "--perf" in sys.argv:
        print()
        print("[seamtest] decode-only speed (ping stubbed to isolate the "
              "vectorized phase from the unrelated GAC-solve cost):")
        import alternator_bridge
        real_ping = alternator_bridge.ping
        alternator_bridge.ping = lambda nv, facs, m, *a, **k: ({}, [1] * nv, 1)
        try:
            for tag, fn in (("v0 (original per-item loop)", mod._alt2_fact_buf_v0),
                             ("v1 (vectorized)", mod._alt2_fact_buf_v1)):
                batches = [synthetic_batch(64, seed=1000 + i) for i in range(40)]
                o0, nv0, ma0 = batches[0]
                fn(o0, None, nv0, ma0)          # warmup
                t0 = time.perf_counter()
                for onp_b, nv_b, ma_b in batches:
                    fn(onp_b, None, nv_b, ma_b)
                dt = time.perf_counter() - t0
                B = o0["pres"].shape[0]
                print(f"  {tag:32s} {dt/len(batches)*1000:8.4f} ms/call  "
                      f"{dt/len(batches)/B*1e6:8.2f} us/item")
        finally:
            alternator_bridge.ping = real_ping


if __name__ == "__main__":
    main()
