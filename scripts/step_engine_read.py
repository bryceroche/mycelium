"""step_engine_read.py — THE STEP-ALTERNATION READ ENGINE v0 (2026-09-03).

ITERATED RE-EXECUTION (the coordinator's ruling): v0 approximates per-step
alternation by running the FULL fused forward repeatedly. Pass 0 runs open
(no mask, no facts — loop_val's pass 1); after EVERY pass the commit
adapter (alt2_fact_buf) runs on the realized outputs, the bridge pings
(inside alt2_fact_buf), and the next pass re-runs with the updated
slot_mask (build_slot_masks on the latest outputs) + updated fact_buf.
Iterates to fixed point (next pass's inputs identical to the previous
pass's -> outputs identical by determinism, forward skipped) or SE_R
conditioned passes. Reuses ONLY the proven ALTMASK/fact_buf seams — zero
surgery on forward(), zero new JIT captures. True per-breath injection is
v1, explicitly out of scope.

RUNG 1 BY CONSTRUCTION: SE_R counts CONDITIONED passes (total forwards =
SE_R + 1). With SE_R=1 this engine IS loop_val.py's two-pass fact-fed
read exactly: pass 0 open -> build_slot_masks + alt2_fact_buf -> pass 1
forward(slot_mask, fact_buf) -> score pass 1 with loop_val's EXACT
fac-exact criterion (copied line-for-line below, dup branch included).
The chain verifies by comparing the printed final fac-exact vs loop_val
on the same ckpt/fixture/envs.

W_FACT-ABSENT HANDLING: fact injection needs the ckpt to carry
W_fact/alt2_g (alt2-era ckpts do; the champion alt21warm242 does NOT).
If the ckpt lacks W_fact (checked by key presence, never a crash) the
engine still runs the mask-iteration loop; facts are computed and
REPORTED per pass (the leak gauge, Blackbird profile) but not injected,
and a one-line notice prints. Injection additionally requires ALG_ALT2=1
in the caller's env (forward()'s own guard).

ATLAS HOOK (optional, diagnostic only): if .cache/step_atlas_current.npz
exists, ALG_INV=1 is set so forward returns fst_s (final factor-slot
states; read-only key, no params), and step_atlas.load_atlas + consult
run on the FINAL pass's mean-pooled states at the last breath_step; the
mean top-1 consult distance prints per decisions-band bucket. No
conditioning in v0 (the conditioning port needs mask-head-era training).

Envs: SE_CKPT (checkpoint), SE_R (conditioned passes, default 3),
SE_THETA (commit threshold, default 0.9) + the caller's ALG_* envs
(reads must carry the TRAINED env, always).

CPU test: --selftest exercises the pure-python bookkeeping (scoring
criterion, fact gauge, fixed-point iteration) on faked numpy inputs and
verifies imports clean — no forward, no GPU, no ckpt.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import numpy as np

from phase1_algebra_head import L_FAC   # CPU-safe: no module-level tinygrad

ATLAS_PATH = ".cache/step_atlas_current.npz"


def score_batch(onp, vg, sl):
    """loop_val.py's EXACT fac-exact criterion, copied line-for-line
    (including the dup branch). Returns (n_ok, n_tot) for this batch;
    padded rows are excluded because sl (not sl_p) drives the loop."""
    n_ok = n_tot = 0
    for bi, i in enumerate(sl):
        i = int(i)
        for j in range(L_FAC):
            if vg["presence"][i, j] < 0.5:
                continue
            n_tot += 1
            ok = (onp["pres"][bi, j] > 0)
            ok &= int(onp["ftype"][bi, j].argmax()) == vg["ftype"][i, j]
            ok &= int(onp["res"][bi, j].argmax()) == vg["res"][i, j]
            if vg["ftype"][i, j] == 0:
                ok &= int(onp["op"][bi, j].argmax()) == vg["op"][i, j]
                gset = set(np.where(vg["args"][i, j] > .5)[0].tolist())
                if len(gset) == 1 and "dup" in onp:
                    ok &= bool(onp["dup"][bi, j] > 0)
                    ok &= int(np.argmax(onp["args"][bi, j])) in gset
                else:
                    top2 = set(np.argsort(-onp["args"][bi, j])[:2].tolist())
                    ok &= top2 == gset
            else:
                ok &= bool((onp["dig"][bi, j].argmax(-1) ==
                            vg["digits"][i, j]).all())
            n_ok += ok
    return n_ok, n_tot


def count_facts(fb, n_real):
    """The leak gauge: forced vars in the commit buffer, real rows only
    (fb[:, :, 0] is the known flag)."""
    return int((fb[:n_real, :, 0] > 0).sum())


def _inputs_equal(a, b):
    """(mask, fact) input-pair equality — the fixed-point test. fact may
    be None on either side (not injectable / open pass)."""
    ma, fa = a
    mb, fb = b
    if (fa is None) != (fb is None):
        return False
    if not np.array_equal(ma, mb):
        return False
    return fa is None or np.array_equal(fa, fb)


def iterate_batch(run_pass, commit, R, injectable):
    """The engine's iteration bookkeeping, forward-agnostic (--selftest
    fakes run_pass/commit; the real path wraps forward + realize).

    run_pass(mask_or_None, fact_or_None) -> realized output dict
    commit(onp) -> (slot_mask, fact_buf) from that pass's outputs

    Pass 0 runs open. Conditioned pass r (1..R) runs with pass r-1's
    mask (+ fact when injectable). If a conditioned pass's inputs equal
    the previous conditioned pass's, the forward output is identical by
    determinism: the pass is skipped and outputs reused (fixed point).

    Returns (outs, facts, conv_at): outs[r] = pass r's output dict,
    facts[r] = commit buffer computed FROM pass r's outputs (reported
    every pass, fed forward only when injectable), conv_at = first
    conditioned pass at fixed point (>= 2) or None."""
    onp = run_pass(None, None)                     # pass 0: open
    outs = [onp]
    mk, fb = commit(onp)
    facts = [fb]
    last_in = None
    conv_at = None
    for r in range(1, R + 1):
        cur_in = (mk, fb if injectable else None)
        if last_in is not None and _inputs_equal(cur_in, last_in):
            if conv_at is None:
                conv_at = r
            outs.append(outs[-1])                  # identical by determinism
            facts.append(facts[-1])
            continue
        onp = run_pass(mk, fb if injectable else None)
        outs.append(onp)
        mk, fb = commit(onp)
        facts.append(fb)
        last_in = cur_in
    return outs, facts, conv_at


def main():
    from phase1_algebra_head import (build_params, forward, load_alg,
                                     build_slot_masks, alt2_fact_buf, K_VARS)
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load

    R = int(os.environ.get("SE_R", "3"))
    THETA = float(os.environ.get("SE_THETA", "0.9"))
    ckpt = os.environ["SE_CKPT"]
    atlas_on = os.path.exists(ATLAS_PATH)
    if atlas_on:
        # read-only key request: forward adds out["fst_s"] under ALG_INV;
        # no params, no loss at read — the atlas consult's state source
        os.environ["ALG_INV"] = "1"

    sd = safe_load(ckpt)
    has_wfact = "W_fact" in sd
    alt2_env = bool(int(os.environ.get("ALG_ALT2", "0")))
    injectable = alt2_env and has_wfact
    if not has_wfact:
        print("[step-engine] NOTICE: ckpt lacks W_fact/alt2_g — facts "
              "computed and REPORTED (leak gauge) but NOT injectable; "
              "iterating slot masks only")
    elif not alt2_env:
        print("[step-engine] NOTICE: ckpt carries W_fact but ALG_ALT2 is "
              "unset — injection off (forward's guard); leak gauge only")

    vs, vst, vtk, vg, vse = load_alg("test")
    p = build_params(0)
    assert set(sd.keys()) == set(p.keys()), \
        (sorted(set(sd) - set(p))[:4], sorted(set(p) - set(sd))[:4])
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

    kset = ("fat", "pres", "ftype", "op", "islit", "dig", "args", "res") \
        + (("dup",) if "h_dup" in p else ()) \
        + (("fst_s",) if atlas_on else ())
    n_ok = np.zeros(R + 1, np.int64)
    n_facts = np.zeros(R + 1, np.int64)
    n_tot = 0
    conv_hist = np.zeros(R + 2, np.int64)          # [R+1] = never converged
    n_batches = 0
    atlas_states, atlas_bands = [], []

    for s0 in range(0, len(vs), 8):
        sl = np.arange(s0, min(s0 + 8, len(vs)))
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        ts = Tensor(vst[sl_p].astype(np.float32), dtype=dtypes.float)
        tk = Tensor(vtk[sl_p].astype(np.float32), dtype=dtypes.float)
        se = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)

        def run_pass(mk, fb):
            o = forward(p, ts, tk, se,
                        slot_mask=(None if mk is None
                                   else Tensor(mk, dtype=dtypes.float)),
                        fact_buf=(None if fb is None
                                  else Tensor(fb, dtype=dtypes.float)))
            return {k: o[k].realize().numpy() for k in kset}

        def commit(onp):
            mk = build_slot_masks(onp, vse[sl_p].astype(np.int32))
            _nv = np.array([vs[int(i)].get("n_vars", K_VARS) for i in sl_p])
            _ma = np.array([vs[int(i)].get("m", 0) for i in sl_p])
            fb = alt2_fact_buf(onp, vse[sl_p].astype(np.int32), _nv, _ma,
                               theta=THETA)
            return mk, fb

        outs, facts, conv_at = iterate_batch(run_pass, commit, R, injectable)
        for r in range(R + 1):
            ok_r, tot_r = score_batch(outs[r], vg, sl)
            n_ok[r] += ok_r
            if r == 0:
                n_tot += tot_r
            n_facts[r] += count_facts(facts[r], len(sl))
        conv_hist[conv_at if conv_at is not None else R + 1] += 1
        n_batches += 1
        if atlas_on:
            atlas_states.append(outs[R]["fst_s"][:len(sl)].mean(axis=1))
            atlas_bands.append(vg["band"][sl])

    feed = "fed to next pass" if injectable else "gauge only"
    for r in range(R + 1):
        tag = "open" if r == 0 else f"cond{r}"
        print(f"[step-engine] pass {r} ({tag}): "
              f"fac-exact={n_ok[r] / max(n_tot, 1):.4f} "
              f"facts={int(n_facts[r])} forced vars ({feed})")
    conv = " ".join(f"p{r}:{int(conv_hist[r])}" for r in range(2, R + 1))
    print(f"[step-engine] fixed-point batches by pass: "
          f"{conv or '(R<2: n/a)'} never:{int(conv_hist[R + 1])} "
          f"of {n_batches}")
    print(f"[step-engine] {ckpt} R={R} theta={THETA} "
          f"inject={int(injectable)} "
          f"final fac-exact={n_ok[R] / max(n_tot, 1):.4f} (n={n_tot})")

    if atlas_on:
        try:
            from mycelium.step_atlas import load_atlas, consult, K_STEPS
            atlas = load_atlas(ATLAS_PATH)
            S = np.concatenate(atlas_states).astype(np.float64)
            Bd = np.concatenate(atlas_bands)
            idx, dist, _ = consult(atlas, K_STEPS - 1, S, k=1)
            for b in sorted(set(int(x) for x in Bd)):
                m = Bd == b
                print(f"[step-engine] atlas band={b}: mean top-1 "
                      f"dist={float(dist[m, 0].mean()):.4f} "
                      f"(n={int(m.sum())})")
        except Exception as e:
            print(f"[step-engine] atlas consult SKIPPED (diagnostic): {e}")


def selftest():
    """CPU-only: py_compile + faked-numpy checks of the pure-python
    pieces (scoring criterion incl. dup branch, leak gauge, fixed-point
    iteration bookkeeping). Never touches forward, ckpts, or the GPU."""
    import py_compile
    py_compile.compile(os.path.abspath(__file__), doraise=True)
    import phase1_algebra_head                      # imports clean, no GPU
    assert not hasattr(phase1_algebra_head, "Tensor")   # no module-level GPU

    # --- score_batch: 2 items, 3 gold factors, known outcome ---
    B, KV, ND = 2, 24, 3
    vg = {"presence": np.zeros((B, L_FAC)), "ftype": np.zeros((B, L_FAC), int),
          "res": np.zeros((B, L_FAC), int), "op": np.zeros((B, L_FAC), int),
          "args": np.zeros((B, L_FAC, KV)), "digits": np.zeros((B, L_FAC, ND), int)}
    # item 0, slot 0: rel add(1,2)->3 ; slot 1: given var4 = 042
    vg["presence"][0, 0] = 1; vg["ftype"][0, 0] = 0; vg["op"][0, 0] = 0
    vg["args"][0, 0, [1, 2]] = 1; vg["res"][0, 0] = 3
    vg["presence"][0, 1] = 1; vg["ftype"][0, 1] = 1; vg["res"][0, 1] = 4
    vg["digits"][0, 1] = [0, 4, 2]
    # item 1, slot 0: rel dup mul(5,5)->6
    vg["presence"][1, 0] = 1; vg["ftype"][1, 0] = 0; vg["op"][1, 0] = 1
    vg["args"][1, 0, 5] = 1; vg["res"][1, 0] = 6
    onp = {"pres": np.full((B, L_FAC), -2.0), "ftype": np.zeros((B, L_FAC, 3)),
           "op": np.zeros((B, L_FAC, 2)), "args": np.full((B, L_FAC, KV), -3.0),
           "res": np.zeros((B, L_FAC, KV)), "dig": np.zeros((B, L_FAC, ND, 10)),
           "dup": np.full((B, L_FAC), -1.0)}
    onp["pres"][0, [0, 1]] = 2.0; onp["pres"][1, 0] = 2.0
    onp["ftype"][0, 0, 0] = 5; onp["ftype"][0, 1, 1] = 5; onp["ftype"][1, 0, 0] = 5
    onp["op"][0, 0, 0] = 5; onp["op"][1, 0, 1] = 5
    onp["args"][0, 0, [1, 2]] = 3.0; onp["args"][1, 0, 5] = 3.0
    onp["res"][0, 0, 3] = 5; onp["res"][0, 1, 4] = 5; onp["res"][1, 0, 6] = 5
    onp["dig"][0, 1, 0, 0] = 5; onp["dig"][0, 1, 1, 4] = 5
    onp["dig"][0, 1, 2, 7] = 5                      # last digit WRONG (7!=2)
    onp["dup"][1, 0] = 1.0
    ok, tot = score_batch(onp, vg, np.array([0, 1]))
    assert (ok, tot) == (2, 3), (ok, tot)           # given fails on digits
    onp2 = {k: v for k, v in onp.items() if k != "dup"}
    ok2, tot2 = score_batch(onp2, vg, np.array([0, 1]))
    assert (ok2, tot2) == (1, 3), (ok2, tot2)       # dup gold needs dup head
    onp["dig"][0, 1, 2, 7] = 0; onp["dig"][0, 1, 2, 2] = 5
    ok3, _ = score_batch(onp, vg, np.array([0, 1]))
    assert ok3 == 3, ok3                            # digits fixed -> all pass

    # --- count_facts: the leak gauge ignores padded rows ---
    fb = np.zeros((2, KV, 4), np.float32)
    fb[0, 3, 0] = 1.0; fb[1, 5, 0] = 1.0; fb[1, 6, 0] = 1.0
    assert count_facts(fb, 2) == 3 and count_facts(fb, 1) == 1

    # --- _inputs_equal ---
    m = np.ones((2, 2)); f = np.zeros((2, 3))
    assert _inputs_equal((m, f), (m.copy(), f.copy()))
    assert _inputs_equal((m, None), (m.copy(), None))
    assert not _inputs_equal((m, None), (m, f))
    assert not _inputs_equal((m, f), (m + 1, f))

    # --- iterate_batch: fixed point detected, forward skipped after ---
    calls = {"n": 0}

    def fake_run(mk, fb):
        calls["n"] += 1
        return {"v": 0 if mk is None else int(mk[0])}

    def fake_commit(onp):
        nxt = min(onp["v"] + 1, 2)                  # commits saturate at 2
        return np.array([nxt]), np.full((1, KV, 4), float(nxt), np.float32)

    outs, facts, conv_at = iterate_batch(fake_run, fake_commit, 4, False)
    assert [o["v"] for o in outs] == [0, 1, 2, 2, 2]
    assert conv_at == 3 and calls["n"] == 3         # passes 3-4 reused
    assert np.array_equal(facts[3], facts[2])
    # injectable path: same trajectory, fact arrays join the equality test
    calls["n"] = 0
    outs, _, conv_at = iterate_batch(fake_run, fake_commit, 3, True)
    assert [o["v"] for o in outs] == [0, 1, 2, 2] and conv_at == 3
    # SE_R=1 shape: exactly two passes (open + one conditioned), no skip
    calls["n"] = 0
    outs, facts, conv_at = iterate_batch(fake_run, fake_commit, 1, False)
    assert len(outs) == 2 and conv_at is None and calls["n"] == 2

    print("[step-engine] selftest PASS: score criterion (incl. dup + "
          "no-dup branches), leak gauge, input equality, fixed-point "
          "iteration + reuse, SE_R=1 two-pass shape; imports clean, "
          "zero GPU")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
