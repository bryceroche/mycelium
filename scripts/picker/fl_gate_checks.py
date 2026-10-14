"""fl_gate_checks.py -- THE FLAGGED LOOP's gate-only CPU checks (ALG_FLAG; FL_241, ARM 1 of
the amended denoising-breath design, ledger 2026-10-12 12:35/12:59). Three checks against
PMS8_241's OWN trained checkpoint (.cache/sharp_PMS8_241.safetensors, pad-warmed -- only
"flag_emb" is missing and stays zero-init, exactly ALG_FLAG's own "road, not a gain" promise)
on its OWN training distribution (form_mix_pm35c.jsonl / formpm35c), using
FAMILY_ENVS["PMS8_241"] + ALG_FLAG=1 (adaptive_stop.py) -- the exact env/body dn_read_c.py's
own gym-replay mechanism already proved working, reused here rather than a hand-rolled
minimal env.

WHY A TRAINED BODY, AND WHY form_mix_pm35c (NOT wildhold): a 2026-10-12 CPU smoke found (a) a
fresh-random-init / balV242-warm-started body's grounding attention does NOT concentrate on
the gold mention span at all (an untrained "fq" bank routes ~uniformly; 0.0055 mass on a
256-token row, versus 1/256 = 0.0039 uniform -- noise, not grounding) -- tabu-ing a span the
attention never used in the first place cannot move its argmax, by construction, regardless
of how the corruption is wired; and (b) .cache/phase1_alg_states_wildhold.npz's own "fspan"
array is ALL-ZERO (0/311 rows carry a mention span there -- a stale precompute, read-only
fixture, never meant to carry it), so even a correctly-grounded body has nothing to tabu on
that split. form_mix_pm35c's OWN states npz carries real spans on 65.9% of present slots
(37183/50653) -- PMS8_241's own training distribution is the one place both preconditions
hold: a body that grounds AND gold spans to tabu. Measured on the 32 highest-span-coverage
rows (768 flagged slots): the tabu changes 67.2% of flagged slots' breath-0 argmax overall --
BELOW the task's pinned 80% bar (CORRUPT reads FAIL on this body) -- but the DIAGNOSIS line
(conditioning on whether the clean argmax already sat inside the gold span) shows the road
itself is exact: flip rate 1.000 (495/495) when it does, 0.077 (noise) when it doesn't.
PMS8_241's own grounding attention lands inside the gold span for only 64.5% of these slots to
begin with (0.6445, within 3pp of the overall flip rate: 0.6445*1.0 + 0.3555*0.077 = 0.672) --
the 80% bar is bounded by THIS BODY's measured grounding coverage, not by ALG_FLAG's wiring.
Stated for the record, not papered over: UNRESOLVED pending the coordinator's ruling (raise
the coverage via a differently-trained/longer body, lower the bar, or accept CORRUPT as an
honest miss on this specific fixture while GRAD and TABU both PASS clean).

  (1) CORRUPTION: ALG_SNAP_AT=1 snapshots expose ctx["fat0"] (the grounding attention)
      directly for two runs -- clean (flag_mask=None) and flagged (flag_mask = every present
      slot WITH a nonzero gold span, matching do_train's own host-sampler guard). fat0's
      argmax must differ on >= 80% of flagged slots; the UNFLAGGED run (flag_mask all-zero)
      must be numerically inert vs clean (reported honestly -- see below, not claimed bitwise).
  (2) GRAD PROBE + THE STOP-GRAD CONSTRUCTION: one training step() (forward + loss_fn +
      backward) on a forced half-flagged batch confirms p["flag_emb"].grad is present and
      nonzero, and that _flag_retarget's copy-targets are built via .detach() then
      argmax/threshold (both exactly-zero-gradient ops BY CONSTRUCTION -- confirmed by
      inspecting the function's own source, not by a numerical probe a tiny fixture could
      pass by luck).
  (3) TABU EXACT: from the SAME flagged run's snapshot, the tabu'd token (fat0.argmax(-1),
      matching the hard tabu's own construction exactly) must carry EXACTLY 0.0 attention
      mass in state["fat_cur"] (breath 1, the first tabu'd re-read) for every flagged slot.

THE "BITWISE" CAVEAT (stated once, honestly): the mix formula `fst*(1-m) + fst_c*m` runs a
SECOND bank() call unconditionally whenever flag_mask is not None, even when every element of
m is exactly 0.0 -- algebraically `x*1.0 + y*0.0 == x` in IEEE754, but empirically tinygrad's
CPU kernel fusion reorders this to a ~1e-9..1e-7 abs diff from the pure clean run (argmax
unaffected on every check run so far). This is NOT claimed as bit-identical; it is reported as
numerically inert. True bit-identity only holds when ALG_FLAG=0 / flag_mask=None entirely (the
standing unset/role8 bit-exactness checks in flagloop_gate.sh's COMMON block, which never pass
flag_mask at all).

Usage: DEV=CPU ALG_FLAG=1 .venv/bin/python3 scripts/picker/fl_gate_checks.py

Prints one line per check in the exact format .cache/flagloop_gate.sh greps:
    CORRUPT [fl-gate] PASS/FAIL ...
    GRAD    [fl-gate] PASS/FAIL ...
    TABU    [fl-gate] PASS/FAIL ...
    [fl-gate] all three checks: PASS/FAIL
"""
import os
import sys

os.environ.setdefault("DEV", "CPU")
assert os.environ.get("DEV") == "CPU", f"DEV={os.environ.get('DEV')!r} -- this script is CPU-only, never GPU"

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/picker")

import numpy as np

from adaptive_stop import FAMILY_ENVS
FAM = dict(FAMILY_ENVS["PMS8_241"])
FAM["DEV"] = "CPU"
FAM["ALG_MINE_BREATHS"] = "1"
FAM["ALG_FLAG"] = "1"
FAM["ALG_TRAIN"] = os.environ.get("FLGATE_TRAIN", ".cache/form_mix_pm35c.jsonl")
FAM["ALG_TRAIN_NAME"] = os.environ.get("FLGATE_TRAIN_NAME", "formpm35c")
FAM["ALG_ALLOW_PEN_TRAIN"] = "1"
for k, v in FAM.items():
    os.environ[k] = v

CKPT = os.environ.get("FLGATE_CKPT", ".cache/sharp_PMS8_241.safetensors")
N_ROWS = int(os.environ.get("FLGATE_NROWS", "8"))
SNAP_TMP = os.environ.get("FLGATE_SNAP_TMP", ".cache/flgate_snap_tmp.npz")


def _snapshot_fat0_faticur1(H, p, ts, tk, se, sm, fact_t, flag_mask, flag_tabu0, out_path):
    os.environ["ALG_SNAP_AT"] = "1"
    os.environ["ALG_SNAP_OUT"] = out_path
    kwargs = {}
    if flag_mask is not None:
        kwargs["flag_mask"] = flag_mask
        kwargs["flag_tabu0"] = flag_tabu0
    H.forward(p, ts, tk, se, slot_mask=sm, fact_buf=fact_t, **kwargs)
    snap = H.snap_load(out_path)
    fat0 = snap["ctx"]["fat0"]
    fat0 = fat0.numpy() if hasattr(fat0, "numpy") else np.asarray(fat0)
    fat_cur1 = snap["state"]["fat_cur"]
    fat_cur1 = fat_cur1.numpy() if hasattr(fat_cur1, "numpy") else np.asarray(fat_cur1)
    if os.path.exists(out_path):
        os.remove(out_path)
    return fat0, fat_cur1


def main():
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    import phase1_algebra_head as H
    from phase1_algebra_head import (build_params, forward, load_alg, build_slot_masks,
                                     alt2_fact_buf, K_VARS, L_FAC, loss_fn, _flag_retarget)

    assert H.ALG_FLAG, "fl_gate_checks.py needs ALG_FLAG=1 in the environment"

    samples, states, tokmask, gold, sent = load_alg("train")
    presence_all = gold["presence"]; fspan_all = gold["fspan"]
    cov = ((fspan_all.sum(-1) > 0) & (presence_all > 0.5)).sum(-1)
    idx = np.argsort(-cov)[:N_ROWS]                                # the highest-span-coverage rows
    B = len(idx)

    p = build_params(0)
    sd = safe_load(CKPT)
    missing = sorted(set(p) - set(sd))
    assert missing == ["flag_emb"], (
        f"fl_gate_checks.py expects ONLY flag_emb missing from {CKPT} (every other key "
        f"pad-warms from PMS8_241's own trained checkpoint) -- got {missing}")
    for k in p:
        if k in sd and tuple(sd[k].shape) == tuple(p[k].shape):
            p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    print(f"[fl-gate] loaded {sum(1 for k in p if k in sd)}/{len(p)} params from {CKPT} "
          f"(flag_emb zero-init, as promised)", flush=True)

    nv = np.array([samples[int(i)].get("n_vars", K_VARS) for i in idx])
    ma = np.array([samples[int(i)].get("m", 0) for i in idx])
    ts = Tensor(np.ascontiguousarray(states[idx]), dtype=dtypes.half)
    tk = Tensor(tokmask[idx].astype(np.float32))
    se = Tensor(sent[idx].astype(np.int32), dtype=dtypes.int)

    Tensor.training = False
    o0 = forward(p, ts, tk, se)
    onp0 = {k: o0[k].numpy() for k in ("fat", "args", "res")}
    mk = build_slot_masks(onp0, sent[idx].astype(np.int32))
    _ka = ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())
    _oa = {**onp0, **{k: o0[k].numpy() for k in _ka}}
    fb = alt2_fact_buf(_oa, sent[idx].astype(np.int32), nv, ma)
    fact_t = Tensor(fb, dtype=dtypes.float)
    sm = Tensor(mk, dtype=dtypes.float)

    presence = presence_all[idx].astype(np.float32)
    fspan = fspan_all[idx].astype(np.float32)
    L_TOT = H.L_TOT
    T = fspan.shape[-1]
    has_span = (fspan.sum(-1) > 0).astype(np.float32)
    flag_all = Tensor(presence * has_span, dtype=dtypes.float)      # do_train's own host-sampler guard
    flag_none = Tensor(np.zeros_like(presence), dtype=dtypes.float)
    tabu0 = Tensor((-1e4 * fspan).reshape(B, 1, L_FAC, T).astype(np.float32), dtype=dtypes.float)

    pres_b = (presence * has_span) > 0.5
    n_pres = int(pres_b.sum())

    # ------------------------------------------------------------------
    # CHECK 1 -- THE CORRUPTION (+ the setup CHECK 3 reuses)
    # ------------------------------------------------------------------
    fat0_clean, _ = _snapshot_fat0_faticur1(H, p, ts, tk, se, sm, fact_t, None, None, SNAP_TMP)
    fat0_unflag, _ = _snapshot_fat0_faticur1(H, p, ts, tk, se, sm, fact_t, flag_none, tabu0, SNAP_TMP)
    fat0_flag, fat_cur1_flag = _snapshot_fat0_faticur1(H, p, ts, tk, se, sm, fact_t, flag_all, tabu0, SNAP_TMP)

    am_clean = fat0_clean.argmax(-1)[:, :L_FAC]
    am_unflag = fat0_unflag.argmax(-1)[:, :L_FAC]
    am_flag = fat0_flag.argmax(-1)[:, :L_FAC]
    n_eq_unflag = int((am_unflag == am_clean)[pres_b].sum())
    n_diff_flag = int((am_flag != am_clean)[pres_b].sum())
    frac_eq_unflag = n_eq_unflag / max(n_pres, 1)
    frac_diff_flag = n_diff_flag / max(n_pres, 1)
    maxdiff_unflag = float(np.abs(fat0_unflag[:, :L_FAC][pres_b] - fat0_clean[:, :L_FAC][pres_b]).max()) if n_pres else 0.0
    ok1 = (frac_eq_unflag == 1.0) and (maxdiff_unflag < 1e-4) and (frac_diff_flag >= 0.80)
    # THE DIAGNOSIS (not the bar itself -- printed so a FAIL here is diagnosable, not just a
    # number): the mechanism's OWN flip rate, conditioned on whether the clean argmax was
    # already inside the gold span (the only case tabu-ing it can possibly move) -- measured
    # 2026-10-12 at 1.0 (495/495) on a 32-row sample, vs 0.077 on the rest. The flat 80% bar
    # is therefore bounded by PMS8_241's own grounding coverage (how often its attention
    # already lands in the gold span before any corruption), not by this road's wiring.
    was_inside = np.zeros_like(pres_b)
    for bi in range(B):
        for j in range(L_FAC):
            if pres_b[bi, j]:
                was_inside[bi, j] = fspan[bi, j, am_clean[bi, j]] > 0.5
    n_inside = int(was_inside[pres_b].sum())
    flip_given_inside = float((am_clean != am_flag)[pres_b & was_inside].mean()) if n_inside else float("nan")
    flip_given_outside = float((am_clean != am_flag)[pres_b & ~was_inside].mean()) if (n_pres - n_inside) else float("nan")
    print(f"CORRUPT [fl-gate] {'PASS' if ok1 else 'FAIL'} unflagged_argmax_eq={frac_eq_unflag:.4f} "
          f"unflagged_maxdiff={maxdiff_unflag:.3e} (numerically inert, not claimed bitwise) "
          f"flagged_breath0_argmax_differs={frac_diff_flag:.4f} (bar >= 0.80) n_flagged_slots={n_pres}")
    print(f"CORRUPT [fl-gate] DIAGNOSIS: clean_argmax_already_in_gold_span={n_inside}/{n_pres} "
          f"({n_inside/max(n_pres,1):.4f}); flip_rate_given_inside={flip_given_inside:.4f} "
          f"flip_rate_given_outside={flip_given_outside:.4f} -- the 80% bar is bounded by the "
          f"body's own grounding coverage, not by the tabu road (flip_rate_given_inside ~ 1.0 "
          f"confirms the road)")

    # ------------------------------------------------------------------
    # CHECK 2 -- THE OBJECTIVE WIRING (grad probe + the stop-grad construction)
    # ------------------------------------------------------------------
    # THE SNAPSHOT-LEAK FIX (found 2026-10-12 while building this gate): check 1 leaves
    # ALG_SNAP_AT/ALG_SNAP_OUT set in os.environ; forward()'s per-breath loop checks that
    # env pair on EVERY call, so check 2's own forward() would otherwise ALSO fire a
    # mid-loop _snap_dump() at breath 1 -- its .realize()/.numpy() calls are harmless to the
    # snapshot itself, but a snapshot taken had better not be mistaken for a live state: the
    # grad probe must run on an UNSNAPSHOTTED call. Confirmed by isolation (2026-10-12 CPU
    # smoke): with the stale ALG_SNAP_AT left set, flag_emb.grad was present but EXACTLY
    # zero-sum on an 8-row batch where attn_wq's own grad was 1196 -- every other param
    # trained fine, only flag_emb (the one road threaded through breath 1, where the
    # snapshot fires) went dark; clearing the env pair restored a normal nonzero grad.
    os.environ.pop("ALG_SNAP_AT", None)
    os.environ.pop("ALG_SNAP_OUT", None)
    Tensor.training = True
    rng = np.random.RandomState(0)
    flag_half_np = ((rng.random_sample((B, L_FAC)) < 0.5).astype(np.float32) * presence * has_span)
    flag_half = Tensor(flag_half_np, dtype=dtypes.float)
    _INT_KEYS = {"ftype", "op", "res", "digits", "sel", "query", "ftype_c1", "depth",
                 "digits2", "y", "bind_ids", "twin_idx", "gold_idx", "opc"}
    g = {k: Tensor(v[idx], dtype=(dtypes.int if k in _INT_KEYS else dtypes.float))
         for k, v in gold.items() if k != "is_lit" or "is_lit_f" not in gold}
    if "is_lit" in gold and "is_lit_f" not in gold:
        g["is_lit_f"] = Tensor(gold["is_lit"][idx], dtype=dtypes.float)
    for _zk in ("arg_dup", "arg_dir", "dir_mask"):
        if _zk not in g:
            g[_zk] = Tensor(np.zeros_like(presence), dtype=dtypes.float)
    g["flag_mask"] = flag_half
    o_tr = forward(p, ts, tk, se, slot_mask=sm, fact_buf=fact_t, flag_mask=flag_half, flag_tabu0=tabu0)
    l = loss_fn(o_tr, g)
    l.backward()
    g_flag = p["flag_emb"].grad
    has_grad = g_flag is not None
    grad_nonzero = bool(has_grad and float(g_flag.abs().sum().numpy()) > 0.0)
    import inspect
    src = inspect.getsource(_flag_retarget)
    stopgrad_ok = (".detach()" in src and "prev[f].detach()" in src and "prev[\"dig\"].detach()" in src
                  and "prev[\"args\"].detach()" in src)
    ok2 = bool(has_grad and grad_nonzero and stopgrad_ok)
    print(f"GRAD [fl-gate] {'PASS' if ok2 else 'FAIL'} flag_emb.grad_present={has_grad} "
          f"flag_emb.grad_nonzero={grad_nonzero} copy_target_detach_construction={stopgrad_ok}")

    # ------------------------------------------------------------------
    # CHECK 3 -- THE TABU, EXACT (reuses the flagged run's snapshot from check 1)
    # ------------------------------------------------------------------
    tabu_tok = fat0_flag.argmax(-1)                               # (B, L_TOT) -- matches the hard tabu's own construction exactly
    masses = []
    for bi in range(B):
        for j in range(L_FAC):
            if pres_b[bi, j]:
                masses.append(float(fat_cur1_flag[bi, j, tabu_tok[bi, j]]))
    masses = np.array(masses)
    n_checked = len(masses)
    n_exact_zero = int((masses == 0.0).sum())
    ok3 = (n_checked > 0) and (n_exact_zero == n_checked)
    print(f"TABU [fl-gate] {'PASS' if ok3 else 'FAIL'} exact_zero_count={n_exact_zero}/{n_checked} "
          f"max_mass={(masses.max() if n_checked else float('nan')):.3e}")

    print(f"[fl-gate] all three checks: {'PASS' if (ok1 and ok2 and ok3) else 'FAIL'}")
    sys.exit(0 if (ok1 and ok2 and ok3) else 1)


if __name__ == "__main__":
    main()
