"""chalk_dryrun.py -- THE FIRST USE: THE CHALKBOARD DRY-RUN (2026-10-07, worktree mycelium-wt8,
branch replay; THE BUILD item (4), docs/phase1_skeleton_spec.md 2026-10-07 17:27).

A MECHANISM READ, NOT A CLAIM (stated per the brief): with RK_241's snapshot at kb=2 (after
consult 1), on 64 wild rows, build the chalk block as rack3's chalk_pack would (the solver's
DERIVED facts from consult 1 -- chalk_pack ported verbatim from branch rack3 commit 5f5ecb85,
scripts/phase1_algebra_head.py:925, into THIS branch's phase1_algebra_head.py, cited there; gen-
weights has no ALG_CHALK organ of its own, so this is a re-implementation, not an import -- rack3
is a sibling lineage of gen-weights, not an ancestor, and the task's own note says the import is
impractical for exactly that reason). Inject it via overrides into the bank's last ALG_CHALK_N=8
pad positions two ways:
  (A) NULL embedding: chalk_dig/chalk_tag zeroed (the checkpoint never trained them -- RK_241 has
      no such params at all; this run allocates them fresh under ALG_CHALK=1 then zeros them) --
      tests whether the READ attends to chalk positions AT ALL under a content-free signal.
  (B) TEXT-COPY embedding: for chalk slots whose numeral also appears as a digit-run in the row's
      own text, the copied embedding is that matching token's OWN waist row (bypassing chalk_dig/
      chalk_tag entirely -- spliced directly into `waist`'s last ALG_CHALK_N columns via replay()'s
      `overrides["waist"]`/["tokmask"]`, not through _make_bank's trained-table path at all, since
      a per-row arbitrary vector cannot be expressed through a shared digit-id -> table lookup).
Then replay kb 3..6 under each variant and report: rows with >=1 chalk token; the attention mass
on chalk positions per slot per breath (3..6); any change in the decoded graph vs the UN-INJECTED
replay (same kb=2 snapshot, no chalk) -- which rows changed, whether the changed slots now point
at chalk, and the custody-key correctness (ps_legal-style: does the row's args/res/dig graph still
match the gold key) before/after on those rows.

DEV=CPU asserted; .cache/gpu.lock never touched (no flock in this file).
"""
import os
import sys
import time

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("DEV", "CPU")
assert os.environ.get("DEV") == "CPU"

import numpy as np

N_ROWS = 64
SNAP_KB = 2

_FAM = {
    "DEV": "CPU", "ALG2": "1", "ALG_FTYPES": "9", "ALG_DUP": "1",
    "ALG_HW": "512", "ALG_WIDE": "1", "ALG_BREATH": "7",
    "ALG_NOTEBOOK": "1", "ALG_SIXWAVE": "1", "NB_PERSLOT": "1",
    "ALG_BINDBUS": "7", "ALG_BIND_D": "512",
    "ALG_BUSGARAGE": "2", "ALG_SHELF_CIRCLE": "2", "ALG_ALTMASK": "1",
    "ALG_ALT21": "1", "ALG_ALT2": "1", "ALG_MASKHEAD": "1",
    "ALG_FED": "1", "ALG_POLAR": "1", "ALG_POLAR_D": "128",
    "ALG_POLAR_EM": "0.1",
    "ALG_POLAR_D_INIT": ".cache/polar_waist_init_d128u.npz",
    "ALG_PRUNE": "pforms,s4,fednl0,lane2", "ALG_SLOT_ALL": "1",
    "ALG_STELLAR": "2", "ALG_CLOCK_CANON": "1", "SC_EVAL": "0",
    "ALG_TEST": ".cache/wild_admitted_holdout.jsonl", "ALG_TEST_NAME": "wildhold",
    "ALG_ROUTER": "2", "R_GAIN_INIT": "1.0", "ALG_FREEZE": "r_gain", "ALG_ROUTER_PTR": "0.0",
    "ALG_SPAN_ALL": "1", "ALG_SPAN_ARGS": "1", "ALG_SPAN_OP": "1", "ALG_SPAN_RCUE": "1",
    "ALG_SPAN_ARCUE": "1", "ALG_PTR_SURF": "role:add:2.0", "BIND_CODES": ".cache/bindbus_codes512r.npz",
    "ALG_HIER_READ": "1", "ALG_HIER_WAIST": "1", "ALG_HIER_DAMP": "2,4,0", "ALG_HIER_TAU": "0",
    "ALG_ALT3": "1", "ALG_CERT": "2.0", "ALG_RACK": "1",
    "ALG_RACK_TESTS": "given_unique", "ALG_RACK_FREEZE": "leaf",
    "ALG_CHALK": "1", "ALG_CHALK_N": "8",   # THE CHALKBOARD: allocates chalk_dig/chalk_tag in build_params
}
CKPT = ".cache/sharp_RK_241.safetensors"


def _set_env():
    for k, v in _FAM.items():
        os.environ[k] = v


def _load():
    _set_env()
    import importlib
    import phase1_algebra_head as H
    importlib.reload(H)
    from tinygrad.nn.state import safe_load
    p = H.build_params(0)
    sd = safe_load(CKPT)
    missing = set(p.keys()) - set(sd.keys())
    extra = set(sd.keys()) - set(p.keys())
    assert missing == {"chalk_dig", "chalk_tag"}, (
        f"unexpected param mismatch beyond THE CHALKBOARD's own new params: "
        f"missing={sorted(missing)[:6]} extra={sorted(extra)[:6]}")
    assert not extra, sorted(extra)[:6]
    for k in sd:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    print(f"[chalk-dryrun] loaded {CKPT}; chalk_dig/chalk_tag left at build_params' random "
          f"init (RK_241 was never trained with ALG_CHALK -- stated, not hidden)", flush=True)
    return H, p


def main():
    t0 = time.time()
    H, p = _load()
    from tinygrad import Tensor, dtypes

    vs, vst_, vtk, vg, vse = H.load_alg("test")
    n = min(N_ROWS, len(vs))
    sl = np.arange(n)
    ts = Tensor(np.ascontiguousarray(vst_[sl]), dtype=dtypes.half)
    tk = Tensor(vtk[sl].astype(np.float32), dtype=dtypes.float)
    se = Tensor(vse[sl].astype(np.int32), dtype=dtypes.int)
    texts = [vs[int(i)]["text"] for i in sl]
    _nv = np.array([vs[int(i)].get("n_vars", H.K_VARS) for i in sl])
    _ma = np.array([vs[int(i)].get("m", 0) for i in sl])

    # pass-1 slot mask (membrane_rack.py's own convention)
    o0 = H.forward(p, ts, tk, se)
    onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
    mk = H.build_slot_masks(onp0, se.numpy().astype(np.int32))
    slot_mask = Tensor(mk, dtype=dtypes.float)

    # consult 1 (kb=2) -- the SAME cycle membrane_rack.py's collect() runs, and chalk_dryrun's
    # own copy of _final_pass_kwargs (replay_gate_check.py), inlined here so this script stands
    # alone
    ck3 = ("pres", "ftype", "op", "dig", "args", "res") + (("dup",) if "dup" in o0 else ())
    oa3 = H.forward(p, ts, tk, se, slot_mask=slot_mask, stop_after=2)
    onp3 = {k: oa3[k].realize().numpy() for k in ck3}
    fb3 = H.alt2_fact_buf(onp3, se.numpy().astype(np.int32), _nv, _ma)
    f3_t = Tensor(fb3, dtype=dtypes.float)
    rows3 = [{k: onp3[k][bi] for k in onp3} for bi in range(n)]
    c3_t = Tensor(H.certifier_bias(rows3, fb3, texts, H.T_ALG, H.ALG_CERT, H.ALG_CERT_IMPLIED), dtype=dtypes.float)
    r3_np = H.rack_pack(rows3, fb3, texts, H.T_ALG, H.ALG_RACK_TESTS)
    r3_t = Tensor(r3_np, dtype=dtypes.float)

    # consult 2 (kb=4), needed so replay from kb=2 can inject facts5/cert5/rack5 at kb==4
    # exactly as the full pipeline does (threaded via the snapshot; replay() does not re-
    # derive it live -- stated in replay()'s own docstring)
    ob3 = H.forward(p, ts, tk, se, slot_mask=slot_mask, stop_after=4, facts3=f3_t, cert3=c3_t, rack3=r3_t)
    onp5 = {k: ob3[k].realize().numpy() for k in ck3}
    fb5 = H.alt2_fact_buf(onp5, se.numpy().astype(np.int32), _nv, _ma)
    f5_t = Tensor(fb5, dtype=dtypes.float)
    rows5 = [{k: onp5[k][bi] for k in onp5} for bi in range(n)]
    c5_t = Tensor(H.certifier_bias(rows5, fb5, texts, H.T_ALG, H.ALG_CERT, H.ALG_CERT_IMPLIED), dtype=dtypes.float)
    r5_np = H.rack_pack(rows5, fb5, texts, H.T_ALG, H.ALG_RACK_TESTS, prev=r3_np)
    r5_t = Tensor(r5_np, dtype=dtypes.float)

    # THE SNAPSHOT at kb=2, from the FULL final-pass call (facts3/facts5/cert3/cert5/rack3/
    # rack5 all threaded -- the real pipeline, exactly as the gate's RK_241 arm)
    snap_path = ".cache/chalk_dryrun_snap_kb2.npz"
    os.environ["ALG_SNAP_AT"] = str(SNAP_KB); os.environ["ALG_SNAP_OUT"] = snap_path
    _ = H.forward(p, ts, tk, se, slot_mask=slot_mask, facts3=f3_t, facts5=f5_t,
                  cert3=c3_t, cert5=c5_t, rack3=r3_t, rack5=r5_t)
    os.environ.pop("ALG_SNAP_AT"); os.environ.pop("ALG_SNAP_OUT")
    snap = H.snap_load(snap_path)
    print(f"[chalk-dryrun] snapshot at kb={SNAP_KB} written ({time.time() - t0:.0f}s so far)", flush=True)

    # THE CHALK BLOCK from consult 1's own derived facts (chalk_pack ported from rack3:925,
    # cited above _make_bank in phase1_algebra_head.py)
    chalk3_np, n_dropped = H.chalk_pack(rows3, fb3, vtk[sl].astype(np.float32), H.T_ALG, H.ALG_CHALK_N)
    # chalk5 from consult 2's own derived facts -- superseding, as the trained form would read it
    chalk5_np, n_dropped5 = H.chalk_pack(rows5, fb5, vtk[sl].astype(np.float32), H.T_ALG, H.ALG_CHALK_N)
    n_chalk_rows = int((chalk3_np[:, :, 0].sum(1) > 0).sum())
    print(f"[chalk-dryrun] chalk3: {n_chalk_rows}/{n} rows carry >=1 chalk token "
          f"({n_dropped} dropped for no room); chalk5 rows: "
          f"{int((chalk5_np[:, :, 0].sum(1) > 0).sum())}/{n} ({n_dropped5} dropped)", flush=True)

    # ---- baseline: the UN-INJECTED replay from the SAME kb=2 snapshot (no chalk at all) ----
    out_base = H.replay(p, snap, SNAP_KB)

    # ---- variant A: NULL embedding (chalk_dig/chalk_tag zeroed) ----
    p["chalk_dig"].assign(np.zeros((3, 10, H.H_W), np.float32)).realize()
    p["chalk_tag"].assign(np.zeros((H.H_W,), np.float32)).realize()
    chalk3_t = Tensor(chalk3_np, dtype=dtypes.float)
    chalk5_t = Tensor(chalk5_np, dtype=dtypes.float)
    out_null = H.replay(p, snap, SNAP_KB, overrides={"chalk3": chalk3_t, "chalk5": chalk5_t})

    # ---- variant B: TEXT-COPY embedding (bypasses chalk_dig/tag; splices a matching text
    # token's own waist row directly into the last ALG_CHALK_N positions) ----
    waist_np = snap["waist"].realize().numpy()            # (B, T, H_W) float32, from the snapshot
    tokmask_np = snap["tokmask"].realize().numpy()         # (B, T)
    T_ALG = H.T_ALG; N_CHALK = H.ALG_CHALK_N
    waist_b = waist_np.copy()
    tokmask_b = tokmask_np.copy()
    n_copy_found = 0
    for bi in range(n):
        ids_row = None
        for ci in range(N_CHALK):
            if chalk3_np[bi, ci, 0] <= 0:
                continue
            d0, d1, d2 = (int(chalk3_np[bi, ci, 1]), int(chalk3_np[bi, ci, 2]), int(chalk3_np[bi, ci, 3]))
            val = d0 * 100 + d1 * 10 + d2
            if ids_row is None:
                from tokenizers import Tokenizer
                tok = Tokenizer.from_file(H.TOKENIZER_JSON)
                enc = tok.encode(texts[bi])
                ids_row = enc.ids[:T_ALG]
            # a same-valued digit RUN in this row's own text (membrane_rack.py's own
            # _digit_runs convention, reused via membrane_scale's helper by import)
            import membrane_scale as MS
            runs = MS._digit_runs(tok, np.array(ids_row + [0] * (T_ALG - len(ids_row))), T_ALG)
            matches = [(a, b, v) for a, b, v in runs if v == val]
            pos = T_ALG - N_CHALK + ci
            if matches:
                a, b, v = matches[0]
                waist_b[bi, pos, :] = waist_np[bi, a, :]   # the matching token's OWN embedding
                tokmask_b[bi, pos] = 1.0
                n_copy_found += 1
            else:
                waist_b[bi, pos, :] = 0.0
                tokmask_b[bi, pos] = 0.0
    print(f"[chalk-dryrun] text-copy: {n_copy_found} chalk slots (of "
          f"{int(chalk3_np[:, :, 0].sum())} candidates) found a same-valued text token to copy",
          flush=True)
    out_text = H.replay(p, snap, SNAP_KB,
                        overrides={"waist": Tensor(waist_b, dtype=dtypes.float),
                                   "tokmask": Tensor(tokmask_b, dtype=dtypes.float)})

    # ---- report: decode agreement + custody-key correctness, baseline vs each variant ----
    def _row_dicts(out, keys):
        np_ = {k: out[k].realize().numpy() for k in keys if k in out}
        return [{k: np_[k][i] for k in np_} for i in range(n)]

    need = ["pres", "res", "ftype", "op", "dig", "args", "query"] + \
           [k for k in ("sgn", "dup", "dargs") if k in out_base]
    rows_base = _row_dicts(out_base, need)
    rows_null = _row_dicts(out_null, need)
    rows_text = _row_dicts(out_text, need)

    gold = {"presence": vg["presence"][sl], "ftype": vg["ftype"][sl], "digits": vg["digits"][sl]}

    def _given_key_ok(dec, row_i):
        """Custody-key correctness: for each decoded GIVEN fact, does its (var, value) match
        the gold given at that var (presence+ftype+digits)? A coarse, zero-GPU proxy for
        ps_legal's own join (full re-derivation of ps_legal_wild needs the mixed-register
        pipeline this script does not run) -- stated as a proxy, not the registered metric."""
        facs_, _qv = dec   # decode() returns (facs, query_argmax)
        ok_all = True
        for f in facs_:
            if f["ftype"] != "given":
                continue
            v = int(f["var"])
            if v >= gold["presence"].shape[1] or gold["presence"][row_i, v] <= 0 or int(gold["ftype"][row_i, v]) != 1:
                ok_all = False
                continue
            gval = int("".join(str(int(x)) for x in gold["digits"][row_i, v]))
            if f["value"] != gval:
                ok_all = False
        return ok_all

    for tag, rows_v, chalk_np in (("NULL", rows_null, chalk3_np), ("TEXT-COPY", rows_text, chalk3_np)):
        n_changed = 0
        changed_rows = []
        for i in range(n):
            dec_base = H.decode(rows_base[i])
            dec_v = H.decode(rows_v[i])
            if dec_base != dec_v:
                n_changed += 1
                changed_rows.append(i)
        key_before = sum(_given_key_ok(H.decode(rows_base[i]), i) for i in changed_rows)
        key_after = sum(_given_key_ok(H.decode(rows_v[i]), i) for i in changed_rows)
        print(f"[chalk-dryrun] variant={tag}: decode changed on {n_changed}/{n} rows "
              f"(rows: {changed_rows[:12]}{'...' if len(changed_rows) > 12 else ''}); "
              f"custody-key-ok before={key_before}/{len(changed_rows)} "
              f"after={key_after}/{len(changed_rows)}", flush=True)

    # ---- attention mass on chalk positions, per slot per breath 3..6 (NULL variant; the
    # cleanest read since text-copy's splice is itself a real-token copy, not a chalk-
    # specific signal to attribute) ----
    # Re-run breath-by-breath so each breath's bank attention (state["fat_cur"]) is captured;
    # this duplicates replay()'s loop body on purpose (a bespoke instrumented probe, not a
    # second generic entry point) -- see the module docstring.
    ctx = dict(snap["ctx"]); state = dict(snap["state"])
    sent = ctx["mc_sent"]
    waist0 = snap["waist"]; tokmask0 = snap["tokmask"]   # already Tensors (snap_load reconstructs them)
    chalk3_t2 = Tensor(chalk3_np, dtype=dtypes.float); chalk5_t2 = Tensor(chalk5_np, dtype=dtypes.float)
    bank = H._make_bank(p, waist0, tokmask0, n, sent=sent, tree=snap["tree"],
                        chalk3=chalk3_t2, chalk5=chalk5_t2)
    ctx["bank"] = bank; ctx["rot2"] = H.rot2_interleaved
    ctx["waist"] = waist0; ctx["tokmask"] = tokmask0
    vst = ctx["vst"]
    facts3_t = snap["facts3"]; facts5_t = snap["facts5"]   # already Tensors
    mass_by_breath = {}
    T_ALG = H.T_ALG; N_CHALK = H.ALG_CHALK_N
    for kb in range(SNAP_KB + 1, int(snap["K_B"])):
        H.breath_step(p, state, kb, ctx)
        if H.conductor(kb).facts3_inject and facts3_t is not None:
            vst = H._fact_inject(p, vst, facts3_t); ctx["vst"] = vst
        if H.conductor(kb).facts5_inject and facts5_t is not None:
            vst = H._fact_inject(p, vst, facts5_t); ctx["vst"] = vst
        fat_cur = state.get("fat_cur")
        if fat_cur is None:
            continue
        fc = fat_cur.realize().numpy()   # (B, L_FAC, T) or (B, L_TOT, T) -- per-slot token mass
        chalk_mass = fc[:, :, T_ALG - N_CHALK:].sum(-1)   # (B, L_FAC_or_L_TOT): mass on the 8 chalk cols
        rows_with_chalk = np.where(chalk3_np[:, :, 0].sum(1) > 0)[0]
        if len(rows_with_chalk):
            per_row_mean = chalk_mass[rows_with_chalk].mean()
            per_row_max = chalk_mass[rows_with_chalk].max()
        else:
            per_row_mean = per_row_max = float("nan")
        mass_by_breath[kb] = (float(per_row_mean), float(per_row_max))
        print(f"[chalk-dryrun] breath {kb}: mean slot->chalk mass (rows with chalk) = "
              f"{per_row_mean:.5f}, max = {per_row_max:.5f}", flush=True)

    print(f"[chalk-dryrun] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
