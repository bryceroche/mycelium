"""snap_smoke_sort.py -- THE REPLAY GYM / gen-weights merge gate, item (2c) (2026-10-11, worktree
mycelium-wt8, branch replay): a snapshot smoke test under the CURRENT family env (SURF8 + the
sorting room, ALG_SORT=4 ALG_SORT_KEY=1 -- SRK_241's own recipe, .cache/rack_chain_SRK.sh) to
answer, empirically, the merge gate's own question: does ALG_SNAP_AT/ALG_SNAP_OUT's snapshot
carry the token side's organs (the re-sorted token tensor, the owner-key shuffle's per-token key)
and the notebook (ALG_NOTEBOOK/NB_PERSLOT), now that the replay harness (built 2026-10-07, before
the sorting room existed) has been merged forward onto a head that has them?

METHOD: load SRK_241's own checkpoint (.cache/sharp_SRK_241.safetensors), SRK_241's own env
(SURF8 + ALG_SORT=4 ALG_SORT_KEY=1, the rack_chain_SRK.sh recipe), N_ROWS=8 wild rows, dump a
snapshot at kb=2 (ALG_SNAP_AT=2), load it back (snap_load), and report:
  - the top-level manifest keys and each one's shape/dtype where applicable
  - whether `waist` (top-level) is the SORTED tensor (it must be -- forward() rebinds the local
    `waist` to _sort_room()/_sort_room_hg()'s output BEFORE the bank is built, and the snap hook
    reads whatever `waist` CURRENTLY IS at dump time, which is already the sorted one; there is
    no separate "pre-sort" waist anywhere in forward() to lose)
  - whether ctx["tok_key"] (the owner-key shuffle's raw per-token key, _sort_tok_key) is present
    and non-None (only when ALG_SORT_KEY=1 and "sort_key_w" in p)
  - whether state["nb"] / state["nb_st"] (the notebook) are present (breath_step always sets
    both keys on _bs_state, independent of ALG_SORT; carried through the generic serializer with
    everything else in `state`)
  - the on-disk npz size, total and divided by N_ROWS, vs the harness's original ~3-4 MB/row
    estimate (2026-10-07, pre-sorting-room)

DEV=CPU asserted; .cache/gpu.lock never touched (no flock in this file); this is a READ (no
training), so it never needs the GPU regardless.
"""
import os
import sys
import time

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("DEV", "CPU")
assert os.environ.get("DEV") == "CPU", f"DEV={os.environ.get('DEV')!r}: this script is CPU-only"

import numpy as np

N_ROWS = 8
SNAP_KB = 2
CKPT = ".cache/sharp_SRK_241.safetensors"
SNAP_OUT = ".cache/replay_gate_snap_smoke_SRK.npz"

# SRK_241's own recipe (.cache/rack_chain_SRK.sh): FAM + SURF8 + ALG_SORT=4 ALG_SORT_KEY=1,
# evaluated against wild_admitted_holdout (the gym's own standing fixture).
_FAM = {
    "DEV": "CPU", "ALG2": "1", "ALG_FTYPES": "9", "ALG_DUP": "1",
    "ALG_HW": "512", "ALG_WIDE": "1", "ALG_BREATH": "7",
    "ALG_NOTEBOOK": "1", "ALG_SIXWAVE": "1", "NB_PERSLOT": "1",
    "ALG_BINDBUS": "7", "ALG_BIND_D": "512", "BIND_CODES": ".cache/bindbus_codes512r.npz",
    "ALG_BUSGARAGE": "2", "ALG_SHELF_CIRCLE": "2", "ALG_ALTMASK": "1",
    "ALG_ALT21": "1", "ALG_ALT2": "1", "ALG_MASKHEAD": "1",
    "ALG_FED": "1", "ALG_POLAR": "1", "ALG_POLAR_D": "128",
    "ALG_POLAR_EM": "0.1", "ALG_POLAR_D_INIT": ".cache/polar_waist_init_d128u.npz",
    "ALG_PRUNE": "pforms,s4,fednl0,lane2", "ALG_SLOT_ALL": "1",
    "ALG_STELLAR": "2", "ALG_CLOCK_CANON": "1", "SC_EVAL": "0",
    "ALG_TEST": ".cache/wild_admitted_holdout.jsonl", "ALG_TEST_NAME": "wildhold",
    "ALG_ROUTER": "2", "R_GAIN_INIT": "1.0", "ALG_FREEZE": "r_gain", "ALG_ROUTER_PTR": "0.0",
    "ALG_SPAN_ALL": "1", "ALG_SPAN_ARGS": "1", "ALG_SPAN_OP": "1", "ALG_SPAN_RCUE": "1",
    "ALG_SPAN_ARCUE": "1", "ALG_PTR_SURF": "role:add:2.0",
    "ALG_SORT": "4", "ALG_SORT_KEY": "1",
}
for k, v in _FAM.items():
    os.environ[k] = v


def _sizeof(x):
    if isinstance(x, np.ndarray):
        return x.nbytes
    if isinstance(x, dict):
        return sum(_sizeof(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return sum(_sizeof(v) for v in x)
    return 0


def _describe(x, depth=0, max_depth=2):
    if isinstance(x, np.ndarray):
        return f"ndarray{x.shape} {x.dtype}"
    if isinstance(x, dict):
        if depth >= max_depth:
            return f"dict({len(x)} keys)"
        return "{" + ", ".join(f"{k}: {_describe(v, depth + 1, max_depth)}" for k, v in x.items()) + "}"
    if isinstance(x, (list, tuple)):
        if not x:
            return "[]"
        return f"[{len(x)}] of " + _describe(x[0], depth + 1, max_depth)
    return repr(x)[:60]


def main():
    t0 = time.time()
    import phase1_algebra_head as H
    from tinygrad.nn.state import safe_load
    from tinygrad import Tensor, dtypes

    assert os.path.exists(CKPT), f"missing checkpoint {CKPT} -- SRK_241's own, from rack_chain_SRK.sh"
    p = H.build_params(0)
    sd = safe_load(CKPT)
    assert set(sd.keys()) == set(p.keys()), (sorted(set(sd) - set(p))[:4], sorted(set(p) - set(sd))[:4])
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    print(f"[snap-smoke] loaded {CKPT} ({time.time()-t0:.0f}s so far); "
          f"ALG_SORT={H.ALG_SORT} ALG_SORT_KEY={H.ALG_SORT_KEY} ALG_SORT_HG={H.ALG_SORT_HG}", flush=True)
    assert H.ALG_SORT == 4 and H.ALG_SORT_KEY == 1, "env did not take -- check module-level ALG_SORT/ALG_SORT_KEY"
    assert "sort_key_w" in p, "sort_key_w not allocated -- ALG_SORT_KEY=1 did not reach build_params"

    vs, vst, vtk, vg, vse = H.load_alg("test")
    n = min(N_ROWS, len(vs))
    sl = np.arange(n)
    ts = Tensor(np.ascontiguousarray(vst[sl]), dtype=dtypes.half)
    tk = Tensor(vtk[sl].astype(np.float32), dtype=dtypes.float)
    se = Tensor(vse[sl].astype(np.int32), dtype=dtypes.int)

    # THE IDENTITY INSTRUMENT'S BLIND SPOT (ledger 2026-10-11 11:16): forward() WITHOUT a
    # slot_mask only runs breath 0's grounding read -- the per-breath loop (line ~9220's
    # `if K_B > 1 and slot_mask is not None and "W_bo" in p:` gate), where the ALG_SNAP_AT
    # hook lives, never executes. Build a real slot_mask first (replay_gate_check.py's own
    # _final_pass_kwargs pattern, no-consult form), exactly like every other reader does.
    o0 = H.forward(p, ts, tk, se)
    onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
    mk = H.build_slot_masks(onp0, se.numpy().astype(np.int32))
    slot_mask = Tensor(mk, dtype=dtypes.float)

    if os.path.exists(SNAP_OUT):
        os.remove(SNAP_OUT)
    os.environ["ALG_SNAP_AT"] = str(SNAP_KB)
    os.environ["ALG_SNAP_OUT"] = SNAP_OUT
    out_dump = H.forward(p, ts, tk, se, slot_mask=slot_mask)
    os.environ.pop("ALG_SNAP_AT", None); os.environ.pop("ALG_SNAP_OUT", None)
    assert os.path.exists(SNAP_OUT), "forward() with ALG_SNAP_AT set did not write a snapshot"
    sz = os.path.getsize(SNAP_OUT)
    print(f"[snap-smoke] snapshot written: {SNAP_OUT} ({sz} bytes = {sz/1e6:.2f} MB total, "
          f"{sz/n/1e6:.3f} MB/row for n={n})", flush=True)

    snap = H.snap_load(SNAP_OUT)
    print(f"[snap-smoke] top-level manifest keys: {sorted(snap.keys())}", flush=True)
    ctx = snap["ctx"]; state = snap["state"]
    print(f"[snap-smoke] ctx keys: {sorted(ctx.keys())}", flush=True)
    print(f"[snap-smoke] state keys: {sorted(state.keys())}", flush=True)

    waist = snap["waist"]
    print(f"[snap-smoke] top-level waist: {_describe(waist)}", flush=True)
    tok_key = ctx.get("tok_key")
    print(f"[snap-smoke] ctx['tok_key'] (owner-key shuffle): "
          f"{'ABSENT' if tok_key is None else _describe(tok_key)}", flush=True)
    nb = state.get("nb"); nb_st = state.get("nb_st")
    print(f"[snap-smoke] state['nb'] (notebook): {'ABSENT' if nb is None else _describe(nb)}", flush=True)
    print(f"[snap-smoke] state['nb_st'] (notebook state): "
          f"{'ABSENT' if nb_st is None else _describe(nb_st)}", flush=True)

    # Cross-check: is the dumped `waist` actually the SORTED tensor, not the raw waist_w
    # projection? Recompute the pre-sort projection by hand and diff -- if ALG_SORT fired,
    # the two must differ (a sort room with "sort0_wq" in p is never a no-op at these weights).
    sent = Tensor(vse[sl].astype(np.int32), dtype=dtypes.int)
    waist_pre = (ts.cast(dtypes.float) @ p["waist_w"] + p["waist_b"]).gelu() + p["sent_emb"][sent]
    wpre_np = waist_pre.realize().numpy()
    wsnap_np = waist.realize().numpy() if hasattr(waist, "realize") else np.asarray(waist)
    maxdiff = float(np.max(np.abs(wpre_np.astype(np.float64) - wsnap_np.astype(np.float64))))
    print(f"[snap-smoke] max|dumped waist - pre-sort waist_w projection| = {maxdiff:.4g} "
          f"({'SORTED (differs from raw projection, as expected)' if maxdiff > 1e-6 else 'SUSPICIOUS: identical to the pre-sort projection'})",
          flush=True)

    # replay from this snapshot and confirm it still runs end to end on this body (decode-level
    # smoke, not the full bit-identical gate -- replay_gate_check.py owns that on PMS8_241/RK_241).
    out_replay = H.replay(p, snap, SNAP_KB)
    need = ["pres", "res", "ftype", "op", "dig", "args", "query"]   # decode() always reads "query"
    fnp = {k: out_dump[k].realize().numpy() for k in need if k in out_dump}
    rnp = {k: out_replay[k].realize().numpy() for k in need if k in out_replay}
    n_diff = 0
    for i in range(n):
        row_f = {k: fnp[k][i] for k in fnp}
        row_r = {k: rnp[k][i] for k in rnp}
        if H.decode(row_f) != H.decode(row_r):
            n_diff += 1
    print(f"[snap-smoke] replay from kb={SNAP_KB} on SRK_241: decode differs on {n_diff}/{n} rows "
          f"({'OK' if n_diff == 0 else 'MISMATCH'})", flush=True)
    print(f"[snap-smoke] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
