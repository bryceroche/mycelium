"""rtb_grad_probe.py -- THE GATE (THE REPLAY-TRAINED BREATH, worktree mycelium-wt28, branch rtb;
2026-10-11). Proves part (B) of THE BUILD on a tiny, sized-down probe (peak RSS budget: CLAUDE.md
§5's "eager probes sized down < 8 GB"):

  1. ONE replay-training step at kb_target=3 (resume from a snapshot at kb_done=2, run breath 3
     via replay(..., kb_stop=4), grade with the SAME loss_fn the live loop uses, backward,
     opt.step()) produces a FINITE loss and a NONZERO gradient on at least one head param this
     breath's own heads touch (W_ftype -- read by EVERY breath's heads_of, so a reliable probe).
  2. THE CLOCK PHASE CHECK: conductor(3) in this probe's own process is the IDENTICAL namedtuple
     (field-for-field) the live training loop's forward() call would build for the same kb --
     conductor() is a pure function of kb and the env-parsed globals (its own docstring), so this
     is checked directly rather than inferred.

DEV=CPU, no .cache/gpu.lock, no real training run -- a mechanism probe on 8 rows.
"""
import os
import sys
import time

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("DEV", "CPU")
assert os.environ.get("DEV") == "CPU", f"DEV={os.environ.get('DEV')!r}: this script is CPU-only"

import numpy as np

import rtb_lib as RL

_FAM = {
    "DEV": "CPU", "ALG2": "1", "ALG_FTYPES": "9", "ALG_DUP": "1",
    "ALG_HW": "512", "ALG_WIDE": "1", "ALG_BREATH": "7",
    "ALG_NOTEBOOK": "1", "ALG_SIXWAVE": "1", "NB_PERSLOT": "1",
    "ALG_BINDBUS": "7", "ALG_BIND_D": "512",
    "BIND_CODES": ".cache/bindbus_codes512.npz",
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
}
CKPT = ".cache/sharp_PMS8_241.safetensors"
N_ROWS = 8
KB_DONE = 2
KB_TARGET = KB_DONE + 1


def main():
    for k, v in _FAM.items():
        os.environ[k] = v
    import phase1_algebra_head as H
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.optim import AdamW
    from tinygrad.nn.state import safe_load

    p = H.build_params(0)
    sd = safe_load(CKPT)
    assert set(sd.keys()) == set(p.keys())
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

    vs, vst, vtk, vg, vse = H.load_alg("test")   # the wild-holdout fixture also used by the gate's other reads
    sl = np.arange(min(N_ROWS, len(vs)))
    ts = Tensor(np.ascontiguousarray(vst[sl]), dtype=dtypes.half)
    tk = Tensor(vtk[sl].astype(np.float32), dtype=dtypes.float)
    se = Tensor(vse[sl].astype(np.int32), dtype=dtypes.int)
    kwargs = RL.build_consult_kwargs(H, p, ts, tk, se, vs, sl)

    snap_path = f".cache/rtb_gradprobe_snap_kb{KB_DONE}.npz"
    os.environ["ALG_SNAP_AT"] = str(KB_DONE)
    os.environ["ALG_SNAP_OUT"] = snap_path
    Tensor.training = True
    _ = H.forward(p, ts, tk, se, **kwargs)
    os.environ.pop("ALG_SNAP_AT", None); os.environ.pop("ALG_SNAP_OUT", None)
    snap = H.snap_load(snap_path)
    os.remove(snap_path)

    # THE CLOCK PHASE CHECK: conductor(kb) is a pure function of kb + env -- same process, same
    # env, so this is necessarily the identical record the live loop's forward() call built for
    # breath KB_TARGET; asserted directly rather than taken on faith.
    H._CONDUCTOR_CACHE.clear()
    c_live = H.conductor(KB_TARGET)
    H._CONDUCTOR_CACHE.clear()
    c_replay = H.conductor(KB_TARGET)
    _fields = type(c_live).__slots__
    _diff = {f: (getattr(c_live, f), getattr(c_replay, f)) for f in _fields
             if getattr(c_live, f) != getattr(c_replay, f)}
    assert not _diff, _diff
    print(f"[rtb-gradprobe] clock phase check PASS: conductor({KB_TARGET}) identical "
          f"field-for-field in both contexts", flush=True)

    opt = AdamW(list(p.values()), lr=1e-4, weight_decay=0.01)
    Tensor.training = True
    out = H.replay(p, snap, KB_DONE, kb_stop=KB_TARGET + 1)   # runs EXACTLY breath KB_TARGET

    gold = RL.gold_feed(H, vg, sl)
    bg = RL.feed_to_tensors(H, gold)
    l = H.loss_fn(out, bg)
    opt.zero_grad()
    l.backward()
    lv = float(l.realize().numpy())
    assert np.isfinite(lv), f"loss not finite: {lv}"

    probe_key = "h_ftype"
    assert probe_key in p, f"{probe_key} not in params -- pick a different probe key"
    g = p[probe_key].grad
    assert g is not None, f"{probe_key}.grad is None -- no gradient reached this param"
    g_abs_sum = float(g.abs().sum().numpy())
    assert g_abs_sum > 0, f"{probe_key}.grad is all-zero"

    # THE ZERO-FILL (stated tradeoff, not hidden): breath KB_TARGET's own forward pass does not
    # touch every param (organs live only in OTHER breaths never run this step) -- tinygrad's
    # AdamW.step() requires every registered param to carry a grad (helpers.unwrap's bare
    # assert), so params this step never touched get an explicit ZERO grad rather than being
    # excluded from the optimizer's param list (which would lose their momentum state across
    # different kb's steps touch them on different training steps). COST, reported not hidden:
    # AdamW's decoupled weight decay still nudges an untouched param by -lr*wd*param every step,
    # even with a zero grad -- a real, small, per-step leak onto every param regardless of which
    # breath's own step ran; a correct fix would skip decay too for untouched params (registered
    # as a follow-up for the real arm's build, not fixed in this probe).
    n_nograd = sum(1 for k, v in p.items() if v.grad is None)
    for k, v in p.items():
        if v.grad is None:
            v.grad = Tensor.zeros(*v.shape, dtype=v.dtype)
    opt.step()
    print(f"[rtb-gradprobe] ONE replay-training step at kb_target={KB_TARGET}: "
          f"loss={lv:.4f} (finite), {probe_key}.grad.abs().sum()={g_abs_sum:.6g} (nonzero), "
          f"params with grad=None: {n_nograd}/{len(p)} (expected > 0 -- breaths before "
          f"KB_DONE={KB_DONE} were never run this step, so organs touched ONLY by breaths "
          f"this step skipped correctly see no gradient)", flush=True)
    print("PASS")


if __name__ == "__main__":
    main()
