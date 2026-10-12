"""rtb_exposure_read.py -- THE EXPOSURE READ (THE REPLAY-TRAINED BREATH, part C of THE BUILD;
worktree mycelium-wt28, branch rtb; 2026-10-11). Per breath kb, on a sample of wild rows:
compares per-breath SLOT-CORRECTNESS when the state at that breath comes from

  (i)  the body's own previous breath (today's full 7-breath unroll, on-policy end to end), vs
  (ii) a buffer snapshot of the SAME rows at kb_done=kb-1 (replay(..., kb_stop=kb) -- one breath
       resumed from whatever checkpoint WROTE the buffer, which may be STALE relative to the
       checkpoint this read is otherwise using).

The gap between the two is the measurement the kill criterion ("masked down > 0.020 = the
exposure gap bites") reads off of: run this ONCE with a buffer written by the SAME checkpoint
being read (gap should be ~0, modulo the decode-exact float noise already characterized by
scripts/rtb_identity_check.py) and again with a buffer written by an OLDER checkpoint (gap
measures real exposure bias) -- the real arm's own chain does the second form; this script does
either, given whatever --buffer it is pointed at.

METRIC (stated as a proxy, not the official fac-exact pipeline): per PRESENT gold slot, 1.0 if
ftype/res argmax AND the args argmax (for rel slots) all match gold, else 0.0, averaged over
present slots per row then over rows -- cheap, decode-free, same spirit as the per-slot accuracy
_loss_single already computes internally, built fresh here because the exposure comparison needs
breath-by-breath per-SLOT correctness, which the solver-backed fac-exact pipeline (decode() +
answer-key check) does not expose per breath.

DEV=CPU asserted; no .cache/gpu.lock touch.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("DEV", "CPU")
assert os.environ.get("DEV") == "CPU", f"DEV={os.environ.get('DEV')!r}: this script is CPU-only"

import numpy as np

import rtb_lib as RL


def _slot_correct(heads, gold, idx):
    """(n_rows,) mean per-row slot-correctness proxy -- heads: realized numpy dict (ftype, res,
    args, pres); gold: the load_alg gold dict; idx: the row ids these heads correspond to."""
    ftype_p = heads["ftype"].argmax(-1)
    res_p = heads["res"].argmax(-1)
    args_p = heads["args"].argmax(-1)
    pres_g = gold["presence"][idx] > 0.5
    ftype_g = gold["ftype"][idx]
    res_g = gold["res"][idx]
    args_g = gold["args"][idx].argmax(-1)
    ok = (ftype_p == ftype_g) & (res_p == res_g) & (args_p == args_g)
    n_rows = pres_g.shape[0]
    out = np.zeros(n_rows, np.float64)
    for i in range(n_rows):
        m = pres_g[i]
        out[i] = float(ok[i][m].mean()) if m.sum() else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="the checkpoint doing the READING (today's body)")
    ap.add_argument("--buffer", required=True, help="an rtb_buffer.py output dir (may be stale or fresh)")
    args = ap.parse_args()

    import phase1_algebra_head as H
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load

    p = H.build_params(0)
    sd = safe_load(args.ckpt)
    assert set(sd.keys()) == set(p.keys())
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

    manifest = RL.load_manifest(args.buffer)
    K_B = manifest["K_B"]
    vs, vst, vtk, vg, vse = H.load_alg(manifest.get("split", "test"))

    per_breath_own = {kb: [] for kb in range(K_B)}
    per_breath_buf = {kb_done + 1: [] for kb_done in manifest["kb_done_range"]}
    t0 = time.time()
    for c in manifest["chunks"]:
        rows = np.array(c["rows"])
        ts = Tensor(np.ascontiguousarray(vst[rows]), dtype=dtypes.half)
        tk = Tensor(vtk[rows].astype(np.float32), dtype=dtypes.float)
        se = Tensor(vse[rows].astype(np.int32), dtype=dtypes.int)
        kwargs = RL.build_consult_kwargs(H, p, ts, tk, se, vs, rows)

        out_full = H.forward(p, ts, tk, se, **kwargs)
        breaths_np = [{k: t.realize().numpy() for k, t in b.items() if k in ("ftype", "res", "args")}
                      for b in out_full["breaths"]]
        for kb, b in enumerate(breaths_np):
            per_breath_own[kb].append(_slot_correct(b, vg, rows))

        for kb_done in manifest["kb_done_range"]:
            kb_target = kb_done + 1
            snap = H.snap_load(c["files"][str(kb_done)])
            out_r = H.replay(p, snap, kb_done, kb_stop=kb_target + 1)
            heads = {k: out_r[k].realize().numpy() for k in ("ftype", "res", "args")}
            per_breath_buf[kb_target].append(_slot_correct(heads, vg, rows))

    print(f"[rtb-exposure] ckpt={args.ckpt} buffer={args.buffer} "
          f"({len(manifest['chunks'])} chunks, {sum(len(c['rows']) for c in manifest['chunks'])} rows) "
          f"in {time.time() - t0:.1f}s", flush=True)
    print(f"{'kb':>4} {'own-unroll':>12} {'from-buffer':>12} {'gap(own-buf)':>14}")
    gaps = {}
    for kb in range(K_B):
        own = np.nanmean(np.concatenate(per_breath_own[kb])) if per_breath_own[kb] else float("nan")
        if kb in per_breath_buf and per_breath_buf[kb]:
            buf = np.nanmean(np.concatenate(per_breath_buf[kb]))
            gap = own - buf
            gaps[kb] = gap
            print(f"{kb:>4} {own:>12.4f} {buf:>12.4f} {gap:>14.4f}")
        else:
            print(f"{kb:>4} {own:>12.4f} {'--':>12} {'--':>14}")
    if gaps:
        print(f"[rtb-exposure] mean |gap| over breaths with a buffered comparison: "
              f"{np.mean([abs(g) for g in gaps.values()]):.4f}")


if __name__ == "__main__":
    main()
