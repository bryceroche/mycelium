"""rtb_buffer.py -- THE BUFFER WRITER (THE REPLAY-TRAINED BREATH, gap (a); 2026-10-11, worktree
mycelium-wt28, branch rtb; ledger 2026-10-11 17:21 registration, docs/replay_gym.md §5).

Runs the CURRENT checkpoint (--ckpt) over a sample of N diet rows (--train / --n) with the full
breath loop, host-side identical to scripts/loop_val.py's own read() recipe for an ALT2 (no
ALT3) family (see rtb_lib.build_consult_kwargs's docstring for why no live consult needs
replaying here), and dumps a snapshot at every kb in KB_DONE_RANGE (default 1..ALG_BREATH-2) --
the exact set of breath-done points THE REPLAY-TRAINED BREATH's training step needs to resume
from (kb_target = kb_done+1 covers kb_target in 2..ALG_BREATH-1; breaths 0/1 train on-policy,
never from this buffer -- the spec's own "breath 0 always on-policy", and kb_done=0 is not even
capturable by the snapshot hook, which only fires for kb>=1 inside the loop).

Rows are chunked at CHUNK (default 8, matching this family's own training BATCH -- each snapshot
file is then literally "one ready-made training batch", no slicing needed downstream) and one
forward() call per chunk writes ALL of KB_DONE_RANGE's snapshots at once (the multi-point
ALG_SNAP_AT extension in phase1_algebra_head.py, this branch) instead of one call per kb.

Writes <out>/index.json (the manifest: ckpt path + sha256 + step, K_B, kb_done_range, chunk
size, and per-chunk {kb_done: npz path, row indices}) and <out>/chunk_<i>_kb<kb>.npz (the
snapshots themselves, via _snap_dump -- read back with snap_load/replay(), never np.load
directly). THE MEMMAP LAW: this script writes one chunk at a time and never holds the whole
buffer in RAM; a consumer (do_train under ALG_RTB=1) mmaps each npz as needed, per CLAUDE.md §5.

ALG_RTB_WRITE=1 is required in the environment as a safety door -- this script performs forward
passes and disk writes but never a training step and never touches .cache/gpu.lock itself (the
caller's env/device choice decides CPU vs the real card; this script does not flock anything),
consistent with the brief's "ALG_RTB_WRITE=1; a script scripts/rtb_buffer.py" naming.

Usage:
  ALG_RTB_WRITE=1 DEV=CPU <FAM env...> .venv/bin/python3 scripts/rtb_buffer.py \\
      --ckpt .cache/sharp_PK_241.safetensors --out .cache/rtb_buffer_PK_241 --n 64 --chunk 8
"""
import argparse
import os
import sys
import time

sys.path.insert(0, "."); sys.path.insert(0, "scripts")

assert int(os.environ.get("ALG_RTB_WRITE", "0")), (
    "ALG_RTB_WRITE=1 is required -- the safety door THE BUILD names for this script")

import numpy as np

import rtb_lib as RL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=64, help="number of diet rows to sample")
    ap.add_argument("--chunk", type=int, default=8, help="rows per snapshot file (= training BATCH)")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kb-done", default="", help="comma list overriding the default 1..K_B-2")
    args = ap.parse_args()

    import phase1_algebra_head as H
    from tinygrad.nn.state import safe_load

    K_B = int(os.environ.get("ALG_BREATH", "1"))
    assert K_B >= 3, f"ALG_BREATH={K_B}: the buffer needs at least one resumable breath (kb_done in 1..K_B-2)"
    kb_done_range = ([int(x) for x in args.kb_done.split(",") if x]
                      if args.kb_done else list(range(1, K_B - 1)))
    assert all(1 <= kb < K_B for kb in kb_done_range), kb_done_range

    p = H.build_params(0)
    sd = safe_load(args.ckpt)
    assert set(sd.keys()) == set(p.keys()), (
        sorted(set(sd) - set(p.keys()))[:4], sorted(set(p.keys()) - set(sd))[:4])
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    ckhash = RL.ckpt_hash(args.ckpt)

    vs, vst, vtk, vg, vse = H.load_alg(args.split)
    rng = np.random.RandomState(args.seed)
    n_pick = min(args.n, len(vs))
    row_ids = rng.choice(len(vs), n_pick, replace=False)
    row_ids.sort()

    os.makedirs(args.out, exist_ok=True)
    manifest = {
        "ckpt": os.path.abspath(args.ckpt), "ckpt_sha16": ckhash,
        "K_B": K_B, "kb_done_range": kb_done_range, "chunk": args.chunk,
        "written_at": time.time(), "n_rows": int(n_pick), "split": args.split,
        "chunks": [],   # list of {"idx": i, "rows": [orig row ids...], "files": {kb: path}}
    }

    from tinygrad import Tensor, dtypes
    t0 = time.time()
    n_chunks = 0
    for c0 in range(0, n_pick, args.chunk):
        sl = row_ids[c0:c0 + args.chunk]
        ts = Tensor(np.ascontiguousarray(vst[sl]), dtype=dtypes.half)
        tk = Tensor(vtk[sl].astype(np.float32), dtype=dtypes.float)
        se = Tensor(vse[sl].astype(np.int32), dtype=dtypes.int)
        kwargs = RL.build_consult_kwargs(H, p, ts, tk, se, vs, sl)

        kb_list = ",".join(str(kb) for kb in kb_done_range)
        tmpl = os.path.join(args.out, f"chunk_{n_chunks}_kb{{kb}}.npz")
        os.environ["ALG_SNAP_AT"] = kb_list
        os.environ["ALG_SNAP_OUT"] = tmpl
        _ = H.forward(p, ts, tk, se, **kwargs)
        os.environ.pop("ALG_SNAP_AT", None); os.environ.pop("ALG_SNAP_OUT", None)

        files = {str(kb): tmpl.format(kb=kb) for kb in kb_done_range}
        for kb, path in files.items():
            assert os.path.exists(path), f"snapshot missing: {path} (kb={kb})"
        manifest["chunks"].append({"idx": n_chunks, "rows": [int(x) for x in sl], "files": files})
        n_chunks += 1
        if n_chunks % 10 == 0 or c0 + args.chunk >= n_pick:
            print(f"[rtb-buffer] chunk {n_chunks}/{(n_pick + args.chunk - 1) // args.chunk} "
                  f"written ({time.time() - t0:.1f}s)", flush=True)

    RL.save_manifest(args.out, manifest)
    n_files = n_chunks * len(kb_done_range)
    size_mb = sum(os.path.getsize(f["files"][str(kb)])
                  for f in manifest["chunks"] for kb in kb_done_range) / 1e6
    print(f"[rtb-buffer] wrote {args.out}/index.json: {n_chunks} chunks x {len(kb_done_range)} "
          f"kb points = {n_files} files, {size_mb:.1f} MB total "
          f"({size_mb / max(n_pick, 1):.3f} MB/row), ckpt={args.ckpt} ({ckhash}) "
          f"in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
