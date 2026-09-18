"""facts_pool.py — THE FACTS PASS ACROSS CPU CORES (perf audit #3, 2026-09-11).
Rows are independent; the solver's propagation per row is python. A spawn
pool whose children set DEV=CPU before importing the head (they never touch
the GPU — the AM single-process law) runs alt2_fact_buf on row chunks and the
parent concatenates in order: the same bytes as the serial pass (the
mask-prep cache asserts it)."""
import os, sys
import numpy as np
_H = None
_POOL = None


def _init():
    os.environ["DEV"] = "CPU"


def _work(args):
    global _H
    if _H is None:
        sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
        import phase1_algebra_head as H
        _H = H
    onp, se, nv, ma, want_mass = args
    mo = np.zeros((len(nv), _H.K_VARS), np.float32) if want_mass else None
    fb = _H.alt2_fact_buf(onp, se, nv, ma, mass_out=mo)
    return fb, mo


def _tagged(args):
    i, t = args
    return i, _work(t)


def _map_hangproof(tasks, workers, wall=None):
    """THE HANG-PROOF MAP (2026-09-17): Pool.map has no timeout — a worker that dies takes its chunk with it
    and the map waits forever (the diet-v3 arm: 64 min asleep, GPU idle). Results are collected unordered
    with a wall per result; a chunk that never returns is recomputed — once more in a fresh pool, then
    in-process (the facts are needed for every row; the serial pass is the reference)."""
    global _POOL
    import multiprocessing as mp
    wall = wall or float(os.environ.get("ALG_FACTS_WALL", "600"))
    out = [None] * len(tasks); todo = list(range(len(tasks)))
    for attempt in range(2):
        if not todo: break
        if _POOL is None:
            _POOL = mp.get_context("spawn").Pool(workers, initializer=_init)
        it = _POOL.imap_unordered(_tagged, [(i, tasks[i]) for i in todo], chunksize=1)
        try:
            for _ in range(len(todo)):
                i, r = it.next(timeout=wall); out[i] = r
        except mp.TimeoutError:
            pass
        todo = [i for i in todo if out[i] is None]
        if todo:
            print(f"[facts-pool] {len(todo)} chunk(s) never returned within {wall:.0f} s (attempt {attempt + 1}) — the pool is rebuilt", flush=True)
            try: _POOL.terminate()
            except Exception: pass
            _POOL = None
    for i in todo:
        print(f"[facts-pool] chunk {i} computed in-process", flush=True); out[i] = _work(tasks[i])
    return out


def run(onp, se, nv, ma, mass_out=None, workers=None):
    global _POOL
    B = int(se.shape[0])
    # THE WORKER CAP (2026-09-17): 14 children each importing the head beside a ~18 GB parent on a 30 GB box
    # swapped and (most likely) lost a worker to the OOM killer — the map then waited forever. 8 by default.
    workers = workers or int(os.environ.get("ALG_FACTS_WORKERS", "0")) or max(1, min(8, (os.cpu_count() or 2) - 2))
    n = min(workers, B)
    chunks = [c for c in np.array_split(np.arange(B), n) if len(c)]
    tasks = [({k: np.ascontiguousarray(v[c]) for k, v in onp.items()}, se[c], nv[c], ma[c], mass_out is not None)
             for c in chunks]
    if n <= 1:
        res = [_work(t) for t in tasks]
    else:
        res = _map_hangproof(tasks, workers)
    fb = np.concatenate([r[0] for r in res], 0)
    if mass_out is not None:
        mass_out[:] = np.concatenate([r[1] for r in res], 0)
    return fb
