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


def run(onp, se, nv, ma, mass_out=None, workers=None):
    global _POOL
    B = int(se.shape[0])
    workers = workers or int(os.environ.get("ALG_FACTS_WORKERS", "0")) or max(1, (os.cpu_count() or 2) - 2)
    n = min(workers, B)
    chunks = [c for c in np.array_split(np.arange(B), n) if len(c)]
    tasks = [({k: np.ascontiguousarray(v[c]) for k, v in onp.items()}, se[c], nv[c], ma[c], mass_out is not None)
             for c in chunks]
    if n <= 1:
        res = [_work(t) for t in tasks]
    else:
        if _POOL is None:
            import multiprocessing as mp
            _POOL = mp.get_context("spawn").Pool(workers, initializer=_init)
        res = _POOL.map(_work, tasks, chunksize=1)
    fb = np.concatenate([r[0] for r in res], 0)
    if mass_out is not None:
        mass_out[:] = np.concatenate([r[1] for r in res], 0)
    return fb
