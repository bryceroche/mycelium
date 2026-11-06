"""gsm_bt_profile.py -- THE STEP PROFILE (ledger 2026-11-06 06:35, the perf line (a); a measurement, no bars).

The tinygrad Trainer.step (scripts/gsm_breath_tg.py, imported, never edited) re-timed in segments: host batch build (BR.batch + to_np), host
gold / masks / notebook (numpy), uploads (Tensor builds), the JIT call to the loss's sync. BT4RB8's flags (set by the caller's env), batch 64,
TG_STEPS steps (300), steady frame from step 21. Then the periodic eval (BR.score on ORIG + TEST 600, torch CPU) and the FINAL scoring (all
splits) timed once on the init weights.  -> .cache/gsm_bt_profile.json + printed table
"""
import json, math, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gsm_breath_reader as BR          # FIRST (the 11-04 import rule)
import gsm_breath_tg as TG
import torch
from tinygrad import Tensor, TinyJit

torch.set_num_threads(int(os.environ.get("TOY_THREADS", "4")))
Dt, states = BR.TR.load(); tree = BR.TR.load_tree()
BR.RF = np.load(BR.GD.V1.C(BR.rolef_name()), mmap_mode="r")
K, NCYC, NT = BR.K, BR.NCYC, TG.NT
SEG = ("batch", "host_glue", "upload", "jit")


def step(tr, its, rng, T):
    t0 = time.perf_counter()
    t, E = BR.batch(its, states, tree, rng); t = TG.to_np(t); L = t["pad"].shape[1]
    for b, it in enumerate(its): E[b]["vals"] = it["vals"]
    t1 = time.perf_counter()
    progs, gt = BR.gold(E)
    m = np.array([e["m"] for e in E]); rr = rng.random(len(E))[:, None]
    masked = rng.random((len(E), K)) < rr
    masked[np.arange(len(E)), rng.integers(0, np.minimum(m + 1, K))] = True
    cyc = np.clip(np.floor(NCYC * (1 - rr[:, 0])), 0, NCYC - 1).astype(np.int32)
    soft = BR.SOFT and rng.random() < 0.5; assert not soft
    Lq = L - K; Lb = max(64, int(math.ceil(Lq / 32) * 32)); t = TG.repad(t, Lb)
    nb = BR.notebook(E, Lb + K, progs, ~masked, gt, np.ones((len(E), K), np.float32))
    t2 = time.perf_counter()
    tt = TG.tens(t); nbt = TG.nb_t(nb); gtt = Tensor(gt.astype(np.int32)); cyct = Tensor(cyc)
    sp = Tensor.zeros(len(E), K, NT).contiguous()
    args = [*[tt[k] for k in TG.FIELDS if k in tt], *nbt, gtt, Tensor(masked.astype(np.float32)),
            Tensor((t["am"] & masked).astype(np.float32)), Tensor(t["va"]), Tensor(t["vb"]), cyct, sp]
    Tensor.realize(*[a for a in args if isinstance(a, Tensor)])
    t3 = time.perf_counter()
    key = (Lb, soft)
    if key not in tr.jits: tr.jits[key] = TinyJit(tr._step)
    lo = float(tr.jits[key](*args).numpy())
    t4 = time.perf_counter()
    for k_, a, b in zip(SEG, (t0, t1, t2, t3), (t1, t2, t3, t4)): T[k_].append(b - a)
    return lo, key


if __name__ == "__main__":
    STEPS = int(os.environ.get("TG_STEPS", "300")); SEED = BR.SEED
    un = TG.unused_params(Dt, states, tree)
    torch.manual_seed(SEED); tn = BR.make_net(); P = TG.from_torch(tn.state_dict())
    tr = TG.Trainer(P, BR.LR, un); train = Dt["TRAIN"]; rng = np.random.default_rng(SEED)
    T = {k: [] for k in SEG}; keys = []; tw = []
    for s in range(1, STEPS + 1):
        its = [train[int(i)] for i in rng.integers(0, len(train), BR.BATCH)]
        w0 = time.perf_counter(); lo, key = step(tr, its, rng, T); tw.append(time.perf_counter() - w0); keys.append(key)
        if s % 50 == 0: print(f"[prof] step {s} loss {lo:.4f} wall {np.mean(tw[-50:]):.3f}s/step buckets {len(tr.jits)}", flush=True)
    # steady frame: steps 21.. and only steps whose bucket was already captured (a first call per bucket captures)
    seen, steady = set(), []
    for j, k_ in enumerate(keys):
        if j >= 20 and k_ in seen: steady.append(j)
        seen.add(k_)
    o = dict(steps=STEPS, steady_n=len(steady), buckets=sorted(map(list, tr.jits)), wall=float(np.mean([tw[j] for j in steady])),
             **{k: float(np.mean([T[k][j] for j in steady])) for k in SEG})
    try: o["kernels_per_step"] = {str(k): len(j.captured.jit_cache) for k, j in tr.jits.items() if getattr(j, "captured", None)}
    except Exception as e: o["kernels_per_step"] = repr(e)
    TG.to_torch(P, tn); tn.eval()
    e0 = time.perf_counter(); [BR.score(tn, Dt[k][:600], states, tree) for k in ("ORIG", "TEST")]; o["eval_1000"] = time.perf_counter() - e0
    o["final_splits"] = {}
    for k in BR.SPLITS:
        e0 = time.perf_counter(); BR.score(tn, Dt[k], states, tree); o["final_splits"][k] = time.perf_counter() - e0
    json.dump(o, open(".cache/gsm_bt_profile.json", "w"), indent=1)
    w = o["wall"]
    print(f"[prof] steady steps {o['steady_n']} | wall {w:.3f} s/step | " + " | ".join(f"{k} {o[k]:.3f} ({o[k] / w:.0%})" for k in SEG))
    print(f"[prof] kernels per captured step: {o['kernels_per_step']}")
    fin = sum(o["final_splits"].values()); run = lambda n: n * w + (n // 1000) * o["eval_1000"] + fin
    print(f"[prof] eval every 1000 steps {o['eval_1000']:.0f}s | FINAL scoring {fin:.0f}s {o['final_splits']}")
    print(f"[prof] modeled run: 18k {run(18000) / 3600:.2f} h | 5k {run(5000) / 3600:.2f} h (train {18000 * w / 3600:.2f} / {5000 * w / 3600:.2f} h)")
