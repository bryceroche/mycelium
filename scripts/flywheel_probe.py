"""flywheel_probe.py — the EMA flywheel's first probe (2026-07-23,
un-parked by the wobble census's own verdict). Uniform average of the
last three annealed snapshots (s3000/s3500/s4000) -> eval bigtest +
alg4. PASS = the average matches or beats the best constituent on
bigtest (smoothing recovers noise); FAIL = averaging blurs (basins
disagree), the flywheel re-parks with a measured no."""
import json, sys, os, re, subprocess
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from tinygrad.nn.state import safe_load, safe_save
from tinygrad import Tensor, dtypes

SNAPS = [".cache/g17c_s3000.safetensors", ".cache/g17c_s3500.safetensors",
         ".cache/g17c_s4000.safetensors"]
acc = None
for s in SNAPS:
    sd = safe_load(s)
    if acc is None:
        acc = {k: sd[k].numpy().astype(np.float64) for k in sd}
    else:
        for k in acc:
            acc[k] += sd[k].numpy().astype(np.float64)
out = {k: Tensor((v / len(SNAPS)).astype(np.float32), dtype=dtypes.float)
       for k, v in acc.items()}
safe_save(out, ".cache/g17c_flywheel.safetensors")
print(f"[flywheel] averaged {len(SNAPS)} snapshots -> .cache/g17c_flywheel.safetensors")

ENV = {"DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "8", "ALG_DUP": "1"}
res = {}
for name, path in [("bigtest", ".cache/algebra_nl_bigtest.jsonl"),
                   ("alg4test", ".cache/algebra4_nl_test.jsonl"),
                   ("h3held", ".cache/gen17_hundreds_held.jsonl")]:
    env = dict(os.environ); env.update(ENV)
    env.update({"ALG_CKPT": ".cache/g17c_flywheel.safetensors",
                "ALG_TEST": path, "ALG_TEST_NAME": name})
    r = subprocess.run(".venv/bin/python3 scripts/phase1_algebra_head.py --eval",
                       shell=True, env=env, capture_output=True, text=True)
    m = re.search(r"TOTAL: \d+/\d+ graph-solve, (\d+)/\d+ ANSWER", r.stdout + r.stderr)
    res[name] = int(m.group(1)) if m else -1
    print(f"[flywheel] {name}: {res[name]}", flush=True)
best_const = 1219  # s3000, the constituents' best bigtest
verdict = ("FLYWHEEL REAL — averaging recovers noise" if res["bigtest"] >= best_const
           else "FLYWHEEL BLURS — basins disagree; re-park with a measured no")
print(f"[flywheel] VERDICT: {verdict} (avg bigtest {res['bigtest']} vs best constituent {best_const})")
json.dump(dict(res=res, best_constituent=best_const, verdict=verdict),
          open(".cache/flywheel_probe.json", "w"), indent=1)
