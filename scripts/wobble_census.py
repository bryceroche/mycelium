"""wobble_census.py — the checkpoint-variance census (gut #57's rider,
fired as the 17c kill's autopsy). Evals the 8 trajectory snapshots on
bigtest + alg4test; reports the wobble band per fixture."""
import json, os, subprocess, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import re

ENV = {"DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "8", "ALG_DUP": "1"}
SNAPS = [f".cache/g17c_s{s}.safetensors" for s in range(500, 4001, 500)]
FIX = [("bigtest", ".cache/algebra_nl_bigtest.jsonl"),
       ("alg4test", ".cache/algebra4_nl_test.jsonl")]
out = {}
for snap in SNAPS:
    step = snap.split("_s")[1].split(".")[0]
    out[step] = {}
    for name, path in FIX:
        env = dict(os.environ); env.update(ENV)
        env.update({"ALG_CKPT": snap, "ALG_TEST": path, "ALG_TEST_NAME": name})
        r = subprocess.run(".venv/bin/python3 scripts/phase1_algebra_head.py --eval",
                           shell=True, env=env, capture_output=True, text=True)
        m = re.search(r"TOTAL: \d+/\d+ graph-solve, (\d+)/\d+ ANSWER",
                      r.stdout + r.stderr)
        out[step][name] = int(m.group(1)) if m else -1
        print(f"[wobble] s{step} {name}: {out[step][name]}", flush=True)
for name, _ in FIX:
    vals = [out[s][name] for s in out]
    print(f"[wobble] {name}: min {min(vals)} max {max(vals)} "
          f"band {max(vals)-min(vals)}")
json.dump(out, open(".cache/wobble_census.json", "w"), indent=1)
print("[wobble] banked -> .cache/wobble_census.json")
