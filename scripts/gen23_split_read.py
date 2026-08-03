"""gen23_split_read.py — B1's per-arm split (the countersigned rider:
which terminal TOOK?) on the same held-out seeds, full guard stack."""
import os, sys, json, random, signal
os.environ["ALG_WIDE"] = "1"
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_HW", "512"); os.environ.setdefault("ALG_DUP", "1")
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from gen23_bars import mint_arm, parse_solve  # same instrument, same guards
res = {}
for cls, seed in (("sign", 104100), ("wide", 104200), ("both", 104300)):
    arm = mint_arm(cls, 40, seed)
    ok = parse_solve(arm)
    res[cls] = {"ok": ok, "n": len(arm)}
    print(f"[split] {cls}: {ok}/{len(arm)} = {ok/len(arm):.1%}", flush=True)
json.dump(res, open(".cache/gen23_split.json", "w"), indent=1)
print("[saved] .cache/gen23_split.json", flush=True)
