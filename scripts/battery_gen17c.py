"""battery_gen17b.py (generated from battery_gen17.py) — THE GEN-17 PROMOTION BATTERY (2026-07-23, fired on
the word; the zener-convened charter's arms). Candidates: g17_armF (flat
16k incumbent) and g17_armR (4x4k SGDR). Prefix PARAMETERIZED (the
gen-16 seam paid). Stages per candidate:
  1. nine-fixture eval sweep (seven standing + the two from-zero held
     fixtures: hundreds {3..9} givens, add-dup)
  2. bigtest member votes -> cert-v2 join vs banked armB/cap2x
  3. acceptance (book-1 paired gate)
  4. adversarial fixture walk (20 specimens)
Then gen17_verdict.py — the only pen — reads bars and writes the
manifest or refuses. Everything alongside gen-16; nothing touched.
"""
import json, os, subprocess, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")

GEN = "17c"
ENV = {"DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "8", "ALG_DUP": "1"}
CANDS = {"C": ".cache/g17c.safetensors"}
FIXTURES = [("bigtest", ".cache/algebra_nl_bigtest.jsonl"),
            ("alg4test", ".cache/algebra4_nl_test.jsonl"),
            ("alg2test", ".cache/algebra2_nl_test.jsonl"),
            ("vtest", ".cache/algv_test_verbose.jsonl"),
            ("dagtest", ".cache/dag_test.jsonl"),
            ("dag7btest", ".cache/dag7b_test.jsonl"),
            ("dag8test", ".cache/dag8_test.jsonl"),
            ("h3held", ".cache/gen17_hundreds_held.jsonl"),
            ("adupheld", ".cache/gen17_adup_held.jsonl")]


def sh(cmd, extra=None, tail=2, logf=None):
    env = dict(os.environ); env.update(ENV); env.update(extra or {})
    r = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True)
    out = (r.stdout + r.stderr)
    if logf:
        open(logf, "a").write(out)
    for l in out.strip().splitlines()[-tail:]:
        print(f"    {l}", flush=True)
    if r.returncode != 0:
        raise RuntimeError(f"stage failed: {cmd[:90]}")
    return out


# one-time state precompute for the two NEW held fixtures (standing
# fixtures' states are banked under their standing names)
for name, path in [("h3held", ".cache/gen17_hundreds_held.jsonl"),
                   ("adupheld", ".cache/gen17_adup_held.jsonl")]:
    if not os.path.exists(f".cache/phase1_alg_states_{name}.npz"):
        print(f"=== precompute held fixture {name} ===", flush=True)
        sh(".venv/bin/python3 scripts/phase1_algebra_head.py --precompute",
           {"ALG_TRAIN": path, "ALG_TRAIN_NAME": name, "PRECOMPUTE_ONLY": name},
           tail=1)

for cand, ckpt in CANDS.items():
    log = f".cache/gen{GEN}_{cand}.log"
    open(log, "w").write(f"=== GEN-{GEN} BATTERY: candidate {cand} = {ckpt} ===\n")
    for name, path in FIXTURES:
        print(f"=== [{cand}] eval {name} ===", flush=True)
        open(log, "a").write(f"--- {name} ---\n")
        sh(".venv/bin/python3 scripts/phase1_algebra_head.py --eval",
           {"ALG_CKPT": ckpt, "ALG_TEST": path, "ALG_TEST_NAME": name},
           tail=1, logf=log)
    print(f"=== [{cand}] member votes (bigtest, standing seeds) ===", flush=True)
    sh(".venv/bin/python3 scripts/lattice_member_votes.py",
       {"MEMBER_CKPT": ckpt, "MEMBER_HW": "512", "MEMBER_DUP": "1",
        "OUT": f".cache/lattice_gen{GEN}_{cand}.json"}, tail=1)
    print(f"=== [{cand}] acceptance (book-1 paired gate) ===", flush=True)
    open(log, "a").write("--- acceptance ---\n")
    sh(".venv/bin/python3 scripts/book1_paired_gate.py",
       {"GATE_CKPT": ckpt}, tail=3, logf=log)
    print(f"=== [{cand}] adversarial walk ===", flush=True)
    sh(f".venv/bin/python3 scripts/adversarial_walk.py {cand} {ckpt}", tail=2)

print("=== BATTERY COLLECTED — the verdict holds the pen ===", flush=True)
sh(f".venv/bin/python3 scripts/gen{GEN}_verdict.py", tail=40)
