"""battery_gen16.py — THE GEN-16 PROMOTION BATTERY (candidate: the record-holding crown reader; macro column at saturation vintage, reported never barred) (2026-07-20, fired on
the word). Candidates: fire_armA (prime control, record 1204) and
fire_armC1 (paired-spread, the lean). Stages per candidate:
  1. seven-fixture eval sweep (the standing rows)
  2. bigtest member votes (standing view seeds -> cert-v2 join vs banked
     armB/cap2x + sentinel columns)
  3. acceptance (book-1 paired gate under the candidate)
  4. adversarial fixture walk (20 specimens, 5-view + guard flags)
Then gen15_verdict.py — the only pen — reads bars and writes the
manifest or refuses. Everything alongside gen-14; nothing touched.
"""
import json, os, subprocess, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")

ENV = {"DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "8", "ALG_DUP": "1"}
CANDS = {"V4": ".cache/crown_reader_v4.safetensors"}
FIXTURES = [("bigtest", ".cache/algebra_nl_bigtest.jsonl"),
            ("alg4test", ".cache/algebra4_nl_test.jsonl"),
            ("alg2test", ".cache/algebra2_nl_test.jsonl"),
            ("vtest", ".cache/algv_test_verbose.jsonl"),
            ("dagtest", ".cache/dag_test.jsonl"),
            ("dag7btest", ".cache/dag7b_test.jsonl"),
            ("dag8test", ".cache/dag8_test.jsonl")]


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


for cand, ckpt in CANDS.items():
    log = f".cache/gen15_{cand}.log"
    open(log, "w").write(f"=== GEN-15 BATTERY: candidate {cand} = {ckpt} ===\n")
    for name, path in FIXTURES:
        print(f"=== [{cand}] eval {name} ===", flush=True)
        open(log, "a").write(f"--- {name} ---\n")
        sh(".venv/bin/python3 scripts/phase1_algebra_head.py --eval",
           {"ALG_CKPT": ckpt, "ALG_TEST": path, "ALG_TEST_NAME": name},
           tail=1, logf=log)
    print(f"=== [{cand}] member votes (bigtest, standing seeds) ===", flush=True)
    sh(".venv/bin/python3 scripts/lattice_member_votes.py",
       {"MEMBER_CKPT": ckpt, "MEMBER_HW": "512", "MEMBER_DUP": "1",
        "OUT": f".cache/lattice_gen15_{cand}.json"}, tail=1)
    print(f"=== [{cand}] acceptance (book-1 paired gate) ===", flush=True)
    open(log, "a").write("--- acceptance ---\n")
    sh(".venv/bin/python3 scripts/book1_paired_gate.py",
       {"GATE_CKPT": ckpt}, tail=3, logf=log)
    print(f"=== [{cand}] adversarial walk ===", flush=True)
    sh(f".venv/bin/python3 scripts/adversarial_walk.py {cand} {ckpt}", tail=2)

print("=== BATTERY COLLECTED — the verdict holds the pen ===", flush=True)
sh(".venv/bin/python3 scripts/gen16_verdict.py", tail=30)
