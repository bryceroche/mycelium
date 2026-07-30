"""battery_gen22.py — THE GEN-22 PROMOTION BATTERY (2026-07-30, fired on
the word; the diet fire's candidate). Candidate: g22 (fdiv-on-derived
diet at 400x10 / 4.85%, SGDR 4x4k gentle continuation from g21).
Stages (the gen-21 harness, candidate swapped):
  1. nine-fixture eval sweep (seven standing + hundreds-held + add-dup)
  2. bigtest member votes -> cert-v2 join vs the manifest's panel
  3. acceptance (book-1 paired gate)
  4. adversarial fixture walk (20 specimens)
  5. band evals on the final segment's annealed snapshots (bar-noise law)
Then gen22_verdict.py — the only pen — reads bars and writes the
manifest (hash recompute mechanical, the deep-clean repair) or refuses.
Everything alongside gen-21; nothing touched until PROMOTED prints."""
import json, os, subprocess, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")

GEN = "22"
ENV = {"DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "8", "ALG_DUP": "1"}
CANDS = {"H": ".cache/g22.safetensors"}
# two-home fix 2026-07-30: the roster's one authority
from gate_fixtures import FIXTURES


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

# BAND EVALS (bar-noise law): 2 annealed snapshots from the FINAL segment
for cand in CANDS:
    for st in ("3000", "3500"):
        snap = f".cache/g22_seg4_s{st}.safetensors"
        log = f".cache/gen{GEN}_{cand}.log"
        for name, path in [("bigtest", ".cache/algebra_nl_bigtest.jsonl"),
                           ("alg4test", ".cache/algebra4_nl_test.jsonl")]:
            print(f"=== [{cand}] band eval s{st} {name} ===", flush=True)
            open(log, "a").write(f"--- band_{st}_{name} ---\n")
            sh(".venv/bin/python3 scripts/phase1_algebra_head.py --eval",
               {"ALG_CKPT": snap, "ALG_TEST": path, "ALG_TEST_NAME": name},
               tail=1, logf=log)

print("=== BATTERY COLLECTED — the verdict holds the pen ===", flush=True)
sh(f".venv/bin/python3 scripts/gen{GEN}_verdict.py", tail=40)
