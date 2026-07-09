"""generation_manifest.py — the versioned-generation manifest (2026-07-10,
Bryce's sync gut + relay candidates 1+4). Synchronization as MECHANISM:
every deployable artifact stamped with one generation; loaders can refuse
cross-generation mixes. --write captures the current consistent set;
--check verifies files exist + env matches; the full atomic
generation-bump (mine -> specialist -> centroids -> thresholds -> manifest)
is the registered v1.
USAGE: .venv/bin/python3 scripts/generation_manifest.py --write|--check
"""
import argparse, hashlib, json, os, sys

MANIFEST = ".cache/GENERATION.json"

GEN4 = {
    "gen_id": 4,
    "date": "2026-07-10",
    "env": {"ALG2": "1", "ALG_FTYPES": "6"},
    "parser_ckpt": ".cache/phase1_algebra4_head.safetensors",
    "specialist_ckpt": ".cache/phase1_algebra4_nack.safetensors",
    "monitor_centroids": ".cache/monitor_centroids_alg2.npz",
    "corpora": {
        "train": ".cache/algebra_mixed4_train.jsonl",
        "test": ".cache/algebra4_nl_test.jsonl",
        "repair": ".cache/algebra4_repair.jsonl",
    },
    "regression_bars": {"alg2test_oneshot": 541, "bigtest_oneshot": 959,
                        "alg4test_oneshot_raw": 331,
                        "alg4test_composed": 143},
    "known_stale": ["monitor_centroids (still tranche-1 space)",
                    "lattice thresholds (gen-1 fit)"],
    "notes": "gen-4 = apposition corpus + in-generation specialist; first "
             "transactional bump (manifest-last, everything alongside gen-3).",
}

GEN3 = {
    "gen_id": 3,
    "date": "2026-07-10",
    "env": {"ALG2": "1", "ALG_FTYPES": "6"},
    "parser_ckpt": ".cache/phase1_algebra3_head.safetensors",
    "specialist_ckpt": ".cache/phase1_algebra3_nack.safetensors",
    "monitor_centroids": ".cache/monitor_centroids_alg2.npz",
    "corpora": {
        "train": ".cache/algebra_mixed3_train.jsonl",
        "test": ".cache/algebra3_nl_test.jsonl",
        "repair": ".cache/algebra3_repair.jsonl",
    },
    "regression_bars": {"alg2test_oneshot": 505, "bigtest_oneshot": 915,
                        "alg3test_oneshot_raw": 233},
    "known_stale": ["monitor_centroids (tranche-1 space — rebuild rides the "
                    "generation bump)", "lattice thresholds (fitted gen-1)"],
    "notes": "gen-3 = tranche-2 head + in-generation specialist. The alg3 "
             "corpus carries the letter-starved term-var flaw; gen-4 = the "
             "apposition fix + regenerated corpus + full atomic bump.",
}


def fhash(p):
    if not os.path.exists(p):
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def write():
    m = dict(GEN4)
    m["hashes"] = {k: fhash(v) for k, v in
                   [("parser", m["parser_ckpt"]),
                    ("specialist", m["specialist_ckpt"]),
                    ("centroids", m["monitor_centroids"])] +
                   list(m["corpora"].items())}
    missing = [k for k, v in m["hashes"].items() if v is None]
    assert not missing, f"manifest refuses to write with missing: {missing}"
    json.dump(m, open(MANIFEST, "w"), indent=1)
    print(f"[gen] wrote {MANIFEST} (gen {m['gen_id']}); "
          f"{len(m['hashes'])} artifacts pinned")


def check():
    m = json.load(open(MANIFEST))
    bad = []
    for k, v in m["hashes"].items():
        cur = fhash({"parser": m["parser_ckpt"],
                     "specialist": m["specialist_ckpt"],
                     "centroids": m["monitor_centroids"],
                     **m["corpora"]}[k])
        if cur != v:
            bad.append(k)
    for ek, ev in m["env"].items():
        if os.environ.get(ek) != ev:
            bad.append(f"env:{ek}={os.environ.get(ek)}!={ev}")
    if bad:
        print(f"[gen] OUT OF SYNC (gen {m['gen_id']}): {bad}")
        sys.exit(1)
    print(f"[gen] consistent (gen {m['gen_id']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    write() if a.write else check()
