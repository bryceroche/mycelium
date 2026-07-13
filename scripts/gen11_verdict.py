"""gen11_verdict.py — the manifest-writing verdict (2026-07-13; the law's
first enforcement: PROSE PROMOTIONS DON'T MOVE MACHINES). Parses the
gen-11 chain log, checks EVERY pinned bar mechanically, and either writes
GENERATION.json (specialist carried with an EXPLICIT one-generation
waiver) and prints PROMOTED — or prints the kill. The word and the JSON
are one act: no manifest write, no PROMOTED.
Usage: gen11_verdict.py <chain-log-path>
"""
import json, re, sys, hashlib

log = open(sys.argv[1]).read()

def answer_of(row):
    m = re.search(rf"--- {row} ---\s*\n\s*TOTAL: \d+/\d+ graph-solve, (\d+)/\d+ ANSWER",
                  log)
    return int(m.group(1)) if m else -1

def perkind(kind):
    m = re.findall(rf"^\s+{kind}\s*:\s*\d+/\s*\d+ = ([0-9.]+)", log, re.M)
    return float(m[-1]) if m else -1.0

acc = re.findall(r"ACCEPTANCE: (\d)/8 banked", log)
acc = int(acc[-1]) if acc else -1
certm = re.findall(r"CERT-V2\s*:\s*coverage\s*\d+ \([0-9.]+%\) precision ([0-9.]+)", log)
cert = float(certm[-1]) if certm else -1.0

BARS = [
    ("bigtest", answer_of("bigtest"), 1130),
    ("alg4test", answer_of("alg4test"), 380),
    ("alg2test", answer_of("alg2test"), 560),
    ("vtest", answer_of("vtest"), 598),
    ("dagtest", answer_of("dagtest"), 660),
    ("dag7btest", answer_of("dag7btest"), 500),
    ("dag8test", answer_of("dag8test"), 500),
    ("acceptance", acc, 8),
    ("sq", perkind("sq"), 0.70),
    ("fdiv", perkind("fdiv"), 0.62),
    ("coupled-ish", perkind("coupled-ish"), 0.65),
    ("ladder", perkind("ladder"), 0.50),
    ("cert-v2-precision", cert, 0.998),
]
fails = [(n, v, b) for n, v, b in BARS if v < b]
print("=== GEN-11 VERDICT (mechanical) ===")
for n, v, b in BARS:
    print(f"  {'PASS' if v >= b else 'FAIL':4s} {n:18s} {v} (bar {b})")
if fails:
    print(f"\nKILL: {len(fails)} bar(s) broken -> NO PROMOTION; "
          f"gate remains gen-9b; manifest untouched.")
    sys.exit(0)

def fh(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]

m = json.load(open(".cache/GENERATION.json"))
m.update({
    "gen_id": "11", "date": "2026-07-13",
    "parser_ckpt": ".cache/phase1_gen11_head.safetensors",
    "corpora": {"train": ".cache/algebra_mixed11_train.jsonl",
                "test": ".cache/dag8_test.jsonl"},
    "regression_bars": {n: b for n, _, b in BARS},
})
m["waivers"] = {"specialist": "ONE GENERATION (trained vs gen-9b errors); "
                              "remine rides the next entourage pass",
                "panel": "cert-v2 members armB + cap2x (panel-eligible bench)"}
m["hashes"]["parser"] = fh(".cache/phase1_gen11_head.safetensors")
m["hashes"]["train"] = fh(".cache/algebra_mixed11_train.jsonl")
json.dump(m, open(".cache/GENERATION.json", "w"), indent=1)
print("\n[manifest] WRITTEN: gen-11 is the gate "
      "(specialist waivered one generation, explicitly).")
print("PROMOTED")
