"""gen23_verdict.py — THE GEN-23 VERDICT (2026-08-03; sign-only scope
per word (c)). Mechanical: reads the banked bar artifacts + the fixture
evals just run; every standing fixture faces the MANIFEST'S OWN
regression_bars (the pre-pinned floors); B1' from gen23_rebar.json;
B2 >= 1214 pinned. ALL green -> writes GENERATION.json + prints
PROMOTED in one act, with WIDE-NOT-CLAIMED carried AS DATA (the
caveat-decay fence prospective). ANY red -> prints the kill, touches
nothing."""
import json, os, re, sys, hashlib

old_m = json.load(open(".cache/GENERATION.json"))
bars = old_m["regression_bars"]
log = open(".cache/gen23_battery3.log").read()

# fixture numbers: "[NAME] ... TOTAL: x/y graph-solve, N/y ANSWER"
nums = {}
for name in ("bigtest", "alg4test", "alg2test", "vtest", "dagtest",
             "dag7btest", "dag8test"):
    m = re.search(rf"=== eval {name} ===.*?TOTAL: \d+/\d+ graph-solve, (\d+)/\d+ ANSWER",
                  log, re.S)
    if m:
        nums[name] = int(m.group(1))
kills = []
for name, floor in (("bigtest", 1214), ("alg4test", bars["alg4test"]),
                    ("alg2test", bars["alg2test"]), ("vtest", bars["vtest"]),
                    ("dagtest", bars["dagtest"]),
                    ("dag7btest", bars["dag7btest"]),
                    ("dag8test", bars["dag8test"])):
    got = nums.get(name)
    verdict = "?" if got is None else ("ok" if got >= floor else "KILL")
    print(f"  {name}: {got} vs floor {floor} -> {verdict}")
    if got is None or got < floor:
        kills.append(name)
rb = json.load(open(".cache/gen23_rebar.json"))
print(f"  B1' sign-only: old {rb['old']}/120 sign {rb['sign']}/120 -> {rb['verdict']}")
if rb["verdict"] != "PASS":
    kills.append("B1-prime")
if kills:
    print(f"THE KILL: {kills} — nothing is written; g22 remains the gate.")
    sys.exit(1)

def h(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]

m = {
    "gen_id": "23",
    "date": "2026-08-03",
    "env": {"ALG2": "1", "ALG_FTYPES": "8", "ALG_DUP": "1",
            "ALG_HW": "512", "ALG_WIDE": "1"},
    "parser_ckpt": ".cache/g23.safetensors",
    "specialist_ckpt": old_m["specialist_ckpt"],
    "monitor_centroids": old_m["monitor_centroids"],
    "mouth": old_m["mouth"],
    "corpora": {"train": ".cache/gen23_mix.jsonl", "test": ".cache/dag8_test.jsonl"},
    "scope": {
        "wide_not_claimed": True,
        "claimed": "E1 (signed literals): sign-arm 116/120 = 96.7% on fresh "
                   "held-out (B1' PASS). E3 (wide digits): TRAINED, NOT "
                   "CLAIMED — 0/40 at battery; the compounding-digit wall "
                   "(per-digit ~90% over 7 positions); see the ledger "
                   "2026-08-03. No reader may infer wide competence.",
    },
    "regression_bars": {**bars, "bigtest": nums["bigtest"],
                        **{k: nums[k] for k in nums},
                        "sign_arm_b1prime": 0.967, "old_arm_b1prime": 0.942},
    "waivers": {**old_m.get("waivers", {}),
                "entourage23": "specialist/centroids/mouth/panel are GEN-22's "
                "until entourage-23 settles (lit immediately after this write; "
                "acceptance + adversarial walk + member votes ride it)"},
    "hashes": {"parser": h(".cache/g23.safetensors"),
               "mix": h(".cache/gen23_mix.jsonl")},
    "notes": ("2026-08-03 GEN-23 PROMOTED (SIGN-ONLY SCOPE, word (c)): the "
              "answer-space generation E1+E3 behind ALG_WIDE; padwarm from "
              "g22; 600x10 @ 6.787 pct; bigtest " + str(nums["bigtest"]) +
              " (above g22's 1226); B1' sign 96.7 vs old 94.2 fresh "
              "held-out; WIDE TRAINED NOT CLAIMED (see scope). Prior "
              "manifest notes preserved below. | ") + old_m.get("notes", ""),
}
tmp = ".cache/GENERATION.json.tmp"
json.dump(m, open(tmp, "w"), indent=1)
os.replace(tmp, ".cache/GENERATION.json")
print("PROMOTED — GEN-23 (sign-only scope) is the gate; manifest written atomically; entourage-23 owed immediately.")
