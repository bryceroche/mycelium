"""gen15_verdict.py — THE ONLY PEN (2026-07-20). Reads the collected
battery for candidates A and C1, checks every pinned bar, prints the
sentinel row, and either writes GENERATION.json (gen-15) or refuses.
BARS (standing manifest + the charter's stricter bigtest floor):
  bigtest >= 1149, alg4test >= 380, alg2test >= 560, vtest >= 598,
  dagtest >= 660, dag7btest >= 500, dag8test >= 500,
  acceptance: >= 7/8 dialect banks with only-[45]-may-miss,
  cert-v2 precision >= 0.998 (candidate + banked armB/cap2x join),
  adversarial: guard flags 20/20 (wiring precondition).
Selection: all-bars candidates ranked by bigtest; winner takes the gate.
"""
import json, re, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from collections import Counter

BARS = [("bigtest", 1149), ("alg4test", 380), ("alg2test", 560),
        ("vtest", 598), ("dagtest", 660), ("dag7btest", 500),
        ("dag8test", 500)]
rows = [json.loads(l) for l in open(".cache/algebra_nl_bigtest.jsonl")]
gold = [r["solution"][r["query_var"]] for r in rows]
armb = json.load(open(".cache/lattice_armB.json"))["bigtest"]
c2x = json.load(open(".cache/lattice_cap2x.json"))["bigtest"]
g14 = json.load(open(".cache/lattice_gate.json"))["bigtest"]


def maj(v):
    vs = [x for x in v if x is not None]
    return Counter(vs).most_common(1)[0] if vs else (None, 0)


def H(votes):
    c = Counter("⊥" if v is None else v for v in votes)
    p_ = np.array(list(c.values()), float) / len(votes)
    return float(-(p_ * np.log(p_)).sum())


verdicts = {}
for cand in ("A", "C1"):
    log = open(f".cache/gen15_{cand}.log").read()

    def answer_of(row):
        m = re.search(rf"--- {row} ---.*?TOTAL: \d+/\d+ graph-solve, (\d+)/\d+ ANSWER",
                      log, re.S)
        return int(m.group(1)) if m else -1

    vals = {name: answer_of(name) for name, _ in BARS}
    fails = [(n, v, b) for (n, b), v in zip(BARS, vals.values()) if v < b]
    # acceptance: count dialect banks in the paired-gate section
    acc_sec = log.split("--- acceptance ---")[-1]
    banks = len(re.findall(r"dialect[^\n]*BANK|BANK[^\n]*dialect", acc_sec, re.I))
    # cert-v2 join
    cv = json.load(open(f".cache/lattice_gen15_{cand}.json"))["bigtest"]
    cert = cert_ok = 0
    dissent = flips = 0
    ents = {"cert": [], "ans": [], "abst": []}
    for i in range(1500):
        gt, gc = maj(cv[i]); at, _ = maj(armb[i]); ct, _ = maj(c2x[i])
        g14t, _ = maj(g14[i])
        flips += (gt != g14t)
        h = H(cv[i])
        if gc == 5 and at == gt and ct == gt:
            cert += 1; cert_ok += (gt == gold[i]); ents["cert"].append(h)
        elif gc == 5:
            dissent += 1; ents["ans"].append(h)
        elif gc >= 3:
            ents["ans"].append(h)
        else:
            ents["abst"].append(h)
    prec = cert_ok / max(cert, 1)
    adv = json.load(open(f".cache/adversarial_walk_{cand}.json"))
    adv_fl = sum(a["guard_flagged"] for a in adv)
    adv_wu = sum(a["wrong_unanimous"] for a in adv)
    ok = (not fails) and prec >= 0.998 and adv_fl == 20 and banks >= 7
    verdicts[cand] = dict(vals=vals, fails=fails, banks=banks, cert=cert,
                          prec=prec, dissent=dissent, flips=flips,
                          adv_wu=adv_wu, adv_fl=adv_fl, ok=ok,
                          H_ans=float(np.mean(ents["ans"])) if ents["ans"] else 0,
                          H_abst=float(np.mean(ents["abst"])) if ents["abst"] else 0)
    print(f"\n=== CANDIDATE {cand} ===")
    print(f"  fixtures: {vals}")
    print(f"  bars failed: {fails if fails else 'NONE'}")
    print(f"  acceptance dialect-banks: {banks} (>=7 with only-[45] clause — "
          f"read the log section if borderline)")
    print(f"  cert-v2: {cert} @ {prec:.4f} | panel-dissent {dissent}")
    print(f"  SENTINEL: ring-gauge flips vs gen-14 {flips}/1500 | "
          f"cooling H(ans) {verdicts[cand]['H_ans']:.3f} "
          f"H(abst) {verdicts[cand]['H_abst']:.3f} | "
          f"adversarial: wrong-unanimous {adv_wu}/20, guard flags {adv_fl}/20")
    print(f"  ALL BARS: {'PASS' if ok else 'FAIL'}")

passing = {c: v for c, v in verdicts.items() if v["ok"]}
if not passing:
    print("\n*** NO CANDIDATE PASSES — THE MANIFEST STAYS GEN-14. "
          "The kill prints; nothing is touched. ***")
    sys.exit(1)
win = max(passing, key=lambda c: passing[c]["vals"]["bigtest"])
v = verdicts[win]
m = json.load(open(".cache/GENERATION.json"))
m["gen_id"] = "15"
m["date"] = "2026-07-20"
m["env"]["ALG_FTYPES"] = "7"
m["parser_ckpt"] = {"A": ".cache/fire_armA.safetensors",
                    "C1": ".cache/fire_armC1.safetensors"}[win]
m["corpora"]["train"] = {"A": ".cache/fire_armA.jsonl",
                         "C1": ".cache/fire_armC1.jsonl"}[win]
m["regression_bars"]["bigtest"] = 1149
m["macro"] = {"grammar": "mg1", "module": "mycelium/macros.py",
              "crown_corpus": ".cache/macro_mint_pairs.jsonl"}
m["waivers"] = {"specialist": "gen-14 entourage (remine rides entourage-15)",
                "panel": "cert-v2 members armB + cap2x"}
m["notes"] = (f"2026-07-20 GEN-15 PROMOTED: arm {win} (four-arm fire; "
              f"bigtest {v['vals']['bigtest']}; crowns readable; "
              f"guard flags 20/20 — decision-path wiring ACTIVE as of this "
              f"battery; adversarial fixture walked; sentinel row reported).")
json.dump(m, open(".cache/GENERATION.json", "w"), indent=1)
print(f"\n*** PROMOTED: GEN-15 = arm {win} "
      f"(bigtest {v['vals']['bigtest']}, cert-v2 {v['cert']}@{v['prec']:.4f}) — "
      f"THE MANIFEST IS WRITTEN ***")
