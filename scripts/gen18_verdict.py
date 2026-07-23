"""gen18_verdict.py (generated; band bars + zone column) — THE ONLY PEN, gen-17 (2026-07-23). Reads the
collected battery for arms F and R, checks every charter-pinned bar,
prints the sentinel row, and either writes GENERATION.json (gen-17) or
refuses. BARS (the charter, registered before the mint):
  PROMOTE: bigtest >= 1223 (the record — no backsliding into the crown)
           AND h3held >= 170/200 (hundreds-given read >= 0.85, from 0.00)
           AND adupheld >= 180/200 (add-dup parse >= 0.90, from 0.00)
           AND alg4test >= 402 AND standing floors (alg2 560, vtest 598,
           dagtest 660, dag7b 500, dag8 500)
           AND cert-v2 precision >= 0.998 AND adversarial guard 20/20
           AND acceptance >= 7 dialect banks.
  KILL: both arms bigtest < 1208 -> keep gen-16, bank the negative.
  ARM CHOICE: higher bigtest; ties to F (restarts must earn the jostle).
"""
import json, re, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from collections import Counter

GEN = "18"
BARS = [("bigtest", 1223), ("alg4test", 402), ("alg2test", 560),
        ("vtest", 598), ("dagtest", 660), ("dag7btest", 500),
        ("dag8test", 500), ("h3held", 170), ("adupheld", 180)]
BAND_MIN = {"bigtest": 1213, "alg4test": 396}   # bar-noise law: record - floor
KILL_FLOOR = 1208
rows = [json.loads(l) for l in open(".cache/algebra_nl_bigtest.jsonl")]
gold = [r["solution"][r["query_var"]] for r in rows]
armb = json.load(open(".cache/lattice_armB.json"))["bigtest"]
c2x = json.load(open(".cache/lattice_cap2x.json"))["bigtest"]
g16 = json.load(open(".cache/lattice_gen16_V4.json"))["bigtest"]


def maj(v):
    vs = [x for x in v if x is not None]
    return Counter(vs).most_common(1)[0] if vs else (None, 0)


def H(votes):
    c = Counter("⊥" if v is None else v for v in votes)
    p_ = np.array(list(c.values()), float) / len(votes)
    return float(-(p_ * np.log(p_)).sum())


verdicts = {}
for cand in ("A", "B"):
    log = open(f".cache/gen{GEN}_{cand}.log").read()

    def answer_of(row):
        m = re.search(rf"--- {row} ---.*?TOTAL: \d+/\d+ graph-solve, (\d+)/\d+ ANSWER",
                      log, re.S)
        return int(m.group(1)) if m else -1

    vals = {name: answer_of(name) for name, _ in BARS}
    fails = [(n, v, b) for (n, b), v in zip(BARS, vals.values()) if v < b]
    # band check (bar-noise law): headline + 2 annealed snapshots all >= band-min
    bands = {}
    for fx, bmin in BAND_MIN.items():
        pts = [vals[fx]] + [answer_of(f"band_{st}_{fx}") for st in ("3000", "3500")]
        bands[fx] = pts
        if min(pts) < bmin:
            fails.append((f"{fx}-band", min(pts), bmin))
    acc_sec = log.split("--- acceptance ---")[-1]
    banks = len(re.findall(r"dialect[^\n]*BANK|BANK[^\n]*dialect", acc_sec, re.I))
    cv = json.load(open(f".cache/lattice_gen{GEN}_{cand}.json"))["bigtest"]
    zu = zp = zd = 0
    for i in range(1500):
        vs = [x for x in cv[i] if x is not None]
        t_, c_ = maj(cv[i])
        if c_ == 5 and t_ == gold[i]:
            zu += 1
        elif gold[i] in vs:
            zp += 1
        else:
            zd += 1
    print(f"  ZONE COLUMN: umbra {zu} / penumbra {zp} / dark {zd}")
    cert = cert_ok = 0
    dissent = flips = 0
    ents = {"cert": [], "ans": [], "abst": []}
    for i in range(1500):
        gt, gc = maj(cv[i]); at, _ = maj(armb[i]); ct, _ = maj(c2x[i])
        g16t, _ = maj(g16[i])
        flips += (gt != g16t)
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
    print(f"\n=== ARM {cand} ===")
    print(f"  fixtures: {vals}")
    print(f"  bands (headline+s3000+s3500): {bands}")
    # ZONE COLUMN (gut #56): masses from the member votes

    print(f"  bars failed: {fails if fails else 'NONE'}")
    print(f"  acceptance dialect-banks: {banks} (>=7)")
    print(f"  cert-v2: {cert} @ {prec:.4f} | panel-dissent {dissent}")
    print(f"  SENTINEL: ring-gauge flips vs gen-16 {flips}/1500 | "
          f"cooling H(ans) {verdicts[cand]['H_ans']:.3f} "
          f"H(abst) {verdicts[cand]['H_abst']:.3f} | "
          f"adversarial: wrong-unanimous {adv_wu}/20, guard flags {adv_fl}/20")
    print(f"  ALL BARS: {'PASS' if ok else 'FAIL'}")

if all(v["vals"]["bigtest"] < KILL_FLOOR for v in verdicts.values()):
    print(f"\n*** KILL: both arms bigtest < {KILL_FLOOR} — THE MANIFEST STAYS "
          "GEN-16. The diet lines return to the docket re-priced. ***")
    sys.exit(1)
passing = {c: v for c, v in verdicts.items() if v["ok"]}
if not passing:
    print("\n*** NO ARM PASSES ALL BARS — THE MANIFEST STAYS GEN-16. "
          "The kill prints; nothing is touched. ***")
    sys.exit(1)
win = max(passing, key=lambda c: (passing[c]["vals"]["bigtest"],
                                  1 if c == "A" else 0))
v = verdicts[win]
m = json.load(open(".cache/GENERATION.json"))
m["gen_id"] = "17"
m["date"] = "2026-07-23"
m["parser_ckpt"] = f".cache/g18_arm{win}.safetensors"
m["corpora"]["train"] = ".cache/gen18_mix.jsonl"
m["macro"] = {"grammar": "mg2", "module": "mycelium/macros.py",
              "crown_corpus": ".cache/gen17_crowns.jsonl"}
m["regression_bars"]["bigtest"] = 1223
m["waivers"] = {"specialist": "gen-16 entourage (remine rides entourage-17)",
                "panel": "cert-v2 members armB + cap2x"}
m["notes"] = (f"2026-07-23 GEN-18 PROMOTED: g18_arm{win} "
              f"({'no-ration control' if win == 'A' else 'hot-phase ration'}; "
              f"bigtest {v['vals']['bigtest']}; hundreds-held "
              f"{v['vals']['h3held']}/200 from 0; add-dup-held "
              f"{v['vals']['adupheld']}/200 from 0; the diet-wall and census "
              f"holes closed by audited mint lines; zener-convened fire).")
json.dump(m, open(".cache/GENERATION.json", "w"), indent=1)
print(f"\n*** PROMOTED: GEN-17 = g17_arm{win} "
      f"(bigtest {v['vals']['bigtest']}, h3held {v['vals']['h3held']}/200, "
      f"adup {v['vals']['adupheld']}/200, cert-v2 {v['cert']}@{v['prec']:.4f}) — "
      f"THE MANIFEST IS WRITTEN ***")
