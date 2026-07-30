"""gen22_verdict.py — THE ONLY PEN, gen-22 (2026-07-30). Reads the
collected battery for the diet-fire candidate, checks every bar, prints
the sentinel row, and either writes GENERATION.json (gen-22) or refuses.
BARS (per the word: regression bars per the manifest; band deltas per
the bar-noise law, same margins as gen-21's charter −10/−6):
  PROMOTE: bigtest >= 1223 AND band-min >= 1213; alg4test >= 380,
           band-min >= 374; standing floors (alg2 560, vtest 598,
           dagtest 660, dag7b 500, dag8 500); h3held >= 170;
           adupheld >= 180; cert-v2 precision >= 0.998; adversarial
           guard 20/20; acceptance >= 7 dialect banks.
  KILL: bigtest < 1213 -> the manifest stays gen-21, negative banked.
HASH RECOMPUTE IS MECHANICAL (the deep-clean repair): PROMOTED and a
true manifest are one atomic act. Entourage duties (specialist remine,
centroids, mouth recal, panel re-audition) ride BEFORE the gate reads
anything, per standing discipline."""
import json, re, sys, os, hashlib
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from collections import Counter

GEN = "22"
BARS = [("bigtest", 1223), ("alg4test", 380), ("alg2test", 560),
        ("vtest", 598), ("dagtest", 660), ("dag7btest", 500),
        ("dag8test", 500), ("h3held", 170), ("adupheld", 180)]
BAND_MIN = {"bigtest": 1213, "alg4test": 374}
KILL_FLOOR = 1213
from gate_fixtures import BIGTEST   # two-home fix: same authority as the battery
rows = [json.loads(l) for l in open(BIGTEST)]
gold = [r["solution"][r["query_var"]] for r in rows]
_m = json.load(open(".cache/GENERATION.json"))
_lat = _m["panel"]["lattices"]
assert set(_lat.keys()) == set(_m["panel"]["members"].keys()), \
    "APPARATUS INVARIANT VIOLATED: panel-in-code != panel-in-manifest"
_lf = sorted(_lat.values())
# freshness assert on exam inputs — the wake >= the head
_ck = os.path.getmtime(".cache/g22.safetensors")
for _st in ("3000", "3500"):
    _sp = f".cache/g22_seg4_s{_st}.safetensors"
    assert os.path.getmtime(_sp) >= _ck - 7200, \
        f"APPARATUS: stale band snapshot {_sp} (older than the candidate)"
print(f"[panel] members from manifest: {sorted(_lat.keys())} -> {_lf}")
armb = json.load(open(_lf[0]))["bigtest"]
c2x = json.load(open(_lf[1]))["bigtest"]
g16 = json.load(open(".cache/lattice_gen16_V4.json"))["bigtest"]


def maj(v):
    vs = [x for x in v if x is not None]
    return Counter(vs).most_common(1)[0] if vs else (None, 0)


def H(votes):
    c = Counter("⊥" if v is None else v for v in votes)
    p_ = np.array(list(c.values()), float) / len(votes)
    return float(-(p_ * np.log(p_)).sum())


verdicts = {}
for cand in ("H",):
    log = open(f".cache/gen{GEN}_{cand}.log").read()

    def answer_of(row):
        m = re.search(rf"--- {row} ---.*?TOTAL: \d+/\d+ graph-solve, (\d+)/\d+ ANSWER",
                      log, re.S)
        return int(m.group(1)) if m else -1

    vals = {name: answer_of(name) for name, _ in BARS}
    fails = [(n, v, b) for (n, b), v in zip(BARS, vals.values()) if v < b]
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
    print(f"  ZONE DELTA vs the gen-20 baseline (998/410/92): "
          f"{zu-998:+d}/{zp-410:+d}/{zd-92:+d} (read against gen-21's own "
          f"promoted column at the bench; the mint diet predicts near-zero "
          f"disturbance — the diet is in-register, not organic prose)")
    sums = [vals["bigtest"] + vals["alg4test"] + vals["h3held"]]
    print(f"  RECALIBRATION SHEET (prints first by rule): headline sum {sums[0]}"
          f" | bands {bands}")
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
    print(f"\n=== CANDIDATE {cand} (g22, the diet fire) ===")
    print(f"  fixtures: {vals}")
    print(f"  bands (headline+seg4_s3000+seg4_s3500): {bands}")
    print(f"  bars failed: {fails if fails else 'NONE'}")
    print(f"  acceptance dialect-banks: {banks} (>=7)")
    print(f"  cert-v2: {cert} @ {prec:.4f} | panel-dissent {dissent}")
    print(f"  SENTINEL: ring-gauge flips vs gen-16 {flips}/1500 | "
          f"cooling H(ans) {verdicts[cand]['H_ans']:.3f} "
          f"H(abst) {verdicts[cand]['H_abst']:.3f} | "
          f"adversarial: wrong-unanimous {adv_wu}/20, guard flags {adv_fl}/20")
    print(f"  ALL BARS: {'PASS' if ok else 'FAIL'}")

if all(v["vals"]["bigtest"] < KILL_FLOOR for v in verdicts.values()):
    print(f"\n*** KILL: bigtest < {KILL_FLOOR} — THE MANIFEST STAYS GEN-21. "
          "The diet line returns to the docket re-priced (the cure printed "
          "at the fixture; the regression cost is the finding). ***")
    sys.exit(1)
passing = {c: v for c, v in verdicts.items() if v["ok"]}
if not passing:
    print("\n*** BARS FAILED — THE MANIFEST STAYS GEN-21. The kill prints; "
          "nothing is touched. ***")
    sys.exit(1)
win = "H"
v = verdicts[win]
m = json.load(open(".cache/GENERATION.json"))
m["gen_id"] = "22"
m["date"] = "2026-07-30"
m["parser_ckpt"] = ".cache/g22.safetensors"
m["corpora"]["train"] = ".cache/gen22_mix.jsonl"
m["regression_bars"]["bigtest"] = max(1223, v["vals"]["bigtest"])
m["waivers"] = {"specialist": "gen-21 entourage until the gen-22 remine lands "
                              "(entourage duty rides BEFORE the gate reads anything)",
                "panel": "cert-v2 members per panel.lattices (re-audition rides entourage-22)"}
m["notes"] = (f"2026-07-30 GEN-22 PROMOTED: g22 (the fdiv-on-derived diet fire, "
              f"400x10 @ 4.85% on gen-21's register, SGDR 4x4k warm-from-g21; "
              f"bigtest {v['vals']['bigtest']}; d1 fixture 1/8->7/8 with the "
              f"sel-basin attractor 0.53->0.00 GONE; DEPLOYED wild forty "
              f"40/40 mouth-refused unchanged; the template hole closed).")
# HASH RECOMPUTE (mechanical since the deep clean): PROMOTED and a true
# manifest are one atomic act.
def _h16(path): return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
m["hashes"] = {"parser": _h16(m["parser_ckpt"]),
               "specialist": _h16(m["specialist_ckpt"]),
               "centroids": _h16(m["monitor_centroids"]),
               "mouth": _h16(m["mouth"]),
               "train": _h16(m["corpora"]["train"])}
json.dump(m, open(".cache/GENERATION.json", "w"), indent=1)
print(f"\n*** PROMOTED: GEN-22 = g22 "
      f"(bigtest {v['vals']['bigtest']}, cert-v2 {v['cert']}@{v['prec']:.4f}, "
      f"the diet fire) — THE MANIFEST IS WRITTEN, HASHES RECOMPUTED. "
      f"ENTOURAGE DUTY NOW OWED: specialist remine, centroids, mouth recal, "
      f"panel re-audition — before the gate reads anything. ***")
