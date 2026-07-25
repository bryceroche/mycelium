import json, os, subprocess, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
def sh(cmd, extra=None, tail=2):
    env = dict(os.environ); env.update({"DEV":"AMD","ALG2":"1","ALG_FTYPES":"8","ALG_DUP":"1"}); env.update(extra or {})
    r = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True)
    for l in (r.stdout + r.stderr).strip().splitlines()[-tail:]:
        print("   ", l, flush=True)
    if r.returncode != 0:
        raise RuntimeError(f"stage failed: {cmd[:90]}")
print("=== E20 7/8: DISSENT-OVERLAP READ (the owed column) ===", flush=True)
S7 = r'''
import json
from collections import Counter
def maj(v):
    vs=[x for x in v if x is not None]
    return Counter(vs).most_common(1)[0] if vs else (None,0)
armb = json.load(open(".cache/lattice_armB.json"))["bigtest"]
c2x = json.load(open(".cache/lattice_cap2x.json"))["bigtest"]
def dissent_set(gate_votes):
    out = set()
    for i in range(1500):
        gt, gc = maj(gate_votes[i]); at,_ = maj(armb[i]); ct,_ = maj(c2x[i])
        if gc == 5 and not (at == gt and ct == gt):
            out.add(i)
    return out
d14 = dissent_set(json.load(open(".cache/lattice_gen16_V4.json"))["bigtest"])
d15 = dissent_set(json.load(open(".cache/lattice_gen20_G.json"))["bigtest"])
ov = d14 & d15
print(f"[dissent-overlap] gen-16: {len(d14)} | gen-20: {len(d15)} | "
      f"OVERLAP {len(ov)} ({len(ov)/max(len(d15),1):.0%} of gen-16's) — "
      f"{'a STABLE dissent family (structural)' if len(ov) >= 0.5*len(d15) else 'dissent ROTATES (population-driven)'}")
json.dump({"d14": sorted(d14), "d15": sorted(d15), "overlap": sorted(ov)},
          open(".cache/dissent_overlap_20.json", "w"))
'''
open(".cache/_e16_s7.py", "w").write(S7)
sh(".venv/bin/python3 .cache/_e16_s7.py", tail=1)

print("=== E20 8/9: THE COLLAPSE RE-READ under v4 (dashboard accrual) ===", flush=True)
sh(".venv/bin/python3 scripts/collapse_probe.py",
   {"COLLAPSE_CKPT": ".cache/g20.safetensors"}, tail=3)

print("=== E20 9/9: manifest member refresh ===", flush=True)
m = json.load(open(".cache/GENERATION.json"))
m["specialist_ckpt"] = NEW_NACK
m["monitor_centroids"] = ".cache/monitor_centroids_gen20.npz"
m["mouth"] = ".cache/recognition_mouth_gen20.npz"
m["waivers"] = {"panel": "cert-v2 members armB + cap2x (panel-eligible bench, "
                "now incl. crown-readers fire_armC1/B)"}
m["notes"] = (m.get("notes", "") +
              " | 2026-07-21 ENTOURAGE-15 PAID (entourage15.py, the committed "
              "chain's first edit): specialist remined vs gen-16, centroids "
              "(8 kinds incl. macro) + mouth rebuilt on the g20fam family, "
              "dissent-overlap column banked, specialist waiver RETIRED.")
json.dump(m, open(".cache/GENERATION.json", "w"), indent=1)
print("[manifest] entourage-20 complete — members refreshed, waiver retired", flush=True)
print("=== ENTOURAGE-20 DONE ===", flush=True)

# === E20 APPENDED DUTIES (the era's standing instruments) ===
import subprocess, os as _os
print("=== E20 10/12: delta-probe THIRD point (gen-20 centroids) ===", flush=True)
env = dict(_os.environ); env["DELTA_CENTROIDS"] = ".cache/monitor_centroids_gen20.npz"
env["DELTA_OUT"] = ".cache/delta_probe_gen20.json"
subprocess.run(".venv/bin/python3 scripts/delta_probe.py", shell=True, env=env)
print("=== E20 11/12: zone baseline under gen-20 (from the ninth exam's votes) ===", flush=True)
import json as _json
from collections import Counter as _C
cv = _json.load(open(".cache/lattice_gen20_G.json"))["bigtest"]
rows_ = [_json.loads(l) for l in open(".cache/algebra_nl_bigtest.jsonl")]
gold_ = [r["solution"][r["query_var"]] for r in rows_]
zu = zp = zd = 0
for i in range(1500):
    vs = [x for x in cv[i] if x is not None]
    t_, c_ = (_C(vs).most_common(1)[0] if vs else (None, 0))
    if c_ == 5 and t_ == gold_[i]: zu += 1
    elif gold_[i] in vs: zp += 1
    else: zd += 1
_json.dump({"umbra": zu, "penumbra": zp, "dark": zd},
           open(".cache/zone_baseline_gen20.json", "w"))
print(f"[zone-20] umbra {zu} / penumbra {zp} / dark {zd} (the umbra-trend column's second row)", flush=True)
print("=== E20 12/12: discharge walk ===", flush=True)
subprocess.run(".venv/bin/python3 scripts/discharge_check.py", shell=True)
print("=== ENTOURAGE-20 DONE (12 stages) ===", flush=True)
