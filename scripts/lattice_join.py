"""lattice_join.py — the lattice probe's CPU join (2026-07-13).
Reads the three members' vote files; computes the pinned reads:
(a) cert-v2 precision/coverage vs gate-only 5/5 (+ the coverage gap),
(b) the disagreement autopsy (which axis breaks false certificates),
(c) the deep-wrong read (cross-examination on stable-wrongs).
"""
import json
from collections import Counter

gate = json.load(open(".cache/lattice_gate.json"))
armb = json.load(open(".cache/lattice_armB.json"))
c2x = json.load(open(".cache/lattice_cap2x.json"))
rows = [json.loads(l) for l in open(".cache/algebra_nl_bigtest.jsonl")]
gold = [r["solution"][r["query_var"]] for r in rows]

def maj(votes):
    vs = [v for v in votes if v is not None]
    if not vs:
        return None, 0
    t, c = Counter(vs).most_common(1)[0]
    return t, c

n = len(gold)
gate5 = cert2 = gate5_ok = cert2_ok = 0
fc_total = fc_armb_breaks = fc_c2x_breaks = fc_both = 0
sw_total = sw_broken = 0
for i in range(n):
    gt, gc = maj(gate["bigtest"][i])
    at, _ = maj(armb["bigtest"][i])
    ct, _ = maj(c2x["bigtest"][i])
    is5 = (gc == 5)
    if is5:
        gate5 += 1
        gate5_ok += (gt == gold[i])
        agree = (at == gt) and (ct == gt)
        if agree:
            cert2 += 1
            cert2_ok += (gt == gold[i])
        if gt != gold[i]:                       # false certificate
            fc_total += 1
            ab = at != gt
            cb = ct != gt
            fc_armb_breaks += ab
            fc_c2x_breaks += cb
            fc_both += (ab and cb)
    if gc >= 3 and gt != gold[i]:               # stable-wrong (deep-wrong class)
        sw_total += 1
        sw_broken += ((at != gt) or (ct != gt))

print("=== THE LATTICE PROBE (bigtest, n=%d) ===" % n)
print(f"  GATE-ONLY 5/5 : coverage {gate5:4d} ({gate5/n:.1%}) precision "
      f"{gate5_ok/max(gate5,1):.4f}")
print(f"  CERT-V2       : coverage {cert2:4d} ({cert2/n:.1%}) precision "
      f"{cert2_ok/max(cert2,1):.4f}")
print(f"  coverage gap (the lineage-disagreement instrument): {gate5-cert2}")
print(f"\n  AUTOPSY — gate false certificates: {fc_total}")
print(f"    broken by armB (lineage axis): {fc_armb_breaks}")
print(f"    broken by cap2x (width axis):  {fc_c2x_breaks}")
print(f"    broken by both:                {fc_both}")
print(f"    survived cert-v2 (uncaught):   {cert2 - cert2_ok}")
print(f"\n  DEEP-WRONG READ — gate stable-wrongs (>=3 consistent, wrong): {sw_total}")
print(f"    broken by cross-examination: {sw_broken} "
      f"({sw_broken/max(sw_total,1):.0%}; fifth-jurisdiction bar: >=1/3)")

# census pool: raw-prose stable-wrongs under the gate, cross-examined
gsw = bsw = 0
for i in range(len(gate["census"])):
    gt, gc = maj(gate["census"][i])
    if gt is not None and gc >= 3:
        at, _ = maj(armb["census"][i])
        ct, _ = maj(c2x["census"][i])
        gsw += 1
        bsw += ((at != gt) or (ct != gt))
print(f"\n  CENSUS POOL — gate stable-vote raw-prose parses: {gsw}; "
      f"cross-examination dissents on {bsw} (no gold here; the mouth's jurisdiction)")
