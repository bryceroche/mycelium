"""flux_continuity_audit.py — GUT #21, READ (a): THE CONTINUITY AUDIT
(2026-07-16). The chain of custody as flow: every fixture item must exit
through EXACTLY ONE labeled surface — certify / answer / vote-abstain
(repair-lane intake) — with intake = outflow, zero double-counts, zero
vanishings. Upstream armor for the paper's public tables (912/1,500 at
1.0000; 1,195 one-shot): the same arithmetic, recomputed item-by-item
from the banked member votes, by an independent walker.

Zero GPU. Artifacts: .cache/lattice_{gate,armB,cap2x}.json (gen-14
freeze members, 5 views x 1,500 items) + the fixture's gold.
"""
import json
from collections import Counter

gate = json.load(open(".cache/lattice_gate.json"))
armb = json.load(open(".cache/lattice_armB.json"))
c2x = json.load(open(".cache/lattice_cap2x.json"))
rows = [json.loads(l) for l in open(".cache/algebra_nl_bigtest.jsonl")]
gold = [r["solution"][r["query_var"]] for r in rows]
n = len(gold)
assert n == len(gate["bigtest"]) == len(armb["bigtest"]) == len(c2x["bigtest"]) == 1500
print(f"[continuity] intake: {n} items | members: gate={gate['ckpt']}")


def maj(votes):
    vs = [v for v in votes if v is not None]
    if not vs:
        return None, 0
    t, c = Counter(vs).most_common(1)[0]
    return t, c


# --- assign exactly one exit per item (the paper's decision path) ---
exits = []                     # (channel, majority_answer, correct)
for i in range(n):
    gt, gc = maj(gate["bigtest"][i])
    at, _ = maj(armb["bigtest"][i])
    ct, _ = maj(c2x["bigtest"][i])
    if gc == 5 and at == gt and ct == gt:
        ch = "certify"
    elif gc == 5:
        ch = "answer(panel-dissent)"   # unanimous but a sibling dissents
    elif gc >= 3:
        ch = "answer(majority)"
    else:
        ch = "vote-abstain->repair"    # the repair lane's intake surface
    exits.append((ch, gt, gt == gold[i] if gt is not None else False))

# --- conservation: sum of surfaces == intake, one exit per item ---
counts = Counter(ch for ch, _, _ in exits)
total = sum(counts.values())
print(f"\n=== SURFACES (each item exits once by construction; sum must be {n}) ===")
for ch, c in counts.most_common():
    ok = sum(1 for e in exits if e[0] == ch and e[2])
    print(f"  {ch:26s} {c:5d}  correct {ok:5d}  precision {ok/max(c,1):.4f}")
print(f"  {'TOTAL':26s} {total:5d}  {'BALANCED' if total == n else f'LEAK: {n-total}'}")
assert total == n

# --- the public dials, recomputed ---
cert = counts["certify"]
cert_ok = sum(1 for e in exits if e[0] == "certify" and e[2])
gate5 = cert + counts.get("answer(panel-dissent)", 0)
oneshot = None  # one-shot = unpermuted straight parse; view 0 is the straight view
v0_ok = sum(1 for i in range(n)
            if gate["bigtest"][i][0] is not None and gate["bigtest"][i][0] == gold[i])
maj_ok = sum(1 for e in exits if e[1] is not None and e[2])
print(f"\n=== PUBLIC DIALS, INDEPENDENTLY RECOMPUTED ===")
print(f"  cert-v2 certified: {cert}/{n} ({cert/n:.1%})  precision {cert_ok}/{cert} = {cert_ok/cert:.4f}")
print(f"  gate 5/5 unanimous (pre-panel): {gate5}")
print(f"  straight-view (one-shot) correct: {v0_ok}/{n}")
print(f"  majority-vote correct (any channel): {maj_ok}/{n}")

# --- pressure point: does any certified item carry a wrong answer? ---
wrong_cert = [(i, e[1], gold[i]) for i, e in enumerate(exits)
              if e[0] == "certify" and not e[2]]
print(f"\n  certified-wrong items: {len(wrong_cert)}"
      + (f"  {wrong_cert[:5]}" if wrong_cert else "  (none — 1.0000 stands)"))

# --- pressure point: the repair lane's intake, enumerated for its exit ledger ---
ra = [i for i, e in enumerate(exits) if e[0] == "vote-abstain->repair"]
print(f"  repair-lane intake (vote-abstain): {len(ra)} items -> {ra[:12]}{'...' if len(ra) > 12 else ''}")
print("  (their exits live in the specialist's ledger — the audit hands"
      " this list to entourage-14's remine as its conservation check)")

json.dump({"n": n, "surfaces": dict(counts), "cert": cert, "cert_ok": cert_ok,
           "gate5": gate5, "oneshot_v0": v0_ok, "repair_intake": ra},
          open(".cache/flux_continuity_audit.json", "w"))
print("\n[continuity] banked -> .cache/flux_continuity_audit.json")
