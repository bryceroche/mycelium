"""erasure_share_correlation.py — THE LANDAUER CHECK (2026-07-11,
zero-GPU). Registered prediction: per-register erosion gen-6 -> gen-7b
tracks INVERSE mix-share of fresh rehearsal — the least-rehearsed
registers pay the erasure bill first. Provenance recovered exactly by
matching mixed7b texts against their source corpora; erosion read from
the banked eval tables (ANSWER column).
"""
import json

SOURCES = {
    "nl-core":  [".cache/algebra_nl_train.jsonl"],
    "alg2":     [".cache/algebra2_nl_train.jsonl"],
    "alg3":     [".cache/algebra3_nl_train.jsonl"],
    "alg4":     [".cache/algebra4_nl_train.jsonl"],
    "verbose":  [".cache/algv_train_verbose.jsonl"],
    "dag6":     [".cache/dag_train.jsonl"],
    "dag7":     [".cache/dag7_train.jsonl", ".cache/dag7b_train.jsonl"],
}
text2reg = {}
for reg, paths in SOURCES.items():
    for path in paths:
        try:
            for l in open(path):
                text2reg[json.loads(l)["text"]] = reg
        except FileNotFoundError:
            print(f"  [warn] missing source {path}")

mix = {}
unknown = 0
for l in open(".cache/algebra_mixed7b_train.jsonl"):
    reg = text2reg.get(json.loads(l)["text"])
    if reg is None:
        unknown += 1
    else:
        mix[reg] = mix.get(reg, 0) + 1
total = sum(mix.values()) + unknown
print(f"=== MIXED7B COMPOSITION (n={total}, unknown {unknown}) ===")
for reg, c in sorted(mix.items(), key=lambda kv: -kv[1]):
    print(f"  {reg:8s}: {c:6d}  ({c/total:.1%})")

# Erosion: gen-6 -> gen-7b ANSWER deltas from the banked eval tables.
# (gen-6 row / gen-7b row, both measured this session, same fixtures.)
EROSION = {
    # register     eval row      gen6   gen7b  fresh rows since gen-6
    "nl-core":  ("bigtest",      1000,  901),
    "alg2":     ("alg2test",      551,  575),
    "alg4":     ("alg4test",      371,  319),
    "verbose":  ("vtest",         600,  600),
    "dag6":     ("dagtest",       563,  670),
}
print("\n=== EROSION vs FRESH-REHEARSAL SHARE (the Landauer table) ===")
rows = []
for reg, (row, g6, g7b) in EROSION.items():
    share = mix.get(reg, 0) / total
    # fresh rehearsal = rows aligned with this register ADDED since gen-6
    # (dag7/dag7b rows rehearse dag-style composition + fdiv, not the
    #  tranche registers) — dag6's own rows persist AND dag7 rehearses it.
    fresh = {"dag6": mix.get("dag7", 0) + mix.get("dag6", 0)}.get(reg, 0)
    if reg == "alg4":
        fresh = 0     # pct/seq: zero new rows (fdiv rows use dag register)
    delta = (g7b - g6) / g6
    rows.append((reg, row, share, fresh / total, delta))
    print(f"  {reg:8s} ({row:8s}): mix-share {share:5.1%} | fresh-share "
          f"{fresh/total:5.1%} | erosion {delta:+.1%}")

# directional check: rank correlation between fresh-share and delta
import math
def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    for pos, i in enumerate(order):
        r[i] = pos
    return r
fr = rank([r[3] for r in rows]); dr = rank([r[4] for r in rows])
n = len(rows)
num = sum((fr[i] - (n-1)/2) * (dr[i] - (n-1)/2) for i in range(n))
den = math.sqrt(sum((fr[i] - (n-1)/2)**2 for i in range(n)) *
                sum((dr[i] - (n-1)/2)**2 for i in range(n)))
rho = num / den if den else float("nan")
print(f"\n  Spearman(fresh-rehearsal share, erosion delta): rho={rho:+.2f} (n={n})")
print("  PREDICTION: rho strongly positive (least-rehearsed pays first).")
