"""adoption_read.py — GUT #26 FOLLOW-UP 4 (2026-07-17): THE ADOPTION READ.
The battery speaks first: no repair lane takes the word until the COMPOSITE
table is stated with every lane's emissions decomposed and the bar pinned.

PINNED ADOPTION BAR (before any number prints): the new composite
precision over ALL emitted answers must be >= the CURRENT composite's —
certified 912 (912 right) + panel-dissent 56 (56) + majority 212 (211):
1179/1180 = 0.99915. A lane tier that drops the composite below this
fails adoption AT THAT TIER; the decision table reports every count
threshold so the word can be given at any bar Bryce prefers (benchmark
scoring may price recall differently — that is a POLICY line, stated,
not smuggled). Cert non-contact is structural: tiers only touch the 320
vote-abstain items; the certify/dissent/majority channels are read from
banked artifacts and cannot move.

Part A (GPU-minor): the incumbent lane's emission decomposition on the
320 (stage0 / withhold-2 / specialist — emitted answer vs gold vs silent),
which follow-up 1 graded for recovery but never for precision.
Part B (zero-GPU): composite decision table — lattice tiers by
plurality-count threshold, incumbent lane, and incumbent-then-lattice
composition, each with end-to-end answered and composite precision.
"""
import json, os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("ALG2", "1")
os.environ.setdefault("ALG_FTYPES", "6")
os.environ.setdefault("ALG_DUP", "1")
import numpy as np
from collections import Counter

from phase1_algebra_head import T_ALG, L_FAC, build_params, decode
from phase1_algebra_nack import N_FIELDS, build_cond_params
from nack_incumbent_read import (run, slot_conf, withheld, forced, fixture,
                                 rows, gold, p_plat, p_re, c_re, c_zero)

# NOTE: importing nack_incumbent_read re-runs its pipeline (flat script) —
# by design: same artifacts, and we need its loaded params + helpers. Its
# banked JSON re-banks identically.

BASE_RIGHT, BASE_EMIT = 1179, 1180        # certified 912 + dissent 56 + majority 211/212
BAR = BASE_RIGHT / BASE_EMIT
print(f"\n[adoption] pinned composite bar: {BASE_RIGHT}/{BASE_EMIT} = {BAR:.5f}")

# ---- Part A: incumbent lane EMISSIONS on the 320 ----
zf = (np.zeros((len(fixture), T_ALG), np.float32),
      np.zeros((len(fixture), 1), np.float32),
      np.zeros((len(fixture), L_FAC, N_FIELDS), np.float32))
blank = run(p_plat, c_zero, fixture, *zf)
inc = {}
survivors, surv_wh = [], {}
for i in fixture:
    o = blank[i]
    facs, q = decode(o)
    a = forced(facs, q, i)
    if a is not None:
        inc[i] = a                          # stage0 emits
        continue
    sub, wh_j = withheld(o, facs, 2)
    a = forced(sub, q, i)
    if a is not None:
        inc[i] = a                          # stage1 emits
        continue
    survivors.append(i)
    surv_wh[i] = wh_j

ffld_s = np.zeros((len(survivors), L_FAC, N_FIELDS), np.float32)
for r_, i in enumerate(survivors):
    for j in surv_wh[i]:
        ffld_s[r_, j, :] = 1.0
re = run(p_re, c_re, survivors,
         np.zeros((len(survivors), T_ALG), np.float32),
         np.ones((len(survivors), 1), np.float32), ffld_s)
for i in survivors:
    o = re[i]
    facs, q = decode(o)
    a = forced(facs, q, i)
    if a is None:
        sub, _ = withheld(o, facs, 2)
        a = forced(sub, q, i)
    if a is not None:
        inc[i] = a                          # specialist emits

inc_emit = len(inc)
inc_right = sum(1 for i, a in inc.items() if a == gold[i])
print(f"[A] INCUMBENT lane emissions: {inc_emit} (right {inc_right}, "
      f"wrong {inc_emit-inc_right}, silent {len(fixture)-inc_emit})  "
      f"lane precision {inc_right/max(inc_emit,1):.4f}")

# ---- Part B: the decision table ----
ex = json.load(open(".cache/residue_portrait.json"))["per_item_exits"]


def lattice_tier(cmin):
    emit = {int(i): v for i, v in ex.items()
            if v["exit"] != "abstain" and v["plurality"] >= cmin}
    right = sum(1 for i, v in emit.items() if v["exit"] == "right")
    return emit, right


def composite(extra_emit, extra_right):
    e, r = BASE_EMIT + extra_emit, BASE_RIGHT + extra_right
    return r, e, r / e


print(f"\n=== THE DECISION TABLE (composite bar {BAR:.5f}; base answered "
      f"{BASE_RIGHT}/1500 = {BASE_RIGHT/1500:.1%}) ===")
print(f"  {'lane / tier':34s} {'+emit':>6s} {'+right':>7s} {'answered':>9s} "
      f"{'precision':>10s}  bar")
table = {}
for cmin in (4, 5, 6, 7, 8, 10):
    emit, right = lattice_tier(cmin)
    r, e, p = composite(len(emit), right)
    ok = p >= BAR
    table[f"lattice_count>={cmin}"] = {"emit": len(emit), "right": right,
                                       "answered": r, "precision": p, "passes": ok}
    print(f"  lattice count>={cmin:2d}                   {len(emit):6d} {right:7d} "
          f"{r:5d} ({r/1500:.1%}) {p:10.5f}  {'PASS' if ok else 'fail'}")

r, e, p = composite(inc_emit, inc_right)
print(f"  incumbent (specialist stack)       {inc_emit:6d} {inc_right:7d} "
      f"{r:5d} ({r/1500:.1%}) {p:10.5f}  {'PASS' if p >= BAR else 'fail'}")
table["incumbent"] = {"emit": inc_emit, "right": inc_right, "answered": r,
                      "precision": p, "passes": bool(p >= BAR)}

# composition: lattice tier first (higher-precision voice), incumbent silent
for cmin in (5, 8):
    emit, right = lattice_tier(cmin)
    add_e, add_r = len(emit), right
    for i, a in inc.items():
        if i not in emit:
            add_e += 1
            add_r += (a == gold[i])
    r, e, p = composite(add_e, add_r)
    ok = p >= BAR
    table[f"lattice>={cmin}_then_incumbent"] = {"emit": add_e, "right": add_r,
                                                "answered": r, "precision": p,
                                                "passes": ok}
    print(f"  lattice>={cmin} then incumbent        {add_e:6d} {add_r:7d} "
          f"{r:5d} ({r/1500:.1%}) {p:10.5f}  {'PASS' if ok else 'fail'}")

passing = {k: v for k, v in table.items() if v["passes"]}
best = max(passing.items(), key=lambda kv: kv[1]["answered"]) if passing else None
print(f"\n[adoption] best bar-passing option: "
      f"{best[0] if best else 'NONE'} -> answered {best[1]['answered']}/1500 "
      f"({best[1]['answered']/1500:.1%}) at {best[1]['precision']:.5f}"
      if best else "[adoption] no option passes the strict bar")
print("[adoption] POLICY LINE, stated not smuggled: benchmark scoring has no "
      "wrong-answer penalty; sub-bar tiers trade published precision for "
      "recall — that trade is Bryce's word, with this table as its price sheet.")
json.dump({"bar": BAR, "incumbent_lane": table["incumbent"], "table": table},
          open(".cache/adoption_read.json", "w"))
print("[adoption] banked -> .cache/adoption_read.json")
