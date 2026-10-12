"""dn_read_a.py -- READ A: THE CORRECTION RATE (ledger 2026-10-12 12:35, denoising-breath reads).

Zero-GPU, banked artifacts only. Wild: .cache/rawslots_breaths_wild_PMS8_241.pkl (the FIXED-tap
per-breath dump, 2026-10-06 22:11 regeneration) x gold .cache/phase1_alg_states_wildhold.npz's g_*
arrays. Diet: the valid2/diet per-breath dump does NOT exist (only final-breath dumps
.cache/rawslots_slicevalid2_PMS8_241.pkl / .cache/dirtb_rawslots_diet_PMS8_241.pkl survive) and its
gold-slot states npz (.cache/phase1_alg_states_pm35cslicevalid2.npz) was deleted in the 2026-10-11
disk cleanup ("retired forms' states deleted") -- regenerating needs a fresh frozen-trunk L0-L3
CPU precompute, which is a new precompute pass, not a read of a banked artifact, so this script
reports wild only and says so explicitly (the task's own "else say so" clause).

Output: .cache/dnread_read_a_PMS8_241.json + .txt
"""
import os
import sys
import json
import pickle
import collections

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/picker")
import numpy as np
from dn_common import load_gold, score_row_slots, index_breaths

RAW_WILD = ".cache/rawslots_breaths_wild_PMS8_241.pkl"
GOLD_WILD = ".cache/phase1_alg_states_wildhold.npz"
OUT_JSON = ".cache/dnread_read_a_PMS8_241.json"
OUT_TXT = ".cache/dnread_read_a_PMS8_241.txt"


def transition(a, b):
    """a,b: parallel bool arrays (same slots, earlier vs later breath). Returns RR,RW,WR,WW counts."""
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    rr = int((a & b).sum()); rw = int((a & ~b).sum()); wr = int((~a & b).sum()); ww = int((~a & ~b).sum())
    return dict(RR=rr, RW=rw, WR=wr, WW=ww, n=len(a),
                correction_rate=(wr / (wr + ww)) if (wr + ww) else float("nan"),
                regression_rate=(rw / (rw + rr)) if (rw + rr) else float("nan"))


def main():
    print("[dn-read-a] loading wild per-breath dump + gold...", flush=True)
    rb = pickle.load(open(RAW_WILD, "rb"))
    kbs = sorted(rb.keys())
    kb0, kbF = kbs[0], kbs[-1]
    idx = index_breaths(rb)
    vg = load_gold(GOLD_WILD)
    rows = sorted(set(idx[kb0]) & set(idx[kbF]))
    print(f"[dn-read-a] K_B breaths {kbs}, breath0={kb0} breathF={kbF}, rows {len(rows)}", flush=True)

    overall0, overallF = [], []
    row_overall0, row_overallF = [], []
    fam = collections.defaultdict(lambda: ([], []))   # family -> (breath0 bools, breathF bools)
    per_row_families = collections.defaultdict(list)  # error family tag -> list of (row, slot)

    for i in rows:
        s0 = score_row_slots(idx[kb0][i], vg, i)
        sF = score_row_slots(idx[kbF][i], vg, i)
        js = sorted(set(s0) & set(sF))
        if not js:
            continue
        row_ok0 = all(s0[j]["ok"] for j in js)
        row_okF = all(sF[j]["ok"] for j in js)
        row_overall0.append(row_ok0); row_overallF.append(row_okF)
        for j in js:
            d0, dF = s0[j], sF[j]
            overall0.append(d0["ok"]); overallF.append(dF["ok"])
            fam["ftype"][0].append(d0["f_ftype"]); fam["ftype"][1].append(dF["f_ftype"])
            fam["res"][0].append(d0["f_res"]); fam["res"][1].append(dF["f_res"])
            if d0["is_rel"]:
                fam["args"][0].append(d0["f_args"]); fam["args"][1].append(dF["f_args"])
                fam["op"][0].append(d0["f_op"]); fam["op"][1].append(dF["f_op"])
            else:
                fam["digits"][0].append(d0["f_dig"]); fam["digits"][1].append(dF["f_dig"])
            if (not d0["ok"]) and dF["ok"]:
                per_row_families["corrected"].append((i, j, "rel" if d0["is_rel"] else "nonrel"))

    result = {
        "tag": "PMS8_241", "fixture": "wildhold", "breath0": kb0, "breathF": kbF,
        "n_rows": len(rows),
        "overall_slot": transition(overall0, overallF),
        "overall_row": transition(row_overall0, row_overallF),
        "by_family": {k: transition(v[0], v[1]) for k, v in fam.items()},
        "first_look_acc_slot": float(np.mean(overall0)),
        "final_acc_slot": float(np.mean(overallF)),
        "first_look_acc_row": float(np.mean(row_overall0)),
        "final_acc_row": float(np.mean(row_overallF)),
        "n_corrected_examples": per_row_families["corrected"][:20],
        "diet": {"available": False,
                 "why": ("no per-breath dump exists for the valid2/diet slice (only final-breath "
                         "dumps rawslots_slicevalid2_PMS8_241.pkl / dirtb_rawslots_diet_PMS8_241.pkl "
                         "survive) and the diet's gold-slot states npz "
                         "(.cache/phase1_alg_states_pm35cslicevalid2.npz) was deleted in the "
                         "2026-10-11 disk cleanup ('retired forms states deleted'); regenerating "
                         "needs a fresh frozen-trunk CPU precompute pass, not a banked-artifact read "
                         "-- out of scope for this zero-GPU read, reported honestly rather than "
                         "fabricated")},
    }
    json.dump(result, open(OUT_JSON, "w"), indent=1)

    lines = []
    lines.append(f"READ A -- THE CORRECTION RATE (PMS8_241, wildhold, breath {kb0} -> breath {kbF}, n_rows={len(rows)})")
    lines.append("")
    o = result["overall_slot"]
    lines.append(f"OVERALL per-slot transition (n={o['n']}): RR={o['RR']} RW={o['RW']} WR={o['WR']} WW={o['WW']}")
    lines.append(f"  correction rate (wrong@0 -> right@F) = {o['correction_rate']:.4f}   [PREDICTION: ~=0]")
    lines.append(f"  regression rate (right@0 -> wrong@F) = {o['regression_rate']:.4f}")
    lines.append(f"  first-look acc (breath {kb0}) = {result['first_look_acc_slot']:.4f}  | final acc (breath {kbF}) = {result['final_acc_slot']:.4f}")
    r = result["overall_row"]
    lines.append("")
    lines.append(f"PER-ROW (all gold slots correct) transition (n={r['n']}): RR={r['RR']} RW={r['RW']} WR={r['WR']} WW={r['WW']}")
    lines.append(f"  row correction rate = {r['correction_rate']:.4f}  | row first-look acc = {result['first_look_acc_row']:.4f} | row final acc = {result['final_acc_row']:.4f}")
    lines.append("")
    lines.append("BY FAMILY:")
    for k, v in result["by_family"].items():
        lines.append(f"  {k:8s} n={v['n']:5d}  RR={v['RR']:4d} RW={v['RW']:4d} WR={v['WR']:4d} WW={v['WW']:4d}  "
                      f"correction={v['correction_rate']:.4f}  regression={v['regression_rate']:.4f}")
    lines.append("")
    lines.append("DIET: " + ("available" if result["diet"]["available"] else "NOT AVAILABLE -- " + result["diet"]["why"]))
    txt = "\n".join(lines)
    open(OUT_TXT, "w").write(txt + "\n")
    print(txt)
    print(f"\n[dn-read-a] wrote {OUT_JSON} and {OUT_TXT}", flush=True)


if __name__ == "__main__":
    main()
