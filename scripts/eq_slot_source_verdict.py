"""eq_slot_source_verdict.py -- the pinned bars for the two eq_slot gap reads (ledger 2026-10-27 12:4x). Reads
.cache/eq_slot_src_{ARM}_STU_s{0,1}.json (eq_slot_source_read.py); zero training. Verdicts on BREATH (seed mean); VANILLA reported.
  S1  GAP = val GSM8K SEEN-problem final exact - holdout final exact. >= 0.15 THE GAP IS SEEN-PROBLEM (then RECALL EXPLAINS IT iff
      the sibling LOOKUP >= seen - 0.05, else BEYOND RECALL); < 0.08 NO SEEN ADVANTAGE; else BETWEEN
  S2  holdout rows whose block-1 decode != final decode and whose two permutations differ in train frequency: share where the
      final permutation is the MORE frequent in training. >= 0.65 DRIFT TO PRIOR; <= 0.50 NO DRIFT; else BETWEEN"""
import json, os

import numpy as np

OUT = ".cache/eq_slot_source_verdict.txt"
if __name__ == "__main__":
    assert not os.path.exists(OUT), "refusing to overwrite a banked read"
    out = open(OUT, "w")

    def say(s):
        print(s, flush=True); out.write(s + "\n")
    J = {(a, s): json.load(open(f".cache/eq_slot_src_{a}_STU_s{s}.json")) for a in ("BREATH", "VANILLA") for s in (0, 1)
         if os.path.exists(f".cache/eq_slot_src_{a}_STU_s{s}.json")}
    assert ("BREATH", 0) in J and ("BREATH", 1) in J, "both BREATH seeds required"
    for (a, s), j in sorted(J.items()):
        for c, v in j["cells"].items():
            say(f"{a:7s} s{s} {c:22s} n {v['n']:4d} | exact block1 {v['exact'][0]:.3f} final {v['exact'][-1]:.3f}"
                + (f" | LOOKUP {v['lookup']:.3f} (n {v['lookup_n']})" if v["lookup"] is not None else ""))
    say("VERDICTS (eq_slot gap reads)")
    for a in ("BREATH", "VANILLA"):
        js = [J[a, s] for s in (0, 1) if (a, s) in J]
        if not js: continue
        seen = np.mean([j["cells"]["val gsm8k seen"]["exact"][-1] for j in js]); hold = np.mean([j["cells"]["holdout"]["exact"][-1] for j in js])
        look = js[0]["cells"]["val gsm8k seen"]["lookup"]; gap = seen - hold
        v = (("THE GAP IS SEEN-PROBLEM / " + ("RECALL EXPLAINS IT" if look >= seen - 0.05 else "BEYOND RECALL")) if gap >= 0.15
             else "NO SEEN ADVANTAGE" if gap < 0.08 else "BETWEEN")
        say(f"S1 {a}{'' if a == 'BREATH' else ' (read)'}: {v} (seen {seen:.3f} - holdout {hold:.3f} = {gap:+.3f}; LOOKUP {look:.3f}; seeds {len(js)})")
        d = [r for j in js for r in j["drift"]]; ch = [r for r in d if r["changed"] and r["f1"] != r["ff"]]
        sh = float(np.mean([r["ff"] > r["f1"] for r in ch])) if ch else float("nan")
        v2 = "DRIFT TO PRIOR" if sh >= 0.65 else "NO DRIFT" if sh <= 0.50 else "BETWEEN"
        tr = {k: sum((r["ok1"], r["okf"]) == k for r in d if r["changed"]) for k in ((True, False), (False, True), (False, False))}
        say(f"S2 {a}{'' if a == 'BREATH' else ' (read)'}: {v2} (final more frequent in {sh:.3f} of {len(ch)} changed rows; changed {sum(r['changed'] for r in d)}/{len(d)}; "
            f"right->wrong {tr[True, False]}, wrong->right {tr[False, True]}, wrong->wrong {tr[False, False]}; identity block1 "
            f"{np.mean([r['id1'] for r in d]):.3f} -> final {np.mean([r['idf'] for r in d]):.3f})")
