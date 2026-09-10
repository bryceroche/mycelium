"""refusal_grade.py — THE STEERING WHEEL, read W1 part 2 (CPU): refusal
visibility and core localization on the FINAL parses. Per row: solve the
parse completely; on a certified refusal, the minimal unsatisfiable core.
Reports P(refuse | wrong row), P(refuse | right row), core size, and the
core's PRECISION (fraction of core slots that are actually wrong) and
RECALL (fraction of the row's wrong slots the core names). BARS (ledger
2026-09-10): live if P(refuse | wrong) >= 0.15 with median core <= 3."""
import sys, json, collections, statistics, time
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
from alternator_bridge import refuse_and_core
rows = [json.loads(l) for l in open(sys.argv[1])]
st = collections.Counter(); cores = []; prec = []; rec = []; t0 = time.time()
n_wrong = n_right = ref_wrong = ref_right = 0
for r in rows:
    wrong = bool(r["wrong_slots"]) or bool(r["extra_slots"])
    res = refuse_and_core(r["n_vars"], r["parse"], r["m"])
    st[res["status"]] += 1
    refused = res["status"] == "unsat"
    if wrong: n_wrong += 1; ref_wrong += refused
    else: n_right += 1; ref_right += refused
    if refused:
        core_slots = {r["parse"][k]["_slot"] for k in res["core"]}
        bad = set(r["wrong_slots"]) | set(r["extra_slots"])
        cores.append(len(core_slots))
        prec.append(len(core_slots & bad) / max(len(core_slots), 1))
        rec.append(len(core_slots & bad) / max(len(bad), 1) if bad else 0.0)
pw = ref_wrong / max(n_wrong, 1); pr = ref_right / max(n_right, 1)
print(f"[refusal-grade] rows={len(rows)} statuses={dict(st)} ({time.time()-t0:.0f}s)")
print(f"[refusal-grade] wrong rows={n_wrong} refused={ref_wrong} P(refuse|wrong)={pw:.3f} (bar >= 0.15) | right rows={n_right} refused={ref_right} P(refuse|right)={pr:.3f}")
if cores:
    print(f"[refusal-grade] core size median={statistics.median(cores):.0f} mean={statistics.mean(cores):.2f} (bar median <= 3) | core precision mean={statistics.mean(prec):.3f} recall mean={statistics.mean(rec):.3f}")
live = pw >= 0.15 and cores and statistics.median(cores) <= 3
print(f"[refusal-grade] W1 {'LIVE' if live else 'FAIL'} — {'the wheel has something to steer with on this fixture' if live else 'the wheel has too little to steer with here'}")
