"""license_table_v2.py — TRANCHE-4 AUDITION (2026-08-06; the slope
probe's pen tranche; the pen proposes, verify_entry licenses)."""
import sys, json
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from mycelium.aug_table import SEED_ENTRIES, verify_entry, TRANCHE3, TRANCHE4
rng = np.random.RandomState(82000)
table, refused = [], []
ALL = [(c,t,f,None,False) for c,t,f in SEED_ENTRIES] + list(TRANCHE3) + list(TRANCHE4)
for cons, tid, fmt, constraint, latex in ALL:
    ok, fails = verify_entry(cons, fmt, rng, constraint=constraint, latex=latex)
    (table if ok else refused).append({"construction": cons, "id": tid, "fmt": fmt,
                                       "fails": [f[0] for f in fails[:2]]})
    print(f"[{cons:6s}] {tid:13s} {'LICENSED' if ok else 'REFUSED ' + str(fails[:2])}", flush=True)
json.dump({"licensed": table, "refused": refused, "version": "v5-tranche4-slope-probe",
           "note": "tranche 4: the slope probe's 120-entry substrate; axis variety by design"},
          open('.cache/aug_table_v2.json','w'), indent=1)
print(f"[table v5] licensed {len(table)}/{len(ALL)}  refused {len(refused)}")
