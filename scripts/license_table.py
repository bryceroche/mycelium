import sys, json
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from mycelium.aug_table import SEED_ENTRIES, verify_entry
rng = np.random.RandomState(82000)
table, refused = [], []
for cons, tid, fmt in SEED_ENTRIES:
    ok, fails = verify_entry(cons, fmt, rng)
    (table if ok else refused).append({"construction": cons, "id": tid, "fmt": fmt,
                                       "fails": [f[0] for f in fails[:2]]})
    print(f"[{cons:5s}] {tid:11s} {'LICENSED' if ok else 'REFUSED ' + str(fails[:2])}", flush=True)
json.dump({"licensed": table, "refused": refused, "version": "v2-symbolic-tranche",
           "note": "tranche 2: symbolic class; refusals = dialect-edge findings; recursion guard holds"},
          open('.cache/aug_table_v1.json','w'), indent=1)
print(f"[table v2] licensed {len(table)}/{len(SEED_ENTRIES)}  refused {len(refused)}")
