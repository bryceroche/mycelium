import sys, json
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from mycelium.aug_table import SEED_ENTRIES, verify_entry
rng = np.random.RandomState(82000)
table, refused = [], []
from mycelium.aug_table import TRANCHE3
ALL = [(c,t,f,None,False) for c,t,f in SEED_ENTRIES] + list(TRANCHE3)
for cons, tid, fmt, constraint, latex in ALL:
    ok, fails = verify_entry(cons, fmt, rng, constraint=constraint, latex=latex)
    (table if ok else refused).append({"construction": cons, "id": tid, "fmt": fmt,
                                       "fails": [f[0] for f in fails[:2]]})
    print(f"[{cons:5s}] {tid:11s} {'LICENSED' if ok else 'REFUSED ' + str(fails[:2])}", flush=True)
json.dump({"licensed": table, "refused": refused, "version": "v3-latex-wordfam-subadd",
           "note": "tranche 2: symbolic class; refusals = dialect-edge findings; recursion guard holds"},
          open('.cache/aug_table_v1.json','w'), indent=1)
print(f"[table v2] licensed {len(table)}/{len(ALL)}  refused {len(refused)}")
