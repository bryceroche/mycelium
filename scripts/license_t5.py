import sys, json
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from mycelium.aug_table import SEED_ENTRIES, verify_entry, TRANCHE3, TRANCHE4, TRANCHE5
rng = np.random.RandomState(82000)
table, refused = [], []
ALL = [(c,t,f,None,False) for c,t,f in SEED_ENTRIES] + list(TRANCHE3) + list(TRANCHE4) + list(TRANCHE5)
for cons, tid, fmt, constraint, latex in ALL:
    ok, fails = verify_entry(cons, fmt, rng, constraint=constraint, latex=latex)
    (table if ok else refused).append({"construction": cons, "id": tid, "fmt": fmt, "fails": [f[0] for f in fails[:2]]})
json.dump({"licensed": table, "refused": refused, "version": "v6-tranche5-dup-hunt",
           "note": "dup 11->19 for the threshold hunt; rest = v120"},
          open('.cache/aug_table_v3.json','w'), indent=1)
print(f"licensed {len(table)}/{len(ALL)} refused {len(refused)} | dup:",
      sum(1 for e in table if e['construction']=='dup'), flush=True)
