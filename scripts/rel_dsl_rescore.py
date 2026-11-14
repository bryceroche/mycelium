"""rel_dsl_rescore.py -- (ledger 11-13 16:15) rescore saved checkpoints (.cache/rel_dsl_{TAG}_P.npz) with the tinygrad scorer (the
forward the arm trained with; env RD_ARM / RD_COV / RD_UNIT / RD_SCHED as the run) -> .cache/rel_dsl_{TAG}_tg.json. Usage: tags as args"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import rel_dsl_tg as T
from tinygrad import Tensor

_, shallow, mid, deep = T.RO.load_depth()
for tag in sys.argv[1:]:
    z = np.load(f".cache/rel_dsl_{tag}_P.npz"); P = {k: Tensor(z[k]).contiguous().realize() for k in z.files}
    S = {k: T.tg_score(P, v, T.UNIT) for k, v in (("shallow", shallow), ("mid", mid), ("deep", deep))}
    bd = {d: float(np.mean([o for o, t in zip(S["deep"], deep) if t["depth"] == d])) for d in range(5, 10) if any(t["depth"] == d for t in deep)}
    json.dump(dict(tag=tag, scorer="tinygrad", unit=T.UNIT, sched=T.SCHED, **{k: float(v.mean()) for k, v in S.items()},
                   **{k + "_per": [int(x) for x in v] for k, v in S.items()}, deep_by_depth=bd), open(f".cache/rel_dsl_{tag}_tg.json", "w"))
    print(f"[rescore] {tag}: SHALLOW {S['shallow'].mean():.3f} MID {S['mid'].mean():.3f} DEEP {S['deep'].mean():.3f} | by depth "
          + " ".join(f"d{d} {v:.3f}" for d, v in bd.items()), flush=True)
