"""BT8 DIAG (ledger 2026-11-04; zero-training, banked weights): WHICH factor did the factored silhouette lose?
Scores BT_LOAD with the schedule commit (the FINAL read's own loop, BR.score) and, per real gold step, splits the committed type into
its factors: op / kind a / kind b, and the joint. Run once per checkpoint (BT8 with BT_FACT=1, BT4M8 without) -> .cache/gsmb_bt8diag_{TAG}.json
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import gsm_breath_reader as BR

assert BR.LOAD and os.path.exists(BR.LOAD)
torch.set_num_threads(int(os.environ.get("TOY_THREADS", "4")))
Dt, states = BR.TR.load(); tree = BR.TR.load_tree()
if BR.ROLE: BR.RF = np.load(BR.GD.V1.C(BR.rolef_name()), mmap_mode="r")
net = BR.make_net(); net.load_state_dict(torch.load(BR.LOAD)); net.eval()
CAP = []
_gold = BR.gold


def gold_tap(E):   # BR.score calls gold(E) after the last cycle; the committed types live in its frame -> read them off the caller
    out = _gold(E)
    f = sys._getframe(1)
    if "typ" in f.f_locals and "progs" in f.f_locals and f.f_code.co_name == "score":
        typ, gt = f.f_locals["typ"], out[1]
        for b, e in enumerate(E):
            for k in range(e["m"]): CAP.append((int(typ[b, k]), int(gt[b, k])))
    return out


BR.gold = gold_tap
TAG = BR.TAG; out = dict(tag=TAG)
F = np.array([(BR.GD.OPS.index(o), BR.KINDS.index(a), BR.KINDS.index(b)) for o, a, b in BR.TYPES] + [(-1, -1, -1)])
for split in ("ORIG", "TEST"):
    CAP.clear(); BR.score(net, Dt[split], states, tree)
    p, g = np.array(CAP).T; fp, fg = F[p], F[g]
    ok = fp == fg
    out[split] = dict(n=len(p), op=float(ok[:, 0].mean()), ka=float(ok[:, 1].mean()), kb=float(ok[:, 2].mean()), joint=float((p == g).mean()),
                      ka_given_op=float(ok[ok[:, 0], 1].mean()), kb_given_op=float(ok[ok[:, 0], 2].mean()))
    o = out[split]
    print(f"[diag] {TAG} {split} n {o['n']} op {o['op']:.4f} | kind a {o['ka']:.4f} (given op right {o['ka_given_op']:.4f}) | "
          f"kind b {o['kb']:.4f} (given op right {o['kb_given_op']:.4f}) | joint {o['joint']:.4f}", flush=True)
json.dump(out, open(f".cache/gsmb_bt8diag_{TAG}.json", "w"), indent=1)
