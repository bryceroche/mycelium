"""eq_slot_source_read.py -- WHERE THE EQ_SLOT GAP LIVES (ledger 2026-10-27 12:4x, word given; zero GPU, zero training). Reads a banked
eq_slot checkpoint (EQ_ARM / EQ_SEED / EQ_READER select it, exactly as the trainer names it) and writes .cache/eq_slot_src_{TAG}.json:
  READ 1 (seen / unseen / source): exact-order per block on the val split cut by source and by whether the val row's PROBLEM
         (gen.src_idx / gen.pool) also has rows in training (205 of 220 GSM8K val rows do: resampled variants of a trained
         problem), + the holdout (unseen GSM8K problems). LOOKUP baseline on seen val rows: predict the most common gold
         permutation among the row's training siblings (same problem, same stated-slot count) -- what pure recall scores.
  READ 2 (drift): on holdout rows whose block-1 decode differs from the final decode, the TRAIN frequency (rows with the same
         stated-slot count) of the block-1 permutation vs the final one, and the identity share at block 1 vs final."""
import collections, hashlib, json, os, sqlite3, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
import eq_slot_breath as EB


def prob_key(g):
    if not isinstance(g, dict): return None
    if "src_idx" in g: return f"{g.get('src')}:{g['src_idx']}"
    if "pool" in g: return f"{g.get('src')}:pool:{json.dumps(g['pool'], sort_keys=True)}"
    return None


def source(g):
    if isinstance(g, dict) and g.get("src") in ("gsm8k", "asdiv", "svamp"): return g["src"]
    return "other"


def logits(net, rows, mm):
    import torch
    out = []
    with torch.no_grad():
        for s0 in range(0, len(rows), 64):
            rs, X, xm, ct, cm, sm, _ = EB.batch(rows, range(s0, min(s0 + 64, len(rows))), mm)
            lg = net(X, xm, ct, cm, sm).numpy()
            out += [(r, lg[i]) for i, r in enumerate(rs)]
    return out


if __name__ == "__main__":
    import torch
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "16")))
    net = EB.make_net(); net.load_state_dict(torch.load(os.environ.get("SR_CKPT", f".cache/eq_slot_{EB.TAG}.pt"))); net.eval()
    gen = [json.loads(l).get("gen") for l in open(EB.EG.MIX["train"])]
    tr, tm_tr = EB.build("train"); ho, tm_ho = EB.build("holdout")
    T_ho, _ = EB.EG.rows("holdout"); ho = [r for r in ho if r["row"] in {t["row"] for t in T_ho}]
    mm_tr = (np.load(EB.NPY.format(EB.SPLIT_STATES["train"]), mmap_mode="r"), tm_tr)
    mm_ho = (np.load(EB.NPY.format(EB.SPLIT_STATES["holdout"]), mmap_mode="r"), tm_ho)
    trn = [r for r in tr if not r["val"]]; val = [r for r in tr if r["val"]]
    sib = collections.defaultdict(collections.Counter); pfreq = collections.defaultdict(collections.Counter)
    for r in trn:
        pfreq[r["n"]][tuple(r["tgt"])] += 1
        k = prob_key(gen[r["row"]])
        if k: sib[(k, r["n"])][tuple(r["tgt"])] += 1
    tprob = {prob_key(gen[r["row"]]) for r in trn} - {None}
    cells = collections.defaultdict(list)   # cell -> [(per-block exact list, lookup-correct or None)]
    for r, lg in logits(net, val, mm_tr):
        g = gen[r["row"]]; k = prob_key(g); seen = k in tprob if k is not None else None
        ex = [EB.decode(lg[b], r["n"]) == r["tgt"] for b in range(EB.NB)]
        lk = (list(sib[(k, r["n"])].most_common(1)[0][0]) == r["tgt"]) if (k, r["n"]) in sib else None
        cells[f"val {source(g)} {'nokey' if seen is None else 'seen' if seen else 'unseen'}"].append((ex, lk))
    drift = []
    for r, lg in logits(net, ho, mm_ho):
        ex = [EB.decode(lg[b], r["n"]) == r["tgt"] for b in range(EB.NB)]
        cells["holdout"].append((ex, None))
        p1, pf = tuple(EB.decode(lg[0], r["n"])), tuple(EB.decode(lg[-1], r["n"]))
        drift.append(dict(row=r["row"], n=r["n"], changed=p1 != pf, ok1=p1 == tuple(r["tgt"]), okf=pf == tuple(r["tgt"]),
                          f1=pfreq[r["n"]][p1], ff=pfreq[r["n"]][pf], id1=p1 == tuple(range(r["n"])), idf=pf == tuple(range(r["n"]))))
    summ = {c: dict(n=len(v), exact=np.mean([e for e, _ in v], 0).tolist(),
                    lookup=(float(np.mean([l for _, l in v if l is not None])) if any(l is not None for _, l in v) else None),
                    lookup_n=sum(l is not None for _, l in v)) for c, v in sorted(cells.items())}
    json.dump(dict(tag=EB.TAG, arm=EB.ARM, cells=summ, drift=drift), open(f".cache/eq_slot_src_{EB.TAG}.json", "w"))
    for c, s in summ.items():
        print(f"[src] {EB.TAG} {c:22s} n {s['n']:4d} | exact block1 {s['exact'][0]:.3f} final {s['exact'][-1]:.3f}"
              + (f" | LOOKUP {s['lookup']:.3f} (n {s['lookup_n']})" if s["lookup"] is not None else ""), flush=True)
