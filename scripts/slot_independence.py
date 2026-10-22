"""slot_independence.py -- THE INDEPENDENCE CHECK (2026-10-21, zero GPU; ledger same day). Do a body's wild slot
failures fall independently across a row (whole-row accuracy = the product of per-slot accuracies) or cluster in hard
rows? Uses the dump's gold ftype per slot and ps_legal's per-slot verdict. -> .cache/slot_independence.txt"""
import collections, os, pickle, sys
import numpy as np
OUT = ".cache/slot_independence.txt"
assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
L = []
for tag in os.environ.get("SI_TAGS", "CLK3_241,SRK3_241,K3_242").split(","):
    z = np.load(f".cache/ps_legal_wild_{tag}.npz")
    ok = {(int(r), int(s)): bool(o) for r, s, o in zip(z["rows"], z["slots"], z["ok"])}
    D = pickle.load(open(f".cache/dump_wild_{tag}.pkl", "rb"))
    ft = {(t[0], t[1]): t[2] for t in D}
    rows = collections.defaultdict(list)
    for (r, s), o in ok.items():
        rows[r].append((ft[(r, s)], o))
    acc_ft = {f: np.mean([o for v in rows.values() for g, o in v if g == f]) for f in {g for v in rows.values() for g, _ in v}}
    p = np.mean([o for v in rows.values() for _, o in v])
    obs = np.mean([all(o for _, o in v) for v in rows.values()])
    pred_ft = np.mean([np.prod([acc_ft[g] for g, _ in v]) for v in rows.values()])
    pred_g = np.mean([p ** len(v) for v in rows.values()])
    ns = np.array([len(v) for v in rows.values()])
    # by row length: observed vs predicted
    byn = collections.defaultdict(lambda: [0, 0.0, 0])
    for v in rows.values():
        b = min(len(v), 9); byn[b][0] += all(o for _, o in v); byn[b][1] += np.prod([acc_ft[g] for g, _ in v]); byn[b][2] += 1
    rf = obs / pred_ft
    verdict = "INDEPENDENT" if 0.67 <= rf <= 1.5 else ("CLUSTERED" if rf > 1.5 else "ANTI-CLUSTERED")
    L.append(f"== {tag}: rows {len(rows)}, slots {sum(ns)} (per row mean {ns.mean():.2f}), per-slot acc {p:.3f}; per-ftype acc "
             + ", ".join(f"{g}:{a:.3f}" for g, a in sorted(acc_ft.items())))
    L.append(f"   OBSERVED all-slots-right {obs:.4f} ({int(round(obs * len(rows)))}/{len(rows)}) | PREDICTED per-ftype product {pred_ft:.4f}, "
             f"global p^n {pred_g:.4f} | ratio obs/pred {rf:.2f} -> {verdict}")
    L.append("   by slots per row (n rows: observed / predicted): " + "  ".join(
        f"{b}{'+' if b == 9 else ''}: {byn[b][2]} {byn[b][0]}/{byn[b][1]:.1f}" for b in sorted(byn)))
print("\n".join(L)); open(OUT, "w").write("\n".join(L) + "\n")
