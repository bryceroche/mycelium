"""gsm_recency_ceiling.py -- R-CEIL (ledger 2026-10-30, Bryce "T6 clock to the rescue"): do GSM8K gold programs chain by RECENCY, so a
per-step clock tick (slot k at tick S + k) would address results by a fixed relative rotation? Zero-training, TEST gold + VANILLA_s0
picks. R1 RECENCY (gold slot operands at lag 1 vs a random earlier step), R2 ERROR LOCUS (VANILLA's wrong operands whose gold holds the
lag-1 slot); reads: lag histogram, text order of numeral operands. Writes .cache/gsm_recency_ceiling.txt."""
import collections, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
import gsm_bench_data as GD

if __name__ == "__main__":
    Dt = json.load(open(GD.OUT)); P = json.load(open(GD.V1.C("gsmb_picks_VANILLA_s0.json")))
    lag1, plac, lags, r2, r2kind, order = [], [], collections.Counter(), [], collections.Counter(), []
    for it, pk in zip(Dt["TEST"], P["TEST"]):
        sid, seen = it["sid"], -1
        for k, (_, A, B) in enumerate(it["prog"]):
            for S in (A, B):
                if S["s"]:
                    j = S["s"][0]; lags[min(k - j, 5)] += 1; lag1.append(k - 1 in S["s"]); plac.append(1.0 / k)
                if S["t"]: order.append(sid[S["t"][0]] >= seen)
            seen = max([seen] + [sid[S["t"][0]] for S in (A, B) if S["t"]])
            if k >= len(pk["slots"]): continue
            _, a, b = pk["slots"][k]
            for o, S in ((a, A), (b, B)):
                if not S["s"] or o[1] in S[o[0]]: continue
                r2.append(k - 1 in S["s"])
                if k - 1 in S["s"]: r2kind["picked " + {"t": "a numeral", "s": "another slot", "c": "a constant"}[o[0]]] += 1
    a1, p1, a2 = np.mean(lag1), np.mean(plac), np.mean(r2); n = sum(lags.values())
    L = [f"R1 RECENCY: {a1:.4f} of gold slot operands are lag 1 vs placebo {p1:.4f} (margin {a1 - p1:+.4f}; n {len(lag1)}; bar >= 0.50 and margin >= +0.20)",
         f"R2 ERROR LOCUS: {a2:.4f} of VANILLA's wrong operands with a slot in gold have the lag-1 slot in gold (n {len(r2)}; bar >= 0.50)",
         "read: lag histogram (first member) " + " ".join(f"{('>=' if l == 5 else '')}{l}:{lags[l] / n:.3f}" for l in sorted(lags)),
         f"read: TEXT ORDER {np.mean(order):.4f} of gold numeral operands lie at or after the latest sentence earlier steps used (n {len(order)})",
         "read: at lag-1 gold, VANILLA " + "; ".join(f"{k} {v / max(sum(r2kind.values()), 1):.3f}" for k, v in r2kind.most_common())]
    ok1, ok2 = a1 >= 0.50 and a1 - p1 >= 0.20, a2 >= 0.50
    L.append(f"R1 {'PASS' if ok1 else 'FAIL'} | R2 {'PASS' if ok2 else 'FAIL'} -> R-CEIL {'PASS' if ok1 and ok2 else 'FAIL'}")
    print("\n".join(L)); open(GD.V1.C("gsm_recency_ceiling.txt"), "w").write("\n".join(L) + "\n")
