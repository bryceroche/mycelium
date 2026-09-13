"""THE PAIRED READ (2026-09-12): compare two per-slot outcome files written by
loop_val (LV_PER_SLOT=...) on the same checkpoint and rows — e.g. open vs
ALG_IDLE_BREATHS=3,4,5. Reports the aggregate difference, the discordant slots
(b = right only under A, c = right only under B), McNemar's z, and the paired
SE of the difference (the aggregate hides all of this).
usage: paired_read.py A.npz B.npz [labelA] [labelB]"""
import sys, math, numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2]); la = sys.argv[3] if len(sys.argv) > 3 else "A"; lb = sys.argv[4] if len(sys.argv) > 4 else "B"
ka = list(zip(a["rows"].tolist(), a["slots"].tolist())); kb = list(zip(b["rows"].tolist(), b["slots"].tolist()))
assert ka == kb, "the two reads cover different slots"
oa, ob = a["ok"].astype(bool), b["ok"].astype(bool); n = len(oa)
only_a = int((oa & ~ob).sum()); only_b = int((~oa & ob).sum()); both = int((oa & ob).sum())
d = oa.mean() - ob.mean(); se_paired = math.sqrt((only_a + only_b) / n - ((only_a - only_b) / n) ** 2) / math.sqrt(n)
z = (only_a - only_b) / math.sqrt(only_a + only_b) if only_a + only_b else 0.0
print(f"[paired-read] n={n} slots | {la} {oa.mean():.4f} | {lb} {ob.mean():.4f} | diff {d:+.4f} | discordant: right only under {la} = {only_a}, only under {lb} = {only_b}, both {both} | paired SE {se_paired:.4f} | McNemar z {z:+.2f}")
