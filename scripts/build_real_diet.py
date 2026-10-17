"""build_real_diet.py -- THE REAL-ONLY DIET (2026-10-17; Bryce: "keep GSM8K and discard our mint problems"; the two-arm
run's arm A). From form_mix_pm35c (the SRK3 lineage's diet, original positional frame):
  DROP   every LETTERED row (a letter list 'a, b, c' in the text = a minted or pen-dialect algebra row): the mint.
  DROP   every row whose text is a WILD fixture text (the r7 handshakes leak, ledger 10-16 18:20) -- fixture custody.
  KEEP   the rest: GSM8K (verbatim + number-swapped), ASDiv, SVAMP, the book / r7 / prose-v0 MATH-harvest rows.
Out: .cache/form_mix_real.jsonl + .cache/form_mix_real.idx.npy (row indices into form_mix_pm35c, for the state slice)."""
import collections, json, re
import numpy as np

SRC = ".cache/form_mix_pm35c.jsonl"
LL = re.compile(r"\b[a-z], [a-z](, [a-z])*\b")
norm = lambda s: re.sub(r"\s+", " ", s).strip()
wild = {norm(json.loads(l)["text"]) for l in open(".cache/wild_admitted_holdout.jsonl")}


def src(r):
    g = r.get("gen")
    if not isinstance(g, dict):
        return f"untagged:{g}"
    return g.get("src") or ("book" if "book" in g else g.get("shape") or "mint-dict")


rows = [json.loads(l) for l in open(SRC)]
keep, C = [], collections.Counter()
for i, r in enumerate(rows):
    if LL.search(r["text"]):
        C["drop_lettered"] += 1; continue
    if norm(r["text"]) in wild:
        C["drop_wild_text"] += 1; print(f"  struck (wild text): row {i} {src(r)} {r['text'][:80]!r}"); continue
    keep.append(i); C["keep:" + str(src(r)) + (":resampled" if (r.get("gen") or {}).get("resample") else "")] += 1
with open(".cache/form_mix_real.jsonl", "w") as fh:
    for i in keep:
        fh.write(json.dumps(rows[i]) + "\n")
np.save(".cache/form_mix_real.idx.npy", np.array(keep, dtype=np.int64))
print(f"form_mix_pm35c {len(rows)} -> form_mix_real {len(keep)}")
for k, v in sorted(C.items()):
    print(f"  {k:32s} {v}")
