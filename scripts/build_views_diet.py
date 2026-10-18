"""build_views_diet.py -- VIEWS, NOT REPS (2026-10-18; Bryce's cross-join ruling, word given for the queue).
form_mix_real carries 8,094 EXACT REPEATS (a text's 2nd..kth occurrence; 1,165 GSM8K problems sit there 6x). This builds
the diet that spends the same rows on DIFFERENT VIEWS: the o-th repeat of a text (o = 1..6) is replaced by photobooth view
v=o of that same row (PB_WIDE=1, the 10-18 audited photobooth: listed names only, hyphen-aware). A row with no swappable
name keeps its exact copy (counted: the declared residue). Rows, reps and card are UNCHANGED vs form_mix_real.
  src      -> .cache/form_real_VNsrc.jsonl + .idx.npy (the repeat rows, their form_mix_real index; the views' source split)
              .cache/form_real_VNkeep.idx.npy (the first occurrences, kept as themselves)
              .cache/form_real_VNsrc.occ.npy (o per repeat row)
  compose  (after photobooth.py form_real_VNsrc.jsonl .cache/form_real_VN 6) -> .cache/form_real_VNviews.jsonl (row k = view
              v=occ[k] of src row k; row-aligned to the source split) + .cache/form_mix_realVN.jsonl (keep rows, then views)
usage: build_views_diet.py src|compose"""
import collections, json, sys
import numpy as np

REAL = ".cache/form_mix_real.jsonl"
mode = sys.argv[1]
rows = [json.loads(l) for l in open(REAL)]
if mode == "src":
    seen, keep, rep, occ = collections.Counter(), [], [], []
    for i, r in enumerate(rows):
        o = seen[r["text"]]; seen[r["text"]] += 1
        (keep if o == 0 else rep).append(i)
        if o:
            occ.append(o)
    assert max(occ) <= 6, f"a text repeats {max(occ) + 1}x; the photobooth makes 6 views"
    with open(".cache/form_real_VNsrc.jsonl", "w") as fh:
        for i in rep:
            fh.write(json.dumps(rows[i]) + "\n")
    np.save(".cache/form_real_VNsrc.idx.npy", np.array(rep, np.int64)); np.save(".cache/form_real_VNkeep.idx.npy", np.array(keep, np.int64))
    np.save(".cache/form_real_VNsrc.occ.npy", np.array(occ, np.int64))
    print(f"[views-diet] {len(rows)} rows = {len(keep)} first occurrences + {len(rep)} repeats (o: {sorted(collections.Counter(occ).items())})")
elif mode == "compose":
    src = [json.loads(l) for l in open(".cache/form_real_VNsrc.jsonl")]
    occ = np.load(".cache/form_real_VNsrc.occ.npy"); keep = np.load(".cache/form_real_VNkeep.idx.npy")
    V = {v: [json.loads(l) for l in open(f".cache/form_real_VN_v{v}.jsonl")] for v in range(1, 7)}
    out, C, texts = [], collections.Counter(), collections.defaultdict(list)
    for k, (r, o) in enumerate(zip(src, occ)):
        w = V[int(o)][k]
        assert (w.get("gen") or {}).get("photobooth", {}).get("source_text", r["text"]) == r["text"] or w["text"] == r["text"], f"view row {k} is not of src row {k}"
        C["view_changed" if w["text"] != r["text"] else "copy_kept_no_name"] += 1
        out.append(w)
    for r, w in zip(src, out):
        if w["text"] != r["text"]:
            texts[r["text"]].append(w["text"])
    dup = sum(len(L) - len(set(L)) for L in texts.values())
    with open(".cache/form_real_VNviews.jsonl", "w") as fh:
        for w in out:
            fh.write(json.dumps(w) + "\n")
    with open(".cache/form_mix_realVN.jsonl", "w") as fh:
        for i in keep:
            fh.write(json.dumps(rows[i]) + "\n")
        for w in out:
            fh.write(json.dumps(w) + "\n")
    n_txt = len({rows[i]["text"] for i in keep} | {w["text"] for w in out})
    print(f"[views-diet] repeats {len(src)}: {dict(C)} | distinct texts {len({rows[i]['text'] for i in range(len(rows))})} -> {n_txt} "
          f"| within-problem collisions (two changed views of one problem identical) {dup}")
    print(f"[views-diet] form_mix_realVN.jsonl {len(keep) + len(out)} rows (= form_mix_real {len(rows)})")
    assert len(keep) + len(out) == len(rows)
