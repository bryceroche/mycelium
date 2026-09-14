"""REBUILD A SPLIT'S GOLD NPZ (2026-09-14): the precompute's npz carries the gold
arrays (build_gold at precompute time); a new jsonl with the SAME texts (e.g.
the aligner's spans) shares the states memmap but needs its own npz. This is the
precompute's metadata half — tokenize + build_gold + sent_indices + savez — with
the states untouched. usage: <family env with ALG_TRAIN/ALG_TRAIN_NAME set to the
new file/split> rebuild_gold_npz.py"""
import os, sys, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import phase1_algebra_head as H
split, path = H.TRAIN_NAME, os.environ["ALG_TRAIN"]
assert os.path.exists(H.STATES_NPY.format(split=split)), f"no states memmap for {split} — this tool never computes states"
samples, ids, mask, offsets = H.tokenize(path)
n = len(samples); st = np.load(H.STATES_NPY.format(split=split), mmap_mode="r")
assert st.shape[0] == n, f"states rows {st.shape[0]} != jsonl rows {n}"
gold = H.build_gold(samples, offsets)
sent = np.stack([H.sent_indices(s["text"], o, mask[i]) for i, (s, o) in enumerate(zip(samples, offsets))])
np.savez(H.STATES_NPZ.format(split=split), tokmask=mask.astype(np.uint8), sent=sent.astype(np.int8), **{f"g_{k}": v for k, v in gold.items()})
fs = gold["fspan"]; isg = (gold["ftype"] != 0) & (gold["presence"] > 0.5)
print(f"[rebuild-npz] {split}: {n} rows -> {H.STATES_NPZ.format(split=split)}; given slots with fspan tokens {int(((fs.sum(-1) > 0) & isg).sum())}/{int(isg.sum())}")
