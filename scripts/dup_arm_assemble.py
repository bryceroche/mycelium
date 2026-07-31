"""dup_arm_assemble.py ARM — one arm's mix + states assembly + gold +
sentinels (2026-07-31; the disk law: one arm resident at a time — the
first launch's ENOSPC lesson). Usage: dup_arm_assemble.py dry_d05"""
import sys, os, json
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_HW", "512"); os.environ.setdefault("ALG_DUP", "1")
import numpy as np
from phase1_algebra_head import T_ALG, sent_indices, TOKENIZER_JSON, build_gold
import phase1_algebra_head as PH
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer

ARM = sys.argv[1]
WET = ARM.startswith("wet")
SHARE = {"d02": 0.02, "d05": 0.05, "d12": 0.12}[ARM.split("_")[1]]
BASE_N = 82400
tok = Tokenizer.from_file(TOKENIZER_JSON)
pool = [json.loads(l) for l in open(".cache/dup_pool.jsonl")]
wet = [json.loads(l) for l in open(".cache/book8_wet_block.jsonl")]
pool_states = np.load(".cache/dupfire_pool_states.npy", mmap_mode="r")
wet_states = np.load(".cache/dupfire_wet_states.npy", mmap_mode="r")
base_states = np.load(".cache/phase1_alg_states_g22_states.npy", mmap_mode="r")
base_lines = open(".cache/gen22_mix.jsonl").read().splitlines()
assert base_states.shape[0] == len(base_lines) == BASE_N

bn = BASE_N + (len(wet) * 10 if WET else 0)
uniq = int(round(SHARE / (1.0 - SHARE) * bn / 10))
block_rows, block_srcs = [], []
if WET:
    for i, r in enumerate(wet):
        for _ in range(10):
            block_rows.append(r); block_srcs.append(("wet", i))
for i in range(uniq):
    for _ in range(10):
        block_rows.append(pool[i]); block_srcs.append(("pool", i))
n_tot = BASE_N + len(block_rows)
mixp = f".cache/dupfire_{ARM}_mix.jsonl"
npyp = f".cache/phase1_alg_states_g23{ARM}_states.npy"
npzp = f".cache/phase1_alg_states_g23{ARM}.npz"
with open(mixp, "w") as f:
    f.write("\n".join(base_lines) + "\n")
    for r in block_rows: f.write(json.dumps(r) + "\n")
print(f"[{ARM}] mix {n_tot} rows (uniq {uniq}, share {SHARE:.0%}, wet={WET})")
out = np.lib.format.open_memmap(npyp, mode="w+", dtype=np.float16,
                                shape=(n_tot, T_ALG, base_states.shape[-1]))
CH = 4096
for s0 in range(0, BASE_N, CH):
    out[s0:min(s0+CH, BASE_N)] = base_states[s0:min(s0+CH, BASE_N)]
for j, (src, i) in enumerate(block_srcs):
    out[BASE_N + j] = (wet_states if src == "wet" else pool_states)[i]
out.flush(); del out
samples, ids, mask, offsets = PH.tokenize(mixp)
gold = build_gold(samples, offsets)
sent = np.stack([sent_indices(s["text"], o, mask[i])
                 for i, (s, o) in enumerate(zip(samples, offsets))])
np.savez(npzp, tokmask=mask.astype(np.uint8), sent=sent.astype(np.int8),
         **{f"g_{k}": v for k, v in gold.items()})
print(f"[{ARM}] states assembled + gold built")
rows = [json.loads(l) for l in open(mixp)]
st = np.load(npyp, mmap_mode="r")
picks = [0, BASE_N - 1, BASE_N, n_tot - 1, BASE_N + len(block_rows)//2, 40000]
ids2 = np.zeros((8, T_ALG), np.int32); msk = np.zeros((8, T_ALG), np.float32)
for i, ridx in enumerate(picks):
    e = tok.encode(rows[ridx]["text"]); Ln = min(len(e.ids), T_ALG)
    ids2[i, :Ln] = e.ids[:Ln]; msk[i, :Ln] = 1.0
live = recompute_states(ids2).astype(np.float32)
for i, ridx in enumerate(picks):
    m_ = msk[i] > 0
    a = live[i][m_]; b = np.asarray(st[ridx], np.float32)[m_]
    cos = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cos > 0.9999, f"SENTINEL FAIL {ARM} row {ridx} cos {cos}"
print(f"[{ARM}] sentinels 6/6 — assembly TRUSTED")
