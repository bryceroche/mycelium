"""build_numeral_table.py — THE NUMERAL TABLE (2026-09-15): the mean frozen-
trunk state of each numeral token 0..999 over the diet (the lexicon probe's
by_v means, saved). The lexicon's substitution road writes a matched
word-span's token state as its numeral's mean state, so the head reads
"a dozen" the way it reads "12". CPU; the precomputed memmap."""
import json, re, sys, numpy as np
from tokenizers import Tokenizer
N_ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
rng = np.random.default_rng(0)
rows = [json.loads(l) for l in open(".cache/form_mix12_aligned.jsonl")]
S = np.load(".cache/phase1_alg_states_form12al_states.npy", mmap_mode="r"); T = 256
tok = Tokenizer.from_file(".cache/llama-3.2-1b-weights/tokenizer.json")
acc = np.zeros((1000, 2048), np.float64); cnt = np.zeros(1000, np.int64)
for i in sorted(rng.choice(len(rows), N_ROWS, replace=False).tolist()):
    e = tok.encode(rows[i]["text"]); ids = e.ids[:T]; st = None
    for t_i, tid in enumerate(ids):
        s_ = tok.decode([tid]).strip()
        if s_.isdigit() and len(s_) <= 3:
            v = int(s_); st = S[i] if st is None else st
            acc[v] += st[t_i]; cnt[v] += 1
mean = np.where(cnt[:, None] > 0, acc / np.maximum(cnt[:, None], 1), 0).astype(np.float16)
np.savez(".cache/numeral_table.npz", mean=mean, count=cnt, note="mean L0-L3 trunk state per numeral token 0..999 over the diet")
print(f"[numeral-table] values with >= 5 tokens: {(cnt >= 5).sum()}; 0..100 covered {(cnt[:101] >= 5).sum()}/101; 101..999 covered {(cnt[101:] >= 5).sum()}/899; total tokens {cnt.sum()}")
