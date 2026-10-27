"""reader_vocab.py -- THE PRUNED READER VOCABULARY (ledger 2026-10-27, word given; zero GPU). Builds the kept token set K and
each dropped token's FALLBACK PIECES, so a pruned reader never meets a hard out-of-vocabulary token:
  K = every token in our training texts (form_mix_pm35c) + the MATH harvest problems + GSM8K TRAIN questions and answers
      EXCLUDING the wild holdout's problems (so the holdout exercises the fallback) + every single-character token (the
      byte-level base: any string decomposes into them)
  a dropped token's pieces = greedy longest-prefix segmentation of its token string over K's strings
At read time a dropped token's input row = the MEAN of its pieces' rows (the sequence and its positions are unchanged, so every
downstream span, mask and pointer is untouched). Writes .cache/reader_vocab_k.npz (keep, oov_ids, piece_ptr, pieces) + a census."""
import json, os, sys

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = ".cache/reader_vocab_k.npz"
P = ".cache/gsm8k/datasets--openai--gsm8k/snapshots/740312add88f781978c0658806c59bc2815b9866/main/"
HOLDOUT = ".cache/wild_admitted_holdout.jsonl"


def segment(s, kept_str):
    out, i = [], 0
    while i < len(s):
        for j in range(len(s), i, -1):
            if s[i:j] in kept_str:
                out.append(kept_str[s[i:j]]); i = j; break
        else:
            raise ValueError(f"no kept piece covers {s[i:]!r}")
    return out


if __name__ == "__main__":
    tok = Tokenizer.from_file(".cache/llama-3.2-1b-weights/tokenizer.json"); vocab = tok.get_vocab(); inv = {i: s for s, i in vocab.items()}
    ho = set(json.loads(l)["text"].strip() for l in open(HOLDOUT))
    K = set()
    for l in open(".cache/form_mix_pm35c.jsonl"):
        t = json.loads(l)["text"]
        if t.strip() not in ho: K.update(tok.encode(t).ids)
    for l in open(".cache/math_harvest_v0.jsonl"): K.update(tok.encode(json.loads(l)["problem"]).ids)
    tb = pq.read_table(P + "train-00000-of-00001.parquet")
    n_fenced = 0
    for q, a in zip(tb.column("question").to_pylist(), tb.column("answer").to_pylist()):
        if q.strip() in ho: n_fenced += 1; continue
        K.update(tok.encode(q).ids); K.update(tok.encode(a).ids)
    singles = {i for s, i in vocab.items() if len(s) == 1}; K |= singles
    kept_str = {inv[i]: i for i in K}
    oov = sorted(set(range(len(inv))) - K); ptr, pieces = [0], []
    for i in oov:
        pieces += segment(inv[i], kept_str); ptr.append(len(pieces))
    np.savez(OUT, keep=np.array(sorted(K), np.int32), oov_ids=np.array(oov, np.int32), piece_ptr=np.array(ptr, np.int64),
             pieces=np.array(pieces, np.int32))
    hid = [tok.encode(json.loads(l)["text"]).ids for l in open(HOLDOUT)]
    n = sum(map(len, hid)); miss = sum(i not in K for e in hid for i in e); tch = sum(any(i not in K for i in e) for e in hid)
    gt = [tok.encode(q).ids for q in pq.read_table(P + "test-00000-of-00001.parquet").column("question").to_pylist()]
    gn = sum(map(len, gt)); gm = sum(i not in K for e in gt for i in e); gtch = sum(any(i not in K for i in e) for e in gt)
    msg = (f"K = {len(K)} of {len(inv)} tokens ({len(K) * 512 / 1e6:.1f}M params at 512d; singles {len(singles)}); holdout problems "
           f"fenced from GSM8K-train {n_fenced} | HOLDOUT token OOV {miss / n:.4f}, rows touched {tch}/{len(hid)} | GSM8K TEST token OOV "
           f"{gm / gn:.4f}, problems touched {gtch / len(gt):.3f} | mean pieces per dropped token {len(pieces) / len(oov):.2f}")
    print(msg); open(OUT.replace(".npz", "_census.txt"), "w").write(msg + "\n")
