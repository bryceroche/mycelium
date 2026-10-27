"""reader_shrink_verdict.py -- THE SHRUNK READER pinned bars (ledger 2026-10-27). Zero training; reads the banked per-slot files.
  FOLD   EMB @ w_in folded into one fp16 128,256 x 512 table: |HCS on STUF - HCS on STU| <= 0.005 paired -> EXACT (and the states'
         max |diff| on real tokens reported)
  PRUNE  16,729 kept rows + dropped tokens composed from fallback pieces: HCS on STUP >= HCS on STU - 0.010 paired -> PASS
  READ   the prune read on the holdout rows that contain a dropped token (96/311) vs the rest"""
import json, math, os

import numpy as np
from tokenizers import Tokenizer

OUT = ".cache/reader_shrink_verdict.txt"
if __name__ == "__main__":
    assert not os.path.exists(OUT), "refusing to overwrite a banked read"
    out = open(OUT, "w")

    def say(s):
        print(s, flush=True); out.write(s + "\n")
    tm = np.load(".cache/phase1_alg_states_wildholdSTU.npz")["tokmask"] > 0
    a, f = (np.load(f".cache/phase1_alg_states_wildhold{s}_states.npy", mmap_mode="r") for s in ("STU", "STUF"))
    d = np.abs(np.asarray(a, np.float32) - np.asarray(f, np.float32))[tm]
    say(f"FOLD states vs STU on real tokens: max |diff| {d.max():.4f}, mean {d.mean():.6f} (state scale: mean |x| {np.abs(np.asarray(a, np.float32)[tm]).mean():.4f})")
    R = {s: np.load(f".cache/ps_{'headcont_HCS' if s == 'STU' else 'shrink_' + s}.npz") for s in ("STU", "STUF", "STUP")}
    ok = {s: R[s]["ok"].astype(bool) for s in R}; assert all((R[s]["rows"] == R["STU"]["rows"]).all() for s in R)

    def paired(x, y, m=None):
        m = np.ones(len(ok[x]), bool) if m is None else m; a_, b_ = ok[x][m], ok[y][m]; n = len(a_)
        b, c = int((a_ & ~b_).sum()), int((~a_ & b_).sum())
        return b_.mean() - a_.mean(), math.sqrt(max((b + c) / n - ((b - c) / n) ** 2, 0)) / math.sqrt(n), b, c, n
    for s in ("STUF", "STUP"):
        dd, se, b, c, n = paired("STU", s); say(f"{s}: {ok[s].mean():.4f} vs STU {ok['STU'].mean():.4f} | diff {dd:+.4f} (paired SE {se:.4f}; discordant {b}/{c})")
    say("VERDICTS (the shrunk reader)")
    dF = paired("STU", "STUF")[0]; dP = paired("STU", "STUP")[0]
    say(f"FOLD (262.7M -> 65.7M, exact by construction): {'EXACT' if abs(dF) <= 0.005 else 'NOT EXACT'} (diff {dF:+.4f})")
    say(f"PRUNE (-> 16,729 rows = 8.6M + fallback pieces): {'PASS' if dP >= -0.010 else 'FAIL'} (diff {dP:+.4f}, bar -0.010)")
    z = np.load(".cache/reader_vocab_k.npz"); K = set(z["keep"].tolist()); tok = Tokenizer.from_file(".cache/llama-3.2-1b-weights/tokenizer.json")
    touched = np.array([any(i not in K for i in tok.encode(json.loads(l)["text"]).ids) for l in open(".cache/wild_admitted_holdout.jsonl")])
    m = touched[R["STU"]["rows"]]
    for name, mm in (("rows WITH a dropped token", m), ("rows without", ~m)):
        dd, se, b, c, n = paired("STU", "STUP", mm); say(f"READ prune on {name}: {n} slots, diff {dd:+.4f} (paired SE {se:.4f}; discordant {b}/{c})")
