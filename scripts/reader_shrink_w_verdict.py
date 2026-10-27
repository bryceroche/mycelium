"""reader_shrink_w_verdict.py -- THE WHOLE-WORD BACKSTOP ARM (ledger 2026-10-27). Zero training.
  PRUNE-W  K + every space-prefixed whole-word token (48,170 rows = 24.7M) + fallback pieces: HCS on STUW >= HCS on STU - 0.010
           paired -> PASS. Reads beside: STUW vs STUP (what the whole words buy), and STUW on the 33 holdout rows still touched."""
import json, math, os

import numpy as np
from tokenizers import Tokenizer

OUT = ".cache/reader_shrink_w_verdict.txt"
if __name__ == "__main__":
    assert not os.path.exists(OUT), "refusing to overwrite a banked read"
    out = open(OUT, "w")

    def say(s):
        print(s, flush=True); out.write(s + "\n")
    R = {s: np.load(f".cache/ps_{'headcont_HCS' if s == 'STU' else 'shrink_' + s}.npz") for s in ("STU", "STUP", "STUW")}
    ok = {s: R[s]["ok"].astype(bool) for s in R}; assert all((R[s]["rows"] == R["STU"]["rows"]).all() for s in R)

    def paired(x, y, m=None):
        m = np.ones(len(ok[x]), bool) if m is None else m; a_, b_ = ok[x][m], ok[y][m]; n = len(a_)
        b, c = int((a_ & ~b_).sum()), int((~a_ & b_).sum())
        return b_.mean() - a_.mean(), math.sqrt(max((b + c) / n - ((b - c) / n) ** 2, 0)) / math.sqrt(n), b, c, n
    for x, y in (("STU", "STUW"), ("STUP", "STUW")):
        d, se, b, c, n = paired(x, y); say(f"{y} vs {x}: {ok[y].mean():.4f} vs {ok[x].mean():.4f} | diff {d:+.4f} (paired SE {se:.4f}; discordant {b}/{c})")
    dW = paired("STU", "STUW")[0]
    say("VERDICT (the whole-word backstop)")
    say(f"PRUNE-W (-> 48,170 rows = 24.7M + fallback pieces): {'PASS' if dW >= -0.010 else 'FAIL'} (diff {dW:+.4f}, bar -0.010)")
    K = set(np.load(".cache/reader_vocab_kw.npz")["keep"].tolist()); tok = Tokenizer.from_file(".cache/llama-3.2-1b-weights/tokenizer.json")
    touched = np.array([any(i not in K for i in tok.encode(json.loads(l)["text"]).ids) for l in open(".cache/wild_admitted_holdout.jsonl")])
    m = touched[R["STU"]["rows"]]
    for name, mm in (("rows still touched", m), ("rows untouched", ~m)):
        d, se, b, c, n = paired("STU", "STUW", mm); say(f"READ STUW on {name}: {n} slots, diff {d:+.4f} (paired SE {se:.4f}; discordant {b}/{c})")
