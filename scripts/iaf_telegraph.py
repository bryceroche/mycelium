"""iaf_telegraph.py — THE ATTENTION-GRAIN TELEGRAPH READ (2026-08-27,
word given): Bryce's original ping-pong memory, measured on its own
data — the S3 IAF corpus (Qwen CoT generations on MATH; 10 monitored
heads L5..L25; per-generated-token top-20 attended positions).
ANALYSIS-ONLY corpus (measured-set custody). Per head:
  (a) NEAR/FAR mode telegraph: lookback distance thresholded ->
      binary channel; transition rate vs shuffled null; dwell-time
      histogram (the dots and dashes);
  (b) ANCHOR RETURNS: A->B->A top-position revisits vs null;
  (c) CROSS-HEAD ANTI-PHASE: correlation matrix of the binary channels.
Pilot: chunk_000 (50 records, local scratch). Stats only ever printed.
"""
import sys, json, re
sys.path.insert(0, '.')
import numpy as np
from collections import Counter

SCRATCH = '/tmp/claude-1000/-home-bryce-mycelium/d21a0793-1f54-4ba7-b544-12222df5f8cb/scratchpad'
NEAR = 8            # lookback <= NEAR tokens = local mode
rng = np.random.default_rng(0)
NPERM = 50

def main():
    data = json.load(open(f'{SCRATCH}/chunk000.json'))
    print(f"[iaf] records {len(data)}", flush=True)
    heads = None
    tr_obs = Counter(); tr_null = Counter(); ret_obs = Counter(); ret_null = Counter()
    nsteps = Counter()
    dwell = {}
    chans = {}
    for r in data:
        tp = r['top_positions']
        il = r['input_len']
        if heads is None:
            heads = sorted(tp[0].keys(),
                           key=lambda k: (int(re.match(r'L(\d+)', k).group(1)),
                                          int(re.search(r'H(\d+)', k).group(1))))
            for h in heads: dwell[h] = []; chans[h] = []
        seqs = {h: [] for h in heads}
        for t, entry in enumerate(tp):
            for h in heads:
                lst = entry.get(h)
                if lst:
                    seqs[h].append(int(lst[0]['pos']))
        for h in heads:
            s = np.array(seqs[h])
            if len(s) < 20: continue
            cur = il + np.arange(len(s))          # absolute position of step t
            look = cur - s
            ch = (look > NEAR).astype(np.int8)    # 1 = FAR/retrieval mode
            chans[h].append(ch)
            tr = int((ch[1:] != ch[:-1]).sum())
            tr_obs[h] += tr; nsteps[h] += len(ch) - 1
            for _ in range(NPERM):
                cp = rng.permutation(ch)
                tr_null[h] += int((cp[1:] != cp[:-1]).sum())
            runs = np.diff(np.flatnonzero(np.concatenate(
                ([True], ch[1:] != ch[:-1], [True]))))
            dwell[h].extend(runs.tolist())
            ret = sum(1 for t in range(2, len(s))
                      if s[t] == s[t-2] and s[t] != s[t-1])
            ret_obs[h] += ret
            for _ in range(NPERM):
                sp = rng.permutation(s)
                ret_null[h] += sum(1 for t in range(2, len(sp))
                                   if sp[t] == sp[t-2] and sp[t] != sp[t-1])
    print(f"[iaf] heads: {heads}", flush=True)
    print("[iaf] head | far%  | trans/step obs vs null | A-B-A/step obs vs null | median dwell (dot,dash)")
    for h in heads:
        n = max(nsteps[h], 1)
        far = np.mean([c.mean() for c in chans[h]])
        to = tr_obs[h] / n; tn = tr_null[h] / (n * NPERM)
        ro = ret_obs[h] / n; rn = ret_null[h] / (n * NPERM)
        d = np.array(dwell[h])
        print(f"[iaf] {h:7s}| {far:.2f} | {to:.3f} vs {tn:.3f} ({to/max(tn,1e-9):.2f}x) "
              f"| {ro:.4f} vs {rn:.4f} ({ro/max(rn,1e-9):.2f}x) "
              f"| dwell med {np.median(d):.0f} p90 {np.percentile(d,90):.0f}",
              flush=True)
    # cross-head anti-phase (pilot: mean pairwise correlation of channels)
    print("[iaf] cross-head channel correlations (negative = anti-phase pairs):")
    M = []
    for h in heads:
        cat = np.concatenate(chans[h]).astype(np.float32)
        M.append(cat[:min(len(c) for c in (np.concatenate(chans[x]) for x in heads))])
    L = min(len(m) for m in M)
    M = np.stack([m[:L] for m in M])
    C = np.corrcoef(M)
    ij = np.unravel_index(np.argsort(C, axis=None), C.shape)
    pairs = [(heads[i], heads[j], C[i, j]) for i, j in zip(*ij) if i < j]
    for a, b, c in pairs[:4]:
        print(f"[iaf]   {a} x {b}: {c:+.2f}")
    print(f"[iaf]   most positive: {pairs[-1][0]} x {pairs[-1][1]}: {pairs[-1][2]:+.2f}")

if __name__ == "__main__":
    main()
