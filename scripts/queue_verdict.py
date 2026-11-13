"""queue_verdict.py -- BIC-x / TSEG-x verdicts (ledger 2026-11-12 "THE QUEUE PINNED"): STACK (BEAM4+CMOTIF) TEST-U per arm, 4 k-seeds x
{local, full}, minus the banked STACK; mean over seeds per source; PASS iff >= -0.010 on both, GAIN iff >= +0.020 on both."""
import re

def f(src, k, spans="", suf=""):
    return f".cache/gsm_ladder{'' if src == 'full' else '_' + src}{spans}{'' if k == 0 else f'_k{k}'}_chain_beam4_cmotif1.0{suf}.txt"
def stack(path):
    m = re.search(r"BEAM4\+CMOTIF : TEST-U ([0-9.]+)", open(path).read()); assert m, path; return float(m.group(1))
ARMS = {"BIC (x-means + law)": dict(suf="_bic_law"), "LAW (KS + law, unbarred)": dict(suf="_law"), "TREE-SEG": dict(spans="_tree")}
out = []
for name, kw in ARMS.items():
    means = {}
    for src in ("local", "full"):
        b = [stack(f(src, k)) for k in range(4)]; a = [stack(f(src, k, **kw)) for k in range(4)]; d = [x - y for x, y in zip(a, b)]
        means[src] = sum(d) / 4
        out.append(f"  {name:26s} {src:5s}: arm " + " / ".join(f"{x:.4f}" for x in a) + " | banked " + " / ".join(f"{x:.4f}" for x in b) + f" | diff mean {means[src]:+.4f}")
    v = "GAIN" if min(means.values()) >= 0.020 else ("PASS" if min(means.values()) >= -0.010 else "FAIL")
    if not name.startswith("LAW"): out.append(f"{'BIC-x' if name.startswith('BIC') else 'TSEG-x'}: local {means['local']:+.4f}, full {means['full']:+.4f} -> {v}")
print("\n".join(out)); open(".cache/queue_verdict.txt", "w").write("\n".join(out) + "\n")
