"""Read a training log, extract pb_ce per step, compute the smoothness of the
per-breath CE ladder.

A "smooth" ladder is one where each breath reduces remaining CE by roughly the
same FRACTION (uniform % change). CE is log-scale, so absolute deltas are not
the right metric — multiplicative reduction is.

For each step:
  pb_ce[k] = breath-k cross-entropy
  pct_change[k] = (pb_ce[k] - pb_ce[k+1]) / pb_ce[k]    for k in [0, K-2]
  smoothness   = std(pct_change) / mean(pct_change)     (coefficient of variation)

Smoother → CV closer to 0. Cliff pattern → high CV (one transition >> others).

Also reports the theoretical "target ladder" for the given range:
  c = 1 - (pb_ce[K-1] / pb_ce[0])^(1/(K-1))
  → uniform % reduction implied by current endpoints

Usage:
  python scripts/analyze_pb_ce_ladder.py [LOG_PATH] [--last N]
"""
import re
import sys
import os
import numpy as np

PB_CE_RE = re.compile(
    r"^step\s+(\d+)\s+K=(\d+)\s+loss=([\d.]+)\s+pb_ce=\[([\d.\s]+)\]"
)


def parse_log(path: str):
    """Yield (step, K, loss, pb_ce_arr) per matching line."""
    with open(path) as f:
        for line in f:
            m = PB_CE_RE.match(line.strip())
            if not m:
                continue
            step = int(m.group(1))
            K = int(m.group(2))
            loss = float(m.group(3))
            ce_strs = m.group(4).split()
            ce = np.array([float(x) for x in ce_strs])
            yield step, K, loss, ce


def analyze(ce: np.ndarray) -> dict:
    """Compute ladder smoothness metrics for one pb_ce row."""
    K = len(ce)
    # Avoid div-by-zero / negative deltas (allow them, but flag).
    pct_change = []
    for k in range(K - 1):
        if ce[k] <= 1e-6:
            pct_change.append(0.0)
        else:
            pct_change.append((ce[k] - ce[k + 1]) / ce[k])
    pct_change = np.array(pct_change)
    # Target ladder: uniform % reduction implied by endpoints.
    if ce[0] > 1e-6 and ce[-1] > 0:
        target_c = 1.0 - (ce[-1] / ce[0]) ** (1.0 / (K - 1))
    else:
        target_c = 0.0
    # Coefficient of variation — lower = smoother.
    pos_mask = pct_change > 0
    if pos_mask.sum() > 0:
        mean_pct = pct_change[pos_mask].mean()
        std_pct = pct_change.std()  # over all transitions, even negatives
        cv = std_pct / mean_pct if mean_pct > 0 else float("inf")
    else:
        mean_pct = 0.0
        cv = float("inf")
    # Max-cliff: largest negative-direction percent-change (the biggest "drop").
    max_drop_k = int(np.argmax(pct_change))
    max_drop_val = float(pct_change[max_drop_k])
    return dict(
        pct_change=pct_change,
        target_c=target_c,
        cv=cv,
        max_drop_k=max_drop_k,
        max_drop_val=max_drop_val,
        mean_pct=mean_pct,
    )


def format_row(step: int, ce: np.ndarray, metrics: dict) -> str:
    """One-line summary of a single step's ladder."""
    K = len(ce)
    pc = metrics["pct_change"]
    ce_str = " ".join(f"{x:5.2f}" for x in ce)
    pct_str = " ".join(f"{p*100:+5.1f}%" for p in pc)
    return (
        f"step {step:>4}  ce=[{ce_str}]  "
        f"%Δ=[{pct_str}]  "
        f"CV={metrics['cv']:.2f}  "
        f"target_c={metrics['target_c']*100:.1f}%  "
        f"max_drop@L{metrics['max_drop_k']}→L{metrics['max_drop_k']+1}"
        f"={metrics['max_drop_val']*100:.1f}%"
    )


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/bryce/mycelium/.cache/v80_smoke_train.log"
    last_n = None
    if "--last" in sys.argv:
        last_n = int(sys.argv[sys.argv.index("--last") + 1])

    if not os.path.exists(log_path):
        print(f"ERROR: log not found at {log_path}", file=sys.stderr)
        sys.exit(1)

    rows = list(parse_log(log_path))
    if not rows:
        print(f"No pb_ce rows in {log_path}.", file=sys.stderr)
        sys.exit(1)

    if last_n is not None:
        rows = rows[-last_n:]

    print(f"=== Per-step ladder shape from {log_path} ===")
    print(f"(showing {len(rows)} rows)\n")
    print("Headers: ce=[L0..LK-1]   %Δ between consecutive breaths   CV=std/mean")
    print("         target_c = uniform %-reduction implied by endpoints")
    print("         max_drop@k = biggest %-drop transition (cliff location)\n")

    for step, K, loss, ce in rows:
        m = analyze(ce)
        print(format_row(step, ce, m))

    # Summary trajectory
    if len(rows) >= 3:
        print()
        print("=== Smoothness trajectory ===")
        steps = [r[0] for r in rows]
        cvs = []
        target_cs = []
        max_drops = []
        for _step, _K, _loss, ce in rows:
            m = analyze(ce)
            cvs.append(m["cv"])
            target_cs.append(m["target_c"])
            max_drops.append(m["max_drop_val"])
        print(f"  CV at step {steps[0]}:  {cvs[0]:.2f}  (lower = smoother)")
        print(f"  CV at step {steps[-1]}: {cvs[-1]:.2f}")
        print(f"  Mean CV last 5: {np.mean(cvs[-5:]):.2f}")
        print(f"  target_c at step {steps[-1]}: {target_cs[-1]*100:.1f}% (uniform-ladder target)")
        print(f"  Max-cliff %-drop last 5 steps mean: {np.mean(max_drops[-5:])*100:.1f}%")
        print()
        # Honest verdict
        recent_cv = np.mean(cvs[-5:])
        if recent_cv < 0.3:
            verdict = "SMOOTH — uniform % reduction per breath, ladder is well-shaped"
        elif recent_cv < 0.7:
            verdict = "MILD CLIFF — one transition is doing more work than others"
        else:
            verdict = "CLIFF — one transition dominates, format needs adjustment"
        print(f"  Verdict (recent 5 steps): {verdict}")


if __name__ == "__main__":
    main()
