"""convergence.py — THE CONVERGENCE CONTRACT (hill 5, ledger 2026-10-04 08:09).

No model is scored until its loss has LEVELLED, by a stated rule — else a
"null" verdict may just mean "not done" (the U-Net bake-off's own honest
caveat: its loss was still falling when read). Call before a chain's
first loop_val read, pointed at the training log that --train just wrote:

    .venv/bin/python3 scripts/contracts/convergence.py .cache/sharp_$C.log || exit 1

Parses lines of the shape `step <N> loss=<F> ...` — the head's
`  step  19800 loss=0.21823 lr=...` (scripts/phase1_algebra_head.py,
printed every 500 steps by default) and the U-Net's `[unet/train] step
19400 loss=0.31294 ...` (scripts/unet/train.py, every 200) both match;
neither script's name nor the printing cadence matters to the regex.

THE RULE (over the last 10% of recorded STEPS, by step-number range —
`step >= 0.9 * max_step`, not by line count, so a log whose print cadence
changed mid-run still slices correctly):
  1. TREND — smooth the window's (step, loss) series with a small moving
     median (window `--smooth`, default 3, odd, clipped to the slice
     length) to cut single-step noise, then take the relative change
     smoothed[-1] vs smoothed[0]: |end - start| / |start|. PASS needs
     this under `--rule` percent (default 1.0 -> 1%).
  2. SPIKE GUARD — the window's single RAW-loss median vs every raw loss
     in the window: no point may exceed 1.5x that median. This is the
     one that catches the head's/U-Net's own well-known last-line jump
     (a val/restore print logged in the same `step N loss=F` shape but
     not a training-step loss) — a FAIL from this guard naming the last
     step specifically is that known artifact, not a surprise; it is
     still reported as a FAIL because the artifact is real in the log
     and this contract does not special-case it away.
Both must hold for PASS. Exit code: 0 (PASS) else 1 (FAIL/no data).
"""
import argparse
import re
import statistics
import sys

STEP_RE = re.compile(r"\bstep\s+(\d+)\s+loss=([0-9.eE+-]+)")


def parse_log(path):
    pts = []
    for line in open(path, errors="replace"):
        m = STEP_RE.search(line)
        if not m:
            continue
        try:
            pts.append((int(m.group(1)), float(m.group(2))))
        except ValueError:
            continue
    # a step can be reprinted (RESUME segments restate step 0) — keep the
    # LAST occurrence of each step, in first-to-last file order
    by_step = {}
    order = []
    for s, l in pts:
        if s not in by_step:
            order.append(s)
        by_step[s] = l
    return [(s, by_step[s]) for s in order]


def moving_median(ys, window):
    window = max(1, min(window, len(ys)))
    if window % 2 == 0:
        window -= 1
    if window <= 1:
        return list(ys)
    half = window // 2
    out = []
    for i in range(len(ys)):
        lo, hi = max(0, i - half), min(len(ys), i + half + 1)
        out.append(statistics.median(ys[lo:hi]))
    return out


def check(points, rule_pct=1.0, smooth=3):
    if len(points) < 4:
        return {"ok": False, "reason": f"only {len(points)} (step, loss) lines parsed — too few to judge"}
    max_step = points[-1][0]
    cutoff = 0.9 * max_step
    window = [(s, l) for s, l in points if s >= cutoff]
    if len(window) < 3:
        window = points[-max(3, len(points) // 10):]
    steps = [s for s, _ in window]
    losses = [l for _, l in window]

    smoothed = moving_median(losses, smooth)
    start, end = smoothed[0], smoothed[-1]
    rel_change = abs(end - start) / abs(start) if start else float("inf")
    trend_ok = rel_change < (rule_pct / 100.0)

    med = statistics.median(losses)
    worst_i = max(range(len(losses)), key=lambda i: losses[i])
    worst_step, worst_loss = steps[worst_i], losses[worst_i]
    spike_ratio = (worst_loss / med) if med else float("inf")
    spike_ok = spike_ratio <= 1.5

    return {
        "ok": trend_ok and spike_ok,
        "max_step": max_step, "n_total": len(points), "n_window": len(window),
        "window_steps": (steps[0], steps[-1]),
        "smoothed_start": start, "smoothed_end": end, "rel_change": rel_change,
        "rule_pct": rule_pct, "trend_ok": trend_ok,
        "window_median": med, "worst_step": worst_step, "worst_loss": worst_loss,
        "spike_ratio": spike_ratio, "spike_ok": spike_ok,
        "is_last_point": worst_step == steps[-1],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="training log to read (scripts/phase1_algebra_head.py or scripts/unet/train.py stdout)")
    ap.add_argument("--rule", type=float, default=1.0, help="trend threshold in percent (default 1.0 = 1%%)")
    ap.add_argument("--smooth", type=int, default=3, help="moving-median window, odd (default 3)")
    args = ap.parse_args()

    points = parse_log(args.log)
    r = check(points, rule_pct=args.rule, smooth=args.smooth)
    if "window_steps" not in r:
        print(f"[convergence] {args.log}: FAIL — {r['reason']}")
        return 1

    print(f"[convergence] {args.log}: {r['n_total']} (step,loss) lines; "
          f"last 10% window = steps {r['window_steps'][0]}..{r['window_steps'][1]} ({r['n_window']} pts)")
    print(f"  TREND: smoothed(median w={args.smooth}) {r['smoothed_start']:.5f} -> {r['smoothed_end']:.5f} | "
          f"relative change {r['rel_change']*100:.3f}% (rule: < {r['rule_pct']:.3f}%) -> "
          f"{'PASS' if r['trend_ok'] else 'FAIL'}")
    print(f"  SPIKE GUARD: window median {r['window_median']:.5f} | worst = step {r['worst_step']} "
          f"loss={r['worst_loss']:.5f} ({r['spike_ratio']:.2f}x median, limit 1.50x) -> "
          f"{'PASS' if r['spike_ok'] else 'FAIL'}"
          + (" [the known last-line jump — a val/restore print, not a training step]" if r["is_last_point"] and not r["spike_ok"] else ""))
    print(f"[convergence] {args.log}: {'PASS' if r['ok'] else 'FAIL'}")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
