"""perceiver_report.py — re-report a saved LoopHealth record (CPU, no GPU):
the calibration table (signed AUROC per meter per breath) and THE WHAT-STANDS
CURVE at slot level (precision vs coverage by the slot's own margin).
usage: perceiver_report.py record.npz [field=args] [breath=-1]"""
import sys
sys.path.insert(0, ".")
from mycelium.perceiver import LoopHealth
hl = LoopHealth.load(sys.argv[1]); field = sys.argv[2] if len(sys.argv) > 2 else "args"; k = int(sys.argv[3]) if len(sys.argv) > 3 else -1
hl.report()
for f in ("args", "res"):
    print(f"[what-stands] slot-level, sorted by the slot's own {f} margin at breath {k}: " + "; ".join(f"cov {c:.0%} -> precision {p:.3f} (n={n})" for c, p, n in hl.coverage_curve(f, k)))
