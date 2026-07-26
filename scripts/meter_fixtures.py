"""meter_fixtures.py — known-signal fixtures for probe-side meters
(the fresh-apparatus corollary, 2026-07-25). ALL must pass before any
real read banks. Would have caught all three of tonight's ruler faults.
"""
import sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from dart_cluster_probe import auc_mann_whitney

fails = []
# 1. AUC on perfectly separable synthetic labels must be 1.0 (catches
#    argument-order faults: swapped args here yield 0.5).
rng = np.random.default_rng(1)
s = np.concatenate([rng.normal(5, .1, 200), rng.normal(-5, .1, 800)])
y = np.concatenate([np.ones(200, bool), np.zeros(800, bool)])
a = auc_mann_whitney(s, y)
print(f"[fixture 1] separable AUC = {a:.3f} (must be 1.000)"); a == 1.0 or fails.append(1)
# 2. AUC on shuffled labels must sit near 0.5 (sanity of the null).
a2 = auc_mann_whitney(s, rng.permutation(y))
print(f"[fixture 2] shuffled AUC = {a2:.3f} (must be in [.45,.55])"); .45 < a2 < .55 or fails.append(2)
# 3. t95 must locate a planted rise (catches ceiling/floor degeneracy:
#    the fixture curve rises 0.2->0.8 with knee at breath 6).
def t95(c):
    c = np.array(c); return int(np.argmax(c >= 0.95*c[-1]))
curve = [0.2,0.3,0.4,0.5,0.6,0.7,0.78,0.79,0.80,0.80,0.80,0.80,0.80,0.80,0.80,0.80]
t = t95(curve)
print(f"[fixture 3] planted-knee t95 = {t} (must be 6)"); t == 6 or fails.append(3)
# 4. range criterion must REJECT a flat curve (either altitude) and
#    ACCEPT the planted rise.
def range_ok(c, base): c = np.array(c); return (c.max()-c[0]) >= 0.05 and bool(np.all(c > base))
r_flat_hi = range_ok([0.9]*16, 0.5); r_flat_lo = range_ok([0.5]*16, 0.5); r_rise = range_ok(curve, 0.1)
print(f"[fixture 4] range: flat-high {r_flat_hi} flat-low {r_flat_lo} rise {r_rise} (must be False/False/True)")
(not r_flat_hi and not r_flat_lo and r_rise) or fails.append(4)
# 5. ridge probe on synthetic linear signal must beat 0.9 AUC (end-to-end).
X = rng.normal(size=(2000, 64)).astype(np.float32); w0 = rng.normal(size=64)
yb = (X @ w0 > 0)
Xf, Xr, yf, yr = X[:1500], X[1500:], yb[:1500], yb[1500:]
w = np.linalg.solve(Xf.T@Xf + 10*np.eye(64, dtype=np.float32), Xf.T@(2.*yf-1).astype(np.float32))
a5 = auc_mann_whitney(Xr@w, yr)
print(f"[fixture 5] end-to-end ridge AUC = {a5:.3f} (must be > 0.9)"); a5 > 0.9 or fails.append(5)
print("METER FIXTURES:", "ALL PASS" if not fails else f"FAIL {fails}")
sys.exit(1 if fails else 0)
