"""profile_summary.py — rank kernels by device time from a tinygrad PROFILE
pickle (tinygrad.device ProfileRangeEvent list). Usage: profile_summary.py <pkl> [top]"""
import sys, pickle, collections
evs = pickle.load(open(sys.argv[1], "rb")); top = int(sys.argv[2]) if len(sys.argv) > 2 else 25
by = collections.defaultdict(lambda: [0.0, 0]); total = 0.0; copies = 0.0
def add(nm, dev, d, is_copy=False):
    global total, copies
    if not str(dev).startswith(("AMD", "PCI")): return
    if is_copy: copies += d; nm = "[copy] " + str(nm)[:40]
    k = str(nm)[:70]; by[k][0] += d; by[k][1] += 1; total += d
for e in evs:
    if hasattr(e, "ents") and hasattr(e, "sigs"):            # ProfileGraphEvent (HCQ graphs)
        for ent in e.ents:
            add(ent.name, ent.device, float(e.sigs[ent.en_id] - e.sigs[ent.st_id]))
    elif hasattr(e, "st") and hasattr(e, "en"):               # ProfileRangeEvent
        add(getattr(e, "name", "?"), getattr(e, "device", ""), float(e.en - e.st), getattr(e, "is_copy", False))
rows = sorted(by.items(), key=lambda kv: -kv[1][0])
print(f"[profile] {len(evs)} events; device time total={total/1e3:.1f} ms (copies {copies/1e3:.1f} ms); distinct names={len(by)}")
cum = 0.0
for nm, (d, n) in rows[:top]:
    cum += d; print(f"  {d/1e3:8.1f} ms  {100*d/total:5.1f}%  cum {100*cum/total:5.1f}%  x{n:<5d} {nm}")
