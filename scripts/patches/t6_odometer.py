"""Patch (ledger 2026-11-02): THE ODOMETER -- the discrete mirror of the T6 gears (CLAUDE.md §9). Applied after lfq5's last launch."""
p = "mycelium/rotor_clock.py"; s = open(p).read()
assert "def t6_odometer" not in s, "already applied"
anchor = "# ------------------------------------------------------------- self-tests\n"
code = '''def t6_radix(speeds=T6_SPEEDS):
    """Per-gear digit radix: gear 0 has period P0, gear g turns P_g / P_(g-1) bins per revolution (T6: 2,3,2,2,2,2 = 96)."""
    P = t6_periods(speeds); r = [P[0]] + [P[g] / P[g - 1] for g in range(1, len(P))]
    assert all(abs(x - round(x)) < 1e-9 for x in r), f"odometer needs integer period ratios: {r}"
    return [int(round(x)) for x in r]


def t6_odometer(tick, speeds=T6_SPEEDS):
    """THE ODOMETER (ledger 11-02): the gears SNAPPED to ticks. Digit g = floor(gear g's angle / (360 / radix_g)): the same gears that
    turn continuously in attention, read as a car odometer (gear 0 = the fast wheel). Mixed radix, so 96 distinct readings per
    revolution of the slowest gear. Integer-exact (no float angle is binned)."""
    t, P, R = int(tick), [int(round(p)) for p in t6_periods(speeds)], t6_radix(speeds)
    return tuple((t % P[g]) * R[g] // P[g] for g in range(len(P)))


def odometer_tick(digits, speeds=T6_SPEEDS):
    """Inverse of t6_odometer within one revolution: the tick in [0, prod(radix))."""
    R, t, w = t6_radix(speeds), 0, 1
    for d, r in zip(digits, R):
        assert 0 <= d < r, f"digit {d} out of radix {r}"
        t += d * w; w *= r
    return t


'''
assert s.count(anchor) == 1; s = s.replace(anchor, code + anchor)
test_anchor = "    check_range(T6_SPEEDS, 48)\n"
test = '''    # 6d. the odometer: 96 unique readings per revolution, round-trips, wraps, and agrees with the gears' snapped angles
    assert t6_radix() == [2, 3, 2, 2, 2, 2]
    reads = [t6_odometer(t) for t in range(96)]
    assert len(set(reads)) == 96 and all(odometer_tick(r) == t for t, r in enumerate(reads)) and t6_odometer(96) == reads[0]
    for t in range(96):
        ang = np.round(np.degrees(t6_phases(t, 6)), 6) % 360.0
        assert tuple(int(a // (360.0 / r)) for a, r in zip(ang, t6_radix())) == reads[t], f"snap != odometer at {t}"
'''
assert s.count(test_anchor) == 1; s = s.replace(test_anchor, test_anchor + test); open(p, "w").write(s); print("applied t6_odometer")
