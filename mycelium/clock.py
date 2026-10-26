"""clock.py -- THE T6 GEAR CLOCK, the project standard for the breath rotor (Bryce's ruling 2026-10-26: "lets make T6 clock
the project standard"; ledger "THE T6 CLOCK RULING").
Six gears, each a CONTINUOUS rotation (cos, sin of speed x tick) at its own speed; every gear turns every tick (clock hands,
no odometer carry). Six independent circles = the torus T^6; one tick counter traces a closed loop on it (the speeds are
commensurate: the loop closes after 360 / min(speed) ticks). Relative time is a rotation: clock(s) * conj(clock(now)).
The deployed head's six-wave (one period-6 wheel read at six 60-degree phases, phase1_algebra_head ALG_SIXWAVE / ALG_SW_TICK)
is NOT this clock (T2m: FLAT6 0.906 vs GEARS 1.000 in-dist); it stays under its own flags until a registered run replaces it.

THE CLOCK-RANGE RULE (T3a-AUTOPSY + T3b): a gate deciding on relative phase d over a span of S distinct values needs
REDUNDANT alias-free gears (period > S - 1). Faster gears alias distant d onto near d; one alias-free gear can be out-voted
by the aliasing ones (four-wave at NB = 12 had exactly one, 15 deg, and seed 1 built its step from the others)."""
T6_SPEEDS = (180.0, 60.0, 30.0, 15.0, 7.5, 3.75)                             # degrees per tick; periods 2, 6, 12, 24, 48, 96
MIN_ALIAS_FREE = 2                                                          # T3b/T3d: four-wave's single free gear failed


def periods(speeds=T6_SPEEDS):
    return [360.0 / s for s in speeds]


def alias_free(speeds, span):
    """The gears whose period exceeds the largest |d1 - d2| over a span of `span` distinct relative phases."""
    return [s for s in speeds if 360.0 / s > span - 1]


def check_range(speeds, span, minimum=MIN_ALIAS_FREE):
    free = alias_free(speeds, span)
    assert len(free) >= minimum, (f"clock-range rule: span {span} needs >= {minimum} alias-free gears (period > {span - 1}); "
                                  f"speeds {tuple(speeds)} have {len(free)}: {free}")
    return free


def plane_speeds(n_planes, speeds=T6_SPEEDS):
    """Assign the gears to n_planes rotation planes cyclically (plane p turns at speeds[p % len(speeds)])."""
    return [speeds[p % len(speeds)] for p in range(n_planes)]
