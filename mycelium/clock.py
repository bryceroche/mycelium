"""clock.py -- a thin alias: THE T6 GEAR CLOCK lives in mycelium/rotor_clock.py (the master clock's single source of truth,
sync enforced by the import graph; ruling 2026-10-26). Import from there; this name exists for the toys that used it."""
from mycelium.rotor_clock import (T6_SPEEDS, MIN_ALIAS_FREE, alias_free, check_range, t6_periods as periods,  # noqa: F401
                                  t6_plane_speeds as plane_speeds, t6_phases)
