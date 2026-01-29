"""State Manager stub - minimal for cleanup."""

from dataclasses import dataclass
from typing import Optional

@dataclass
class WelfordStats:
    """Welford algorithm stats."""
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        return self.variance ** 0.5

class StateManager:
    """Stub state manager."""

    def get_welford_stats(self, key: str) -> WelfordStats:
        return WelfordStats()

    def update_welford(self, key: str, value: float) -> WelfordStats:
        return WelfordStats(count=1, mean=value, m2=0.0)

    def get_metadata(self, key: str, default=None):
        return default

    def set_metadata(self, key: str, value):
        pass

_state_manager: Optional[StateManager] = None

def get_state_manager() -> StateManager:
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager()
    return _state_manager
