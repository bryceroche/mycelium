"""Decay stub - minimal for cleanup."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

class DecayStatus(Enum):
    ACTIVE = "active"
    DECAYING = "decaying"
    DORMANT = "dormant"

@dataclass
class DecayState:
    status: DecayStatus = DecayStatus.ACTIVE
    uses: int = 0

@dataclass
class DecayAction:
    action: str = "none"

@dataclass
class DecayReport:
    checked: int = 0
    decayed: int = 0

class DecayManager:
    """Stub decay manager."""

    def __init__(self, **kwargs):
        pass

    def check_signature(self, sig_id: int) -> DecayAction:
        return DecayAction()

def run_decay_cycle(**kwargs) -> DecayReport:
    """Stub - no-op."""
    return DecayReport()

def get_signature_decay_state(sig_id: int) -> DecayState:
    """Stub."""
    return DecayState()

def get_decay_summary() -> dict:
    """Stub."""
    return {"active": 0, "decaying": 0, "dormant": 0}
