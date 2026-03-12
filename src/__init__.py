"""
Mycelium v7 Source Package

Core modules:
    - oracle: SymPy telegram execution and answer verification
    - energy_landscape: Learned energy function for solution quality
    - ode_solver: Gradient descent refinement via ODE integration
    - ode_integration: Full pipeline wiring everything together
"""

from .oracle import (
    execute_telegram,
    execute_sequence,
    compare_answers,
    parse_telegram_expr,
)

from .energy_landscape import (
    EnergyLandscape,
    NodeEnergyMLP,
    PairEnergyMLP,
    create_energy_landscape,
    HIDDEN_DIM,
)

from .ode_solver import (
    ODESolver,
    AdaptiveODESolver,
    ODEResult,
    create_ode_solver,
)

from .ode_integration import (
    ODEIntegration,
    RefinementResult,
    load_integration,
)

__all__ = [
    # Oracle
    'execute_telegram',
    'execute_sequence',
    'compare_answers',
    'parse_telegram_expr',
    # Energy
    'EnergyLandscape',
    'NodeEnergyMLP',
    'PairEnergyMLP',
    'create_energy_landscape',
    'HIDDEN_DIM',
    # ODE
    'ODESolver',
    'AdaptiveODESolver',
    'ODEResult',
    'create_ode_solver',
    # Integration
    'ODEIntegration',
    'RefinementResult',
    'load_integration',
]
