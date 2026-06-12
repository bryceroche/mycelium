"""v200 legacy functional API — re-exported from factor_graph_v200.

Stage 1B namespace hygiene: the training driver (scripts/v200_train.py) uses
the functional API (attach_fg_params_v200, fg_breathing_forward_v200, etc.)
which lives in the top half of factor_graph_v200.py alongside the new class
API (FactorGraphV200, V200Config, etc.).

Import this module when you need the functional/legacy API; import
mycelium.factor_graph_v200 when you need the class API. Both work; this
module makes the intent explicit and keeps the training driver's imports
unambiguous for Stage 1C.

All names re-exported below are the exact names scripts/v200_train.py imports.
"""

from mycelium.factor_graph_v200 import (
    # Constants
    V200_K_MAX,
    V200_N_LATENTS,
    V200_N_VAR_LAT,
    V200_N_DIGITS,
    V200_N_MAX,
    V200_F_MAX,
    V200_T_MAX,
    V200_CALIB_WEIGHT,
    V200_FACTOR_AUX,
    V200_STAGE2A_WAIST,
    V200_WAIST_DIM,
    # Helpers
    _fourier_orthogonal_init,
    _cross_attend_v200,
    _apply_waist_v200,
    _embed_fg_tokens_v200,
    # Functional API
    attach_fg_params_v200,
    fg_v200_parameters,
    fg_v200_state_dict,
    fg_breathing_forward_v200,
    fg_accuracy_v200,
    compute_drift_v200,
    _compile_jit_fg_step_v200,
    compile_jit_eval_v200,
)

__all__ = [
    "V200_K_MAX", "V200_N_LATENTS", "V200_N_VAR_LAT", "V200_N_DIGITS",
    "V200_N_MAX", "V200_F_MAX", "V200_T_MAX", "V200_CALIB_WEIGHT",
    "V200_FACTOR_AUX", "V200_STAGE2A_WAIST", "V200_WAIST_DIM",
    "_fourier_orthogonal_init", "_cross_attend_v200",
    "_apply_waist_v200", "_embed_fg_tokens_v200",
    "attach_fg_params_v200", "fg_v200_parameters", "fg_v200_state_dict",
    "fg_breathing_forward_v200", "fg_accuracy_v200", "compute_drift_v200",
    "_compile_jit_fg_step_v200", "compile_jit_eval_v200",
]
