"""
Mycelium Factor Graph

Energy-based inference for template selection and operand binding.
Uses ODE dynamics to converge beliefs over templates.

Key insight: Execution is the classifier. We try templates and see which
ones execute successfully on the available operands.
"""

import re
import inspect
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

import numpy as np
import sympy
from sympy import Symbol, sympify

from template_library import (
    TemplateLibrary, CLUSTER_TEMPLATE_MAP, SUPER_CATEGORIES,
    CLUSTER_TO_SUPER, get_template_function, get_cluster_templates
)


# ================================================================
# DATA STRUCTURES
# ================================================================

@dataclass
class Operand:
    """An operand extracted by C3 or from text."""
    value: Any  # SymPy expression, number, or Symbol
    position: int  # Token position in window (-1 for backref)
    is_backref: bool = False
    source: str = "c3"  # "c3", "text", "previous"


@dataclass
class Segment:
    """A reasoning segment from C1-A."""
    idx: int
    text: str
    operands: List[Operand] = field(default_factory=list)

    # Beliefs over templates (updated by factor graph)
    template_beliefs: Dict[str, float] = field(default_factory=dict)

    # Selected template and result after inference
    selected_template: Optional[str] = None
    result: Any = None
    execution_success: bool = False


@dataclass
class FactorGraphState:
    """State of the factor graph during inference."""
    segments: List[Segment]
    results: Dict[int, Any] = field(default_factory=dict)  # segment_idx -> result
    energy_history: List[float] = field(default_factory=list)
    converged: bool = False


# ================================================================
# OPERAND EXTRACTION
# ================================================================

def extract_numbers_from_text(text: str) -> List[Operand]:
    """Extract numeric operands from text, including LaTeX fractions."""
    operands = []
    seen_positions = set()

    # LaTeX fractions: \frac{a}{b} or \cfrac{a}{b}
    for match in re.finditer(r'\\c?frac\{(\d+)\}\{(\d+)\}', text):
        pos = match.start()
        if pos not in seen_positions:
            seen_positions.add(pos)
            try:
                num, den = int(match.group(1)), int(match.group(2))
                operands.append(Operand(
                    value=sympy.Rational(num, den),
                    position=pos,
                    is_backref=False,
                    source="text"
                ))
            except:
                pass

    # Match integers, decimals, fractions
    patterns = [
        (r'-?\d+\.\d+', 'decimal'),
        (r'-?\d+/\d+', 'fraction'),
        (r'(?<![a-zA-Z\d])-?\d+(?![a-zA-Z\d/])', 'integer'),  # Standalone integers
    ]

    for pattern, num_type in patterns:
        for match in re.finditer(pattern, text):
            pos = match.start()
            # Check we haven't already captured this position
            if any(abs(pos - p) < 3 for p in seen_positions):
                continue
            seen_positions.add(pos)

            try:
                val_str = match.group()
                if num_type == 'fraction':
                    num, den = val_str.split('/')
                    value = sympy.Rational(int(num), int(den))
                elif num_type == 'decimal':
                    value = sympy.Float(val_str)
                else:
                    value = sympy.Integer(val_str)

                operands.append(Operand(
                    value=value,
                    position=pos,
                    is_backref=False,
                    source="text"
                ))
            except:
                pass

    return operands


def extract_variables_from_text(text: str) -> List[Operand]:
    """Extract variable names from text."""
    operands = []

    # Match single letters that look like variables
    for match in re.finditer(r'\b([a-zA-Z])\b', text):
        var_name = match.group(1)
        # Skip common words
        if var_name.lower() in ['a', 'i', 'I']:
            continue
        operands.append(Operand(
            value=Symbol(var_name),
            position=match.start(),
            is_backref=False,
            source="text"
        ))

    return operands


def merge_operands(c3_operands: List[Operand], text_operands: List[Operand]) -> List[Operand]:
    """Merge C3 operands with text-extracted operands."""
    # Start with C3 operands
    merged = list(c3_operands)

    # Add text operands that don't overlap
    c3_positions = {op.position for op in c3_operands}
    for op in text_operands:
        if op.position not in c3_positions:
            merged.append(op)

    # Sort by position
    merged.sort(key=lambda x: x.position if x.position >= 0 else 999)
    return merged


# ================================================================
# ENERGY FUNCTIONS
# ================================================================

def execution_energy(template_name: str, operands: List[Any],
                     previous_results: Dict[int, Any]) -> Tuple[float, Any]:
    """
    Try executing a template with given operands.
    Returns (energy, result) where lower energy = better fit.
    """
    template_fn = get_template_function(template_name)
    if template_fn is None:
        return 10.0, None

    # Resolve operands (handle backrefs, filter None)
    resolved = []
    for op in operands:
        if isinstance(op, Operand):
            if op.is_backref and previous_results:
                recent_idx = max(previous_results.keys())
                resolved.append(previous_results[recent_idx])
            elif op.value is not None:
                resolved.append(op.value)
        elif op is not None:
            resolved.append(op)

    # Filter out None values
    resolved = [r for r in resolved if r is not None]

    if not resolved:
        return 9.0, None

    # Special handling for different template types
    try:
        # ARITHMETIC - needs 2 numbers (GOOD DEFAULT for most problems)
        if template_name == 'arithmetic':
            numbers = [r for r in resolved if hasattr(r, 'is_number') and r.is_number]
            if len(numbers) >= 2:
                # Try operations in order of likely usefulness
                for op in ['mul', 'div', 'add', 'sub']:
                    try:
                        result = template_fn(numbers[0], numbers[1], op)
                        if result is not None and hasattr(result, 'is_finite') and result.is_finite:
                            # Bonus for non-trivial results
                            if result != 0 and result != numbers[0] and result != numbers[1]:
                                return -3.0, result
                            return -1.5, result
                    except:
                        pass
            return 5.0, None  # Lower penalty - arithmetic is often useful

        # EVALUATE - can work with one operand
        elif template_name in ['evaluate_expression', 'compute_numeric', 'simplify_expression']:
            if resolved:
                result = template_fn(resolved[0])
                if result is not None:
                    if hasattr(result, 'is_number') and result.is_number:
                        return -3.0, result
                    return -1.5, result
            return 5.0, None

        # ASSIGN_VALUE - just returns the input
        elif template_name == 'assign_value':
            if resolved:
                return -1.0, resolved[0]
            return 5.0, None

        # SOLVE - needs actual equation structure, not just symbols
        elif template_name in ['solve_for_variable']:
            # Need an expression with both symbols AND numbers (an actual equation)
            exprs = [r for r in resolved if hasattr(r, 'free_symbols') and r.free_symbols
                     and hasattr(r, 'is_polynomial') and len(str(r)) > 3]
            symbols = [r for r in resolved if isinstance(r, Symbol)]

            if exprs and symbols:
                # Only if the expression is non-trivial (has numbers and symbols)
                expr = exprs[0]
                if expr.free_symbols and any(hasattr(r, 'is_number') and r.is_number
                                              for r in resolved):
                    try:
                        result = template_fn(expr, symbols[0])
                        if result and result != [0] and result != []:
                            return -3.0, result[0] if isinstance(result, list) else result
                    except:
                        pass
            return 7.0, None  # High penalty - don't use unless clearly needed

        # SUBSTITUTE - needs expression + variable + value
        elif template_name in ['substitute_value', 'substitute_expression']:
            if len(resolved) >= 2:
                # Try expr.subs(var, value) patterns
                exprs = [r for r in resolved if hasattr(r, 'subs')]
                values = [r for r in resolved if hasattr(r, 'is_number') and r.is_number]
                if exprs and values:
                    try:
                        result = exprs[0].subs(Symbol('x'), values[0])
                        return -2.0, result
                    except:
                        pass
            return 6.0, None

        # APPLY_PYTHAGOREAN - needs exactly 2-3 numbers
        elif template_name == 'apply_pythagorean':
            numbers = [r for r in resolved if hasattr(r, 'is_number') and r.is_number]
            if len(numbers) == 2:
                try:
                    result = template_fn(a=numbers[0], b=numbers[1])
                    if result is not None and hasattr(result, 'is_finite') and result.is_finite:
                        return -2.0, result
                except:
                    pass
            return 7.0, None  # High energy unless exactly right

        # QUADRATIC_FORMULA - needs exactly 3 numbers
        elif template_name == 'apply_quadratic_formula':
            numbers = [r for r in resolved if hasattr(r, 'is_number') and r.is_number]
            if len(numbers) >= 3:
                try:
                    result = template_fn(numbers[0], numbers[1], numbers[2])
                    if result:
                        return -3.0, result[0]
                except:
                    pass
            return 7.0, None

        # EXPAND/FACTOR - need symbolic expression
        elif template_name in ['expand_expression', 'factor_expression']:
            exprs = [r for r in resolved if hasattr(r, 'free_symbols') and r.free_symbols]
            if exprs:
                try:
                    result = template_fn(exprs[0])
                    return -2.0, result
                except:
                    pass
            return 6.0, None

        # DEFAULT - try generic execution
        else:
            sig = inspect.signature(template_fn)
            n_required = sum(1 for p in sig.parameters.values()
                             if p.default == inspect.Parameter.empty)

            if len(resolved) >= n_required:
                try:
                    result = template_fn(*resolved[:len(sig.parameters)])
                    if result is not None:
                        return -1.5, result
                except:
                    pass

            return 6.0, None

    except Exception as e:
        return 8.0, None


def type_compatibility_energy(template_name: str, operands: List[Operand]) -> float:
    """Score type compatibility between template expectations and operands."""
    energy = 0.0

    # Templates that expect numbers
    numeric_templates = ['arithmetic', 'compute_numeric', 'count_combinations',
                         'count_permutations', 'apply_pythagorean',
                         'apply_quadratic_formula', 'compute_gcd', 'compute_lcm']

    # Templates that expect expressions
    expr_templates = ['simplify_expression', 'expand_expression', 'factor_expression',
                      'solve_for_variable', 'substitute_value', 'evaluate_expression']

    has_numbers = any(hasattr(op.value, 'is_number') and op.value.is_number
                      for op in operands if isinstance(op, Operand))
    has_symbols = any(isinstance(op.value, Symbol)
                      for op in operands if isinstance(op, Operand))

    if template_name in numeric_templates and not has_numbers:
        energy += 2.0
    if template_name in expr_templates and not (has_numbers or has_symbols):
        energy += 1.0

    return energy


def context_energy(segment_idx: int, template_name: str,
                   segments: List[Segment], previous_results: Dict[int, Any]) -> float:
    """Score template based on context (neighboring segments, position)."""
    energy = 0.0
    n_segments = len(segments)

    # Common/simple templates get bonus everywhere
    simple_templates = ['arithmetic', 'evaluate_expression', 'compute_numeric',
                        'assign_value', 'simplify_expression']
    if template_name in simple_templates:
        energy -= 0.5

    # First segment: setup or direct computation
    if segment_idx == 0:
        setup_templates = ['assign_value', 'parse_equation', 'define_variable',
                           'arithmetic', 'evaluate_expression']
        if template_name in setup_templates:
            energy -= 1.0

    # Middle segments: computation templates
    if 0 < segment_idx < n_segments - 1:
        compute_templates = ['arithmetic', 'substitute_value', 'evaluate_expression',
                             'simplify_expression']
        if template_name in compute_templates:
            energy -= 0.5

    # Last segment: should produce final result
    if segment_idx == n_segments - 1:
        final_templates = ['evaluate_expression', 'compute_numeric', 'simplify_expression',
                           'arithmetic', 'solve_for_variable']
        if template_name in final_templates:
            energy -= 1.0

    # Penalize complex templates when simpler ones might work
    complex_templates = ['apply_pythagorean', 'apply_quadratic_formula', 'apply_vietas',
                         'solve_system', 'solve_inequality']
    if template_name in complex_templates:
        energy += 1.0  # Need strong evidence to use these

    # Templates needing previous results
    if template_name.startswith('substitute') or 'local_' in template_name:
        if previous_results:
            energy -= 0.5
        else:
            energy += 2.0  # Strong penalty if no previous results

    return energy


def total_energy(segment: Segment, template_name: str,
                 all_segments: List[Segment], previous_results: Dict[int, Any]) -> Tuple[float, Any]:
    """Compute total energy for a template choice."""

    # Execution energy (most important)
    exec_energy, result = execution_energy(
        template_name, segment.operands, previous_results
    )

    # Type compatibility
    type_energy = type_compatibility_energy(template_name, segment.operands)

    # Context energy
    ctx_energy = context_energy(segment.idx, template_name, all_segments, previous_results)

    # Weighted sum
    total = exec_energy * 2.0 + type_energy * 1.0 + ctx_energy * 0.5

    return total, result


# ================================================================
# ODE DYNAMICS
# ================================================================

def softmax(energies: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute softmax probabilities from negative energies."""
    # Negate energies (lower energy = higher probability)
    logits = -energies / temperature
    logits = logits - np.max(logits)  # numerical stability
    exp_logits = np.exp(logits)
    return exp_logits / (np.sum(exp_logits) + 1e-10)


def update_beliefs_ode(segment: Segment, all_segments: List[Segment],
                       previous_results: Dict[int, Any],
                       dt: float = 0.1, temperature: float = 1.0) -> Dict[str, Tuple[float, Any]]:
    """
    Single ODE step to update beliefs over templates.
    Returns dict mapping template_name -> (belief, result).
    """
    # Get candidate templates
    # Start with broad set, let energy guide selection
    candidate_templates = set()

    # Add templates from all clusters (we don't have C2)
    for cluster_name, cluster_info in CLUSTER_TEMPLATE_MAP.items():
        for template in cluster_info["primary"]:
            candidate_templates.add(template)

    # Also add common fallbacks
    candidate_templates.update(['arithmetic', 'evaluate_expression',
                                 'simplify_expression', 'assign_value'])

    # Compute energies for each template
    template_energies = {}
    template_results = {}

    for template_name in candidate_templates:
        energy, result = total_energy(segment, template_name, all_segments, previous_results)
        template_energies[template_name] = energy
        template_results[template_name] = result

    # Convert to arrays for softmax
    templates = list(template_energies.keys())
    energies = np.array([template_energies[t] for t in templates])

    # Current beliefs (uniform if not set)
    if not segment.template_beliefs:
        current_beliefs = np.ones(len(templates)) / len(templates)
    else:
        current_beliefs = np.array([segment.template_beliefs.get(t, 1e-6) for t in templates])
        current_beliefs = current_beliefs / (current_beliefs.sum() + 1e-10)

    # Target distribution from energies
    target_beliefs = softmax(energies, temperature)

    # ODE update: move toward target
    new_beliefs = current_beliefs + dt * (target_beliefs - current_beliefs)
    new_beliefs = new_beliefs / (new_beliefs.sum() + 1e-10)

    # Update segment beliefs
    segment.template_beliefs = {t: float(b) for t, b in zip(templates, new_beliefs)}

    return {t: (segment.template_beliefs[t], template_results[t]) for t in templates}


def run_ode_dynamics(state: FactorGraphState,
                     n_steps: int = 20,
                     dt: float = 0.1,
                     temperature_schedule: List[float] = None) -> FactorGraphState:
    """
    Run ODE dynamics to convergence.
    Temperature annealing: start high (exploration), end low (exploitation).
    """
    if temperature_schedule is None:
        # Anneal from 2.0 to 0.1
        temperature_schedule = [2.0 - 1.9 * (i / n_steps) for i in range(n_steps)]

    for step in range(n_steps):
        temperature = temperature_schedule[min(step, len(temperature_schedule) - 1)]
        total_energy = 0.0

        for segment in state.segments:
            # Get results from previous segments
            previous_results = {i: state.results[i] for i in state.results if i < segment.idx}

            # Update beliefs
            beliefs_and_results = update_beliefs_ode(
                segment, state.segments, previous_results, dt, temperature
            )

            # Track best template's energy
            if segment.template_beliefs:
                best_template = max(segment.template_beliefs, key=segment.template_beliefs.get)
                best_belief = segment.template_beliefs[best_template]

                # If confident enough, execute and store result
                if best_belief > 0.5:
                    _, result = beliefs_and_results.get(best_template, (0, None))
                    if result is not None:
                        state.results[segment.idx] = result
                        segment.result = result
                        segment.selected_template = best_template
                        segment.execution_success = True

                total_energy += -np.log(best_belief + 1e-10)

        state.energy_history.append(total_energy)

        # Check convergence
        if len(state.energy_history) > 3:
            recent = state.energy_history[-3:]
            if max(recent) - min(recent) < 0.01:
                state.converged = True
                break

    return state


# ================================================================
# INFERENCE PIPELINE
# ================================================================

def create_segments_from_windows(windows: List[Dict], c3_outputs: List[Dict]) -> List[Segment]:
    """Create Segment objects from C1-A windows and C3 outputs."""
    segments = []

    for i, (window, c3_out) in enumerate(zip(windows, c3_outputs)):
        text = window.get('text', '')

        # ALWAYS extract operands from text first (most reliable)
        text_operands = extract_numbers_from_text(text)
        text_operands.extend(extract_variables_from_text(text))

        # Check for C3 backrefs
        c3_operands = c3_out.get('operands', [])
        has_backref = any(op.get('is_backref', False) for op in c3_operands)

        # Build merged operand list
        merged = list(text_operands)  # Start with all text operands

        # Add backref if C3 detected one
        if has_backref:
            merged.insert(0, Operand(
                value=None,
                position=-1,
                is_backref=True,
                source="c3"
            ))

        # Sort by position (backrefs last)
        merged.sort(key=lambda x: x.position if x.position >= 0 else 999)

        segments.append(Segment(
            idx=i,
            text=text,
            operands=merged
        ))

    return segments


def run_factor_graph(windows: List[Dict], c3_outputs: List[Dict],
                     n_steps: int = 20) -> Tuple[Any, FactorGraphState]:
    """
    Run factor graph inference on a problem.
    Returns (final_result, state).
    """
    # Create segments
    segments = create_segments_from_windows(windows, c3_outputs)

    if not segments:
        return None, FactorGraphState(segments=[])

    # Initialize state
    state = FactorGraphState(segments=segments)

    # Run ODE dynamics
    state = run_ode_dynamics(state, n_steps=n_steps)

    # Get final result (from last successful segment)
    final_result = None
    for segment in reversed(state.segments):
        if segment.execution_success and segment.result is not None:
            final_result = segment.result
            break

    return final_result, state


# ================================================================
# EVALUATION HELPERS
# ================================================================

def evaluate_result(result: Any, ground_truth: str) -> Dict[str, Any]:
    """Compare factor graph result to ground truth."""
    if result is None:
        return {'correct': False, 'status': 'no_result'}

    try:
        # Parse ground truth
        gt_clean = ground_truth.strip()
        gt_clean = re.sub(r'\$([^$]+)\$', r'\1', gt_clean)
        gt_clean = gt_clean.replace('\\frac{', '(').replace('}{', ')/(').replace('}', ')')
        gt_clean = gt_clean.replace('\\cdot', '*').replace('\\times', '*')

        expected = sympify(gt_clean)

        # Compare
        diff = sympy.simplify(result - expected)
        if diff == 0:
            return {'correct': True, 'status': 'exact_match', 'result': str(result)}

        # Numeric comparison
        try:
            if abs(float(result.evalf()) - float(expected.evalf())) < 1e-6:
                return {'correct': True, 'status': 'numeric_match', 'result': str(result)}
        except:
            pass

        return {'correct': False, 'status': 'wrong_answer',
                'result': str(result), 'expected': str(expected)}

    except Exception as e:
        return {'correct': False, 'status': 'eval_error', 'error': str(e)}


def print_inference_trace(state: FactorGraphState):
    """Print detailed trace of factor graph inference."""
    print("\n" + "="*60)
    print("FACTOR GRAPH INFERENCE TRACE")
    print("="*60)

    for segment in state.segments:
        print(f"\nSegment {segment.idx}: {segment.text[:60]}...")
        print(f"  Operands: {[str(op.value) for op in segment.operands[:4]]}")

        if segment.template_beliefs:
            # Top 3 templates
            sorted_beliefs = sorted(segment.template_beliefs.items(),
                                   key=lambda x: -x[1])[:3]
            print(f"  Top templates:")
            for t, b in sorted_beliefs:
                print(f"    {t}: {b:.3f}")

        if segment.selected_template:
            print(f"  Selected: {segment.selected_template}")
            print(f"  Result: {segment.result}")

    print(f"\nConverged: {state.converged}")
    print(f"Energy history: {[f'{e:.2f}' for e in state.energy_history[-5:]]}")


if __name__ == "__main__":
    # Simple test
    windows = [
        {"text": "If x + 2 = 5, find x"},
        {"text": "Solve for x: x = 5 - 2 = 3"},
    ]
    c3_outputs = [
        {"operands": [{"position": 3, "is_backref": False},
                      {"position": 7, "is_backref": False}]},
        {"operands": [{"position": 15, "is_backref": False}]},
    ]

    result, state = run_factor_graph(windows, c3_outputs)
    print_inference_trace(state)
    print(f"\nFinal result: {result}")
