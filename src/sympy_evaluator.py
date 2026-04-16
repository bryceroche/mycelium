"""
SymPyEvaluator: Safe sandboxed execution of SymPy expressions.

This module provides a secure way to evaluate mathematical expressions
using SymPy, with protection against arbitrary code execution.
"""

import sympy
import signal
import threading
from contextlib import contextmanager
from typing import Dict, Any, Optional


class TimeoutError(Exception):
    """Raised when code execution exceeds the timeout."""
    pass


class SymPyEvaluator:
    """Safe, sandboxed SymPy expression evaluator."""

    # Allowed SymPy names that can be used in expressions
    ALLOWED_NAMES = {
        # Core types
        'Symbol', 'symbols', 'Rational', 'Integer', 'Float',
        # Arithmetic functions
        'sqrt', 'Abs', 'ceiling', 'floor', 'Mod',
        'log', 'ln', 'exp', 'pow',
        # Trigonometric
        'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
        # Simplification
        'simplify', 'expand', 'factor', 'collect',
        # Solving
        'solve', 'Eq',
        # Constants
        'pi', 'E', 'oo', 'I',
        # Summation and products
        'Sum', 'Product', 'factorial',
        # Rounding
        'N', 'nsimplify',
    }

    # Dangerous strings that should never appear in code
    DANGEROUS_PATTERNS = [
        'import ',      # Any import statement
        'from ',        # From imports
        '__',           # Dunder access (e.g., __builtins__, __class__)
        'exec(',        # Code execution
        'exec (',
        'eval(',        # Expression evaluation
        'eval (',
        'open(',        # File access
        'open (',
        'compile(',     # Code compilation
        'compile (',
        'globals(',     # Global namespace access
        'globals (',
        'locals(',      # Local namespace access
        'locals (',
        'getattr(',     # Attribute access
        'getattr (',
        'setattr(',     # Attribute modification
        'setattr (',
        'delattr(',     # Attribute deletion
        'delattr (',
        'hasattr(',     # Attribute checking (can be used for introspection)
        'hasattr (',
        'dir(',         # Directory listing
        'dir (',
        'vars(',        # Variable access
        'vars (',
        'type(',        # Type introspection
        'type (',
        'isinstance(',  # Type checking (could be used for introspection)
        'input(',       # User input
        'input (',
        'breakpoint(',  # Debugger
        'breakpoint (',
        'exit(',        # System exit
        'exit (',
        'quit(',        # System quit
        'quit (',
        'help(',        # Help system
        'help (',
        'memoryview(',  # Memory access
        'memoryview (',
        'bytearray(',   # Byte manipulation
        'bytearray (',
    ]

    @staticmethod
    def _is_safe(code_str: str) -> bool:
        """
        Check if the code string is safe to execute.

        Args:
            code_str: The code to check.

        Returns:
            True if the code appears safe, False otherwise.
        """
        code_lower = code_str.lower()

        for pattern in SymPyEvaluator.DANGEROUS_PATTERNS:
            if pattern.lower() in code_lower:
                return False

        return True

    @staticmethod
    def _build_namespace() -> Dict[str, Any]:
        """
        Build a restricted namespace with only allowed SymPy functions.

        Returns:
            Dictionary mapping allowed names to their SymPy implementations.
        """
        namespace = {}

        # Add allowed SymPy functions
        for name in SymPyEvaluator.ALLOWED_NAMES:
            if hasattr(sympy, name):
                namespace[name] = getattr(sympy, name)

        # Add common math operations that might be needed
        namespace['abs'] = abs  # Python's abs (for simple numbers)
        namespace['int'] = int  # Integer conversion
        namespace['float'] = float  # Float conversion
        namespace['round'] = round  # Rounding
        namespace['min'] = min  # Minimum
        namespace['max'] = max  # Maximum
        namespace['sum'] = sum  # Sum (Python's built-in for lists)
        namespace['len'] = len  # Length
        namespace['range'] = range  # Range
        namespace['True'] = True
        namespace['False'] = False
        namespace['None'] = None

        return namespace

    @staticmethod
    def safe_eval(code_str: str, timeout_sec: float = 5.0) -> Dict[str, float]:
        """
        Execute SymPy code safely and return variable bindings.

        Args:
            code_str: The SymPy code to execute.
            timeout_sec: Maximum execution time in seconds (default 5).

        Returns:
            Dictionary of {variable_name: numeric_value} for all computed values.
            Returns empty dict on any failure (safe fallback).
        """
        # Basic sanitization
        if not code_str or not isinstance(code_str, str):
            return {}

        code_str = code_str.strip()
        if not code_str:
            return {}

        # Check for dangerous patterns
        if not SymPyEvaluator._is_safe(code_str):
            return {}

        # Build restricted namespace
        namespace = SymPyEvaluator._build_namespace()

        # Result container for thread
        result_container = {'result': {}, 'error': None}

        def execute_code():
            try:
                # Execute with restricted builtins
                exec(code_str, {"__builtins__": {}}, namespace)

                # Extract numeric results
                results = {}
                skip_names = set(SymPyEvaluator.ALLOWED_NAMES) | {
                    'abs', 'int', 'float', 'round', 'min', 'max',
                    'sum', 'len', 'range', 'True', 'False', 'None'
                }

                for name, value in namespace.items():
                    # Skip internal names, builtins, and allowed names
                    if name.startswith('_') or name in skip_names:
                        continue

                    # Try to convert to numeric value
                    try:
                        # First, try direct float conversion
                        numeric = float(value)
                        results[name] = numeric
                    except (TypeError, ValueError):
                        # Try SymPy's N() for symbolic expressions
                        try:
                            numeric = float(sympy.N(value))
                            results[name] = numeric
                        except Exception:
                            # Skip non-numeric values
                            pass

                result_container['result'] = results

            except Exception as e:
                result_container['error'] = str(e)

        # Execute in a thread with timeout
        thread = threading.Thread(target=execute_code)
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout_sec)

        if thread.is_alive():
            # Timeout occurred - thread is still running
            # Note: We can't forcefully kill the thread in Python,
            # but the daemon flag ensures it won't block program exit
            return {}

        if result_container['error']:
            return {}

        return result_container['result']


def test_sympy_evaluator():
    """Test the SymPyEvaluator with various cases."""

    print("Testing SymPyEvaluator...")
    print("=" * 50)

    # Test 1: Simple arithmetic
    print("\nTest 1: Simple arithmetic")
    result = SymPyEvaluator.safe_eval("x = 48 / 2")
    expected = {'x': 24.0}
    assert result == expected, f"Expected {expected}, got {result}"
    print(f"  'x = 48 / 2' -> {result}")
    print("  PASSED")

    # Test 2: Multi-variable
    print("\nTest 2: Multi-variable")
    result = SymPyEvaluator.safe_eval("a = 10; b = a * 3")
    expected = {'a': 10.0, 'b': 30.0}
    assert result == expected, f"Expected {expected}, got {result}"
    print(f"  'a = 10; b = a * 3' -> {result}")
    print("  PASSED")

    # Test 3: SymPy functions
    print("\nTest 3: SymPy functions (sqrt)")
    result = SymPyEvaluator.safe_eval("x = sqrt(16)")
    expected = {'x': 4.0}
    assert result == expected, f"Expected {expected}, got {result}"
    print(f"  'x = sqrt(16)' -> {result}")
    print("  PASSED")

    # Test 4: Dangerous code rejection - import os
    print("\nTest 4: Dangerous code rejection - import os")
    result = SymPyEvaluator.safe_eval("import os; x = os.system('echo hacked')")
    expected = {}
    assert result == expected, f"Expected {expected}, got {result}"
    print(f"  'import os; ...' -> {result}")
    print("  PASSED")

    # Test 5: Dangerous code rejection - import sys
    print("\nTest 5: Dangerous code rejection - import sys")
    result = SymPyEvaluator.safe_eval("import sys")
    expected = {}
    assert result == expected, f"Expected {expected}, got {result}"
    print(f"  'import sys' -> {result}")
    print("  PASSED")

    # Test 6: Dangerous code rejection - exec
    print("\nTest 6: Dangerous code rejection - exec")
    result = SymPyEvaluator.safe_eval("exec('x = 1')")
    expected = {}
    assert result == expected, f"Expected {expected}, got {result}"
    print(f"  \"exec('x = 1')\" -> {result}")
    print("  PASSED")

    # Test 7: Dangerous code rejection - eval
    print("\nTest 7: Dangerous code rejection - eval")
    result = SymPyEvaluator.safe_eval("x = eval('1 + 1')")
    expected = {}
    assert result == expected, f"Expected {expected}, got {result}"
    print(f"  \"x = eval('1 + 1')\" -> {result}")
    print("  PASSED")

    # Test 8: Dangerous code rejection - dunder access
    print("\nTest 8: Dangerous code rejection - dunder access")
    result = SymPyEvaluator.safe_eval("x = ().__class__.__bases__[0]")
    expected = {}
    assert result == expected, f"Expected {expected}, got {result}"
    print(f"  'x = ().__class__...' -> {result}")
    print("  PASSED")

    # Test 9: Dangerous code rejection - open
    print("\nTest 9: Dangerous code rejection - open")
    result = SymPyEvaluator.safe_eval("f = open('/etc/passwd', 'r')")
    expected = {}
    assert result == expected, f"Expected {expected}, got {result}"
    print(f"  \"f = open('/etc/passwd', 'r')\" -> {result}")
    print("  PASSED")

    # Test 10: Timeout handling - infinite loop
    print("\nTest 10: Timeout handling - infinite loop")
    result = SymPyEvaluator.safe_eval("while True: pass", timeout_sec=1.0)
    expected = {}
    assert result == expected, f"Expected {expected}, got {result}"
    print(f"  'while True: pass' (1s timeout) -> {result}")
    print("  PASSED")

    # Test 11: More complex arithmetic
    print("\nTest 11: Complex arithmetic chain")
    result = SymPyEvaluator.safe_eval("""
n_april = 48
n_may = n_april / 2
total = n_april + n_may
""")
    expected = {'n_april': 48.0, 'n_may': 24.0, 'total': 72.0}
    assert result == expected, f"Expected {expected}, got {result}"
    print(f"  Multi-line calculation -> {result}")
    print("  PASSED")

    # Test 12: SymPy Rational (exact fractions)
    print("\nTest 12: SymPy Rational")
    result = SymPyEvaluator.safe_eval("x = Rational(1, 3) * 9")
    expected = {'x': 3.0}
    assert result == expected, f"Expected {expected}, got {result}"
    print(f"  'x = Rational(1, 3) * 9' -> {result}")
    print("  PASSED")

    # Test 13: SymPy Abs
    print("\nTest 13: SymPy Abs")
    result = SymPyEvaluator.safe_eval("x = Abs(-5)")
    expected = {'x': 5.0}
    assert result == expected, f"Expected {expected}, got {result}"
    print(f"  'x = Abs(-5)' -> {result}")
    print("  PASSED")

    # Test 14: Empty input
    print("\nTest 14: Empty input")
    result = SymPyEvaluator.safe_eval("")
    expected = {}
    assert result == expected, f"Expected {expected}, got {result}"
    print(f"  '' -> {result}")
    print("  PASSED")

    # Test 15: Syntax error handling
    print("\nTest 15: Syntax error handling")
    result = SymPyEvaluator.safe_eval("x = = 5")
    expected = {}
    assert result == expected, f"Expected {expected}, got {result}"
    print(f"  'x = = 5' (syntax error) -> {result}")
    print("  PASSED")

    # Test 16: Floor and ceiling
    print("\nTest 16: Floor and ceiling")
    result = SymPyEvaluator.safe_eval("a = floor(3.7); b = ceiling(3.2)")
    expected = {'a': 3.0, 'b': 4.0}
    assert result == expected, f"Expected {expected}, got {result}"
    print(f"  'a = floor(3.7); b = ceiling(3.2)' -> {result}")
    print("  PASSED")

    # Test 17: Mod operation
    print("\nTest 17: Mod operation")
    result = SymPyEvaluator.safe_eval("x = Mod(17, 5)")
    expected = {'x': 2.0}
    assert result == expected, f"Expected {expected}, got {result}"
    print(f"  'x = Mod(17, 5)' -> {result}")
    print("  PASSED")

    print("\n" + "=" * 50)
    print("All tests PASSED!")
    print("=" * 50)


if __name__ == "__main__":
    test_sympy_evaluator()
