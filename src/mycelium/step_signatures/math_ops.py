"""Pure Math Operations for DSL Execution.

These are domain-specific math functions used by the DSL executor.
All functions are pure and have no dependencies on the rest of the system.
"""

import math
from typing import Optional


# =============================================================================
# Algebra Functions
# =============================================================================

def extract_coefficient(expr: str, var: str = "x") -> Optional[float]:
    """Extract coefficient of variable from expression."""
    try:
        import sympy
        x = sympy.Symbol(var)
        parsed = sympy.sympify(expr)
        coeff = parsed.coeff(x)
        return float(coeff) if coeff.is_number else None
    except Exception:
        return None


def apply_quadratic_formula(a: float, b: float, c: float) -> Optional[tuple[float, float]]:
    """Apply quadratic formula to solve ax^2 + bx + c = 0."""
    if a == 0:
        return None  # Not quadratic
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return None  # No real roots
    sqrt_d = math.sqrt(discriminant)
    return ((-b + sqrt_d) / (2*a), (-b - sqrt_d) / (2*a))


def complete_square(a: float, b: float, c: float) -> Optional[str]:
    """Complete the square for ax^2 + bx + c.

    Returns string in form: a(x + h)^2 + k
    """
    if a == 0:
        return None  # Not quadratic
    h = -b / (2*a)
    k = c - b**2 / (4*a)
    return f"{a}*(x + {h})^2 + {k}"


def solve_linear(a: float, b: float) -> Optional[float]:
    """Solve ax + b = 0 for x."""
    if a == 0:
        return None
    return -b / a


def evaluate_polynomial(coeffs: list[float], x: float) -> float:
    """Evaluate polynomial with coefficients [a_n, ..., a_1, a_0] at x."""
    result = 0.0
    for coeff in coeffs:
        result = result * x + coeff
    return result


# =============================================================================
# Number Theory Functions
# =============================================================================

def euclidean_gcd(a: int, b: int) -> int:
    """Euclidean algorithm for GCD."""
    a, b = int(abs(a)), int(abs(b))
    while b:
        a, b = b, a % b
    return a


def modinv(a: int, m: int) -> Optional[int]:
    """Modular multiplicative inverse of a mod m using extended Euclidean algorithm."""
    a, m = int(a), int(m)
    if m == 1:
        return 0
    m0, x0, x1 = m, 0, 1
    while a > 1:
        q = a // m
        m, a = a % m, m
        x0, x1 = x1 - q * x0, x0
    return x1 + m0 if x1 < 0 else x1


def divisors(n: int) -> list[int]:
    """Find all divisors of n."""
    n = int(abs(n))
    if n == 0:
        return []
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs)


def divisor_count(n: int) -> int:
    """Count the number of divisors of n.

    For n = p1^a1 * p2^a2 * ... * pk^ak,
    count = (a1+1) * (a2+1) * ... * (ak+1)
    """
    n = int(abs(n))
    if n == 0:
        return 0
    if n == 1:
        return 1

    # Use the formula based on prime factorization
    count = 1
    d = 2
    while d * d <= n:
        exp = 0
        while n % d == 0:
            exp += 1
            n //= d
        if exp > 0:
            count *= (exp + 1)
        d += 1
    if n > 1:  # Remaining prime factor
        count *= 2
    return count


def factorization_exponents(n: int) -> dict[int, int]:
    """Get prime factorization as {prime: exponent} dict.

    Example: factorization_exponents(196) -> {2: 2, 7: 2}
    """
    n = int(abs(n))
    exponents = {}
    d = 2
    while d * d <= n:
        exp = 0
        while n % d == 0:
            exp += 1
            n //= d
        if exp > 0:
            exponents[d] = exp
        d += 1
    if n > 1:
        exponents[n] = 1
    return exponents


def divisor_count_from_factors(factors) -> int:
    """Count divisors from a list of prime factors.

    Takes a list like [2, 2, 7, 7] (from prime_factors) and returns
    the divisor count using the formula (a1+1)(a2+1)...(ak+1).

    Example: divisor_count_from_factors([2, 2, 7, 7]) -> 9
    (2^2 * 7^2 has (2+1)(2+1) = 9 divisors)

    Also handles:
    - String representation like '[2, 2, 7, 7]'
    - Dict representation like {2: 2, 7: 2}
    """
    # Handle string input (from step context)
    if isinstance(factors, str):
        factors = factors.strip()
        # Try to parse as list
        if factors.startswith('[') and factors.endswith(']'):
            import ast
            try:
                factors = ast.literal_eval(factors)
            except (ValueError, SyntaxError):
                return 0
        # Try to parse as dict
        elif factors.startswith('{') and factors.endswith('}'):
            import ast
            try:
                exponents = ast.literal_eval(factors)
                if isinstance(exponents, dict):
                    count = 1
                    for exp in exponents.values():
                        count *= (int(exp) + 1)
                    return count
            except (ValueError, SyntaxError):
                return 0
        else:
            return 0

    # Handle dict input (from factorization_exponents)
    if isinstance(factors, dict):
        count = 1
        for exp in factors.values():
            count *= (int(exp) + 1)
        return count

    # Handle list input (from prime_factors)
    if not isinstance(factors, (list, tuple)):
        return 0

    if not factors:
        return 1  # Empty factorization = 1 (divisors of 1)

    # Count occurrences of each prime
    from collections import Counter
    exponents = Counter(factors)

    # Apply formula: (a1+1)(a2+1)...(ak+1)
    count = 1
    for exp in exponents.values():
        count *= (exp + 1)
    return count


def prime_factors(n: int) -> list[int]:
    """Find prime factorization of n."""
    n = int(abs(n))
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def is_prime(n: int) -> bool:
    """Check if n is prime."""
    n = int(n)
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def mod_pow(base: int, exp: int, mod: int) -> int:
    """Modular exponentiation: base^exp mod mod."""
    base, exp, mod = int(base), int(exp), int(mod)
    result = 1
    base = base % mod
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        exp = exp >> 1
        base = (base * base) % mod
    return result


# =============================================================================
# Base Conversion Functions
# =============================================================================

def int_to_base(n: int, base: int) -> str:
    """Convert integer n to string representation in given base."""
    n, base = int(n), int(base)
    if n == 0:
        return "0"
    if base < 2 or base > 36:
        raise ValueError(f"Base must be 2-36, got {base}")

    negative = n < 0
    n = abs(n)
    digits = []
    while n:
        remainder = n % base
        if remainder < 10:
            digits.append(str(remainder))
        else:
            digits.append(chr(ord('a') + remainder - 10))
        n //= base

    result = ''.join(reversed(digits))
    return '-' + result if negative else result


def clean_base_input(s) -> str:
    """Clean input for base conversion - handle floats like 2012.0."""
    s_str = str(s)
    if '.' in s_str:
        return str(int(float(s)))
    return s_str.strip()


def from_base(s, base: int) -> int:
    """Convert string s from given base to integer."""
    return int(clean_base_input(s), int(base))


def base_multiply(a, b, base: int) -> str:
    """Multiply two numbers in given base, return result in same base."""
    a_dec = int(clean_base_input(a), int(base))
    b_dec = int(clean_base_input(b), int(base))
    product = a_dec * b_dec
    return int_to_base(product, int(base))


def base_add(a, b, base: int) -> str:
    """Add two numbers in given base, return result in same base."""
    a_dec = int(clean_base_input(a), int(base))
    b_dec = int(clean_base_input(b), int(base))
    return int_to_base(a_dec + b_dec, int(base))


# =============================================================================
# Combinatorics Functions
# =============================================================================

def binomial(n: int, k: int) -> int:
    """Binomial coefficient C(n, k)."""
    n, k = int(n), int(k)
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def permutations(n: int, r: int) -> int:
    """Permutations P(n, r) = n! / (n-r)!"""
    n, r = int(n), int(r)
    if r < 0 or r > n:
        return 0
    result = 1
    for i in range(n, n - r, -1):
        result *= i
    return result


def combinations(n: int, r: int) -> int:
    """Combinations C(n, r) = n! / (r! * (n-r)!)"""
    return binomial(n, r)


# =============================================================================
# Miscellaneous Functions
# =============================================================================

def day_of_week(year: int, month: int, day: int) -> int:
    """Zeller's formula: 0=Saturday, 1=Sunday, ..., 6=Friday."""
    year, month, day = int(year), int(month), int(day)
    if month < 3:
        month += 12
        year -= 1
    k = year % 100
    j = year // 100
    h = (day + (13 * (month + 1)) // 5 + k + k // 4 + j // 4 - 2 * j) % 7
    return h


def triangular_number(n: int) -> int:
    """nth triangular number: 1 + 2 + ... + n = n(n+1)/2."""
    n = int(n)
    return n * (n + 1) // 2


def fibonacci(n: int) -> int:
    """nth Fibonacci number (0-indexed: F(0)=0, F(1)=1)."""
    n = int(n)
    if n < 0:
        return 0
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
