"""
Function Registry - Curated Python math function pointers organized by tier.

This module provides a centralized registry of mathematical functions that can be
used throughout the mycelium system for DSL execution and computation graphs.
"""

import operator
import math
import statistics
from typing import Any, Callable, List, Optional, Union

# Type alias for numbers
Number = Union[int, float]


# =============================================================================
# FLEXIBLE ARITHMETIC FUNCTIONS (handle variable args)
# =============================================================================

def _add(*args) -> Number:
    """Add any combination of numbers - handles args or list."""
    if len(args) == 0:
        return 0
    # If single arg that's a list/tuple, sum it
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        return sum(args[0])
    return sum(args)


def _mul(*args) -> Number:
    """Multiply any combination of numbers - handles args or list."""
    if len(args) == 0:
        return 1
    # If single arg that's a list/tuple, multiply all
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        args = args[0]
    result = 1
    for x in args:
        result *= x
    return result

# Try to import sympy for symbolic operations (tier 7)
try:
    import sympy
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False


# =============================================================================
# FUNCTION REGISTRY
# =============================================================================

FUNCTION_REGISTRY = {
    # =========================================================================
    # TIER 1: Arithmetic
    # =========================================================================
    "add": {
        "func": _add,
        "arity": -1,  # Variable arity
        "tier": 1,
        "module": "custom",
        "description": "Add two or more numbers",
    },
    "sub": {
        "func": operator.sub,
        "arity": 2,
        "tier": 1,
        "module": "operator",
        "description": "Subtract second number from first",
    },
    "mul": {
        "func": _mul,
        "arity": -1,  # Variable arity
        "tier": 1,
        "module": "custom",
        "description": "Multiply two or more numbers",
    },
    "truediv": {
        "func": operator.truediv,
        "arity": 2,
        "tier": 1,
        "module": "operator",
        "description": "Divide first number by second (true division)",
    },
    "floordiv": {
        "func": operator.floordiv,
        "arity": 2,
        "tier": 1,
        "module": "operator",
        "description": "Divide first number by second (floor division)",
    },
    "mod": {
        "func": operator.mod,
        "arity": 2,
        "tier": 1,
        "module": "operator",
        "description": "Modulo (remainder) of first number divided by second",
    },
    "pow": {
        "func": operator.pow,
        "arity": 2,
        "tier": 1,
        "module": "operator",
        "description": "Raise first number to the power of second",
    },
    "neg": {
        "func": operator.neg,
        "arity": 1,
        "tier": 1,
        "module": "operator",
        "description": "Negate a number",
    },
    "abs": {
        "func": abs,
        "arity": 1,
        "tier": 1,
        "module": "builtins",
        "description": "Absolute value of a number",
    },
    "sqrt": {
        "func": math.sqrt,
        "arity": 1,
        "tier": 1,
        "module": "math",
        "description": "Square root of a number",
    },
    "cbrt": {
        "func": math.cbrt,
        "arity": 1,
        "tier": 1,
        "module": "math",
        "description": "Cube root of a number",
    },
    "floor": {
        "func": math.floor,
        "arity": 1,
        "tier": 1,
        "module": "math",
        "description": "Floor of a number (largest integer <= x)",
    },
    "ceil": {
        "func": math.ceil,
        "arity": 1,
        "tier": 1,
        "module": "math",
        "description": "Ceiling of a number (smallest integer >= x)",
    },
    "trunc": {
        "func": math.trunc,
        "arity": 1,
        "tier": 1,
        "module": "math",
        "description": "Truncate a number (integer part only)",
    },

    # =========================================================================
    # TIER 2: Comparison
    # =========================================================================
    "eq": {
        "func": operator.eq,
        "arity": 2,
        "tier": 2,
        "module": "operator",
        "description": "Test equality of two values",
    },
    "ne": {
        "func": operator.ne,
        "arity": 2,
        "tier": 2,
        "module": "operator",
        "description": "Test inequality of two values",
    },
    "lt": {
        "func": operator.lt,
        "arity": 2,
        "tier": 2,
        "module": "operator",
        "description": "Test if first value is less than second",
    },
    "le": {
        "func": operator.le,
        "arity": 2,
        "tier": 2,
        "module": "operator",
        "description": "Test if first value is less than or equal to second",
    },
    "gt": {
        "func": operator.gt,
        "arity": 2,
        "tier": 2,
        "module": "operator",
        "description": "Test if first value is greater than second",
    },
    "ge": {
        "func": operator.ge,
        "arity": 2,
        "tier": 2,
        "module": "operator",
        "description": "Test if first value is greater than or equal to second",
    },
    "max": {
        "func": max,
        "arity": -1,  # variadic
        "tier": 2,
        "module": "builtins",
        "description": "Return the maximum of the given values",
    },
    "min": {
        "func": min,
        "arity": -1,  # variadic
        "tier": 2,
        "module": "builtins",
        "description": "Return the minimum of the given values",
    },

    # =========================================================================
    # TIER 3: Trigonometry
    # =========================================================================
    "sin": {
        "func": math.sin,
        "arity": 1,
        "tier": 3,
        "module": "math",
        "description": "Sine of angle (in radians)",
    },
    "cos": {
        "func": math.cos,
        "arity": 1,
        "tier": 3,
        "module": "math",
        "description": "Cosine of angle (in radians)",
    },
    "tan": {
        "func": math.tan,
        "arity": 1,
        "tier": 3,
        "module": "math",
        "description": "Tangent of angle (in radians)",
    },
    "asin": {
        "func": math.asin,
        "arity": 1,
        "tier": 3,
        "module": "math",
        "description": "Arc sine (inverse sine), returns radians",
    },
    "acos": {
        "func": math.acos,
        "arity": 1,
        "tier": 3,
        "module": "math",
        "description": "Arc cosine (inverse cosine), returns radians",
    },
    "atan": {
        "func": math.atan,
        "arity": 1,
        "tier": 3,
        "module": "math",
        "description": "Arc tangent (inverse tangent), returns radians",
    },
    "atan2": {
        "func": math.atan2,
        "arity": 2,
        "tier": 3,
        "module": "math",
        "description": "Arc tangent of y/x, returns radians in correct quadrant",
    },
    "degrees": {
        "func": math.degrees,
        "arity": 1,
        "tier": 3,
        "module": "math",
        "description": "Convert radians to degrees",
    },
    "radians": {
        "func": math.radians,
        "arity": 1,
        "tier": 3,
        "module": "math",
        "description": "Convert degrees to radians",
    },
    "hypot": {
        "func": math.hypot,
        "arity": -1,  # variadic in Python 3.8+
        "tier": 3,
        "module": "math",
        "description": "Euclidean distance (hypotenuse) from origin",
    },

    # =========================================================================
    # TIER 4: Logarithms/Exponentials
    # =========================================================================
    "log": {
        "func": math.log,
        "arity": -1,  # 1 or 2 args (value, optional base)
        "tier": 4,
        "module": "math",
        "description": "Natural logarithm (or log with given base)",
    },
    "log10": {
        "func": math.log10,
        "arity": 1,
        "tier": 4,
        "module": "math",
        "description": "Base-10 logarithm",
    },
    "log2": {
        "func": math.log2,
        "arity": 1,
        "tier": 4,
        "module": "math",
        "description": "Base-2 logarithm",
    },
    "exp": {
        "func": math.exp,
        "arity": 1,
        "tier": 4,
        "module": "math",
        "description": "e raised to the power x",
    },
    "exp2": {
        "func": math.exp2,
        "arity": 1,
        "tier": 4,
        "module": "math",
        "description": "2 raised to the power x",
    },

    # =========================================================================
    # TIER 5: Number Theory
    # =========================================================================
    "gcd": {
        "func": math.gcd,
        "arity": -1,  # variadic in Python 3.9+
        "tier": 5,
        "module": "math",
        "description": "Greatest common divisor",
    },
    "lcm": {
        "func": math.lcm,
        "arity": -1,  # variadic in Python 3.9+
        "tier": 5,
        "module": "math",
        "description": "Least common multiple",
    },
    "factorial": {
        "func": math.factorial,
        "arity": 1,
        "tier": 5,
        "module": "math",
        "description": "Factorial of a non-negative integer",
    },
    "comb": {
        "func": math.comb,
        "arity": 2,
        "tier": 5,
        "module": "math",
        "description": "Number of combinations (n choose k)",
    },
    "perm": {
        "func": math.perm,
        "arity": 2,
        "tier": 5,
        "module": "math",
        "description": "Number of permutations (n permute k)",
    },
    "isqrt": {
        "func": math.isqrt,
        "arity": 1,
        "tier": 5,
        "module": "math",
        "description": "Integer square root (floor of sqrt)",
    },

    # =========================================================================
    # TIER 6: Statistics
    # =========================================================================
    "mean": {
        "func": statistics.mean,
        "arity": 1,  # takes iterable
        "tier": 6,
        "module": "statistics",
        "description": "Arithmetic mean of data",
    },
    "median": {
        "func": statistics.median,
        "arity": 1,  # takes iterable
        "tier": 6,
        "module": "statistics",
        "description": "Median (middle value) of data",
    },
    "mode": {
        "func": statistics.mode,
        "arity": 1,  # takes iterable
        "tier": 6,
        "module": "statistics",
        "description": "Mode (most common value) of data",
    },
    "stdev": {
        "func": statistics.stdev,
        "arity": 1,  # takes iterable
        "tier": 6,
        "module": "statistics",
        "description": "Sample standard deviation of data",
    },
    "variance": {
        "func": statistics.variance,
        "arity": 1,  # takes iterable
        "tier": 6,
        "module": "statistics",
        "description": "Sample variance of data",
    },
    "sum": {
        "func": _add,  # Same as add - handles both args and list
        "arity": -1,
        "tier": 6,
        "module": "custom",
        "description": "Sum of all values",
    },
    "len": {
        "func": len,
        "arity": 1,  # takes iterable
        "tier": 6,
        "module": "builtins",
        "description": "Number of items in an iterable",
    },
}

# =========================================================================
# TIER 7: Symbolic (optional, requires sympy)
# =========================================================================
if SYMPY_AVAILABLE:
    FUNCTION_REGISTRY.update({
        "solve": {
            "func": sympy.solve,
            "arity": -1,  # variadic
            "tier": 7,
            "module": "sympy",
            "description": "Solve algebraic equations",
        },
        "simplify": {
            "func": sympy.simplify,
            "arity": 1,
            "tier": 7,
            "module": "sympy",
            "description": "Simplify a symbolic expression",
        },
        "expand": {
            "func": sympy.expand,
            "arity": 1,
            "tier": 7,
            "module": "sympy",
            "description": "Expand a symbolic expression",
        },
        "factor": {
            "func": sympy.factor,
            "arity": 1,
            "tier": 7,
            "module": "sympy",
            "description": "Factor a symbolic expression",
        },
        "diff": {
            "func": sympy.diff,
            "arity": -1,  # variadic
            "tier": 7,
            "module": "sympy",
            "description": "Differentiate a symbolic expression",
        },
        "integrate": {
            "func": sympy.integrate,
            "arity": -1,  # variadic
            "tier": 7,
            "module": "sympy",
            "description": "Integrate a symbolic expression",
        },
        "limit": {
            "func": sympy.limit,
            "arity": 3,
            "tier": 7,
            "module": "sympy",
            "description": "Compute the limit of a symbolic expression",
        },
    })


# =============================================================================
# TIER 8: Percentages & Ratios (Word Problems)
# =============================================================================

def _percent_of(percent, whole):
    """Calculate percent of a whole. E.g., 20% of 50 = 10."""
    return (percent / 100) * whole

def _what_percent(part, whole):
    """What percent is part of whole?"""
    if whole == 0:
        return None
    return (part / whole) * 100

def _percent_change(old, new):
    """Percent change from old to new."""
    if old == 0:
        return None
    return ((new - old) / old) * 100

def _percent_increase(value, percent):
    """Increase value by percent. E.g., 100 increased by 20% = 120."""
    return value * (1 + percent / 100)

def _percent_decrease(value, percent):
    """Decrease value by percent. E.g., 100 decreased by 20% = 80."""
    return value * (1 - percent / 100)

def _ratio(a, b):
    """Ratio a:b as a decimal."""
    if b == 0:
        return None
    return a / b

def _proportion_solve(a, b, c):
    """Solve a/b = c/x for x."""
    if a == 0:
        return None
    return (b * c) / a

def _remaining_after(total, spent):
    """What remains after spending."""
    return total - spent

def _split_equally(total, parts):
    """Split total into equal parts."""
    if parts == 0:
        return None
    return total / parts

def _combine_parts(part, count):
    """Combine multiple equal parts. E.g., 5 items at $3 each = $15."""
    return part * count


FUNCTION_REGISTRY.update({
    "percent_of": {
        "func": _percent_of,
        "arity": 2,
        "tier": 8,
        "module": "local",
        "description": "Calculate percent of a whole (e.g., 20% of 50)",
    },
    "what_percent": {
        "func": _what_percent,
        "arity": 2,
        "tier": 8,
        "module": "local",
        "description": "What percent is part of whole",
    },
    "percent_change": {
        "func": _percent_change,
        "arity": 2,
        "tier": 8,
        "module": "local",
        "description": "Percent change from old to new value",
    },
    "percent_increase": {
        "func": _percent_increase,
        "arity": 2,
        "tier": 8,
        "module": "local",
        "description": "Increase value by a percentage",
    },
    "percent_decrease": {
        "func": _percent_decrease,
        "arity": 2,
        "tier": 8,
        "module": "local",
        "description": "Decrease value by a percentage",
    },
    "ratio": {
        "func": _ratio,
        "arity": 2,
        "tier": 8,
        "module": "local",
        "description": "Ratio a:b as a decimal",
    },
    "proportion_solve": {
        "func": _proportion_solve,
        "arity": 3,
        "tier": 8,
        "module": "local",
        "description": "Solve proportion a/b = c/x for x",
    },
    "remaining_after": {
        "func": _remaining_after,
        "arity": 2,
        "tier": 8,
        "module": "local",
        "description": "What remains after subtracting (total - spent)",
    },
    "split_equally": {
        "func": _split_equally,
        "arity": 2,
        "tier": 8,
        "module": "local",
        "description": "Split total into equal parts",
    },
    "combine_parts": {
        "func": _combine_parts,
        "arity": 2,
        "tier": 8,
        "module": "local",
        "description": "Combine multiple equal parts (part × count)",
    },
})


# =============================================================================
# TIER 9: Geometry
# =============================================================================

def _area_rectangle(length, width):
    """Area of a rectangle."""
    return length * width

def _area_square(side):
    """Area of a square."""
    return side * side

def _area_triangle(base, height):
    """Area of a triangle (1/2 × base × height)."""
    return 0.5 * base * height

def _area_circle(radius):
    """Area of a circle (π × r²)."""
    return math.pi * radius * radius

def _area_trapezoid(base1, base2, height):
    """Area of a trapezoid."""
    return 0.5 * (base1 + base2) * height

def _perimeter_rectangle(length, width):
    """Perimeter of a rectangle."""
    return 2 * (length + width)

def _perimeter_square(side):
    """Perimeter of a square."""
    return 4 * side

def _circumference(radius):
    """Circumference of a circle (2 × π × r)."""
    return 2 * math.pi * radius

def _volume_cube(side):
    """Volume of a cube."""
    return side ** 3

def _volume_box(length, width, height):
    """Volume of a rectangular box."""
    return length * width * height

def _volume_sphere(radius):
    """Volume of a sphere (4/3 × π × r³)."""
    return (4/3) * math.pi * radius ** 3

def _volume_cylinder(radius, height):
    """Volume of a cylinder (π × r² × h)."""
    return math.pi * radius ** 2 * height

def _surface_area_cube(side):
    """Surface area of a cube (6 × s²)."""
    return 6 * side ** 2

def _surface_area_sphere(radius):
    """Surface area of a sphere (4 × π × r²)."""
    return 4 * math.pi * radius ** 2

def _pythagorean_c(a, b):
    """Find hypotenuse given two legs."""
    return math.sqrt(a ** 2 + b ** 2)

def _pythagorean_leg(c, a):
    """Find leg given hypotenuse and other leg."""
    val = c ** 2 - a ** 2
    if val < 0:
        return None
    return math.sqrt(val)

def _distance_2d(x1, y1, x2, y2):
    """Euclidean distance between two points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def _midpoint(a, b):
    """Midpoint between two numbers."""
    return (a + b) / 2


FUNCTION_REGISTRY.update({
    "area_rectangle": {
        "func": _area_rectangle,
        "arity": 2,
        "tier": 9,
        "module": "local",
        "description": "Area of a rectangle (length × width)",
    },
    "area_square": {
        "func": _area_square,
        "arity": 1,
        "tier": 9,
        "module": "local",
        "description": "Area of a square (side²)",
    },
    "area_triangle": {
        "func": _area_triangle,
        "arity": 2,
        "tier": 9,
        "module": "local",
        "description": "Area of a triangle (½ × base × height)",
    },
    "area_circle": {
        "func": _area_circle,
        "arity": 1,
        "tier": 9,
        "module": "local",
        "description": "Area of a circle (π × r²)",
    },
    "area_trapezoid": {
        "func": _area_trapezoid,
        "arity": 3,
        "tier": 9,
        "module": "local",
        "description": "Area of a trapezoid",
    },
    "perimeter_rectangle": {
        "func": _perimeter_rectangle,
        "arity": 2,
        "tier": 9,
        "module": "local",
        "description": "Perimeter of a rectangle",
    },
    "perimeter_square": {
        "func": _perimeter_square,
        "arity": 1,
        "tier": 9,
        "module": "local",
        "description": "Perimeter of a square",
    },
    "circumference": {
        "func": _circumference,
        "arity": 1,
        "tier": 9,
        "module": "local",
        "description": "Circumference of a circle (2πr)",
    },
    "volume_cube": {
        "func": _volume_cube,
        "arity": 1,
        "tier": 9,
        "module": "local",
        "description": "Volume of a cube",
    },
    "volume_box": {
        "func": _volume_box,
        "arity": 3,
        "tier": 9,
        "module": "local",
        "description": "Volume of a rectangular box",
    },
    "volume_sphere": {
        "func": _volume_sphere,
        "arity": 1,
        "tier": 9,
        "module": "local",
        "description": "Volume of a sphere",
    },
    "volume_cylinder": {
        "func": _volume_cylinder,
        "arity": 2,
        "tier": 9,
        "module": "local",
        "description": "Volume of a cylinder",
    },
    "surface_area_cube": {
        "func": _surface_area_cube,
        "arity": 1,
        "tier": 9,
        "module": "local",
        "description": "Surface area of a cube",
    },
    "surface_area_sphere": {
        "func": _surface_area_sphere,
        "arity": 1,
        "tier": 9,
        "module": "local",
        "description": "Surface area of a sphere",
    },
    "pythagorean_c": {
        "func": _pythagorean_c,
        "arity": 2,
        "tier": 9,
        "module": "local",
        "description": "Find hypotenuse (c = √(a² + b²))",
    },
    "pythagorean_leg": {
        "func": _pythagorean_leg,
        "arity": 2,
        "tier": 9,
        "module": "local",
        "description": "Find leg given hypotenuse and other leg",
    },
    "distance_2d": {
        "func": _distance_2d,
        "arity": 4,
        "tier": 9,
        "module": "local",
        "description": "Distance between two 2D points",
    },
    "midpoint": {
        "func": _midpoint,
        "arity": 2,
        "tier": 9,
        "module": "local",
        "description": "Midpoint between two values",
    },
})


# =============================================================================
# TIER 10: Financial / Money
# =============================================================================

def _simple_interest(principal, rate, time):
    """Simple interest: P × r × t (rate as decimal)."""
    return principal * rate * time

def _compound_interest(principal, rate, n, time):
    """Compound interest amount: P × (1 + r/n)^(n×t)."""
    return principal * ((1 + rate / n) ** (n * time))

def _discount(original, percent_off):
    """Price after discount."""
    return original * (1 - percent_off / 100)

def _markup(cost, percent_markup):
    """Price after markup."""
    return cost * (1 + percent_markup / 100)

def _profit(revenue, cost):
    """Profit = Revenue - Cost."""
    return revenue - cost

def _profit_margin(profit, revenue):
    """Profit margin as percentage."""
    if revenue == 0:
        return None
    return (profit / revenue) * 100

def _unit_price(total, quantity):
    """Price per unit."""
    if quantity == 0:
        return None
    return total / quantity

def _total_cost(unit_price, quantity):
    """Total cost = unit price × quantity."""
    return unit_price * quantity

def _tax_amount(price, tax_rate):
    """Tax amount (rate as percent)."""
    return price * (tax_rate / 100)

def _price_with_tax(price, tax_rate):
    """Price including tax."""
    return price * (1 + tax_rate / 100)

def _tip_amount(bill, tip_percent):
    """Tip amount."""
    return bill * (tip_percent / 100)


FUNCTION_REGISTRY.update({
    "simple_interest": {
        "func": _simple_interest,
        "arity": 3,
        "tier": 10,
        "module": "local",
        "description": "Simple interest (P × r × t)",
    },
    "compound_interest": {
        "func": _compound_interest,
        "arity": 4,
        "tier": 10,
        "module": "local",
        "description": "Compound interest amount",
    },
    "discount": {
        "func": _discount,
        "arity": 2,
        "tier": 10,
        "module": "local",
        "description": "Price after discount",
    },
    "markup": {
        "func": _markup,
        "arity": 2,
        "tier": 10,
        "module": "local",
        "description": "Price after markup",
    },
    "profit": {
        "func": _profit,
        "arity": 2,
        "tier": 10,
        "module": "local",
        "description": "Profit (revenue - cost)",
    },
    "profit_margin": {
        "func": _profit_margin,
        "arity": 2,
        "tier": 10,
        "module": "local",
        "description": "Profit margin as percentage",
    },
    "unit_price": {
        "func": _unit_price,
        "arity": 2,
        "tier": 10,
        "module": "local",
        "description": "Price per unit",
    },
    "total_cost": {
        "func": _total_cost,
        "arity": 2,
        "tier": 10,
        "module": "local",
        "description": "Total cost (unit price × quantity)",
    },
    "tax_amount": {
        "func": _tax_amount,
        "arity": 2,
        "tier": 10,
        "module": "local",
        "description": "Tax amount",
    },
    "price_with_tax": {
        "func": _price_with_tax,
        "arity": 2,
        "tier": 10,
        "module": "local",
        "description": "Price including tax",
    },
    "tip_amount": {
        "func": _tip_amount,
        "arity": 2,
        "tier": 10,
        "module": "local",
        "description": "Tip amount",
    },
})


# =============================================================================
# TIER 11: Rate / Time / Distance / Work
# =============================================================================

def _distance_formula(rate, time):
    """Distance = Rate × Time."""
    return rate * time

def _rate_formula(distance, time):
    """Rate = Distance / Time."""
    if time == 0:
        return None
    return distance / time

def _time_formula(distance, rate):
    """Time = Distance / Rate."""
    if rate == 0:
        return None
    return distance / rate

def _combined_work_rate(rate1, rate2):
    """Combined work rate when working together."""
    return rate1 + rate2

def _time_working_together(rate1, rate2):
    """Time to complete 1 job working together."""
    combined = rate1 + rate2
    if combined == 0:
        return None
    return 1 / combined

def _average_speed(total_distance, total_time):
    """Average speed."""
    if total_time == 0:
        return None
    return total_distance / total_time

def _relative_speed_opposite(speed1, speed2):
    """Relative speed when moving toward each other."""
    return speed1 + speed2

def _relative_speed_same(speed1, speed2):
    """Relative speed when moving in same direction."""
    return abs(speed1 - speed2)

def _meeting_time(distance, speed1, speed2):
    """Time until two objects meet (moving toward each other)."""
    combined = speed1 + speed2
    if combined == 0:
        return None
    return distance / combined


FUNCTION_REGISTRY.update({
    "distance_formula": {
        "func": _distance_formula,
        "arity": 2,
        "tier": 11,
        "module": "local",
        "description": "Distance = Rate × Time",
    },
    "rate_formula": {
        "func": _rate_formula,
        "arity": 2,
        "tier": 11,
        "module": "local",
        "description": "Rate = Distance / Time",
    },
    "time_formula": {
        "func": _time_formula,
        "arity": 2,
        "tier": 11,
        "module": "local",
        "description": "Time = Distance / Rate",
    },
    "combined_work_rate": {
        "func": _combined_work_rate,
        "arity": 2,
        "tier": 11,
        "module": "local",
        "description": "Combined work rate",
    },
    "time_working_together": {
        "func": _time_working_together,
        "arity": 2,
        "tier": 11,
        "module": "local",
        "description": "Time to complete job together",
    },
    "average_speed": {
        "func": _average_speed,
        "arity": 2,
        "tier": 11,
        "module": "local",
        "description": "Average speed",
    },
    "relative_speed_opposite": {
        "func": _relative_speed_opposite,
        "arity": 2,
        "tier": 11,
        "module": "local",
        "description": "Relative speed (moving toward each other)",
    },
    "relative_speed_same": {
        "func": _relative_speed_same,
        "arity": 2,
        "tier": 11,
        "module": "local",
        "description": "Relative speed (same direction)",
    },
    "meeting_time": {
        "func": _meeting_time,
        "arity": 3,
        "tier": 11,
        "module": "local",
        "description": "Time until objects meet",
    },
})


# =============================================================================
# TIER 12: Sequences & Series
# =============================================================================

def _arithmetic_term(a1, d, n):
    """nth term of arithmetic sequence: a1 + (n-1)×d."""
    return a1 + (n - 1) * d

def _arithmetic_sum(a1, an, n):
    """Sum of arithmetic sequence: n×(a1 + an)/2."""
    return n * (a1 + an) / 2

def _geometric_term(a1, r, n):
    """nth term of geometric sequence: a1 × r^(n-1)."""
    return a1 * (r ** (n - 1))

def _geometric_sum(a1, r, n):
    """Sum of geometric sequence."""
    if r == 1:
        return a1 * n
    return a1 * (1 - r ** n) / (1 - r)

def _sum_consecutive(start, end):
    """Sum of consecutive integers from start to end (inclusive)."""
    n = end - start + 1
    return n * (start + end) // 2

def _sum_first_n(n):
    """Sum of first n positive integers: n×(n+1)/2."""
    return n * (n + 1) // 2

def _triangular_number(n):
    """nth triangular number: n×(n+1)/2."""
    return n * (n + 1) // 2

def _square_number(n):
    """nth square number: n²."""
    return n * n


FUNCTION_REGISTRY.update({
    "arithmetic_term": {
        "func": _arithmetic_term,
        "arity": 3,
        "tier": 12,
        "module": "local",
        "description": "nth term of arithmetic sequence",
    },
    "arithmetic_sum": {
        "func": _arithmetic_sum,
        "arity": 3,
        "tier": 12,
        "module": "local",
        "description": "Sum of arithmetic sequence",
    },
    "geometric_term": {
        "func": _geometric_term,
        "arity": 3,
        "tier": 12,
        "module": "local",
        "description": "nth term of geometric sequence",
    },
    "geometric_sum": {
        "func": _geometric_sum,
        "arity": 3,
        "tier": 12,
        "module": "local",
        "description": "Sum of geometric sequence",
    },
    "sum_consecutive": {
        "func": _sum_consecutive,
        "arity": 2,
        "tier": 12,
        "module": "local",
        "description": "Sum of consecutive integers",
    },
    "sum_first_n": {
        "func": _sum_first_n,
        "arity": 1,
        "tier": 12,
        "module": "local",
        "description": "Sum of first n positive integers",
    },
    "triangular_number": {
        "func": _triangular_number,
        "arity": 1,
        "tier": 12,
        "module": "local",
        "description": "nth triangular number",
    },
    "square_number": {
        "func": _square_number,
        "arity": 1,
        "tier": 12,
        "module": "local",
        "description": "nth square number",
    },
})


# =============================================================================
# TIER 13: Linear Algebra
# =============================================================================

def _dot_product(a: List[Number], b: List[Number]) -> Number:
    """Dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))

def _vector_magnitude(v: List[Number]) -> Number:
    """Magnitude of a vector."""
    return math.sqrt(sum(x**2 for x in v))

def _vector_add(a: List[Number], b: List[Number]) -> List[Number]:
    """Add two vectors."""
    return [x + y for x, y in zip(a, b)]

def _vector_subtract(a: List[Number], b: List[Number]) -> List[Number]:
    """Subtract two vectors."""
    return [x - y for x, y in zip(a, b)]

def _scalar_multiply(v: List[Number], scalar: Number) -> List[Number]:
    """Multiply vector by scalar."""
    return [x * scalar for x in v]

def _vector_normalize(v: List[Number]) -> Optional[List[Number]]:
    """Normalize a vector to unit length."""
    mag = math.sqrt(sum(x**2 for x in v))
    if mag == 0:
        return None
    return [x / mag for x in v]

def _cross_product_3d(a: List[Number], b: List[Number]) -> List[Number]:
    """Cross product of two 3D vectors."""
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ]

def _vector_angle(a: List[Number], b: List[Number]) -> Optional[Number]:
    """Angle between two vectors in radians."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x**2 for x in a))
    mag_b = math.sqrt(sum(x**2 for x in b))
    if mag_a == 0 or mag_b == 0:
        return None
    cos_theta = dot / (mag_a * mag_b)
    # Clamp to [-1, 1] to handle floating point errors
    cos_theta = max(-1, min(1, cos_theta))
    return math.acos(cos_theta)

def _matrix_multiply_2x2(a: List[List[Number]], b: List[List[Number]]) -> List[List[Number]]:
    """Multiply two 2x2 matrices."""
    return [
        [a[0][0]*b[0][0] + a[0][1]*b[1][0], a[0][0]*b[0][1] + a[0][1]*b[1][1]],
        [a[1][0]*b[0][0] + a[1][1]*b[1][0], a[1][0]*b[0][1] + a[1][1]*b[1][1]]
    ]

def _determinant_2x2(m: List[List[Number]]) -> Number:
    """Determinant of a 2x2 matrix."""
    return m[0][0] * m[1][1] - m[0][1] * m[1][0]

def _determinant_3x3(m: List[List[Number]]) -> Number:
    """Determinant of a 3x3 matrix."""
    return (m[0][0] * (m[1][1]*m[2][2] - m[1][2]*m[2][1]) -
            m[0][1] * (m[1][0]*m[2][2] - m[1][2]*m[2][0]) +
            m[0][2] * (m[1][0]*m[2][1] - m[1][1]*m[2][0]))


FUNCTION_REGISTRY.update({
    "dot_product": {
        "func": _dot_product,
        "arity": 2,
        "tier": 13,
        "module": "local",
        "description": "Dot product of two vectors",
    },
    "vector_magnitude": {
        "func": _vector_magnitude,
        "arity": 1,
        "tier": 13,
        "module": "local",
        "description": "Magnitude of a vector",
    },
    "vector_add": {
        "func": _vector_add,
        "arity": 2,
        "tier": 13,
        "module": "local",
        "description": "Add two vectors",
    },
    "vector_subtract": {
        "func": _vector_subtract,
        "arity": 2,
        "tier": 13,
        "module": "local",
        "description": "Subtract two vectors",
    },
    "scalar_multiply": {
        "func": _scalar_multiply,
        "arity": 2,
        "tier": 13,
        "module": "local",
        "description": "Multiply vector by scalar",
    },
    "vector_normalize": {
        "func": _vector_normalize,
        "arity": 1,
        "tier": 13,
        "module": "local",
        "description": "Normalize vector to unit length",
    },
    "cross_product_3d": {
        "func": _cross_product_3d,
        "arity": 2,
        "tier": 13,
        "module": "local",
        "description": "Cross product of two 3D vectors",
    },
    "vector_angle": {
        "func": _vector_angle,
        "arity": 2,
        "tier": 13,
        "module": "local",
        "description": "Angle between two vectors (radians)",
    },
    "matrix_multiply_2x2": {
        "func": _matrix_multiply_2x2,
        "arity": 2,
        "tier": 13,
        "module": "local",
        "description": "Multiply two 2x2 matrices",
    },
    "determinant_2x2": {
        "func": _determinant_2x2,
        "arity": 1,
        "tier": 13,
        "module": "local",
        "description": "Determinant of a 2x2 matrix",
    },
    "determinant_3x3": {
        "func": _determinant_3x3,
        "arity": 1,
        "tier": 13,
        "module": "local",
        "description": "Determinant of a 3x3 matrix",
    },
})


# =============================================================================
# TIER 14: More Geometry
# =============================================================================

def _slope(x1: Number, y1: Number, x2: Number, y2: Number) -> Optional[Number]:
    """Slope between two points."""
    if x2 == x1:
        return None  # Vertical line
    return (y2 - y1) / (x2 - x1)

def _area_ellipse(a: Number, b: Number) -> Number:
    """Area of ellipse (pi * a * b)."""
    return math.pi * a * b

def _arc_length(radius: Number, angle_degrees: Number) -> Number:
    """Arc length given radius and central angle in degrees."""
    return radius * (angle_degrees * math.pi / 180)

def _sector_area(radius: Number, angle_degrees: Number) -> Number:
    """Area of circular sector."""
    return 0.5 * radius**2 * (angle_degrees * math.pi / 180)

def _diagonal_rectangle(length: Number, width: Number) -> Number:
    """Diagonal of rectangle."""
    return math.sqrt(length**2 + width**2)

def _diagonal_cube(side: Number) -> Number:
    """Space diagonal of cube."""
    return side * math.sqrt(3)

def _area_rhombus(d1: Number, d2: Number) -> Number:
    """Area of rhombus given diagonals."""
    return 0.5 * d1 * d2

def _area_parallelogram(base: Number, height: Number) -> Number:
    """Area of parallelogram."""
    return base * height

def _area_regular_polygon(n: int, side: Number) -> Number:
    """Area of regular polygon with n sides."""
    return (n * side**2) / (4 * math.tan(math.pi / n))

def _perimeter_regular_polygon(n: int, side: Number) -> Number:
    """Perimeter of regular polygon."""
    return n * side

def _volume_cone(radius: Number, height: Number) -> Number:
    """Volume of cone."""
    return (1/3) * math.pi * radius**2 * height

def _volume_pyramid(base_area: Number, height: Number) -> Number:
    """Volume of pyramid."""
    return (1/3) * base_area * height

def _surface_area_cylinder(radius: Number, height: Number) -> Number:
    """Total surface area of cylinder."""
    return 2 * math.pi * radius * (radius + height)

def _surface_area_cone(radius: Number, slant_height: Number) -> Number:
    """Total surface area of cone."""
    return math.pi * radius * (radius + slant_height)

def _heron_area(a: Number, b: Number, c: Number) -> Optional[Number]:
    """Area of triangle using Heron's formula."""
    s = (a + b + c) / 2
    val = s * (s - a) * (s - b) * (s - c)
    if val < 0:
        return None  # Invalid triangle
    return math.sqrt(val)

def _angle_from_sides(a: Number, b: Number, c: Number) -> Optional[Number]:
    """Angle C (opposite to side c) using law of cosines, returns degrees."""
    if a <= 0 or b <= 0 or c <= 0:
        return None
    cos_c = (a**2 + b**2 - c**2) / (2 * a * b)
    if cos_c < -1 or cos_c > 1:
        return None
    return math.degrees(math.acos(cos_c))

def _side_from_angle(a: Number, b: Number, angle_c_degrees: Number) -> Number:
    """Side c using law of cosines."""
    angle_rad = math.radians(angle_c_degrees)
    return math.sqrt(a**2 + b**2 - 2 * a * b * math.cos(angle_rad))


FUNCTION_REGISTRY.update({
    "slope": {
        "func": _slope,
        "arity": 4,
        "tier": 14,
        "module": "local",
        "description": "Slope between two points",
    },
    "area_ellipse": {
        "func": _area_ellipse,
        "arity": 2,
        "tier": 14,
        "module": "local",
        "description": "Area of ellipse",
    },
    "arc_length": {
        "func": _arc_length,
        "arity": 2,
        "tier": 14,
        "module": "local",
        "description": "Arc length from radius and angle",
    },
    "sector_area": {
        "func": _sector_area,
        "arity": 2,
        "tier": 14,
        "module": "local",
        "description": "Area of circular sector",
    },
    "diagonal_rectangle": {
        "func": _diagonal_rectangle,
        "arity": 2,
        "tier": 14,
        "module": "local",
        "description": "Diagonal of rectangle",
    },
    "diagonal_cube": {
        "func": _diagonal_cube,
        "arity": 1,
        "tier": 14,
        "module": "local",
        "description": "Space diagonal of cube",
    },
    "area_rhombus": {
        "func": _area_rhombus,
        "arity": 2,
        "tier": 14,
        "module": "local",
        "description": "Area of rhombus from diagonals",
    },
    "area_parallelogram": {
        "func": _area_parallelogram,
        "arity": 2,
        "tier": 14,
        "module": "local",
        "description": "Area of parallelogram",
    },
    "area_regular_polygon": {
        "func": _area_regular_polygon,
        "arity": 2,
        "tier": 14,
        "module": "local",
        "description": "Area of regular polygon",
    },
    "perimeter_regular_polygon": {
        "func": _perimeter_regular_polygon,
        "arity": 2,
        "tier": 14,
        "module": "local",
        "description": "Perimeter of regular polygon",
    },
    "volume_cone": {
        "func": _volume_cone,
        "arity": 2,
        "tier": 14,
        "module": "local",
        "description": "Volume of cone",
    },
    "volume_pyramid": {
        "func": _volume_pyramid,
        "arity": 2,
        "tier": 14,
        "module": "local",
        "description": "Volume of pyramid",
    },
    "surface_area_cylinder": {
        "func": _surface_area_cylinder,
        "arity": 2,
        "tier": 14,
        "module": "local",
        "description": "Surface area of cylinder",
    },
    "surface_area_cone": {
        "func": _surface_area_cone,
        "arity": 2,
        "tier": 14,
        "module": "local",
        "description": "Surface area of cone",
    },
    "heron_area": {
        "func": _heron_area,
        "arity": 3,
        "tier": 14,
        "module": "local",
        "description": "Triangle area using Heron's formula",
    },
    "angle_from_sides": {
        "func": _angle_from_sides,
        "arity": 3,
        "tier": 14,
        "module": "local",
        "description": "Angle from triangle sides (law of cosines)",
    },
    "side_from_angle": {
        "func": _side_from_angle,
        "arity": 3,
        "tier": 14,
        "module": "local",
        "description": "Side from angle (law of cosines)",
    },
})


# =============================================================================
# TIER 15: More Number Theory
# =============================================================================

def _is_prime_helper(n: int) -> bool:
    """Check if n is prime (helper function)."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def _is_prime(n: int) -> bool:
    """Check if n is prime."""
    return _is_prime_helper(n)

def _is_perfect_square(n: int) -> bool:
    """Check if n is a perfect square."""
    if n < 0:
        return False
    root = int(math.sqrt(n))
    return root * root == n

def _is_perfect_cube(n: int) -> bool:
    """Check if n is a perfect cube."""
    if n < 0:
        root = -int(round(abs(n) ** (1/3)))
    else:
        root = int(round(n ** (1/3)))
    return root ** 3 == n

def _next_prime(n: int) -> int:
    """Find next prime after n."""
    candidate = n + 1
    while not _is_prime_helper(candidate):
        candidate += 1
    return candidate

def _count_primes_up_to(n: int) -> int:
    """Count primes up to n."""
    count = 0
    for i in range(2, n + 1):
        if _is_prime_helper(i):
            count += 1
    return count

def _prime_factors(n: int) -> List[int]:
    """Return list of prime factors."""
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

def _sum_of_divisors(n: int) -> int:
    """Sum of all divisors of n."""
    total = 0
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            total += i
            if i != n // i:
                total += n // i
    return total

def _count_divisors(n: int) -> int:
    """Count number of divisors of n."""
    count = 0
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            count += 1
            if i != n // i:
                count += 1
    return count

def _is_coprime(a: int, b: int) -> bool:
    """Check if a and b are coprime (GCD = 1)."""
    return math.gcd(a, b) == 1

def _euler_totient(n: int) -> int:
    """Euler's totient function (count of coprimes up to n)."""
    result = n
    p = 2
    temp_n = n
    while p * p <= temp_n:
        if temp_n % p == 0:
            while temp_n % p == 0:
                temp_n //= p
            result -= result // p
        p += 1
    if temp_n > 1:
        result -= result // temp_n
    return result

def _digital_root(n: int) -> int:
    """Digital root (repeated digit sum until single digit)."""
    n = abs(n)
    if n == 0:
        return 0
    return 1 + (n - 1) % 9

def _sum_of_digits(n: int) -> int:
    """Sum of digits of n."""
    return sum(int(d) for d in str(abs(n)))


FUNCTION_REGISTRY.update({
    "is_prime": {
        "func": _is_prime,
        "arity": 1,
        "tier": 15,
        "module": "local",
        "description": "Check if number is prime",
    },
    "is_perfect_square": {
        "func": _is_perfect_square,
        "arity": 1,
        "tier": 15,
        "module": "local",
        "description": "Check if number is a perfect square",
    },
    "is_perfect_cube": {
        "func": _is_perfect_cube,
        "arity": 1,
        "tier": 15,
        "module": "local",
        "description": "Check if number is a perfect cube",
    },
    "next_prime": {
        "func": _next_prime,
        "arity": 1,
        "tier": 15,
        "module": "local",
        "description": "Find next prime after n",
    },
    "count_primes_up_to": {
        "func": _count_primes_up_to,
        "arity": 1,
        "tier": 15,
        "module": "local",
        "description": "Count primes up to n",
    },
    "prime_factors": {
        "func": _prime_factors,
        "arity": 1,
        "tier": 15,
        "module": "local",
        "description": "List of prime factors",
    },
    "sum_of_divisors": {
        "func": _sum_of_divisors,
        "arity": 1,
        "tier": 15,
        "module": "local",
        "description": "Sum of all divisors",
    },
    "count_divisors": {
        "func": _count_divisors,
        "arity": 1,
        "tier": 15,
        "module": "local",
        "description": "Count of divisors",
    },
    "is_coprime": {
        "func": _is_coprime,
        "arity": 2,
        "tier": 15,
        "module": "local",
        "description": "Check if two numbers are coprime",
    },
    "euler_totient": {
        "func": _euler_totient,
        "arity": 1,
        "tier": 15,
        "module": "local",
        "description": "Euler's totient function",
    },
    "digital_root": {
        "func": _digital_root,
        "arity": 1,
        "tier": 15,
        "module": "local",
        "description": "Digital root of a number",
    },
    "sum_of_digits": {
        "func": _sum_of_digits,
        "arity": 1,
        "tier": 15,
        "module": "local",
        "description": "Sum of digits",
    },
})


# =============================================================================
# TIER 16: More Statistics
# =============================================================================

def _weighted_mean(values: List[Number], weights: List[Number]) -> Optional[Number]:
    """Weighted arithmetic mean."""
    if len(values) != len(weights) or not weights:
        return None
    total_weight = sum(weights)
    if total_weight == 0:
        return None
    return sum(v * w for v, w in zip(values, weights)) / total_weight

def _harmonic_mean(values: List[Number]) -> Optional[Number]:
    """Harmonic mean."""
    if not values or any(v == 0 for v in values):
        return None
    return len(values) / sum(1/v for v in values)

def _geometric_mean(values: List[Number]) -> Optional[Number]:
    """Geometric mean."""
    if not values or any(v <= 0 for v in values):
        return None
    product = 1
    for v in values:
        product *= v
    return product ** (1/len(values))

def _percentile(values: List[Number], p: Number) -> Optional[Number]:
    """Calculate p-th percentile (0-100)."""
    if not values or p < 0 or p > 100:
        return None
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_vals) else f
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])

def _quartile(values: List[Number], q: int) -> Optional[Number]:
    """Calculate quartile (1, 2, or 3)."""
    if q not in [1, 2, 3]:
        return None
    return _percentile(values, q * 25)

def _interquartile_range(values: List[Number]) -> Optional[Number]:
    """Interquartile range (Q3 - Q1)."""
    q1 = _quartile(values, 1)
    q3 = _quartile(values, 3)
    if q1 is None or q3 is None:
        return None
    return q3 - q1

def _range_stat(values: List[Number]) -> Optional[Number]:
    """Range of values (max - min)."""
    if not values:
        return None
    return max(values) - min(values)

def _coefficient_of_variation(values: List[Number]) -> Optional[Number]:
    """Coefficient of variation (stdev/mean * 100)."""
    if not values or len(values) < 2:
        return None
    m = statistics.mean(values)
    if m == 0:
        return None
    return (statistics.stdev(values) / m) * 100

def _z_score(value: Number, mean: Number, stdev: Number) -> Optional[Number]:
    """Z-score (standard score)."""
    if stdev == 0:
        return None
    return (value - mean) / stdev

def _population_variance(values: List[Number]) -> Optional[Number]:
    """Population variance."""
    if not values:
        return None
    return statistics.pvariance(values)

def _population_stdev(values: List[Number]) -> Optional[Number]:
    """Population standard deviation."""
    if not values:
        return None
    return statistics.pstdev(values)


FUNCTION_REGISTRY.update({
    "weighted_mean": {
        "func": _weighted_mean,
        "arity": 2,
        "tier": 16,
        "module": "local",
        "description": "Weighted arithmetic mean",
    },
    "harmonic_mean": {
        "func": _harmonic_mean,
        "arity": 1,
        "tier": 16,
        "module": "local",
        "description": "Harmonic mean",
    },
    "geometric_mean": {
        "func": _geometric_mean,
        "arity": 1,
        "tier": 16,
        "module": "local",
        "description": "Geometric mean",
    },
    "percentile": {
        "func": _percentile,
        "arity": 2,
        "tier": 16,
        "module": "local",
        "description": "Calculate p-th percentile",
    },
    "quartile": {
        "func": _quartile,
        "arity": 2,
        "tier": 16,
        "module": "local",
        "description": "Calculate quartile (1, 2, or 3)",
    },
    "interquartile_range": {
        "func": _interquartile_range,
        "arity": 1,
        "tier": 16,
        "module": "local",
        "description": "Interquartile range",
    },
    "range_stat": {
        "func": _range_stat,
        "arity": 1,
        "tier": 16,
        "module": "local",
        "description": "Range of values",
    },
    "coefficient_of_variation": {
        "func": _coefficient_of_variation,
        "arity": 1,
        "tier": 16,
        "module": "local",
        "description": "Coefficient of variation",
    },
    "z_score": {
        "func": _z_score,
        "arity": 3,
        "tier": 16,
        "module": "local",
        "description": "Z-score (standard score)",
    },
    "population_variance": {
        "func": _population_variance,
        "arity": 1,
        "tier": 16,
        "module": "local",
        "description": "Population variance",
    },
    "population_stdev": {
        "func": _population_stdev,
        "arity": 1,
        "tier": 16,
        "module": "local",
        "description": "Population standard deviation",
    },
})


# =============================================================================
# TIER 17: Unit Conversions
# =============================================================================

def _celsius_to_fahrenheit(c: Number) -> Number:
    """Convert Celsius to Fahrenheit."""
    return c * 9/5 + 32

def _fahrenheit_to_celsius(f: Number) -> Number:
    """Convert Fahrenheit to Celsius."""
    return (f - 32) * 5/9

def _km_to_miles(km: Number) -> Number:
    """Convert kilometers to miles."""
    return km * 0.621371

def _miles_to_km(miles: Number) -> Number:
    """Convert miles to kilometers."""
    return miles * 1.60934

def _kg_to_pounds(kg: Number) -> Number:
    """Convert kilograms to pounds."""
    return kg * 2.20462

def _pounds_to_kg(pounds: Number) -> Number:
    """Convert pounds to kilograms."""
    return pounds * 0.453592

def _hours_to_minutes(hours: Number) -> Number:
    """Convert hours to minutes."""
    return hours * 60

def _minutes_to_hours(minutes: Number) -> Number:
    """Convert minutes to hours."""
    return minutes / 60

def _days_to_hours(days: Number) -> Number:
    """Convert days to hours."""
    return days * 24

def _hours_to_days(hours: Number) -> Number:
    """Convert hours to days."""
    return hours / 24

def _meters_to_feet(meters: Number) -> Number:
    """Convert meters to feet."""
    return meters * 3.28084

def _feet_to_meters(feet: Number) -> Number:
    """Convert feet to meters."""
    return feet * 0.3048

def _liters_to_gallons(liters: Number) -> Number:
    """Convert liters to US gallons."""
    return liters * 0.264172

def _gallons_to_liters(gallons: Number) -> Number:
    """Convert US gallons to liters."""
    return gallons * 3.78541

def _inches_to_cm(inches: Number) -> Number:
    """Convert inches to centimeters."""
    return inches * 2.54

def _cm_to_inches(cm: Number) -> Number:
    """Convert centimeters to inches."""
    return cm / 2.54

def _kelvin_to_celsius(k: Number) -> Number:
    """Convert Kelvin to Celsius."""
    return k - 273.15

def _celsius_to_kelvin(c: Number) -> Number:
    """Convert Celsius to Kelvin."""
    return c + 273.15


FUNCTION_REGISTRY.update({
    "celsius_to_fahrenheit": {
        "func": _celsius_to_fahrenheit,
        "arity": 1,
        "tier": 17,
        "module": "local",
        "description": "Convert Celsius to Fahrenheit",
    },
    "fahrenheit_to_celsius": {
        "func": _fahrenheit_to_celsius,
        "arity": 1,
        "tier": 17,
        "module": "local",
        "description": "Convert Fahrenheit to Celsius",
    },
    "km_to_miles": {
        "func": _km_to_miles,
        "arity": 1,
        "tier": 17,
        "module": "local",
        "description": "Convert kilometers to miles",
    },
    "miles_to_km": {
        "func": _miles_to_km,
        "arity": 1,
        "tier": 17,
        "module": "local",
        "description": "Convert miles to kilometers",
    },
    "kg_to_pounds": {
        "func": _kg_to_pounds,
        "arity": 1,
        "tier": 17,
        "module": "local",
        "description": "Convert kilograms to pounds",
    },
    "pounds_to_kg": {
        "func": _pounds_to_kg,
        "arity": 1,
        "tier": 17,
        "module": "local",
        "description": "Convert pounds to kilograms",
    },
    "hours_to_minutes": {
        "func": _hours_to_minutes,
        "arity": 1,
        "tier": 17,
        "module": "local",
        "description": "Convert hours to minutes",
    },
    "minutes_to_hours": {
        "func": _minutes_to_hours,
        "arity": 1,
        "tier": 17,
        "module": "local",
        "description": "Convert minutes to hours",
    },
    "days_to_hours": {
        "func": _days_to_hours,
        "arity": 1,
        "tier": 17,
        "module": "local",
        "description": "Convert days to hours",
    },
    "hours_to_days": {
        "func": _hours_to_days,
        "arity": 1,
        "tier": 17,
        "module": "local",
        "description": "Convert hours to days",
    },
    "meters_to_feet": {
        "func": _meters_to_feet,
        "arity": 1,
        "tier": 17,
        "module": "local",
        "description": "Convert meters to feet",
    },
    "feet_to_meters": {
        "func": _feet_to_meters,
        "arity": 1,
        "tier": 17,
        "module": "local",
        "description": "Convert feet to meters",
    },
    "liters_to_gallons": {
        "func": _liters_to_gallons,
        "arity": 1,
        "tier": 17,
        "module": "local",
        "description": "Convert liters to gallons",
    },
    "gallons_to_liters": {
        "func": _gallons_to_liters,
        "arity": 1,
        "tier": 17,
        "module": "local",
        "description": "Convert gallons to liters",
    },
    "inches_to_cm": {
        "func": _inches_to_cm,
        "arity": 1,
        "tier": 17,
        "module": "local",
        "description": "Convert inches to centimeters",
    },
    "cm_to_inches": {
        "func": _cm_to_inches,
        "arity": 1,
        "tier": 17,
        "module": "local",
        "description": "Convert centimeters to inches",
    },
    "kelvin_to_celsius": {
        "func": _kelvin_to_celsius,
        "arity": 1,
        "tier": 17,
        "module": "local",
        "description": "Convert Kelvin to Celsius",
    },
    "celsius_to_kelvin": {
        "func": _celsius_to_kelvin,
        "arity": 1,
        "tier": 17,
        "module": "local",
        "description": "Convert Celsius to Kelvin",
    },
})


# =============================================================================
# TIER 18: Age/Time Problems
# =============================================================================

def _age_in_years(birth_year: int, current_year: int) -> int:
    """Calculate age given birth year and current year."""
    return current_year - birth_year

def _years_until(current_age: int, target_age: int) -> int:
    """Years until reaching target age."""
    return target_age - current_age

def _age_sum(age1: int, age2: int) -> int:
    """Sum of two ages."""
    return age1 + age2

def _age_difference(age1: int, age2: int) -> int:
    """Difference between two ages."""
    return abs(age1 - age2)

def _age_ratio(age1: int, age2: int) -> Optional[Number]:
    """Ratio of two ages."""
    if age2 == 0:
        return None
    return age1 / age2

def _age_after_years(current_age: int, years: int) -> int:
    """Age after a number of years."""
    return current_age + years

def _age_before_years(current_age: int, years: int) -> int:
    """Age before a number of years (years ago)."""
    return current_age - years

def _combined_age_in_years(age1: int, age2: int, years: int) -> int:
    """Combined age after given years."""
    return (age1 + years) + (age2 + years)

def _time_difference_hours(start_hour: Number, end_hour: Number) -> Number:
    """Time difference in hours (handles day wrap)."""
    diff = end_hour - start_hour
    if diff < 0:
        diff += 24
    return diff

def _time_to_minutes(hours: Number, minutes: Number) -> Number:
    """Convert time to total minutes."""
    return hours * 60 + minutes

def _minutes_to_time(total_minutes: Number) -> tuple:
    """Convert total minutes to hours and minutes."""
    hours = int(total_minutes) // 60
    mins = int(total_minutes) % 60
    return (hours, mins)


FUNCTION_REGISTRY.update({
    "age_in_years": {
        "func": _age_in_years,
        "arity": 2,
        "tier": 18,
        "module": "local",
        "description": "Calculate age from birth year",
    },
    "years_until": {
        "func": _years_until,
        "arity": 2,
        "tier": 18,
        "module": "local",
        "description": "Years until target age",
    },
    "age_sum": {
        "func": _age_sum,
        "arity": 2,
        "tier": 18,
        "module": "local",
        "description": "Sum of two ages",
    },
    "age_difference": {
        "func": _age_difference,
        "arity": 2,
        "tier": 18,
        "module": "local",
        "description": "Difference between ages",
    },
    "age_ratio": {
        "func": _age_ratio,
        "arity": 2,
        "tier": 18,
        "module": "local",
        "description": "Ratio of two ages",
    },
    "age_after_years": {
        "func": _age_after_years,
        "arity": 2,
        "tier": 18,
        "module": "local",
        "description": "Age after given years",
    },
    "age_before_years": {
        "func": _age_before_years,
        "arity": 2,
        "tier": 18,
        "module": "local",
        "description": "Age before given years",
    },
    "combined_age_in_years": {
        "func": _combined_age_in_years,
        "arity": 3,
        "tier": 18,
        "module": "local",
        "description": "Combined age after years",
    },
    "time_difference_hours": {
        "func": _time_difference_hours,
        "arity": 2,
        "tier": 18,
        "module": "local",
        "description": "Time difference in hours",
    },
    "time_to_minutes": {
        "func": _time_to_minutes,
        "arity": 2,
        "tier": 18,
        "module": "local",
        "description": "Convert hours:minutes to total minutes",
    },
    "minutes_to_time": {
        "func": _minutes_to_time,
        "arity": 1,
        "tier": 18,
        "module": "local",
        "description": "Convert minutes to hours:minutes",
    },
})


# =============================================================================
# TIER 19: Mixture/Concentration
# =============================================================================

def _mixture_concentration(conc1: Number, vol1: Number, conc2: Number, vol2: Number) -> Number:
    """Concentration after mixing two solutions."""
    total_solute = conc1 * vol1 + conc2 * vol2
    total_volume = vol1 + vol2
    return total_solute / total_volume if total_volume > 0 else 0

def _dilution(initial_conc: Number, initial_vol: Number, final_vol: Number) -> Number:
    """Concentration after dilution."""
    return (initial_conc * initial_vol) / final_vol if final_vol > 0 else 0

def _volume_for_concentration(initial_conc: Number, initial_vol: Number, target_conc: Number) -> Optional[Number]:
    """Final volume needed to achieve target concentration."""
    if target_conc == 0:
        return None
    return (initial_conc * initial_vol) / target_conc

def _alloy_mixture(pure1: Number, weight1: Number, pure2: Number, weight2: Number) -> Number:
    """Purity of alloy mixture."""
    if weight1 + weight2 == 0:
        return 0
    return (pure1 * weight1 + pure2 * weight2) / (weight1 + weight2)

def _evaporation_concentration(initial_conc: Number, initial_vol: Number, evaporated: Number) -> Optional[Number]:
    """Concentration after evaporation."""
    final_vol = initial_vol - evaporated
    if final_vol <= 0:
        return None
    return (initial_conc * initial_vol) / final_vol

def _add_pure_solute(initial_conc: Number, initial_vol: Number, added_solute: Number) -> Number:
    """Concentration after adding pure solute."""
    total_solute = (initial_conc / 100) * initial_vol + added_solute
    return (total_solute / initial_vol) * 100

def _replacement_concentration(initial_conc: Number, total_vol: Number, replaced_vol: Number, new_conc: Number) -> Number:
    """Concentration after replacing part of solution."""
    remaining = (initial_conc / 100) * (total_vol - replaced_vol)
    added = (new_conc / 100) * replaced_vol
    return ((remaining + added) / total_vol) * 100


FUNCTION_REGISTRY.update({
    "mixture_concentration": {
        "func": _mixture_concentration,
        "arity": 4,
        "tier": 19,
        "module": "local",
        "description": "Concentration after mixing solutions",
    },
    "dilution": {
        "func": _dilution,
        "arity": 3,
        "tier": 19,
        "module": "local",
        "description": "Concentration after dilution",
    },
    "volume_for_concentration": {
        "func": _volume_for_concentration,
        "arity": 3,
        "tier": 19,
        "module": "local",
        "description": "Volume needed for target concentration",
    },
    "alloy_mixture": {
        "func": _alloy_mixture,
        "arity": 4,
        "tier": 19,
        "module": "local",
        "description": "Purity of alloy mixture",
    },
    "evaporation_concentration": {
        "func": _evaporation_concentration,
        "arity": 3,
        "tier": 19,
        "module": "local",
        "description": "Concentration after evaporation",
    },
    "add_pure_solute": {
        "func": _add_pure_solute,
        "arity": 3,
        "tier": 19,
        "module": "local",
        "description": "Concentration after adding solute",
    },
    "replacement_concentration": {
        "func": _replacement_concentration,
        "arity": 4,
        "tier": 19,
        "module": "local",
        "description": "Concentration after replacement",
    },
})


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_function(name: str) -> Callable:
    """
    Get function by name.

    Args:
        name: The function name as registered in FUNCTION_REGISTRY.

    Returns:
        The callable function.

    Raises:
        KeyError: If the function name is not found in the registry.
    """
    if name not in FUNCTION_REGISTRY:
        raise KeyError(f"Function '{name}' not found in registry. "
                       f"Available functions: {list(FUNCTION_REGISTRY.keys())}")
    return FUNCTION_REGISTRY[name]["func"]


def list_functions(tier: Optional[int] = None) -> List[str]:
    """
    List function names, optionally filtered by tier.

    Args:
        tier: If provided, only return functions from this tier.
              If None, return all function names.

    Returns:
        List of function names.
    """
    if tier is None:
        return list(FUNCTION_REGISTRY.keys())
    return [name for name, info in FUNCTION_REGISTRY.items() if info["tier"] == tier]


def call_function(name: str, *args) -> Any:
    """
    Call a function by name with given arguments.

    Args:
        name: The function name as registered in FUNCTION_REGISTRY.
        *args: Arguments to pass to the function.

    Returns:
        The result of calling the function with the given arguments.

    Raises:
        KeyError: If the function name is not found in the registry.
    """
    func = get_function(name)
    return func(*args)


def get_function_info(name: str) -> dict:
    """
    Get full info dict for a function.

    Args:
        name: The function name as registered in FUNCTION_REGISTRY.

    Returns:
        Dictionary containing func, arity, tier, module, and description.

    Raises:
        KeyError: If the function name is not found in the registry.
    """
    if name not in FUNCTION_REGISTRY:
        raise KeyError(f"Function '{name}' not found in registry. "
                       f"Available functions: {list(FUNCTION_REGISTRY.keys())}")
    return FUNCTION_REGISTRY[name].copy()


def get_tiers() -> List[int]:
    """
    Get list of all available tiers.

    Returns:
        Sorted list of unique tier numbers.
    """
    return sorted(set(info["tier"] for info in FUNCTION_REGISTRY.values()))


def get_tier_description(tier: int) -> str:
    """
    Get a human-readable description of a tier.

    Args:
        tier: The tier number.

    Returns:
        Description string for the tier.
    """
    descriptions = {
        1: "Arithmetic",
        2: "Comparison",
        3: "Trigonometry",
        4: "Logarithms/Exponentials",
        5: "Number Theory",
        6: "Statistics",
        7: "Symbolic (requires sympy)",
        8: "Percentages & Ratios",
        9: "Geometry",
        10: "Financial",
        11: "Rate/Time/Distance",
        12: "Sequences & Series",
        13: "Linear Algebra",
        14: "More Geometry",
        15: "More Number Theory",
        16: "More Statistics",
        17: "Unit Conversions",
        18: "Age/Time Problems",
        19: "Mixture/Concentration",
    }
    return descriptions.get(tier, f"Tier {tier}")


def count_functions() -> int:
    """Count total registered functions."""
    return len(FUNCTION_REGISTRY)


def execute(func_name: str, *args) -> Any:
    """Execute a function by name (alias for call_function)."""
    return call_function(func_name, *args)


# Alias for flat architecture compatibility
REGISTRY = {name: info["func"] for name, info in FUNCTION_REGISTRY.items()}


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    print("Running function_registry tests...\n")

    # -------------------------------------------------------------------------
    # Tier 1: Arithmetic tests
    # -------------------------------------------------------------------------
    print("Tier 1: Arithmetic")
    assert call_function("add", 2, 3) == 5, "add failed"
    assert call_function("sub", 10, 4) == 6, "sub failed"
    assert call_function("mul", 3, 7) == 21, "mul failed"
    assert call_function("truediv", 15, 4) == 3.75, "truediv failed"
    assert call_function("floordiv", 15, 4) == 3, "floordiv failed"
    assert call_function("mod", 17, 5) == 2, "mod failed"
    assert call_function("pow", 2, 10) == 1024, "pow failed"
    assert call_function("neg", 5) == -5, "neg failed"
    assert call_function("abs", -42) == 42, "abs failed"
    assert call_function("sqrt", 16) == 4.0, "sqrt failed"
    assert call_function("cbrt", 27) == 3.0, "cbrt failed"
    assert call_function("floor", 3.7) == 3, "floor failed"
    assert call_function("ceil", 3.2) == 4, "ceil failed"
    assert call_function("trunc", -3.7) == -3, "trunc failed"
    print("  All Tier 1 tests passed!")

    # -------------------------------------------------------------------------
    # Tier 2: Comparison tests
    # -------------------------------------------------------------------------
    print("Tier 2: Comparison")
    assert call_function("eq", 5, 5) is True, "eq failed"
    assert call_function("ne", 5, 3) is True, "ne failed"
    assert call_function("lt", 3, 5) is True, "lt failed"
    assert call_function("le", 5, 5) is True, "le failed"
    assert call_function("gt", 7, 3) is True, "gt failed"
    assert call_function("ge", 7, 7) is True, "ge failed"
    assert call_function("max", 1, 5, 3) == 5, "max failed"
    assert call_function("min", 1, 5, 3) == 1, "min failed"
    print("  All Tier 2 tests passed!")

    # -------------------------------------------------------------------------
    # Tier 3: Trigonometry tests
    # -------------------------------------------------------------------------
    print("Tier 3: Trigonometry")
    import math as m
    assert abs(call_function("sin", m.pi / 2) - 1.0) < 1e-10, "sin failed"
    assert abs(call_function("cos", 0) - 1.0) < 1e-10, "cos failed"
    assert abs(call_function("tan", 0) - 0.0) < 1e-10, "tan failed"
    assert abs(call_function("asin", 1) - m.pi / 2) < 1e-10, "asin failed"
    assert abs(call_function("acos", 1) - 0.0) < 1e-10, "acos failed"
    assert abs(call_function("atan", 0) - 0.0) < 1e-10, "atan failed"
    assert abs(call_function("atan2", 1, 1) - m.pi / 4) < 1e-10, "atan2 failed"
    assert abs(call_function("degrees", m.pi) - 180.0) < 1e-10, "degrees failed"
    assert abs(call_function("radians", 180) - m.pi) < 1e-10, "radians failed"
    assert call_function("hypot", 3, 4) == 5.0, "hypot failed"
    print("  All Tier 3 tests passed!")

    # -------------------------------------------------------------------------
    # Tier 4: Logarithms/Exponentials tests
    # -------------------------------------------------------------------------
    print("Tier 4: Logarithms/Exponentials")
    assert abs(call_function("log", m.e) - 1.0) < 1e-10, "log (natural) failed"
    assert abs(call_function("log", 100, 10) - 2.0) < 1e-10, "log (base 10) failed"
    assert abs(call_function("log10", 1000) - 3.0) < 1e-10, "log10 failed"
    assert abs(call_function("log2", 8) - 3.0) < 1e-10, "log2 failed"
    assert abs(call_function("exp", 1) - m.e) < 1e-10, "exp failed"
    assert call_function("exp2", 3) == 8.0, "exp2 failed"
    print("  All Tier 4 tests passed!")

    # -------------------------------------------------------------------------
    # Tier 5: Number Theory tests
    # -------------------------------------------------------------------------
    print("Tier 5: Number Theory")
    assert call_function("gcd", 48, 18) == 6, "gcd failed"
    assert call_function("lcm", 4, 6) == 12, "lcm failed"
    assert call_function("factorial", 5) == 120, "factorial failed"
    assert call_function("comb", 5, 2) == 10, "comb failed"
    assert call_function("perm", 5, 2) == 20, "perm failed"
    assert call_function("isqrt", 17) == 4, "isqrt failed"
    print("  All Tier 5 tests passed!")

    # -------------------------------------------------------------------------
    # Tier 6: Statistics tests
    # -------------------------------------------------------------------------
    print("Tier 6: Statistics")
    assert call_function("mean", [1, 2, 3, 4, 5]) == 3.0, "mean failed"
    assert call_function("median", [1, 2, 3, 4, 5]) == 3, "median failed"
    assert call_function("mode", [1, 2, 2, 3, 3, 3]) == 3, "mode failed"
    assert abs(call_function("stdev", [2, 4, 4, 4, 5, 5, 7, 9]) - 2.138089935299395) < 1e-10, "stdev failed"
    assert abs(call_function("variance", [2, 4, 4, 4, 5, 5, 7, 9]) - 4.571428571428571) < 1e-10, "variance failed"
    assert call_function("sum", [1, 2, 3, 4, 5]) == 15, "sum failed"
    assert call_function("len", [1, 2, 3, 4, 5]) == 5, "len failed"
    print("  All Tier 6 tests passed!")

    # -------------------------------------------------------------------------
    # Tier 7: Symbolic tests (if sympy available)
    # -------------------------------------------------------------------------
    if SYMPY_AVAILABLE:
        print("Tier 7: Symbolic (sympy)")
        x = sympy.Symbol('x')
        # Test simplify - verify it simplifies sin^2 + cos^2 to 1
        expr = sympy.sin(x)**2 + sympy.cos(x)**2
        simplified = call_function("simplify", expr)
        assert simplified == 1, "simplify failed"
        # Test expand
        expanded = call_function("expand", (x + 1)**2)
        assert expanded == x**2 + 2*x + 1, "expand failed"
        # Test factor
        factored = call_function("factor", x**2 - 1)
        assert factored == (x - 1)*(x + 1), "factor failed"
        # Test diff
        diff_result = call_function("diff", x**3, x)
        assert diff_result == 3*x**2, "diff failed"
        # Test integrate
        int_result = call_function("integrate", 2*x, x)
        assert int_result == x**2, "integrate failed"
        # Test solve
        solutions = call_function("solve", x**2 - 4, x)
        assert set(solutions) == {-2, 2}, "solve failed"
        print("  All Tier 7 tests passed!")
    else:
        print("Tier 7: Symbolic (sympy) - SKIPPED (sympy not installed)")

    # -------------------------------------------------------------------------
    # Helper function tests
    # -------------------------------------------------------------------------
    print("\nHelper function tests:")

    # Test get_function
    add_func = get_function("add")
    assert add_func(1, 2) == 3, "get_function failed"
    print("  get_function: passed")

    # Test get_function KeyError
    try:
        get_function("nonexistent")
        assert False, "get_function should raise KeyError"
    except KeyError:
        pass
    print("  get_function KeyError: passed")

    # Test list_functions
    all_funcs = list_functions()
    assert len(all_funcs) > 0, "list_functions returned empty"
    assert "add" in all_funcs, "add not in list_functions"
    print(f"  list_functions (all): {len(all_funcs)} functions")

    tier1_funcs = list_functions(tier=1)
    assert "add" in tier1_funcs, "add not in tier 1"
    assert "sin" not in tier1_funcs, "sin should not be in tier 1"
    print(f"  list_functions (tier 1): {len(tier1_funcs)} functions")

    # Test get_function_info
    add_info = get_function_info("add")
    assert add_info["arity"] == 2, "add arity wrong"
    assert add_info["tier"] == 1, "add tier wrong"
    assert add_info["module"] == "operator", "add module wrong"
    print("  get_function_info: passed")

    # Test get_tiers
    tiers = get_tiers()
    assert 1 in tiers and 6 in tiers, "get_tiers missing expected tiers"
    print(f"  get_tiers: {tiers}")

    # Test get_tier_description
    assert get_tier_description(1) == "Arithmetic", "tier 1 description wrong"
    assert get_tier_description(3) == "Trigonometry", "tier 3 description wrong"
    print("  get_tier_description: passed")

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)
