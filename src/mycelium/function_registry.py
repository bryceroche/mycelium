"""
Function Registry - Curated Python math function pointers organized by tier.

This module provides a centralized registry of mathematical functions that can be
used throughout the mycelium system for DSL execution and computation graphs.
"""

import operator
import math
import statistics
from typing import Any, Callable, List, Optional

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
        "func": operator.add,
        "arity": 2,
        "tier": 1,
        "module": "operator",
        "description": "Add two numbers",
    },
    "sub": {
        "func": operator.sub,
        "arity": 2,
        "tier": 1,
        "module": "operator",
        "description": "Subtract second number from first",
    },
    "mul": {
        "func": operator.mul,
        "arity": 2,
        "tier": 1,
        "module": "operator",
        "description": "Multiply two numbers",
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
        "func": sum,
        "arity": 1,  # takes iterable
        "tier": 6,
        "module": "builtins",
        "description": "Sum of all values in an iterable",
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
