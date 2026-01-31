"""Seed the signature database with initial prototypes.

This script creates 3-5 initial signatures per function with diverse natural language
descriptions. These prototypes teach the LLM how to describe each operation and provide
a foundation for the k-NN classification system.

Usage:
    uv run python scripts/seed_signatures.py          # Add new signatures
    uv run python scripts/seed_signatures.py --clear  # Clear and reseed
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from mycelium.step_signatures.db import get_step_db, reset_step_db
from mycelium.function_registry import FUNCTION_REGISTRY
from mycelium.embedder import Embedder

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# Diverse descriptions for each function category
# Each description should be a natural language phrase that an LLM might use
# to describe the step when decomposing a math problem
SEED_DESCRIPTIONS = {
    # =========================================================================
    # TIER 1: Arithmetic
    # =========================================================================
    "add": [
        "combine two prices",
        "sum the quantities",
        "total of the amounts",
        "add the values together",
        "find the combined total",
    ],
    "sub": [
        "subtract from the total",
        "remaining after spending",
        "difference between values",
        "how much less",
        "take away from",
    ],
    "mul": [
        "multiply the numbers",
        "calculate total cost",
        "compute the product",
        "find area by multiplying",
        "times the quantity",
    ],
    "truediv": [
        "divide evenly",
        "split into equal parts",
        "calculate the quotient",
        "how many times does it fit",
        "ratio of values",
    ],
    "floordiv": [
        "divide and round down",
        "integer division",
        "how many whole groups",
        "floor of division",
    ],
    "mod": [
        "find the remainder",
        "what is left over after dividing",
        "modulo operation",
        "remainder when divided",
    ],
    "pow": [
        "raise to a power",
        "compute exponent",
        "calculate power",
        "squared or cubed value",
    ],
    "neg": [
        "negate the value",
        "make negative",
        "opposite sign",
    ],
    "abs": [
        "absolute value",
        "distance from zero",
        "remove negative sign",
        "magnitude of number",
    ],
    "sqrt": [
        "square root",
        "find the root",
        "what number squared equals this",
    ],
    "cbrt": [
        "cube root",
        "third root",
        "what number cubed equals this",
    ],
    "floor": [
        "round down to integer",
        "floor function",
        "largest integer less than",
    ],
    "ceil": [
        "round up to integer",
        "ceiling function",
        "smallest integer greater than",
    ],
    "trunc": [
        "truncate to integer",
        "remove decimal part",
        "integer portion only",
    ],

    # =========================================================================
    # TIER 2: Comparison
    # =========================================================================
    "max": [
        "find the maximum",
        "largest value",
        "biggest number",
        "highest of the values",
    ],
    "min": [
        "find the minimum",
        "smallest value",
        "least number",
        "lowest of the values",
    ],

    # =========================================================================
    # TIER 5: Number Theory
    # =========================================================================
    "gcd": [
        "greatest common divisor",
        "largest common factor",
        "find GCD",
    ],
    "lcm": [
        "least common multiple",
        "smallest common multiple",
        "find LCM",
    ],
    "factorial": [
        "factorial of number",
        "n factorial",
        "product of all positive integers up to n",
    ],
    "comb": [
        "combinations of n choose k",
        "number of ways to choose",
        "binomial coefficient",
    ],
    "perm": [
        "permutations of n items",
        "number of arrangements",
        "ordered selections",
    ],

    # =========================================================================
    # TIER 6: Statistics
    # =========================================================================
    "mean": [
        "calculate average",
        "find the mean",
        "arithmetic average",
        "sum divided by count",
    ],
    "median": [
        "find the median",
        "middle value",
        "central value when sorted",
    ],
    "mode": [
        "find the mode",
        "most frequent value",
        "most common number",
    ],
    "stdev": [
        "standard deviation",
        "measure of spread",
        "how spread out the data is",
    ],
    "variance": [
        "calculate variance",
        "squared deviation from mean",
        "measure of dispersion",
    ],
    "sum": [
        "sum all values",
        "total of all items",
        "add up everything",
    ],
    "len": [
        "count the items",
        "number of elements",
        "how many items",
    ],

    # =========================================================================
    # TIER 8: Percentages & Ratios
    # =========================================================================
    "percent_of": [
        "calculate 20% of the total",
        "find the percentage amount",
        "compute the tip",
        "determine the discount amount",
        "what is X percent of Y",
    ],
    "what_percent": [
        "what percent is this of the total",
        "calculate percentage of whole",
        "express as a percentage",
    ],
    "percent_change": [
        "percent increase from old to new",
        "calculate the percentage change",
        "how much did it grow in percent",
        "rate of change as percentage",
    ],
    "percent_increase": [
        "increase the value by a percentage",
        "add percentage markup",
        "grow by percent",
    ],
    "percent_decrease": [
        "decrease by percentage",
        "reduce by percent",
        "apply percentage discount",
    ],
    "ratio": [
        "find the ratio",
        "express as a ratio",
        "proportion between two values",
    ],
    "proportion_solve": [
        "solve the proportion",
        "find missing value in ratio",
        "cross multiply to solve",
    ],
    "remaining_after": [
        "what remains after",
        "leftover amount",
        "how much is left",
    ],
    "split_equally": [
        "divide equally among",
        "split evenly",
        "share equally",
    ],
    "combine_parts": [
        "combine multiple items",
        "total of repeated items",
        "quantity times unit",
    ],

    # =========================================================================
    # TIER 9: Geometry
    # =========================================================================
    "area_rectangle": [
        "area of rectangle",
        "length times width",
        "rectangular area",
        "space inside rectangle",
    ],
    "area_square": [
        "area of square",
        "side squared",
        "square area",
    ],
    "area_circle": [
        "area of circle",
        "circular area",
        "pi r squared",
    ],
    "area_triangle": [
        "area of triangle",
        "half base times height",
        "triangular area",
    ],
    "area_trapezoid": [
        "area of trapezoid",
        "average of bases times height",
    ],
    "perimeter_rectangle": [
        "perimeter of rectangle",
        "distance around rectangle",
        "sum of all sides",
    ],
    "perimeter_square": [
        "perimeter of square",
        "four times the side",
        "distance around square",
    ],
    "circumference": [
        "circumference of circle",
        "distance around circle",
        "perimeter of circle",
    ],
    "volume_cube": [
        "volume of cube",
        "side cubed",
        "cubic volume",
    ],
    "volume_box": [
        "volume of rectangular box",
        "length times width times height",
        "box volume",
    ],
    "volume_sphere": [
        "volume of sphere",
        "spherical volume",
        "four thirds pi r cubed",
    ],
    "volume_cylinder": [
        "volume of cylinder",
        "pi r squared times height",
        "cylindrical volume",
    ],
    "surface_area_cube": [
        "surface area of cube",
        "six times side squared",
    ],
    "surface_area_sphere": [
        "surface area of sphere",
        "four pi r squared",
    ],
    "pythagorean_c": [
        "find hypotenuse",
        "pythagorean theorem for c",
        "longest side of right triangle",
    ],
    "pythagorean_leg": [
        "find leg of right triangle",
        "missing side using pythagorean",
    ],
    "distance_2d": [
        "distance between two points",
        "euclidean distance",
        "how far apart",
    ],
    "midpoint": [
        "find the midpoint",
        "halfway between",
        "center point",
    ],

    # =========================================================================
    # TIER 10: Financial / Money
    # =========================================================================
    "simple_interest": [
        "calculate simple interest",
        "interest earned",
        "principal times rate times time",
    ],
    "compound_interest": [
        "calculate compound interest",
        "interest on interest",
        "compounded amount",
    ],
    "discount": [
        "price after discount",
        "sale price",
        "reduced price",
    ],
    "markup": [
        "price after markup",
        "add profit margin",
        "increase by percentage",
    ],
    "profit": [
        "calculate profit",
        "revenue minus cost",
        "earnings",
    ],
    "profit_margin": [
        "profit margin percentage",
        "profit as percent of revenue",
    ],
    "unit_price": [
        "price per unit",
        "cost per item",
        "unit cost",
    ],
    "total_cost": [
        "total cost of items",
        "unit price times quantity",
        "how much for all items",
    ],
    "tax_amount": [
        "calculate tax",
        "tax on purchase",
        "sales tax amount",
    ],
    "price_with_tax": [
        "price including tax",
        "total with tax",
        "after-tax price",
    ],
    "tip_amount": [
        "calculate tip",
        "gratuity amount",
        "service tip",
    ],

    # =========================================================================
    # TIER 11: Rate / Time / Distance / Work
    # =========================================================================
    "distance_formula": [
        "distance traveled",
        "rate times time",
        "how far",
    ],
    "rate_formula": [
        "calculate speed",
        "distance divided by time",
        "how fast",
    ],
    "time_formula": [
        "time to travel",
        "distance divided by speed",
        "how long to get there",
    ],
    "combined_work_rate": [
        "combined work rate",
        "working together rate",
        "total work capacity",
    ],
    "time_working_together": [
        "time when working together",
        "combined work time",
    ],
    "average_speed": [
        "average speed",
        "total distance over total time",
        "mean velocity",
    ],
    "relative_speed_opposite": [
        "relative speed approaching",
        "combined closing speed",
    ],
    "relative_speed_same": [
        "relative speed same direction",
        "difference in speeds",
    ],
    "meeting_time": [
        "when do they meet",
        "time until meeting",
        "intersection time",
    ],

    # =========================================================================
    # TIER 12: Sequences & Series
    # =========================================================================
    "arithmetic_term": [
        "nth term of arithmetic sequence",
        "find term in sequence",
        "arithmetic progression term",
    ],
    "arithmetic_sum": [
        "sum of arithmetic sequence",
        "arithmetic series sum",
    ],
    "geometric_term": [
        "nth term of geometric sequence",
        "geometric progression term",
    ],
    "geometric_sum": [
        "sum of geometric sequence",
        "geometric series sum",
    ],
    "sum_consecutive": [
        "sum of consecutive integers",
        "add numbers from start to end",
    ],
    "sum_first_n": [
        "sum of first n integers",
        "triangular sum",
        "1 plus 2 plus ... plus n",
    ],
    "triangular_number": [
        "nth triangular number",
        "sum of first n",
    ],
    "square_number": [
        "nth square number",
        "n squared",
        "perfect square",
    ],
}


def seed_database(clear_first: bool = False) -> dict:
    """Seed the database with initial signatures.

    Args:
        clear_first: If True, clear all existing signatures before seeding.

    Returns:
        Stats dictionary with counts of created/skipped signatures.
    """
    if clear_first:
        logger.info("Clearing existing signatures...")
        db = get_step_db()
        db.clear_all_data(force=True)
        reset_step_db()

    db = get_step_db()
    embedder = Embedder.get_instance()

    stats = {"created": 0, "skipped": 0, "functions": 0, "missing_funcs": []}

    logger.info(f"Seeding database with {len(SEED_DESCRIPTIONS)} functions...")
    logger.info(f"Using embedder: {embedder.model_name}")
    logger.info("")

    for func_name, descriptions in SEED_DESCRIPTIONS.items():
        if func_name not in FUNCTION_REGISTRY:
            logger.warning(f"  [SKIP] {func_name} not in registry")
            stats["missing_funcs"].append(func_name)
            continue

        stats["functions"] += 1
        logger.info(f"[{func_name}]")

        for desc in descriptions:
            # Embed the description
            embedding = embedder.embed(desc)

            # Create signature
            sig, created = db.find_or_create(
                step_text=desc,
                embedding=embedding,
                func_name=func_name,
            )

            if created:
                stats["created"] += 1
                logger.info(f"  + Created: '{desc}' (id={sig.id})")
            else:
                stats["skipped"] += 1
                logger.info(f"  = Exists: '{desc}' (matched sig={sig.id})")

        logger.info("")

    logger.info("=" * 60)
    logger.info("SEEDING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Functions processed: {stats['functions']}")
    logger.info(f"Signatures created: {stats['created']}")
    logger.info(f"Signatures skipped (duplicates): {stats['skipped']}")

    if stats["missing_funcs"]:
        logger.warning(f"Missing from registry: {stats['missing_funcs']}")

    # Print summary
    logger.info("")
    logger.info("Database summary:")
    logger.info(f"  Total signatures: {db.count_signatures()}")
    func_names = db.get_all_func_names()
    logger.info(f"  Functions with signatures: {len(func_names)}")

    return stats


def verify_seeding() -> None:
    """Verify the seeding worked by printing stats."""
    db = get_step_db()

    logger.info("")
    logger.info("=" * 60)
    logger.info("VERIFICATION")
    logger.info("=" * 60)

    total = db.count_signatures()
    logger.info(f"Total signatures: {total}")

    func_names = db.get_all_func_names()
    logger.info(f"Functions with signatures: {len(func_names)}")

    logger.info("")
    logger.info("Signatures per function:")
    for func_name in sorted(func_names):
        sigs = db.get_signatures_by_func(func_name)
        logger.info(f"  {func_name}: {len(sigs)} signatures")


if __name__ == "__main__":
    clear = "--clear" in sys.argv
    verify_only = "--verify" in sys.argv

    if verify_only:
        verify_seeding()
    else:
        seed_database(clear_first=clear)
        verify_seeding()
