#!/usr/bin/env python3
"""Analyze 23K templates to find ungeneralized dimensions.

Currently templates generalize: PERSON, ITEM, N (numbers)
But patterns may still contain specific:
- Locations (farmers market, school, store)
- Time periods (Monday, morning, January)
- Units (dollars, miles, pounds, gallons)
- Occupations (teacher, farmer, baker)
- Colors (red, blue, green)
- Animals (dogs, cats, horses)
- Food items (apples, cookies, bread)

This script identifies what's leaking through and how often.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


# Known generalization tokens
GENERIC_TOKENS = {"[PERSON1]", "[PERSON2]", "[PERSON3]", "[ITEM1]", "[ITEM2]", "[ITEM3]", "[N]"}

# Dimension detectors
LOCATIONS = {
    "store", "shop", "market", "mall", "school", "gym", "park", "library",
    "hospital", "restaurant", "cafe", "bakery", "farm", "garden", "zoo",
    "beach", "pool", "church", "office", "factory", "warehouse", "garage",
    "kitchen", "bathroom", "bedroom", "classroom", "playground", "stadium",
    "theater", "museum", "bank", "grocery", "supermarket", "pharmacy",
    "salon", "barber", "laundromat", "airport", "station", "harbor",
    "farmers market", "flea market", "pet store", "book store", "toy store",
}

TEMPORAL = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "morning", "afternoon", "evening", "night", "noon", "midnight",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "spring", "summer", "fall", "autumn", "winter",
    "week", "month", "year", "day", "hour", "minute",
    "today", "yesterday", "tomorrow", "daily", "weekly", "monthly", "yearly",
    "birthday", "christmas", "halloween", "thanksgiving", "easter",
}

UNITS = {
    "dollar", "dollars", "cent", "cents",
    "mile", "miles", "kilometer", "kilometers", "km",
    "pound", "pounds", "kilogram", "kilograms", "kg", "gram", "grams",
    "gallon", "gallons", "liter", "liters", "quart", "quarts", "pint", "pints",
    "inch", "inches", "foot", "feet", "yard", "yards", "meter", "meters",
    "minute", "minutes", "hour", "hours", "second", "seconds",
    "page", "pages", "piece", "pieces", "slice", "slices",
    "cup", "cups", "tablespoon", "teaspoon", "ounce", "ounces",
    "dozen", "pair", "pairs", "set", "sets",
    "percent", "percentage",
}

OCCUPATIONS = {
    "teacher", "farmer", "baker", "doctor", "nurse", "driver",
    "carpenter", "plumber", "mechanic", "painter", "chef", "cook",
    "waiter", "waitress", "cashier", "clerk", "manager", "owner",
    "worker", "employee", "student", "professor", "principal",
    "dentist", "vet", "veterinarian", "lawyer", "judge",
    "firefighter", "police", "officer", "soldier", "pilot",
    "artist", "musician", "singer", "dancer", "actor", "actress",
    "coach", "referee", "athlete", "swimmer", "runner",
}

COLORS = {
    "red", "blue", "green", "yellow", "orange", "purple", "pink",
    "black", "white", "brown", "gray", "grey", "gold", "silver",
}

ANIMALS = {
    "dog", "dogs", "cat", "cats", "horse", "horses", "cow", "cows",
    "pig", "pigs", "chicken", "chickens", "duck", "ducks", "goose", "geese",
    "fish", "sheep", "goat", "goats", "rabbit", "rabbits",
    "bird", "birds", "parrot", "parrots", "hamster", "hamsters",
    "turtle", "turtles", "snake", "snakes", "frog", "frogs",
    "lion", "lions", "tiger", "tigers", "bear", "bears", "elephant", "elephants",
    "monkey", "monkeys", "deer", "mouse", "mice", "rat", "rats",
}

FOODS = {
    "apple", "apples", "orange", "oranges", "banana", "bananas",
    "cookie", "cookies", "cake", "cakes", "pie", "pies",
    "bread", "sandwich", "sandwiches", "pizza", "pizzas",
    "candy", "candies", "chocolate", "chocolates",
    "egg", "eggs", "milk", "juice", "water", "soda",
    "rice", "pasta", "noodles", "soup", "salad",
    "meat", "chicken", "beef", "pork", "fish",
    "vegetable", "vegetables", "fruit", "fruits",
    "tomato", "tomatoes", "potato", "potatoes", "carrot", "carrots",
    "strawberry", "strawberries", "blueberry", "blueberries",
    "grape", "grapes", "peach", "peaches", "mango", "mangoes",
    "lemon", "lemons", "cherry", "cherries", "pear", "pears",
    "watermelon", "pumpkin", "corn",
}

CONTAINERS = {
    "box", "boxes", "bag", "bags", "basket", "baskets",
    "bottle", "bottles", "jar", "jars", "can", "cans",
    "bucket", "buckets", "barrel", "barrels", "crate", "crates",
    "tray", "trays", "plate", "plates", "bowl", "bowls",
    "shelf", "shelves", "rack", "racks", "row", "rows",
    "pile", "piles", "stack", "stacks",
}


def analyze_templates(templates):
    """Analyze templates for ungeneralized specifics."""
    dimension_counts = defaultdict(Counter)
    dimension_template_examples = defaultdict(lambda: defaultdict(list))

    for t in templates:
        pattern = t.get("pattern", "").lower()
        # Remove generic tokens for analysis
        clean = pattern
        for tok in GENERIC_TOKENS:
            clean = clean.replace(tok.lower(), " ")

        words = set(re.findall(r'\b[a-z]+\b', clean))

        # Check each dimension
        for dim_name, dim_words in [
            ("LOCATION", LOCATIONS),
            ("TEMPORAL", TEMPORAL),
            ("UNIT", UNITS),
            ("OCCUPATION", OCCUPATIONS),
            ("COLOR", COLORS),
            ("ANIMAL", ANIMALS),
            ("FOOD", FOODS),
            ("CONTAINER", CONTAINERS),
        ]:
            found = words & dim_words
            for word in found:
                dimension_counts[dim_name][word] += t.get("count", 1)
                if len(dimension_template_examples[dim_name][word]) < 3:
                    dimension_template_examples[dim_name][word].append(pattern)

    return dimension_counts, dimension_template_examples


def main():
    project_root = Path(__file__).parent.parent
    templates_path = project_root / "qwen_templates.json"

    print(f"Loading templates from {templates_path}...")
    with open(templates_path) as f:
        templates = json.load(f)
    print(f"Loaded {len(templates)} templates\n")

    dim_counts, dim_examples = analyze_templates(templates)

    # Report
    total_spans = sum(t.get("count", 1) for t in templates)
    print(f"Total spans across all templates: {total_spans}\n")

    print("=" * 80)
    print("UNGENERALIZED DIMENSIONS ANALYSIS")
    print("=" * 80)

    for dim_name in sorted(dim_counts.keys(), key=lambda d: -sum(dim_counts[d].values())):
        counts = dim_counts[dim_name]
        total = sum(counts.values())
        n_unique = len(counts)
        pct = total / total_spans * 100

        print(f"\n{'='*60}")
        print(f"{dim_name}: {total} spans ({pct:.1f}%), {n_unique} unique words")
        print(f"{'='*60}")

        for word, count in counts.most_common(15):
            examples = dim_examples[dim_name][word][:2]
            ex_str = " | ".join(examples)[:80]
            print(f"  {word:<20} {count:>5} spans   e.g.: {ex_str}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY: Dimensions that should be generalized")
    print(f"{'='*80}")
    for dim_name in sorted(dim_counts.keys(), key=lambda d: -sum(dim_counts[d].values())):
        total = sum(dim_counts[dim_name].values())
        n_unique = len(dim_counts[dim_name])
        pct = total / total_spans * 100
        suggested_token = f"[{dim_name}1]"
        print(f"  {dim_name:<15} → {suggested_token:<15} {total:>6} spans ({pct:.1f}%), {n_unique} unique words")

    # Check for words in [ITEM1/2/3] that probably should be in other categories
    print(f"\n{'='*80}")
    print("NOTE: Many of these words may already be captured as [ITEM1], [ITEM2], etc.")
    print("The question is whether they SHOULD have their own generic token for better")
    print("template deduplication (e.g., all food items → [FOOD1], all locations → [LOC1])")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
