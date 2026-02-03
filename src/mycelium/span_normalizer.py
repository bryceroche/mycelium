"""Span normalizer - makes spans parameter-agnostic for clustering."""

import re
from typing import Tuple, Dict, List, Any

# Common nouns in math word problems (items/objects, not people)
COMMON_NOUNS = {
    "apples", "apple", "oranges", "orange", "bananas", "banana",
    "cookies", "cookie", "candies", "candy", "cakes", "cake",
    "books", "book", "toys", "toy", "balls", "ball",
    "coins", "coin", "dollars", "dollar", "cents", "cent",
    "stickers", "sticker", "marbles", "marble", "cards", "card",
    "pencils", "pencil", "pens", "pen", "flowers", "flower",
    "eggs", "egg", "miles", "mile", "hours", "hour", "minutes", "minute",
    "days", "day", "weeks", "week", "months", "month", "years", "year",
    "gallons", "gallon", "liters", "liter", "pounds", "pound",
    "kilograms", "kilogram", "meters", "meter", "feet", "foot",
    "inches", "inch", "pages", "page", "problems", "problem",
    "questions", "question", "answers", "answer", "points", "point",
    "tickets", "ticket", "stamps", "stamp", "shells", "shell",
    "rocks", "rock", "stones", "stone", "beads", "bead",
    "buttons", "button", "ribbons", "ribbon", "boxes", "box",
    "bags", "bag", "baskets", "basket", "jars", "jar",
    "bottles", "bottle", "cups", "cup", "glasses", "glass",
    "plates", "plate", "bowls", "bowl", "slices", "slice",
    "pieces", "piece", "groups", "group", "sets", "set",
    "rows", "row", "columns", "column", "piles", "pile",
}

# Common names in math word problems
COMMON_NAMES = {
    "lisa", "tom", "john", "mary", "sarah", "mike", "anna", "david",
    "emma", "jack", "jane", "bob", "alice", "peter", "susan", "james",
    "amy", "ben", "chris", "dan", "emily", "frank", "grace", "henry",
    "kate", "luke", "maria", "nick", "olivia", "paul", "rachel", "sam",
    "tina", "victor", "wendy", "xavier", "yolanda", "zach",
    "adam", "betty", "carl", "diana", "eric", "fiona", "george", "helen",
    "ivan", "julia", "kevin", "linda", "mark", "nancy", "oscar", "patricia",
    "quinn", "robert", "sally", "tim", "uma", "vincent", "william", "yvonne",
    "alex", "brian", "cathy", "derek", "elena", "fred", "gloria", "harry",
    "ian", "jenny", "ken", "laura", "michael", "nina", "oliver", "penny",
    "rick", "steve", "teresa", "ursula", "vera", "walter", "zoe",
    "joe", "max", "kim", "lee", "pat", "sue", "ann", "ron", "jim", "meg",
    # People nouns that act like names in math problems
    "friend", "friends", "student", "students", "child", "children",
    "person", "people", "boy", "boys", "girl", "girls",
    "brother", "brothers", "sister", "sisters", "mother", "father",
    "mom", "dad", "teacher", "teachers", "classmate", "classmates",
}

# Subject pronouns
SUBJECT_PRONOUNS = {"she", "he", "they", "it", "i", "we", "you"}

# Object pronouns
OBJECT_PRONOUNS = {"her", "him", "them", "it", "me", "us", "you"}

# Possessive pronouns (including possessive adjectives)
POSSESSIVE_PRONOUNS = {"her", "his", "their", "its", "my", "our", "your", "hers", "theirs", "ours", "yours", "mine"}


def normalize_span(text: str) -> Tuple[str, Dict[str, List[Any]]]:
    """Normalize a span to be parameter-agnostic.

    Args:
        text: Original span like "Lisa sold 5 apples"

    Returns:
        Tuple of (normalized_text, extracted_values)
        e.g., ("[NAME] sold [N] [ITEM]", {"names": ["Lisa"], "numbers": [5], "items": ["apples"]})
    """
    extracted = {
        "names": [],
        "numbers": [],
        "items": [],
        "subject_pronouns": [],
        "object_pronouns": [],
        "possessive_pronouns": [],
    }

    # Tokenize while preserving structure
    # Handle decimal numbers specially, then split on word boundaries
    # Pattern captures: decimal numbers, words with apostrophes, or punctuation
    tokens = re.findall(r"-?\d+\.\d+|-?\d+|\b[\w']+\b|[^\w\s]", text)
    normalized_tokens = []

    i = 0
    while i < len(tokens):
        token = tokens[i]
        token_lower = token.lower()

        # Check for numbers (integers and decimals)
        if re.match(r'^-?\d+\.?\d*$', token):
            extracted["numbers"].append(float(token) if '.' in token else int(token))
            normalized_tokens.append("[N]")
            i += 1
            continue

        # Check for possessive pronouns first (before object pronouns, since "her" can be both)
        # Context: possessive if followed by a noun or name
        if token_lower in POSSESSIVE_PRONOUNS:
            # Look ahead to determine if possessive or object
            if token_lower == "her":
                # "her" is possessive if followed by a noun/name, object otherwise
                if i + 1 < len(tokens):
                    next_token_lower = tokens[i + 1].lower()
                    if next_token_lower in COMMON_NOUNS or next_token_lower in COMMON_NAMES:
                        extracted["possessive_pronouns"].append(token)
                        normalized_tokens.append("[POSS]")
                        i += 1
                        continue
                # Default to object pronoun for "her"
                extracted["object_pronouns"].append(token)
                normalized_tokens.append("[OBJ]")
                i += 1
                continue
            else:
                # Other possessives like "his", "their", etc.
                extracted["possessive_pronouns"].append(token)
                normalized_tokens.append("[POSS]")
                i += 1
                continue

        # Check for subject pronouns
        if token_lower in SUBJECT_PRONOUNS:
            extracted["subject_pronouns"].append(token)
            normalized_tokens.append("[SUBJ]")
            i += 1
            continue

        # Check for object pronouns (excluding "her" which is handled above)
        if token_lower in OBJECT_PRONOUNS and token_lower != "her":
            extracted["object_pronouns"].append(token)
            normalized_tokens.append("[OBJ]")
            i += 1
            continue

        # Check for names
        if token_lower in COMMON_NAMES:
            extracted["names"].append(token)
            normalized_tokens.append("[NAME]")
            i += 1
            continue

        # Check for common nouns (items)
        if token_lower in COMMON_NOUNS:
            extracted["items"].append(token)
            normalized_tokens.append("[ITEM]")
            i += 1
            continue

        # Keep token as-is
        normalized_tokens.append(token)
        i += 1

    # Reconstruct the normalized text
    normalized_text = _reconstruct_text(normalized_tokens)

    return normalized_text, extracted


def _reconstruct_text(tokens: List[str]) -> str:
    """Reconstruct text from tokens, handling spacing around punctuation."""
    if not tokens:
        return ""

    result = [tokens[0]]
    for i in range(1, len(tokens)):
        token = tokens[i]
        prev_token = tokens[i - 1]

        # No space before punctuation that attaches to previous word
        if token in {",", ".", "!", "?", ";", ":", "'s", "'t", "'re", "'ve", "'ll", "'d"}:
            result.append(token)
        # No space after opening brackets/quotes
        elif prev_token in {"(", "[", "{", '"', "'"}:
            result.append(token)
        # No space before closing brackets/quotes
        elif token in {")", "]", "}", '"', "'"}:
            result.append(token)
        else:
            result.append(" " + token)

    return "".join(result)


def batch_normalize(spans: List[str]) -> List[Tuple[str, Dict[str, List[Any]]]]:
    """Normalize multiple spans at once.

    Args:
        spans: List of original spans

    Returns:
        List of (normalized_text, extracted_values) tuples
    """
    return [normalize_span(span) for span in spans]


def get_canonical_form(text: str) -> str:
    """Get just the normalized text without extracted values.

    Useful for clustering/comparison where you only need the canonical form.
    """
    normalized, _ = normalize_span(text)
    return normalized


# =============================================================================
# Test cases
# =============================================================================

def test_normalize_span():
    """Test the span normalizer with various examples."""
    test_cases = [
        # Basic case
        ("Lisa sold 5 apples", "[NAME] sold [N] [ITEM]"),

        # Pronouns with context
        ("She gave her friend 3 cookies", "[SUBJ] gave [POSS] [NAME] [N] [ITEM]"),

        # Multiple names
        ("Tom has twice as many books as Mary", "[NAME] has twice as many [ITEM] as [NAME]"),

        # Numbers with decimals
        ("John spent 12.50 dollars", "[NAME] spent [N] [ITEM]"),

        # Object pronouns
        ("Mary gave him 4 apples", "[NAME] gave [OBJ] [N] [ITEM]"),

        # Possessive his
        ("Tom ate his 5 cookies", "[NAME] ate [POSS] [N] [ITEM]"),

        # Complex sentence
        ("Emma bought 3 books and gave 2 to her friend Jack",
         "[NAME] bought [N] [ITEM] and gave [N] to [POSS] [NAME] [NAME]"),

        # Mixed pronouns
        ("She told him that they have 10 marbles",
         "[SUBJ] told [OBJ] that [SUBJ] have [N] [ITEM]"),

        # Preserving structure words
        ("If Lisa has 5 more apples than Tom",
         "If [NAME] has [N] more [ITEM] than [NAME]"),
    ]

    print("Testing span normalizer...")
    print("=" * 60)

    all_passed = True
    for original, expected in test_cases:
        normalized, extracted = normalize_span(original)
        status = "PASS" if normalized == expected else "FAIL"
        if status == "FAIL":
            all_passed = False

        print(f"\nInput:    {original}")
        print(f"Expected: {expected}")
        print(f"Got:      {normalized}")
        print(f"Extracted: {extracted}")
        print(f"Status:   {status}")

    print("\n" + "=" * 60)
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return all_passed


if __name__ == "__main__":
    test_normalize_span()
