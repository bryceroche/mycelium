"""
Verb-based operation classification module.

Uses verb taxonomy to classify operations when KNN similarity is ambiguous.
Research shows verbs carry operational semantics - "sold" means SUB, "found" means ADD.

This is the SINGLE SOURCE OF TRUTH for verb patterns.
Import from here rather than duplicating patterns elsewhere.
"""

import re
from typing import Optional, Tuple

# Verb taxonomy mapping verbs to operations
# Includes both past and present tense forms for broader coverage
VERB_TAXONOMY = {
    "SUB": [
        # Past tense
        "sold", "gave", "spent", "lost", "ate", "used", "traded", "removed",
        "dropped", "threw", "donated", "paid", "left", "shared", "sent",
        # Present tense
        "sells", "gives", "spends", "loses", "eats", "uses", "trades", "removes",
        "drops", "throws", "donates", "pays", "leaves", "shares", "sends",
        # Base/infinitive
        "sell", "give", "spend", "lose", "eat", "use", "trade", "remove",
        "drop", "throw", "donate", "pay", "leave", "share", "send",
        # Gerund (-ing)
        "selling", "giving", "spending", "losing", "eating", "using",
        # Additional verbs
        "bakes", "baked", "bake", "baking",  # Uses ingredients = subtraction
        "takes", "took", "take", "taking",
    ],
    "ADD": [
        # Past tense
        "found", "received", "earned", "won", "picked", "got", "gained",
        "collected", "bought", "acquired", "added", "obtained", "gathered",
        # Present tense
        "finds", "receives", "earns", "wins", "picks", "gets", "gains",
        "collects", "buys", "acquires", "adds", "obtains", "gathers",
        # Base/infinitive
        "find", "receive", "earn", "win", "pick", "get", "gain",
        "collect", "buy", "acquire", "add", "obtain", "gather",
        # Gerund
        "finding", "receiving", "earning", "winning", "collecting",
    ],
    "SET": [
        "has", "have", "had", "having",
        "starts", "started", "start", "starting",
        "begins", "began", "begin", "beginning",
        "owns", "owned", "own", "owning",
        "contains", "contained", "contain", "containing",
        "holds", "held", "hold", "holding",
        # Production/creation verbs (initial quantity)
        "lays", "laid", "lay", "laying",  # "lays 16 eggs"
        "produces", "produced", "produce", "producing",
        "makes", "made", "make", "making",
        "grows", "grew", "grow", "growing",
    ],
    "MUL": [
        "times", "multiplied", "multiply", "multiplying",
        "doubled", "double", "doubles", "doubling",
        "tripled", "triple", "triples", "tripling",
        "each",
    ],
    "DIV": [
        "split", "splits", "splitting",
        "divided", "divide", "divides", "dividing",
        "shared equally", "shares equally",
        "distributed", "distribute", "distributes",
        "grouped", "group", "groups",
    ],
}

# =============================================================================
# Frozensets for fast O(1) membership checking
# These are the canonical pattern sets - import these, don't duplicate them!
# =============================================================================

# ADD patterns: words that indicate increase/addition
ADD_PATTERNS = frozenset([
    "more", "added", "gained", "found", "received", "bought", "collected",
    "earned", "got", "picked", "additional", "extra", "plus", "won",
    "acquired", "obtained", "gathered"
])

# SUB patterns: words that indicate decrease/subtraction
SUB_PATTERNS = frozenset([
    "less", "fewer", "sold", "gave", "lost", "spent", "used", "ate",
    "removed", "took", "subtracted", "minus", "traded", "dropped",
    "threw", "donated", "paid", "left", "shared", "sent"
])

# MUL patterns: words that indicate multiplication
MUL_PATTERNS = frozenset([
    "times", "twice", "double", "triple", "multiplied", "doubled", "tripled",
    "half", "third", "quarter", "each"
])

# DIV patterns: words that indicate division
DIV_PATTERNS = frozenset([
    "split", "divided", "shared", "distributed", "grouped", "each", "per", "every"
])

# Reference patterns: words that indicate comparison/reference to another entity
REFERENCE_PATTERNS = frozenset([
    "than", "as", "of"
])

# Build reverse lookup: verb -> (operation, base_confidence)
_VERB_TO_OP: dict[str, str] = {}
for op, verbs in VERB_TAXONOMY.items():
    for verb in verbs:
        _VERB_TO_OP[verb.lower()] = op


def extract_verb(text: str) -> Optional[str]:
    """
    Extract the main action verb from a span.

    Uses simple heuristic: find the first word that matches a known verb.
    This works well for simple math word problem spans.

    Args:
        text: Input text span

    Returns:
        The matched verb if found, None otherwise
    """
    # Normalize text
    text_lower = text.lower()

    # First check for multi-word phrases (like "shared equally")
    for op, verbs in VERB_TAXONOMY.items():
        for verb in verbs:
            if " " in verb and verb in text_lower:
                return verb

    # Extract words, handling punctuation
    words = re.findall(r'\b[a-z]+\b', text_lower)

    # Find first matching verb
    for word in words:
        if word in _VERB_TO_OP:
            return word

    return None


def classify_by_verb(text: str) -> Optional[Tuple[str, float]]:
    """
    Classify operation based on verb found in text.

    Args:
        text: Input text span

    Returns:
        Tuple of (operation, confidence) if a known verb is found, None otherwise.
        Confidence is based on verb specificity:
        - SUB/ADD verbs: 0.85 (highly specific)
        - MUL/DIV verbs: 0.80 (fairly specific)
        - SET verbs: 0.70 (less specific, "has" can be ambiguous)
    """
    verb = extract_verb(text)

    if verb is None:
        return None

    operation = _VERB_TO_OP.get(verb)
    if operation is None:
        return None

    # Special case: "gave him/her/them" - the pronoun is RECEIVING
    # "His mom gave him 5" → main entity (him) gains 5 = ADD
    # Check if verb is "gave" and followed by object pronoun
    if verb == "gave":
        text_lower = text.lower()
        # Object pronouns that indicate receiving
        object_pronouns = ["him", "her", "them", "me", "us"]
        for pronoun in object_pronouns:
            # Check for "gave [pronoun]" pattern
            if re.search(rf'\bgave\s+{pronoun}\b', text_lower):
                # The pronoun is receiving, so it's ADD not SUB
                return ("ADD", 0.85)

    # Special case: "sells for $X per" - price calculation = MUL
    # "She sells eggs for $2 per egg" → revenue = quantity * price = MUL
    if verb in ("sells", "sell", "sold", "selling"):
        text_lower = text.lower()
        # Check for price patterns: "for $X per", "at $X each", "for $X a"
        if re.search(r'for\s+\$?\d+.*\bper\b', text_lower):
            return ("MUL", 0.85)
        if re.search(r'at\s+\$?\d+.*\b(each|per)\b', text_lower):
            return ("MUL", 0.85)
        if re.search(r'for\s+\$?\d+\s+(a|an|each)\b', text_lower):
            return ("MUL", 0.85)

    # Assign confidence based on operation type
    # SUB/ADD verbs are highly specific to their operations
    # SET verbs like "has" can be more ambiguous
    confidence_map = {
        "SUB": 0.85,
        "ADD": 0.85,
        "SET": 0.70,
        "MUL": 0.80,
        "DIV": 0.80,
    }

    confidence = confidence_map.get(operation, 0.75)

    return (operation, confidence)


if __name__ == "__main__":
    test_cases = [
        ("She sold 5 apples", "SUB"),
        ("He found 3 more coins", "ADD"),
        ("Lisa has 12 apples", "SET"),
        ("They shared equally among 4 friends", "DIV"),
        ("He doubled his money", "MUL"),
        ("Mary gave away 7 books", "SUB"),
        ("Tom received 10 dollars", "ADD"),
        ("She started with 20 candies", "SET"),
    ]

    print("Verb Classifier Test Results")
    print("=" * 50)

    passed = 0
    failed = 0

    for text, expected_op in test_cases:
        result = classify_by_verb(text)
        verb = extract_verb(text)

        if result is None:
            status = "FAIL (no verb found)"
            failed += 1
        elif result[0] == expected_op:
            status = f"PASS (verb='{verb}', conf={result[1]:.2f})"
            passed += 1
        else:
            status = f"FAIL (got {result[0]}, expected {expected_op})"
            failed += 1

        print(f"  '{text}'")
        print(f"    Expected: {expected_op} -> {status}")
        print()

    print("=" * 50)
    print(f"Results: {passed}/{len(test_cases)} passed")
