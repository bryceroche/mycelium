"""Simple regex-based segmentation for math word problems.

NO GPU. NO QWEN. Just regex for:
1. Extracting numbers from text
2. Detecting entity names (capitalized words)
3. Basic clause structure

This replaces attention_segmenter.py for CPU-only inference.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Segment:
    """A segmented portion of text."""
    text: str
    segment_type: str  # "entity", "number", "operation"
    start_idx: int = 0
    end_idx: int = 0
    tokens: List[str] = field(default_factory=list)
    similarity_score: float = 1.0


@dataclass
class SegmentedSpan:
    """Result of segmenting a span."""
    original_text: str
    segments: List[Segment]
    reference_entity: Optional[str]  # Detected entity reference
    numbers: List[float]  # Extracted numbers


class SimpleSegmenter:
    """Lightweight regex-based segmenter for math word problems.

    Extracts:
    - Numbers (integers and decimals)
    - Named entities (capitalized words)
    - Reference entities (pronouns that refer to previous entities)
    """

    # Import from span_normalizer (single source of truth)
    from mycelium.span_normalizer import PRONOUNS

    # Number pattern: integers, decimals, with optional commas
    NUMBER_PATTERN = re.compile(r'\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\b')

    # Named entity pattern: Capitalized words (not at sentence start)
    ENTITY_PATTERN = re.compile(r'(?<=[.!?\s])\s*([A-Z][a-z]+)\b|^([A-Z][a-z]+)\b')

    def __init__(self):
        self._last_entity: Optional[str] = None

    def segment(self, text: str) -> SegmentedSpan:
        """Segment a clause into components.

        Args:
            text: A single clause like "Lisa has 12 apples"

        Returns:
            SegmentedSpan with numbers, entities, and reference
        """
        # Extract numbers
        numbers = []
        for match in self.NUMBER_PATTERN.finditer(text):
            num_str = match.group(1).replace(',', '')
            try:
                numbers.append(float(num_str))
            except ValueError:
                pass

        # Extract named entities (capitalized words)
        entities = []
        words = text.split()
        for i, word in enumerate(words):
            # Check if word starts with capital (and isn't first word or after punctuation)
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word and clean_word[0].isupper() and clean_word.lower() not in self.PRONOUNS:
                entities.append(clean_word)

        # Determine reference entity
        reference_entity = None
        text_lower = text.lower()

        # Check for pronouns that reference last entity
        for pronoun in self.PRONOUNS:
            if re.search(rf'\b{pronoun}\b', text_lower):
                reference_entity = self._last_entity
                break

        # If we found a new named entity, update last entity
        if entities:
            self._last_entity = entities[0]
            if reference_entity is None:
                reference_entity = entities[0]

        # Build segments
        segments = []

        # Add entity segment if found
        if entities:
            segments.append(Segment(
                text=entities[0],
                segment_type="entity",
                tokens=[entities[0]],
            ))

        # Add number segments
        for num in numbers:
            segments.append(Segment(
                text=str(num),
                segment_type="number",
                tokens=[str(num)],
            ))

        # Add operation segment (the whole text for now)
        segments.append(Segment(
            text=text,
            segment_type="operation",
            tokens=words,
        ))

        return SegmentedSpan(
            original_text=text,
            segments=segments,
            reference_entity=reference_entity,
            numbers=numbers,
        )

    def reset(self):
        """Reset state between problems."""
        self._last_entity = None


def test_segmenter():
    """Test the simple segmenter."""
    segmenter = SimpleSegmenter()

    test_cases = [
        "Lisa has 12 apples",
        "She sold 5 apples",
        "Tom found 3 more coins",
        "He gave 2 to Mary",
    ]

    print("=== Simple Segmenter Test ===\n")

    for text in test_cases:
        result = segmenter.segment(text)
        print(f"Text: {text}")
        print(f"  Numbers: {result.numbers}")
        print(f"  Reference: {result.reference_entity}")
        print(f"  Segments: {len(result.segments)}")
        print()


if __name__ == "__main__":
    test_segmenter()
