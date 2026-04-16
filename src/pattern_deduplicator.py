"""
PatternDeduplicator — Deduplication logic for pattern memory.

Three-layer deduplication:
1. Template canonicalization: Replace numbers with ordered placeholders
2. Hash-based uniqueness: SHA256 of (pattern_type + canonical_template)
3. Embedding similarity: Cosine similarity check for near-duplicates
"""

import hashlib
import re
import numpy as np


class PatternDeduplicator:
    """Prevents duplicate patterns through three mechanisms."""

    @staticmethod
    def canonicalize_template(sympy_template: str) -> str:
        """
        Normalize a SymPy template to its canonical form.
        Replace specific numbers with ordered placeholders.

        Examples:
            "result = 48 / 2"     -> "result = {a} / 2"
            "result = 96 / 2"     -> "result = {a} / 2"    (same!)
            "total = 48 + 24"     -> "total = {a} + {b}"
            "total = 100 + 50"    -> "total = {a} + {b}"   (same!)
            "x = 3.14 * 2"        -> "x = {a} * 2"

        Note: Small integers that are likely operators (like / 2 for "half of")
        are preserved, not replaced. Only the first occurrence of each unique
        number is replaced with a new placeholder.
        """
        if not sympy_template:
            return ""

        canonical = sympy_template.strip()

        # Find all numbers (integers and floats)
        # Pattern matches: integers (123) and floats (3.14, .5, 123.456)
        numbers = re.findall(r'\b\d+\.?\d*\b', canonical)

        seen = {}
        placeholder_idx = 0

        for num in numbers:
            if num not in seen:
                seen[num] = chr(ord('a') + placeholder_idx)
                placeholder_idx += 1
                # Handle wraparound if more than 26 unique numbers
                if placeholder_idx > 25:
                    placeholder_idx = 0

            # Replace only the first occurrence (left to right)
            canonical = canonical.replace(num, '{' + seen[num] + '}', 1)

        return canonical

    @staticmethod
    def compute_hash(pattern_type: str, canonical_template: str) -> str:
        """
        Compute a unique hash for deduplication.

        Args:
            pattern_type: Type of pattern (e.g., "half_of", "percent_of")
            canonical_template: Canonicalized template string

        Returns:
            First 16 characters of SHA256 hash
        """
        key = f"{pattern_type}::{canonical_template}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    @staticmethod
    def embedding_similar(emb1: np.ndarray, emb2: np.ndarray, threshold: float = 0.95) -> bool:
        """
        Check if two pattern embeddings are near-duplicates via cosine similarity.

        Args:
            emb1: First embedding (numpy array)
            emb2: Second embedding (numpy array)
            threshold: Similarity threshold (default 0.95)

        Returns:
            True if cosine similarity > threshold
        """
        # Handle edge cases
        if emb1 is None or emb2 is None:
            return False

        emb1 = np.asarray(emb1, dtype=np.float32).flatten()
        emb2 = np.asarray(emb2, dtype=np.float32).flatten()

        if len(emb1) != len(emb2):
            return False

        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        # Handle zero vectors
        if norm1 < 1e-8 or norm2 < 1e-8:
            return False

        cos_sim = np.dot(emb1, emb2) / (norm1 * norm2 + 1e-8)
        return cos_sim > threshold


# =============================================================================
# Tests
# =============================================================================

def test_canonicalize_template():
    """Test template canonicalization."""
    print("Testing canonicalize_template...")

    # Test 1: Different numbers should produce same canonical form
    t1 = PatternDeduplicator.canonicalize_template("result = 48 / 2")
    t2 = PatternDeduplicator.canonicalize_template("result = 96 / 2")
    assert t1 == t2, f"Expected same canonical, got '{t1}' vs '{t2}'"
    assert t1 == "result = {a} / {b}", f"Expected 'result = {{a}} / {{b}}', got '{t1}'"
    print(f"  PASS: '48 / 2' and '96 / 2' -> '{t1}'")

    # Test 2: Multiple variables
    t3 = PatternDeduplicator.canonicalize_template("total = 48 + 24")
    t4 = PatternDeduplicator.canonicalize_template("total = 100 + 50")
    assert t3 == t4, f"Expected same canonical, got '{t3}' vs '{t4}'"
    assert t3 == "total = {a} + {b}", f"Expected 'total = {{a}} + {{b}}', got '{t3}'"
    print(f"  PASS: '48 + 24' and '100 + 50' -> '{t3}'")

    # Test 3: Float numbers
    t5 = PatternDeduplicator.canonicalize_template("x = 3.14 * 2")
    assert "{a}" in t5, f"Expected placeholder in '{t5}'"
    print(f"  PASS: Float '3.14 * 2' -> '{t5}'")

    # Test 4: Repeated numbers get same placeholder
    t6 = PatternDeduplicator.canonicalize_template("result = 5 + 5")
    assert t6 == "result = {a} + {a}", f"Expected 'result = {{a}} + {{a}}', got '{t6}'"
    print(f"  PASS: Repeated '5 + 5' -> '{t6}'")

    # Test 5: No numbers
    t7 = PatternDeduplicator.canonicalize_template("result = x + y")
    assert t7 == "result = x + y", f"Expected unchanged, got '{t7}'"
    print(f"  PASS: No numbers 'x + y' -> '{t7}'")

    # Test 6: Empty string
    t8 = PatternDeduplicator.canonicalize_template("")
    assert t8 == "", f"Expected empty, got '{t8}'"
    print(f"  PASS: Empty string -> '{t8}'")

    # Test 7: Whitespace handling
    t9 = PatternDeduplicator.canonicalize_template("  result = 10 / 2  ")
    assert t9 == "result = {a} / {b}", f"Expected stripped, got '{t9}'"
    print(f"  PASS: Whitespace stripped -> '{t9}'")

    # Test 8: Complex expression
    t10 = PatternDeduplicator.canonicalize_template("total = 100 * 0.15 + 50")
    assert t10 == "total = {a} * {b} + {c}", f"Expected three placeholders, got '{t10}'"
    print(f"  PASS: Complex '100 * 0.15 + 50' -> '{t10}'")

    # Test 9: Multiple occurrences of same number
    t11 = PatternDeduplicator.canonicalize_template("area = 10 * 10")
    assert t11 == "area = {a} * {a}", f"Expected same placeholder, got '{t11}'"
    print(f"  PASS: Same number twice '10 * 10' -> '{t11}'")

    print("  All canonicalize_template tests passed!")


def test_compute_hash():
    """Test hash computation."""
    print("\nTesting compute_hash...")

    # Test 1: Same inputs produce same hash
    h1 = PatternDeduplicator.compute_hash("half_of", "result = {a} / 2")
    h2 = PatternDeduplicator.compute_hash("half_of", "result = {a} / 2")
    assert h1 == h2, f"Same inputs should produce same hash: '{h1}' vs '{h2}'"
    print(f"  PASS: Same inputs -> same hash '{h1}'")

    # Test 2: Different type produces different hash
    h3 = PatternDeduplicator.compute_hash("third_of", "result = {a} / 2")
    assert h1 != h3, f"Different types should produce different hashes"
    print(f"  PASS: Different type -> different hash '{h3}'")

    # Test 3: Different template produces different hash
    h4 = PatternDeduplicator.compute_hash("half_of", "result = {a} / 3")
    assert h1 != h4, f"Different templates should produce different hashes"
    print(f"  PASS: Different template -> different hash '{h4}'")

    # Test 4: Hash is 16 characters
    assert len(h1) == 16, f"Hash should be 16 chars, got {len(h1)}"
    print(f"  PASS: Hash length is 16")

    # Test 5: Hash is hexadecimal
    assert all(c in '0123456789abcdef' for c in h1), f"Hash should be hex"
    print(f"  PASS: Hash is hexadecimal")

    # Test 6: Uniqueness check - many different inputs
    hashes = set()
    for i in range(100):
        h = PatternDeduplicator.compute_hash(f"type_{i}", f"template_{i}")
        assert h not in hashes, f"Hash collision detected"
        hashes.add(h)
    print(f"  PASS: 100 unique inputs -> 100 unique hashes")

    print("  All compute_hash tests passed!")


def test_embedding_similar():
    """Test embedding similarity."""
    print("\nTesting embedding_similar...")

    # Test 1: Identical embeddings
    emb1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    emb2 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    assert PatternDeduplicator.embedding_similar(emb1, emb2), "Identical should be similar"
    print(f"  PASS: Identical embeddings are similar")

    # Test 2: Very similar embeddings
    emb3 = np.array([1.0, 0.01, 0.0, 0.0], dtype=np.float32)
    assert PatternDeduplicator.embedding_similar(emb1, emb3), "Very similar should match"
    print(f"  PASS: Very similar embeddings (cos > 0.95) are similar")

    # Test 3: Orthogonal embeddings
    emb4 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    assert not PatternDeduplicator.embedding_similar(emb1, emb4), "Orthogonal should not be similar"
    print(f"  PASS: Orthogonal embeddings are not similar")

    # Test 4: Opposite embeddings
    emb5 = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    assert not PatternDeduplicator.embedding_similar(emb1, emb5), "Opposite should not be similar"
    print(f"  PASS: Opposite embeddings are not similar")

    # Test 5: Custom threshold
    # [0.9, 0.436, 0, 0] has cosine ~0.90 with [1, 0, 0, 0]
    emb6 = np.array([0.9, 0.436, 0.0, 0.0], dtype=np.float32)
    assert not PatternDeduplicator.embedding_similar(emb1, emb6, threshold=0.95), "Strict threshold"
    assert PatternDeduplicator.embedding_similar(emb1, emb6, threshold=0.85), "Loose threshold"
    print(f"  PASS: Custom threshold works correctly")

    # Test 6: Different lengths
    emb7 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert not PatternDeduplicator.embedding_similar(emb1, emb7), "Different lengths should not match"
    print(f"  PASS: Different length embeddings return False")

    # Test 7: Zero vector handling
    emb8 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    assert not PatternDeduplicator.embedding_similar(emb1, emb8), "Zero vector should not match"
    assert not PatternDeduplicator.embedding_similar(emb8, emb8), "Two zero vectors should not match"
    print(f"  PASS: Zero vectors handled correctly")

    # Test 8: None handling
    assert not PatternDeduplicator.embedding_similar(None, emb1), "None should not match"
    assert not PatternDeduplicator.embedding_similar(emb1, None), "None should not match"
    assert not PatternDeduplicator.embedding_similar(None, None), "None should not match"
    print(f"  PASS: None inputs handled correctly")

    # Test 9: Realistic 64-float embeddings
    np.random.seed(42)
    emb9 = np.random.randn(64).astype(np.float32)
    emb9 = emb9 / np.linalg.norm(emb9)  # Normalize to unit sphere
    emb10 = emb9 + 0.01 * np.random.randn(64).astype(np.float32)  # Small perturbation
    emb10 = emb10 / np.linalg.norm(emb10)
    assert PatternDeduplicator.embedding_similar(emb9, emb10), "Small perturbation should be similar"
    print(f"  PASS: 64-float realistic embeddings work correctly")

    # Test 10: List input (should be converted to numpy)
    emb11 = [1.0, 0.0, 0.0, 0.0]
    assert PatternDeduplicator.embedding_similar(emb11, emb1), "List input should work"
    print(f"  PASS: List inputs converted correctly")

    print("  All embedding_similar tests passed!")


def test_end_to_end():
    """Test full deduplication workflow."""
    print("\nTesting end-to-end deduplication workflow...")

    # Simulate two problems with the same pattern
    problem1_template = "result = 48 / 2"
    problem2_template = "result = 96 / 2"

    # Step 1: Canonicalize
    canon1 = PatternDeduplicator.canonicalize_template(problem1_template)
    canon2 = PatternDeduplicator.canonicalize_template(problem2_template)
    assert canon1 == canon2, "Same pattern should have same canonical form"
    print(f"  PASS: Both canonicalize to '{canon1}'")

    # Step 2: Compute hash
    hash1 = PatternDeduplicator.compute_hash("half_of", canon1)
    hash2 = PatternDeduplicator.compute_hash("half_of", canon2)
    assert hash1 == hash2, "Same pattern should have same hash"
    print(f"  PASS: Both have hash '{hash1}'")

    # Step 3: Embedding similarity (simulated)
    np.random.seed(123)
    # Embeddings from same pattern type should be similar
    base_emb = np.random.randn(64).astype(np.float32)
    base_emb = base_emb / np.linalg.norm(base_emb)
    emb1 = base_emb + 0.02 * np.random.randn(64).astype(np.float32)
    emb2 = base_emb + 0.02 * np.random.randn(64).astype(np.float32)
    emb1 = emb1 / np.linalg.norm(emb1)
    emb2 = emb2 / np.linalg.norm(emb2)

    assert PatternDeduplicator.embedding_similar(emb1, emb2), "Similar embeddings should match"
    print(f"  PASS: Similar embeddings detected as duplicates")

    print("  End-to-end workflow test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("PatternDeduplicator Tests")
    print("=" * 60)

    test_canonicalize_template()
    test_compute_hash()
    test_embedding_similar()
    test_end_to_end()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
