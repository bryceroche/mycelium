"""
PatternMemory: SQLite-backed long-term memory for mathematical reasoning patterns.

The model queries by page embedding similarity, retrieves templates like
"half of X -> X/2", and uses them to formulate SymPy expressions. Patterns
accumulate over training with success/failure tracking. Deduplication prevents
bloat. The database is the system's long-term memory.

Three memory timescales:
- Atoms (attention control): HOW to read the problem (microsecond)
- Pages (notebook): WHAT we know about THIS problem (second)
- Pattern DB (SQLite): WHAT works for problems LIKE this (permanent)
"""

import sqlite3
import hashlib
import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import numpy as np

# Try to import PatternDeduplicator from separate module
# If not available, use a stub implementation
try:
    from src.pattern_deduplicator import PatternDeduplicator
except ImportError:
    class PatternDeduplicator:
        """
        Stub implementation of PatternDeduplicator.
        Provides template canonicalization and hashing for deduplication.
        Will be replaced by full implementation in src/pattern_deduplicator.py
        """

        @staticmethod
        def canonicalize_template(sympy_template: str) -> str:
            """
            Normalize a SymPy template to its canonical form.
            Replace specific numbers with placeholders.

            "result = 48 / 2"     -> "result = {a} / 2"
            "result = 96 / 2"     -> "result = {a} / 2"    (same!)
            "total = 48 + 24"     -> "total = {a} + {b}"
            """
            canonical = sympy_template.strip()

            # Replace numbers with ordered placeholders
            numbers = re.findall(r'\b\d+\.?\d*\b', canonical)
            seen = {}
            placeholder_idx = 0

            for num in numbers:
                if num not in seen:
                    seen[num] = chr(ord('a') + placeholder_idx)
                    placeholder_idx += 1
                    # Only go up to 'z'
                    if placeholder_idx > 25:
                        placeholder_idx = 25
                canonical = canonical.replace(num, '{' + seen[num] + '}', 1)

            return canonical

        @staticmethod
        def compute_hash(pattern_type: str, canonical_template: str) -> str:
            """Unique hash for deduplication."""
            key = f"{pattern_type}::{canonical_template}"
            return hashlib.sha256(key.encode()).hexdigest()[:16]

        @staticmethod
        def embedding_similar(emb1: np.ndarray, emb2: np.ndarray,
                             threshold: float = 0.95) -> bool:
            """Check if two pattern embeddings are near-duplicates."""
            cos_sim = np.dot(emb1, emb2) / (
                np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8
            )
            return float(cos_sim) > threshold


class PatternMemory:
    """
    Long-term memory for mathematical reasoning patterns.
    SQLite-backed. Deduplication built in. Self-improving.

    Six stored procedures:
    1. query()          - find similar patterns by cosine similarity
    2. store()          - store new pattern with deduplication
    3. record_outcome() - track success/failure
    4. prune()          - remove bad/stale patterns
    5. stats()          - print summary statistics
    6. export_json()    - export for analysis
    """

    def __init__(self, db_path: Union[str, Path] = "pattern_memory.db"):
        """
        Initialize pattern memory with SQLite backend.

        Args:
            db_path: Path to SQLite database file. Use ":memory:" for testing.
        """
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self._create_tables()

    def _create_tables(self):
        """Create the patterns table and indexes."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                -- Pattern identity
                pattern_type TEXT NOT NULL,
                pattern_hash TEXT UNIQUE NOT NULL,

                -- Content
                sympy_template TEXT NOT NULL,
                canonical_template TEXT NOT NULL,
                description TEXT,
                example_problem TEXT,

                -- Embedding for similarity search (64 floats as BLOB)
                page_embedding BLOB NOT NULL,

                -- Success tracking
                success_count INTEGER DEFAULT 1,
                fail_count INTEGER DEFAULT 0,

                -- Metadata
                created_epoch INTEGER DEFAULT 0,
                last_used_epoch INTEGER DEFAULT 0,
                last_updated TEXT DEFAULT (datetime('now'))
            );

            -- Index for fast type-based lookup
            CREATE INDEX IF NOT EXISTS idx_pattern_type ON patterns(pattern_type);

            -- Unique index enforces hash-based deduplication
            CREATE UNIQUE INDEX IF NOT EXISTS idx_pattern_hash ON patterns(pattern_hash);
        """)
        self.conn.commit()

    def _embedding_to_numpy(self, embedding) -> np.ndarray:
        """Convert embedding (tensor or ndarray) to float32 numpy array."""
        if hasattr(embedding, 'detach'):
            # PyTorch tensor - convert to float32 BEFORE numpy (BFloat16 not supported)
            return embedding.detach().float().cpu().numpy()
        elif isinstance(embedding, np.ndarray):
            return embedding.astype(np.float32)
        else:
            return np.array(embedding, dtype=np.float32)

    # =========================================
    # STORED PROCEDURE 1: QUERY (retrieve patterns)
    # =========================================
    def query(self, page_embedding, top_k: int = 3,
              min_success_rate: float = 0.3) -> List[Dict[str, Any]]:
        """
        Find patterns similar to the current problem's page embedding.

        Called at: pass 1 (after initial problem encoding)

        Args:
            page_embedding: 64-float embedding (tensor or ndarray)
            top_k: Number of results to return
            min_success_rate: Filter out patterns with lower success rate
                              (only if 5+ uses)

        Returns:
            List of dicts with keys: score, pattern_id, template, canonical,
            type, description, success_rate, total_uses

        Scoring: cosine_similarity * (0.5 + 0.5 * success_rate)
        """
        query_np = self._embedding_to_numpy(page_embedding)

        rows = self.conn.execute(
            "SELECT id, page_embedding, sympy_template, pattern_type, "
            "canonical_template, description, success_count, fail_count "
            "FROM patterns"
        ).fetchall()

        if not rows:
            return []

        scored = []
        for (row_id, emb_bytes, template, ptype, canonical,
             desc, successes, failures) in rows:

            stored_np = np.frombuffer(emb_bytes, dtype=np.float32)

            # Cosine similarity
            query_norm = np.linalg.norm(query_np)
            stored_norm = np.linalg.norm(stored_np)
            if query_norm < 1e-8 or stored_norm < 1e-8:
                cos_sim = 0.0
            else:
                cos_sim = float(np.dot(query_np, stored_np) /
                               (query_norm * stored_norm))

            # Success rate
            total_uses = successes + failures
            success_rate = successes / max(total_uses, 1)

            # Skip patterns with poor track record (but only if enough uses)
            if total_uses >= 5 and success_rate < min_success_rate:
                continue

            # Combined score: similarity weighted by success rate
            score = cos_sim * (0.5 + 0.5 * success_rate)

            scored.append({
                'score': score,
                'pattern_id': row_id,
                'template': template,
                'canonical': canonical,
                'type': ptype,
                'description': desc,
                'success_rate': success_rate,
                'total_uses': total_uses,
            })

        scored.sort(key=lambda x: x['score'], reverse=True)
        return scored[:top_k]

    # =========================================
    # STORED PROCEDURE 2: STORE (add new pattern)
    # =========================================
    def store(self, page_embedding, sympy_template: str,
              pattern_type: str = "auto", description: str = "",
              example_problem: str = "", epoch: int = 0) -> int:
        """
        Store a successful reasoning pattern with deduplication.

        Called at: end of training step when answer is correct

        Args:
            page_embedding: 64-float embedding (tensor or ndarray)
            sympy_template: SymPy expression template (e.g., "result = {a} / 2")
            pattern_type: Category (e.g., "half_of", "percent_of")
            description: Human-readable description
            example_problem: Example problem text (truncated)
            epoch: Current training epoch

        Returns:
            pattern_id (new or existing)

        Deduplication:
            1. Canonicalize template (replace numbers with placeholders)
            2. Compute hash(type + canonical)
            3. If hash exists: increment success_count (merge)
            4. Otherwise: insert new pattern
        """
        embedding_np = self._embedding_to_numpy(page_embedding)
        embedding_bytes = embedding_np.tobytes()

        # Canonicalize for deduplication
        canonical = PatternDeduplicator.canonicalize_template(sympy_template)
        pattern_hash = PatternDeduplicator.compute_hash(pattern_type, canonical)

        # Try insert (hash-based dedup via UNIQUE constraint)
        try:
            cursor = self.conn.execute(
                "INSERT INTO patterns "
                "(pattern_type, pattern_hash, sympy_template, canonical_template, "
                "description, example_problem, page_embedding, "
                "success_count, created_epoch, last_used_epoch) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
                (pattern_type, pattern_hash, sympy_template, canonical,
                 description, example_problem, embedding_bytes, epoch, epoch)
            )
            self.conn.commit()
            return cursor.lastrowid

        except sqlite3.IntegrityError:
            # Hash collision -> duplicate pattern. Merge: increment success count
            # Update embedding to most recent (could use running average instead)
            self.conn.execute(
                "UPDATE patterns SET success_count = success_count + 1, "
                "last_used_epoch = ?, last_updated = datetime('now'), "
                "page_embedding = ? "
                "WHERE pattern_hash = ?",
                (epoch, embedding_bytes, pattern_hash)
            )
            self.conn.commit()

            cursor = self.conn.execute(
                "SELECT id FROM patterns WHERE pattern_hash = ?", (pattern_hash,)
            )
            return cursor.fetchone()[0]

    # =========================================
    # STORED PROCEDURE 3: RECORD_OUTCOME (track success/failure)
    # =========================================
    def record_outcome(self, pattern_id: int, success: bool,
                       epoch: int = 0) -> None:
        """
        Update a pattern's track record after using it.

        Called at: end of training/inference step

        Args:
            pattern_id: ID of the pattern that was used
            success: Whether the answer was correct
            epoch: Current epoch
        """
        if success:
            self.conn.execute(
                "UPDATE patterns SET success_count = success_count + 1, "
                "last_used_epoch = ?, last_updated = datetime('now') "
                "WHERE id = ?", (epoch, pattern_id)
            )
        else:
            self.conn.execute(
                "UPDATE patterns SET fail_count = fail_count + 1, "
                "last_used_epoch = ?, last_updated = datetime('now') "
                "WHERE id = ?", (epoch, pattern_id)
            )
        self.conn.commit()

    # =========================================
    # STORED PROCEDURE 4: PRUNE (remove bad patterns)
    # =========================================
    def prune(self, min_uses: int = 10, max_failure_rate: float = 0.7,
              stale_epochs: int = 20, current_epoch: int = 0) -> tuple:
        """
        Remove patterns that consistently fail or haven't been used recently.

        Called at: end of each epoch

        Args:
            min_uses: Minimum uses before considering for failure-based pruning
            max_failure_rate: Remove patterns with failure rate above this
            stale_epochs: Remove patterns not used in this many epochs
            current_epoch: Current training epoch

        Returns:
            (deleted_bad, deleted_stale) counts

        Removes:
            - Patterns used min_uses+ times with >max_failure_rate failure rate
            - Patterns not used in stale_epochs+ epochs (if low total uses)
        """
        # Remove high-failure patterns
        cursor = self.conn.execute(
            "DELETE FROM patterns WHERE "
            "(success_count + fail_count) >= ? AND "
            "CAST(fail_count AS REAL) / (success_count + fail_count) > ?",
            (min_uses, max_failure_rate)
        )
        deleted_bad = cursor.rowcount

        # Remove stale patterns (but not well-established ones)
        cursor = self.conn.execute(
            "DELETE FROM patterns WHERE "
            "(? - last_used_epoch) > ? AND (success_count + fail_count) < ?",
            (current_epoch, stale_epochs, min_uses)
        )
        deleted_stale = cursor.rowcount

        self.conn.commit()

        if deleted_bad + deleted_stale > 0:
            print(f"Pruned {deleted_bad} failed + {deleted_stale} stale patterns")

        return deleted_bad, deleted_stale

    # =========================================
    # STORED PROCEDURE 5: STATS (inspect memory)
    # =========================================
    def stats(self) -> Dict[str, Any]:
        """
        Print and return summary statistics about the pattern memory.

        Called at: end of epoch, for diagnostics

        Returns:
            Dict with total, avg_success_rate, top_patterns, type_counts
        """
        total = self.conn.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]

        if total == 0:
            print("Pattern memory: empty")
            return {'total': 0, 'avg_success_rate': 0.0,
                    'top_patterns': [], 'type_counts': []}

        # Average success rate
        avg_success = self.conn.execute(
            "SELECT AVG(CAST(success_count AS REAL) / "
            "MAX(success_count + fail_count, 1)) FROM patterns"
        ).fetchone()[0] or 0.0

        # Top patterns by usage
        top_patterns = self.conn.execute(
            "SELECT pattern_type, canonical_template, success_count, fail_count "
            "FROM patterns ORDER BY success_count DESC LIMIT 5"
        ).fetchall()

        # Pattern type distribution
        type_counts = self.conn.execute(
            "SELECT pattern_type, COUNT(*), SUM(success_count) "
            "FROM patterns GROUP BY pattern_type ORDER BY COUNT(*) DESC LIMIT 10"
        ).fetchall()

        # Print summary
        print(f"Pattern memory: {total} patterns")
        print(f"Avg success rate: {avg_success:.1%}")

        if top_patterns:
            print(f"\nTop 5 most-used patterns:")
            for ptype, template, succ, fail in top_patterns:
                rate = succ / max(succ + fail, 1)
                print(f"  [{ptype}] {template} ({succ} wins, {fail} fails, {rate:.0%})")

        if type_counts:
            print(f"\nPattern types:")
            for ptype, count, total_succ in type_counts:
                total_succ = total_succ or 0
                print(f"  {ptype}: {count} patterns, {int(total_succ)} total successes")

        return {
            'total': total,
            'avg_success_rate': avg_success,
            'top_patterns': top_patterns,
            'type_counts': type_counts,
        }

    # =========================================
    # STORED PROCEDURE 6: EXPORT_JSON (save for analysis)
    # =========================================
    def export_json(self, path: Union[str, Path] = "pattern_memory_export.json"
                   ) -> int:
        """
        Export all patterns to JSON for analysis.

        Args:
            path: Output file path

        Returns:
            Number of patterns exported
        """
        rows = self.conn.execute(
            "SELECT id, pattern_type, canonical_template, sympy_template, "
            "description, success_count, fail_count, created_epoch, "
            "last_used_epoch, example_problem "
            "FROM patterns ORDER BY success_count DESC"
        ).fetchall()

        patterns = []
        for row in rows:
            patterns.append({
                'id': row[0],
                'type': row[1],
                'canonical': row[2],
                'template': row[3],
                'description': row[4],
                'successes': row[5],
                'failures': row[6],
                'success_rate': row[5] / max(row[5] + row[6], 1),
                'created_epoch': row[7],
                'last_used_epoch': row[8],
                'example_problem': row[9],
            })

        with open(path, 'w') as f:
            json.dump(patterns, f, indent=2)

        print(f"Exported {len(patterns)} patterns to {path}")
        return len(patterns)

    # =========================================
    # UTILITY METHODS
    # =========================================
    def close(self):
        """Close the database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def count(self) -> int:
        """Return total number of patterns."""
        return self.conn.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]

    def get_pattern(self, pattern_id: int) -> Optional[Dict[str, Any]]:
        """Get a single pattern by ID."""
        row = self.conn.execute(
            "SELECT id, pattern_type, canonical_template, sympy_template, "
            "description, success_count, fail_count, created_epoch, "
            "last_used_epoch, page_embedding "
            "FROM patterns WHERE id = ?", (pattern_id,)
        ).fetchone()

        if row is None:
            return None

        return {
            'id': row[0],
            'type': row[1],
            'canonical': row[2],
            'template': row[3],
            'description': row[4],
            'successes': row[5],
            'failures': row[6],
            'success_rate': row[5] / max(row[5] + row[6], 1),
            'created_epoch': row[7],
            'last_used_epoch': row[8],
            'embedding': np.frombuffer(row[9], dtype=np.float32),
        }


def classify_pattern(sympy_steps: List[str]) -> str:
    """
    Auto-classify a list of SymPy steps into a pattern type.
    Simple keyword heuristic -- not ML, just string matching.

    Args:
        sympy_steps: List of SymPy expression strings

    Returns:
        Pattern type string
    """
    combined = " ".join(sympy_steps).lower()

    if "/ 2" in combined or "* 0.5" in combined:
        return "half_of"
    elif "/ 3" in combined:
        return "third_of"
    elif "/ 100" in combined or "percent" in combined or "* 0.01" in combined:
        return "percent_of"
    elif combined.count("+") >= 2:
        return "sum_of_parts"
    elif "*" in combined and "+" in combined:
        return "multiply_then_add"
    elif "-" in combined:
        return "difference"
    elif "*" in combined:
        return "multiply"
    elif "/" in combined:
        return "divide"
    else:
        return "other"


# =========================================
# TESTS
# =========================================
def run_tests():
    """Comprehensive tests for PatternMemory."""
    import tempfile
    import os

    print("=" * 60)
    print("Running PatternMemory Tests")
    print("=" * 60)

    # Test 1: Basic initialization with in-memory database
    print("\n[Test 1] Initialization...")
    memory = PatternMemory(":memory:")
    assert memory.count() == 0, "New database should be empty"
    print("  PASSED: Database initialized empty")

    # Test 2: Store a pattern
    print("\n[Test 2] Store pattern...")
    embedding1 = np.random.randn(64).astype(np.float32)
    embedding1 = embedding1 / np.linalg.norm(embedding1) * np.sqrt(64)  # Hypersphere

    pattern_id = memory.store(
        page_embedding=embedding1,
        sympy_template="result = 48 / 2",
        pattern_type="half_of",
        description="Half of a quantity",
        example_problem="What is half of 48?",
        epoch=1
    )
    assert pattern_id == 1, f"First pattern should have ID 1, got {pattern_id}"
    assert memory.count() == 1, "Should have 1 pattern"
    print(f"  PASSED: Stored pattern with ID {pattern_id}")

    # Test 3: Deduplication - same pattern canonical form
    print("\n[Test 3] Deduplication (same canonical form)...")
    embedding2 = np.random.randn(64).astype(np.float32)
    embedding2 = embedding2 / np.linalg.norm(embedding2) * np.sqrt(64)

    # "96 / 2" canonicalizes to same "{a} / 2" as "48 / 2"
    pattern_id2 = memory.store(
        page_embedding=embedding2,
        sympy_template="result = 96 / 2",
        pattern_type="half_of",
        description="Half of a quantity",
        epoch=2
    )
    assert pattern_id2 == 1, f"Should return existing ID 1, got {pattern_id2}"
    assert memory.count() == 1, "Should still have 1 pattern (deduplicated)"

    # Check success count incremented
    pattern = memory.get_pattern(1)
    assert pattern['successes'] == 2, f"Success count should be 2, got {pattern['successes']}"
    print("  PASSED: Duplicate merged, success count incremented to 2")

    # Test 4: Store different pattern
    print("\n[Test 4] Store different pattern...")
    embedding3 = np.random.randn(64).astype(np.float32)
    embedding3 = embedding3 / np.linalg.norm(embedding3) * np.sqrt(64)

    pattern_id3 = memory.store(
        page_embedding=embedding3,
        sympy_template="total = 10 + 20 + 30",
        pattern_type="sum_of_parts",
        description="Sum of multiple parts",
        epoch=2
    )
    assert pattern_id3 == 2, f"New pattern should have ID 2, got {pattern_id3}"
    assert memory.count() == 2, "Should have 2 patterns"
    print(f"  PASSED: New pattern stored with ID {pattern_id3}")

    # Test 5: Query by embedding similarity
    print("\n[Test 5] Query by embedding similarity...")
    # Query with embedding1 should find the "half_of" pattern with high similarity
    # Note: embedding1 was stored, so querying with it should return perfect match
    results = memory.query(embedding1, top_k=3)
    assert len(results) > 0, "Should find at least one result"
    # The best match should be half_of since we query with its own embedding
    # But note: embedding was overwritten by embedding2 during dedup test
    # So we check that query works and returns reasonable results
    assert len(results) == 2, f"Should find 2 patterns, got {len(results)}"
    # Score can be negative for random embeddings - just verify we get results sorted
    assert results[0]['score'] >= results[1]['score'], "Results should be sorted by score"
    scores_str = ", ".join([f"{r['score']:.3f}" for r in results])
    print(f"  PASSED: Query returned {len(results)} results, scores: [{scores_str}]")

    # Test 6: Record outcome
    print("\n[Test 6] Record outcome...")
    memory.record_outcome(pattern_id=1, success=True, epoch=3)
    pattern = memory.get_pattern(1)
    assert pattern['successes'] == 3, f"Success count should be 3, got {pattern['successes']}"

    memory.record_outcome(pattern_id=1, success=False, epoch=3)
    pattern = memory.get_pattern(1)
    assert pattern['failures'] == 1, f"Fail count should be 1, got {pattern['failures']}"
    print("  PASSED: Outcome recorded correctly")

    # Test 7: Pruning - bad patterns (use separate memory instance)
    print("\n[Test 7] Pruning bad patterns...")
    prune_mem = PatternMemory(":memory:")

    # Add a pattern that will fail a lot
    embedding_bad = np.random.randn(64).astype(np.float32)
    bad_id = prune_mem.store(
        page_embedding=embedding_bad,
        sympy_template="wrong = 1 + 1",
        pattern_type="bad_pattern",
        epoch=0
    )
    # Record many failures (16 total = 1 success from store + 15 failures = 93.75% fail rate)
    for _ in range(15):
        prune_mem.record_outcome(bad_id, success=False, epoch=1)

    pattern_before = prune_mem.get_pattern(bad_id)
    assert pattern_before is not None, "Bad pattern should exist before prune"

    deleted_bad, deleted_stale = prune_mem.prune(
        min_uses=10, max_failure_rate=0.7, stale_epochs=20, current_epoch=5
    )

    pattern_after = prune_mem.get_pattern(bad_id)
    assert pattern_after is None, "Bad pattern should be deleted after prune"
    assert deleted_bad >= 1, f"Should have deleted at least 1 bad pattern, got {deleted_bad}"
    print(f"  PASSED: Pruned {deleted_bad} bad patterns")
    prune_mem.close()

    # Test 8: Pruning - stale patterns (use separate memory instance)
    print("\n[Test 8] Pruning stale patterns...")
    stale_mem = PatternMemory(":memory:")

    embedding_stale = np.random.randn(64).astype(np.float32)
    stale_id = stale_mem.store(
        page_embedding=embedding_stale,
        sympy_template="old = 5 * 5",
        pattern_type="stale_pattern",
        epoch=0
    )
    # Don't use it for many epochs
    deleted_bad, deleted_stale = stale_mem.prune(
        min_uses=10, max_failure_rate=0.7, stale_epochs=20, current_epoch=50
    )

    pattern_after = stale_mem.get_pattern(stale_id)
    assert pattern_after is None, "Stale pattern should be deleted after prune"
    assert deleted_stale >= 1, f"Should have deleted at least 1 stale pattern, got {deleted_stale}"
    print(f"  PASSED: Pruned {deleted_stale} stale patterns")
    stale_mem.close()

    # Test 9: Stats
    print("\n[Test 9] Stats...")
    stats = memory.stats()
    assert 'total' in stats, "Stats should include total"
    assert 'avg_success_rate' in stats, "Stats should include avg_success_rate"
    print("  PASSED: Stats returned correctly")

    # Test 10: Export JSON
    print("\n[Test 10] Export JSON...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name

    try:
        count = memory.export_json(temp_path)
        assert count >= 2, f"Should export at least 2 patterns, got {count}"

        with open(temp_path, 'r') as f:
            exported = json.load(f)
        assert len(exported) >= 2, "Exported JSON should have at least 2 patterns"
        assert 'type' in exported[0], "Exported pattern should have 'type'"
        assert 'success_rate' in exported[0], "Exported pattern should have 'success_rate'"
        print(f"  PASSED: Exported {count} patterns to JSON")
    finally:
        os.unlink(temp_path)

    # Test 11: Pattern canonicalization
    print("\n[Test 11] Template canonicalization...")
    dedup = PatternDeduplicator()

    canon1 = dedup.canonicalize_template("result = 48 / 2")
    canon2 = dedup.canonicalize_template("result = 96 / 2")
    assert canon1 == canon2, f"Canonical forms should match: '{canon1}' vs '{canon2}'"

    canon3 = dedup.canonicalize_template("total = 10 + 20 + 30")
    assert canon3 == "total = {a} + {b} + {c}", f"Wrong canonical: {canon3}"

    canon4 = dedup.canonicalize_template("x = 3.14 * 2")
    assert canon4 == "x = {a} * {b}", f"Should handle floats: {canon4}"
    print("  PASSED: Canonicalization works correctly")

    # Test 12: classify_pattern function
    print("\n[Test 12] Pattern classification...")
    assert classify_pattern(["result = x / 2"]) == "half_of"
    assert classify_pattern(["result = x * 0.5"]) == "half_of"
    assert classify_pattern(["result = x / 3"]) == "third_of"
    assert classify_pattern(["result = x * y / 100"]) == "percent_of"
    assert classify_pattern(["total = a + b + c"]) == "sum_of_parts"
    assert classify_pattern(["result = a * b + c"]) == "multiply_then_add"
    assert classify_pattern(["result = a - b"]) == "difference"
    assert classify_pattern(["result = a * b"]) == "multiply"
    assert classify_pattern(["result = a / b"]) == "divide"
    assert classify_pattern(["result = sqrt(x)"]) == "other"
    print("  PASSED: Pattern classification works correctly")

    # Test 13: Context manager
    print("\n[Test 13] Context manager...")
    with PatternMemory(":memory:") as mem:
        mem.store(np.random.randn(64), "test = 1 + 1", "test")
        assert mem.count() == 1
    print("  PASSED: Context manager works")

    # Test 14: File-based database persistence
    print("\n[Test 14] File-based persistence...")
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name

    try:
        # Create and populate
        with PatternMemory(db_path) as mem:
            mem.store(np.random.randn(64), "persist = 42 * 2", "multiply")
            assert mem.count() == 1

        # Reopen and verify
        with PatternMemory(db_path) as mem:
            assert mem.count() == 1, "Pattern should persist across sessions"
            results = mem.query(np.random.randn(64), top_k=5)
            assert len(results) == 1, "Should find the persisted pattern"
        print("  PASSED: Database persists correctly")
    finally:
        os.unlink(db_path)

    # Test 15: Query with min_success_rate filter
    print("\n[Test 15] Query with success rate filter...")
    memory2 = PatternMemory(":memory:")

    # Pattern with good success rate
    good_emb = np.random.randn(64).astype(np.float32)
    good_id = memory2.store(good_emb, "good = 1 + 1", "good")
    for _ in range(10):
        memory2.record_outcome(good_id, success=True, epoch=1)

    # Pattern with poor success rate
    bad_emb = np.random.randn(64).astype(np.float32)
    bad_id = memory2.store(bad_emb, "bad = 2 + 2", "bad")
    for _ in range(4):
        memory2.record_outcome(bad_id, success=True, epoch=1)
    for _ in range(6):
        memory2.record_outcome(bad_id, success=False, epoch=1)
    # Bad pattern: 5 successes, 6 fails = 45% success rate

    # Query with 50% min threshold should filter out bad pattern
    results = memory2.query(bad_emb, top_k=10, min_success_rate=0.5)
    for r in results:
        if r['total_uses'] >= 5:
            assert r['success_rate'] >= 0.5, f"Should filter low success: {r['success_rate']}"
    print("  PASSED: Success rate filter works")

    # Test 16: Handle zero-norm embeddings gracefully
    print("\n[Test 16] Zero-norm embedding handling...")
    memory3 = PatternMemory(":memory:")
    memory3.store(np.random.randn(64), "test = 1", "test")

    zero_emb = np.zeros(64, dtype=np.float32)
    results = memory3.query(zero_emb, top_k=3)
    # Should not crash, returns results with 0 similarity
    assert isinstance(results, list), "Should return list even for zero embedding"
    print("  PASSED: Zero-norm embedding handled gracefully")

    # Test 17: Embedding similarity check in deduplicator
    print("\n[Test 17] Embedding similarity check...")
    emb_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb_b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb_c = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    assert PatternDeduplicator.embedding_similar(emb_a, emb_b, 0.95) == True
    assert PatternDeduplicator.embedding_similar(emb_a, emb_c, 0.95) == False
    print("  PASSED: Embedding similarity check works")

    memory.close()
    memory2.close()
    memory3.close()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
