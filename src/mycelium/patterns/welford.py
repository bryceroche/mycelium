"""Welford's algorithm for online variance computation.

Tracks running statistics for similarity distributions to enable adaptive thresholds.
Database Statistics -> Welford -> Tree Structure (per CLAUDE.md "The Flow")
"""
import sqlite3
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple

from mycelium.config import (
    ADAPTIVE_THRESHOLD_MIN_SAMPLES,
    ADAPTIVE_THRESHOLD_K,
    ADAPTIVE_THRESHOLD_MIN,
    ADAPTIVE_THRESHOLD_MAX,
)

logger = logging.getLogger(__name__)

DB_PATH = Path.home() / ".mycelium" / "welford_stats.db"


@dataclass
class WelfordState:
    """Welford accumulator state for a single distribution."""
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0  # Sum of squared deviations

    def update(self, x: float) -> None:
        """Update statistics with a new observation."""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        """Population variance."""
        if self.n < 2:
            return 0.0
        return self.m2 / self.n

    @property
    def stddev(self) -> float:
        """Population standard deviation."""
        return self.variance ** 0.5

    def adaptive_threshold(self) -> Optional[float]:
        """Compute adaptive threshold: mean - k * stddev, clamped to bounds."""
        if self.n < ADAPTIVE_THRESHOLD_MIN_SAMPLES:
            return None  # Not enough samples

        threshold = self.mean - ADAPTIVE_THRESHOLD_K * self.stddev
        return max(ADAPTIVE_THRESHOLD_MIN, min(ADAPTIVE_THRESHOLD_MAX, threshold))


def _get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize Welford stats tables."""
    conn = _get_connection()
    conn.executescript('''
        -- Per-pattern similarity statistics
        CREATE TABLE IF NOT EXISTS pattern_stats (
            pattern_name TEXT PRIMARY KEY,
            n INTEGER DEFAULT 0,
            mean REAL DEFAULT 0.0,
            m2 REAL DEFAULT 0.0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Global similarity statistics (across all patterns)
        CREATE TABLE IF NOT EXISTS global_stats (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            n INTEGER DEFAULT 0,
            mean REAL DEFAULT 0.0,
            m2 REAL DEFAULT 0.0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Individual observations for analysis
        CREATE TABLE IF NOT EXISTS similarity_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_name TEXT NOT NULL,
            similarity REAL NOT NULL,
            was_correct INTEGER,  -- NULL if unknown, 1 if correct, 0 if wrong
            problem_hash TEXT,  -- For deduplication
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_obs_pattern ON similarity_observations(pattern_name);
        CREATE INDEX IF NOT EXISTS idx_obs_correct ON similarity_observations(was_correct);

        -- Per-example (signature) statistics for two-signal variance tracking
        -- Embedding variance: tracks variance of similarity scores when this example is matched
        -- Outcome variance: tracks variance of success/failure (1/0) outcomes
        CREATE TABLE IF NOT EXISTS example_stats (
            example_id TEXT PRIMARY KEY,
            pattern_name TEXT NOT NULL,
            -- Embedding variance Welford state
            emb_n INTEGER DEFAULT 0,
            emb_mean REAL DEFAULT 0.0,
            emb_m2 REAL DEFAULT 0.0,
            -- Outcome variance Welford state
            out_n INTEGER DEFAULT 0,
            out_mean REAL DEFAULT 0.0,
            out_m2 REAL DEFAULT 0.0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_example_pattern ON example_stats(pattern_name);

        -- Initialize global stats row
        INSERT OR IGNORE INTO global_stats (id, n, mean, m2) VALUES (1, 0, 0.0, 0.0);
    ''')
    conn.commit()
    conn.close()


def record_similarity(
    pattern_name: str,
    similarity: float,
    was_correct: Optional[bool] = None,
    problem_hash: Optional[str] = None
) -> None:
    """Record a similarity observation and update Welford stats."""
    conn = _get_connection()

    # Record observation
    conn.execute('''
        INSERT INTO similarity_observations (pattern_name, similarity, was_correct, problem_hash)
        VALUES (?, ?, ?, ?)
    ''', (pattern_name, similarity, 1 if was_correct else (0 if was_correct is False else None), problem_hash))

    # Update pattern stats using Welford's algorithm
    row = conn.execute(
        'SELECT n, mean, m2 FROM pattern_stats WHERE pattern_name = ?',
        (pattern_name,)
    ).fetchone()

    if row:
        n, mean, m2 = row['n'], row['mean'], row['m2']
    else:
        n, mean, m2 = 0, 0.0, 0.0

    # Welford update
    n += 1
    delta = similarity - mean
    mean += delta / n
    delta2 = similarity - mean
    m2 += delta * delta2

    conn.execute('''
        INSERT INTO pattern_stats (pattern_name, n, mean, m2, updated_at)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(pattern_name) DO UPDATE SET
            n = excluded.n,
            mean = excluded.mean,
            m2 = excluded.m2,
            updated_at = excluded.updated_at
    ''', (pattern_name, n, mean, m2))

    # Update global stats
    grow = conn.execute('SELECT n, mean, m2 FROM global_stats WHERE id = 1').fetchone()
    gn, gmean, gm2 = grow['n'], grow['mean'], grow['m2']

    gn += 1
    gdelta = similarity - gmean
    gmean += gdelta / gn
    gdelta2 = similarity - gmean
    gm2 += gdelta * gdelta2

    conn.execute('''
        UPDATE global_stats SET n = ?, mean = ?, m2 = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = 1
    ''', (gn, gmean, gm2))

    conn.commit()
    conn.close()

    logger.debug(f"[welford] Recorded sim={similarity:.3f} for {pattern_name} (n={n}, mean={mean:.3f})")


def get_pattern_stats(pattern_name: str) -> WelfordState:
    """Get Welford state for a specific pattern."""
    conn = _get_connection()
    row = conn.execute(
        'SELECT n, mean, m2 FROM pattern_stats WHERE pattern_name = ?',
        (pattern_name,)
    ).fetchone()
    conn.close()

    if row:
        return WelfordState(n=row['n'], mean=row['mean'], m2=row['m2'])
    return WelfordState()


def get_global_stats() -> WelfordState:
    """Get global Welford state across all patterns."""
    conn = _get_connection()
    row = conn.execute('SELECT n, mean, m2 FROM global_stats WHERE id = 1').fetchone()
    conn.close()

    if row:
        return WelfordState(n=row['n'], mean=row['mean'], m2=row['m2'])
    return WelfordState()


def get_adaptive_threshold(pattern_name: Optional[str] = None) -> float:
    """Get adaptive threshold for coverage decisions.

    Uses pattern-specific stats if available and sufficient samples,
    otherwise falls back to global stats, then to config default.
    """
    # Try pattern-specific first
    if pattern_name:
        stats = get_pattern_stats(pattern_name)
        threshold = stats.adaptive_threshold()
        if threshold is not None:
            logger.debug(f"[welford] Using pattern threshold for {pattern_name}: {threshold:.3f}")
            return threshold

    # Fall back to global
    global_stats = get_global_stats()
    threshold = global_stats.adaptive_threshold()
    if threshold is not None:
        logger.debug(f"[welford] Using global threshold: {threshold:.3f}")
        return threshold

    # Fall back to config default
    from mycelium.config import MIN_MATCH_THRESHOLD
    logger.debug(f"[welford] Using default threshold: {MIN_MATCH_THRESHOLD}")
    return MIN_MATCH_THRESHOLD


def get_all_pattern_stats() -> Dict[str, WelfordState]:
    """Get Welford states for all patterns."""
    conn = _get_connection()
    rows = conn.execute('SELECT pattern_name, n, mean, m2 FROM pattern_stats').fetchall()
    conn.close()

    return {
        row['pattern_name']: WelfordState(n=row['n'], mean=row['mean'], m2=row['m2'])
        for row in rows
    }


def get_coverage_gap_stats() -> Dict[str, Tuple[int, int]]:
    """Get stats on coverage gaps (low similarity but correct answers).

    Returns dict of pattern_name -> (gap_count, total_count) where
    gap_count is observations where similarity < adaptive_threshold and was_correct.
    """
    conn = _get_connection()

    results = {}
    patterns = conn.execute('SELECT DISTINCT pattern_name FROM similarity_observations').fetchall()

    for row in patterns:
        pattern_name = row['pattern_name']
        threshold = get_adaptive_threshold(pattern_name)

        total = conn.execute(
            'SELECT COUNT(*) as cnt FROM similarity_observations WHERE pattern_name = ?',
            (pattern_name,)
        ).fetchone()['cnt']

        gaps = conn.execute(
            'SELECT COUNT(*) as cnt FROM similarity_observations WHERE pattern_name = ? AND similarity < ? AND was_correct = 1',
            (pattern_name, threshold)
        ).fetchone()['cnt']

        results[pattern_name] = (gaps, total)

    conn.close()
    return results


# ============================================================================
# Per-Example (Signature) Welford Stats
# Two-signal variance tracking per CLAUDE.md:
# - Embedding variance: tracks variance of similarity scores when matched
# - Outcome variance: tracks variance of success/failure (1/0) outcomes
# ============================================================================

@dataclass
class ExampleWelfordStats:
    """Two-signal Welford stats for an example/signature."""
    example_id: str
    pattern_name: str
    embedding: WelfordState  # Variance of similarity scores
    outcome: WelfordState    # Variance of success/failure (1/0)

    @property
    def high_embedding_variance(self) -> bool:
        """High embedding variance = matches diverse problem types (may need decomposition)."""
        # Consider high if stddev > 0.15 (15% variation in similarity scores)
        return self.embedding.stddev > 0.15 if self.embedding.n >= 5 else False

    @property
    def high_outcome_variance(self) -> bool:
        """High outcome variance = inconsistent success (may need refinement).

        For binary outcomes (0/1), max variance is 0.25 (50% success rate).
        Consider "high" if variance > 0.20 (roughly 30-70% success range).
        """
        return self.outcome.variance > 0.20 if self.outcome.n >= 5 else False


def record_example_match(
    example_id: str,
    pattern_name: str,
    similarity: float,
    was_correct: bool
) -> None:
    """Record a match for an example, updating both embedding and outcome Welford stats.

    Args:
        example_id: Identifier for the example/signature
        pattern_name: Which pattern this example belongs to
        similarity: Cosine similarity score for this match
        was_correct: Whether the match resulted in correct execution
    """
    conn = _get_connection()

    # Get existing stats or defaults
    row = conn.execute(
        '''SELECT emb_n, emb_mean, emb_m2, out_n, out_mean, out_m2
           FROM example_stats WHERE example_id = ?''',
        (example_id,)
    ).fetchone()

    if row:
        emb_n, emb_mean, emb_m2 = row['emb_n'], row['emb_mean'], row['emb_m2']
        out_n, out_mean, out_m2 = row['out_n'], row['out_mean'], row['out_m2']
    else:
        emb_n, emb_mean, emb_m2 = 0, 0.0, 0.0
        out_n, out_mean, out_m2 = 0, 0.0, 0.0

    # Welford update for embedding variance (similarity scores)
    emb_n += 1
    emb_delta = similarity - emb_mean
    emb_mean += emb_delta / emb_n
    emb_delta2 = similarity - emb_mean
    emb_m2 += emb_delta * emb_delta2

    # Welford update for outcome variance (1/0 for success/failure)
    outcome_val = 1.0 if was_correct else 0.0
    out_n += 1
    out_delta = outcome_val - out_mean
    out_mean += out_delta / out_n
    out_delta2 = outcome_val - out_mean
    out_m2 += out_delta * out_delta2

    # Upsert
    conn.execute('''
        INSERT INTO example_stats (
            example_id, pattern_name,
            emb_n, emb_mean, emb_m2,
            out_n, out_mean, out_m2,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(example_id) DO UPDATE SET
            pattern_name = excluded.pattern_name,
            emb_n = excluded.emb_n,
            emb_mean = excluded.emb_mean,
            emb_m2 = excluded.emb_m2,
            out_n = excluded.out_n,
            out_mean = excluded.out_mean,
            out_m2 = excluded.out_m2,
            updated_at = excluded.updated_at
    ''', (example_id, pattern_name, emb_n, emb_mean, emb_m2, out_n, out_mean, out_m2))

    conn.commit()
    conn.close()

    logger.debug(
        f"[welford] Example {example_id}: sim={similarity:.3f} (n={emb_n}, var={emb_m2/emb_n if emb_n > 1 else 0:.4f}), "
        f"outcome={was_correct} (n={out_n}, mean={out_mean:.2f})"
    )


def get_example_stats(example_id: str) -> Optional[ExampleWelfordStats]:
    """Get both embedding and outcome Welford states for an example.

    Returns None if example has no recorded stats.
    """
    conn = _get_connection()
    row = conn.execute(
        '''SELECT example_id, pattern_name,
                  emb_n, emb_mean, emb_m2,
                  out_n, out_mean, out_m2
           FROM example_stats WHERE example_id = ?''',
        (example_id,)
    ).fetchone()
    conn.close()

    if not row:
        return None

    return ExampleWelfordStats(
        example_id=row['example_id'],
        pattern_name=row['pattern_name'],
        embedding=WelfordState(n=row['emb_n'], mean=row['emb_mean'], m2=row['emb_m2']),
        outcome=WelfordState(n=row['out_n'], mean=row['out_mean'], m2=row['out_m2']),
    )


def get_high_variance_examples(
    embedding_threshold: float = 0.15,
    outcome_threshold: float = 0.20,
    min_samples: int = 5
) -> Dict[str, ExampleWelfordStats]:
    """Get examples with high embedding OR outcome variance.

    Useful for MCTS post-mortem analysis per CLAUDE.md:
    - High embedding variance = signature matches diverse problem types (may need decomposition)
    - High outcome variance = inconsistent success (may need refinement)

    Args:
        embedding_threshold: Stddev threshold for embedding variance (default 0.15)
        outcome_threshold: Variance threshold for outcome variance (default 0.20)
        min_samples: Minimum observations required (default 5)

    Returns:
        Dict of example_id -> ExampleWelfordStats for high-variance examples
    """
    conn = _get_connection()
    rows = conn.execute(
        '''SELECT example_id, pattern_name,
                  emb_n, emb_mean, emb_m2,
                  out_n, out_mean, out_m2
           FROM example_stats
           WHERE emb_n >= ? OR out_n >= ?''',
        (min_samples, min_samples)
    ).fetchall()
    conn.close()

    results = {}
    for row in rows:
        emb_state = WelfordState(n=row['emb_n'], mean=row['emb_mean'], m2=row['emb_m2'])
        out_state = WelfordState(n=row['out_n'], mean=row['out_mean'], m2=row['out_m2'])

        # Check if either variance is high
        high_emb = emb_state.n >= min_samples and emb_state.stddev > embedding_threshold
        high_out = out_state.n >= min_samples and out_state.variance > outcome_threshold

        if high_emb or high_out:
            results[row['example_id']] = ExampleWelfordStats(
                example_id=row['example_id'],
                pattern_name=row['pattern_name'],
                embedding=emb_state,
                outcome=out_state,
            )

    return results


def get_all_example_stats() -> Dict[str, ExampleWelfordStats]:
    """Get Welford stats for all examples."""
    conn = _get_connection()
    rows = conn.execute(
        '''SELECT example_id, pattern_name,
                  emb_n, emb_mean, emb_m2,
                  out_n, out_mean, out_m2
           FROM example_stats'''
    ).fetchall()
    conn.close()

    return {
        row['example_id']: ExampleWelfordStats(
            example_id=row['example_id'],
            pattern_name=row['pattern_name'],
            embedding=WelfordState(n=row['emb_n'], mean=row['emb_mean'], m2=row['emb_m2']),
            outcome=WelfordState(n=row['out_n'], mean=row['out_mean'], m2=row['out_m2']),
        )
        for row in rows
    }


# Initialize on import
init_db()
