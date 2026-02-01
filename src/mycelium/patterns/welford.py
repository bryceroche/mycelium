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


# Initialize on import
init_db()
