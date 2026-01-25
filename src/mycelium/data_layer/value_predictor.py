"""Value Predictor for amplitude_post computation.

AlphaGo-inspired approach: Learn V(dag_step_type, node_id) - the value of using
a specific node for a specific step type. Just like AlphaGo learns V(board_position),
we learn V(step_type, node) = expected probability of success.

Key insight from AlphaGo:
- V(state) = expected outcome from this state
- We learn V(dag_step_type, node_id) = expected outcome when using this node for this step

Update rule (TD-learning):
- V(s) ← V(s) + α * (outcome - V(s))
- Where α = learning rate, outcome = 1 (won) or 0 (lost)

The value is then used to compute amplitude_post:
- High V (node works well here) → boost amplitude
- Low V (node fails here) → penalize amplitude
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from mycelium.config import (
    VALUE_PREDICTOR_ENABLED,
    VALUE_PREDICTOR_MIN_SAMPLES,
    VALUE_PREDICTOR_LEARNING_RATE,
    POSTMORTEM_REINFORCE_MULT,
    POSTMORTEM_BOOST_MULT,
    POSTMORTEM_MILD_PENALTY_MULT,
    POSTMORTEM_STRONG_PENALTY_MULT,
    POSTMORTEM_HIGH_CONF_THRESHOLD,
    POSTMORTEM_AMPLITUDE_MIN,
    POSTMORTEM_AMPLITUDE_MAX,
)
from mycelium.data_layer import get_db

logger = logging.getLogger(__name__)


# Schema for position values (AlphaGo-style)
VALUE_PREDICTOR_SCHEMA = """
-- Core value table: V(dag_step_type, node_id) like AlphaGo's V(position)
-- This is the learned value of using a specific node for a specific step type
CREATE TABLE IF NOT EXISTS position_values (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dag_step_type TEXT NOT NULL,          -- The step type (e.g., "compute_sum")
    node_id INTEGER NOT NULL,             -- The signature/node being evaluated

    -- Learned value and statistics
    value REAL DEFAULT 0.5,               -- V(s) ∈ [0, 1], initialized to 0.5 (neutral)
    updates INTEGER DEFAULT 0,            -- Number of TD updates (for confidence)
    total_outcomes REAL DEFAULT 0.0,      -- Sum of outcomes (for avg calculation)

    -- Metadata
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,

    UNIQUE(dag_step_type, node_id)
);
CREATE INDEX IF NOT EXISTS idx_pv_lookup ON position_values(dag_step_type, node_id);
CREATE INDEX IF NOT EXISTS idx_pv_node ON position_values(node_id);
CREATE INDEX IF NOT EXISTS idx_pv_value ON position_values(value);

-- Training samples for analysis/debugging (optional, can be pruned)
CREATE TABLE IF NOT EXISTS value_predictor_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dag_step_type TEXT,
    node_id INTEGER,
    amplitude REAL NOT NULL,
    similarity_score REAL,
    won INTEGER NOT NULL,
    dag_id TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_vps_pair ON value_predictor_samples(dag_step_type, node_id);
"""


@dataclass
class PredictorSample:
    """Training sample for value predictor."""

    dag_step_type: str  # The "position" - what type of step
    node_id: int  # The "move" - which node was chosen
    amplitude: float  # Prior routing confidence
    similarity_score: Optional[float]  # How well step matched node
    ucb1_gap: Optional[float]  # Decision confidence
    was_undecided: bool  # Did we branch?
    won: bool  # Ground truth outcome
    dag_id: Optional[str] = None


class ValuePredictor:
    """AlphaGo-style value predictor for (dag_step_type, node_id) pairs.

    Learns V(step_type, node) = probability of success when using this node
    for this type of step. Uses TD-learning to update values based on outcomes.
    """

    # Prior value for unseen pairs (neutral)
    PRIOR_VALUE = 0.5

    # Minimum updates before trusting learned value
    MIN_UPDATES_FOR_TRUST = 3

    def __init__(self):
        """Initialize predictor."""
        self._initialized: bool = False
        self._total_samples: int = 0

    def _ensure_schema(self) -> None:
        """Ensure tables exist."""
        db = get_db()
        with db.connection() as conn:
            conn.executescript(VALUE_PREDICTOR_SCHEMA)
            conn.commit()

    def _get_sample_count(self) -> int:
        """Get total number of training samples."""
        db = get_db()
        try:
            cursor = db.execute("SELECT COUNT(*) FROM value_predictor_samples")
            return cursor.fetchone()[0]
        except Exception:
            return 0

    def initialize(self) -> None:
        """Initialize predictor (ensure schema, load stats)."""
        if self._initialized:
            return

        self._ensure_schema()
        self._total_samples = self._get_sample_count()
        self._initialized = True

        logger.info(
            "[value_predictor] Initialized: %d training samples",
            self._total_samples
        )

    def get_position_value(
        self,
        dag_step_type: str,
        node_id: int,
    ) -> tuple[float, int]:
        """Get learned value V(dag_step_type, node_id).

        Returns:
            Tuple of (value, update_count)
            - value: learned V ∈ [0, 1], or PRIOR_VALUE if unseen
            - update_count: number of TD updates (0 if unseen)
        """
        if not self._initialized:
            self.initialize()

        self._ensure_schema()

        db = get_db()
        cursor = db.execute(
            """
            SELECT value, updates FROM position_values
            WHERE dag_step_type = ? AND node_id = ?
            """,
            (dag_step_type, node_id),
        )
        row = cursor.fetchone()

        if row is None:
            return (self.PRIOR_VALUE, 0)

        return (row[0], row[1])

    def update_position_value(
        self,
        dag_step_type: str,
        node_id: int,
        outcome: float,  # 1.0 for win, 0.0 for loss
        learning_rate: float = None,
    ) -> float:
        """Update V(dag_step_type, node_id) using TD-learning.

        TD update: V(s) ← V(s) + α * (outcome - V(s))

        Args:
            dag_step_type: The step type
            node_id: The node that was used
            outcome: 1.0 if won, 0.0 if lost
            learning_rate: Optional override for α

        Returns:
            New value after update
        """
        if not self._initialized:
            self.initialize()

        self._ensure_schema()

        lr = learning_rate or VALUE_PREDICTOR_LEARNING_RATE
        db = get_db()
        now = datetime.now(timezone.utc).isoformat()

        # Get current value or create new entry
        current_value, updates = self.get_position_value(dag_step_type, node_id)

        # TD update: V ← V + α(outcome - V)
        td_error = outcome - current_value
        new_value = current_value + lr * td_error

        # Clamp to [0, 1]
        new_value = max(0.0, min(1.0, new_value))

        # Upsert the value
        db.execute(
            """
            INSERT INTO position_values
            (dag_step_type, node_id, value, updates, total_outcomes, created_at, updated_at)
            VALUES (?, ?, ?, 1, ?, ?, ?)
            ON CONFLICT(dag_step_type, node_id) DO UPDATE SET
                value = ?,
                updates = updates + 1,
                total_outcomes = total_outcomes + ?,
                updated_at = ?
            """,
            (
                dag_step_type, node_id, new_value, outcome, now, now,
                new_value, outcome, now,
            ),
        )

        logger.debug(
            "[value_predictor] V(%s, node_%d): %.3f → %.3f (outcome=%.1f, td_error=%.3f)",
            dag_step_type, node_id, current_value, new_value, outcome, td_error
        )

        return new_value

    def record_sample(self, sample: PredictorSample) -> int:
        """Record a training sample and update position value.

        This does TWO things:
        1. Records sample for analysis/debugging
        2. Updates V(dag_step_type, node_id) via TD-learning

        Returns:
            Row ID of inserted sample
        """
        if not self._initialized:
            self.initialize()

        self._ensure_schema()

        db = get_db()
        now = datetime.now(timezone.utc).isoformat()

        # Record sample
        cursor = db.execute(
            """
            INSERT INTO value_predictor_samples
            (dag_step_type, node_id, amplitude, similarity_score, won, dag_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sample.dag_step_type,
                sample.node_id,
                sample.amplitude,
                sample.similarity_score,
                1 if sample.won else 0,
                sample.dag_id,
                now,
            ),
        )
        sample_id = cursor.lastrowid
        self._total_samples += 1

        # Update position value via TD-learning
        if sample.dag_step_type and sample.node_id:
            outcome = 1.0 if sample.won else 0.0
            self.update_position_value(
                sample.dag_step_type,
                sample.node_id,
                outcome,
            )

        return sample_id

    def predict(self, sample: PredictorSample) -> float:
        """Predict win probability for this (dag_step_type, node_id) pair.

        Blends learned position value with routing confidence signals.

        Returns:
            Predicted probability of success ∈ [0, 1]
        """
        if not self._initialized:
            self.initialize()

        # Get learned position value
        position_value, updates = self.get_position_value(
            sample.dag_step_type,
            sample.node_id,
        )

        # If we don't have enough updates, trust routing signals more
        if updates < self.MIN_UPDATES_FOR_TRUST:
            # Blend: heavily weight routing confidence during cold start
            routing_confidence = sample.amplitude
            blend_weight = updates / self.MIN_UPDATES_FOR_TRUST  # 0 → 1 as updates grow
            prediction = (1 - blend_weight) * routing_confidence + blend_weight * position_value
        else:
            # Trust position value, but still consider routing confidence
            # Position value is primary (70%), routing confidence secondary (30%)
            prediction = 0.7 * position_value + 0.3 * sample.amplitude

        return max(0.0, min(1.0, prediction))

    def compute_amplitude_post(
        self,
        sample: PredictorSample,
        use_predictor: bool = True,
    ) -> float:
        """Compute amplitude_post using learned position values.

        Args:
            sample: Input features including outcome
            use_predictor: If True and enough data, use learned values

        Returns:
            amplitude_post value (clamped to valid range)
        """
        if use_predictor and VALUE_PREDICTOR_ENABLED and self._total_samples >= VALUE_PREDICTOR_MIN_SAMPLES:
            # Use learned position value to scale amplitude
            position_value, updates = self.get_position_value(
                sample.dag_step_type,
                sample.node_id,
            )

            # Scale amplitude based on position value and outcome
            # High V + won → big boost (predicted correctly)
            # Low V + lost → expected, mild penalty
            # High V + lost → prediction was wrong, moderate penalty
            # Low V + won → pleasant surprise, boost

            if sample.won:
                if position_value >= 0.5:
                    # Predicted success, got success → reinforce
                    scale = 1.0 + (position_value - 0.5) * 0.8  # 1.0 to 1.4
                else:
                    # Predicted failure, got success → boost (nice surprise)
                    scale = 1.0 + (0.5 - position_value) * 0.3  # 1.0 to 1.15
            else:
                if position_value < 0.5:
                    # Predicted failure, got failure → mild penalty (expected)
                    scale = 0.9 - (0.5 - position_value) * 0.1  # 0.85 to 0.9
                else:
                    # Predicted success, got failure → strong penalty
                    scale = 0.7 - (position_value - 0.5) * 0.4  # 0.5 to 0.7

            amplitude_post = sample.amplitude * scale
        else:
            # Fall back to fixed multipliers during cold start
            amplitude_post = self._fixed_multiplier_fallback(sample)

        # Clamp to valid range
        return max(POSTMORTEM_AMPLITUDE_MIN, min(POSTMORTEM_AMPLITUDE_MAX, amplitude_post))

    def _fixed_multiplier_fallback(self, sample: PredictorSample) -> float:
        """Fallback to fixed multiplier logic (pre-predictor behavior)."""
        amp = sample.amplitude
        is_high_conf = amp >= POSTMORTEM_HIGH_CONF_THRESHOLD

        if sample.won and is_high_conf:
            return amp * POSTMORTEM_REINFORCE_MULT
        elif sample.won and not is_high_conf:
            return amp * POSTMORTEM_BOOST_MULT
        elif not sample.won and not is_high_conf:
            return amp * POSTMORTEM_MILD_PENALTY_MULT
        else:  # not won and is_high_conf
            return amp * POSTMORTEM_STRONG_PENALTY_MULT

    def get_stats(self) -> dict:
        """Get predictor statistics."""
        if not self._initialized:
            self.initialize()

        self._ensure_schema()

        db = get_db()

        # Count unique positions
        cursor = db.execute("SELECT COUNT(*) FROM position_values")
        position_count = cursor.fetchone()[0]

        # Get value distribution
        cursor = db.execute(
            """
            SELECT
                AVG(value) as avg_value,
                MIN(value) as min_value,
                MAX(value) as max_value,
                AVG(updates) as avg_updates
            FROM position_values
            """
        )
        row = cursor.fetchone()

        return {
            "total_samples": self._total_samples,
            "unique_positions": position_count,
            "avg_value": row[0] or 0.5,
            "min_value": row[1] or 0.5,
            "max_value": row[2] or 0.5,
            "avg_updates": row[3] or 0,
            "min_samples_for_predictor": VALUE_PREDICTOR_MIN_SAMPLES,
            "using_predictor": self._total_samples >= VALUE_PREDICTOR_MIN_SAMPLES,
        }


# Global predictor instance
_predictor: Optional[ValuePredictor] = None


def get_predictor() -> ValuePredictor:
    """Get the global value predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = ValuePredictor()
    return _predictor


def record_predictor_sample(
    dag_step_type: str,
    node_id: int,
    amplitude: float,
    won: bool,
    similarity_score: Optional[float] = None,
    ucb1_gap: Optional[float] = None,
    was_undecided: bool = False,
    dag_id: Optional[str] = None,
) -> int:
    """Convenience function to record a training sample.

    This records the sample AND updates V(dag_step_type, node_id).

    Returns:
        Row ID of inserted sample
    """
    sample = PredictorSample(
        dag_step_type=dag_step_type,
        node_id=node_id,
        amplitude=amplitude,
        similarity_score=similarity_score,
        ucb1_gap=ucb1_gap,
        was_undecided=was_undecided,
        won=won,
        dag_id=dag_id,
    )

    predictor = get_predictor()
    return predictor.record_sample(sample)


def get_position_value(dag_step_type: str, node_id: int) -> tuple[float, int]:
    """Get learned value V(dag_step_type, node_id).

    Returns:
        Tuple of (value, update_count)
    """
    predictor = get_predictor()
    return predictor.get_position_value(dag_step_type, node_id)


def compute_amplitude_post(
    dag_step_type: str,
    node_id: int,
    amplitude: float,
    won: bool,
    similarity_score: Optional[float] = None,
    use_predictor: bool = True,
) -> float:
    """Compute amplitude_post using learned position values.

    Args:
        dag_step_type: The step type (position)
        node_id: The node that was used (move)
        amplitude: Prior confidence
        won: Whether the thread won
        similarity_score: Routing similarity
        use_predictor: If True and enough data, use learned values

    Returns:
        amplitude_post value
    """
    sample = PredictorSample(
        dag_step_type=dag_step_type,
        node_id=node_id,
        amplitude=amplitude,
        similarity_score=similarity_score,
        ucb1_gap=None,
        was_undecided=False,
        won=won,
    )

    predictor = get_predictor()
    return predictor.compute_amplitude_post(sample, use_predictor=use_predictor)
