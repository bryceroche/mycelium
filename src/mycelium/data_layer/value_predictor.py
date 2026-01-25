"""Value Predictor for amplitude_post computation.

Per beads mycelium-8150: AlphaGo-inspired approach - train a predictor instead
of using fixed multipliers. Collects (features, outcome) pairs and learns
the actual relationship between confidence signals and win probability.

Features:
- amplitude (prior confidence)
- similarity_score (routing similarity)
- ucb1_gap (decision confidence)
- node_success_rate (historical performance)
- signature_uses (maturity)

Target:
- 1.0 if won (normalize amplitude for credit)
- 0.0 if lost (normalize amplitude for blame)
"""

import json
import logging
import sqlite3
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
)
from mycelium.data_layer import get_db

logger = logging.getLogger(__name__)


# Schema for training data collection
VALUE_PREDICTOR_SCHEMA = """
CREATE TABLE IF NOT EXISTS value_predictor_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Input features
    amplitude REAL NOT NULL,
    similarity_score REAL,
    ucb1_gap REAL,
    was_undecided INTEGER,
    node_success_rate REAL,
    node_uses INTEGER,
    step_idx INTEGER,
    total_steps INTEGER,

    -- Outcome (ground truth)
    won INTEGER NOT NULL,  -- 1=thread won, 0=thread lost

    -- Metadata
    dag_id TEXT,
    node_id INTEGER,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_vps_created ON value_predictor_samples(created_at);
CREATE INDEX IF NOT EXISTS idx_vps_won ON value_predictor_samples(won);

-- Learned weights table (simple linear model)
CREATE TABLE IF NOT EXISTS value_predictor_weights (
    feature_name TEXT PRIMARY KEY,
    weight REAL NOT NULL,
    updated_at TEXT NOT NULL
);
"""


@dataclass
class PredictorSample:
    """Training sample for value predictor."""

    amplitude: float
    similarity_score: Optional[float]
    ucb1_gap: Optional[float]
    was_undecided: bool
    node_success_rate: Optional[float]
    node_uses: int
    step_idx: int
    total_steps: int
    won: bool
    dag_id: Optional[str] = None
    node_id: Optional[int] = None


class ValuePredictor:
    """Linear value predictor for amplitude_post.

    Learns weights for features to predict win probability.
    Falls back to fixed multipliers when insufficient training data.
    """

    # Feature names in order (matches weight vector)
    FEATURES = [
        "amplitude",
        "similarity_score",
        "ucb1_gap",
        "was_undecided",
        "node_success_rate",
        "step_position",  # step_idx / total_steps (normalized)
        "bias",
    ]

    def __init__(self):
        """Initialize predictor with default weights."""
        self._weights: dict[str, float] = {}
        self._sample_count: int = 0
        self._initialized: bool = False

    def _ensure_schema(self) -> None:
        """Ensure training data tables exist."""
        db = get_db()
        with db.connection() as conn:
            conn.executescript(VALUE_PREDICTOR_SCHEMA)
            conn.commit()

    def _load_weights(self) -> dict[str, float]:
        """Load trained weights from database."""
        conn = get_db()
        cursor = conn.execute(
            "SELECT feature_name, weight FROM value_predictor_weights"
        )
        weights = {row[0]: row[1] for row in cursor.fetchall()}

        # Initialize missing features with defaults
        defaults = {
            "amplitude": 0.3,  # Higher amplitude -> higher value
            "similarity_score": 0.2,  # Higher similarity -> higher value
            "ucb1_gap": 0.1,  # Higher gap (confident) -> higher value
            "was_undecided": -0.1,  # Undecided -> slightly lower value
            "node_success_rate": 0.3,  # Historical success matters
            "step_position": -0.1,  # Later steps slightly riskier
            "bias": 0.5,  # Start at 50%
        }

        for feat, default in defaults.items():
            if feat not in weights:
                weights[feat] = default

        return weights

    def _save_weights(self, weights: dict[str, float]) -> None:
        """Save trained weights to database."""
        conn = get_db()
        now = datetime.now(timezone.utc).isoformat()

        for feat, weight in weights.items():
            conn.execute(
                """
                INSERT OR REPLACE INTO value_predictor_weights
                (feature_name, weight, updated_at) VALUES (?, ?, ?)
                """,
                (feat, weight, now),
            )

    def _get_sample_count(self) -> int:
        """Get number of training samples."""
        conn = get_db()
        cursor = conn.execute("SELECT COUNT(*) FROM value_predictor_samples")
        return cursor.fetchone()[0]

    def initialize(self) -> None:
        """Initialize predictor (load weights, check sample count)."""
        if self._initialized:
            return

        self._ensure_schema()
        self._weights = self._load_weights()
        self._sample_count = self._get_sample_count()
        self._initialized = True

        logger.info(
            "[value_predictor] Initialized: %d samples, weights=%s",
            self._sample_count,
            {k: f"{v:.3f}" for k, v in self._weights.items()}
        )

    def record_sample(self, sample: PredictorSample) -> int:
        """Record a training sample to database.

        Returns:
            Row ID of inserted sample
        """
        if not self._initialized:
            self.initialize()

        # Always ensure schema exists (handles test isolation with fresh DBs)
        self._ensure_schema()

        conn = get_db()
        now = datetime.now(timezone.utc).isoformat()

        cursor = conn.execute(
            """
            INSERT INTO value_predictor_samples
            (amplitude, similarity_score, ucb1_gap, was_undecided,
             node_success_rate, node_uses, step_idx, total_steps,
             won, dag_id, node_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sample.amplitude,
                sample.similarity_score,
                sample.ucb1_gap,
                1 if sample.was_undecided else 0,
                sample.node_success_rate,
                sample.node_uses,
                sample.step_idx,
                sample.total_steps,
                1 if sample.won else 0,
                sample.dag_id,
                sample.node_id,
                now,
            ),
        )

        self._sample_count += 1
        return cursor.lastrowid

    def _extract_features(self, sample: PredictorSample) -> dict[str, float]:
        """Extract feature vector from sample."""
        # Normalize step position (0 = first step, 1 = last step)
        step_position = sample.step_idx / max(sample.total_steps, 1)

        return {
            "amplitude": sample.amplitude,
            "similarity_score": sample.similarity_score or 0.0,
            "ucb1_gap": sample.ucb1_gap or 0.0,
            "was_undecided": 1.0 if sample.was_undecided else 0.0,
            "node_success_rate": sample.node_success_rate or 0.5,
            "step_position": step_position,
            "bias": 1.0,
        }

    def predict(self, sample: PredictorSample) -> float:
        """Predict win probability (0-1) for a sample.

        Falls back to fixed multipliers if insufficient training data.
        """
        if not self._initialized:
            self.initialize()

        # Fall back to fixed multipliers if not enough data
        if self._sample_count < VALUE_PREDICTOR_MIN_SAMPLES:
            return self._fixed_multiplier_prediction(sample)

        # Linear prediction: sum(weight_i * feature_i)
        features = self._extract_features(sample)
        raw_score = sum(
            self._weights.get(feat, 0.0) * val
            for feat, val in features.items()
        )

        # Sigmoid activation for probability output
        import math
        probability = 1.0 / (1.0 + math.exp(-raw_score))

        return probability

    def _fixed_multiplier_prediction(self, sample: PredictorSample) -> float:
        """Fallback to fixed multiplier logic (pre-predictor behavior).

        Returns normalized amplitude_post value.
        """
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

    def compute_amplitude_post(
        self,
        sample: PredictorSample,
        use_predictor: bool = True,
    ) -> float:
        """Compute amplitude_post using predictor or fixed multipliers.

        Args:
            sample: Input features and outcome
            use_predictor: If True and enough data, use learned predictor

        Returns:
            amplitude_post value (clamped to valid range)
        """
        from mycelium.config import POSTMORTEM_AMPLITUDE_MIN, POSTMORTEM_AMPLITUDE_MAX

        if use_predictor and VALUE_PREDICTOR_ENABLED:
            # Predictor outputs probability, scale to amplitude
            win_prob = self.predict(sample)
            # Scale: 0.5 prob -> amp unchanged, 1.0 -> max boost, 0.0 -> max penalty
            amplitude_post = sample.amplitude * (0.5 + win_prob)
        else:
            amplitude_post = self._fixed_multiplier_prediction(sample)

        # Clamp to valid range
        return max(POSTMORTEM_AMPLITUDE_MIN, min(POSTMORTEM_AMPLITUDE_MAX, amplitude_post))

    def train_batch(self, batch_size: int = 1000) -> dict:
        """Train on recent samples using online gradient descent.

        Returns:
            Training statistics
        """
        if not self._initialized:
            self.initialize()

        conn = get_db()

        # Get recent samples
        cursor = conn.execute(
            """
            SELECT amplitude, similarity_score, ucb1_gap, was_undecided,
                   node_success_rate, node_uses, step_idx, total_steps, won
            FROM value_predictor_samples
            ORDER BY id DESC
            LIMIT ?
            """,
            (batch_size,),
        )

        samples = cursor.fetchall()
        if not samples:
            return {"samples": 0, "loss": 0.0}

        total_loss = 0.0
        lr = VALUE_PREDICTOR_LEARNING_RATE

        for row in samples:
            sample = PredictorSample(
                amplitude=row[0],
                similarity_score=row[1],
                ucb1_gap=row[2],
                was_undecided=bool(row[3]),
                node_success_rate=row[4],
                node_uses=row[5] or 0,
                step_idx=row[6] or 0,
                total_steps=row[7] or 1,
                won=bool(row[8]),
            )

            # Forward pass
            features = self._extract_features(sample)
            prediction = self.predict(sample)
            target = 1.0 if sample.won else 0.0

            # Loss (mean squared error)
            error = prediction - target
            total_loss += error ** 2

            # Backward pass (gradient descent)
            # d(loss)/d(weight_i) = 2 * error * prediction * (1 - prediction) * feature_i
            grad_scale = 2 * error * prediction * (1 - prediction)

            for feat, val in features.items():
                gradient = grad_scale * val
                self._weights[feat] = self._weights.get(feat, 0.0) - lr * gradient

        # Save updated weights
        self._save_weights(self._weights)

        avg_loss = total_loss / len(samples)
        logger.info(
            "[value_predictor] Trained on %d samples, avg_loss=%.4f",
            len(samples), avg_loss
        )

        return {
            "samples": len(samples),
            "loss": avg_loss,
            "weights": dict(self._weights),
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
    amplitude: float,
    won: bool,
    similarity_score: Optional[float] = None,
    ucb1_gap: Optional[float] = None,
    was_undecided: bool = False,
    node_success_rate: Optional[float] = None,
    node_uses: int = 0,
    step_idx: int = 0,
    total_steps: int = 1,
    dag_id: Optional[str] = None,
    node_id: Optional[int] = None,
) -> int:
    """Convenience function to record a training sample.

    Returns:
        Row ID of inserted sample
    """
    sample = PredictorSample(
        amplitude=amplitude,
        similarity_score=similarity_score,
        ucb1_gap=ucb1_gap,
        was_undecided=was_undecided,
        node_success_rate=node_success_rate,
        node_uses=node_uses,
        step_idx=step_idx,
        total_steps=total_steps,
        won=won,
        dag_id=dag_id,
        node_id=node_id,
    )

    predictor = get_predictor()
    return predictor.record_sample(sample)


def compute_amplitude_post(
    amplitude: float,
    won: bool,
    similarity_score: Optional[float] = None,
    ucb1_gap: Optional[float] = None,
    was_undecided: bool = False,
    node_success_rate: Optional[float] = None,
    node_uses: int = 0,
    step_idx: int = 0,
    total_steps: int = 1,
    use_predictor: bool = True,
) -> float:
    """Compute amplitude_post using value predictor.

    Args:
        amplitude: Prior confidence
        won: Whether the thread won
        similarity_score: Routing similarity
        ucb1_gap: UCB1 decision gap
        was_undecided: Whether routing was undecided
        node_success_rate: Historical success rate of node
        node_uses: Number of times node has been used
        step_idx: Step index in problem (0-based)
        total_steps: Total steps in problem
        use_predictor: If True and enough data, use learned predictor

    Returns:
        amplitude_post value
    """
    sample = PredictorSample(
        amplitude=amplitude,
        similarity_score=similarity_score,
        ucb1_gap=ucb1_gap,
        was_undecided=was_undecided,
        node_success_rate=node_success_rate,
        node_uses=node_uses,
        step_idx=step_idx,
        total_steps=total_steps,
        won=won,
    )

    predictor = get_predictor()
    return predictor.compute_amplitude_post(sample, use_predictor=use_predictor)
