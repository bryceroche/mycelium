"""DSL Negative Examples: Embedding-based semantic validation for DSL execution.

Instead of hardcoded keyword rules, we use embeddings to detect when a DSL
shouldn't run. When a DSL fails on a task, we store that task's embedding.
Before executing, we check if the current task is similar to known failures.

This is per-DSL lift tracking at the embedding level.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Path to store negative examples
NEGATIVE_EXAMPLES_PATH = Path(__file__).parent.parent / "dsl_negative_examples.json"

# Similarity threshold - if task is this similar to a negative example, skip DSL
NEGATIVE_EXAMPLE_THRESHOLD = 0.85

# Max negative examples per signature (prevent unbounded growth)
MAX_NEGATIVE_EXAMPLES = 20


class DSLNegativeExamples:
    """Tracks task embeddings where DSL execution failed."""

    _instance: Optional["DSLNegativeExamples"] = None

    def __init__(self):
        # signature_id -> list of (task_text, embedding) tuples
        self._examples: dict[str, list[tuple[str, list[float]]]] = {}
        self._dirty = False
        self._load()

    @classmethod
    def get_instance(cls) -> "DSLNegativeExamples":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load(self) -> None:
        """Load negative examples from disk."""
        if NEGATIVE_EXAMPLES_PATH.exists():
            try:
                with open(NEGATIVE_EXAMPLES_PATH) as f:
                    data = json.load(f)
                self._examples = data.get("negative_examples", {})
                logger.info(
                    "[dsl_negative] Loaded %d signatures with negative examples",
                    len(self._examples)
                )
            except Exception as e:
                logger.warning("[dsl_negative] Failed to load: %s", e)
                self._examples = {}
        else:
            self._examples = {}

    def save(self) -> None:
        """Save negative examples to disk."""
        if not self._dirty:
            return
        try:
            with open(NEGATIVE_EXAMPLES_PATH, "w") as f:
                json.dump({"negative_examples": self._examples}, f, indent=2)
            self._dirty = False
            logger.debug("[dsl_negative] Saved negative examples")
        except Exception as e:
            logger.warning("[dsl_negative] Failed to save: %s", e)

    def add_negative_example(
        self,
        signature_id: str,
        task_text: str,
        task_embedding: np.ndarray
    ) -> None:
        """Record a task where DSL failed for this signature."""
        if signature_id not in self._examples:
            self._examples[signature_id] = []

        examples = self._examples[signature_id]

        # Check if we already have a very similar example (avoid duplicates)
        task_emb_list = task_embedding.tolist()
        for existing_text, existing_emb in examples:
            sim = self._cosine_similarity(task_embedding, np.array(existing_emb))
            if sim > 0.95:  # Very similar, skip
                logger.debug("[dsl_negative] Skipping duplicate negative example")
                return

        # Add new example
        examples.append((task_text, task_emb_list))

        # Trim to max size (keep most recent)
        if len(examples) > MAX_NEGATIVE_EXAMPLES:
            self._examples[signature_id] = examples[-MAX_NEGATIVE_EXAMPLES:]

        self._dirty = True
        logger.debug(
            "[dsl_negative] Added negative example for sig=%s task='%s'",
            signature_id[:8], task_text[:50]
        )

    def should_skip_dsl(
        self,
        signature_id: str,
        task_embedding: np.ndarray,
        threshold: float = NEGATIVE_EXAMPLE_THRESHOLD
    ) -> tuple[bool, Optional[str]]:
        """Check if DSL should be skipped based on negative examples.

        Returns:
            (should_skip, reason) - True if task is too similar to known failures
        """
        if signature_id not in self._examples:
            return False, None

        examples = self._examples[signature_id]
        if not examples:
            return False, None

        for task_text, emb_list in examples:
            sim = self._cosine_similarity(task_embedding, np.array(emb_list))
            if sim >= threshold:
                reason = f"Similar to failed task: '{task_text[:40]}...' (sim={sim:.2f})"
                logger.debug("[dsl_negative] Skip DSL: %s", reason)
                return True, reason

        return False, None

    def get_stats(self) -> dict:
        """Get statistics about negative examples."""
        total_examples = sum(len(v) for v in self._examples.values())
        return {
            "signatures_with_negatives": len(self._examples),
            "total_negative_examples": total_examples,
        }

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


# Convenience functions
def get_negative_examples() -> DSLNegativeExamples:
    """Get the singleton instance."""
    return DSLNegativeExamples.get_instance()


def should_skip_dsl_semantic(
    signature_id: str,
    task_embedding: np.ndarray
) -> tuple[bool, Optional[str]]:
    """Check if DSL should be skipped based on semantic similarity to past failures."""
    return get_negative_examples().should_skip_dsl(signature_id, task_embedding)


def record_dsl_failure(
    signature_id: str,
    task_text: str,
    task_embedding: np.ndarray
) -> None:
    """Record that DSL failed on this task."""
    get_negative_examples().add_negative_example(signature_id, task_text, task_embedding)


def save_negative_examples() -> None:
    """Persist negative examples to disk."""
    get_negative_examples().save()
