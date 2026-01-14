"""Utility functions for step signatures."""

import json
from typing import Optional, Union
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def pack_embedding(embedding: Union[np.ndarray, list]) -> str:
    """Pack a numpy array into JSON string for SQLite storage."""
    if isinstance(embedding, np.ndarray):
        embedding = embedding.tolist()
    return json.dumps(embedding)


def unpack_embedding(data: str) -> Optional[np.ndarray]:
    """Unpack JSON string into numpy array."""
    if data is None:
        return None
    if isinstance(data, str):
        return np.array(json.loads(data), dtype=np.float32)
    if isinstance(data, (list, tuple)):
        return np.array(data, dtype=np.float32)
    if isinstance(data, np.ndarray):
        return data.astype(np.float32)
    return None
