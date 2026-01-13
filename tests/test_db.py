"""Tests for the MyceliumDB class."""

import numpy as np
from mycelium.db import MyceliumDB, DB


class TestMyceliumDB:
    def test_db_alias(self):
        assert DB is MyceliumDB

    def test_add_signature(self):
        db = MyceliumDB()
        emb = np.random.randn(384).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        sig = db.add_signature(
            step_type="test_step",
            method_name="test_method",
            method_template="Test template",
            initial_embedding=emb,
        )

        assert sig.id is not None
        assert sig.step_type == "test_step"

    def test_find_similar(self):
        db = MyceliumDB()
        emb = np.random.randn(384).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        db.add_signature(
            step_type="test_step",
            method_name="test_method",
            method_template="Test template",
            initial_embedding=emb,
        )

        results = db.find_similar(emb, threshold=0.9)
        assert len(results) == 1
        assert results[0][1] > 0.99

    def test_get_stats(self):
        db = MyceliumDB()
        emb = np.random.randn(384).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        db.add_signature(
            step_type="test_step",
            method_name="test_method",
            method_template="Test template",
            initial_embedding=emb,
        )

        stats = db.get_stats()
        assert stats["signatures"] == 1
