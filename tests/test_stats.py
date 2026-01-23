"""Tests for step-level statistics module."""

import os
import tempfile
import pytest

from mycelium.step_signatures.stats import (
    StepExecution,
    SignatureStats,
    StepStatsCollector,
)
from mycelium.data_layer.schema import init_db
import sqlite3


@pytest.fixture
def stats_db():
    """Create a temp database with schema for stats tests."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    conn.close()

    yield path

    try:
        os.unlink(path)
    except Exception:
        pass


class TestStepExecution:
    """Tests for StepExecution dataclass."""

    def test_creation(self):
        """Test basic creation."""
        exec_record = StepExecution(
            signature_id=1,
            step_text="Calculate 5 * 3",
            execution_time_ms=15.5,
            step_completed=True,
            was_injected=True,
            dsl_type="math",
            param_count=2,
            param_extraction_success=True,
        )
        assert exec_record.signature_id == 1
        assert exec_record.execution_time_ms == 15.5
        assert exec_record.step_completed is True
        assert exec_record.dsl_type == "math"
        assert exec_record.created_at  # Auto-populated

    def test_auto_timestamp(self):
        """Test that created_at is auto-populated."""
        exec1 = StepExecution(
            signature_id=1,
            step_text="test",
            execution_time_ms=10.0,
            step_completed=True,
            was_injected=False,
            dsl_type="math",
            param_count=0,
            param_extraction_success=True,
        )
        exec2 = StepExecution(
            signature_id=1,
            step_text="test",
            execution_time_ms=10.0,
            step_completed=True,
            was_injected=False,
            dsl_type="math",
            param_count=0,
            param_extraction_success=True,
            created_at="2024-01-01T00:00:00Z",
        )

        assert exec1.created_at != ""  # Auto-populated
        assert exec2.created_at == "2024-01-01T00:00:00Z"  # Preserved


class TestSignatureStats:
    """Tests for SignatureStats dataclass."""

    def test_empty_stats(self):
        """Test default values for empty stats."""
        stats = SignatureStats(signature_id=1)
        assert stats.total_executions == 0
        assert stats.success_rate == 0.0
        assert stats.dsl_execution_ratio == 0.0

    def test_efficiency_score(self):
        """Test efficiency score calculation."""
        # New signature (no data)
        stats = SignatureStats(signature_id=1)
        assert stats.efficiency_score == 0.5  # Neutral

        # Good signature
        stats = SignatureStats(
            signature_id=1,
            total_executions=100,
            success_rate=0.9,
            avg_execution_ms=50.0,
            param_extraction_rate=0.95,
            dsl_execution_ratio=0.8,
        )
        score = stats.efficiency_score
        assert score > 0.7  # Should be high

        # Poor signature
        stats = SignatureStats(
            signature_id=1,
            total_executions=100,
            success_rate=0.2,
            avg_execution_ms=2000.0,
            param_extraction_rate=0.3,
            dsl_execution_ratio=0.1,
        )
        score = stats.efficiency_score
        assert score < 0.3  # Should be low


class TestStepStatsCollector:
    """Tests for StepStatsCollector."""

    def test_schema_migration(self, stats_db):
        """Test that schema migration adds required columns."""
        collector = StepStatsCollector(stats_db)

        # Verify columns exist
        conn = sqlite3.connect(stats_db)
        cursor = conn.execute("PRAGMA table_info(step_usage_log)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()

        assert "execution_time_ms" in columns
        assert "dsl_type" in columns
        assert "param_extraction_success" in columns

    def test_record_execution(self, stats_db):
        """Test recording an execution."""
        collector = StepStatsCollector(stats_db)

        row_id = collector.record_execution(
            signature_id=1,
            step_text="Calculate 5 * 3",
            execution_time_ms=25.5,
            step_completed=True,
            was_injected=True,
            dsl_type="math",
            params_extracted={"a": 5, "b": 3},
            param_extraction_success=True,
        )

        assert row_id is not None
        assert row_id > 0

    def test_get_signature_stats_empty(self, stats_db):
        """Test getting stats for signature with no executions."""
        collector = StepStatsCollector(stats_db)
        stats = collector.get_signature_stats(999)

        assert stats.signature_id == 999
        assert stats.total_executions == 0
        assert stats.success_rate == 0.0

    def test_get_signature_stats(self, stats_db):
        """Test getting aggregated stats."""
        collector = StepStatsCollector(stats_db)

        # Record some executions
        for i in range(10):
            collector.record_execution(
                signature_id=1,
                step_text=f"step {i}",
                execution_time_ms=100.0 + i * 10,  # 100-190ms
                step_completed=i < 8,  # 8/10 success
                was_injected=i < 6,  # 6/10 injected
                dsl_type="math" if i < 7 else "decompose",  # 7 math, 3 decompose
                params_extracted={"x": i},
                param_extraction_success=i < 9,  # 9/10 success
            )

        stats = collector.get_signature_stats(1)

        assert stats.total_executions == 10
        assert stats.success_count == 8
        assert stats.success_rate == 0.8
        assert stats.injection_count == 6
        assert stats.injection_rate == 0.6
        assert stats.dsl_executions == 7
        assert stats.decompose_executions == 3
        assert stats.param_extraction_rate == 0.9

        # Check timing
        assert stats.avg_execution_ms == 145.0  # (100+190)/2
        assert stats.min_execution_ms == 100.0
        assert stats.max_execution_ms == 190.0
