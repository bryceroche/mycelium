"""Tests for the slow decay lifecycle system."""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta

from mycelium.step_signatures.decay import (
    DecayStatus,
    DecayState,
    DecayAction,
    DecayReport,
    DecayManager,
)


class TestDecayStatus:
    """Tests for DecayStatus enum."""

    def test_all_statuses_exist(self):
        """Verify all expected statuses are defined."""
        assert DecayStatus.HEALTHY.value == "healthy"
        assert DecayStatus.WARNING.value == "warning"
        assert DecayStatus.CRITICAL.value == "critical"
        assert DecayStatus.ARCHIVED.value == "archived"
        assert DecayStatus.RECOVERING.value == "recovering"


class TestDecayState:
    """Tests for DecayState dataclass."""

    def test_default_values(self):
        """New state should have sensible defaults."""
        state = DecayState(signature_id=1)
        assert state.status == DecayStatus.HEALTHY
        assert state.current_traffic_share == 0.0
        assert state.min_traffic_share_seen == 1.0
        assert state.max_traffic_share_seen == 0.0
        assert state.recovery_attempts == 0

    def test_is_improving_positive_trend(self):
        """Should detect improving trend."""
        state = DecayState(signature_id=1, trend_7d=0.05)
        assert state.is_improving() is True

    def test_is_improving_negative_trend(self):
        """Should detect declining trend."""
        state = DecayState(signature_id=1, trend_7d=-0.05)
        assert state.is_improving() is False

    def test_is_improving_flat_trend(self):
        """Flat trend is not improving."""
        state = DecayState(signature_id=1, trend_7d=0.005)
        assert state.is_improving() is False

    def test_should_archive_not_critical(self):
        """Should not archive healthy signatures."""
        state = DecayState(signature_id=1, status=DecayStatus.HEALTHY)
        assert state.should_archive() is False

    def test_should_archive_already_archived(self):
        """Should not re-archive already archived signatures."""
        state = DecayState(signature_id=1, status=DecayStatus.ARCHIVED)
        assert state.should_archive() is False

    def test_should_archive_critical_too_short(self):
        """Should not archive critical signatures that are too recent."""
        now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        state = DecayState(
            signature_id=1,
            status=DecayStatus.CRITICAL,
            entered_critical_at=now,
        )
        assert state.should_archive() is False

    def test_should_archive_critical_recovering(self):
        """Should not archive critical signatures that are recovering."""
        old_date = (datetime.now(timezone.utc) - timedelta(days=60)).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        state = DecayState(
            signature_id=1,
            status=DecayStatus.CRITICAL,
            entered_critical_at=old_date,
            trend_7d=0.10,  # Improving
        )
        assert state.should_archive() is False


class TestDecayAction:
    """Tests for DecayAction dataclass."""

    def test_action_creation(self):
        """Should create action with all fields."""
        action = DecayAction(
            signature_id=1,
            action="warn",
            reason="Traffic dropped",
            old_status=DecayStatus.HEALTHY,
            new_status=DecayStatus.WARNING,
            traffic_share=0.005,
        )
        assert action.signature_id == 1
        assert action.action == "warn"


class TestDecayReport:
    """Tests for DecayReport dataclass."""

    def test_default_values(self):
        """New report should have zero counts."""
        report = DecayReport()
        assert report.total_signatures == 0
        assert report.healthy_count == 0
        assert report.warning_count == 0
        assert report.critical_count == 0
        assert report.archived_count == 0
        assert report.actions_taken == []
        assert report.errors == []

    def test_summary(self):
        """Should generate readable summary."""
        report = DecayReport(
            total_signatures=100,
            healthy_count=80,
            warning_count=15,
            critical_count=5,
            avg_traffic_share=0.01,
        )
        summary = report.summary()
        assert "100 signatures" in summary
        assert "Healthy: 80" in summary
        assert "Warning: 15" in summary


class TestDecayManager:
    """Tests for DecayManager class."""

    @pytest.fixture
    def manager(self, clean_test_db):
        """Create manager with test database."""
        from mycelium.config import DB_PATH
        return DecayManager(db_path=DB_PATH)

    def test_init(self, manager):
        """Manager should initialize properly."""
        assert manager._db is not None
        assert manager._last_run_at == 0

    def test_ensure_decay_table(self, manager):
        """Should create decay tables without error."""
        manager._ensure_decay_table()
        # Second call should also work (idempotent)
        manager._ensure_decay_table()

    def test_compute_decay_status_healthy(self, manager):
        """High traffic share should be healthy."""
        status = manager._compute_decay_status(0.02, None)  # Above TRAFFIC_MIN_SHARE
        assert status == DecayStatus.HEALTHY

    def test_compute_decay_status_warning(self, manager):
        """Low traffic share should trigger warning."""
        # TRAFFIC_MIN_SHARE = 0.01, WARNING_THRESHOLD = 0.50
        # So warning threshold = 0.01 * 0.50 = 0.005
        status = manager._compute_decay_status(0.006, None)
        assert status == DecayStatus.WARNING

    def test_compute_decay_status_critical(self, manager):
        """Very low traffic share should be critical."""
        # TRAFFIC_MIN_SHARE = 0.01, DEMOTE_THRESHOLD = 0.20
        # So demote threshold = 0.01 * 0.20 = 0.002
        status = manager._compute_decay_status(0.001, None)
        assert status == DecayStatus.CRITICAL

    def test_compute_decay_status_recovering(self, manager):
        """Should detect recovery from critical state."""
        # TRAFFIC_MIN_SHARE = 0.01, RECOVERY_THRESHOLD = 0.80
        # So recovery threshold = 0.01 * 0.80 = 0.008
        current_state = DecayState(
            signature_id=1,
            status=DecayStatus.CRITICAL,
        )
        status = manager._compute_decay_status(0.009, current_state)
        assert status == DecayStatus.RECOVERING

    def test_save_and_load_decay_state(self, manager):
        """Should persist and retrieve decay state."""
        import numpy as np
        from mycelium.step_signatures.db import StepSignatureDB

        # Create StepSignatureDB first to initialize step_signatures table
        # (signature_decay has FK to step_signatures)
        db = StepSignatureDB()
        sig = db.create_signature(
            step_text="test signature for decay",
            embedding=np.random.randn(3072).astype(np.float32),
        )

        manager._ensure_decay_table()

        state = DecayState(
            signature_id=sig.id,
            status=DecayStatus.WARNING,
            current_traffic_share=0.005,
            recovery_attempts=2,
        )
        manager._save_decay_state(state)

        loaded = manager._load_decay_state(sig.id)
        assert loaded is not None
        assert loaded.status == DecayStatus.WARNING
        assert loaded.current_traffic_share == 0.005
        assert loaded.recovery_attempts == 2

    def test_load_nonexistent_state(self, manager):
        """Should return None for missing state."""
        manager._ensure_decay_table()
        loaded = manager._load_decay_state(99999)
        assert loaded is None

    def test_get_decay_summary(self, manager):
        """Should return summary dict."""
        manager._ensure_decay_table()
        summary = manager.get_decay_summary()
        assert isinstance(summary, dict)
        assert "healthy" in summary
        assert "warning" in summary

    def test_run_decay_cycle_cold_start(self, manager):
        """Should skip decay during cold start."""
        # No problems solved yet
        report = manager.run_decay_cycle(force=True)
        # Should return early without errors
        assert report.errors == [] or report.total_problems < 50


class TestDecayIntegration:
    """Integration tests for decay with real signatures."""

    @pytest.fixture
    def db_with_signatures(self, clean_test_db):
        """Create DB with some signatures."""
        import numpy as np
        from mycelium.step_signatures.db import StepSignatureDB
        from mycelium.config import EMBEDDING_DIM

        db = StepSignatureDB()

        # Create a few signatures with different usage patterns
        for i, (uses, successes) in enumerate([
            (100, 80),  # High traffic, healthy
            (10, 5),    # Medium traffic
            (2, 1),     # Low traffic (will decay)
        ]):
            emb = np.random.randn(EMBEDDING_DIM).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            sig = db.create_signature(
                step_text=f"Test signature {i}",
                embedding=emb,
            )
            with db._connection() as conn:
                conn.execute(
                    """UPDATE step_signatures
                       SET uses = ?, successes = ?, is_semantic_umbrella = 0
                       WHERE id = ?""",
                    (uses, successes, sig.id)
                )

        # Set total problems high enough to exit cold start
        with db._connection() as conn:
            conn.execute("""
                INSERT INTO db_metadata (key, value, updated_at)
                VALUES ('total_problems_solved', '1000', datetime('now'))
                ON CONFLICT(key) DO UPDATE SET value = '1000'
            """)
            conn.commit()

        return db

    def test_decay_cycle_with_signatures(self, db_with_signatures):
        """Should analyze signatures and produce report."""
        from mycelium.config import DB_PATH
        from mycelium.data_layer.schema import migrate_db
        import sqlite3

        manager = DecayManager(db_path=DB_PATH)

        # Run migrations to add is_archived column
        conn = sqlite3.connect(DB_PATH)
        migrate_db(conn)
        conn.close()

        # Manually ensure we're past cold start by setting total_problems
        manager._ensure_decay_table()
        with manager._connection() as conn:
            conn.execute("""
                INSERT INTO db_metadata (key, value, updated_at)
                VALUES ('total_problems_solved', '1000', datetime('now'))
                ON CONFLICT(key) DO UPDATE SET value = '1000'
            """)
            conn.commit()

        report = manager.run_decay_cycle(force=True)

        # Should have analyzed our signatures
        assert report.total_signatures >= 3
        assert report.total_problems == 1000


class TestDecayModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_get_decay_manager_singleton(self):
        """Should return same instance on repeated calls."""
        from mycelium.step_signatures.decay import get_decay_manager

        # Reset singleton for test
        import mycelium.step_signatures.decay as decay_module
        decay_module._decay_manager = None

        m1 = get_decay_manager()
        m2 = get_decay_manager()
        assert m1 is m2

    def test_run_decay_cycle_function(self, clean_test_db):
        """Convenience function should work."""
        from mycelium.step_signatures.decay import run_decay_cycle

        report = run_decay_cycle(force=True)
        assert isinstance(report, DecayReport)

    def test_get_decay_summary_function(self, clean_test_db):
        """Convenience function should work."""
        from mycelium.step_signatures.decay import get_decay_summary

        summary = get_decay_summary()
        assert isinstance(summary, dict)
