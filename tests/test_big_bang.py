"""Tests for big bang smooth fork probability functions."""

import pytest
from unittest.mock import patch


class TestGetSystemMaturity:
    """Tests for get_system_maturity()."""

    def test_zero_signatures_is_immature(self):
        """Empty DB should have maturity near 0."""
        from mycelium.step_signatures.db import get_system_maturity

        maturity = get_system_maturity(0)
        assert maturity == 0.0

    def test_maturity_increases_with_signatures(self):
        """Maturity should increase as signature count grows."""
        from mycelium.step_signatures.db import get_system_maturity

        m1 = get_system_maturity(100)
        m2 = get_system_maturity(500)
        m3 = get_system_maturity(1000)

        assert m1 < m2 < m3

    def test_maturity_approaches_one(self):
        """Large signature count should approach maturity 1.0."""
        from mycelium.step_signatures.db import get_system_maturity

        maturity = get_system_maturity(10000)
        assert maturity > 0.9  # Should be close to 1.0

    def test_maturity_bounded_zero_to_one(self):
        """Maturity should always be between 0 and 1."""
        from mycelium.step_signatures.db import get_system_maturity

        for sig_count in [0, 10, 100, 1000, 10000, 100000]:
            maturity = get_system_maturity(sig_count)
            assert 0.0 <= maturity <= 1.0


class TestGetForkCenter:
    """Tests for get_fork_center()."""

    def test_cold_start_at_min_fork_depth(self):
        """Cold start (maturity=0) should have fork center at MIN_FORK_DEPTH."""
        from mycelium.step_signatures.db import get_fork_center
        from mycelium.config import MIN_FORK_DEPTH

        center = get_fork_center(0.0)
        assert center == MIN_FORK_DEPTH

    def test_fork_center_drifts_toward_root(self):
        """Fork center should drift toward root as maturity increases."""
        from mycelium.step_signatures.db import get_fork_center

        c1 = get_fork_center(0.0)
        c2 = get_fork_center(0.5)
        c3 = get_fork_center(1.0)

        assert c1 > c2 > c3  # Closer to root as maturity increases

    def test_fork_center_never_below_one(self):
        """Fork center should never go below level 1 (root never forks)."""
        from mycelium.step_signatures.db import get_fork_center

        center = get_fork_center(1.0)
        assert center >= 1.0


class TestSigmoid:
    """Tests for internal _sigmoid function."""

    def test_sigmoid_at_zero(self):
        """Sigmoid at x=0 should be 0.5."""
        from mycelium.step_signatures.db import _sigmoid

        assert _sigmoid(0.0) == pytest.approx(0.5)

    def test_sigmoid_monotonic(self):
        """Sigmoid should be monotonically increasing."""
        from mycelium.step_signatures.db import _sigmoid

        prev = _sigmoid(-10.0)
        for x in range(-9, 11):
            curr = _sigmoid(float(x))
            assert curr >= prev
            prev = curr

    def test_sigmoid_bounded(self):
        """Sigmoid should be bounded between 0 and 1."""
        from mycelium.step_signatures.db import _sigmoid

        for x in [-100, -10, -1, 0, 1, 10, 100]:
            result = _sigmoid(float(x))
            assert 0.0 <= result <= 1.0


class TestComputeForkProbability:
    """Tests for compute_fork_probability()."""

    def test_root_never_forks(self):
        """Depth 0 (root) should always have 0 fork probability."""
        from mycelium.step_signatures.db import compute_fork_probability

        prob = compute_fork_probability(
            depth=0,
            sig_count=100,
            best_similarity=0.5,
            fork_threshold=0.7,
        )
        assert prob == 0.0

    def test_shallow_depths_zero_probability(self):
        """Shallow depths (protected levels) should have ZERO fork probability."""
        from mycelium.step_signatures.db import compute_fork_probability
        from mycelium.config import MIN_FORK_DEPTH

        # Test all protected levels (below MIN_FORK_DEPTH)
        for depth in range(1, MIN_FORK_DEPTH):
            prob = compute_fork_probability(
                depth=depth,
                sig_count=0,  # Cold start
                best_similarity=0.3,  # Large gap
                fork_threshold=0.7,
            )
            # Hard cutoff: protected levels NEVER fork
            assert prob == 0.0, f"depth={depth} should have 0 probability, got {prob}"

    def test_deep_depths_higher_probability(self):
        """Deep depths (forking zone) should have higher fork probability."""
        from mycelium.step_signatures.db import compute_fork_probability
        from mycelium.config import MIN_FORK_DEPTH

        # At MIN_FORK_DEPTH with cold start and large similarity gap
        prob_deep = compute_fork_probability(
            depth=MIN_FORK_DEPTH + 2,  # Well into forking zone
            sig_count=0,  # Cold start (more aggressive)
            best_similarity=0.3,  # Large gap from threshold
            fork_threshold=0.7,
        )

        prob_shallow = compute_fork_probability(
            depth=2,  # Shallow, protected
            sig_count=0,
            best_similarity=0.3,
            fork_threshold=0.7,
        )

        assert prob_deep > prob_shallow

    def test_hysteresis_increases_probability(self):
        """Having existing forks at a level should increase fork probability."""
        from mycelium.step_signatures.db import compute_fork_probability
        from mycelium.config import MIN_FORK_DEPTH

        prob_no_hysteresis = compute_fork_probability(
            depth=MIN_FORK_DEPTH + 1,
            sig_count=100,
            best_similarity=0.5,
            fork_threshold=0.7,
            has_existing_forks_at_level=False,
        )

        prob_with_hysteresis = compute_fork_probability(
            depth=MIN_FORK_DEPTH + 1,
            sig_count=100,
            best_similarity=0.5,
            fork_threshold=0.7,
            has_existing_forks_at_level=True,
        )

        assert prob_with_hysteresis >= prob_no_hysteresis

    def test_larger_gap_higher_probability(self):
        """Larger similarity gap from threshold should increase fork probability."""
        from mycelium.step_signatures.db import compute_fork_probability
        from mycelium.config import MIN_FORK_DEPTH

        prob_small_gap = compute_fork_probability(
            depth=MIN_FORK_DEPTH + 1,
            sig_count=100,
            best_similarity=0.65,  # Close to threshold
            fork_threshold=0.7,
        )

        prob_large_gap = compute_fork_probability(
            depth=MIN_FORK_DEPTH + 1,
            sig_count=100,
            best_similarity=0.3,  # Far from threshold
            fork_threshold=0.7,
        )

        assert prob_large_gap >= prob_small_gap

    def test_probability_bounded(self):
        """Fork probability should be within configured bounds (for allowed depths)."""
        from mycelium.step_signatures.db import compute_fork_probability
        from mycelium.config import (
            BIG_BANG_MIN_FORK_PROB, BIG_BANG_MAX_FORK_PROB, MIN_FORK_DEPTH
        )

        # Test protected levels (0 to MIN_FORK_DEPTH-1) → should be 0
        for depth in range(0, MIN_FORK_DEPTH):
            prob = compute_fork_probability(
                depth=depth,
                sig_count=100,
                best_similarity=0.3,
                fork_threshold=0.7,
            )
            assert prob == 0.0, f"Protected depth {depth} should be 0"

        # Test allowed levels (MIN_FORK_DEPTH and above) → bounded by config
        for depth in range(MIN_FORK_DEPTH, MIN_FORK_DEPTH + 10):
            for sig_count in [0, 100, 1000, 10000]:
                for best_sim in [0.0, 0.3, 0.5, 0.7, 0.9]:
                    prob = compute_fork_probability(
                        depth=depth,
                        sig_count=sig_count,
                        best_similarity=best_sim,
                        fork_threshold=0.7,
                    )
                    assert BIG_BANG_MIN_FORK_PROB <= prob <= BIG_BANG_MAX_FORK_PROB


class TestShouldForkAtDepth:
    """Tests for should_fork_at_depth() probabilistic decision."""

    def test_root_never_returns_true(self):
        """Depth 0 (root) should never fork."""
        from mycelium.step_signatures.db import should_fork_at_depth

        # Run many times - should always be False
        for _ in range(100):
            result = should_fork_at_depth(
                depth=0,
                sig_count=100,
                best_similarity=0.3,
                fork_threshold=0.7,
            )
            assert result is False

    def test_returns_boolean(self):
        """should_fork_at_depth should always return a boolean."""
        from mycelium.step_signatures.db import should_fork_at_depth
        from mycelium.config import MIN_FORK_DEPTH

        result = should_fork_at_depth(
            depth=MIN_FORK_DEPTH + 1,
            sig_count=100,
            best_similarity=0.5,
            fork_threshold=0.7,
        )
        assert isinstance(result, bool)

    def test_probabilistic_behavior(self):
        """With moderate probability, should see both True and False over many runs."""
        from mycelium.step_signatures.db import should_fork_at_depth
        from mycelium.config import MIN_FORK_DEPTH

        # Run many times with moderate parameters
        results = []
        for _ in range(100):
            result = should_fork_at_depth(
                depth=MIN_FORK_DEPTH + 1,
                sig_count=100,
                best_similarity=0.4,  # Below threshold
                fork_threshold=0.7,
            )
            results.append(result)

        # Should see both outcomes (probabilistic)
        # With moderate probability, we expect variation
        true_count = sum(results)
        # Allow for some variability but expect not all same
        # This test may occasionally fail due to randomness
        assert 0 < true_count < 100, f"Expected mixed results, got {true_count} True out of 100"
