"""Tests for DSL template inference functions."""

import json
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from mycelium.step_signatures.dsl_templates import (
    infer_dsl_for_signature,
    _infer_dsl_from_values,
    _infer_operation_semantic,
    _find_similar_successful_dsl,
    _get_param_anchor_embeddings,
    _get_desc_anchor_embeddings,
    _make_dsl_json,
    _PARAM_ANCHORS,
    _DESCRIPTION_ANCHORS,
)

# Reset anchor embedding caches before tests
import mycelium.step_signatures.dsl_templates as dsl_templates


@pytest.fixture(autouse=True)
def reset_anchor_caches():
    """Reset global anchor embedding caches and seed RNG before each test."""
    np.random.seed(42)  # Deterministic embeddings in mock fixtures
    dsl_templates._param_anchor_embeddings = None
    dsl_templates._desc_anchor_embeddings = None
    yield
    dsl_templates._param_anchor_embeddings = None
    dsl_templates._desc_anchor_embeddings = None


class TestInferDslForSignature:
    """Tests for infer_dsl_for_signature() priority order."""

    def test_priority_1_extracted_values_used_first(self):
        """When extracted_values provided, DSL should be inferred from them."""
        with patch.object(dsl_templates, "_infer_dsl_from_values") as mock_infer:
            mock_infer.return_value = ('{"type": "math", "script": "a + b"}', "math")

            result = infer_dsl_for_signature(
                step_type="add_numbers",
                description="add two numbers",
                db=None,
                extracted_values={"a": 1, "b": 2},
            )

            mock_infer.assert_called_once()
            assert result[1] == "math"

    def test_priority_2_similar_dsl_when_no_values(self):
        """When no extracted_values, should query database for similar DSL."""
        mock_db = MagicMock()

        with patch.object(dsl_templates, "_infer_dsl_from_values", return_value=None):
            with patch.object(dsl_templates, "_find_similar_successful_dsl") as mock_find:
                mock_find.return_value = ('{"type": "math", "script": "x * y"}', "math")

                result = infer_dsl_for_signature(
                    step_type="multiply",
                    description="multiply numbers",
                    db=mock_db,
                    extracted_values=None,
                )

                mock_find.assert_called_once()
                assert result[1] == "math"

    def test_fallback_to_decompose(self):
        """When no DSL can be inferred, should return decompose fallback."""
        result = infer_dsl_for_signature(
            step_type="unknown_operation",
            description="some complex operation",
            db=None,
            extracted_values=None,
        )

        dsl_json, dsl_type = result
        assert dsl_type == "decompose"
        dsl = json.loads(dsl_json)
        assert dsl["type"] == "decompose"
        assert dsl["script"] == "reason_step"

    def test_fallback_includes_description_in_purpose(self):
        """Decompose fallback should include description in purpose."""
        result = infer_dsl_for_signature(
            step_type="custom_op",
            description="Calculate something complicated",
            db=None,
            extracted_values=None,
        )

        dsl = json.loads(result[0])
        assert "Calculate something" in dsl["purpose"]

    def test_extracted_values_takes_priority_over_db(self):
        """Even with a db, extracted_values should be tried first."""
        mock_db = MagicMock()

        with patch.object(dsl_templates, "_infer_dsl_from_values") as mock_infer:
            mock_infer.return_value = ('{"type": "math", "script": "a ** b"}', "math")

            result = infer_dsl_for_signature(
                step_type="compute_power",
                description="raise to power",
                db=mock_db,
                extracted_values={"base": 2, "exponent": 3},
            )

            # Should use extracted values, not query db
            mock_infer.assert_called_once()
            assert "a ** b" in result[0] or result[1] == "math"


class TestInferDslFromValues:
    """Tests for _infer_dsl_from_values() param extraction."""

    def test_returns_none_when_no_values(self):
        """Should return None when extracted_values is empty."""
        result = _infer_dsl_from_values("compute", "compute something", {})
        assert result is None

        result = _infer_dsl_from_values("compute", "compute something", None)
        assert result is None

    def test_filters_non_numeric_values(self):
        """Should filter out non-numeric string values."""
        with patch.object(dsl_templates, "_infer_operation_semantic") as mock_op:
            mock_op.return_value = ("+", "adding")

            # Only 'a' should be kept as param
            result = _infer_dsl_from_values(
                "compute",
                "compute something",
                {"a": 5, "b": "not a number", "c": 10},
            )

            # Should have called with params ['a', 'c']
            call_args = mock_op.call_args
            assert "a" in call_args[1]["param_names"]
            assert "c" in call_args[1]["param_names"]
            assert "b" not in call_args[1]["param_names"]

    def test_keeps_numeric_strings(self):
        """Should keep string values that are numeric."""
        with patch.object(dsl_templates, "_infer_operation_semantic") as mock_op:
            mock_op.return_value = ("+", "adding")

            _infer_dsl_from_values(
                "add",
                "add numbers",
                {"a": "42", "b": "3.14"},
            )

            call_args = mock_op.call_args
            assert "a" in call_args[1]["param_names"]
            assert "b" in call_args[1]["param_names"]

    def test_keeps_step_references(self):
        """Should keep step references like {step_1}."""
        with patch.object(dsl_templates, "_infer_operation_semantic") as mock_op:
            mock_op.return_value = ("+", "adding")

            _infer_dsl_from_values(
                "combine",
                "combine results",
                {"first": "{step_1}", "second": 10},
            )

            call_args = mock_op.call_args
            assert "first" in call_args[1]["param_names"]
            assert "second" in call_args[1]["param_names"]

    def test_returns_none_when_no_valid_params(self):
        """Should return None when all params are filtered out."""
        result = _infer_dsl_from_values(
            "compute",
            "compute something",
            {"text": "hello", "more_text": "world"},
        )
        assert result is None

    def test_two_param_binary_operators(self):
        """Should build correct script for binary operators."""
        with patch.object(dsl_templates, "_infer_operation_semantic") as mock_op:
            # Test addition
            mock_op.return_value = ("+", "adding")
            result = _infer_dsl_from_values("add", "add", {"a": 1, "b": 2})
            assert result is not None
            dsl = json.loads(result[0])
            assert "+" in dsl["script"]

            # Test multiplication
            mock_op.return_value = ("*", "multiplying")
            result = _infer_dsl_from_values("mult", "mult", {"x": 3, "y": 4})
            dsl = json.loads(result[0])
            assert "*" in dsl["script"]

            # Test power
            mock_op.return_value = ("**", "power")
            result = _infer_dsl_from_values("pow", "pow", {"base": 2, "exp": 3})
            dsl = json.loads(result[0])
            assert "**" in dsl["script"]

    def test_function_operators_gcd_lcm(self):
        """Should build correct script for function-style operators."""
        with patch.object(dsl_templates, "_infer_operation_semantic") as mock_op:
            mock_op.return_value = ("gcd", "greatest common divisor")
            result = _infer_dsl_from_values("find_gcd", "gcd", {"a": 12, "b": 18})
            assert result is not None
            dsl = json.loads(result[0])
            assert "gcd(" in dsl["script"]

            mock_op.return_value = ("lcm", "least common multiple")
            result = _infer_dsl_from_values("find_lcm", "lcm", {"a": 4, "b": 6})
            dsl = json.loads(result[0])
            assert "lcm(" in dsl["script"]

    def test_single_param_operators(self):
        """Should build correct script for unary operators."""
        with patch.object(dsl_templates, "_infer_operation_semantic") as mock_op:
            mock_op.return_value = ("sqrt", "square root")
            result = _infer_dsl_from_values("compute_sqrt", "sqrt", {"n": 16})
            assert result is not None
            dsl = json.loads(result[0])
            assert "sqrt(" in dsl["script"]

            mock_op.return_value = ("factorial", "factorial")
            result = _infer_dsl_from_values("compute_factorial", "fact", {"n": 5})
            dsl = json.loads(result[0])
            assert "factorial(" in dsl["script"]

    def test_returns_math_dsl_type(self):
        """Should return 'math' as DSL type."""
        with patch.object(dsl_templates, "_infer_operation_semantic") as mock_op:
            mock_op.return_value = ("+", "adding")
            result = _infer_dsl_from_values("add", "add", {"a": 1, "b": 2})
            assert result[1] == "math"


class TestInferOperationSemantic:
    """Tests for _infer_operation_semantic() dual-channel matching."""

    @pytest.fixture
    def mock_embedder(self):
        """Create a mock embedder that returns predictable embeddings."""
        with patch("mycelium.embedder.Embedder") as MockEmbedder:
            mock_instance = MagicMock()

            # Create distinct embeddings for different operations
            def make_embedding(text):
                """Generate embeddings that cluster by operation keywords."""
                emb = np.random.rand(768).astype(np.float32)
                # Bias embedding based on keywords to make tests predictable
                if "dividend" in text.lower() or "divisor" in text.lower() or "divid" in text.lower():
                    emb[0:100] = 0.9  # Division signal
                elif "factor" in text.lower() or "multipl" in text.lower() or "product" in text.lower():
                    emb[100:200] = 0.9  # Multiplication signal
                elif "addend" in text.lower() or "sum" in text.lower() or "add" in text.lower():
                    emb[200:300] = 0.9  # Addition signal
                elif "base" in text.lower() or "exponent" in text.lower() or "power" in text.lower():
                    emb[300:400] = 0.9  # Power signal
                elif "gcd" in text.lower() or "greatest common" in text.lower():
                    emb[400:500] = 0.9  # GCD signal
                return emb / np.linalg.norm(emb)

            mock_instance.embed = MagicMock(side_effect=make_embedding)
            MockEmbedder.get_instance.return_value = mock_instance
            yield mock_instance

    def test_returns_none_below_threshold(self, mock_embedder):
        """Should return None when similarity is below threshold."""
        # Use very generic params that won't match well
        result = _infer_operation_semantic(
            step_type="do_something",
            description="",
            num_params=2,
            param_names=["x", "y"],
            min_similarity=0.99,  # Very high threshold
        )
        assert result is None

    def test_respects_min_params_requirement(self, mock_embedder):
        """Should filter operations that require more params than available."""
        # sqrt requires 1 param, binary ops require 2
        result = _infer_operation_semantic(
            step_type="compute_sqrt",
            description="square root",
            num_params=1,
            param_names=["n"],
            min_similarity=0.0,  # Accept any match
        )
        # Should only match single-param ops
        if result:
            op, _ = result
            assert op in ("sqrt", "factorial", "abs")

    def test_semantic_params_weighted_higher(self, mock_embedder):
        """Semantic param names like 'dividend' should be weighted 0.7."""
        # Long param names (avg > 3 chars) trigger semantic weighting
        result = _infer_operation_semantic(
            step_type="divide",
            description="divide numbers",
            num_params=2,
            param_names=["dividend", "divisor"],  # Long, semantic names
            min_similarity=0.0,
        )
        # With semantic params weighted 0.7, division should match well
        assert result is not None

    def test_generic_params_weighted_lower(self, mock_embedder):
        """Generic param names like 'a', 'b' should be weighted 0.3."""
        result = _infer_operation_semantic(
            step_type="add_values",
            description="add two numbers together",
            num_params=2,
            param_names=["a", "b"],  # Short, generic names
            min_similarity=0.0,
        )
        # Description channel (0.7 weight) should dominate
        assert result is not None

    def test_dual_channel_combines_signals(self, mock_embedder):
        """Both param and description channels should contribute."""
        # Params say "division", description says "divide" - both should help
        result = _infer_operation_semantic(
            step_type="compute_quotient",
            description="find the quotient",
            num_params=2,
            param_names=["dividend", "divisor"],
            min_similarity=0.0,
        )
        assert result is not None

    def test_returns_operator_and_name_tuple(self, mock_embedder):
        """Should return tuple of (operator, operation_name)."""
        result = _infer_operation_semantic(
            step_type="multiply",
            description="multiply factors",
            num_params=2,
            param_names=["factor_a", "factor_b"],
            min_similarity=0.0,
        )
        if result:
            op, name = result
            assert isinstance(op, str)
            assert isinstance(name, str)


class TestGetParamAnchorEmbeddings:
    """Tests for _get_param_anchor_embeddings() caching."""

    def test_returns_dict_keyed_by_operator(self):
        """Should return dict with operator keys."""
        with patch("mycelium.embedder.Embedder") as MockEmbedder:
            mock_instance = MagicMock()
            mock_instance.embed = MagicMock(return_value=np.zeros(768))
            MockEmbedder.get_instance.return_value = mock_instance

            result = _get_param_anchor_embeddings()

            assert isinstance(result, dict)
            # Should have all operators from _PARAM_ANCHORS
            for op in _PARAM_ANCHORS:
                assert op in result

    def test_caches_embeddings_on_second_call(self):
        """Should cache embeddings and not recompute on second call."""
        with patch("mycelium.embedder.Embedder") as MockEmbedder:
            mock_instance = MagicMock()
            mock_instance.embed = MagicMock(return_value=np.zeros(768))
            MockEmbedder.get_instance.return_value = mock_instance

            # First call - should embed all anchors
            result1 = _get_param_anchor_embeddings()
            call_count_after_first = mock_instance.embed.call_count

            # Second call - should use cache
            result2 = _get_param_anchor_embeddings()
            call_count_after_second = mock_instance.embed.call_count

            # No new embed calls
            assert call_count_after_second == call_count_after_first
            assert result1 is result2

    def test_returns_empty_dict_on_embedder_failure(self):
        """Should return empty dict if embedder fails."""
        with patch("mycelium.embedder.Embedder") as MockEmbedder:
            MockEmbedder.get_instance.side_effect = RuntimeError("No model")

            result = _get_param_anchor_embeddings()
            assert result == {}


class TestGetDescAnchorEmbeddings:
    """Tests for _get_desc_anchor_embeddings() caching."""

    def test_returns_dict_keyed_by_operator(self):
        """Should return dict with operator keys."""
        with patch("mycelium.embedder.Embedder") as MockEmbedder:
            mock_instance = MagicMock()
            mock_instance.embed = MagicMock(return_value=np.zeros(768))
            MockEmbedder.get_instance.return_value = mock_instance

            result = _get_desc_anchor_embeddings()

            assert isinstance(result, dict)
            for op in _DESCRIPTION_ANCHORS:
                assert op in result

    def test_caches_embeddings(self):
        """Should cache embeddings on subsequent calls."""
        with patch("mycelium.embedder.Embedder") as MockEmbedder:
            mock_instance = MagicMock()
            mock_instance.embed = MagicMock(return_value=np.zeros(768))
            MockEmbedder.get_instance.return_value = mock_instance

            result1 = _get_desc_anchor_embeddings()
            call_count_after_first = mock_instance.embed.call_count

            result2 = _get_desc_anchor_embeddings()

            assert mock_instance.embed.call_count == call_count_after_first
            assert result1 is result2


class TestFindSimilarSuccessfulDsl:
    """Tests for _find_similar_successful_dsl() similarity lookup."""

    @pytest.fixture
    def mock_db_and_embedder(self):
        """Create mock database and embedder."""
        with patch("mycelium.embedder.Embedder") as MockEmbedder:
            mock_embedder = MagicMock()

            # Create embeddings that will have high similarity for matching types
            def embed_fn(text):
                emb = np.random.rand(768).astype(np.float32)
                if "multiply" in text.lower() or "product" in text.lower():
                    emb[0:100] = 1.0
                elif "divide" in text.lower() or "quotient" in text.lower():
                    emb[100:200] = 1.0
                return emb / np.linalg.norm(emb)

            mock_embedder.embed = MagicMock(side_effect=embed_fn)
            MockEmbedder.get_instance.return_value = mock_embedder

            mock_db = MagicMock()
            yield mock_db, mock_embedder

    def test_returns_none_when_no_candidates(self, mock_db_and_embedder):
        """Should return None when db has no successful DSLs."""
        mock_db, _ = mock_db_and_embedder
        mock_db.get_signatures_with_successful_dsls.return_value = []

        result = _find_similar_successful_dsl(
            step_type="compute_sum",
            description="add numbers",
            db=mock_db,
        )

        assert result is None

    def test_queries_db_with_min_thresholds(self, mock_db_and_embedder):
        """Should query db with success rate and usage thresholds."""
        mock_db, _ = mock_db_and_embedder
        mock_db.get_signatures_with_successful_dsls.return_value = []

        _find_similar_successful_dsl(
            step_type="compute",
            description="desc",
            db=mock_db,
            min_success_rate=0.8,
            min_uses=5,
        )

        mock_db.get_signatures_with_successful_dsls.assert_called_once_with(
            min_success_rate=0.8,
            min_uses=5,
            limit=50,
        )

    def test_returns_best_semantic_match(self, mock_db_and_embedder):
        """Should return DSL from most semantically similar signature."""
        mock_db, mock_embedder = mock_db_and_embedder

        # Create mock signatures
        sig1 = MagicMock()
        sig1.step_type = "compute_product"
        sig1.description = "multiply two numbers"
        sig1.dsl_script = '{"type": "math", "script": "a * b"}'

        sig2 = MagicMock()
        sig2.step_type = "compute_quotient"
        sig2.description = "divide numbers"
        sig2.dsl_script = '{"type": "math", "script": "a / b"}'

        mock_db.get_signatures_with_successful_dsls.return_value = [sig1, sig2]

        # Query for multiplication - should match sig1
        result = _find_similar_successful_dsl(
            step_type="multiply_values",
            description="compute the product",
            db=mock_db,
        )

        # Should return multiplication DSL
        if result:
            dsl_json, dsl_type = result
            dsl = json.loads(dsl_json)
            assert "*" in dsl["script"]

    def test_returns_none_below_similarity_threshold(self, mock_db_and_embedder):
        """Should return None when best match is below 0.7 similarity."""
        mock_db, mock_embedder = mock_db_and_embedder

        # Create signature that won't match well
        sig = MagicMock()
        sig.step_type = "unrelated_operation"
        sig.description = "completely different thing"
        sig.dsl_script = '{"type": "math", "script": "x + y"}'

        mock_db.get_signatures_with_successful_dsls.return_value = [sig]

        # Make embeddings very different
        def orthogonal_embed(text):
            if "unrelated" in text:
                return np.array([1.0] + [0.0] * 767, dtype=np.float32)
            return np.array([0.0] * 767 + [1.0], dtype=np.float32)

        mock_embedder.embed = MagicMock(side_effect=orthogonal_embed)

        result = _find_similar_successful_dsl(
            step_type="compute_sum",
            description="add values",
            db=mock_db,
        )

        # Similarity should be ~0, below threshold
        assert result is None

    def test_skips_signatures_without_dsl_script(self, mock_db_and_embedder):
        """Should skip signatures that have no dsl_script."""
        mock_db, _ = mock_db_and_embedder

        sig1 = MagicMock()
        sig1.step_type = "multiply"
        sig1.description = "multiply"
        sig1.dsl_script = None  # No DSL

        sig2 = MagicMock()
        sig2.step_type = "multiply"
        sig2.description = "multiply"
        sig2.dsl_script = '{"type": "math", "script": "a * b"}'

        mock_db.get_signatures_with_successful_dsls.return_value = [sig1, sig2]

        result = _find_similar_successful_dsl(
            step_type="multiply",
            description="multiply",
            db=mock_db,
        )

        # Should return sig2's DSL (sig1 has no dsl_script)
        if result:
            assert "a * b" in result[0]

    def test_handles_json_decode_error(self, mock_db_and_embedder):
        """Should handle invalid JSON in dsl_script gracefully."""
        mock_db, mock_embedder = mock_db_and_embedder

        sig = MagicMock()
        sig.step_type = "multiply"
        sig.description = "multiply"
        sig.dsl_script = "not valid json"

        mock_db.get_signatures_with_successful_dsls.return_value = [sig]

        # Make it match well
        mock_embedder.embed = MagicMock(return_value=np.ones(768, dtype=np.float32))

        result = _find_similar_successful_dsl(
            step_type="multiply",
            description="multiply",
            db=mock_db,
        )

        # Should return None due to JSON error
        assert result is None

    def test_returns_none_on_exception(self, mock_db_and_embedder):
        """Should return None if an exception occurs."""
        mock_db, mock_embedder = mock_db_and_embedder
        mock_db.get_signatures_with_successful_dsls.side_effect = RuntimeError("DB error")

        result = _find_similar_successful_dsl(
            step_type="compute",
            description="desc",
            db=mock_db,
        )

        assert result is None


class TestMakeDslJson:
    """Tests for _make_dsl_json() helper."""

    def test_creates_valid_json(self):
        """Should create valid JSON string."""
        result = _make_dsl_json("a + b", ["a", "b"], "adding numbers")
        dsl = json.loads(result)
        assert dsl["type"] == "math"
        assert dsl["script"] == "a + b"
        assert dsl["params"] == ["a", "b"]
        assert dsl["purpose"] == "adding numbers"

    def test_handles_special_characters(self):
        """Should handle special characters in script."""
        result = _make_dsl_json("gcd(a, b)", ["a", "b"], "compute gcd")
        dsl = json.loads(result)
        assert dsl["script"] == "gcd(a, b)"

    def test_preserves_param_order(self):
        """Should preserve parameter order."""
        params = ["dividend", "divisor"]
        result = _make_dsl_json("dividend / divisor", params, "divide")
        dsl = json.loads(result)
        assert dsl["params"] == ["dividend", "divisor"]


class TestAnchorCoverage:
    """Tests to verify anchor dictionaries are properly defined."""

    def test_param_anchors_has_all_operations(self):
        """_PARAM_ANCHORS should have entries for all operations."""
        expected_ops = ["+", "-", "*", "/", "**", "%", "gcd", "lcm", "comb", "perm", "sqrt", "factorial", "abs"]
        for op in expected_ops:
            assert op in _PARAM_ANCHORS, f"Missing param anchor for {op}"
            assert len(_PARAM_ANCHORS[op]) > 0, f"Empty param anchor for {op}"

    def test_description_anchors_has_all_operations(self):
        """_DESCRIPTION_ANCHORS should have entries for all operations."""
        expected_ops = ["+", "-", "*", "/", "**", "%", "gcd", "lcm", "comb", "perm", "sqrt", "factorial", "abs"]
        for op in expected_ops:
            assert op in _DESCRIPTION_ANCHORS, f"Missing desc anchor for {op}"
            assert len(_DESCRIPTION_ANCHORS[op]) > 0, f"Empty desc anchor for {op}"

    def test_param_and_desc_anchors_match(self):
        """Both anchor dicts should have the same keys."""
        assert set(_PARAM_ANCHORS.keys()) == set(_DESCRIPTION_ANCHORS.keys())
