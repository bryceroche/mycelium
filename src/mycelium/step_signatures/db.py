"""StepSignatureDB: SQLite-backed database for step-level signatures."""

import json
import math
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, Literal

import numpy as np

from mycelium.data_layer import get_db
from mycelium.data_layer.schema import init_db
from mycelium.step_signatures.models import StepSignature
from mycelium.step_signatures.schema import StepIOSchema
from mycelium.step_signatures.utils import cosine_similarity, pack_embedding, unpack_embedding


MatchMode = Literal["cosine", "essence", "interference", "resonance", "auto"]


class StepSignatureDB:
    """SQLite-backed database for step-level signatures.

    Provides methods for finding, creating, and updating step signatures
    using SQLite for storage and in-memory cosine similarity for matching.
    """

    def __init__(self, db_path: str = None):
        """Initialize the database.

        Args:
            db_path: Optional path to SQLite database. If provided, creates
                     a direct connection instead of using the global singleton.
                     Useful for multiprocessing scenarios.
        """
        if db_path:
            # Direct connection for multiprocessing - bypass singleton
            import sqlite3
            self._direct_conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0)
            self._direct_conn.row_factory = sqlite3.Row
            self._direct_conn.execute("PRAGMA foreign_keys = ON")
            self._direct_conn.execute("PRAGMA journal_mode = WAL")  # Enable WAL for concurrent access
            self._direct_conn.execute("PRAGMA busy_timeout = 30000")  # 30s timeout for locks
            self._db = None
            self._db_path = db_path
        else:
            self._db = get_db()
            self._direct_conn = None
            self._db_path = None
        self._init_schema()

    @contextmanager
    def _connection(self):
        """Get a database connection (works with both singleton and direct modes)."""
        if self._direct_conn:
            yield self._direct_conn
            self._direct_conn.commit()
        else:
            with self._db.connection() as conn:
                yield conn

    def _init_schema(self):
        """Initialize database schema."""
        with self._connection() as conn:
            init_db(conn)

    def find_or_create(
        self,
        step_text: str,
        embedding: np.ndarray,
        min_similarity: float = 0.5,
        parent_problem: str = "",
        match_mode: MatchMode = "cosine",
        origin_depth: int = 0,
    ) -> tuple[StepSignature, bool]:
        """Find a matching signature or create a new one.

        Uses BEGIN IMMEDIATE to prevent race conditions when multiple
        workers try to create signatures for the same step pattern.

        Note: Default match_mode is "cosine" for predictable matching.
        The "auto" mode uses interference scoring which can produce low
        scores (0.1) for new signatures due to amplitude defaults.

        Args:
            step_text: The step description text
            embedding: Embedding vector for the step
            min_similarity: Minimum similarity threshold for matching
            parent_problem: The parent problem this step came from
            match_mode: Matching algorithm to use
            origin_depth: Decomposition depth at which this step was created

        Returns:
            Tuple of (signature, is_new) where is_new=True if newly created
        """
        import time
        max_retries = 10
        retry_delay = 0.05  # 50ms base delay

        for attempt in range(max_retries):
            try:
                return self._find_or_create_atomic(
                    step_text, embedding, min_similarity, parent_problem, match_mode, origin_depth
                )
            except Exception as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                raise

    def _find_or_create_atomic(
        self,
        step_text: str,
        embedding: np.ndarray,
        min_similarity: float,
        parent_problem: str,
        match_mode: MatchMode,
        origin_depth: int = 0,
    ) -> tuple[StepSignature, bool]:
        """Internal atomic find-or-create with transaction locking."""
        with self._connection() as conn:
            # Acquire write lock to prevent race conditions
            # BEGIN IMMEDIATE acquires a RESERVED lock, blocking other writers
            conn.execute("BEGIN IMMEDIATE")
            try:
                # Find best matching signature (inline to use same transaction)
                cursor = conn.execute("SELECT * FROM step_signatures")
                rows = cursor.fetchall()

                best_match = None
                best_score = 0.0

                for row in rows:
                    centroid = unpack_embedding(row["centroid"])
                    if centroid is None:
                        continue

                    sig = self._row_to_signature(row)
                    score = self._compute_score(embedding, centroid, sig, match_mode)

                    # Adaptive threshold based on cohesion
                    cohesion = row["cohesion"] or 0.5
                    adaptive_thresh = self._adaptive_threshold(min_similarity, cohesion)

                    # Further adjust for match modes that produce lower scores
                    if match_mode in ("interference", "resonance"):
                        adaptive_thresh *= 0.5

                    if score >= adaptive_thresh and score > best_score:
                        best_match = sig
                        best_score = score

                if best_match:
                    # Update last_used_at
                    now = datetime.utcnow().isoformat()
                    conn.execute(
                        "UPDATE step_signatures SET last_used_at = ? WHERE id = ?",
                        (now, best_match.id),
                    )
                    conn.commit()
                    return best_match, False

                # Create new signature (inline to use same transaction)
                sig = self._create_signature_atomic(conn, step_text, embedding, parent_problem, origin_depth)
                conn.commit()
                return sig, True

            except Exception:
                conn.rollback()
                raise

    def _create_signature_atomic(
        self,
        conn,
        step_text: str,
        embedding: np.ndarray,
        parent_problem: str = "",
        origin_depth: int = 0,
    ) -> StepSignature:
        """Create a new signature within an existing transaction.

        Used by find_or_create to maintain atomicity.

        Args:
            conn: Database connection
            step_text: The step description text
            embedding: Embedding vector for the step
            parent_problem: The parent problem this step came from
            origin_depth: Decomposition depth at which this step was created
        """
        sig_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        step_type = self._infer_step_type(step_text)
        dsl_script = self._get_default_dsl_script(step_type)

        # Signatures created at depth > 0 are from decomposition, likely atomic
        is_atomic = 1 if origin_depth > 0 else 0

        cursor = conn.execute(
            """INSERT INTO step_signatures
               (signature_id, centroid, step_type, description, method_name,
                method_template, example_count, created_at, dsl_script,
                origin_depth, is_atomic)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                sig_id,
                pack_embedding(embedding),
                step_type,
                step_text[:200],
                step_type,
                f"Solve: {step_text[:100]}",
                1,
                now,
                dsl_script,
                origin_depth,
                is_atomic,
            ),
        )
        row_id = cursor.lastrowid

        # Also add as first example
        conn.execute(
            """INSERT INTO step_examples
               (signature_id, step_text, embedding, parent_problem, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (row_id, step_text, pack_embedding(embedding), parent_problem, now),
        )

        return StepSignature(
            id=row_id,
            signature_id=sig_id,
            centroid=embedding,
            step_type=step_type,
            description=step_text[:200],
            method_name=step_type,
            method_template=f"Solve: {step_text[:100]}",
            example_count=1,
            uses=0,
            successes=0,
            cohesion=None,
            created_at=now,
            dsl_script=dsl_script,
            origin_depth=origin_depth,
            is_atomic=bool(is_atomic),
        )

    def _compute_score(
        self,
        embedding: np.ndarray,
        centroid: np.ndarray,
        sig: StepSignature,
        match_mode: MatchMode,
    ) -> float:
        """Compute similarity score based on match mode.

        Args:
            embedding: Query embedding
            centroid: Signature centroid
            sig: The signature (for amplitude/phase)
            match_mode: Matching algorithm

        Returns:
            Similarity score (higher = better match)
        """
        # Base cosine similarity
        cos_sim = cosine_similarity(embedding, centroid)

        if match_mode == "cosine":
            return cos_sim

        elif match_mode == "essence":
            # Essence mode: weight by high-variance dimensions
            # Simplified: use top 20% of dimensions by absolute value
            n_dims = int(len(embedding) * 0.2)
            indices = np.argsort(np.abs(centroid))[-n_dims:]
            essence_sim = cosine_similarity(embedding[indices], centroid[indices])
            # Blend essence and full cosine
            return 0.7 * essence_sim + 0.3 * cos_sim

        elif match_mode == "interference":
            # Interference mode: cosine * amplitude * gaussian decay
            amplitude = sig.amplitude if sig.amplitude else 0.1
            spread = sig.spread if sig.spread else 0.3
            # Distance in embedding space (1 - cosine as proxy)
            distance = 1.0 - cos_sim
            gaussian = math.exp(-(distance ** 2) / (spread ** 2))
            return cos_sim * amplitude * gaussian

        elif match_mode == "resonance":
            # Resonance mode: add phase-based frequency overlap
            amplitude = sig.amplitude if sig.amplitude else 0.1
            phase = sig.phase if sig.phase else 0.0
            # Frequency overlap based on embedding alignment
            freq_overlap = math.cos(phase) * cos_sim
            return cos_sim * amplitude * (0.5 + 0.5 * freq_overlap)

        elif match_mode == "auto":
            # Auto mode: use interference, fall back to cosine if score too low
            interference_score = self._compute_score(
                embedding, centroid, sig, "interference"
            )
            if interference_score < 0.1:
                return cos_sim
            return interference_score

        return cos_sim

    def _adaptive_threshold(self, base_threshold: float, cohesion: float) -> float:
        """Compute adaptive threshold based on cluster cohesion.

        Tight clusters (high cohesion) → stricter threshold
        Loose clusters (low cohesion) → more lenient threshold

        Formula: threshold = base + (cohesion - 0.5) * 0.2
        - cohesion=0.5 → base threshold (no adjustment)
        - cohesion=1.0 → base + 0.1 (stricter)
        - cohesion=0.0 → base - 0.1 (more lenient)

        Args:
            base_threshold: The base similarity threshold
            cohesion: Cluster cohesion score (0-1)

        Returns:
            Adjusted threshold clamped to [0.3, 0.95]
        """
        adjustment = (cohesion - 0.5) * 0.2
        adjusted = base_threshold + adjustment
        return max(0.3, min(0.95, adjusted))

    def _find_similar(
        self,
        embedding: np.ndarray,
        threshold: float = 0.5,
        limit: int = 10,
        match_mode: MatchMode = "auto",
    ) -> list[tuple[StepSignature, float]]:
        """Find signatures similar to the given embedding.

        Uses adaptive thresholds based on cluster cohesion:
        - Tight clusters need closer matches
        - Loose clusters allow more variance

        Args:
            embedding: Query embedding vector
            threshold: Base minimum similarity threshold
            limit: Maximum number of results
            match_mode: Matching algorithm to use

        Returns:
            List of (signature, similarity) tuples, sorted by similarity descending
        """
        with self._connection() as conn:
            cursor = conn.execute("SELECT * FROM step_signatures")
            rows = cursor.fetchall()

        results = []
        for row in rows:
            centroid = unpack_embedding(row["centroid"])
            if centroid is None:
                continue

            sig = self._row_to_signature(row)
            score = self._compute_score(embedding, centroid, sig, match_mode)

            # Adaptive threshold based on cohesion
            cohesion = row["cohesion"] or 0.5
            adaptive_thresh = self._adaptive_threshold(threshold, cohesion)

            # Further adjust for match modes that produce lower scores
            if match_mode in ("interference", "resonance"):
                adaptive_thresh *= 0.5
            elif match_mode == "essence":
                adaptive_thresh *= 0.7

            if score >= adaptive_thresh:
                results.append((sig, score))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def _create_signature(
        self,
        step_text: str,
        embedding: np.ndarray,
        parent_problem: str = "",
    ) -> StepSignature:
        """Create a new signature from a step.

        Args:
            step_text: The step description
            embedding: Embedding vector
            parent_problem: Parent problem context

        Returns:
            The created StepSignature
        """
        sig_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        # Generate step type from text (simplified)
        step_type = self._infer_step_type(step_text)

        # Get DSL script for known arithmetic step types
        dsl_script = self._get_default_dsl_script(step_type)

        with self._connection() as conn:
            cursor = conn.execute(
                """INSERT INTO step_signatures
                   (signature_id, centroid, step_type, description, method_name,
                    method_template, example_count, created_at, dsl_script)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    sig_id,
                    pack_embedding(embedding),
                    step_type,
                    step_text[:200],
                    step_type,
                    f"Solve: {step_text[:100]}",
                    1,
                    now,
                    dsl_script,
                ),
            )
            row_id = cursor.lastrowid

            # Also add as first example
            conn.execute(
                """INSERT INTO step_examples
                   (signature_id, step_text, embedding, parent_problem, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (row_id, step_text, pack_embedding(embedding), parent_problem, now),
            )

        return StepSignature(
            id=row_id,
            signature_id=sig_id,
            centroid=embedding,
            step_type=step_type,
            description=step_text[:200],
            method_name=step_type,
            method_template=f"Solve: {step_text[:100]}",
            example_count=1,
            created_at=now,
            dsl_script=dsl_script,  # Include DSL if step type has default
        )

    def _infer_step_type(self, step_text: str) -> str:
        """Infer a step type from the step text.

        Classifies steps into specific types to enable DSL generation.
        More specific types = better DSL reuse across similar problems.
        """
        text_lower = step_text.lower()

        # === ARITHMETIC OPERATIONS ===
        # Direct computation patterns
        if any(p in text_lower for p in ["calculate ", "compute ", "find the value", "evaluate "]):
            if "percentage" in text_lower or "percent of" in text_lower:
                return "compute_percentage"
            if " sum " in text_lower or "add " in text_lower or " total" in text_lower:
                return "compute_sum"
            if " product " in text_lower or "multiply" in text_lower:
                return "compute_product"
            if " difference " in text_lower or "subtract" in text_lower:
                return "compute_difference"
            if " quotient " in text_lower or " divided by " in text_lower or " ratio" in text_lower:
                return "compute_quotient"
            if "square root" in text_lower or "sqrt" in text_lower:
                return "compute_sqrt"
            if " square" in text_lower and "root" not in text_lower:
                return "compute_square"
            if "average" in text_lower or " mean" in text_lower:
                return "compute_average"

        # Verb-first arithmetic
        if text_lower.startswith("add "):
            return "compute_sum"
        if text_lower.startswith("subtract "):
            return "compute_difference"
        if text_lower.startswith("multiply "):
            return "compute_product"
        if text_lower.startswith("divide "):
            return "compute_quotient"

        # === COUNTING / COMBINATORICS ===
        if "how many ways" in text_lower or "number of ways" in text_lower:
            if "arrange" in text_lower or "order" in text_lower:
                return "count_permutations"
            if "choose" in text_lower or "select" in text_lower:
                return "count_combinations"
            return "count_ways"
        if "permutation" in text_lower:
            return "count_permutations"
        if "combination" in text_lower or "choose " in text_lower:
            return "count_combinations"
        if "factorial" in text_lower:
            return "compute_factorial"
        if any(p in text_lower for p in ["count ", "number of ", "how many "]):
            return "count_items"

        # === ALGEBRA / EQUATIONS ===
        if "system of equations" in text_lower or "simultaneous" in text_lower:
            return "solve_system"
        if any(p in text_lower for p in ["solve for", "solve the equation", "find x", "find the value of x", "solve the"]):
            if "quadratic" in text_lower:
                return "solve_quadratic"
            if "linear" in text_lower or "= 0" in text_lower:
                return "solve_linear"
            return "solve_equation"
        if "substitute" in text_lower or "plug in" in text_lower:
            return "substitute_value"
        if "simplify" in text_lower:
            if "fraction" in text_lower:
                return "simplify_fraction"
            return "simplify_expression"
        if "expand" in text_lower:
            return "expand_expression"
        if "factor" in text_lower:
            return "factor_expression"
        if "set up" in text_lower and "equation" in text_lower:
            return "setup_equation"

        # === FRACTIONS ===
        if "equivalent fraction" in text_lower or "scale the fraction" in text_lower:
            return "scale_fraction"
        if "reduce" in text_lower and "fraction" in text_lower:
            return "simplify_fraction"
        if "numerator" in text_lower and "denominator" in text_lower:
            return "fraction_parts"

        # === GEOMETRY ===
        if "area" in text_lower:
            if "circle" in text_lower:
                return "area_circle"
            if "triangle" in text_lower:
                return "area_triangle"
            if "rectangle" in text_lower or "square" in text_lower:
                return "area_rectangle"
            return "compute_area"
        if "perimeter" in text_lower or "circumference" in text_lower:
            return "compute_perimeter"
        if "volume" in text_lower:
            return "compute_volume"
        if "radius" in text_lower:
            return "compute_radius"
        if "arc length" in text_lower:
            return "arc_length"
        if "angle" in text_lower and any(p in text_lower for p in ["find", "calculate", "compute"]):
            return "compute_angle"

        # === NUMBER THEORY ===
        if "gcd" in text_lower or "greatest common" in text_lower:
            return "compute_gcd"
        if "lcm" in text_lower or "least common" in text_lower:
            return "compute_lcm"
        if "prime factor" in text_lower:
            return "prime_factorization"
        if "divisible" in text_lower or "divisor" in text_lower:
            return "check_divisibility"
        if "remainder" in text_lower or "modulo" in text_lower or " mod " in text_lower:
            return "compute_remainder"

        # === PROBABILITY ===
        if "probability" in text_lower:
            return "compute_probability"

        # === SEQUENCES / SERIES ===
        if "arithmetic sequence" in text_lower or "arithmetic progression" in text_lower:
            return "arithmetic_sequence"
        if "geometric sequence" in text_lower or "geometric progression" in text_lower:
            return "geometric_sequence"
        if "common difference" in text_lower:
            return "common_difference"
        if "common ratio" in text_lower:
            return "common_ratio"
        if "nth term" in text_lower:
            return "nth_term"

        # === EXPONENTS / LOGARITHMS ===
        if "exponent" in text_lower or " power" in text_lower or "^" in step_text:
            return "compute_power"
        if "logarithm" in text_lower or "log " in text_lower:
            return "compute_logarithm"

        # === BINOMIAL ===
        if "binomial" in text_lower:
            if "coefficient" in text_lower:
                return "binomial_coefficient"
            if "expand" in text_lower:
                return "binomial_expansion"

        # === TRIGONOMETRY ===
        # Be specific to avoid false matches (e.g., "using" contains "sin")
        trig_patterns = ["sin(", "cos(", "tan(", "sine ", "cosine ", "tangent ",
                         "sin ", "cos ", "tan ", " sin", " cos", " tan"]
        if any(p in text_lower for p in trig_patterns):
            return "trig_function"

        # === CONVERSIONS ===
        if "convert" in text_lower:
            if "base" in text_lower or "decimal" in text_lower or "binary" in text_lower:
                return "convert_base"
            if "radian" in text_lower or "degree" in text_lower:
                return "convert_angle"
            if "fraction" in text_lower:
                return "convert_fraction"
            return "convert_units"

        # === FUNCTION EVALUATION ===
        if "f(" in step_text or "g(" in step_text or "evaluate the function" in text_lower:
            return "evaluate_function"
        if "compute f(" in text_lower or "find f(" in text_lower or "calculate f(" in text_lower:
            return "evaluate_function"

        # === VECTOR/MATRIX OPERATIONS ===
        if "magnitude" in text_lower or "norm" in text_lower:
            return "compute_magnitude"
        if "matrix" in text_lower or "determinant" in text_lower:
            return "matrix_operation"
        if "vector" in text_lower:
            return "vector_operation"

        # === LENGTH/DISTANCE ===
        if "length" in text_lower or "distance" in text_lower:
            return "compute_length"

        # === PYTHAGOREAN THEOREM (standalone) ===
        if "pythagorean" in text_lower:
            return "pythagorean_theorem"

        # === LAW OF COSINES/SINES ===
        if "law of cosines" in text_lower or "cosine rule" in text_lower:
            return "law_of_cosines"
        if "law of sines" in text_lower or "sine rule" in text_lower:
            return "law_of_sines"

        # === INVERSE TRIG FUNCTIONS ===
        if any(p in text_lower for p in ["arctan", "arcsin", "arccos", "atan", "asin", "acos"]):
            return "inverse_trig"

        # === ADDITIONAL EQUATION PATTERNS ===
        if "isolate" in text_lower:
            return "solve_equation"
        if "equate" in text_lower:
            return "setup_equation"

        # === DEFINITIONS/FORMULAS ===
        if "define the formula" in text_lower or "write down the" in text_lower:
            return "define_formula"
        if "express" in text_lower and "in terms of" in text_lower:
            return "express_relation"

        # === SYNTHESIS/COMBINATION (high-use patterns) ===
        if any(p in text_lower for p in ["combine the results", "combine results", "synthesize"]):
            return "synthesize_results"

        # === INEQUALITIES ===
        if "am-gm" in text_lower or "arithmetic mean" in text_lower:
            return "apply_amgm"
        if "inequality" in text_lower:
            return "apply_inequality"

        # === COUNTING/CHOICES ===
        if "number of choices" in text_lower or "number of ways" in text_lower:
            return "count_choices"

        # === VARIABLE DEFINITIONS ===
        if "define the variable" in text_lower or "let " in text_lower[:10]:
            return "define_variables"

        # === VECTOR OPERATIONS (additional) ===
        if "normalize" in text_lower:
            return "normalize_vector"

        # === BINOMIAL EXPANSION ===
        if "expand" in text_lower and ("square" in text_lower or "binomial" in text_lower):
            return "expand_binomial"

        # === AREA CALCULATIONS (additional) ===
        if "area of triangle" in text_lower:
            return "area_triangle"

        # === ANGLE CALCULATIONS ===
        if "measure of" in text_lower and "angle" in text_lower:
            return "compute_angle"

        return "general_step"

    def _update_last_used(self, signature_id: int):
        """Update the last_used_at timestamp for a signature."""
        now = datetime.utcnow().isoformat()
        with self._connection() as conn:
            conn.execute(
                "UPDATE step_signatures SET last_used_at = ? WHERE id = ?",
                (now, signature_id),
            )

    def _get_default_dsl_script(self, step_type: str) -> Optional[str]:
        """Get default DSL script for step types that have deterministic solutions.

        Returns DSL for:
        - Simple arithmetic (math type)
        - Combinatorics with known formulas (math type)
        - Algebraic operations (sympy type)
        - Geometry formulas (math type)
        """
        dsl_scripts = {
            # === ARITHMETIC ===
            "compute_sum": json.dumps({
                "type": "math",
                "script": "a + b",
                "params": ["a", "b"],
                "fallback": "guidance"
            }),
            "compute_product": json.dumps({
                "type": "math",
                "script": "a * b",
                "params": ["a", "b"],
                "fallback": "guidance"
            }),
            "compute_difference": json.dumps({
                "type": "math",
                "script": "a - b",
                "params": ["a", "b"],
                "fallback": "guidance"
            }),
            "compute_quotient": json.dumps({
                "type": "math",
                "script": "a / b",
                "params": ["a", "b"],
                "fallback": "guidance"
            }),
            "compute_percentage": json.dumps({
                "type": "math",
                "script": "(percentage / 100) * base_value",
                "params": ["percentage", "base_value"],
                "fallback": "guidance"
            }),
            "compute_sqrt": json.dumps({
                "type": "math",
                "script": "sqrt(x)",
                "params": ["x"],
                "fallback": "guidance"
            }),
            "compute_square": json.dumps({
                "type": "math",
                "script": "x ** 2",
                "params": ["x"],
                "fallback": "guidance"
            }),
            "compute_average": json.dumps({
                "type": "math",
                "script": "(a + b) / 2",
                "params": ["a", "b"],
                "fallback": "guidance"
            }),
            "compute_power": json.dumps({
                "type": "math",
                "script": "base ** exponent",
                "params": ["base", "exponent"],
                "fallback": "guidance"
            }),
            "compute_remainder": json.dumps({
                "type": "math",
                "script": "a % b",
                "params": ["a", "b"],
                "fallback": "guidance"
            }),

            # === COMBINATORICS ===
            "compute_factorial": json.dumps({
                "type": "math",
                "script": "factorial(n)",
                "params": ["n"],
                "fallback": "guidance"
            }),
            "count_permutations": json.dumps({
                "type": "math",
                "script": "factorial(n) / factorial(n - r)",
                "params": ["n", "r"],
                "fallback": "guidance"
            }),
            "count_combinations": json.dumps({
                "type": "math",
                "script": "factorial(n) / (factorial(r) * factorial(n - r))",
                "params": ["n", "r"],
                "fallback": "guidance"
            }),
            "binomial_coefficient": json.dumps({
                "type": "math",
                "script": "factorial(n) / (factorial(k) * factorial(n - k))",
                "params": ["n", "k"],
                "fallback": "guidance"
            }),

            # === ALGEBRA (sympy) ===
            "solve_linear": json.dumps({
                "type": "sympy",
                "script": "solve(a*x + b, x)",
                "params": ["a", "b"],
                "fallback": "guidance"
            }),
            "solve_quadratic": json.dumps({
                "type": "sympy",
                "script": "solve(a*x**2 + b*x + c, x)",
                "params": ["a", "b", "c"],
                "fallback": "guidance"
            }),
            "simplify_expression": json.dumps({
                "type": "sympy",
                "script": "simplify(expr)",
                "params": ["expr"],
                "fallback": "guidance"
            }),
            "expand_expression": json.dumps({
                "type": "sympy",
                "script": "expand(expr)",
                "params": ["expr"],
                "fallback": "guidance"
            }),
            "factor_expression": json.dumps({
                "type": "sympy",
                "script": "factor(expr)",
                "params": ["expr"],
                "fallback": "guidance"
            }),

            # === GEOMETRY ===
            "area_circle": json.dumps({
                "type": "math",
                "script": "pi * r ** 2",
                "params": ["r"],
                "fallback": "guidance"
            }),
            "area_triangle": json.dumps({
                "type": "math",
                "script": "0.5 * base * height",
                "params": ["base", "height"],
                "fallback": "guidance"
            }),
            "area_rectangle": json.dumps({
                "type": "math",
                "script": "length * width",
                "params": ["length", "width"],
                "fallback": "guidance"
            }),
            "compute_perimeter": json.dumps({
                "type": "math",
                "script": "2 * (length + width)",
                "params": ["length", "width"],
                "fallback": "guidance"
            }),
            "compute_volume": json.dumps({
                "type": "math",
                "script": "length * width * height",
                "params": ["length", "width", "height"],
                "fallback": "guidance"
            }),
            "arc_length": json.dumps({
                "type": "math",
                "script": "(angle / 360) * 2 * pi * radius",
                "params": ["angle", "radius"],
                "fallback": "guidance"
            }),

            # === NUMBER THEORY ===
            "compute_gcd": json.dumps({
                "type": "math",
                "script": "gcd(a, b)",
                "params": ["a", "b"],
                "fallback": "guidance"
            }),
            "compute_lcm": json.dumps({
                "type": "math",
                "script": "(a * b) / gcd(a, b)",
                "params": ["a", "b"],
                "fallback": "guidance"
            }),

            # === PROBABILITY ===
            "compute_probability": json.dumps({
                "type": "math",
                "script": "favorable / total",
                "params": ["favorable", "total"],
                "fallback": "guidance"
            }),

            # === SEQUENCES ===
            "arithmetic_sequence": json.dumps({
                "type": "math",
                "script": "a1 + (n - 1) * d",
                "params": ["a1", "n", "d"],
                "fallback": "guidance"
            }),
            "geometric_sequence": json.dumps({
                "type": "math",
                "script": "a1 * r ** (n - 1)",
                "params": ["a1", "r", "n"],
                "fallback": "guidance"
            }),
            "common_difference": json.dumps({
                "type": "math",
                "script": "a2 - a1",
                "params": ["a1", "a2"],
                "fallback": "guidance"
            }),
            "common_ratio": json.dumps({
                "type": "math",
                "script": "a2 / a1",
                "params": ["a1", "a2"],
                "fallback": "guidance"
            }),

            # === GENERIC ALGEBRA (sympy) ===
            "solve_equation": json.dumps({
                "type": "sympy",
                "script": "solve(equation, x)",
                "params": ["equation"],
                "fallback": "guidance"
            }),
            "solve_system": json.dumps({
                "type": "sympy",
                "script": "solve([eq1, eq2], [x, y])",
                "params": ["eq1", "eq2"],
                "fallback": "guidance"
            }),

            # === TRIGONOMETRY (sympy for symbolic) ===
            "trig_function": json.dumps({
                "type": "sympy",
                "script": "simplify(expr)",
                "params": ["expr"],
                "fallback": "guidance"
            }),

            # === CONVERSIONS ===
            "convert_base": json.dumps({
                "type": "custom",
                "script": "int(str(number), from_base)",
                "params": ["number", "from_base"],
                "fallback": "guidance"
            }),
            "convert_angle": json.dumps({
                "type": "math",
                "script": "degrees * pi / 180",  # degrees to radians
                "params": ["degrees"],
                "fallback": "guidance"
            }),
            "convert_fraction": json.dumps({
                "type": "sympy",
                "script": "Rational(numerator, denominator)",
                "params": ["numerator", "denominator"],
                "fallback": "guidance"
            }),

            # === FUNCTION EVALUATION ===
            "evaluate_function": json.dumps({
                "type": "sympy",
                "script": "expr.subs(x, value)",
                "params": ["expr", "value"],
                "fallback": "guidance"
            }),

            # === VECTOR/MATRIX OPERATIONS ===
            "compute_magnitude": json.dumps({
                "type": "math",
                "script": "sqrt(sum(c**2 for c in components))",
                "params": ["components"],
                "fallback": "guidance"
            }),
            "matrix_operation": json.dumps({
                "type": "sympy",
                "script": "Matrix(matrix).det()",
                "params": ["matrix"],
                "fallback": "guidance"
            }),
            "vector_operation": json.dumps({
                "type": "sympy",
                "script": "Matrix(v1).dot(Matrix(v2))",
                "params": ["v1", "v2"],
                "fallback": "guidance"
            }),

            # === LENGTH/DISTANCE ===
            "compute_length": json.dumps({
                "type": "math",
                "script": "sqrt((x2 - x1)**2 + (y2 - y1)**2)",
                "params": ["x1", "y1", "x2", "y2"],
                "fallback": "guidance"
            }),
            "pythagorean_theorem": json.dumps({
                "type": "math",
                "script": "sqrt(a**2 + b**2)",
                "params": ["a", "b"],
                "fallback": "guidance"
            }),

            # === LAW OF COSINES/SINES ===
            "law_of_cosines": json.dumps({
                "type": "math",
                "script": "sqrt(a**2 + b**2 - 2*a*b*cos(C))",
                "params": ["a", "b", "C"],
                "fallback": "guidance"
            }),
            "law_of_sines": json.dumps({
                "type": "sympy",
                "script": "a / sin(A)",
                "params": ["a", "A"],
                "fallback": "guidance"
            }),

            # === INVERSE TRIG ===
            "inverse_trig": json.dumps({
                "type": "math",
                "script": "atan(y / x)",
                "params": ["y", "x"],
                "fallback": "guidance"
            }),

            # === VECTOR OPERATIONS (additional) ===
            "normalize_vector": json.dumps({
                "type": "math",
                "script": "v / sqrt(sum(c**2 for c in v))",
                "params": ["v"],
                "fallback": "guidance"
            }),

            # === BINOMIAL EXPANSION ===
            "expand_binomial": json.dumps({
                "type": "sympy",
                "script": "expand((a + b)**n)",
                "params": ["a", "b", "n"],
                "fallback": "guidance"
            }),

            # === COUNTING ===
            "count_choices": json.dumps({
                "type": "math",
                "script": "n",  # Often just the count itself
                "params": ["n"],
                "fallback": "guidance"
            }),

            # === AM-GM INEQUALITY ===
            "apply_amgm": json.dumps({
                "type": "math",
                "script": "sqrt(a * b)",  # Geometric mean
                "params": ["a", "b"],
                "fallback": "guidance"
            }),

        }
        # Only return DSL for specific step types, NOT general_step
        # General steps need LLM reasoning, not DSL execution
        return dsl_scripts.get(step_type)

    def update_dsl_script(self, signature_id: int, dsl_script: str) -> None:
        """Update the DSL script for a signature.

        Called when DSL is generated for a reliable signature.
        """
        with self._connection() as conn:
            conn.execute(
                "UPDATE step_signatures SET dsl_script = ? WHERE id = ?",
                (dsl_script, signature_id),
            )

    def get_signature_examples(self, signature_id: int, limit: int = 10) -> list[dict]:
        """Get example steps for a signature (for DSL generation context)."""
        with self._connection() as conn:
            cursor = conn.execute(
                """SELECT step_text, result, success FROM step_examples
                   WHERE signature_id = ? ORDER BY created_at DESC LIMIT ?""",
                (signature_id, limit),
            )
            return [dict(row) for row in cursor.fetchall()]

    def _row_to_signature(self, row) -> StepSignature:
        """Convert a database row to a StepSignature object."""
        io_schema = None
        if row["io_schema"]:
            io_schema = StepIOSchema.from_json(row["io_schema"])

        # Get dsl fields from row - handle missing columns gracefully
        row_keys = row.keys()
        dsl_script = row["dsl_script"] if "dsl_script" in row_keys else None
        dsl_version = row["dsl_version"] if "dsl_version" in row_keys else 1
        dsl_version_uses = row["dsl_version_uses"] if "dsl_version_uses" in row_keys else 0
        origin_depth = row["origin_depth"] if "origin_depth" in row_keys else 0
        is_atomic = bool(row["is_atomic"]) if "is_atomic" in row_keys else False

        return StepSignature(
            id=row["id"],
            signature_id=row["signature_id"],
            centroid=unpack_embedding(row["centroid"]),
            step_type=row["step_type"],
            description=row["description"] or "",
            method_name=row["method_name"],
            method_template=row["method_template"],
            example_count=row["example_count"],
            uses=row["uses"],
            successes=row["successes"],
            injected_uses=row["injected_uses"],
            injected_successes=row["injected_successes"],
            non_injected_uses=row["non_injected_uses"],
            non_injected_successes=row["non_injected_successes"],
            cohesion=row["cohesion"],
            amplitude=row["amplitude"],
            phase=row["phase"],
            spread=row["spread"],
            is_canonical=bool(row["is_canonical"]),
            canonical_parent_id=row["canonical_parent_id"],
            variant_count=row["variant_count"],
            origin_depth=origin_depth or 0,
            is_atomic=is_atomic,
            created_at=row["created_at"],
            last_used_at=row["last_used_at"],
            io_schema=io_schema,
            dsl_script=dsl_script,
            dsl_version=dsl_version or 1,
            dsl_version_uses=dsl_version_uses or 0,
            plan_type=row["plan_type"],
            compressed_instruction=row["compressed_instruction"],
            param_schema=json.loads(row["param_schema"]) if row["param_schema"] else None,
            output_format=row["output_format"],
            plan_optimization_method=row["plan_optimization_method"],
            plan_tokens_before=row["plan_tokens_before"],
            plan_tokens_after=row["plan_tokens_after"],
            plan_validation_accuracy=row["plan_validation_accuracy"],
        )

    def record_usage(
        self,
        signature_id: int,
        step_text: str,
        success: bool,
        was_injected: bool = False,
        match_mode: str = "auto",
    ):
        """Record usage of a signature.

        Args:
            signature_id: ID of the signature used
            step_text: The step text that was executed
            success: Whether the step succeeded
            was_injected: Whether a method was injected
            match_mode: The matching mode used
        """
        now = datetime.utcnow().isoformat()

        with self._connection() as conn:
            # Insert usage log
            conn.execute(
                """INSERT INTO step_usage_log
                   (signature_id, step_text, success, was_injected, match_mode, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (signature_id, step_text, int(success), int(was_injected), match_mode, now),
            )

            # Update signature statistics
            if was_injected:
                conn.execute(
                    """UPDATE step_signatures
                       SET uses = uses + 1,
                           successes = successes + ?,
                           injected_uses = injected_uses + 1,
                           injected_successes = injected_successes + ?,
                           last_used_at = ?
                       WHERE id = ?""",
                    (int(success), int(success), now, signature_id),
                )
            else:
                conn.execute(
                    """UPDATE step_signatures
                       SET uses = uses + 1,
                           successes = successes + ?,
                           non_injected_uses = non_injected_uses + 1,
                           non_injected_successes = non_injected_successes + ?,
                           last_used_at = ?
                       WHERE id = ?""",
                    (int(success), int(success), now, signature_id),
                )

    def get_signature(self, signature_id: int) -> Optional[StepSignature]:
        """Get a signature by ID."""
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM step_signatures WHERE id = ?",
                (signature_id,),
            )
            row = cursor.fetchone()

        if row is None:
            return None
        return self._row_to_signature(row)

    def get_stats(self) -> dict:
        """Get database statistics."""
        with self._connection() as conn:
            sig_count = conn.execute(
                "SELECT COUNT(*) FROM step_signatures"
            ).fetchone()[0]

            example_count = conn.execute(
                "SELECT COUNT(*) FROM step_examples"
            ).fetchone()[0]

            usage_count = conn.execute(
                "SELECT COUNT(*) FROM step_usage_log"
            ).fetchone()[0]

            success_count = conn.execute(
                "SELECT SUM(successes) FROM step_signatures"
            ).fetchone()[0] or 0

            total_uses = conn.execute(
                "SELECT SUM(uses) FROM step_signatures"
            ).fetchone()[0] or 0

        return {
            "signatures": sig_count,
            "examples": example_count,
            "usage_logs": usage_count,
            "total_uses": total_uses,
            "total_successes": success_count,
            "success_rate": success_count / total_uses if total_uses > 0 else 0.0,
        }

    def get_signature_hints(self, limit: int = 20) -> list[tuple[str, str]]:
        """Get top signatures as hints for the planner.

        Returns reliable signatures sorted by usage, formatted as
        (step_type, description) tuples for injection into planner prompt.

        Args:
            limit: Maximum number of hints to return

        Returns:
            List of (step_type, description) tuples
        """
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT step_type, description
                FROM step_signatures
                WHERE uses >= 3
                  AND (successes * 1.0 / uses) >= 0.7
                  AND step_type != 'general_step'
                ORDER BY uses DESC, successes DESC
                LIMIT ?
                """,
                (limit,)
            ).fetchall()

        return [(row[0], row[1]) for row in rows]

    def get_negative_lift_signatures(
        self,
        min_uses: int = 5,
        lift_threshold: float = -0.05
    ) -> list[StepSignature]:
        """Get signatures with negative lift (DSL hurts accuracy).

        Used to build the dynamic "avoid" embedding space.

        Args:
            min_uses: Minimum injected AND baseline uses to calculate lift
            lift_threshold: Lift below this is considered negative

        Returns:
            List of signatures where DSL injection hurts accuracy
        """
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM step_signatures
                WHERE dsl_script IS NOT NULL
                  AND injected_uses >= ?
                  AND non_injected_uses >= ?
                  AND (
                      (CAST(injected_successes AS FLOAT) / injected_uses)
                      - (CAST(non_injected_successes AS FLOAT) / non_injected_uses)
                  ) < ?
                """,
                (min_uses, min_uses, lift_threshold)
            ).fetchall()

        return [self._row_to_signature(row) for row in rows]

    def get_signatures_for_dsl_improvement(
        self,
        min_uses: int = 10,
        lift_threshold: float = -0.10
    ) -> list[StepSignature]:
        """Get signatures that need DSL improvement.

        Returns signatures with:
        - DSL script present
        - Sufficient usage data
        - Significantly negative lift (worse than threshold)

        These are candidates for DSL regeneration.
        """
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM step_signatures
                WHERE dsl_script IS NOT NULL
                  AND injected_uses >= ?
                  AND non_injected_uses >= ?
                  AND (
                      (CAST(injected_successes AS FLOAT) / injected_uses)
                      - (CAST(non_injected_successes AS FLOAT) / non_injected_uses)
                  ) < ?
                ORDER BY (
                    (CAST(injected_successes AS FLOAT) / injected_uses)
                    - (CAST(non_injected_successes AS FLOAT) / non_injected_uses)
                ) ASC
                """,
                (min_uses, min_uses, lift_threshold)
            ).fetchall()

        return [self._row_to_signature(row) for row in rows]

    def reset_lift_stats_for_dsl_version(
        self,
        signature_id: int,
        new_dsl_script: str,
        new_dsl_version: int
    ) -> None:
        """Reset lift stats when DSL is improved.

        This ensures the new DSL is evaluated fresh, without being
        penalized by the old DSL's failures.

        Args:
            signature_id: Signature to update
            new_dsl_script: The improved DSL script
            new_dsl_version: Version number (should be old version + 1)
        """
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE step_signatures
                SET dsl_script = ?,
                    dsl_version = ?,
                    dsl_version_uses = 0,
                    injected_uses = 0,
                    injected_successes = 0
                WHERE id = ?
                """,
                (new_dsl_script, new_dsl_version, signature_id)
            )

    # =========================================================================
    # Cluster Merging: Consolidate near-duplicate signatures
    # =========================================================================

    def find_merge_candidates(
        self,
        similarity_threshold: float = 0.90,
        success_rate_tolerance: float = 0.15,
        min_uses: int = 3,
    ) -> list[tuple[StepSignature, StepSignature, float]]:
        """Find pairs of signatures that are candidates for merging.

        Signatures are merge candidates if:
        1. Their centroids have cosine similarity >= threshold
        2. Their success rates are within tolerance of each other
        3. Both have at least min_uses (enough data to trust stats)

        Args:
            similarity_threshold: Minimum cosine similarity between centroids
            success_rate_tolerance: Max difference in success rates (0-1)
            min_uses: Minimum uses required for both signatures

        Returns:
            List of (sig_a, sig_b, similarity) tuples, sorted by similarity desc
        """
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM step_signatures WHERE uses >= ?",
                (min_uses,),
            )
            rows = cursor.fetchall()

        signatures = [(self._row_to_signature(row), unpack_embedding(row["centroid"]))
                      for row in rows]

        candidates = []
        n = len(signatures)

        for i in range(n):
            sig_a, centroid_a = signatures[i]
            if centroid_a is None:
                continue

            for j in range(i + 1, n):
                sig_b, centroid_b = signatures[j]
                if centroid_b is None:
                    continue

                # Check embedding similarity
                similarity = cosine_similarity(centroid_a, centroid_b)
                if similarity < similarity_threshold:
                    continue

                # Check success rate similarity
                rate_diff = abs(sig_a.success_rate - sig_b.success_rate)
                if rate_diff > success_rate_tolerance:
                    continue

                candidates.append((sig_a, sig_b, similarity))

        # Sort by similarity descending (merge most similar first)
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates

    def merge_signatures(
        self,
        survivor_id: int,
        absorbed_id: int,
    ) -> Optional[StepSignature]:
        """Merge two signatures, keeping survivor and deleting absorbed.

        The survivor signature:
        - Gets combined statistics (uses, successes, etc.)
        - Gets a weighted-average centroid
        - Inherits all examples from the absorbed signature

        Args:
            survivor_id: ID of signature to keep
            absorbed_id: ID of signature to merge into survivor and delete

        Returns:
            The updated survivor signature, or None if merge failed
        """
        survivor = self.get_signature(survivor_id)
        absorbed = self.get_signature(absorbed_id)

        if survivor is None or absorbed is None:
            return None

        # Compute weighted average centroid
        if survivor.centroid is not None and absorbed.centroid is not None:
            w1 = survivor.example_count or 1
            w2 = absorbed.example_count or 1
            new_centroid = (w1 * survivor.centroid + w2 * absorbed.centroid) / (w1 + w2)
        else:
            new_centroid = survivor.centroid

        # Combine statistics
        new_uses = survivor.uses + absorbed.uses
        new_successes = survivor.successes + absorbed.successes
        new_example_count = survivor.example_count + absorbed.example_count
        new_injected_uses = survivor.injected_uses + absorbed.injected_uses
        new_injected_successes = survivor.injected_successes + absorbed.injected_successes
        new_non_injected_uses = survivor.non_injected_uses + absorbed.non_injected_uses
        new_non_injected_successes = survivor.non_injected_successes + absorbed.non_injected_successes

        now = datetime.utcnow().isoformat()

        with self._connection() as conn:
            # Update survivor with combined stats
            conn.execute(
                """UPDATE step_signatures SET
                   centroid = ?,
                   uses = ?,
                   successes = ?,
                   example_count = ?,
                   injected_uses = ?,
                   injected_successes = ?,
                   non_injected_uses = ?,
                   non_injected_successes = ?,
                   last_used_at = ?
                   WHERE id = ?""",
                (
                    pack_embedding(new_centroid),
                    new_uses,
                    new_successes,
                    new_example_count,
                    new_injected_uses,
                    new_injected_successes,
                    new_non_injected_uses,
                    new_non_injected_successes,
                    now,
                    survivor_id,
                ),
            )

            # Reassign examples from absorbed to survivor
            conn.execute(
                "UPDATE step_examples SET signature_id = ? WHERE signature_id = ?",
                (survivor_id, absorbed_id),
            )

            # Update usage logs to point to survivor
            conn.execute(
                "UPDATE step_usage_log SET signature_id = ? WHERE signature_id = ?",
                (survivor_id, absorbed_id),
            )

            # Delete absorbed signature
            conn.execute(
                "DELETE FROM step_signatures WHERE id = ?",
                (absorbed_id,),
            )

        return self.get_signature(survivor_id)

    def merge_similar_signatures(
        self,
        similarity_threshold: float = 0.90,
        success_rate_tolerance: float = 0.15,
        min_uses: int = 3,
        max_merges: int = 10,
    ) -> list[dict]:
        """Find and merge similar signatures.

        Runs periodic cluster consolidation to merge near-duplicate signatures
        that have similar embeddings AND similar success rates.

        Args:
            similarity_threshold: Minimum cosine similarity (default 0.90)
            success_rate_tolerance: Max success rate difference (default 0.15)
            min_uses: Minimum uses for both signatures (default 3)
            max_merges: Maximum merges per call (default 10)

        Returns:
            List of merge records: {survivor_id, absorbed_id, similarity}
        """
        candidates = self.find_merge_candidates(
            similarity_threshold=similarity_threshold,
            success_rate_tolerance=success_rate_tolerance,
            min_uses=min_uses,
        )

        merged = []
        merged_ids = set()  # Track already-merged signatures

        for sig_a, sig_b, similarity in candidates:
            if len(merged) >= max_merges:
                break

            # Skip if either signature was already merged this round
            if sig_a.id in merged_ids or sig_b.id in merged_ids:
                continue

            # Survivor is the one with more examples (more established)
            if sig_a.example_count >= sig_b.example_count:
                survivor, absorbed = sig_a, sig_b
            else:
                survivor, absorbed = sig_b, sig_a

            result = self.merge_signatures(survivor.id, absorbed.id)
            if result:
                merged.append({
                    "survivor_id": survivor.id,
                    "survivor_type": survivor.step_type,
                    "absorbed_id": absorbed.id,
                    "absorbed_type": absorbed.step_type,
                    "similarity": similarity,
                    "combined_uses": result.uses,
                })
                merged_ids.add(absorbed.id)

        return merged
