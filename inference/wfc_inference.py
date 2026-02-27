#!/usr/bin/env python3
"""
Mycelium v6: MCTS Inference (Wave Function Collapse)

6-model architecture with UCB1-guided tree search.

Pipeline:
    Problem → C1 (relevance field) → C6 (goal type, backward constraint)
      → C2 (top-k of 100 templates) → C3 (sympy expressions)
      → C5 (DAG wiring) → Sympy (FULL COLLAPSE)

Each model outputs distributions, not hard decisions.
MCTS explores the hypothesis space guided by confidence.
Sympy execution is the only collapse point.

No heuristics. No keywords. No regex. Everything learned.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Protocol
from enum import Enum
import math


# =============================================================================
# MODEL PROTOCOLS (interfaces for trained models)
# =============================================================================

class C1Protocol(Protocol):
    """C1: Relevance Scorer - outputs continuous field, not spans."""

    def predict_relevance(self, problem_text: str) -> List[float]:
        """
        Returns relevance score (0-1) per token.
        High values = tokens the teacher attended to during computation.
        """
        ...

    def predict_relevance_batch(self, texts: List[str]) -> List[List[float]]:
        """Batched version for efficiency."""
        ...


class C2Protocol(Protocol):
    """C2: Classifier - relevance field → IB template."""

    def predict_template(
        self,
        problem_text: str,
        relevance_field: List[float],
        cluster_mask: Optional[List[bool]] = None
    ) -> Dict[str, float]:
        """
        Returns probability distribution over 100 IB templates.
        cluster_mask: which tokens belong to this cluster (optional).
        """
        ...

    def predict_template_batch(
        self,
        texts: List[str],
        relevance_fields: List[List[float]],
        cluster_masks: Optional[List[List[bool]]] = None
    ) -> List[Dict[str, float]]:
        """Batched version."""
        ...


class C3Protocol(Protocol):
    """C3: Extractor - template + relevance → sympy expression."""

    def extract_expression(
        self,
        problem_text: str,
        relevance_field: List[float],
        template: str,
        k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Returns top-k (expression_str, confidence) pairs.
        expression_str is sympy-parseable.
        """
        ...

    def extract_expression_batch(
        self,
        texts: List[str],
        relevance_fields: List[List[float]],
        templates: List[str],
        k: int = 3
    ) -> List[List[Tuple[str, float]]]:
        """Batched version."""
        ...


class C5Protocol(Protocol):
    """C5: Dependency Resolver - pairwise DAG wiring."""

    def predict_dependency(
        self,
        problem_text: str,
        cluster_i: Dict,  # First operation cluster
        cluster_j: Dict,  # Second operation cluster
    ) -> float:
        """
        Returns P(cluster_j depends on cluster_i's output).
        High = j uses i's result. Low = independent.
        """
        ...

    def predict_dag(
        self,
        problem_text: str,
        clusters: List[Dict],
    ) -> List[Tuple[int, int, float]]:
        """
        Returns list of (i, j, prob) dependency edges.
        """
        ...


class C6Protocol(Protocol):
    """C6: Goal Resolver - answer type (backward constraint)."""

    def predict_goal(self, problem_text: str) -> Dict[str, float]:
        """
        Returns probability distribution over answer types:
        - INTEGER, FRACTION, DECIMAL, SET, EXPRESSION, etc.
        """
        ...


class ExecutorProtocol(Protocol):
    """Symbolic executor - sympy evaluation."""

    def execute(self, expression: str, context: Optional[Dict] = None) -> Optional[Any]:
        """
        Execute sympy expression. Returns result or None if invalid.
        """
        ...

    def validate_answer_type(self, result: Any, expected_type: str) -> bool:
        """
        Check if result matches expected answer type (C6 backward constraint).
        """
        ...


# =============================================================================
# ANSWER TYPES (C6 output space)
# =============================================================================

class AnswerType(Enum):
    INTEGER = "INTEGER"
    FRACTION = "FRACTION"
    DECIMAL = "DECIMAL"
    PERCENTAGE = "PERCENTAGE"
    MONEY = "MONEY"
    SET = "SET"
    EXPRESSION = "EXPRESSION"
    UNKNOWN = "UNKNOWN"


# =============================================================================
# DECOHERENCE: Adaptive branching based on confidence
# =============================================================================

class Decoherence:
    """
    Decoherence threshold θ: controls when to collapse vs explore.

    High confidence (max_prob > θ) → collapse to k=1
    Low confidence (max_prob < θ) → explore top-k

    Easy problems → high confidence → minimal branching
    Hard problems → low confidence → full exploration
    """

    def __init__(self, theta: float = 0.85):
        self.theta = theta

    def adaptive_k(self, probs: Dict[str, float], max_k: int) -> int:
        """Return adaptive k based on confidence."""
        if not probs:
            return max_k
        max_prob = max(probs.values())
        return 1 if max_prob >= self.theta else max_k

    def should_collapse(self, probs: Dict[str, float]) -> bool:
        """Check if we should collapse to single hypothesis."""
        if not probs:
            return False
        return max(probs.values()) >= self.theta

    def top_k(self, probs: Dict[str, float], max_k: int) -> List[Tuple[str, float]]:
        """Return top-k items with adaptive k."""
        k = self.adaptive_k(probs, max_k)
        sorted_items = sorted(probs.items(), key=lambda x: -x[1])
        return sorted_items[:k]


# =============================================================================
# MCTS NODE
# =============================================================================

class MCTSStage(Enum):
    """Stages in the MCTS tree."""
    ROOT = "ROOT"           # Initial state
    RELEVANCE = "RELEVANCE" # After C1
    GOAL = "GOAL"           # After C6
    TEMPLATE = "TEMPLATE"   # After C2
    EXPRESSION = "EXPRESSION" # After C3
    DAG = "DAG"             # After C5
    EXECUTED = "EXECUTED"   # After sympy


@dataclass
class MCTSState:
    """State at an MCTS node - accumulates decisions through pipeline."""
    problem_text: str
    relevance_field: Optional[List[float]] = None
    goal_type: Optional[str] = None
    goal_confidence: float = 1.0
    clusters: List[Dict] = field(default_factory=list)  # Identified clusters
    templates: List[Tuple[str, float]] = field(default_factory=list)  # (template, conf) per cluster
    expressions: List[Tuple[str, float]] = field(default_factory=list)  # (expr, conf) per cluster
    dag_edges: List[Tuple[int, int, float]] = field(default_factory=list)
    result: Optional[Any] = None
    valid: bool = False
    stage: MCTSStage = MCTSStage.ROOT

    @property
    def total_confidence(self) -> float:
        """Joint probability across all decisions."""
        conf = self.goal_confidence
        for _, c in self.templates:
            conf *= c
        for _, c in self.expressions:
            conf *= c
        return conf


@dataclass
class MCTSNode:
    """Node in MCTS search tree."""
    state: MCTSState
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0

    @property
    def q_value(self) -> float:
        """Average value (exploitation term)."""
        return self.value / self.visits if self.visits > 0 else 0.0

    def ucb1(self, c: float = 1.414) -> float:
        """UCB1 score for node selection."""
        if self.visits == 0:
            return float('inf')
        if self.parent is None or self.parent.visits == 0:
            return self.q_value
        exploitation = self.q_value
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def best_child(self, c: float = 1.414) -> 'MCTSNode':
        """Select best child by UCB1."""
        return max(self.children, key=lambda n: n.ucb1(c))

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_terminal(self) -> bool:
        return self.state.stage == MCTSStage.EXECUTED


# =============================================================================
# MCTS INFERENCE ENGINE
# =============================================================================

class MCTSInference:
    """
    MCTS-based inference for 6-model architecture.

    Tree structure mirrors pipeline stages:
        ROOT → C1 (relevance) → C6 (goal) → C2 (template) → C3 (expression)
            → C5 (DAG) → Sympy (execute)

    UCB1 balances exploitation (high confidence paths) and exploration.
    Sympy execution provides binary reward. No learned value function.
    """

    def __init__(
        self,
        c1: C1Protocol,
        c2: C2Protocol,
        c3: C3Protocol,
        c5: C5Protocol,
        c6: C6Protocol,
        executor: ExecutorProtocol,
        n_iterations: int = 100,
        ucb_c: float = 1.414,
        theta: float = 0.85,
        template_k: int = 3,
        expression_k: int = 3,
    ):
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c5 = c5
        self.c6 = c6
        self.executor = executor

        self.n_iterations = n_iterations
        self.ucb_c = ucb_c
        self.decoherence = Decoherence(theta=theta)
        self.template_k = template_k
        self.expression_k = expression_k

    def infer(self, problem_text: str) -> Dict[str, Any]:
        """
        Run MCTS inference on a problem.

        Returns:
            answer: best answer found (or None)
            confidence: confidence of answer
            valid_paths: number of valid execution paths
            iterations: MCTS iterations run
            diagnosis: error attribution info
        """
        # Create root
        root_state = MCTSState(problem_text=problem_text, stage=MCTSStage.ROOT)
        root = MCTSNode(state=root_state)

        # Run C1 once (deterministic) - creates single child
        self._expand_c1(root)

        valid_results: List[MCTSState] = []

        for _ in range(self.n_iterations):
            # SELECT: walk down tree using UCB1
            node = self._select(root)

            # EXPAND: add children based on current stage
            if node.is_leaf() and not node.is_terminal():
                self._expand(node)

            # SIMULATE: if we have children, pick one and try to execute
            if node.children:
                child = node.best_child(self.ucb_c)
                reward = self._simulate(child)

                if reward > 0 and child.state.result is not None:
                    valid_results.append(child.state)

                # BACKPROPAGATE
                self._backpropagate(child, reward)
            elif node.is_terminal():
                # Terminal node - evaluate directly
                reward = 1.0 if node.state.valid else 0.0
                if reward > 0:
                    valid_results.append(node.state)
                self._backpropagate(node, reward)

        # Select best valid result
        if valid_results:
            best = max(valid_results, key=lambda s: s.total_confidence)
            return {
                'answer': best.result,
                'confidence': best.total_confidence,
                'valid_paths': len(valid_results),
                'iterations': self.n_iterations,
                'goal_type': best.goal_type,
            }
        else:
            return {
                'answer': None,
                'confidence': 0.0,
                'valid_paths': 0,
                'iterations': self.n_iterations,
                'goal_type': None,
            }

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select leaf node using UCB1."""
        while not node.is_leaf() and not node.is_terminal():
            node = node.best_child(self.ucb_c)
        return node

    def _expand(self, node: MCTSNode) -> None:
        """Expand node based on current pipeline stage."""
        stage = node.state.stage

        if stage == MCTSStage.ROOT:
            self._expand_c1(node)
        elif stage == MCTSStage.RELEVANCE:
            self._expand_c6(node)
        elif stage == MCTSStage.GOAL:
            self._expand_c2(node)
        elif stage == MCTSStage.TEMPLATE:
            self._expand_c3(node)
        elif stage == MCTSStage.EXPRESSION:
            self._expand_c5(node)
        elif stage == MCTSStage.DAG:
            self._expand_execute(node)

    def _expand_c1(self, node: MCTSNode) -> None:
        """C1: Compute relevance field (deterministic, single child)."""
        state = node.state
        relevance = self.c1.predict_relevance(state.problem_text)

        new_state = MCTSState(
            problem_text=state.problem_text,
            relevance_field=relevance,
            stage=MCTSStage.RELEVANCE,
        )
        child = MCTSNode(state=new_state, parent=node)
        node.children.append(child)

    def _expand_c6(self, node: MCTSNode) -> None:
        """C6: Predict goal type (backward constraint)."""
        state = node.state
        goal_probs = self.c6.predict_goal(state.problem_text)

        # Adaptive k based on confidence
        top_goals = self.decoherence.top_k(goal_probs, max_k=2)

        for goal_type, conf in top_goals:
            new_state = MCTSState(
                problem_text=state.problem_text,
                relevance_field=state.relevance_field,
                goal_type=goal_type,
                goal_confidence=conf,
                stage=MCTSStage.GOAL,
            )
            child = MCTSNode(state=new_state, parent=node)
            node.children.append(child)

    def _expand_c2(self, node: MCTSNode) -> None:
        """C2: Classify clusters into templates."""
        state = node.state

        # Identify clusters from relevance field
        clusters = self._identify_clusters(state.relevance_field)

        if not clusters:
            # No clusters found - terminal failure
            new_state = MCTSState(
                problem_text=state.problem_text,
                relevance_field=state.relevance_field,
                goal_type=state.goal_type,
                goal_confidence=state.goal_confidence,
                stage=MCTSStage.EXECUTED,
                valid=False,
            )
            child = MCTSNode(state=new_state, parent=node)
            node.children.append(child)
            return

        # For now: classify first cluster, expand templates
        # TODO: Handle multiple clusters properly
        cluster = clusters[0]
        template_probs = self.c2.predict_template(
            state.problem_text,
            state.relevance_field,
            cluster.get('mask'),
        )

        top_templates = self.decoherence.top_k(template_probs, self.template_k)

        for template, conf in top_templates:
            new_state = MCTSState(
                problem_text=state.problem_text,
                relevance_field=state.relevance_field,
                goal_type=state.goal_type,
                goal_confidence=state.goal_confidence,
                clusters=clusters,
                templates=[(template, conf)],
                stage=MCTSStage.TEMPLATE,
            )
            child = MCTSNode(state=new_state, parent=node)
            node.children.append(child)

    def _expand_c3(self, node: MCTSNode) -> None:
        """C3: Extract expressions for each template."""
        state = node.state

        if not state.templates:
            return

        template, _ = state.templates[0]  # First cluster's template

        expressions = self.c3.extract_expression(
            state.problem_text,
            state.relevance_field,
            template,
            k=self.expression_k,
        )

        for expr, conf in expressions:
            new_state = MCTSState(
                problem_text=state.problem_text,
                relevance_field=state.relevance_field,
                goal_type=state.goal_type,
                goal_confidence=state.goal_confidence,
                clusters=state.clusters,
                templates=state.templates,
                expressions=[(expr, conf)],
                stage=MCTSStage.EXPRESSION,
            )
            child = MCTSNode(state=new_state, parent=node)
            node.children.append(child)

    def _expand_c5(self, node: MCTSNode) -> None:
        """C5: Resolve dependencies (DAG wiring)."""
        state = node.state

        # For single-cluster problems, no dependencies needed
        # Just pass through to execution
        new_state = MCTSState(
            problem_text=state.problem_text,
            relevance_field=state.relevance_field,
            goal_type=state.goal_type,
            goal_confidence=state.goal_confidence,
            clusters=state.clusters,
            templates=state.templates,
            expressions=state.expressions,
            dag_edges=[],  # No edges for single cluster
            stage=MCTSStage.DAG,
        )
        child = MCTSNode(state=new_state, parent=node)
        node.children.append(child)

        # TODO: For multi-cluster, expand C5 dependency hypotheses

    def _expand_execute(self, node: MCTSNode) -> None:
        """Execute expression through sympy."""
        state = node.state

        if not state.expressions:
            new_state = MCTSState(
                problem_text=state.problem_text,
                relevance_field=state.relevance_field,
                goal_type=state.goal_type,
                goal_confidence=state.goal_confidence,
                clusters=state.clusters,
                templates=state.templates,
                expressions=state.expressions,
                stage=MCTSStage.EXECUTED,
                valid=False,
            )
            child = MCTSNode(state=new_state, parent=node)
            node.children.append(child)
            return

        expr, _ = state.expressions[0]

        try:
            result = self.executor.execute(expr)

            # Backward constraint: check answer type matches C6 prediction
            if state.goal_type and result is not None:
                if not self.executor.validate_answer_type(result, state.goal_type):
                    result = None  # Prune incompatible result

            valid = result is not None

            new_state = MCTSState(
                problem_text=state.problem_text,
                relevance_field=state.relevance_field,
                goal_type=state.goal_type,
                goal_confidence=state.goal_confidence,
                clusters=state.clusters,
                templates=state.templates,
                expressions=state.expressions,
                dag_edges=state.dag_edges,
                result=result,
                valid=valid,
                stage=MCTSStage.EXECUTED,
            )
            child = MCTSNode(state=new_state, parent=node)
            node.children.append(child)

        except Exception:
            new_state = MCTSState(
                problem_text=state.problem_text,
                relevance_field=state.relevance_field,
                goal_type=state.goal_type,
                goal_confidence=state.goal_confidence,
                clusters=state.clusters,
                templates=state.templates,
                expressions=state.expressions,
                stage=MCTSStage.EXECUTED,
                valid=False,
            )
            child = MCTSNode(state=new_state, parent=node)
            node.children.append(child)

    def _simulate(self, node: MCTSNode) -> float:
        """
        Simulate from node to terminal state.

        For efficiency, we greedily expand (taking top-1 at each stage)
        until we hit execution, then return reward.
        """
        current = node

        while not current.is_terminal():
            if current.is_leaf():
                self._expand(current)

            if not current.children:
                return 0.0

            # Greedy: take highest confidence child
            current = max(current.children, key=lambda n: n.state.total_confidence)

        return 1.0 if current.state.valid else 0.0

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """Update values up the tree."""
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def _identify_clusters(self, relevance_field: List[float]) -> List[Dict]:
        """
        Identify high-relevance clusters from the relevance field.

        Returns list of cluster dicts with 'mask' and 'peak_score'.
        """
        if not relevance_field:
            return []

        threshold = 0.3  # Minimum relevance to consider
        clusters = []

        # Find contiguous regions above threshold
        in_cluster = False
        cluster_start = 0

        for i, score in enumerate(relevance_field):
            if score >= threshold and not in_cluster:
                in_cluster = True
                cluster_start = i
            elif score < threshold and in_cluster:
                in_cluster = False
                mask = [False] * len(relevance_field)
                for j in range(cluster_start, i):
                    mask[j] = True
                peak = max(relevance_field[cluster_start:i])
                clusters.append({'mask': mask, 'peak_score': peak, 'start': cluster_start, 'end': i})

        # Handle cluster at end
        if in_cluster:
            mask = [False] * len(relevance_field)
            for j in range(cluster_start, len(relevance_field)):
                mask[j] = True
            peak = max(relevance_field[cluster_start:])
            clusters.append({'mask': mask, 'peak_score': peak, 'start': cluster_start, 'end': len(relevance_field)})

        # Sort by peak score
        clusters.sort(key=lambda c: -c['peak_score'])

        return clusters


# =============================================================================
# MOCK MODELS FOR TESTING
# =============================================================================

class MockC1:
    """Mock C1: random relevance scores, higher for digit tokens."""
    def predict_relevance(self, text: str) -> List[float]:
        import random
        tokens = text.split()
        scores = []
        for t in tokens:
            if any(c.isdigit() for c in t):
                scores.append(random.uniform(0.5, 0.9))
            else:
                scores.append(random.uniform(0.0, 0.3))
        return scores

    def predict_relevance_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.predict_relevance(t) for t in texts]


class MockC2:
    """Mock C2: random template probabilities."""
    TEMPLATES = [f"T{i:02d}" for i in range(100)]  # T00-T99

    def predict_template(self, problem_text: str, relevance_field: List[float],
                         cluster_mask: Optional[List[bool]] = None) -> Dict[str, float]:
        import random
        # Random distribution over templates
        weights = [random.random() for _ in self.TEMPLATES]
        total = sum(weights)
        return {t: w/total for t, w in zip(self.TEMPLATES, weights)}

    def predict_template_batch(self, texts, relevance_fields, cluster_masks=None):
        return [self.predict_template(t, r) for t, r in zip(texts, relevance_fields)]


class MockC3:
    """Mock C3: extracts numbers and builds simple expressions."""
    def extract_expression(self, problem_text: str, relevance_field: List[float],
                          template: str, k: int = 3) -> List[Tuple[str, float]]:
        import random
        # Extract numbers
        numbers = []
        current = ""
        for c in problem_text:
            if c.isdigit() or c == '.':
                current += c
            elif current:
                try:
                    numbers.append(float(current))
                except:
                    pass
                current = ""
        if current:
            try:
                numbers.append(float(current))
            except:
                pass

        if len(numbers) >= 2:
            exprs = [
                (f"{numbers[0]} + {numbers[1]}", 0.4),
                (f"{numbers[0]} - {numbers[1]}", 0.3),
                (f"{numbers[0]} * {numbers[1]}", 0.2),
            ]
            return exprs[:k]
        elif len(numbers) == 1:
            return [(str(numbers[0]), 0.9)]
        else:
            return [("0", 0.1)]

    def extract_expression_batch(self, texts, relevance_fields, templates, k=3):
        return [self.extract_expression(t, r, tmpl, k)
                for t, r, tmpl in zip(texts, relevance_fields, templates)]


class MockC5:
    """Mock C5: no dependencies (single cluster)."""
    def predict_dependency(self, problem_text, cluster_i, cluster_j) -> float:
        return 0.0

    def predict_dag(self, problem_text, clusters):
        return []


class MockC6:
    """Mock C6: random goal type."""
    GOAL_TYPES = ["INTEGER", "DECIMAL", "FRACTION", "MONEY"]

    def predict_goal(self, problem_text: str) -> Dict[str, float]:
        import random
        weights = [random.random() for _ in self.GOAL_TYPES]
        total = sum(weights)
        return {g: w/total for g, w in zip(self.GOAL_TYPES, weights)}


class MockExecutor:
    """Mock executor: eval simple arithmetic expressions."""
    def execute(self, expression: str, context: Optional[Dict] = None) -> Optional[Any]:
        try:
            # Safety: only allow basic arithmetic
            allowed = set('0123456789.+-*/() ')
            if not all(c in allowed for c in expression):
                return None
            result = eval(expression)
            return result
        except:
            return None

    def validate_answer_type(self, result: Any, expected_type: str) -> bool:
        if result is None:
            return False
        if expected_type == "INTEGER":
            return isinstance(result, int) or (isinstance(result, float) and result == int(result))
        elif expected_type == "DECIMAL":
            return isinstance(result, (int, float))
        elif expected_type == "FRACTION":
            return True  # Accept any numeric
        elif expected_type == "MONEY":
            return isinstance(result, (int, float)) and result >= 0
        return True


# =============================================================================
# DEMO
# =============================================================================

def demo():
    print("=" * 60)
    print("MCTS INFERENCE: 6-Model Architecture")
    print("=" * 60)

    # Create mock models
    c1 = MockC1()
    c2 = MockC2()
    c3 = MockC3()
    c5 = MockC5()
    c6 = MockC6()
    executor = MockExecutor()

    # Create inference engine
    mcts = MCTSInference(
        c1=c1, c2=c2, c3=c3, c5=c5, c6=c6,
        executor=executor,
        n_iterations=50,
        theta=0.85,
    )

    # Test problem
    problem = "Bob has 5 apples. He gives 2 to Sally. How many apples does Bob have now?"
    print(f"\nProblem: {problem}")

    result = mcts.infer(problem)

    print(f"\nResult:")
    print(f"  Answer: {result['answer']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Valid paths: {result['valid_paths']}")
    print(f"  Goal type: {result['goal_type']}")
    print(f"  Iterations: {result['iterations']}")


if __name__ == "__main__":
    demo()
