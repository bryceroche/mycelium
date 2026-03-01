# Mycelium v7: The GNN Navigator — Online Strategy Recognition from Growing DAGs

## Core Insight

The DAG doesn't exist before inference. It grows node by node as C3 executes operations and creates PRIOR_N edges. A graph neural network placed AFTER C3 watches the DAG assemble in real time and recognizes the emerging strategy — like a chess commentator who sees the first few moves and says "ah, this is a Sicilian Defense."

The GNN doesn't predict the strategy before the game starts. It recognizes the strategy as it unfolds, and uses that recognition to guide what happens next.

---

## Why After C3, Not Before

### Before C3: No graph exists
C2 outputs a SET of operations: `{DIV, ADD, SOLVE}`. Three disconnected nodes. No edges. No topology. A GNN on disconnected nodes is just a set classifier — MiniLM already does that job as C2.

### After C3: Real graph exists
Each time C3 outputs `PRIOR_N`, that creates a directed edge from step N to the current step. After 2-3 C3 executions, there's genuine graph topology — branching, merging, chain length, fan-in, fan-out. This is what GNNs are designed to process.

---

## Architecture

### The Loop

```
C2: What operations? → {DIV, ADD, SOLVE, SIMPLIFY}

Iteration 1:
    GNN(partial_dag=[], remaining=[DIV, ADD, SOLVE, SIMPLIFY]) → DIV (start here)
    C3(DIV, text) → [TEXT_48, IMPLICIT_half]
    C4 → "48 / 2"
    Sympy → 24
    DAG: [DIV:24]

Iteration 2:
    GNN(partial_dag=[DIV:24], remaining=[ADD, SOLVE, SIMPLIFY]) → ADD
    C3(ADD, text + [PRIOR_1:24]) → [PRIOR_1, TEXT_48]
    C4 → "24 + 48"
    Sympy → 72
    DAG: [DIV:24] → [ADD:72]

Iteration 3:
    GNN(partial_dag=[DIV→ADD], remaining=[SOLVE, SIMPLIFY]) → SOLVE
    C3(SOLVE, text + [PRIOR_1:24, PRIOR_2:72]) → [PRIOR_2, TEXT_x]
    C4 → "solve(72 - x, x)"
    Sympy → x = 72
    DAG: [DIV:24] → [ADD:72] → [SOLVE:x=72]

Iteration 4:
    GNN(partial_dag=[DIV→ADD→SOLVE], remaining=[SIMPLIFY]) → SIMPLIFY
    ...
```

### GNN Input (per iteration)

```python
gnn_input = {
    # Existing nodes in the DAG
    "nodes": [
        {"id": 0, "template": "DIV", "result_type": "integer", "result": 24},
        {"id": 1, "template": "ADD", "result_type": "integer", "result": 72},
    ],
    # Existing edges (from C3 provenance labels)
    "edges": [
        {"source": 0, "target": 1, "type": "PRIOR"},  # ADD used DIV's result
    ],
    # Remaining operations from C2 (unexecuted)
    "remaining_ops": ["SOLVE", "SIMPLIFY"],
    # Problem-level features
    "heartbeat_count": 9,
    "difficulty_estimate": 3,
}
```

### GNN Output

```python
gnn_output = {
    # Which remaining operation to execute next
    "next_op": "SOLVE",
    "next_op_confidence": 0.87,
    
    # Strategy recognition (probability distribution)
    "strategy_probs": {
        "LINEAR_CHAIN": 0.72,
        "SOLVE_AND_SUBSTITUTE": 0.15,
        "EVALUATE_FUNCTION": 0.08,
        "UNKNOWN": 0.05,
    },
    
    # Anomaly detection
    "dag_coherence": 0.93,  # High = this looks like a pattern I've seen
                             # Low = something unusual, increase MCTS exploration
}
```

---

## Three Roles of the GNN

### Role 1: Operation Ordering (replaces C5 / MCTS permutations)

Without GNN: try all N! permutations of remaining operations, or use greedy heuristic.

With GNN: predict the single best next operation from the partial DAG structure. The GNN has seen thousands of DAGs and knows that after a DIV→ADD pattern, SOLVE usually comes next — not SIMPLIFY.

This reduces per-step branching from N remaining operations to 1-2 high-confidence choices. For a 5-operation problem, this is 5 GNN forward passes vs 120 permutations.

### Role 2: Strategy Recognition (enables v7 strategy-level MCTS)

As the DAG grows, the GNN continuously updates its strategy estimate. Early on (1-2 nodes), the estimate is vague: "probably arithmetic." After 3-4 nodes, it sharpens: "this is complete-the-square with 85% confidence."

This enables strategy-level MCTS from the v7 vision:
- If the GNN is confident (>80% one strategy), commit to that strategy's remaining operations
- If the GNN is uncertain (no strategy >50%), branch: explore 2-3 strategy interpretations in parallel
- Each MCTS branch follows a different strategy's predicted operation sequence

### Role 3: Error Detection (anomaly signal)

The GNN's dag_coherence score measures how well the partial DAG matches known patterns. If C3 extracts a wrong operand and the resulting sympy value is unexpected, the DAG will look anomalous — a SQRT producing a negative number, an ADD whose result is smaller than both operands.

Low coherence triggers:
- Backtrack: discard the last node, try C3's next beam candidate
- Re-branch: increase MCTS exploration width for subsequent steps
- Flag: report to the user that confidence is low for this problem

---

## GNN Architecture Details

### Graph Representation

Each node in the DAG carries a feature vector:
```python
node_features = {
    "template_embedding": one_hot(template, 15),    # Operation type from C2
    "result_type": one_hot(type, 5),                 # integer, rational, symbolic, etc.
    "result_magnitude": log10(abs(result) + 1),      # Scale of the number
    "step_index": position / total_steps,            # How far through execution
    "operand_sources": multi_hot(sources, 4),        # TEXT, PRIOR, IMPLICIT, CONSTANT
}
```

Each edge carries:
```python
edge_features = {
    "dependency_type": one_hot(type, 4),     # PRIOR, shared operand, etc.
    "source_result_used": boolean,           # Did target use source's numeric result?
}
```

### Network Architecture

```
Node features → Linear(input_dim, 64)
                    ↓
                GraphConv(64, 128) + ReLU     # Message passing round 1
                    ↓
                GraphConv(128, 128) + ReLU    # Message passing round 2
                    ↓
                GraphConv(128, 64) + ReLU     # Message passing round 3
                    ↓
              Global mean pool                 # Graph-level representation
                    ↓
            Concat(graph_repr, remaining_ops_embedding, problem_features)
                    ↓
                Linear(combined, 128) + ReLU
                    ↓
            ┌───────┼───────────┐
            ↓       ↓           ↓
      next_op    strategy    coherence
      softmax    softmax     sigmoid
```

Three message-passing rounds: enough for information to propagate across typical 3-5 node DAGs. Global mean pooling collapses the variable-size graph into a fixed representation. Three output heads for the three roles.

### Why GNN and Not Transformer

A transformer on the DAG node sequence would process nodes in order but lose topology. Node 3 depending on node 1 (skipping node 2) is invisible to sequential attention unless explicitly encoded. GNNs process the GRAPH STRUCTURE natively — message passing follows edges, not sequence position.

Also: the DAGs are small (3-10 nodes). GNN inference on 10 nodes is microseconds. No need for the heavy machinery of self-attention.

---

## Training

### Data Source: v6 DAG Corpus

Every problem solved by the v6 pipeline produces a complete DAG. With 7K+ MATH problems, we have 7K DAGs with:
- Node features (template, result, operand sources)
- Edge structure (PRIOR_N dependencies)
- Ground truth operation order (the order that produced a correct answer)
- Problem category and difficulty labels

### Training Task 1: Next Operation Prediction

For each DAG of N nodes, create N training examples:
```
(partial_dag[0:1], remaining[1:N]) → next = node[1]
(partial_dag[0:2], remaining[2:N]) → next = node[2]
(partial_dag[0:3], remaining[3:N]) → next = node[3]
...
```

Each partial DAG is a prefix of the complete DAG. The GNN learns to predict the next operation from the partial structure.

### Training Task 2: Strategy Classification

Label each complete DAG with its strategy type. Strategy labels come from graph motif mining (gSpan/SUBDUE on the DAG corpus) or from MATH problem category mappings.

Train on complete DAGs first (easy — full structure visible), then curriculum-learn on progressively smaller prefixes (harder — must recognize strategy from partial evidence).

### Training Task 3: Coherence Scoring

Generate "corrupted" DAGs by:
- Swapping two nodes' results
- Replacing an edge with a random connection
- Substituting a wrong template type

The GNN learns to distinguish real DAGs (coherence → 1.0) from corrupted ones (coherence → 0.0). At inference, low coherence on a real DAG indicates something went wrong upstream.

### Training Size

- 7K complete DAGs → ~35K partial DAG training examples (next-op prediction)
- 7K strategy labels (from motif mining)
- 14K coherence examples (7K real + 7K corrupted)
- Small model (~500K parameters), trains in minutes on a single GPU

---

## Integration with MCTS

### Without GNN (current v6)
```
C2: {DIV, ADD, SOLVE}
MCTS: try all 6 orderings × 10 beams per op = 60+ paths
Sympy: execute each path
Vote: majority on valid results
```

### With GNN (v7)
```
C2: {DIV, ADD, SOLVE}

Path 1 (GNN confident → single path):
    GNN: DIV first (0.92) → C3 → C4 → sympy
    GNN: ADD next (0.87) → C3 → C4 → sympy
    GNN: SOLVE last (0.95) → C3 → C4 → sympy
    → answer: 72

Path 2 (GNN less confident → branch at step 2):
    GNN: DIV first (0.92) → C3 → C4 → sympy
    GNN: SOLVE next (0.55) vs ADD (0.40) → BRANCH
        Branch A: SOLVE → ADD → answer: 72
        Branch B: ADD → SOLVE → answer: 72
    → both agree: 72 (high confidence)
```

The GNN controls MCTS branching:
- High confidence → single path, no branching
- Low confidence → branch, let both paths execute
- Very low confidence → full MCTS exploration

This is adaptive search guided by structural pattern recognition, not random permutation.

---

## The Knot Theory Connection

The GNN learns topological features of DAGs. Two DAGs with different node labels but the same topology (same edges, same branching structure) will have similar GNN representations.

This IS the knot invariant idea, but learned instead of computed:
- **Crossing number** → approximated by number of edges that "cross" in a topological ordering
- **Jones polynomial** → approximated by the GNN's graph-level embedding (both are topological invariants)
- **Reidemeister equivalence** → DAGs that differ only by Reidemeister moves will have similar GNN embeddings

The GNN discovers its own topological invariants from data. If two problems have isomorphic DAGs (same topology, different numbers), the GNN will assign them the same strategy — exactly what knot invariants do.

---

## Implementation Plan

### Phase 1: Build DAG Corpus (requires working v6 pipeline)
1. Run v6 (C2→C3→C4→sympy) on 7K MATH problems
2. Record complete DAG for each correctly solved problem
3. Store: node features, edge list, operation order, problem metadata
4. Expected: 2-3K correctly solved problems with complete DAGs

### Phase 2: Train Next-Op GNN
1. Generate partial DAG training examples from complete DAGs
2. Train 3-layer GraphConv with next-op prediction head
3. Validate: does GNN predict correct operation order on held-out DAGs?
4. Expected: >80% next-op accuracy (most orderings are nearly deterministic)

### Phase 3: Strategy Discovery
1. Run graph motif mining (gSpan) on DAG corpus
2. Cluster motifs → 20-50 strategy templates
3. Add strategy classification head to GNN
4. Train on complete DAGs, then curriculum to partial DAGs

### Phase 4: Integrate into Pipeline
1. Insert GNN between C3 iterations (after each node is added)
2. Replace greedy/MCTS ordering with GNN next-op prediction
3. Use strategy confidence to control MCTS branching width
4. Use coherence score for error detection and backtracking

### Phase 5: Evaluate
1. Compare v6 (no GNN, MCTS permutations) vs v7 (GNN-guided)
2. Metrics: accuracy, paths explored, inference time
3. Expected: higher accuracy with fewer paths (structured search > brute force)

---

## Dependencies

- **v6 must work first.** The GNN trains on v6's DAG output. No DAGs → no training data.
- **C3 pointer must be accurate.** If C3 builds wrong DAGs (wrong PRIOR_N edges), the GNN learns wrong patterns.
- **Strategy labels need the DAG corpus.** Can't do motif mining without DAGs.

Critical path: v6 accuracy → DAG corpus → GNN training → v7 integration

---

## Size and Speed

- GNN parameters: ~500K (tiny compared to C2's 22M and C3's 500M)
- GNN inference: <1ms per forward pass (small graph, 3-10 nodes)
- Training: minutes on a single GPU (small model, small dataset)
- Memory: negligible

The GNN adds almost zero cost to the pipeline. Its value is in REDUCING cost — fewer MCTS paths explored, fewer wasted C3 calls, faster convergence to the correct answer.

---

## Summary

The GNN is the navigator. C2 is the map (what operations exist). C3 is the driver (executes each step). C4 is the engine (builds expressions). Sympy is the road (mathematical truth). The GNN watches the route unfold and says "turn left here" — not because it was told the destination, but because it recognizes the road.

```
Without GNN:  Explore every road → expensive, many dead ends
With GNN:     Recognize the route → efficient, structured exploration
```

The DAG builds itself from C3's provenance labels. The GNN reads the DAG as it grows. The strategy emerges from structure, not from labels. The navigator learns the map by driving it.
