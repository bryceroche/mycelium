# Design Doc: Tree-Structured Breathing with Scaled Controller

**Author:** Bryce + Claude (brainstorm session)
**Date:** April 28, 2026
**Status:** Design phase — implement after straight-through gradient fix is validated
**Depends on:** Baked L1, dual-layer LoRA, straight-through controller gradient, K=3 inner loop

---

## Motivation

The breathing loop is currently a flat sequential chain: cycle 1 → cycle 2 → cycle 3. Each cycle claims one target in order. But real math reasoning is hierarchical:

```
"If Sally had $20 less, she would have $80. If Jolly has $20 more,
 she would have $70. How much money do they have altogether?"

                    [MERGE: 100 + 50 = 150]
                   /                        \
    [SOLVE: 80 + 20 = 100]        [SOLVE: 70 - 20 = 50]
         (Sally branch)               (Jolly branch)
```

The current linear chain forces this into cycle 1 → cycle 2 → cycle 3, which works but is unnatural. The model has to implicitly learn "do Sally first, then Jolly, then add" when the problem structure says "Sally and Jolly are independent, then merge." As problems get harder (more entities, nested dependencies), the linear chain becomes increasingly limiting.

**Tree-structured breathing lets the controller express the problem's natural dependency graph.**

---

## Architecture Overview

### Core Idea

Extend the controller to make structural decisions alongside atom selection. Each breathing cycle becomes a **tree node** that can be one of three types:

- **DECOMPOSE:** Split this problem/subproblem into child branches
- **SOLVE:** Generate an equation and claim a target (leaf node)
- **MERGE:** Combine results from child branches (internal node)

The tree structure emerges from the controller's decisions — it's learned, not hardcoded.

### Implicit Tree via Branch Embeddings (Option A — Recommended)

Rather than discrete action selection (which needs REINFORCE), the controller produces a continuous **branch embedding** alongside its existing outputs. The tree structure is implicit in the geometry of these embeddings:

```
Controller outputs per cycle:
  - L2 atom scales        (64 dims)   — existing: which atoms to activate
  - Page vector           (256 dims)  — existing: what to record in notebook
  - Confidence            (1 dim)     — existing: should we stop?
  - Branch embedding      (64 dims)   — NEW: where am I in the tree?
  - Branch action logits  (3 dims)    — NEW: decompose / solve / merge
```

The branch embedding encodes the node's position in the tree. Nodes in the same subtree have similar embeddings. The merge operation attends to nodes with related branch embeddings.

### Tree Execution Flow

```
CYCLE 0: COMPREHEND
  Vanilla Llama reads problem → controller reads hidden states
  Controller decides: DECOMPOSE (this problem has independent subproblems)
  Creates branch embeddings for child subtrees

CYCLE 1: SOLVE (Branch A — e.g., Sally)
  Controller sets branch embedding → selects L2 atoms for this branch
  Llama generates equation → claims target
  Page records result with branch embedding

CYCLE 2: SOLVE (Branch B — e.g., Jolly)
  Controller sets different branch embedding → different L2 atoms
  Llama generates equation → claims target
  Page records result with branch embedding

CYCLE 3: MERGE
  Controller attends to Branch A and Branch B pages
  Selects L2 atoms for combining results
  Llama generates final equation → claims final target

STOP: confidence high or all targets claimed
```

For simple sequential problems (L4.5-style), the tree degenerates to a chain — every cycle is SOLVE with incrementally different branch embeddings. The controller learns when to branch and when to chain.

### Training the Tree Structure

**Branch action (decompose/solve/merge):** Use Gumbel-softmax on the 3 action logits for differentiable discrete selection during training. At inference, take argmax. This avoids REINFORCE entirely.

**Branch embedding contrastive loss:** Pages within the same subtree should have similar branch embeddings; pages in different subtrees should differ. Use a simple contrastive loss:

```python
# Pages from same branch (parent-child or siblings working on same subproblem)
same_branch_loss = (1 - cosine_sim(embed_i, embed_j)).mean()

# Pages from different branches
diff_branch_loss = max(0, cosine_sim(embed_i, embed_k) - margin).mean()

tree_loss = same_branch_loss + diff_branch_loss
```

**Equal-reward with tree structure:** The existing 1/N reward per target still works. But merge nodes can only claim the final target if their children successfully claimed intermediate targets. This creates natural gradient flow from merge → children: if the merge fails because a child got the wrong intermediate, the child's reward is affected.

**Target assignment:** Currently targets can be claimed in any order. With trees, constrain it: leaf nodes claim intermediate targets, merge nodes claim combination targets. The target dependency graph mirrors the tree structure. This can be soft (encouraged via reward shaping) rather than hard (enforced via masking).

---

## Scaled Controller (~350-400M)

### Why Scale Up

The controller is currently 190M and being asked to:
1. Read and understand Llama's 2048-dim hidden states (observation)
2. Track cycle/pass position (temporal awareness)
3. Manage notebook pages (memory)
4. Select 64 atom scales (action)
5. Estimate confidence (meta-cognition)
6. **NEW:** Make tree-structural decisions (planning)
7. **NEW:** Attend to tree-structured notebook (hierarchical memory)

This is qualitatively more complex than the original design. The controller needs more capacity, specifically in its observation and memory components.

### Where to Add Parameters

**State Encoder (~+80M): PRIORITY**
The component that reads Llama's hidden states and forms the controller's "understanding" of the problem state. This is the bottleneck — the controller sees Llama's 1.2B-parameter internal state through a small encoder. Scaling this up means:
- More attention heads in the Perceiver-style cross-attention (4 → 8 heads)
- Wider latent dimensions (512 → 768 or 1024)
- One additional encoder layer

**Tree-Structured Notebook Attention (~+60M): NEW**
Replace flat notebook attention with hierarchical attention that respects tree structure:
- Attend to parent page (what subproblem am I working on?)
- Attend to sibling pages (what have parallel branches produced?)
- Attend to child pages (for merge nodes: what results am I combining?)
- Branch-embedding-gated attention: attention weights modulated by branch embedding similarity

**Decision Heads (~+10M): SMALL**
- Branch embedding output: linear projection (small)
- Branch action logits: linear projection (tiny)
- Scale output, confidence, page output: unchanged

### Parameter Budget

```
Component                  Current    Scaled     Delta
─────────────────────────────────────────────────────
State encoder              ~80M       ~160M      +80M
Notebook attention         ~40M       ~100M      +60M
Scale/page/confidence      ~50M       ~60M       +10M
Branch modules             —          ~30M       +30M
─────────────────────────────────────────────────────
Total                      ~190M      ~350M      +160M
```

### Memory Budget Check

```
Component              Memory (fp16)
─────────────────────────────────────
Llama 1.2B (frozen)    ~2.4 GB
L1 (baked into Llama)  0 GB (absorbed)
L2 atoms (~82M)        ~164 MB
Controller (~350M)     ~700 MB
Optimizer states        ~1.4 GB (controller + L2)
Activations/gradients   ~2-4 GB
─────────────────────────────────────
Total                   ~7-9 GB (fits A10G 24GB)
```

Plenty of headroom. The old L1 atoms (82M) freed up ~164MB + optimizer states, and the controller scaling adds ~320MB + optimizer states. Net increase is small.

---

## Notebook as Tree Data Structure

### Current: Flat List
```python
notebook = [page_0, page_1, page_2, page_3]  # linear append
```

### Proposed: Tree with Parent Pointers
```python
@dataclass
class TreeNode:
    page: Tensor           # controller's recorded understanding (256 dims)
    branch_embed: Tensor   # position in tree (64 dims)
    action: str            # decompose / solve / merge
    parent_idx: int        # index of parent node (-1 for root)
    children_idx: list     # indices of child nodes
    generation: str        # Llama's generated text
    claimed_target: int    # which target was claimed (if any)
```

### Tree Attention Mechanism

The controller attends to the notebook differently based on tree structure:

```python
def tree_attention(query, notebook, current_node):
    # Attend to ancestors (path to root — what's the big picture?)
    ancestor_pages = get_ancestors(notebook, current_node)
    ancestor_ctx = cross_attend(query, ancestor_pages)
    
    # Attend to siblings (parallel branches — what have peers done?)
    sibling_pages = get_siblings(notebook, current_node)
    sibling_ctx = cross_attend(query, sibling_pages)
    
    # Attend to children (for merge — what results are available?)
    child_pages = get_children(notebook, current_node)
    child_ctx = cross_attend(query, child_pages)
    
    # Combine with learned gating
    return gate(ancestor_ctx, sibling_ctx, child_ctx)
```

### Gradient Flow Through Tree

Gradient flows naturally through the tree via the page vectors:
- Merge node's gen_loss → merge node's page attention → child pages → child nodes' controller
- This gives child nodes gradient signal: "your result was used in the merge and it was wrong/right"
- The straight-through estimator applies at each node independently
- No gradient detachment between parent and child nodes

---

## Training Strategy

### Phase 1: Validate on Existing Problems (No Tree Yet)
- Implement scaled controller (~350M) with branch embedding output
- Train on GSM8K with current linear chain
- Verify that scaled controller + straight-through gradient improves accuracy
- The branch embeddings should show sequential structure (ordered along one dimension)

### Phase 2: Enable Tree Actions
- Add Gumbel-softmax branch action selection
- Add tree-structured notebook with parent pointers
- Add contrastive branch embedding loss
- Train on GSM8K — let the controller discover which problems benefit from branching
- Many problems will stay sequential (tree degenerates to chain)
- Multi-entity problems (Sally + Jolly) should start branching

### Phase 3: Tree-Structured Curriculum
- Generate training problems with explicit tree structure:
  - 2-branch: "A does X. B does Y. Together they Z." (two leaves + merge)
  - 3-branch: "A, B, and C each do X. Total?" (three leaves + merge)
  - Nested: "A does X. B does Y based on A's result. C combines." (chain + merge)
- The tree structure provides supervision signal for branch actions
- Curriculum: simple trees → complex trees → GSM8K (discover structure)

### Phase 4: MATH-500
- With tree-structured breathing, the model can handle problems requiring:
  - Multiple independent computations (branches)
  - Sequential derivations (chains)
  - Results that depend on intermediate values (merges)
  - Nested subproblems (subtrees)

---

## Connection to Prior Work

### Factor Graph (from v7)
The tree structure IS the factor graph made operational. The v7 factor graph had: structural_consistency, dependency_flow, execution_validity, verification. The tree encodes dependency_flow directly — parent-child relationships ARE the dependencies.

### Expand-Collapse Dynamics (from early Mycelium)
The core thesis was that reasoning follows an expand-collapse rhythm. The tree structure makes this explicit:
- DECOMPOSE = expand (break problem into parts)
- SOLVE = work (compute each part)  
- MERGE = collapse (combine parts into answer)

### Information Bottleneck Discovery
The IB work showed that operation type is NOT classifiable from features. The tree structure doesn't try to classify operations — it lets the controller learn structural patterns (when to branch, when to chain) from the gradient signal. The operations themselves are handled by L2 atoms.

---

## Risks / Mitigation

| Risk | Mitigation |
|------|-----------|
| Tree structure adds too much complexity, model can't learn it | Phase 1 validates scaled controller without tree; add tree only after controller is learning well |
| Gumbel-softmax action selection is unstable | Start with high temperature (5.0), anneal slowly; fall back to soft mixture if discrete doesn't work |
| Tree degenerates to chain on all problems | This is fine — it means the problems don't need branching. Verify on multi-entity problems specifically |
| Controller too large, slows training | 350M is still <30% of Llama's cost per forward; inner loop iterations are ~20% of Llama cost each |
| Branch contrastive loss conflicts with gen_loss | Weight contrastive loss low (0.01-0.1); it's a structural regularizer, not the main training signal |
| Tree depth explodes on complex problems | Cap max depth at 4; cap total nodes at 8; let confidence signal terminate branches |

---

## Success Criteria

- **Phase 1:** Scaled controller achieves >20% GSM8K accuracy (with straight-through gradient, without tree)
- **Phase 2:** Multi-entity GSM8K problems (Sally + Jolly type) show branching behavior in the tree; accuracy on these problems specifically improves
- **Phase 3:** GSM8K accuracy exceeds 30% with tree structure
- **Phase 4:** MATH-500 shows meaningful accuracy with tree-structured breathing

---

## Implementation Priority

1. Scale controller to ~350M (Phase 1 — validates independently of tree)
2. Add branch embedding output head (cheap, runs alongside Phase 1)
3. Implement Gumbel-softmax action selection (Phase 2)
4. Implement tree-structured notebook with parent pointers (Phase 2)
5. Add tree attention mechanism to controller (Phase 2)
6. Add contrastive branch embedding loss (Phase 2)
7. Generate tree-structured curriculum data (Phase 3)
