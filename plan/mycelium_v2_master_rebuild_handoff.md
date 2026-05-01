# Mycelium v2: Master Rebuild Handoff

**Author:** Bryce + Claude (full-day architecture session)
**Date:** April 28, 2026
**Deadline:** September 1, 2026
**Target:** MATH-500 benchmark
**Infrastructure:** AWS EC2 g5.xlarge (A10G 24GB)

---

## Why Rebuild

Mycelium v1 reached 22% GSM8K accuracy with genuine multi-cycle decomposition — proving the core thesis that a frozen LLM can gain reasoning capability through an external breathing loop. But a series of debugging sessions revealed that the BreathingController has been a **constant function the entire time**. Every result was L2 atoms learning a universal blend without any per-problem steering. The controller's gradient was dead due to:

1. **Gradient routing through 1.2B frozen Llama** — 500x attenuation, effectively zero signal
2. **Double-bounded scales (clamp + tanh)** — hard zero gradient at boundaries
3. **Trunk collapse** — state encoder learned to ignore input during the zero-gradient period

The architecture is sound. The implementation has a foundational flaw: the controller was never able to learn. Rather than patching a broken foundation, we rebuild cleanly with the controller's gradient path as the primary design constraint.

---

## Core Design Principles

### 1. The Controller Gets Direct Gradient — Always
The controller and Llama have **separate loss functions**. The controller never receives gradient through Llama's forward pass. This is not an optimization — it is the foundational architectural rule.

### 2. Test the Controller Before Anything Else
Phase 0 is a smoke test: can the controller produce different outputs for different inputs? No Llama, no atoms, no data pipeline. If this fails, nothing else matters.

### 3. Tree Structure Is the Default
The breathing loop is tree-structured from day one. Linear chains are degenerate trees (every node has exactly one child). The notebook, attention mechanisms, and controller decisions are all designed for hierarchy.

### 4. The Right Geometry for Each Component
Atom scales in bounded Euclidean (independent per-atom control). Branch embeddings in hyperbolic space (natural tree metric). Pages on hypersphere (collapse prevention, high capacity). Hidden states in Euclidean (standard neural network processing).

### 5. Keep What Works
Baked L1 (proven), equal-reward decomposition (proven), bounded tanh scales (correct design), number augmentation (prevents memorization), procedural curriculum (validated progression).

---

## Architecture Specification

### System Overview

```
INPUT: Math problem (text)

FROZEN BASE MODEL: Llama 3.2 1B with baked L1 math-mode LoRA
  - L1 atoms absorbed into base weights permanently
  - Transforms Llama from token predictor → arithmetic engine
  - No runtime cost, no separate forward pass
  - Proven: 99.5% on L4.5 (3-step procedural), template-invariant

CONTROLLER (~400M): Tree-structured reasoning engine
  - Reads Llama's hidden states (observation)
  - Makes tree decisions: DECOMPOSE / SOLVE / MERGE (planning)
  - Selects L2 atom scales per node (action)
  - Estimates energy per node (evaluation)
  - Writes pages to tree notebook (memory)
  - Produces branch embeddings in Poincaré ball (tree position)
  - Gets DIRECT gradient via straight-through estimator (learning)

L2 ATOMS (~82M): Per-node LoRA steering
  - 64 atoms, bounded Euclidean scales via 0.46 * tanh(x)
  - Controller selects blend per tree node
  - Modifies Llama's attention for this specific reasoning step
  - Gets direct gradient through LoRA application

TREE NOTEBOOK: Hierarchical memory
  - Each node stores: page vector (256d hypersphere), branch embedding
    (64d Poincaré ball), generation text, claimed target, parent/child pointers
  - Tree attention: attend to ancestors, siblings, children
  - Gradient flows through page vectors from merge nodes to children

ENERGY HEAD: Learned stopping criterion
  - Scalar energy per node, trained with contrastive loss
  - Low energy = controller believes scales are correct
  - Adaptive inner loop: iterate until energy converges
  - Lyapunov regularization: energy must decrease across inner passes

OUTPUT: Claimed targets (intermediate + final answers)
```

### Controller Architecture (~400M)

```
CONTROLLER COMPONENTS:

State Encoder (~160M):
  - Cross-attention Perceiver reading Llama's hidden states
  - 8 attention heads, latent dim 1024
  - 4 Perceiver layers
  - LayerNorm + GELU activations
  - Input: Llama hidden states (2048d × seq_len)
  - Output: compressed state vector (1024d)

Tree Notebook Attention (~100M):
  - Hierarchical attention over tree nodes
  - Ancestor attention: attend to path from current node to root
  - Sibling attention: attend to nodes sharing same parent
  - Child attention: attend to children (for merge nodes)
  - Routing via hyperbolic distance between branch embeddings
  - 6 attention layers, 8 heads
  - Input: current state (1024d) + notebook pages
  - Output: context-enriched state (1024d)

Trunk (~80M):
  - 4-layer MLP with LayerNorm + GELU
  - Integrates state encoder output + notebook context
  - Input: concatenated state + context (2048d)
  - Output: trunk representation (2048d)

Decision Heads (~60M):
  - Scale head: MLP → 64d → 0.46 * tanh(x) [bounded Euclidean]
  - Branch embedding head: Linear → 64d → Poincaré projection [hyperbolic]
  - Branch action head: Linear → 3d → Gumbel-softmax [decompose/solve/merge]
  - Page head: MLP → 256d → L2 normalize [hypersphere]
  - Energy head: Linear → 1d → sigmoid [bounded scalar]
  - Confidence head: Linear → 1d → sigmoid [bounded scalar]

Positional Embeddings:
  - cycle_embed: which outer loop cycle (learned, 16 dims)
  - pass_embed: which inner loop pass (learned, 16 dims)
  - depth_embed: tree depth of current node (learned, 16 dims)
  - prev_scale_project: Linear(64, 64) for previous scales
```

### Memory Budget

```
Component                    Memory (fp16)   Trainable
──────────────────────────────────────────────────────
Llama 1.2B (frozen + baked)  ~2.4 GB         No
L2 atoms (82M)               ~164 MB         Yes
Controller (400M)            ~800 MB         Yes
Optimizer states             ~1.9 GB         —
Activations/gradients        ~2-4 GB         —
──────────────────────────────────────────────────────
Total                        ~7-9 GB         (fits A10G 24GB)
```

---

## Gradient Architecture

### The Cardinal Rule: Controller Gradient Never Flows Through Llama

```
WRONG (v1 — gradient dies):
  gen_loss → Llama (1.2B frozen) → L2 LoRA → scales → controller
  Result: controller gradient = 0.0000

RIGHT (v2 — direct path):
  gen_loss → scales → straight-through estimator → controller
  Result: controller gradient = full strength
```

### Straight-Through Estimator (Proper Implementation)

```python
# SEPARATE backward passes — never combine into one loss

# Step 1: Llama forward with controller's scales
scales = controller(hidden_states, notebook, cycle, pass)
llama_output = llama_forward(input_ids, l2_atoms, scales)
gen_loss = compute_gen_loss(llama_output, targets)

# Step 2: Atoms get gradient from gen_loss (direct through LoRA)
gen_loss.backward(retain_graph=True)
# atoms2 gradient: ~0.3 (healthy, direct path)

# Step 3: Controller gets gradient via straight-through
scale_grad = torch.autograd.grad(
    gen_loss, scales, retain_graph=False
)[0]
# scale_grad shape: (batch, 64) — per-sample, per-atom direction

# Normalize per sample to unit direction (consistent magnitude)
scale_grad_normalized = scale_grad / (scale_grad.norm(dim=-1, keepdim=True) + 1e-8)

# Direct controller loss — separate backward
st_loss = (scales * scale_grad_normalized.detach()).sum()
st_loss.backward()
# controller gradient: ~0.1-1.0 (healthy, direct path)

# Step 4: Separate optimizer steps
atom_optimizer.step()
controller_optimizer.step()
```

### Energy Head Training (Contrastive)

```python
# After generation, we know if the answer was correct
energy = controller.energy_head(trunk_output)

if answer_matches_target:
    energy_loss = energy                          # push energy DOWN
else:
    energy_loss = F.relu(margin - energy)         # push energy UP

energy_loss.backward()  # direct gradient to controller
```

### Gradient Flow Summary

```
Component        Loss Source              Path                  Expected Magnitude
─────────────────────────────────────────────────────────────────────────────────
L2 atoms         gen_loss                 Direct through LoRA   0.1-1.0
Controller       ST estimator             Direct, bypasses LLM  0.1-1.0
Controller       energy contrastive       Direct                0.01-0.1
Controller       branch contrastive       Direct                0.01-0.1
Controller       Lyapunov regularization  Direct                0.001-0.01
Llama            (none)                   Frozen                0.0
L1 atoms         (none)                   Baked into Llama      0.0
```

---

## Tree-Structured Breathing

### Execution Flow

```
STEP 1: COMPREHEND
  Llama reads problem (vanilla, no L2 atoms)
  Controller reads hidden states → initial tree plan

STEP 2: BUILD TREE (recursive)
  For each node:
    Controller decides action via Gumbel-softmax:

    DECOMPOSE:
      - Create child nodes with branch embeddings pushed outward in Poincaré ball
      - Recurse into children
      - No generation at this node

    SOLVE (leaf node):
      Inner loop (adaptive, Lyapunov stopping):
        for k in range(max_K):
          controller refines L2 scales (reads hidden states, notebook)
          controller estimates energy
          if energy < threshold or energy not decreasing: break
      Llama generates with final L2 scales → "equation #### answer"
      Extract answer → claim target (consumed once)
      Write page to notebook (256d hypersphere)
      Record branch embedding (64d Poincaré ball)

    MERGE (internal node):
      Attend to child pages via tree attention
      Inner loop to select L2 scales for combination step
      Llama generates combination equation → claim final target
      Write page recording merged result

STEP 3: EXTRACT
  Collect claimed targets from all SOLVE and MERGE nodes
  Final answer = MERGE root's claimed target (or last claimed)
```

### Tree Constraints

```
Max depth:       4 levels
Max total nodes: 8 nodes
Max fan-out:     3 children per DECOMPOSE
Min tree:        1 SOLVE node (degenerate chain for simple problems)
```

### Equal-Reward with Tree Structure

The 1/N reward per target still applies. For a problem with N targets:
- Each SOLVE node that claims a correct target earns 1/N reward
- MERGE nodes that claim correct combination targets earn 1/N reward
- Targets are consumed once (no double-claiming)
- The tree structure provides a natural ordering: leaves claim intermediates, merges claim combinations

```python
# Reward assignment
reward_per_target = 1.0 / num_targets

for node in tree.nodes:
    if node.claimed_target in remaining_targets:
        node.reward = reward_per_target      # correct match
        remaining_targets.remove(node.claimed_target)
    elif node.generation_is_self_consistent:
        node.reward = 0.5 * reward_per_target  # plausible but wrong
    else:
        node.reward = 0.3 * reward_per_target  # wrong

    # Weight gen_loss by reward
    node.weighted_gen_loss = node.reward * node.gen_loss
```

---

## Geometric Spaces

### Branch Embeddings: Poincaré Ball (64d)

```python
class PoincareBranchEmbedding(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=64, max_norm=0.95):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.max_norm = max_norm

    def forward(self, x):
        embed = self.proj(x)
        norm = embed.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        scale = (self.max_norm / norm).clamp(max=1.0)
        return embed * scale

def poincare_distance(u, v, eps=1e-6):
    diff_sq = (u - v).pow(2).sum(-1)
    u_sq = u.pow(2).sum(-1).clamp(max=1 - eps)
    v_sq = v.pow(2).sum(-1).clamp(max=1 - eps)
    return torch.acosh(1 + 2 * diff_sq / ((1 - u_sq) * (1 - v_sq) + eps))
```

**Tree structure in Poincaré ball:**
- Root: near origin (radius ~0.1) — general, parent of everything
- DECOMPOSE: children pushed outward (radius increases with depth)
- Siblings: angularly separated at similar radius
- MERGE: result moves inward (combining specifics into general)
- Tree depth ↔ radial distance — natural, distortion-free

**Training:** Use Riemannian SGD (multiply Euclidean gradient by `((1-||x||²)²)/4`) for branch embedding parameters only. All other parameters use standard Adam.

### Atom Scales: Bounded Euclidean (64d)

```python
scales = 0.46 * torch.tanh(raw_scales)  # independent per atom, no coupling
```

No normalization, no hypersphere projection. Each atom is independently controllable.

### Pages: Hypersphere (256d)

```python
page = F.normalize(raw_page, dim=-1)  # unit norm, collapse prevention
```

High capacity (256d sphere surface), natural orthogonality between random pages.

### Tree Attention: Hyperbolic Distance Routing

```python
def tree_attention(query_embed, notebook, temperature=1.0):
    distances = poincare_distance(
        query_embed.unsqueeze(1),
        torch.stack([n.branch_embed for n in notebook]).unsqueeze(0)
    )
    attn = F.softmax(-distances / temperature, dim=-1)
    pages = torch.stack([n.page for n in notebook])
    return (attn.unsqueeze(-1) * pages).sum(dim=1)
```

---

## Energy-Based Inner Loop with Adaptive Stopping

### Energy Head

```python
energy = torch.sigmoid(controller.energy_head(trunk_output))
# 0.0 = maximally confident (low energy)
# 1.0 = maximally uncertain (high energy)
```

### Contrastive Energy Training

```python
if answer_correct:
    energy_loss = energy                        # push toward 0
else:
    energy_loss = F.relu(0.7 - energy)          # push toward 1 (margin=0.7)
```

### Adaptive Inner Loop

```python
def adaptive_inner_loop(controller, hidden_states, notebook, node,
                         max_K=6, energy_threshold=0.15,
                         convergence_threshold=0.01):
    scales = controller.initial_scales(hidden_states)
    prev_energy = 1.0
    energies = []

    for k in range(max_K):
        scales, page, energy = controller(
            hidden_states, scales, notebook, node, pass_num=k
        )
        energies.append(energy)

        if energy < energy_threshold:
            break  # confident enough
        if k > 0 and (prev_energy - energy) < convergence_threshold:
            break  # converged (Lyapunov: V̇ ≈ 0)

        prev_energy = energy

    return scales, page, energy, k, energies
```

### Lyapunov Regularization

```python
# Penalize energy increases across inner passes (V̇ should be < 0)
lyapunov_loss = 0
for k in range(1, len(energies)):
    increase = F.relu(energies[k] - energies[k-1])
    lyapunov_loss += increase

total_loss += 0.01 * lyapunov_loss
```

---

## Data Pipeline

### Procedural Curriculum (Keep from v1)

```
L3:    1-step arithmetic (100% baseline)
L4:    2-step arithmetic (99.5% proven)
L4.5:  3-step arithmetic (99.5% proven)
L5+:   4-5 step arithmetic (new, for tree validation)
```

### Tree-Structured Curriculum (New)

```
T2:    2-branch problems
       "A has X. B has Y. Total?"
       Tree: [DECOMPOSE → [SOLVE(A), SOLVE(B)] → MERGE(A+B)]

T3:    3-branch problems
       "A, B, and C each do X. How many total?"
       Tree: [DECOMPOSE → [SOLVE(A), SOLVE(B), SOLVE(C)] → MERGE]

TN:    Nested problems
       "A does X. B does Y based on A. C combines A and B."
       Tree: [DECOMPOSE → [SOLVE(A), chain(A→SOLVE(B))] → MERGE(A,B→C)]
```

### GSM8K (Keep, with improvements)

```
- Number augmentation: scale 0.8-1.2 (gentle, lets patterns settle)
- 3x weight on number tokens (gradient targets numbers over prose)
- Cycle-level reward: correct=1.0, self-consistent=0.5, wrong=0.3
- Eval: 441 samples (full eval set, low noise)
- Step annotations via Claude API (existing annotate_gsm8k_cycles.py)
```

### MATH-500 (Target)

```
- Diverse problem types: algebra, geometry, number theory, etc.
- Much harder than GSM8K — requires deep decomposition
- Tree structure is essential: many problems have independent subcomputations
- Goal: meaningful accuracy (>15%) by September 1
```

---

## Training Pipeline

### Phase 0: Controller Smoke Test (Half Day)

**Goal:** Verify the controller can produce different outputs for different inputs. No Llama, no atoms.

```python
# Toy environment
# 10 different input vectors → 10 different target scale vectors
# Controller must learn to map each input to its target

inputs = torch.randn(10, 2048)       # 10 different "hidden states"
targets = torch.randn(10, 64)        # 10 different target scales
targets = 0.46 * torch.tanh(targets) # bounded

controller = BreathingController(...)

for step in range(1000):
    for i in range(10):
        scales = controller(inputs[i])
        loss = F.mse_loss(scales, targets[i])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# VERIFY:
# 1. Loss → 0 (controller learned the mapping)
# 2. Different inputs → different outputs (cosine sim < 0.9)
# 3. Gradients flow to all controller components
# 4. No dimension collapse (all 64 dims active)
```

**Do not proceed to Phase 1 until this passes.** This catches:
- Constant-function collapse
- Dead gradient paths
- Tanh saturation issues
- Dimension death

### Phase 1: Linear Breathing on Curriculum (2-3 Days)

**Goal:** Breathing loop with new controller on procedural math. Tree structure exists but all problems use degenerate chain (SOLVE → SOLVE → SOLVE).

**Training sequence:**
1. L3 (1-step) — verify basic atom activation via controller
2. L4 (2-step) — verify 2-cycle decomposition with equal-reward
3. L4.5 (3-step) — verify 3-cycle decomposition, match v1's 99.5%

**Success criteria:**
- `scale_xproblem_cos < 0.9` (controller differentiates between problems)
- `scale_mid_frac > 0.3` (scales not saturated)
- L4.5 accuracy ≥ 95% within 5 epochs
- Controller gradient norm in same order of magnitude as atoms2
- Energy correlates with correctness (low for correct, high for wrong)

**Key diagnostics:**
- Per-problem scale vectors (should differ between problems)
- Inner loop energy trajectory (should decrease per pass)
- Controller gradient norms (should be 0.01-1.0)
- Page diversity (active dims should be >50/64)

### Phase 2: Tree Structure on Curriculum (1-2 Weeks)

**Goal:** Enable tree decisions. Train on T2/T3/TN curriculum where tree structure is natural.

**Training sequence:**
1. T2 (2-branch) — verify DECOMPOSE → 2×SOLVE → MERGE
2. T3 (3-branch) — verify 3-way branching
3. TN (nested) — verify chain+branch combinations
4. Mixed curriculum: L4.5 + T2 + T3 + TN

**Success criteria:**
- Multi-entity problems show branching (not forced into chain)
- Branch embeddings in Poincaré ball: leaves at larger radius than root
- Merge nodes attend primarily to their children (not random pages)
- Tree accuracy ≥ linear chain accuracy on branching problems
- Linear problems still work (tree degenerates to chain)

**Key diagnostics:**
- Branch action distribution (what fraction DECOMPOSE vs SOLVE vs MERGE)
- Poincaré ball visualization (project to 2D, verify tree structure)
- Tree depth distribution per problem type
- Merge node attention weights (should focus on children)

### Phase 3: GSM8K with Trees (2-4 Weeks)

**Goal:** GSM8K accuracy exceeding v1's 22% with genuine per-problem adaptation.

**Training sequence:**
1. Transfer from Phase 2 checkpoint
2. Train on GSM8K with tree structure enabled
3. Monitor which problems branch vs chain

**Success criteria:**
- GSM8K accuracy > 25% (beats v1 baseline)
- Multi-entity problems (Sally + Jolly) show branching behavior
- Controller produces measurably different scales for different problems
- Energy-based stopping reduces average inner loop iterations vs fixed K

**Target sub-milestones:**
- Week 1: 15%+ (controller alive, basic decomposition)
- Week 2: 20%+ (tree structure helping on multi-entity)
- Week 3: 25%+ (matching/exceeding v1)
- Week 4: 30%+ (controller steering provides clear advantage)

### Phase 4: MATH-500 (4-8 Weeks)

**Goal:** Meaningful accuracy on MATH-500.

**Training sequence:**
1. Annotate MATH-500 problems with step decompositions (Claude API)
2. Curriculum: GSM8K → MATH-500 easy → MATH-500 medium → MATH-500 hard
3. Tree depth may need to increase (max_depth 4 → 6 for harder problems)
4. If memory becomes bottleneck: implement adjoint sensitivity method

**Success criteria:**
- MATH-500 accuracy > 15%
- Tree structures emerge that match problem complexity
- Adaptive inner loop averages K=2 for easy, K=5+ for hard
- Clear accuracy improvement from tree structure vs chain baseline

---

## Diagnostics Suite

### Controller Health (Run Every Epoch)

| Metric | Healthy Range | What It Means |
|--------|--------------|---------------|
| `scale_xproblem_cos` | < 0.9 | Controller differentiates between problems |
| `scale_mid_frac` | > 0.3 | Scales not saturated in tanh |
| `dead_dims` | < 10/64 | Most scale dimensions are active |
| `controller_grad_norm` | 0.01-1.0 | Controller is learning |
| `atoms2_grad_norm` | 0.1-1.0 | Atoms are learning |
| `energy_correct_mean` | < 0.3 | Energy head identifies correct answers |
| `energy_wrong_mean` | > 0.7 | Energy head identifies wrong answers |
| `page_xproblem_cos` | < 0.8 | Pages differ between problems |
| `inner_loop_avg_K` | 1.5-4.0 | Adaptive stopping is working |

### Tree Health (Run Every Epoch, Phase 2+)

| Metric | Healthy Range | What It Means |
|--------|--------------|---------------|
| `branch_action_entropy` | > 0.5 | Not collapsed to single action |
| `decompose_fraction` | 0.1-0.5 | Some but not all problems branch |
| `avg_tree_depth` | 1.5-3.0 | Trees are non-trivial |
| `merge_child_attn_frac` | > 0.6 | Merge nodes attend to their children |
| `poincare_radius_leaves` | > 0.5 | Leaves pushed outward |
| `poincare_radius_root` | < 0.3 | Root stays near origin |

### Existing Diagnostics (Keep from v1)

| Script | Purpose |
|--------|---------|
| `diag_debug_cycles.py` | Full chain per problem: scales, pages, generation |
| `diag_deep_single_step.py` | Failure categories: wrong_arithmetic, copying, format |
| `diag_atom_inspect.py` | What each atom does to attention |
| `diag_laplace.py` | Page trajectory: convergence, oscillation |

---

## Failed Experiments — Do Not Repeat

These are validated failures from v1. Do not re-attempt.

| What | Result | Lesson |
|------|--------|--------|
| Answer head (5 versions) | 4% peak | Generation IS the output |
| Separate perceiver + hypernetwork | Constant function | Unified controller reads hidden states |
| Atoms at [-3, 3] volume | 14.6% | Too loud, corrupts arithmetic. Use [-0.5, 0.5] |
| Q,K-only atoms | 0.4% | WORSE than vanilla. V,O does the work |
| Per-layer dance (loud early, quiet late) | 10.7% | Uniform scale beats per-layer schemes |
| Cycle multiplier (2x correct, 0.2x wrong) | 11.5% drop | Too aggressive at low accuracy |
| Split generation (atoms off for arithmetic) | 15.3% drop | Disrupts generation flow |
| No decomposition incentive | Cycle 2=6%, final=0% | Without 1/N reward, model one-shots |
| Full LR on controller | Destabilizes | Controller needs conservative LR |
| Routing controller gradient through Llama | 500x attenuation | Use straight-through estimator |
| clamp + tanh on scales | Dead gradient | Use 0.46 * tanh(x) only |
| Loading collapsed controller weights | Saturated scales | Always reinit scale head with small weights |
| Unfreezing L1 during GSM8K training | Accuracy peaks epoch 2-3, declines | L4.5 math patterns drift. Bake L1 permanently |
| Aggressive number augmentation (0.5-2.0x) | Prevents settling | Use gentle 0.8-1.2x |

---

## Key Discoveries — Carry Forward

These are validated insights that inform the v2 design.

| Discovery | Evidence | Implication |
|-----------|----------|-------------|
| L4.5 atoms are template-invariant | Same atoms at max clamp for all problems, different per cycle | L1 is a universal math mode — bake it permanently |
| V,O projections do the heavy lifting | 1.9%→13.8% V,O only; +6.5pp adding Q,K | Atom design focuses on V,O |
| Equal-reward is essential for decomposition | Without 1/N: cycle 2=6%. With 1/N: cycle 2=99.5% | Never remove equal-reward structure |
| Operation type is NOT classifiable | IB analysis across all feature spaces | Don't try to classify operations — let attention handle it |
| Discovered decomposition beats designed | 7-worker IB pipeline (46%) vs designed gradual descent | Let the model find the decomposition |
| Teaching via auxiliary heads beats hint concatenation | F1 0.741 vs 0.727 ceiling | Hints in input are redundant with backbone knowledge |
| 78.6% of early labels were wrong | Regex vs IB clustering mismatch | Always validate training labels thoroughly |
| DeepMind confirms recurrence is necessary | "Topological Trouble with Transformers" (2604.17121) | External validation of breathing loop thesis |
| Gen loss ≠ accuracy | Gen loss drops while accuracy declines | Language tokens dominate; weight number tokens 3x |

---

## File Structure (v2)

```
mycelium-v2/
├── CLAUDE.md                          # This document (condensed)
├── scripts/
│   ├── controller.py                  # BreathingController (~400M)
│   │   ├── StateEncoder               # Perceiver reading Llama hidden states
│   │   ├── TreeNotebookAttention       # Hierarchical attention with Poincaré routing
│   │   ├── Trunk                       # Integration MLP
│   │   ├── ScaleHead                   # → 64d bounded Euclidean
│   │   ├── BranchEmbeddingHead         # → 64d Poincaré ball
│   │   ├── BranchActionHead            # → 3d Gumbel-softmax
│   │   ├── PageHead                    # → 256d hypersphere
│   │   ├── EnergyHead                  # → 1d sigmoid
│   │   └── ConfidenceHead              # → 1d sigmoid
│   ├── atom_lora.py                    # L2 atoms + AtomAdditiveLoRAManager
│   ├── tree_notebook.py                # TreeNode, TreeNotebook, tree operations
│   ├── poincare.py                     # Poincaré ball ops, distances, RSGD
│   ├── train.py                        # Training loop with tree execution
│   ├── losses.py                       # gen_loss, ST estimator, energy contrastive,
│   │                                   # branch contrastive, Lyapunov regularization
│   ├── data_gen.py                     # L3-L5 + T2/T3/TN procedural generation
│   ├── gsm8k_prep.py                   # GSM8K parsing + annotation
│   ├── eval.py                         # Evaluation with tree execution
│   └── diag/
│       ├── controller_health.py        # scale_xproblem_cos, scale_mid_frac, dead_dims
│       ├── tree_health.py              # branch actions, Poincaré radii, merge attention
│       ├── energy_calibration.py       # energy vs correctness correlation
│       ├── gradient_flow.py            # per-component gradient norms
│       ├── debug_cycles.py             # full chain per problem (updated for trees)
│       └── smoke_test.py              # Phase 0 controller verification
├── plan/
│   ├── master_rebuild_handoff.md       # This document
│   ├── tree_structured_breathing.md    # Tree architecture design
│   ├── geometric_spaces.md             # Manifold choices per component
│   ├── energy_inner_loop.md            # Energy-based adaptive stopping
│   ├── straight_through_gradient.md    # Controller gradient fix
│   └── two_layer_lora.md              # L1/L2 layer separation
├── data/
│   ├── procedural/                     # L3-L5 + T2/T3/TN JSONL
│   ├── gsm8k/                          # Annotated GSM8K
│   └── math500/                        # Annotated MATH-500
├── checkpoints/
│   ├── baked_llama/                    # Llama 1B + baked L1 (frozen, reusable)
│   └── training/                       # Controller + L2 checkpoints
└── logs/
    └── *.log                           # Training logs with diagnostics
```

---

## Baking L1 Into Llama (One-Time Setup)

```python
# Run once, save the result, never touch L1 again

from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
l1_checkpoint = torch.load("checkpoints/l45_best_atoms.pt")

# For each LoRA target module (V, O, Q, K projections):
for name, param in model.named_parameters():
    if is_lora_target(name):
        # LoRA: W' = W + scale * A @ B
        A = l1_checkpoint[f"{name}.lora_A"]
        B = l1_checkpoint[f"{name}.lora_B"]
        scale = 0.46  # proven optimal atom volume
        param.data += scale * (A @ B)

model.save_pretrained("checkpoints/baked_llama/")
# This IS the base model now. No L1 at runtime.
```

---

## Timeline

```
May 2026:
  Week 1:  Phase 0 (smoke test) + Phase 1 start (clean reimplementation)
  Week 2:  Phase 1 complete (L4.5 at 95%+, controller alive)
  Week 3-4: Phase 2 (tree structure on curriculum)

June 2026:
  Week 1-2: Phase 2 complete (tree decisions validated)
  Week 3-4: Phase 3 start (GSM8K with trees)

July 2026:
  Week 1-2: Phase 3 push (GSM8K > 25%)
  Week 3-4: Phase 3 → Phase 4 transition (MATH-500 annotation + curriculum)

August 2026:
  Week 1-4: Phase 4 (MATH-500 training + evaluation)

September 1: Deadline
```

---

## Summary

Mycelium v1 proved the thesis: a frozen LLM can gain reasoning through an external breathing loop. v2 rebuilds around the lesson that the controller's gradient path is everything. Direct gradient via straight-through estimation. Tree structure from day one. Hyperbolic geometry for hierarchical representations. Energy-based adaptive stopping. And most importantly: verify the controller works before building anything on top of it.

The architecture is validated by theory (DeepMind's topological analysis, LeCun's EBM framework), by empirical results (equal-reward decomposition, baked L1, template-invariant math mode), and by failure analysis (every failed experiment in v1 points toward this design). Four months is enough to build it right.
