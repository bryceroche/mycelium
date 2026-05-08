# Mycelium v4: The Breathing Transformer
## Conceptual Architecture

**Author:** Bryce + Claude
**Date:** May 1, 2026
**Deadline:** September 1, 2026
**Platform:** Shadow Glass (AMD 7900 XTX, 24GB) · tinygrad + AM driver · no AMD/ROCm software
**Target:** MATH-500

---

## 1. Vision

A small transformer that loops its own layers, breathing through each problem until it finds the resonant decomposition. Not a frozen LLM with external scaffolding — a model that breathes natively. Trained from scratch.

Each loop is one expand-collapse cycle shaped as a sine wave across 4 specialized layers. Twelve attention heads, each tuned to a different phase angle, scan the problem's spectral content in parallel — identifying which mathematical operations are present simultaneously, like BirdNET identifying multiple overlapping bird species from a single audio clip.

A controller between breaths reads the accumulated understanding, matches it against a library of prime operations, and decides when the analysis is complete. The tree structure of the solution emerges directly from the prime factorization — no separate decomposition classifier needed.

**Three mathematical operations unify the architecture:**

**Oscillation** — sine-wave breath within each loop. Four layers cycle between expansion (broad attention, open to new information) and compression (sharp focus, distilling the essential signal). Temperature modulates the expand-collapse rhythm continuously.

**Alternation** — π-cycled attention across loops. Each loop rotates the viewing angle so the model sees the problem from a different geometric perspective. Twelve heads per breath scan different frequency bands in parallel. Structural diversity that gradient descent cannot erase.

**Integration** — accumulated signal across breaths. A running integral combines the observations from all previous breaths, weighted by novelty. The integral converges when all significant modes have been captured. Lyapunov stopping criterion: breathe until the integral stabilizes.

---

## 2. The Sine-Wave Breath

### Why Specialized Layers

An expand layer and a collapse layer do fundamentally different things. Expansion requires broad attention patterns and FFN weights that increase representational diversity. Compression requires sharp, focused attention and FFN weights that extract essential features. The same weights cannot excel at both. Asking them to is how you get mediocrity.

We discovered this empirically: Llama's own layer diversity traces a wave — diverse early layers, compressed middle layers, re-diversified late layers. The expand-collapse rhythm exists inside transformers naturally. Mycelium makes it explicit and structural.

### The Four Phases

Each of the 4 layers sits at a specific phase of a sine wave. The sine value at each phase continuously controls the layer's behavior.

**RISE (phase = 0).** The breath begins. Transitioning from the previous loop's compressed output toward openness. Moderate attention breadth, balanced between preserving prior understanding and accepting new information.

**PEAK (phase = π/2).** Maximum expansion. Broadest attention, strongest FFN processing, weakest residual connection — maximally open to new information. The model casts the widest net.

**FALL (phase = π).** Compression begins. Attention narrows. The model starts filtering noise from signal, deciding what matters.

**TROUGH (phase = 3π/2).** Maximum compression. Sharpest attention, gentlest processing, strongest residual — protecting the distilled essence. The output is a tight representation carrying only the most important information, ready for the next loop.

### Partial Weight Sharing

The Q and K projections must be unique per layer — they control attention geometry, which is fundamentally different for expand vs collapse. The FFN gate must be unique — it controls what gets amplified vs suppressed. But the V projection, FFN transformation basis, and normalization layers can be shared across all 4 layers. Specialized where it matters, efficient where it doesn't. Roughly 2× the parameters of full weight tying, 54% of full separation.

### Temperature as the Expand-Collapse Lever

Temperature modulates the sharpness of the attention distribution. High temperature at the PEAK produces broad, high-entropy attention — many tokens contribute equally. Low temperature at the TROUGH produces sharp, low-entropy attention — few tokens dominate. The sine wave modulates temperature continuously, controlling the information-theoretic bandwidth of each layer. This is the natural and correct mechanism.

---

## 3. π-Cycled Attention

### The Phase Rotation

Standard rotary position encoding (RoPE) encodes WHERE a token is in the sequence. π-cycled RoPE adds a second dimension: WHICH LOOP you're in. A phase shift of loop × π / max_loops rotates the attention geometry at each loop, making each pass attend to different token relationships.

At phase 0, certain token pairs have high attention. At phase π/2, a maximally different set of pairs dominates. Each loop sees the problem from a genuinely different geometric angle. This is structural — encoded in the position formula, not in learned weights. The gradient cannot erase it.

### Per-Head Spectral Decomposition

Inspired by BirdNET (Cornell Lab of Ornithology), which identifies multiple overlapping bird species from a single audio spectrogram by scanning different frequency bands simultaneously, our 12 attention heads each operate at a different phase offset.

Head 0 at phase 0. Head 1 at phase π/12. Head 2 at 2π/12. Through head 11 at 11π/12. Each head is tuned to a different frequency band of the problem's spectral content. In a single breath, all 12 heads scan the full range in parallel.

BirdNET doesn't separate overlapping bird songs into individual streams. It classifies them simultaneously with independent confidence scores — robin AND cardinal AND wren, all detected from one clip. Our heads work the same way: head 3 responds strongly to subtraction patterns while head 7 responds to entity structure, all in one breath. The combined multi-head response IS the spectral decomposition.

Each loop, all heads rotate together by π/max_loops while maintaining their relative offsets. The bank of heads shifts as a unit, scanning new regions with each breath. Twelve heads × 8 loops = 96 distinct phase angles sampled.

### Why This Can't Collapse

The phase rotation is geometric, not learned. Every previous diversity mechanism in v1-v3 was learned and therefore erasable by gradient descent — scales, soft tokens, codebooks, fingerprints, diversity losses all collapsed to constant within one epoch. π cycling is permanent. Different breaths WILL see different features because the mathematics of the rotation guarantees it.

---

## 4. Integration Across Breaths

### The Running Integral

Without integration, the breathing loop is Markov — each loop only sees the previous loop's output. Early insights get overwritten. Integration fixes this: a running sum accumulates every breath's contribution, weighted by an integration gate that the controller learns.

Breaths that discover something new contribute strongly (high gate). Breaths that confirm known information contribute weakly (low gate). The model works from the normalized integral (running mean of all gated outputs), carrying the accumulated understanding from all previous breaths.

### Bayesian Evidence Combination

Each breath is an observation from a different angle (π cycling). The integration combines independent observations into a posterior belief — Gauss's least-squares estimation applied to breath observations. The π cycling ensures observations are independent (different angles = linearly independent evidence), which is maximally informative.

### Convergence as Stopping Criterion

When the integral stops changing between loops, all significant evidence has been accumulated. This is the Lyapunov criterion: V = magnitude of integral change, V̇ < 0 means converging. Simple problems stabilize in 1-2 breaths. Hard problems need 4-8.

---

## 5. The Controller

### Role: Conductor of the Breathing

The controller sits between loops. It doesn't play the instruments — the 4 transformer layers do that. The controller conducts: controlling the tempo, dynamics, and duration of the breathing process. It owns all the meta-parameters that govern how the model thinks.

### Thinking Space

The controller thinks in 512 dimensions. All observation, memory, decision-making, and pages exist at this dimensionality. No projections between thinking and memory — they share the same space.

### What the Controller Governs

**Temperature.** How broad or sharp each breath's attention should be. The default sine-wave modulation provides the baseline expand-collapse rhythm, but the controller can shift the entire curve — warming it for coarse exploration, cooling it for fine-grained analysis. A small scalar adjustment per breath.

**Phase angle.** Which direction to look next. The default is uniform spacing, but the controller can target specific angles based on what the factorization suggests. If the lookup table indicates subtraction is likely, probe at subtraction's known resonant angle.

**Loop count.** When to stop breathing. The controller monitors integral convergence (understanding has stabilized) and spectral completeness (residual has no strong prime matches). Both must be satisfied.

**Integration gate.** How much each breath contributes to the running integral. Novel observations get high gate values. Redundant observations get low gate values.

**Resolution.** The coarse-to-fine progression. Early breaths at warm temperature scan broadly. Later breaths at cool temperature probe precisely. The controller can accelerate or slow this based on problem complexity.

The controller is learning to be a good scientist: design the experiment (choose temperature, angle, and resolution), run it (let the layers process), observe the result (read the integral), decide if more data is needed (continue or stop), and commit to a conclusion (output the factorization and build the tree).

### What the Controller Produces

**Pages** — 512d vectors on the hypersphere recording what this breath understood.

**Factorization** — prime weights from querying the lookup table. Composite = decompose. Single prime = solve. Coupling matrix = tree shape.

**Breathing parameters** — temperature modulation, phase angle, integration gate, continue/stop signal for the next breath.

### Gradient Separation

The controller's gradient NEVER flows through the transformer. Three days of evidence in v1-v3 proved that any such path collapses to one basin. The controller learns through REINFORCE on outcomes, supervised energy calibration, contrastive page diversity, and structure preservation targeting early-breath hidden states. Complete separation.

---

## 6. Coarse-to-Fine Spectral Analysis

### The Resolution Progression

The first breath shouldn't try to identify all primes precisely. It should produce a coarse estimate. Each subsequent breath refines. This is how spectral analysis works — a short observation window gives coarse frequency resolution, a longer observation gives finer resolution. Each breath adds samples, increasing the spectral precision.

**Breath 1 (coarse).** "What kind of problem is this?" The controller sets warm temperature — broad attention. The 12 heads scan across phase angles. The lookup table returns a rough factorization — maybe 3 entries active with moderate confidence. Low resolution, high recall.

**Breath 2 (medium).** "How many of each operation?" Same layers, rotated phase, slightly cooler. Higher spectral resolution from integrating two independent observations. The factorization sharpens — "two subtractions, not one."

**Breath 3 (fine).** "What are the dependencies?" Same layers, further rotated, cooler still. Enough independent samples to resolve the coupling structure.

**Breath 4+ (verification).** "Is the factorization complete?" The residual after subtracting confirmed primes is examined. Noise = done. Structure remaining = breathe again.

### Why This Makes the Layers Reusable

Each breath doesn't ask the layers to do something fundamentally different. The layers always expand, process, compress, and integrate. What changes is the resolution of the input and the temperature of the attention. The layers are a general-purpose expand-compress pipeline that works at any resolution — like a microscope with the same optics at every magnification.

### The Sine Wave Is Periodic

A sine wave from 0 to 2π starts and ends at the same height. The output of one breath is at exactly the right representational altitude for the input of the next breath. No discontinuity between loops. The looping is smooth because the sine wave IS smooth and periodic. This is by design.

### The Inward Spiral

The triple helix spirals inward across breaths. Alternation rotates the viewing angle. Oscillation provides the expand-collapse rhythm. Integration accumulates evidence. Temperature cools progressively under the controller's direction. Each breath simultaneously rotates AND increases resolution. The spiral converges on the precise factorization — coarse to fine, broad to sharp, hypothesis to confirmation.

---

## 6. The Differentiable Lookup Table

### Primes, Multiplicities, and Couplings

The lookup table stores the fundamental operations of mathematics as orthogonal vectors — a prime basis. For elementary math: addition, subtraction, multiplication, division, fraction-of, comparison, combination, sequential dependency. Start with 8-12 entries. Grow as the model encounters harder math.

When the controller matches its current page against the table, the match weights decompose the problem into its prime factors. But operations can appear multiple times — "subtract 27, then add 83, then subtract 218" contains subtraction twice. The factorization must capture multiplicities, like 12 = 2² × 3.

### Parallel Detection, Iterative Refinement

Breath 1 leverages all 12 per-head phase angles to detect ALL candidate primes simultaneously — multi-label detection, not single-label classification. Like BirdNET identifying three bird species from one clip, all operations are identified in parallel with independent confidence scores.

Subsequent breaths confirm individual primes and count multiplicities through iterative spectral subtraction: confirm a prime, subtract its contribution from the signal, examine the residual. If the same prime still matches the residual, it appears again — ring and subtract until that prime is exhausted. Repeat for remaining primes until the residual is noise.

### What Each Entry Stores

**Pattern.** What this operation looks like in the transformer's representation space. Used for matching.

**Resonant angle.** Which π-cycling phase angle most clearly reveals this operation. Learned through training.

**Subtraction mask.** What to remove from the signal once this operation is confirmed. For computing the residual.

**Confidence threshold.** How much energy must drop to confirm this operation's presence.

### The Coupling Matrix

A small N×N table encoding relationships between co-occurring primes. When subtraction and multiplication appear together, are they independent (parallel — can be computed separately) or coupled (sequential — subtraction depends on multiplication's result)?

The prime counts determine the number of tree nodes. The coupling matrix determines the tree shape. Independent primes become parallel branches. Coupled primes become sequential chains. The combination of parallel results defines the merge points.

### From Factorization to Tree

The complete factorization — primes, multiplicities, couplings — fully determines the solution tree. The controller doesn't need to learn tree construction through trial and error. The spectral analysis provides it directly: identify the primes (what operations), count them (how many nodes), check their couplings (what shape), and the tree writes itself.

---

## 7. The Outer Cycle

### Two Levels of Breathing

Inner breathing (4 layers × N loops) handles understanding — spectral analysis of what operations the problem requires. Outer cycling handles execution — generating one step's answer, recording it, breathing again for the next step.

The first outer cycle breathes until the full factorization is established. Subsequent cycles execute the tree: each cycle takes the next node from the plan, breathes briefly to refine (the prime is already identified, 1-2 breaths suffice), generates the equation, claims the target, advances.

### Equal-Reward Decomposition

Each target earns 1/N reward when claimed. The only way to maximize total reward is to claim all N targets through genuine decomposition. The number of primes with multiplicity equals the number of targets. Each node claims one.

---

## 8. Training Data: Haiku-Curated Decompositions

A small model (Claude Haiku or equivalent) preprocesses GSM8K problems to create multi-pass training data. For each problem, the haiku model identifies the operations, decomposes them into steps, and produces annotated training examples showing what each breath SHOULD discover at each cycle.

This gives the breathing transformer supervised targets for the spectral analysis — "breath 1 should detect subtraction and entity-tracking, breath 2 should confirm subtraction with multiplicity 2." Without this, the model must discover the spectral decomposition purely through REINFORCE, which is slow. The curated data provides a curriculum for learning to breathe.

---

## 9. The Copy Machine Principle

### Why Representation Space, Not Token Space

The breathing transformer does NOT generate reasoning tokens between breaths. All thinking happens in representation space — 512d pages in the controller, 768d hidden states in the transformer. This is not just an efficiency choice — it is an architectural necessity, proven empirically.

### The Copy Machine Problem

Chain-of-thought reasoning forces the model to "photocopy" its understanding into tokens at every reasoning step. Each token is a lossy compression — the model's full 768-dimensional continuous understanding squeezed into a single discrete choice from 50,000 vocabulary items. Information is destroyed at each step. The next step builds on that destroyed information. It's a copy of a copy of a copy.

We observed this directly in validation: when Pythia's layers generate tokens and feed them back in (the autoregressive pipeline), output degrades to "had had had" within 2 loops. Each generated token introduces small errors that become the source material for the next token. The errors compound monotonically — they can never improve, only worsen.

### The Original, Not the Photocopy

The hidden states are the original painting. They carry the model's full continuous understanding — all 768 dimensions, all the nuance, all the per-problem diversity. Our validation proved this: centered cross-problem cosine stays at -0.05 (orthogonal) through 4 loops. Effective rank holds at 15-16. Signal norm actually GROWS across loops (3.9 → 6.4). The original doesn't degrade when you keep working with it.

The breathing transformer works from the original throughout. Each breath refines the hidden states directly — continuous, lossless, without ever compressing through the vocabulary bottleneck. The model only "prints" once: at the very end, when breathing has converged and the integrated representation is ready.

One high-quality copy from a refined original. Not a chain of copies from increasingly degraded intermediates.

### Why This Limits Chain-of-Thought

Chain-of-thought is inherently limited by the copy machine problem. Each reasoning step photocopies the model's understanding into tokens. Each photocopy degrades. The model must reconstruct its full understanding from the degraded copy before taking the next step. Deep reasoning requires many steps, and each step adds degradation.

The breathing transformer's reasoning depth is unlimited in principle — more breaths refine the original without degradation. The compute cost of breathing scales with problem difficulty (more loops through 4 layers), not with verbosity (more tokens consuming context window).

### Empirical Proof

Our looping experiments proved both sides. Hidden states after 4 loops: per-problem diversity preserved (centered_cos = -0.047), effective rank stable (15.2), signal growing (3.9 → 6.4). Generated text after 2 loops with autoregressive decoding between loops: "had had had had had" — complete collapse.

Breathing in representation space: the signal survives and grows. Generating between breaths: the copy machine destroys it. The architecture is designed around this empirical reality.

### Tree Notebook, DAG Execution

The controller's decomposition plan is a tree. The actual computational dependencies can be a DAG — step 3 depending on both step 1 and step 2. The tree with merge nodes handles this naturally.

### Working Backwards (Phase 3)

The controller should eventually learn to verify solutions by reasoning from the claimed answer backward through intermediate steps. Constraint propagation — checking consistency rather than computing from scratch. SymPy serves as a verification oracle outside the execution path.

### Gaussian Framework

Gauss's Fundamental Theorem of Arithmetic: unique prime factorization. The lookup entries must remain independent — regularization penalizes entries drifting toward each other. Gauss's least-squares estimation: the integration is a weighted average of independent observations. The convergence criterion is that residuals have become negligible.

---

## 10. Honest Unknowns

### Will the Looping Layers Work?

This is the biggest open question. Cycling the same 4 transformer layers multiple times with different phase angles is genuinely novel. The layers might only be useful once — the representation after one pass might not benefit from a second pass through the same weights, even with rotated attention.

**Reasons for concern:** Standard transformers use different weights per layer for a reason. The FFN computation is shared across phases and might dominate layer behavior regardless of attention rotation. The transition from TROUGH output back to RISE input between loops might be discontinuous.

**Reasons for optimism:** Deep Equilibrium Models (2019) proved that a single layer iterated to convergence matches deep transformers. Universal Transformers (2018) showed weight-tied layers with step embeddings work on algorithmic tasks. Block Recurrent Transformers (2022) validated recurrence within transformer blocks. Our setup is less extreme — 4 specialized layers with partial sharing, not 1 fully tied layer.

**The mitigation:** Start with forced loop counts on simple problems (L3). Measure whether accuracy increases with loops (more thinking helps) or decreases (representation degrades). If degradation, the looping architecture needs revision — possibly more layers per breath, or different normalization between loops.

### Will 12 Heads at Different Phases Provide Meaningful Spectral Coverage?

Twelve phase angles spanning [0, π) is 12 samples of the spectral space. This might be sufficient for elementary math (8-12 primes) or might be too sparse for complex problems with many overlapping operations. The BirdNET analogy is encouraging (they handle 2000+ species with CNN filters) but the domains are different enough that the analogy might not transfer.

### Will the Prime Factorization Remain Unique Under Training?

The orthogonal initialization guarantees initial uniqueness. But training pushes entries toward whatever reduces loss. If two operations are highly correlated in the training data (subtraction and sequential-dependency often co-occur), their entries might converge. The regularization term helps but might not be sufficient. Monitoring entry pairwise similarity is essential.

### Can a Small Model Learn Math?

Addressed by initializing from Pythia-160M. The pretrained layers already understand English — entity tracking, sentence structure, counterfactuals. The breathing fine-tuning adds mathematical reasoning on top of existing language competence. This eliminates the risk of spending weeks learning basic English parsing before math training can begin.

The remaining risk: Pythia's layers were trained for single-pass processing. They might resist being looped. But the phase-specific weights (Q, K, gate) are reinitialized for the breathing phases — only the shared weights (V, FFN, norms) carry pretrained assumptions, and those components are genuinely phase-independent.

---

## 11. Specifications

### Initialization: Pythia-160M

The breathing transformer initializes from the first 4 layers of EleutherAI's Pythia-160M — a 12-layer, 768-dim model trained on 300B tokens of diverse English text (the Pile). Taking 4 of 12 layers captures a substantial fraction of a complete model, not just the shallow preprocessing stages of a deep narrow network.

Pythia-160M was chosen over alternatives for several reasons. Its 12 standard multi-head attention heads (not grouped query attention) give each head full independence for per-head π-cycling phase offsets — essential for the spectral decomposition. Its hidden dimension of 768 with 12 heads yields head_dim=64, which is standard and well-studied. It uses RoPE which our π cycling extends naturally. It's designed for research, well-documented, and Apache 2.0 licensed.

What we take from Pythia: layers 0-3 (attention + FFN weights), token embeddings (768 × 50304), and the output head. What we modify: Q, K, and FFN gate projections become phase-specific (4 unique copies, initialized from Pythia's layers 0-3 respectively). V, O, FFN up/down, and norms remain shared (initialized from Pythia, averaged or from layer 0).

### Model Dimensions

Hidden dimension 768 (matching Pythia). Twelve attention heads, head dimension 64. FFN intermediate 3072. Vocabulary 50304 (Pythia's tokenizer). Max sequence length 512. Four specialized layers per breath, partially weight-shared. Maximum 8 loops.

### Parameters

Transformer phase-specific (per layer): Q 590K + K 590K + FFN gate 2.36M = ~3.5M. Four layers: ~14M. Shared (one copy): V 590K + O 590K + FFN up 2.36M + FFN down 2.36M + norms = ~5.9M. Embeddings: ~38.6M (tied with output). Transformer total: ~58.5M.

Controller: ~80M (512d thinking space reading 768d transformer space). Lookup table: 8-12 entries × 768d + coupling matrix. Total system: ~140M.

### Memory (7900 XTX, 24GB)

~4GB total in mixed precision. ~20GB headroom. Batch size 64-128 at sequence length 512. The smaller model means faster iteration — more experiments per day.

### Estimated Training

Week 1: Learn to breathe — L3-L4 with forced then adaptive loops. Validate that pretrained layers improve with looping. Week 2: Multi-step — L4.5 with outer cycles. Week 3: GSM8K. Week 4: Push toward MATH-500. Estimated 3-5 minutes per epoch on GSM8K with the smaller model.

### Platform

AMD Radeon RX 7900 XTX (24GB GDDR6, 960 GB/s bandwidth, ~120 TFLOPS FP16). Tinygrad framework with AM custom userspace driver. No ROCm, no AMD GPU drivers, no PyTorch, no CUDA. Tinygrad and Python standard library only.

---

## 12. What We Carry Forward

The expand-collapse breathing pattern, discovered in attention analysis and confirmed in Llama's per-layer diversity spectrum. π cycling of attention, proven to create structural diversity. Equal-reward decomposition for genuine multi-step reasoning. Number augmentation and number-token weighting for training stability.

The controller's gradient must never flow through the transformer — three days of empirical proof. The generation loss landscape has one dominant basin for any controller output. Structural diversity is necessary; learned diversity always collapses. Differentiable lookup tables from the project's origins return as the prime factorization mechanism.

We leave behind: training from scratch (Pythia provides the language foundation), LoRA atoms and continuous scales, the straight-through gradient estimator, soft token diversity mechanisms, and the PyTorch/ROCm software stack.

---

## Summary

**The breath.** Four specialized layers from Pythia-160M, tracing a sine wave from expansion to compression. Phase-specific Q, K, gate weights specialize each layer for its role. Shared V, FFN, norms carry pretrained English understanding. Temperature modulates the rhythm. The sine wave is periodic — smooth, continuous looping with no discontinuity between breaths.

**The triple helix.** Oscillation provides expand-collapse within each breath. Alternation provides complementary viewing angles across breaths via π-cycled attention with per-head phase offsets — 12 heads scanning 12 frequency bands in parallel, inspired by BirdNET's simultaneous multi-species identification. Integration accumulates evidence across all breaths via gated running integral. Stop when the integral stabilizes.

**The controller.** ~80M conductor thinking in 512 dimensions. Reads the normalized integral from the 768d transformer space. Writes 512d pages. Governs temperature, phase angle, loop count, and integration gate. Factorizes subproblems via the lookup table — the tree emerges from the prime spectrum. No separate decomposition classifier.

**The coarse-to-fine spiral.** Early breaths scan broadly at warm temperature. Later breaths probe precisely at cool temperature. Resolution increases with each breath. The triple helix spirals inward — alternation rotates, oscillation breathes, integration accumulates, temperature cools. Convergence on the precise factorization.

**The lookup table.** 8-12 prime entries at 768d (matching hidden dim). Parallel multi-label detection across 12 heads in one breath. Iterative spectral subtraction confirms primes, counts multiplicities, identifies couplings. The complete factorization specifies the solution tree.

**The efficiency.** No chain-of-thought tokens. All reasoning in representation space. 140M total parameters initialized from a pretrained foundation. Parallel spectral detection in one breath. Compute scales with difficulty, not verbosity.

**The honesty.** Looping pretrained layers is unproven. The core bet — that layers trained for single-pass processing can be productive when cycled with π-shifted attention and sine-wave temperature — must be validated in Week 1.

**The guarantee.** Diversity is structural: π cycling, per-head phase offsets, sine-wave temperature. Gradient descent cannot erase them. The one-basin collapse of v1-v3 is architecturally impossible.

**The platform.** AMD 7900 XTX + tinygrad + AM driver. No proprietary software. Local, open-source, hackable.

A 140M model that breathes, alternates, integrates, and factorizes. Four months to September 1.
