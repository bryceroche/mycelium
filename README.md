# Mycelium v4: The Breathing Transformer
## Conceptual Architecture

**Author:** Bryce + Claude
**Date:** May 29, 2026 (vision: May 1 · empirical status: see §14 + §15)
**Deadline:** December 25, 2026
**Platform:** Shadow Glass (AMD 7900 XTX, 24GB) · tinygrad + AM driver · no AMD/ROCm software
**Target:** MATH-500

---

> ## ⚠️ STATUS NOTE (2026-06-01) — read this before §1-§13
>
> Sections **§1-§13** describe the **v1-v95 era conceptual arc**: sine-wave
> breath, π-cycled RoPE, BirdNET-parallel heads, Controller/Notebook/LookupTable
> closed feedback loop. **This was largely deprecated by the v98 Sudoku pivot
> (2026-05-29).** The current architectures (v98 Sudoku, v100-v107 factor graphs,
> v105.1.2 v2 digit-level) implement a different design: iterative shared-weight
> prefill, structured per-head attention masks, learnable per-breath delta_gate,
> per-breath weighted CE supervision (the "ladder"), variant codebooks.
>
> **For current truth:**
> - **§15** — empirical status (v98 Sudoku 79%, v100-v107, v105.1.2 v2)
> - **§17-§18** — the active architectural framing (musical keys, ODE
>   integrator, two-phase comprehension/inference)
> - **`CLAUDE.md` §2a** — the actual implementation built today
> - **`memory/project_v98_sudoku_validates_paradigm.md`** — the breakthrough
>
> The legacy machinery (`Controller`, `Notebook`, `LookupTable`, sine-modulated
> within-breath temperature, controller-emitted gate) lives in
> `mycelium/breathing.py` but **is not called by any current training path**.
> It is preserved as historical context.

---

## 1. Vision

A small transformer that loops its own layers, breathing through each problem until it finds the resonant decomposition. Not a frozen LLM with external scaffolding — a model that breathes natively. Initialized from Pythia-410M layers 0-3 and fine-tuned end-to-end.

Each loop is one expand-collapse cycle shaped as a sine wave across 4 specialized layers. Sixteen attention heads, each tuned to a different phase angle, scan the problem's spectral content in parallel — identifying which mathematical operations are present simultaneously, like BirdNET identifying multiple overlapping bird species from a single audio clip.

A controller between breaths reads the accumulated understanding, matches it against a library of prime operations, and decides when the analysis is complete. The tree structure of the solution emerges directly from the prime factorization — no separate decomposition classifier needed.

**Three mathematical operations unify the architecture:**

**Oscillation** — sine-wave breath within each loop. Four layers cycle between expansion (broad attention, open to new information) and compression (sharp focus, distilling the essential signal). Temperature modulates the expand-collapse rhythm continuously.

**Alternation** — π-cycled attention across loops. Each loop rotates the viewing angle so the model sees the problem from a different geometric perspective. Sixteen heads per breath scan different frequency bands in parallel. Structural diversity that gradient descent cannot erase.

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

Inspired by BirdNET (Cornell Lab of Ornithology), which identifies multiple overlapping bird species from a single audio spectrogram by scanning different frequency bands simultaneously, our 16 attention heads each operate at a different phase offset.

Head 0 at phase 0. Head 1 at phase π/16. Head 2 at 2π/16. Through head 15 at 15π/16. Each head is tuned to a different frequency band of the problem's spectral content. In a single breath, all 16 heads scan the full range in parallel.

BirdNET doesn't separate overlapping bird songs into individual streams. It classifies them simultaneously with independent confidence scores — robin AND cardinal AND wren, all detected from one clip. Our heads work the same way: head 3 responds strongly to subtraction patterns while head 7 responds to entity structure, all in one breath. The combined multi-head response IS the spectral decomposition.

Each loop, all heads rotate together by π/max_loops while maintaining their relative offsets. The bank of heads shifts as a unit, scanning new regions with each breath. Sixteen heads × 8 loops = 128 distinct phase angles sampled.

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

### The Notebook

The notebook is the controller's memory — 512d pages written after each breath, persisting across both inner breathing loops and outer execution cycles. Without it, the controller makes decisions based only on the current breath's observation. With it, the controller compares "what I see now" against "what I saw three breaths ago" — detecting convergence, tracking new primes, monitoring confidence trends.

The notebook also bridges outer cycles. When cycle 2 executes, the controller reads notebook pages from cycle 1 — carrying representation-level understanding that is richer than the generated text tokens alone. The context carries "Step 1: 263" as text. The notebook carries the full 512d page encoding the controller's understanding of WHY 263 was produced, which operation generated it, and what operation comes next.

Tree-structured attention over notebook pages lets the controller read ancestors (what was the original problem?), siblings (what other operations were identified?), and children (what results have been produced?).

---

## 5b. The Closed Feedback Loop

### Why All Components Are Necessary

The breathing transformer's core components — rotation, integration, notebook, lookup table, controller, temperature modulation, and step size — form an irreducible closed feedback loop. Each component amplifies the others. Remove any one and the system degrades.

### The Loop

Rotation provides a new viewing angle. The transformer expands, processes, compresses, and integrates at that angle. Integration accumulates this observation into the running integral alongside all previous observations. The controller writes a notebook page recording what this breath understood. The lookup table matches the page against its library of known prime operations, returning match weights and confidence. The controller reads the match confidence, compares with previous notebook pages, and decides: is the factorization converging? Is confidence sufficient? If not, the controller adjusts the rotation angle (target the weakest-matched prime), the temperature (broader if uncertain, sharper if nearly confirmed), and the step size (smaller if modes are closely spaced). The next breath rotates to the adjusted angle and the loop continues.

The loop terminates when two conditions are met: the integral has stabilized (Lyapunov criterion — new breaths add negligible information) and the spectral residual is noise (all significant primes have been identified and subtracted).

### What Each Component Contributes

**Rotation** provides independent observations. Without it, every breath sees the same view and integration accumulates redundant information.

**Integration** makes observations cumulative. Without it, each breath's insight is forgotten when the next begins. Rotation without integration is amnesia.

**The notebook** provides memory across breaths and cycles. Without it, the controller has no basis for comparing current observations against previous ones. It cannot detect convergence or track the factorization's evolution.

**The lookup table** provides the reference library and target map. Without it, the controller adapts rotation based on energy signals alone — "something is changing" — without knowing WHAT it's looking for. The lookup table transforms blind search into guided search.

**The controller** provides adaptive feedback. Without it, rotation is uniform (fixed step size), temperature is fixed (no adaptation to problem difficulty), and stopping is arbitrary (fixed loop count). The controller is the intelligence of the loop.

**Temperature modulation** controls resolution at each angle. Without it, every breath has the same precision — wasting early breaths on unnecessary precision and starving late breaths of the precision they need.

**Step size (rotation rate)** determines spectral coverage efficiency. Without adaptive step size, closely-spaced modes might be missed (step too large) or the search takes too many breaths (step too small). The Nyquist theorem applies: to resolve two primes separated by Δθ in phase angle, the rotation step must be at most Δθ/2.

### Current State: Full 7/7 Closed Loop Implemented

As of 2026-05-10, all seven components are implemented and wired together:

| # | Component | Status | File |
|---|---|---|---|
| 1 | Rotation (π-cycled RoPE, per-head phase offsets) | ✓ | `breathing.py: RoPE` |
| 2 | Integration (gated running integral across breaths) | ✓ | `breathing.py: BreathingBlock.breathe` |
| 3 | Notebook (512d pages, persisting across cycles) | ✓ | `controller.py: Notebook` |
| 4 | Lookup table (16×1024 cosine matcher, joint-trained) | ✓ | `lookup_table.py: LookupTable` |
| 5 | Controller (state reader + decision heads + notebook attn) | ✓ | `controller.py: Controller` |
| 6 | Temperature modulation (controller scalar × sine baseline) | ✓ | `breathing.py: BreathingLayer (temp_mult)` |
| 7 | Step size / rotation rate (controller emits step_mult) | ✓ | `breathing.py: breathe_controlled` |

The closed loop is invoked via `model.breathe_controlled(tokens, max_loops, notebook)`. Per breath: controller reads the running integrated rep → writes a 512d page into the notebook → notebook attention refines the page over all prior pages → decision heads emit `{temperature, gate, stop_logit, step_mult}` for the next breath → the breath runs at the controller's chosen temperature with the integral weighted by the controller's gate, and RoPE indexed by the controller's adaptive phase.

Gradient separation is enforced by construction:
- The transformer is trained on the main CE loss + a small joint lookup-table aux CE. `model.parameters()` returns just transformer + lookup_table params.
- The controller is trained by `controller_train_step` on a separate optimizer over `model.controller_parameters()`. Its loss is per-breath lookup-CE + stop calibration. The loss reaches transformer params too but those grads are simply discarded on the next `main_opt.zero_grad()`.
- Verified on a 5-step joint smoke: 0/39 transformer params changed in controller training; 61/62 controller params changed.

**The 7/7 implementation is the architecture's first complete realization.** The L3-spaced 65% and L4 10% results in earlier runs were obtained with only rotation + integration active. The full system is now ready for empirical validation: does adding the lookup table, notebook, controller, and adaptive rotation produce gains on L3-spaced, and (critically) on L4-spaced where the controller has the most to contribute (deciding *which step's operation comes next*)?

---

## 6. The Unified Framework: Detect, Refine, Execute

### Three Ideas, One Mechanism

Three concepts that appeared separately during the design process are actually the same mechanism viewed from different angles:

**BirdNET parallel detection.** All operations identified simultaneously across 16 heads, like BirdNET identifying multiple overlapping bird species from a single audio clip. Each head is tuned to a different spectral band. One breath scans all frequencies in parallel.

**Coarse-to-fine resolution.** Each subsequent breath refines the detection — from "some arithmetic" to "specifically subtract 132 from 170." The temperature cools, the attention sharpens, the factorization becomes more precise. Early breaths have high recall and low precision. Later breaths have high precision.

**Equal-reward decomposition.** Each intermediate target earns 1/N reward. The model is incentivized to identify and correctly solve ALL operations, not just the easiest ones. This incentivizes breathing until the resolution is sufficient for correct answers at every target.

These three are one process: the 16 heads detect all operations in parallel (BirdNET), each breath refines the detection from coarse to fine (resolution), and the equal-reward structure incentivizes continuing until every target can be claimed correctly (decomposition). The breathing loop IS the coarse-to-fine refinement. The lookup table IS the detector bank. The equal reward IS the stopping incentive.

### How It Works

**Breath 1 (coarse — warm temperature, broad attention).** All 16 heads ring the problem simultaneously. Each head at a different phase angle scans a different spectral band. The lookup table returns approximate match weights for all prime entries. "There's subtraction and something multiplicative in here." All operations are on the radar. Confidence is low. The model COULD claim all targets now — but would get most of them wrong because the resolution is too coarse.

**Breath 2 (medium — cooler, sharper).** Same layers, π-rotated phase angles. The integration now has two independent observations. Spectral resolution doubles. "Specifically subtraction of 132, and the multiplicative thing is doubling." Prime factorization sharpens. Multiplicities emerging. The model could claim targets with moderate confidence.

**Breath 3 (fine — cooler still).** Three independent observations integrated. "290 - 27 is step 1, then result + 83 is step 2. Sequential coupling." Specific numbers and operations resolved. Couplings identified. The factorization is nearly complete.

**Breath 4 (verification).** The residual after subtracting all confirmed primes is examined. If noise — all primes found, factorization complete, resolution sufficient. If structure remains — another breath needed. The controller monitors both integral convergence (Lyapunov: has the accumulated understanding stabilized?) and spectral completeness (has the residual dropped to noise?).

Each breath simultaneously rotates the viewing angle (alternation), cycles through expand-collapse (oscillation), and accumulates evidence (integration). The triple helix spirals inward toward the precise factorization. Coarse to fine, broad to sharp, hypothesis to confirmation.

### The Differentiable Lookup Table

The lookup table stores the fundamental operations of mathematics as orthogonal vectors — a prime basis. For elementary math: addition, subtraction, multiplication, division, fraction-of, comparison, combination, sequential dependency. Start with 16 entries, each at 1024 dimensions (matching the transformer's hidden space). Grow as the model encounters harder math.

Each entry stores four things:

**Pattern.** What this operation looks like in the transformer's representation space. Used for matching.

**Resonant angle.** Which π-cycling phase angle most clearly reveals this operation. Learned through training. Guides the controller's phase selection.

**Subtraction mask.** What to remove from the signal once this operation is confirmed. For computing the residual during iterative spectral subtraction.

**Confidence threshold.** How much energy must drop to confirm this operation's presence.

Plus a 16×16 coupling matrix encoding relationships between co-occurring primes — independent (parallel branches) vs coupled (sequential chains).

### Iterative Spectral Subtraction with Multiplicities

After breath 1's parallel scan returns approximate match weights, subsequent breaths confirm individual primes and count multiplicities. Confirm a prime, subtract its contribution from the signal, examine the residual. If the same prime still matches the residual, it appears again — ring and subtract until that prime is exhausted. Repeat for remaining primes until the residual is noise.

Operations can appear multiple times in a single problem. "Subtract 27, then add 83, then subtract 218" contains subtraction twice. The factorization captures multiplicities: subtraction² × addition¹. The multiplicity count directly determines the number of execution steps.

### From Factorization to Tree (Instant)

The complete factorization — primes, multiplicities, couplings — fully determines the execution plan. The number of primes with multiplicity equals the number of steps. The coupling matrix determines the shape: independent primes create parallel branches, coupled primes create sequential chains. The tree writes itself from the factorization. No learned heuristic, no trial and error.

### Execution: Light Outer Cycles

Once the factorization is complete and the plan is determined, execution cycles are LIGHT — 1-2 breaths each. The heavy analysis is done. Each cycle generates one equation claiming one target at 1/N reward. The only new information at each execution cycle is the previous step's numerical result.

The efficiency gain over naive re-breathing:

Heavy analysis once (4-8 breaths) + N light execution cycles (1-2 breaths each) = 8 + 2N passes. Versus re-breathing every cycle: 8N passes. For a 3-step problem: 14 vs 24 passes. For a 5-step problem: 18 vs 40. Scales better with problem complexity.

### Why Digit-by-Digit Generation Matters

A critical empirical discovery: Pythia's BPE tokenizer encodes entire numbers as single tokens ("170" → token 15046). This means the model must predict 3-digit arithmetic results in a single softmax over 50K vocabulary entries — a memorized lookup, not computation. Breathing cannot help because there's nothing to iterate on when the answer is one token.

Digit-spaced generation ("1 7 0 - 1 3 2 = 3 8") transforms arithmetic into a sequential prediction problem. Each digit is its own token, its own forward pass, its own softmax. Borrow and carry tracking happens autoregressively across digits. Breathing in representation space refines the computation for each digit position.

With digit spacing, accuracy jumped from 71% (single-token ceiling) to 87.5% on peek samples — and the model is genuinely computing, not memorizing. The single error in 8 samples was one wrong digit (tens-place borrow), not a completely wrong number. A fundamentally different and more correctable failure mode.

### The Sine Wave Is Periodic

A sine wave from 0 to 2π starts and ends at the same height. The output of one breath is at exactly the right representational altitude for the input of the next breath. No discontinuity between loops. The looping is smooth because the sine wave IS smooth and periodic. This is by design — the layers are reusable because the oscillation returns to its starting point.

### Why the Layers Are Reusable

Each breath doesn't ask the layers to do something fundamentally different. The layers always expand, process, compress, and integrate. What changes is the resolution of the input and the temperature of the attention. The layers are a general-purpose expand-compress pipeline that works at any resolution — like a microscope with the same optics at every magnification. The π cycling provides new viewing angles, the integration provides increasing resolution, and the layers just process.

### The Inward Spiral

The triple helix spirals inward across breaths. Alternation rotates the viewing angle. Oscillation provides the expand-collapse rhythm. Integration accumulates evidence. Temperature cools progressively under the controller's direction. Each breath simultaneously rotates AND increases resolution. The spiral converges on the precise factorization — coarse to fine, broad to sharp, hypothesis to confirmation. When all targets can be claimed correctly, the breathing stops and execution begins.

---

## 7. Training Data: Haiku-Curated Decompositions

A small model (Claude Haiku or equivalent) preprocesses GSM8K problems to create multi-pass training data. For each problem, the haiku model identifies the operations, decomposes them into steps, and produces annotated training examples showing what each breath SHOULD discover at each cycle.

This gives the breathing transformer supervised targets for the spectral analysis — "breath 1 should detect subtraction and entity-tracking, breath 2 should confirm subtraction with multiplicity 2." Without this, the model must discover the spectral decomposition purely through REINFORCE, which is slow. The curated data provides a curriculum for learning to breathe.

---

## 8. The Copy Machine Principle

### Why Representation Space, Not Token Space

The breathing transformer does NOT generate reasoning tokens between breaths. All thinking happens in representation space — 512d pages in the controller, 1024d hidden states in the transformer. This is not just an efficiency choice — it is an architectural necessity, proven empirically.

### The Copy Machine Problem

Chain-of-thought reasoning forces the model to "photocopy" its understanding into tokens at every reasoning step. Each token is a lossy compression — the model's full 1024-dimensional continuous understanding squeezed into a single discrete choice from 50,000 vocabulary items. Information is destroyed at each step. The next step builds on that destroyed information. It's a copy of a copy of a copy.

We observed this directly in validation: when Pythia's layers generate tokens and feed them back in (the autoregressive pipeline), output degrades to "had had had" within 2 loops. Each generated token introduces small errors that become the source material for the next token. The errors compound monotonically — they can never improve, only worsen.

### The Original, Not the Photocopy

The hidden states are the original painting. They carry the model's full continuous understanding — all 1024 dimensions, all the nuance, all the per-problem diversity. Our validation proved this: centered cross-problem cosine stays at -0.05 (orthogonal) through 4 loops. Effective rank holds at 15-16. Signal norm actually GROWS across loops (3.9 → 6.4). The original doesn't degrade when you keep working with it.

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

## 9. The Diffusion Connection

### Breathing as Iterative Denoising

The coarse-to-fine resolution progression of the breathing loop mirrors diffusion models with striking precision. Diffusion models start with noise and iteratively refine toward a clear output. Each step reduces entropy — early steps establish coarse structure, late steps add fine detail. The breathing transformer does the same: early breaths establish coarse factorization ("this involves subtraction and entities"), late breaths refine to precise operations and multiplicities.

### The Parallels

The noise-to-signal progression in diffusion corresponds to the coarse-to-fine understanding in breathing — both are iterative refinement from disorder to order, from uncertainty to precision.

The score function in diffusion (the learned denoiser that says "move in this direction") corresponds to the controller. Both are learned guides that steer an iterative refinement process — "this region should become an eye" maps to "this part of the representation should become a subtraction operation."

The noise schedule in diffusion (large denoising steps early, small steps late) corresponds to the temperature schedule. Both are annealing from exploration (broad, coarse) to exploitation (sharp, fine).

The adaptive step count in diffusion (complex images need more steps) corresponds to adaptive loop depth. Both systems benefit from spending more compute on harder inputs.

### The Critical Shared Principle

Both diffusion models and the breathing transformer work in continuous representation space, never producing intermediate discrete outputs. Diffusion models refine continuous pixel values — they never quantize to a discrete image between steps. The breathing transformer refines continuous hidden states — it never generates discrete tokens between breaths. Both avoid the copy machine degradation by staying in continuous space throughout the iterative process. The discrete output (final image, generated tokens) is produced once at the end.

### Classifier-Free Guidance as DC Subtraction

In diffusion, classifier-free guidance runs the denoiser twice — once conditioned on the prompt and once unconditionally — then amplifies the difference. The "guided" signal is the conditional minus unconditional response. This is mathematically identical to our DC subtraction concept: the DC component is the unconditional response (what every problem produces), and the residual is the guided response (what this specific problem uniquely produces). Two domains, same mathematical operation.

### Optimal Temperature from SNR Theory

Diffusion models have well-studied theory for the optimal noise schedule — how much to denoise at each step, derived from the signal-to-noise ratio at that step. We have measured SNR at each breath (0.114 → 0.127 for L0-3 across 8 loops). The same variational framework could derive the optimal temperature progression: breathe broadly when SNR is low (early breaths, high uncertainty), breathe sharply when SNR is high (later breaths, high confidence). This makes the temperature schedule principled rather than heuristic — a theoretically optimal default with small learned adjustments from the controller on top.

### What This Reframes

The breathing transformer may be understood as a diffusion model over mathematical reasoning. The representation space is the "image" being denoised. The prime factorization from the lookup table is the "prompt" guiding the denoising. Each breath is a denoising step. The controller is the score function. The temperature schedule is the noise schedule. The integration is the accumulated denoising trajectory.

This connection to a well-established theoretical framework (score matching, DDPM, variational inference) provides a rich foundation for understanding why the breathing architecture works, how to optimize it, and what its theoretical limits are.

---

## 10. Honest Unknowns

### Will the Looping Layers Work?

This is the biggest open question. Cycling the same 4 transformer layers multiple times with different phase angles is genuinely novel. The layers might only be useful once — the representation after one pass might not benefit from a second pass through the same weights, even with rotated attention.

**Reasons for concern:** Standard transformers use different weights per layer for a reason. The FFN computation is shared across phases and might dominate layer behavior regardless of attention rotation. The transition from TROUGH output back to RISE input between loops might be discontinuous.

**Reasons for optimism:** Deep Equilibrium Models (2019) proved that a single layer iterated to convergence matches deep transformers. Universal Transformers (2018) showed weight-tied layers with step embeddings work on algorithmic tasks. Block Recurrent Transformers (2022) validated recurrence within transformer blocks. Our setup is less extreme — 4 specialized layers with partial sharing, not 1 fully tied layer.

**The mitigation:** Start with forced loop counts on simple problems (L3). Measure whether accuracy increases with loops (more thinking helps) or decreases (representation degrades). If degradation, the looping architecture needs revision — possibly more layers per breath, or different normalization between loops.

### Will 16 Heads at Different Phases Provide Meaningful Spectral Coverage?

Sixteen phase angles spanning [0, π) is 16 samples of the spectral space — denser coverage than the original 12-head design. With 16 entries in the lookup table and 16 heads, there's a natural one-to-one correspondence available: each head could specialize for detecting one prime operation. Whether the model discovers this alignment or uses the heads more fluidly is an empirical question.

### Will the Prime Factorization Remain Unique Under Training?

The orthogonal initialization guarantees initial uniqueness. But training pushes entries toward whatever reduces loss. If two operations are highly correlated in the training data (subtraction and sequential-dependency often co-occur), their entries might converge. The regularization term helps but might not be sufficient. Monitoring entry pairwise similarity is essential.

### Can a Small Model Learn Math?

Addressed by initializing from Pythia-410M. The pretrained layers already understand English — entity tracking, sentence structure, counterfactuals. At h=1024 with 16 heads, the layers have substantially more processing capacity than the previous h=768 design. The breathing fine-tuning adds mathematical reasoning on top of existing language competence.

The remaining risk: Pythia's layers were trained for single-pass processing. They might resist being looped. But the phase-specific weights (Q, K, gate) are reinitialized for the breathing phases — only the shared weights (V, FFN, norms) carry pretrained assumptions, and those components are genuinely phase-independent.

### Will the Gradient Landscape Have Multiple Basins?

The v1-v3 one-basin collapse was through Llama — a model pretrained on trillions of tokens where "math mode" was deeply baked in. Our custom setup, fine-tuned from Pythia with π cycling from the start, might develop a DIFFERENT landscape. The model has never known a world without phase-shifted attention — it might naturally develop multiple basins corresponding to different operation types and phase angles.

Even if the transformer develops one basin, the controller is protected by complete gradient separation. Its learning signal comes from REINFORCE and auxiliary losses, never through the transformer. The architecture is designed to work either way.

---

## 11. Specifications

### Initialization: Pythia-410M

The breathing transformer initializes from the first 4 layers of EleutherAI's Pythia-410M — a 24-layer, 1024-dim model trained on 300B tokens of diverse English text (the Pile). Taking 4 of 24 layers captures the model's early processing stages, which provide strong language understanding without deep single-pass specialization.

Pythia-410M was chosen for its balance of capacity and efficiency. Its 16 standard multi-head attention heads (not grouped query attention) give each head full independence for per-head π-cycling phase offsets — 16 heads at 16 phase angles provides dense spectral coverage with a natural correspondence to the 16-entry lookup table. Its hidden dimension of 1024 with head_dim=64 gives substantially more processing capacity than the h=768 alternative. It uses RoPE which our π cycling extends naturally. Apache 2.0 licensed.

What we take from Pythia: layers 0-3 (attention + FFN weights), token embeddings (1024 × 50304), and the output head. What we modify: Q, K, and FFN gate projections become phase-specific (4 unique copies, initialized from Pythia's layers 0-3). V, O, FFN up/down, and norms remain shared.

### Model Dimensions

Hidden dimension 1024 (matching Pythia-410M). Sixteen attention heads, head dimension 64. FFN intermediate 4096. Vocabulary 50304 (Pythia's tokenizer). Max sequence length 512. Four specialized layers per breath, partially weight-shared. Maximum 8 loops.

### Parameters (Rebalanced)

The parameter budget is balanced between thinking (transformer) and conducting (controller), with neither dominating:

Transformer phase-specific (per layer): Q 1.05M + K 1.05M + FFN gate 4.19M = ~6.3M. Four layers: ~25.2M. Shared (one copy): V 1.05M + O 1.05M + FFN up 4.19M + FFN down 4.19M + norms = ~10.5M. Embeddings: ~51.5M token + ~51.5M untied output head = ~103M. Transformer total: ~139M (35.7M processing + 103M embeddings); training log reports 134.5M trainable params.

Controller: ~40M (20M state reader with 2 Perceiver layers reading 1024d into 512d, 12M notebook attention with 3 layers, 8M decision heads). Slim and decisive — the controller reads the integral, matches against primes, and decides temperature/phase/stop. It doesn't need to be larger than the orchestra it conducts.

Lookup table: 16 entries × 1024d pattern + 1024d subtraction mask + resonant angle + confidence threshold = ~33K. Plus 16×16 coupling matrix = 256 parameters. Living in the transformer's native 1024d space — no projection needed for matching.

Total system: ~127M. Effective processing with 8 loops: 35.7M × 8 = 286M effective transformer capacity + 40M controller = 326M effective.

### Memory (7900 XTX, 24GB)

~5GB total in mixed precision. ~19GB headroom. Batch size 64-128 at sequence length 512. Fast iteration.

For cached inference: K/V buffers are sized to the actual sequence length, not the model's max. For arithmetic at L3-spaced (~30 tokens) with cache_max_len=32, the cache footprint at B=100 is ~2GB instead of ~33GB at the full 512 RoPE table size. The model's compute budget — not its memory — is the binding constraint at inference batch sizes that matter.

### Inference Engine

The breathing transformer's inference path is a JIT-fused KV cache: per-loop, per-layer K/V buffers shared across the batch, with per-batch position tracking so variable-length prompts coexist in one compiled graph. Phase A breathes the prompt once and writes 32 K/V tensors (4 layers × 8 loops). Phase C iterates token-by-token, replaying a single TinyJit graph that fuses the embedding lookup, all 32 layer-loop passes, the integration, the output norm, and the argmax into one launch per generation step.

Compiled graphs live in a per-model dict keyed on (batch_size, n_loops, vocab_active). The first eval cycle compiles each unique configuration once (~45s per graph; for a typical sweep over EVAL_LOOPS=[1,2,4,8] that is ~3 minutes total, paid once). Every subsequent eval cycle is pure replay — zero compile cost, however many times we evaluate. Eval calls are padded up to a fixed batch_size so chunk size never drifts and the same compiled graphs match every run. The pattern is **compile once at the start of training (during the first eval), reuse the compiled kernels for every subsequent eval for the rest of the run**.

Measured against the original uncached B=1 sequential path on the L3-spaced step-600 checkpoint (N=100, LOOPS=8):

| Path | Time | Speedup |
|---|---:|---:|
| Uncached B=1 sequential | 268s | 1× |
| Cached B=1 sequential | 27s | 10× |
| Cached batched, warm-up (one-time JIT compile) | 49s | 5.5× |
| **Cached batched, steady-state (B=100, cache_max_len=32)** | **6.3s** | **42.8×** |

100/100 exact text match against the uncached path — bit-for-bit identical reasoning, just dramatically faster. The accuracy eval that dominated training wall-clock (10+ minutes per checkpoint at B=1 sequential) now runs in ~12 seconds steady-state.

### Estimated Training

Phase 0 (loop consistency): Days 1-3. Phase 1 (breathing curriculum L3-L4.5): Days 4-7. Phase 2 (controller + lookup table): Week 2. Phase 3 (GSM8K): Weeks 3-4. Phase 4 (MATH-500): Months 3-4. Estimated 3-5 minutes per epoch on GSM8K.

### Platform

AMD Radeon RX 7900 XTX (24GB GDDR6, 960 GB/s bandwidth, ~120 TFLOPS FP16). Tinygrad framework with AM custom userspace driver. No ROCm, no AMD GPU drivers, no PyTorch, no CUDA. Tinygrad and Python standard library only. Requires Linux (Ubuntu 24.04 recommended) — tinygrad's AM driver is Linux-only.

---

## 12. What We Carry Forward

The expand-collapse breathing pattern, discovered in attention analysis and confirmed in Llama's per-layer diversity spectrum. π cycling of attention, proven to create structural diversity. Equal-reward decomposition for genuine multi-step reasoning. Number augmentation and number-token weighting for training stability.

The controller's gradient must never flow through the transformer — three days of empirical proof. The generation loss landscape has one dominant basin for any controller output (though our custom setup may develop multiple basins — to be tested). Structural diversity is necessary; learned diversity always collapses in pretrained models. Differentiable lookup tables from the project's origins return as the prime factorization mechanism.

The looping validation proved: signal survives (norm grows 3.9 → 6.4 across 8 loops, ~1.6×), diversity holds (effective rank 16+), SNR improves (0.114 → 0.127 for L0-3). The generation head needs fine-tuning to extract signal from looped representations — the DC component grows linearly. The copy machine principle: reasoning in representation space is necessary. Generate once at the end, never between breaths.

The inference engine works: a JIT-fused KV cache with per-batch position tracking, K/V buffers sized to actual sequence length (not the model's max), and a fixed-batch eval path that compiles once and reuses across every checkpoint. 42.8× over the original uncached path on the L3 eval set, bit-for-bit identical outputs. Eval went from dominating training wall-clock to a rounding error. Faster eval means we can evaluate more often, on more held-out examples, with longer breathing budgets — every architectural question gets answered in seconds instead of minutes.

We leave behind: Llama 1B (replaced by Pythia-410M L0-3), LoRA atoms and continuous scales, the straight-through gradient estimator, soft token diversity mechanisms, the PyTorch/ROCm software stack, and Windows (wipe Shadow Glass, install Ubuntu 24.04).

---

## Summary

### "The Shape of Thought"

**The breath.** Four specialized layers from Pythia-410M (h=1024, 16 heads), tracing a sine wave from expansion to compression. Phase-specific Q, K, gate weights specialize each layer for its role. Shared V, FFN, norms carry pretrained English understanding. Temperature modulates the rhythm. The sine wave is periodic — smooth, continuous looping with no discontinuity between breaths.

**The triple helix spiraling inward.** Oscillation provides expand-collapse within each breath. Alternation provides complementary viewing angles across breaths via π-cycled attention with 16 per-head phase offsets scanning 16 frequency bands in parallel, inspired by BirdNET's simultaneous multi-species identification. Integration accumulates evidence across all breaths via gated running integral. Each breath simultaneously rotates the viewing angle AND increases resolution — coarse to fine, broad to sharp, hypothesis to confirmation. Stop when the integral stabilizes.

**The unified framework.** Three concepts that are actually one mechanism: BirdNET parallel detection (all operations identified simultaneously across 16 heads), coarse-to-fine resolution (each breath sharpens the factorization from "some arithmetic" to "specifically subtract 132 from 170"), and equal-reward decomposition (incentivizes breathing until ALL targets can be claimed correctly). The breathing loop IS the refinement. The lookup table IS the detector bank. The equal reward IS the stopping incentive. Detection, refinement, and execution flow as one continuous process — heavy analysis once, then light execution cycles.

**Digit-by-digit arithmetic.** A critical empirical discovery: BPE tokenizers encode whole numbers as single tokens, forcing the model to predict 3-digit answers in one softmax — memorization, not computation. Digit-spaced generation ("1 7 0 - 1 3 2 = 3 8") transforms arithmetic into sequential digit-by-digit prediction where breathing can refine each position. Accuracy jumped from 71% (single-token ceiling) to 87.5% on peek samples. The model genuinely computes — tracking borrows across digits — rather than memorizing a lookup table.

**The controller.** ~40M conductor thinking in 512 dimensions. Slim and decisive. Reads the normalized integral from the 1024d transformer space. Writes 512d pages. Governs temperature, phase angle, loop count, and integration gate. Rotation guided by the lookup table — the 16 prime entries tell the controller which phase angles to probe for each candidate operation. The tree structure emerges directly from the prime factorization. No separate decomposition classifier.

**The lookup table.** 16 prime entries at 1024d (matching hidden dim), each with a pattern, resonant angle, subtraction mask, and confidence threshold. Plus a 16×16 coupling matrix encoding operation dependencies. Parallel multi-label detection across 16 heads in one breath — all operations identified simultaneously. Iterative spectral subtraction confirms primes, counts multiplicities, identifies couplings. The complete factorization specifies the solution tree.

**The copy machine principle.** All reasoning in representation space. The signal norm grows 3.9 → 6.4 across 8 loops (~1.6×) while diversity holds perfectly (effective rank 15-16, centered cross-problem cosine -0.05). Generate tokens ONCE after breathing converges — never between breaths. Autoregressive mid-breath generation is a copy machine that degrades exponentially. The hidden states are the original painting. Breathe with the original. Print only the final version.

**The parameter balance.** ~35.7M transformer processing + ~103M embeddings (token + untied output) + ~6.6M controller (Step C; spec target ~40M at Step E) = ~145M total. With 8 loops: 286M effective processing from 35.7M reusable parameters. The conductor is appropriately smaller than the effective orchestra (286M). Massive parameter efficiency through weight reuse.

**The empirical foundation.** L3 training (2000 steps): 8-loop accuracy beat 1-loop (70% vs 65%). Loss gap closed 73% (0.77 → 0.20 nats). All depths converge. More thinking produces better answers. The 65% ceiling is arithmetic precision (4-layer model limit), not a breathing limitation. Causal mask verified — all training is honest.

**The inference engine.** JIT-fused KV cache with per-batch position tracking, K/V buffers sized to actual sequence length, fixed-batch eval that compiles once and reuses across checkpoints. 42.8× faster than the original uncached path on the L3 eval set (268s → 6.3s for 100 problems at LOOPS=8). 100/100 exact text match — bit-for-bit identical reasoning, just faster. Eval is no longer a bottleneck.

**The honesty.** Looping pretrained layers requires fine-tuning — frozen weights don't loop for generation (validated empirically). The core bet: fine-tuning can close the generation gap by teaching the output head to extract the rich signal that provably survives looping. The gradient landscape may develop multiple basins in our custom setup — the architecture works either way through complete gradient separation.

**The guarantee.** Diversity is structural: π cycling, 16 per-head phase offsets, sine-wave temperature. Gradient descent cannot erase them. Every breath sees the problem from a different angle at a different resolution. The one-basin collapse of v1-v3 is architecturally impossible.

**The platform.** AMD 7900 XTX + tinygrad + AM driver. Ubuntu 24.04. No proprietary software. Local, open-source, hackable. 75 TFLOPS fp16 matmul. 2.3 min/epoch on GSM8K — 2.5× faster than the best v3 configuration on AWS.

---

## 14. Empirical Status (May 27, 2026)

The architecture above is the design. The empirical evolution has surfaced what's load-bearing and what's scaffolding. This section is the honest current-state record.

**Two paradigms, two champions, plus a third paradigm bridging to GSM8K.**

**Misaligned-decode paradigm champion: v45 reg take 3 step 1000** = **96 / 94 / 93** on L4_MIXED eval (A=1 / A=4 / A=8). File: `.cache/l4_mixed_ckpts/v45_reg_take3_step1000.safetensors`. Warm-started from v24c step 500 (96/94/91) with the regularization stack below.

**Rep-space-thinking paradigm champions (May 19):** v55 step 500 = 89% aligned on L4 (K=2 breaths); v56 step 500 = 89% segmented on L4.5 (K=3); v59 step 1500 = 45% segmented on L4.7 (K=4) with PERFECT step decomposition (per-step accuracy ~78% — breaks the 4-layer 65% arithmetic ceiling). The K-axis is doing real work: at K=3 breaths SPECIALIZE — breath-1 decodes division, breath-2 decodes subtraction on the same problem. The segmented–aligned gap of +14 pt at K=3 is the proof.

**DAG paradigm (v77+, bridge to GSM8K, in progress):** instead of emitting a numeric answer, the final breath emits a SymPy-executable DAG (`x0 = 50 / 60 ; x1 = x0 * 12 ; answer = x1`). SymPy executes the arithmetic. The model only needs to learn STRUCTURAL correctness — operand binding, op selection, dependency graph — not multi-digit arithmetic. K=7 breaths × 7 Haiku-distilled supervision layers (L0: paraphrase → L6: pure DAG). This separates the orchestration problem (which the model handles well — v59 perfect decomposition) from the arithmetic-execution ceiling.

**Waist load-bearing — VALIDATED (May 25):** the WaistController paradigm raises the question: is the 512d waist actually carrying the reasoning, or is the decoder just bypassing it via prompt cross-attn? Direct ablation answers this: with the waist zeroed, CE rises +3.77 on average across 7 breaths (+5.6/+5.9 at L5/L6). With breath-0's waist substituted for breath-k's, CE rises +4.59 — so waists are breath-SPECIFIC, not interchangeable. The architecture is doing what it was designed to do; the WaistController capacity (4 cross-attn layers) is right-sized.

**Regularization stack (validated 2026-05-18):**
- `STOCH_DEPTH_P=0.10` — per-breath Bernoulli drop with ResNet-style 1/(1-p) scaling. Mask-gen guarantees ≥1 active-breath kept per step (no all-dropped catastrophe); skipped entirely at n_loops=1 where SD has no meaning.
- `LABEL_SMOOTHING=0.1` — applied to main answer-CE only. Training-only (eval CE gated on `Tensor.training` so reported val loss stays comparable across runs).
- `WEIGHT_DECAY=0.05` — bumped from the 0.01 default. AdamW typical range for this scale.

**STAGE2_NOTEBOOK bug (caught 2026-05-18):** The v40-era inference code added per-token notebook reads/writes inside `cached_generate_batch`'s Stage 2 decode JIT, intending to "mirror training." But training has no autoregressive decode — adding ~240 OOD notebook updates per problem produced low teacher-forced val loss + 0% generation acc + garbage output ("M 1 1lezIntrIntr..."). Fixed by gating the Stage 2 notebook ops behind `STAGE2_NOTEBOOK` env var (default 0 → restores v24c-compatible behavior). v45 takes 1 and 2 collapsed because of this; take 3 succeeded once the gate was in.

**What ablations established:**
- π-cycled RoPE (per-head phase offset) is the clearly load-bearing closed-loop component: −73 points if ablated, vs −0 to −6 for the others at 150 steps.
- Sine-baseline temperature (2.0 → 0.7 cosine half-period) is load-bearing for warm-start stability.
- Regularization stack (v45, above) — load-bearing for breaking past v24c's overfitting wall and compressing the depth-gradient.
- Integration, notebook, controller decisions (temperature/gate/step_mult), step-size adaptivity — measured as decorative on converged ARITH_HARD. The controller specifically learned `f(breath_idx)`, not `f(rep)`: an open-loop schedule, problem-blind.

**Verification probe — definitive negative:** MLP probes on trained reps (up to 17M params, 1000 steps) cannot distinguish correct from wrong answers (AUC 0.29, anti-correlated). Verification information is not present in the reps as trained. The "7/7 closed feedback loop" is correctly wired but most of it doesn't yet have a job that benefits the loss.

**v80 ladder iterations + DISCOVERY (May 25-27):** 10 sub-versions of v80 chased per-breath CE smoothness via Haiku template tuning ($200+ spent). CV dropped from v1's 1.34 to v3's 0.47 best. But two diagnostics in the final week reframed the project:

1. **Eval bug — missing notebook_pool_mask** silently invalidated 9 iterations of accuracy measurements. After fix, v80_prod_step400 went from 0% → 28.3% DAG parse rate, 1.7% accuracy. The CE chase had been on a model that actually worked under teacher forcing; we just couldn't measure it.
2. **FOUR-mask training requirement** discovered during v81 Phase 2 masking audit. Beyond kv_mask + notebook_pool_mask, we ALSO need main_attn_mask (main self-attn at answer-span KV) and embed_mask (zero input embeddings at answer-span). All 4 are required for clean train/eval consistency. v80 era trained with only 2 of 4.

**v81 paradigm (current, May 27): same task per breath at different granularity.** Architectural insight: the 4 transformer layers are SHARED across K=7 breaths. Different tasks per breath = 7-fold gradient conflict. v81 makes every breath do the SAME job — coarse-to-fine refinement of a multi-list DAG representation. IB clustering on Pythia embeddings yields a 32-leaf codebook tree (4 ops × ~8 sub-clusters). Each breath emits 4 parallel lists — ops, types_path, args1, args2 — with each breath filling/refining ONE list more than the previous. Multi-head WaistController (4 heads, one per list). Full 4-mask training enforced. v81 prod fires now with BATCH=4, FIXED_LEN=256 (post-perf-tuning) — ETA ~5h to step 2000.

**The lesson the empirical curriculum keeps re-establishing:** "train on what you evaluate on" — but verify it BEFORE chasing metrics. Lookahead leaks between training and eval can silently invalidate hundreds of dollars of CE iterations. Always run a masking audit BEFORE celebrating CE drops. v45's regularization stack, v55's per-breath specialization, v79's causal masks, v80's format-ladder, v81's four-mask training are all expressions of the same principle — match the training signal to the target geometry, verify there are no leaks, then the architecture does its job.

**Architectural triad preserved across all paradigms.** Each successive paradigm (multi-cycle → rep-space-thinking → DAG → multi-list parallel) keeps the breathing + π-cycled-RoPE + WaistController triad and reshapes only the supervision interface. The codebook (V78_HEAD_CODEBOOK_N=32 now, was 12) now natively aligns with the IB cluster tree leaves — internal representation and supervision targets share the same indices.

A 127M model that breathes, refines coarse-to-fine across 7 cycles, emits an integer-encoded DAG, and hands off to SymPy. ~7 months to December 25.

---

## 15. The Breakthrough: v98 Sudoku Validates the Paradigm (May 29, 2026)

After v82-v97 plateaued at 0-1.7% on GSM8K (17 architectural variants), we made a strategic pivot to test the breathing paradigm on a constraint-satisfaction problem with explicit factor structure: **Sudoku**.

**The result: v98 reaches 79.0% puzzle accuracy on easy Sudoku (n=200), 97.65% cell accuracy, 87M parameters, 4 hours of training on one AMD 7900 XTX.**

### Why this matters

The 17 GSM8K failures were not architectural — they were **reading comprehension at Pythia-410M scale**. Sudoku isolates constraint propagation from natural language parsing. The breathing transformer's mechanism (iterative attention through shared weights, refined across K=20 breaths) is exactly the right substrate for joint inference on a structured probabilistic graph.

### The framing: breathing transformer = learned approximate BP on a factor graph

```
Variable nodes (81):  one per Sudoku cell, soft distribution over 9 digits
Factor nodes (27):    9 row AllDifferent + 9 column AllDifferent + 9 box AllDifferent
Per-head attention masks: ENCODE the factor topology (row/col/box per head bucket)
K breaths:            K rounds of loopy belief propagation
Per-breath supervision: forces monotonic refinement toward joint MAP
```

This is not a metaphor. It's the literal computation. Transformer attention with softmax = one step of modern Hopfield energy descent (Ramsauer et al., 2020). K breaths of shared-weight attention = K steps of approximate joint MAP inference. The factor topology baked into attention masks IS the structural inductive bias.

### The empirical signature: the constraint energy curve

The most important measurement in the project. K-sweep at K ∈ {1, 3, 5, 8, 12, 15, 18, 20}, n=200 per difficulty:

```
K    easy puzzle  easy cell  medium puzzle  medium cell  hard puzzle  hard cell  avg energy (easy)
 1     0.0%        82.1%       0.0%           69.1%        0.0%         63.6%      21.0
 3    10.0%        91.9%       0.0%           76.6%        0.0%         70.0%       7.2
 5    33.5%        94.8%       0.0%           79.8%        0.0%         72.9%       3.5
 8    56.0%        96.4%       1.0%           81.3%        0.0%         74.3%       1.8
12    72.5%        97.3%       2.5%           82.4%        0.0%         75.3%       1.1
15    75.0%        97.5%       5.5%           82.8%        0.0%         75.8%       0.86
18    77.0%        97.6%       6.0%           83.2%        0.0%         76.0%       0.75
20    79.0%        97.65%      6.5%           83.33%       0.0%         76.16%      0.71
```

The constraint energy decays **geometrically at characteristic rate ~0.5× per ~3 K** — exactly what loopy BP predicts on a factor graph with cycles. This is the mathematical signature of the underlying inference operation, distinct from any phenomenological metric.

### Per-breath convergence diagnostic

For each of K=20 breaths, we measure Δₖ = average number of cells whose argmax prediction changed between breaths k-1 and k:

```
              B1      B5      B10     B15     B19
easy:     13.05 →  1.17 →  0.20 →  0.09 →  0.05  ← CONVERGED to BP fixed point
medium:   19.45 →  3.08 →  0.97 →  0.44 →  0.45  ← Δ ≈ 0.4 floor — still settling
hard:     22.92 →  3.77 →  1.03 →  0.51 →  0.39  ← Δ ≈ 0.4 floor — still settling
```

Higher difficulty = longer mixing time, exactly as BP theory predicts on graphs with longer cycles. K=20 is enough for easy puzzles; medium/hard need more breaths than our training budget allowed.

### The 7-orders-of-magnitude correlation

At medium cell accuracy of 83.3%, independent-cell prediction baseline is 0.833⁸¹ ≈ 3×10⁻⁷. Observed puzzle accuracy: 6.5%. **Ratio: 2×10⁵ above independent baseline.** The model's per-cell errors are correlated within constraint cliques — the model is solving **STRUCTURES**, not classifying independent cells. This is the empirical proof that joint MAP inference is what the architecture computes.

### The six-component recipe (validated)

The architectural ingredients that made v98 work, generalizable to any factor graph task:

1. **Per-breath supervision** (weighted CE on per-iteration logits)
2. **Iterative prefill with shared weights** (K passes through L layers)
3. **Structured inductive bias** (per-head attention masks encoding factor topology)
4. **Aligned init** (embed rows = output codebook rows — the v98 unlock)
5. **Per-breath markers** (orthogonal additive embeddings separating gradients without specializing tasks)
6. **No feedback loops** (no consolidation table, no positive amplification — Bombe elimination, not generation)

### v99: generalizing to arbitrary factor graphs — and discovering the limits

If breathing = BP on factor graphs, the architecture should transfer to any structured inference problem. **v99 (May 29):** the same breathing-transformer mechanism (Pythia-410M L0-L3, K=10 breaths due to AMD JIT capacity, per-problem dynamic attention masks, moment-matching constraint energy) trained on synthetic arithmetic factor graphs.

**Result: 9% cell accuracy on easy, flat across K=1 to K=10, flat across DAG depths 2 to 7.** Energy decays geometrically (4.7M at K=1 → 2.5M at K=2, similar to Sudoku's rate), but accuracy does NOT improve with more breaths.

**The model is doing BP — and converging to the wrong fixed point.** The moment-matching energy has a trivial low-energy attractor: uniform distributions for unobserved variables. The model finds this attractor and stays there. Energy descent IS working; the energy landscape just doesn't single out the gold solution.

### Musical keys: topology determines breathing rhythm

The v99 result sharpens a deeper insight. The breathing transformer's "instrument" is universal — Pythia layers + K iterations + masked attention. But each problem class is in a different **key**, and each key requires a different breathing **rhythm**:

| Problem class | Topology | Key | Right breathing rhythm |
|---|---|---|---|
| Sudoku | Loopy graph, symmetric AllDiff cliques | Cyclic | π-cycled rotation across K breaths |
| Arithmetic DAG (v99) | Tree, asymmetric functional constraints | Directional | Topological staging (one DAG layer per breath) |
| Verification | Forward + backward cycles | Cadence | Alternating direction breathing |
| Multi-modal | Mixed topology | Modulation | Breath rhythm shifts mid-sequence |

**v98's success and v99's failure are the same finding viewed from two angles.** Rotation breathing works for cyclic key (Sudoku); it doesn't work for directional key (DAGs). Same instrument, wrong scale. The piano doesn't know what key the sheet music is in.

### v100: the directional key

For tree-shaped factor graphs (arithmetic, GSM8K-after-parsing), the right breathing pattern is **topological staging**:

```
Breath 0: model can only "see" observed leaves
Breath 1: leaves + depth-1 factor outputs visible
Breath 2: + depth-2 visible
...
Breath D: full DAG visible including query
```

The mask GROWS across breaths. Information has to be EARNED by waiting for predecessor breaths. Forces sequential propagation along the DAG's natural order.

Plus three v98-derived unlocks v99 was missing:
- **Aligned init** for the 100-way domain codebook (state_embed = digit_codebook)
- **Hard head specialization** (heads 0-3 add, 4-7 sub, 8-11 mul, 12-15 div) — drop the soft factor-type embedding
- **Factor-execute auxiliary loss** — direct supervision on factor node hidden states, not just CE on final variable predictions

And one v99-specific fix:
- **Replace moment-matching with KL on convolved distributions** — eliminate the uniform attractor

### The paper

`paper/outline.md` — "The Shape of Thought: Iterative Reasoning Through Learned Energy Descent on Factor Graphs". 

The paper structure expanded:
- **§6.1 Sudoku** (cyclic key, validated): rotational breathing → exponential energy decay → 79% puzzle accuracy
- **§6.2 v99 Arithmetic DAGs** (directional key, NEGATIVE result): rotational breathing → energy decay but accuracy stays at chance → uniform-attractor failure mode
- **§6.3 v100 Arithmetic DAGs with matched rhythm** (TBD): topological staging → expected accuracy lift on the same task

The framing becomes richer than "Sudoku works." It's "**breathing transformers are general substrates for energy descent, but each problem class is in a different topological key, and each key requires a breathing rhythm matched to its symmetry**."

The architecture's principles — per-breath diversity, commitment through compression, energy-based training, iterative convergence — transfer to any domain where the breathing rhythm matches the topology's key signature.

A 87M-parameter model that performs joint inference on a factor graph through K iterations of shared-weight attention. The Shape of Thought is learned approximate belief propagation — when the breathing is in the right key.

---

## 16. The JPEG Codec Mental Model

If the musical keys framework tells you what RHYTHM to breathe in, the JPEG codec analogy tells you what COMPRESSION to apply each breath. Together: rhythm × compression = the architecture.

Each breath is a learned compression codec. The four-step design structure mirrors JPEG/MP3:

| Codec step | Breathing transformer | What it does |
|---|---|---|
| **Transform** | Attention layers rotate into task-relevant basis (π-cycled RoPE for cyclic key, factor-aligned masking for directional key) | Change basis so energy concentrates in few coordinates |
| **Quantize** | Waist projection 1024d → 512d | Deliberately destroy unimportant coordinates |
| **Encode** | Notebook carries compressed state to next breath | Persist the survivors across iterations |
| **Psychoacoustic model** | Next-breath CE loss is the learned model of "what to preserve" | Determine what to throw away based on what the consumer needs |

### Implementation status — honest accounting (May 30, 2026)

The four-step codec is the **design vision**. Different architecture variants implement different subsets:

| Variant | Transform | Quantize (waist) | Encode (carry) | Psychoacoustic (CE) | Notes |
|---|---|---|---|---|---|
| v54-v95 (WaistController paradigm) | ✓ | ✓ (1024→512) | ✓ | ✓ | Full 4-step codec; had AR decode through WaistController |
| **v98 Sudoku** | ✓ | **✗** | ✓ (delta_gate) | ✓ | No AR decode, residual stays at 1024d through all breaths |
| **v99 Factor graph** | ✓ | **✗** | ✓ (delta_gate) | ✓ | Inherited no-waist design from v98 |
| **v100 Factor graph (directional)** | ✓ | **✗** | ✓ (delta_gate) | ✓ | Same; topological staging is the basis rotation |

**v98 and v100 are 3-of-4 codec architectures.** The quantize step was dropped when v98 removed AR decode (the WaistController had served both compression and decode roles). v100 inherited this. The compression that the codec framework describes is not currently in our best architectures.

This may be load-bearing accuracy we're leaving on the table. Open architectural question: would adding back the 1024→512→1024 waist per breath improve v100's accuracy on factor graphs? Hypothesis: yes, because the codec framework predicts that the lossy step IS the commitment mechanism. Without it, every breath is a residual addition rather than a refinement to commitment.

**v101 (planned ablation):** add per-breath waist projection to v100. Compare against current v100. If v101 > v100, the codec framework's quantize step is load-bearing for the directional key as well.

### The psychoacoustic model insight

In MP3, the encoder uses a perceptual model of human hearing to decide which frequencies the listener won't notice are missing. The model is hand-designed from psychoacoustic research — humans don't hear frequencies above 20 kHz, can't distinguish overlapping tones at certain ratios, etc.

In the breathing transformer, the analog is **"what does the next breath need?"** — and unlike MP3, this isn't hand-designed. The model learns it implicitly through end-to-end training. The next breath's CE loss is the gradient signal that says "this information was important; that wasn't." Information that doesn't survive the waist compression is information the model has learned the consumer (next breath) doesn't need.

### Why transformers needed a codec

The architectural insight from v68/v69 work: **transformers don't naturally compress; residual connections add information**. A standard L-layer transformer with residuals preserves and expands; it doesn't deliberately lose. Compression requires an explicit lossy step, and the lossy step IS the commitment mechanism.

The waist (1024d → 512d) is that explicit lossy step. Without it, every breath would be "another forward pass that adds more residual" rather than "a refinement that commits to a hypothesis." The compression FORCES commitment.

### The codec applies to both keys

- **Cyclic key (Sudoku):** Transform = π-cycled rotation, each breath sees the constraint graph from a different angle. The same codec runs every breath, with rotational diversity.
- **Directional key (v100 DAGs):** Transform = topological staging, each breath has access to a different depth slice. The codec runs at progressively deeper layers of the DAG.

The "key" determines the basis transformation. The codec structure (transform→quantize→encode→psychoacoustic-model) is universal.

### Connection to diffusion

The JPEG codec view also connects breathing transformers to diffusion models:
- Diffusion: noise schedule controls what level of detail the model processes at each step
- Breathing: waist compression controls what information survives between steps
- Both implement coarse-to-fine refinement through learned commitment

The breathing transformer is a diffusion process where the "noise" is principled compression, and the "denoising" is iterative refinement through belief propagation in the right key.

---

## 17. The ODE Integrator: A Deeper Mathematical Identity

The musical keys (rhythm) and JPEG codec (compression) are mental models. **The breathing transformer LITERALLY IS a learned ODE integrator for energy descent on factor graphs.** This isn't analogy — it's the underlying math.

### The dynamical system

```
State:     x(t) ∈ ℝ^(N_vars × D_domain)   soft distribution per variable
Energy:    E(x) = Σ_factors constraint_violation(factor, x)
Dynamics:  dx/dt = -∇E(x)                  follow the energy gradient
Fixed pt:  x* where ∇E(x*) = 0             energy minimum = solution
```

Each breath = one integration step. K breaths = K steps. The convergence plateau IS the integrator reaching its fixed point.

### The RK4-stages interpretation is literal

The 4 transformer layers within ONE breath are the 4 stages of a Runge-Kutta-like integrator:

```
RK4:                          Breathing transformer:
  k1 = f(x_n)                    h1 = Layer_0(x_n)
  k2 = f(x_n + h*k1/2)           h2 = Layer_1(h1)
  k3 = f(x_n + h*k2/2)           h3 = Layer_2(h2)
  k4 = f(x_n + h*k3)             h4 = Layer_3(h3)
  x_{n+1} = x_n + h*Σ(k)/6      x_{n+1} = x_n + delta_gate*(h4-x_n)
```

The residual stream is the running sum across stages. The delta_gate is the step size. Higher-order integration in 4 stages per breath.

### Attention IS one Hopfield energy descent step (Ramsauer 2020)

The mathematical foundation: modern Hopfield networks showed that
```
softmax(x · K^T) · V = one step of energy descent on
  E(x) = -log Σ_i exp(x · k_i)
```

So each transformer layer literally computes one energy gradient step. K breaths × 4 layers = 4K energy descent iterations on the (learned) Hopfield energy landscape.

Training aligns the implicit Hopfield energy with the explicit factor graph constraint energy. That IS what "learned approximate belief propagation" means precisely.

### The calibration head is a Dopri5-style error estimator

Adaptive ODE solvers (Dopri5, Cash-Karp) compute two estimates of x_{n+1} at different orders and use their difference as a local error estimate. If the error is small, the step is accepted; otherwise step size is reduced.

Our calibration head plays the same role:
- Output: P(solution correct | current state)
- Adaptive K: when calibration crosses threshold, stop integrating
- Easy puzzles converge at K=5; hard puzzles run to K=20

### Three vocabularies, one mathematical object

```
Computer science    →  Physics            →  Machine learning
──────────────────     ───────────────       ──────────────────────
ODE integrator         Energy descent        Approximate BP
RK4 stages             Higher-order grad     Multi-layer attn per breath
Step size              Damping coefficient   delta_gate
Error estimator        Stability check       Calibration head
Fixed point            Equilibrium           Joint MAP solution
Adaptive step          Adaptive damping      Adaptive K via calibration
```

These aren't three separate theories. They're three vocabularies for the same mathematical object: **a learned approximate iterative solver for joint MAP inference on a factor graph, structured as an ODE integrator with energy-based dynamics**.

---

## 18. The Two-Phase Architecture

The ODE framing makes one thing crisp: **comprehension and inference are different computational regimes that should use different model sizes**.

### Phase 1 — Comprehension (NL → factor graph)

```
Input:   "Janet has 16 eggs. She eats 3 for breakfast.
          She bakes muffins with 4. Sells remaining at $2 each."

Output:  A factor graph (structured tensor)

  Variables: [v0=16, v1=3, v2=4, v3=2, v4=?, v5=?, v6=?]
  Factors:   [f0: sub(v0,v1)→v4,
              f1: sub(v4,v2)→v5,
              f2: mul(v5,v3)→v6]
  Observed:  [v0, v1, v2, v3]
  Query:     v6
```

This is a **language task**. No math. No iteration. Just identifying variables, operations, and dependencies in prose. **Hard for small models** (the 410M GSM8K failure was here). **Easy for large models** (Haiku/Sonnet can do it).

### Phase 2 — Inference (factor graph → answer)

```
Input:   The factor graph from Phase 1

Process: Breathing transformer as ODE solver
  
  dx/dt = -∇E(x, constraints)
  
  Breath 0: x₀ = initial (observed one-hot, unknowns uniform)
  Breath 1: x₁ = x₀ - η∇E(x₀)        → constraints decrease
  Breath 2: x₂ = x₁ - η∇E(x₁)        → further refinement
  ...
  Breath K: x_K ≈ fixed point         → all constraints satisfied

Output:  argmax of each variable's distribution → numerical answer
  v4=13, v5=9, v6=18 → answer: 18
```

This is a **computational task**. No language. Pure constraint propagation via energy descent. **Easy for small models** (the 87M breathing transformer hits 79% on Sudoku). **Wasteful for large models** (they'd just compute directly).

### Why the phases must be separate

The two phases use opposite strengths:

| Phase | Strength needed | Iteration | Model size |
|---|---|---|---|
| 1 (comprehension) | Language understanding | One-shot | Large (1B+) |
| 2 (inference) | Constraint propagation | Iterative (K breaths) | Small (87M) |

Forcing both into one model means one will be wrong-sized for what it has to do. The 410M GSM8K failure was Phase 1 being undersized for comprehension; v98 Sudoku worked because Phase 1 is trivial (the grid IS the factor graph) and the architecture is sized correctly for Phase 2.

### The full GSM8K system

```
┌─────────────────────────────────────────────────────────┐
│ Phase 1: COMPREHENSION (large model, one-shot)           │
│                                                          │
│ "Janet has 16 eggs..."                                   │
│         ↓                                                │
│ NL Parser (Haiku / fine-tuned classifier / rules)        │
│         ↓                                                │
│ Factor Graph: variables, factors, observed, query        │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌──────────────────────┴──────────────────────────────────┐
│ Phase 2: INFERENCE (small model, iterative, on device)   │
│                                                          │
│ Factor Graph                                             │
│         ↓                                                │
│ Breathing Transformer (87M, 377MB)                       │
│ dx/dt = -∇E(x, constraints)                             │
│ K breaths of ODE integration                            │
│         ↓                                                │
│ Converged variable assignments → answer: 18              │
└─────────────────────────────────────────────────────────┘
```

Phase 1 runs ONCE per problem (possibly in the cloud, possibly offline). Phase 2 runs on device, iteratively, as many breaths as needed. The expensive comprehension happens once. The cheap inference iterates until confident.

### What this means for the future

For Sudoku: Phase 1 is trivial (grid → factor graph is a one-line conversion). v98 demonstrated that Phase 2 alone is enough. **79% puzzle accuracy from 87M parameters on a constraint satisfaction problem with explicit factor structure.**

For GSM8K: Phase 1 is the bottleneck. The breathing transformer didn't fail on the reasoning — it failed on the reading. The v100 series demonstrated that GIVEN a factor graph, the architecture can reach 50%+ accuracy with the right key. Building the comprehension front-end (Phase 1) is the path to GSM8K, not redesigning the inference engine.

The Shape of Thought is Stage 2.
A small model that solves by breathing.
An ODE integrator that descends the energy landscape.
One step per breath. Fixed point = solution.
