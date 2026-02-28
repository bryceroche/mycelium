# Section X: Methods, Infrastructure, and the Road Not (Yet) Taken

## X.1 The Data Problem: 33GB from 7,000 Problems

The first engineering challenge was survival. Extracting full attention patterns from a 7B parameter model across 7,000 MATH problems with `output_attentions=True` generates approximately 33GB of dense floating-point tensors. A single A10G GPU has 23GB of VRAM; the training VM has 192GB of system RAM. Naively accumulating attention outputs crashes the machine before processing 200 problems.

### The S3 Daemon

We solved this with a streaming daemon that ships attention data to S3 as it's generated, maintaining a fixed memory footprint regardless of dataset size. The daemon watches a local staging directory, compresses completed chunks (one per problem), uploads them to `s3://mycelium-data/`, and deletes the local copy. The VM never holds more than ~500MB of attention data at any moment, yet the full 33GB corpus accumulates safely in S3.

This pattern — generate locally, stream to object storage, process later — recurs throughout the pipeline. Every expensive extraction (IAF, JSD boundaries, heartbeat analysis, IB clustering) follows the same architecture: a Lambda MapReduce over S3 chunks, with each Lambda invocation processing one chunk independently.

### Lambda MapReduce on S3

AWS Lambda functions with 1-10GB memory and 300-second timeouts process individual IAF chunks in parallel. A coordinator script invokes one Lambda per chunk, collects results, and aggregates. This pattern handled:

- **IAF extraction**: Per-head, per-token attention-to-input fractions across all generation tokens
- **Heartbeat analysis**: Thresholding and run-length encoding of IAF traces to count computation pulses
- **Y-label extraction**: Parsing sympy AST root operators from CoT mathematical expressions
- **IB step aggregation**: Collecting and deduplicating 41,787 computation steps across 5,850 problems
- **C2/C3 training data construction**: Assembling multi-label and expression-extraction pairs

Processing 117 chunks in parallel completes in 20-45 seconds wall-clock time. The same work sequentially on a single machine takes 2-3 hours. Lambda's per-invocation pricing makes this cheaper than maintaining a persistent compute instance for batch processing.

### Practical Note on Large Files

A recurring failure mode throughout development: AI coding assistants (including the ones helping build this pipeline) would freeze when attempting to open or print large JSON files (>50MB). The solution was disciplined: never `cat` a large file; always inspect structure first with `head -c 1000` or targeted Python one-liners that load and summarize without printing contents. This seemingly trivial observation cost multiple hours of debugging time across the project.

## X.2 Attention Extraction: What the Teacher Reveals

### HuggingFace Attention Access

Modern transformer implementations in HuggingFace expose per-layer, per-head attention matrices via `output_attentions=True`. For a model with L layers, H heads, and sequence length S, this produces L × H matrices of dimension S × S — the complete record of what attended to what, at every layer, for every token.

The critical architectural decision: we use HuggingFace's native attention extraction rather than custom hooks or activation patching. This ensures compatibility across model families (Llama, Mistral, Qwen) and versions, at the cost of extracting ALL attention rather than surgically targeting specific layers. The 33GB data volume is a direct consequence of this choice, justified by the S3 daemon making storage tractable and Lambda MapReduce making processing parallel.

### vLLM Where Possible

For inference-only tasks (generating CoT solutions, running the student pipeline), we use vLLM for its significantly higher throughput. vLLM's PagedAttention and continuous batching achieve 3-5x speedup over naive HuggingFace generation. However, vLLM does not expose raw attention matrices, so attention extraction still requires HuggingFace. The pipeline bifurcates: vLLM for fast generation, HuggingFace for attention analysis.

### GPU Vectorization vs Python Loops

Early attention analysis code used Python loops over attention matrices — iterating over layers, heads, and token positions to compute per-head statistics. Processing one problem took ~45 seconds. Rewriting with vectorized PyTorch operations (broadcasting, einsum, masked operations) reduced this to ~0.3 seconds per problem — a 150x speedup. The lesson was consistent: any operation that touches attention tensors must be GPU-vectorized. Python loops over attention matrices are never acceptable, even for "quick analysis scripts."

## X.3 The Signals We Collect

The teacher model's attention patterns contain multiple distinct signals, each encoding different aspects of the reasoning process.

### Input Attention Fraction (IAF)

For each generation token t, IAF measures what fraction of attention goes to input tokens versus previously generated tokens. Formally:

$$\text{IAF}(t, h) = \frac{\sum_{i \in \text{input}} A_{t,i}^{(h)}}{\sum_{j} A_{t,j}^{(h)}}$$

where $A^{(h)}$ is the attention matrix for head h. High IAF indicates the model is reading the problem (grounding in input). Low IAF indicates internal computation (reasoning from previously generated context).

IAF is the foundation signal. It separates reading from reasoning — the dual phases hidden in every forward pass. The binary alternation between high-IAF and low-IAF states produces the heartbeat.

### The Heartbeat: Two Phases of Attention

The IAF trace for a single computing head, viewed as a time series over generation tokens, exhibits binary switching between two states:

- **Systole** (IAF ≈ 1.0): The model reaches out to the input, grounding the next computation in problem data
- **Diastole** (IAF ≈ 0.0): The model turns inward, computing from previously generated context

This is not sinusoidal oscillation — early experiments with FFT confirmed the signal is a telegraph wave, not a sine wave. The transitions are sharp, the states are binary, and the pulse widths vary. Run-length encoding with a threshold of 0.5 and minimum run length of 5 tokens reliably counts computation steps.

The initial hypothesis was that FFT would extract the dominant frequency of oscillation, corresponding to the number of computation steps. Visualization revealed otherwise: the signal is bimodal and bursty, not periodic. The attention head is not emitting a sine wave — it is switching between two states, like a telegraph key tapping out Morse code. The information is in the pulse count and timing, not in a dominant frequency.

Calibration across MATH difficulty levels confirms the signal encodes computational complexity:

| Level | Mean Heartbeats |
|-------|----------------|
| 1     | 3.8            |
| 2     | 5.1            |
| 3     | 7.0            |
| 4     | 9.7            |
| 5     | 14.8           |

The heartbeat count serves as an auxiliary training signal for the operation classifier (C2), providing a problem-level estimate of computational complexity that regularizes multi-label prediction via a dual-head architecture with joint loss: BCE for operation labels plus weighted MSE for heartbeat count.

### Jensen-Shannon Divergence (JSD) Boundaries

JSD between adjacent windows of attention distributions identifies points where the model's attention pattern changes significantly — corresponding to boundaries between reasoning steps in the chain of thought. A sliding window of size w computes:

$$\text{JSD}(t) = \text{JSD}(A_{t-w:t} \| A_{t:t+w})$$

Peaks in JSD correspond to step transitions. This was our initial approach to segmentation before discovering (through three failed C1 training attempts) that position-based segmentation is low-information when operands overlap across steps.

JSD remains valuable for ground truth step counting and for validating heartbeat-derived step counts against an independent signal.

### Attention Entropy

The entropy of each head's attention distribution measures focus versus diffusion:

$$H(t, h) = -\sum_j A_{t,j}^{(h)} \log A_{t,j}^{(h)}$$

High entropy indicates broad, uncertain attention (the model is searching). Low entropy indicates sharp focus (the model has found what it needs). Entropy over time reveals when the model transitions from exploration to extraction — potentially useful for identifying the moment C3 should "lock on" to operands. This signal has been extracted but not yet integrated into the training pipeline.

### Connectivity

The number of tokens receiving significant attention (above a threshold) from each head at each generation step. Related to but distinct from entropy: a head can attend to exactly 2 tokens with equal weight (low connectivity, moderate entropy) or attend to 2 tokens with 95/5 split (low connectivity, low entropy). Connectivity may correlate with operation arity — binary operations (ADD) should show connectivity ≈ 2, while function evaluations (f(x,y,z)) should show connectivity ≈ 3-4. This hypothesis remains untested.

### Received Attention

The total attention each input token accumulates across all generation steps:

$$R(i) = \sum_t \sum_h A_{t,i}^{(h)}$$

Tokens with high received attention are "important" — they were attended to repeatedly during reasoning. This was the basis for C1's relevance scoring, which failed because received attention smears across operations (a token used in step 1 AND step 3 gets high received attention but for conflicting reasons). Received attention is informative at the problem level but ambiguous at the step level.

## X.4 Information Bottleneck Template Discovery

### The Algorithm

The Information Bottleneck (IB) method discovers operation templates by finding clusters in embedding space that preserve information about computational function. Given:

- **X**: Embeddings of CoT computation steps (Qwen-0.5B, 896 dimensions)
- **Y**: The primary sympy operator for each step (the root of the expression AST)

IB finds a clustering T of X that minimizes I(X; T) (compression) while maximizing I(T; Y) (informativeness about what the operation does):

$$\min_T I(X; T) - \beta \cdot I(T; Y)$$

The parameter β controls the tradeoff. Low β produces few coarse clusters. High β produces many fine clusters. β-annealing sweeps from low to high, and **phase transitions** — sudden jumps in I(T; Y) — reveal the natural cluster boundaries in the data.

### The Critical Role of Y

Five iterations of clustering preceded the successful IB run, each producing increasingly sophisticated but fundamentally flawed taxonomies. The root cause, identified after substantial debugging: none of the "IB" implementations actually used a target variable Y. They were performing hierarchical k-means on embeddings — geometric clustering that groups steps by what the text looks like, not by what the computation does.

Without Y, "48/2 = 24" (division) and "48 - 24 = 24" (subtraction) cluster together because the text is similar. With Y, they separate because the sympy operators differ (Mul vs Add). The target variable forces clusters to align with computational semantics.

The iterative debugging process illustrates a broader methodological point: IB without a target variable is not IB. The mathematical formulation requires two random variables (X and Y) and a compression variable (T). Clustering on X alone, regardless of how sophisticated the splitting criterion, is unsupervised clustering — a fundamentally different algorithm with different guarantees.

### Y Extraction via Sympy AST

The Y label for each step is extracted by parsing its mathematical expression through sympy and walking the abstract syntax tree:

```python
def extract_all_ops(expr):
    ops = set()
    for node in sympy.preorder_traversal(expr):
        if isinstance(node, sympy.Function):
            ops.add(type(node).__name__)
        elif isinstance(node, Pow):
            if node.exp == Rational(1, 2):   ops.add("sqrt")
            elif node.exp == 2:              ops.add("square")
            elif node.exp == -1:             ops.add("inverse")
            else:                            ops.add("pow_general")
    if not ops:
        ops.add(type(expr).__name__)
    return ops
```

The initial Y extraction used only the AST root operator, which produced a degenerate distribution: 53% Pow (because sympy represents division as a × b⁻¹, roots as x^(1/2), and squares as x² — all Pow internally). Splitting Pow into semantic subtypes (sqrt, square, inverse, cube, high_pow, frac_pow, nth_root, pow_general) provided the granularity needed for IB to discover meaningful clusters.

### Plateau Detection and the Elbow

β-annealing produces a bifurcation history: at each β value, one cluster splits into two, with an associated I(T; Y) gain. Plotting gain versus cluster count reveals:

- **Large gains**: Real phase transitions where operationally distinct groups separate (e.g., arithmetic splits from trigonometry)
- **Small gains**: Noise-driven splits that don't reflect meaningful computational differences

The natural template count lives at the **elbow** — where gains transition from meaningful to noise. Threshold-based analysis found:

| Gain Threshold | Cluster Count |
|---------------|---------------|
| ≥ 0.30        | 6             |
| ≥ 0.20        | 16            |
| ≥ 0.15        | 30            |
| ≥ 0.10        | 60            |

The 30-cluster solution at gain ≥ 0.15 was selected as the operating point, producing a taxonomy with sufficient granularity for the operation classifier while avoiding over-fragmentation.

### Surface Tension and Singleton Elimination

Small clusters (< min_size) that emerge during annealing are reabsorbed into their parent cluster — a "surface tension" mechanism that prevents proliferation of singleton or near-singleton templates that would provide too few training examples for the classifier. Setting min_size = 50 preserved rare but genuine operations (LOG: 441 steps, COMBINATORICS: 386 steps) while eliminating statistical artifacts.

### Embedding Choice: Why Not Ask Qwen to Classify

A natural question: why not simply prompt Qwen-0.5B (or a larger model) to classify each step's operation type? "Is this addition, multiplication, or division?"

The answer is philosophical and practical. Philosophically, Mycelium's thesis is that operation types are **discovered** from attention patterns, not pre-specified. Asking a model to classify presupposes the taxonomy; IB discovers it. Practically, LLM classification introduces its own error modes — hallucinated labels, inconsistent granularity, sensitivity to prompt phrasing. IB on embeddings is deterministic and reproducible.

The labels emerge from the data through information-theoretic optimization, not from a language model's interpretation of a prompt. This is the difference between asking a spectrometer to identify spectral lines and asking a human to name the colors they see.

### Text Embeddings vs Math-Only Embeddings

Early IB runs on full-text embeddings (including English prose surrounding mathematical expressions) consistently produced taxonomies dominated by topic similarity rather than operational similarity. Trigonometric steps clustered with other trigonometric steps regardless of operation (sin vs cos vs tan), because the surrounding text ("the triangle has angle...") dominated the embedding space.

Extracting only mathematical expressions before embedding — stripping English and retaining LaTeX and symbolic content — improved cluster coherence by forcing the embedding to represent operational structure rather than topical context. However, the dominant factor in template quality was the presence of Y labels in the IB objective, not the embedding modality.

## X.5 Distributed Training: 8 × A10G

### Distributed Data Parallel (DDP)

The C3 expression extractor trains on 28K+ examples using PyTorch's DistributedDataParallel across 8 A10G GPUs (g5.48xlarge instance, ~$16/hr). Each GPU processes a shard of the dataset with synchronized gradient updates. At 94-100% GPU utilization and 14-20GB memory per device, the 8-GPU configuration achieves near-linear scaling for the 0.5B parameter model.

### Training Considerations: EOS Tokens and Data Quality

Two critical training failures informed best practices:

**Missing EOS tokens.** The initial C3 training run produced valid-looking expression prefixes followed by degenerate repetition (e.g., `x+y_25+y_2+y_2...`). The model learned to predict expression tokens but never learned to stop, because the training data omitted explicit end-of-sequence tokens. Adding `tokenizer.eos_token` after every output expression resolved the issue immediately.

**Inconsistent output formats.** The first C3 training data mixed sympy-parseable expressions (`x**2 + 1`), implicit multiplication (`2x + 3`), LaTeX fragments (`\frac{3}{4}`), and trailing punctuation (`2m + 4.`). The model's output reflected this inconsistency — producing a chimeric format that was neither valid sympy nor valid LaTeX. Hard-filtering to only examples where `sympy.sympify()` succeeds — dropping 22% of the data — produced a smaller but consistent training set that the model could learn reliably.

### Dataset Contamination: The GSM8K Landmine

Three separate training runs were contaminated by GSM8K data when training scripts defaulted to the wrong dataset path. C6 (answer type classifier) trained on GSM8K, which is 100% integer answers — producing a model that could not classify fractions, radicals, or expressions. C2 trained on GSM8K with 6 arithmetic labels instead of the full MATH taxonomy of 15+ operation types.

The fix was engineering discipline: renaming legacy files with `_LEGACY_DO_NOT_USE_` prefixes, adding assertions to training scripts (`assert "math" in data_path.lower()`), and moving GSM8K data to an archive directory. Dataset contamination is a silent failure — metrics look reasonable on the wrong distribution.

## X.6 The Wave Function Collapse Framework

### Measurement Order Matters

The central architectural insight — discovered through three failed attempts at input segmentation (C1) — is that the order of measurements determines the quality of information extracted.

The original pipeline measured position first: which tokens belong to which computation step? This produced overlapping, ambiguous assignments because the same token (e.g., "48") participates in multiple operations. In wave function collapse terms, position measurement on entangled tokens produces low-information results.

The revised pipeline measures type first: what operations does this problem require? This is high-information because operation types are discrete and distinguishable. Once the operation type is known, operand extraction becomes constrained and tractable.

The analogy to quantum measurement is more than metaphorical. In quantum mechanics, measuring spin before position yields different information than measuring position before spin — the order determines what collapses and what remains in superposition. In our pipeline, measuring operation type before token position yields clean classifications, while measuring token position before operation type yields overlapping blobs.

### Decoherence Threshold and MCTS

The decoherence threshold θ determines when the pipeline commits to a specific interpretation versus maintaining multiple hypotheses. At each stage:

- If confidence > θ: collapse to a single interpretation (deterministic execution)
- If confidence < θ: maintain multiple hypotheses (MCTS branching)

Monte Carlo Tree Search explores the space of interpretations: which operations, which operands, which dependencies. Each path through the tree is a complete hypothesis about the problem's computational structure. Sympy serves as the oracle — if a path produces a valid, computable expression, it survives; if not, it is pruned. The tree search is guided by model confidences (exploitation) and diversity of unexplored hypotheses (exploration).

This adaptive behavior means easy problems execute in a single pass (no branching, ~15ms) while hard problems explore multiple interpretations (full MCTS, ~150ms). The compute budget automatically scales with problem difficulty, analogous to System 1 vs System 2 thinking.

### Minimax Considerations

For adversarial or competition-style problems where the problem setter deliberately constructs misleading surface features, a minimax perspective may improve robustness: the pipeline should select the interpretation that performs best under the worst-case assumption about which templates are correct. This remains theoretical — the current MCTS uses expected-value optimization, not minimax — but suggests a direction for competition mathematics where problems are designed to exploit common misconceptions.

## X.7 On Overfitting: When Memorization Might Be a Virtue

A counterintuitive question arises in the Mycelium setting: is overfitting actually desirable?

Standard machine learning practice treats memorization as pathological — a model that memorizes training examples fails to generalize. But Mycelium's student models are not generalizing in the traditional sense. C2 is mapping natural language problem descriptions to a fixed taxonomy of ~30 operation types. C3 is extracting mathematical expressions from problem text given an operation hint. These are more like lookup tasks than generalization tasks.

Consider C3's job: given `[TEMPLATE: DIVIDE]` and a problem about selling clips, output `48 / 2`. This is closer to information retrieval than creative generation. If C3 memorized every (problem, template, expression) triple in the training set, it would perform perfectly on those problems. The question is whether the patterns generalize to unseen problems.

Our empirical observation: moderate overfitting (training loss significantly below validation loss) correlates with better sympy parse rates. The model needs to memorize the output FORMAT (always produce `a / b`, never `a ÷ b` or `\frac{a}{b}`) even if the specific numbers generalize. This suggests:

- **Low dropout** (0.05-0.1) for C2 and C3, since we want format memorization
- **Standard layer normalization** for training stability
- **Early stopping on sympy parse rate**, not on validation loss — because lower loss on the wrong format is worse than higher loss on the right format

The distinction between format memorization (desirable) and content memorization (mixed) suggests a two-phase training approach: first overfit to learn the output format with high learning rate, then fine-tune for content generalization with low learning rate and modest regularization.

## X.8 Sympy as the Execution Engine: Why Not a DSL

An early design consideration was whether to define a custom domain-specific language (DSL) for mathematical operations or to use sympy's existing symbolic algebra system.

Arguments for a DSL: potentially simpler parsing, guaranteed coverage of exactly the operations we need, no ambiguity from sympy's internal representation choices (like division as multiplication by inverse).

Arguments for sympy: mature symbolic computation engine with 15+ years of development, handles edge cases (division by zero, complex numbers, symbolic simplification) that a custom DSL would need to reimagine, extensive documentation and community support, and — critically — automatic verification. If sympy evaluates an expression and produces a result, that result is mathematically correct. A custom DSL would need its own verification layer.

We chose sympy. The representation quirks (Pow for everything, Mul for division) caused friction in Y-label extraction but are manageable engineering issues. The alternative — building and debugging a custom symbolic computation engine — would have consumed months of development time for uncertain benefit.

The principle: use the most mature, well-tested tool for execution, even if its interface is imperfect. Engineering effort is better spent on the novel components (attention extraction, IB discovery, specialist training) than on reinventing symbolic algebra.

## X.9 Ideas Explored and Deferred

The following ideas were explored during development and set aside — not because they are wrong, but because they are premature or because simpler approaches sufficed. They represent the frontier for future work.

### Frequency Tensor Analysis

Hypothesis: attention patterns across layers form a 3D tensor (layers × heads × positions) whose frequency decomposition reveals hierarchical reasoning structure. Low-frequency components correspond to global problem understanding; high-frequency components correspond to local computation steps. Status: conceptually compelling, not yet tested. The heartbeat analysis (1D signal from a single head) was a simplified version of this idea. The full tensor analysis awaits a validated single-head pipeline.

### Low-Resolution Pointers

Hypothesis: instead of predicting exact expressions, C3 could predict "pointers" to approximate regions of the input (e.g., "the number in the second sentence"), then extract exact tokens from those regions. This reduces the output vocabulary from all possible expressions to a small set of pointer combinations. Status: superseded by the full-text extraction approach where C3's own attention mechanism learns where to look without explicit pointer supervision.

### Rubato: Adaptive Computation Time

Named after the musical concept of "stolen time" — the idea that different parts of a problem deserve different amounts of computational attention. A rubato mechanism would allocate more transformer layers to complex sub-expressions and fewer to simple ones, analogous to adaptive computation time (ACT). Status: interesting but premature. The current fixed-depth architecture is sufficient for the operation types in MATH.

### Pi Normalization

Normalizing attention weights by π (dividing by π before softmax) as a form of temperature scaling, inspired by the observation that attention entropy distributions cluster around multiples of ln(π). Status: tested briefly, no measurable improvement. The observation about ln(π) clustering may have been coincidental.

### Gramian Symmetrization

Taking the Gram matrix of attention patterns (A^T A) to produce symmetric, positive-semidefinite representations that are invariant to token order. Useful for comparing attention patterns across problems of different lengths. Status: used informally for visualization, not integrated into the pipeline.

### Cross-Tokenizer Alignment

Different models (Llama, Mistral, Qwen) tokenize the same text differently, making attention pattern comparison across model families non-trivial. Cross-tokenizer alignment maps attention weights from one tokenization to another via character-level correspondence. Status: implemented for the cross-model validation study (Llama vs Mistral attention head specialization), not needed for the single-model Mycelium pipeline.

### Savitzky-Golay Smoothing

Applying Savitzky-Golay filters to IAF traces before heartbeat detection, preserving sharp transitions while removing high-frequency noise. Status: tested but unnecessary — the raw binary signal is clean enough for threshold-based detection. Smoothing actually degraded heartbeat counting by rounding off short computation bursts.

### Confidence-Weighted Reciprocal Rank Fusion (RRF)

Combining multiple attention signals (IAF, entropy, connectivity) via weighted RRF to produce a unified "computational importance" score per token. Status: not implemented. The signals serve different purposes and combining them into a single score loses the distinct information each provides.

### Merge Trees for Multi-Grain Segmentation

Using merge trees (from topological data analysis) to simultaneously represent segmentation at multiple granularities — coarse clause boundaries and fine sub-expression boundaries — in a single hierarchical structure. Status: conceptually elegant, rendered unnecessary by the decision to abandon segmentation entirely in favor of full-text operation detection.

### TCP Handshake for Model Communication

Modeling the C2→C3 interaction as a three-phase handshake: C2 proposes operations (SYN), C3 confirms extractability (SYN-ACK), C2 finalizes the operation set (ACK). This would allow C3's feedback to inform C2's final predictions — if C3 cannot extract an expression for a proposed operation, that operation is likely wrong. Status: elegant but adds a round-trip of inference latency. The current fire-and-forget architecture (C2 predicts, C3 extracts, sympy validates) achieves the same filtering effect through sympy rejection, without the additional model inference.
