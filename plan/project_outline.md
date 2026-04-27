# Mycelium: Differentiable Decomposition Through Breathing

## The Thesis

Mathematical reasoning is decomposition. Breaking a hard problem into easy pieces that compose. The intelligence isn't in computing "160 - 63" — any calculator does that. The intelligence is in knowing to break "Jamie had 160 cookies, gave 63 away, then got 20 more" into three pieces, recognizing each piece as a known pattern, and chaining the results.

Small language models can't decompose. They know individual math facts but can't chain them. Llama 3.2 1B scores 70% on single arithmetic operations and 0% on chained operations. The knowledge is there. The decomposition ability is not.

We give the model a differentiable decomposition engine. Each thinking cycle matches one pattern, computes one result, and passes it forward. The model breathes: EXPAND (think in natural language) → COLLAPSE (compress to 64 floats) → repeat. Each breath takes one bite of the problem. The bites accumulate into a solution.

---

## Three Principles

### 1. Decomposition Is Everything

The model doesn't need to be smart. It needs to break hard problems into easy pieces. Each cycle handles one piece. The pieces are easy — Llama can compute "160 - 63" in a single forward pass. The difficulty was never arithmetic. It was knowing WHICH arithmetic to do, in WHICH order, on WHICH numbers.

```
Hard:    "Jamie had 160 cookies, gave 63 away, then got 20 more. How many?"
         Chained reasoning. 0% accuracy for Llama alone.

Easy pieces:
  Piece 1: "Jamie had 160 cookies" → extract 160 (parsing pattern)
  Piece 2: "gave 63 away" → compute 160 - 63 = 97 (subtraction pattern)
  Piece 3: "got 20 more" → compute 97 + 20 = 117 (addition pattern)

Each piece: >90% accuracy for Llama with the right attention.
The decomposition turns an impossible problem into three trivial ones.
```

### 2. You Can't Decompose Without Patterns

Decomposition means "find a known pattern that matches this part of the problem." You can't break something into pieces if you don't know what the pieces look like. The pattern library is the prerequisite for decomposition.

"Half as many clips in May" is one pattern: fractional_relationship(previous, 0.5). "Bought 5 at $3 each" is one pattern: unit_price(quantity=5, price=3). Each pattern maps natural language to a mathematical operation. Without the pattern, the words are just words.

### 3. Match the Largest Pattern That Fits

The Panama hat principle. "Panama hat" is one concept (a fashion item), not "Panama" (a country) + "hat" (headwear). Use the largest pattern that matches. Don't decompose more than necessary.

```
Over-decomposed (too many small patterns):
  "half" → divide_by_2
  "as many" → reference_previous
  "clips" → object_type
  "in May" → time_period
  Four tiny patterns. Four cycles. Fragile recombination.

Right-sized (one large pattern per cycle):
  "half as many clips in May" → fractional_relationship(previous, 0.5)
  One pattern. One cycle. Clean.
```

Larger patterns = fewer cycles = less error compounding = higher accuracy. The model should take the BIGGEST bite it can per cycle. Small bites are the fallback when no large pattern matches.

---

## The Architecture

### The Breathing Loop

Each cycle: EXPAND fully in natural language (Llama forward pass with atom-modified attention), then COLLAPSE through a compression bottleneck (perceiver squeezes to 64 floats). The rhythm repeats until the model is confident.

```
Cycle 1: EXPAND  → Llama reads problem, atoms steer attention to first pattern
         COLLAPSE → perceiver compresses understanding to 64-float page
         Result: "Jamie had 160 cookies." → page encodes 160

Cycle 2: EXPAND  → Llama re-reads with different atoms (informed by page 1)
                    Text injection: "Step 1 result: 160"
                    Atoms steer attention to the second pattern
         COLLAPSE → perceiver compresses to page 2
         Result: "He gave away 63. 160 - 63 = 97 cookies remaining."

Cycle 3: EXPAND  → Llama re-reads with different atoms (informed by pages 1-2)
                    Text injection: "Step 1 result: 160\nStep 2 result: 97"
                    Atoms steer attention to the third pattern
         COLLAPSE → perceiver compresses to page 3
         Result: "He got 20 more. 97 + 20 = 117 cookies remaining."

Answer: 117 (from page delta → answer head)
```

Each cycle breathes once. Full natural language expansion (the inhale). Ruthless compression to 64 floats (the exhale). The cycles coordinate through accumulated pages and injected text results.

### The Pattern Library: 64 LoRA Atoms

64 rank-6 Low-Rank Adaptation atoms. Each atom modifies how Llama pays attention. Each atom is one "primary color" of attention modification. The atoms BLEND continuously — the hypernetwork outputs 64 independent scales and the attention modification is the weighted sum.

```
64 atoms with continuous scales [-1.0 to +1.0]:
  The space of possible attention patterns is a continuous 64-dimensional manifold.
  Every point is a unique pattern recognizer.
  The hypernetwork navigates to the right point per cycle.

Atom 0 + Atom 5 = a COMPOSITE pattern neither produces alone
  (like red + blue = purple — a new color from blending primaries)
```

The atoms are initialized as a Fourier basis — orthogonal by construction. Low-frequency atoms match LARGE patterns (broad attention, whole-phrase recognition). High-frequency atoms match SMALL patterns (fine-grained, single-word focus). The Fourier structure creates a natural multi-scale pattern library.

```
Low-frequency atoms (0-15):   broad patterns
  "half as many in May" as ONE unit
  "bought N at $M each" as ONE unit
  Match whole phrases. Few cycles needed.

High-frequency atoms (48-63): fine patterns
  "48" → a number
  "half" → divide by 2
  Match individual tokens. More cycles needed.

Easy problems: mostly low-frequency atoms (big bites, few cycles)
Hard problems: mix of all frequencies (big and small bites, more cycles)
```

### The Compression Bottleneck: 64-Float Pages

The perceiver reads ALL 16 layers of Llama's hidden states and compresses everything into 64 floating point numbers. This bottleneck is intentional — it forces the model to commit to WHAT MATTERS at each cycle. You can't dump your entire understanding into 64 numbers. You can capture one or two key insights per breath.

Pages accumulate cycle by cycle. Each page is appended, never overwritten. The residual gate blends new and old information with frequency-aware persistence — low-frequency dimensions (coarse info) preserved, high-frequency dimensions (fine details) updated each cycle.

Pi-harmonic encoding gives each page dimension a frequency identity. The page is a structured representation where dimension 0 carries the coarsest information and dimension 63 carries the finest detail.

### The Page Delta: Isolating Each Cycle's Contribution

The answer head reads the PAGE DELTA (page_k - page_{k-1}), not the raw page. This isolates what each cycle ADDED — its new computation — from what previous cycles stored. Without the delta, the answer head reads persisted information from earlier cycles and produces copies instead of new computations.

```
Without delta: answer head reads page_2 → predicts cycle 1's number (COPY)
With delta:    answer head reads (page_2 - page_1) → predicts cycle 2's computation (CORRECT)

Result: 5% → 89% accuracy on cycle 2. One line of code.
```

### Text Injection: Speaking Llama's Language

Previous cycle results are injected as ACTUAL TEXT into Llama's input. Not pseudo-tokens. Not continuous vectors. Text — because that's Llama's native format.

```
Cycle 3 input: "Step 1 result: 160\nStep 2 result: 97\nJamie had 160 cookies..."
```

The page carries compressed understanding through the bottleneck. The text injection carries specific numbers in the format Llama was pretrained on. Two complementary channels: compressed context (pages) and explicit intermediates (text).

### Dynamic Detach: The Frontier Advances

Once a cycle reaches 90% accuracy, it GRADUATES — its page is detached from the computation graph. Graduated cycles are protected from later cycles' gradient.

```python
for cycle in range(num_cycles):
    page = model.think_one_pass(...)
    
    if per_cycle_accuracy[cycle] > 0.90:
        state_pages.append(page.detach())  # graduated
    else:
        state_pages.append(page)           # still learning
```

The frontier advances automatically. Each cycle masters its patterns and locks in. The next cycle builds on a stable foundation. The pattern library grows one cycle at a time.

### The Cycle Message: Direct Signal

A 16-float message alongside the 64-float page that bypasses the compression bottleneck. The page carries compressed multi-layer understanding (the formal record). The message carries a direct signal from Llama's last layer (the memo). Complementary channels — the page says "what was understood," the message says "what was important."

### The Hypernetwork: Pattern Matcher

The 10M-parameter hypernetwork reads accumulated pages and messages, then navigates the 64-dimensional atom space to the right blend. It IS the pattern matcher:

```
Reads:    pages (what previous cycles found)
Outputs:  64 atom scales (which attention pattern to apply)

"Given what I know so far, what pattern should I match next?"
```

---

## The Training Recipe

### Per-Cycle Intermediate Targets

Each cycle gets its OWN target — the intermediate result for that step. Not the final answer. The ONE number this cycle should produce.

```
Cycle 1 target: 160    (extract first quantity)
Cycle 2 target: 97     (compute 160 - 63)
Cycle 3 target: 117    (compute 97 + 20)
```

### Natural Sentence Generation

Each cycle generates a FULL NATURAL SENTENCE containing the computation. Not a bare number. Not an isolated equation. A sentence — because that's how Llama breathes.

```
Cycle 1: "Jamie had 160 cookies."
Cycle 2: "He gave away 63 cookies. 160 - 63 = 97 cookies remaining."
Cycle 3: "Then he got 20 more. 97 + 20 = 117 cookies total."
```

### Hybrid Loss

Generation loss POWERS learning (1000x gradient to atoms). Answer head loss SHAPES correctness. Different weights per cycle — parsing cycles use generation-dominant loss, computation cycles use answer-head-dominant loss.

```
Cycle 1 (parsing):      1.0 × gen_loss + 0.5 × head_loss
Cycle 2+ (computation): 0.1 × gen_loss + 5.0 × head_loss
```

### Curriculum: Building the Pattern Library

Each level adds patterns to the atom library and trains one more cycle:

```
L3 (1-step):    basic patterns (single operations in context)
L4 (2-step):    composition patterns (two chained operations)
L4.5 (3-step):  longer chains (three operations, mixed types)
L4.7 (4-step):  complex patterns (percentages, rates, multi-quantity)
L5 (GSM8K):     combines all learned patterns (3-8 steps)
```

Dynamic detach protects graduated cycles at each level. The frontier advances through the curriculum automatically.

---

## Results

```
Task              Baseline    Breathing      Improvement
──────────────────────────────────────────────────────────
L3 (1-step)       18.8%       94.5%          5.0x
L4 (2-step)       6.0%        91.0%          15.2x
L4.5 (3-step)     ???         training now   —
GSM8K             4-5%        target >30%    —
MATH-500          ???         July 1 target  —
```

Key breakthroughs in order:
1. Per-cycle targets — made cycles non-redundant (17% → per-cycle learning)
2. Hybrid generation loss — 1000x gradient to atoms (0.0002 → 0.24)
3. Page delta — broke the copying ceiling (5% → 89% on cycle 2)
4. Text injection — bridged format gap (Llama reads numbers as text)
5. Natural sentences — allowed full expansion (the model breathes)
6. Dynamic detach — protected graduated cycles (no oscillation)

---

## The Complete System

```
LLAMA (frozen, 1.2B):        Comprehends — reads with atom-modified attention
ATOMS (64 × rank 6, 82M):   Pattern library — Fourier-initialized, continuous blending
HYPERNETWORK (10M):          Pattern matcher — navigates 64-dim atom space
PERCEIVER (105M):            Compressor — 16 layers → 64 floats (the exhale)
PAGES (64 floats × N):      Memory — accumulated insights, one per breath
PAGE DELTA:                  Isolator — each cycle's NEW contribution only
TEXT INJECTION:              Bridge — previous results in Llama's native format
MESSAGES (16 floats):        Memo — direct uncompressed signal between cycles
RESIDUAL GATE:               Preserver — coarse persists, fine updates
ANSWER HEAD (100K):          Verifier — digit prediction from page delta
CONFIDENCE HEAD (79K):       Judge — when to stop breathing
DYNAMIC DETACH:              Curriculum — graduated cycles are protected

Total: ~1.43B | Trainable: ~203M (14.2%) | Frozen: 1.23B (85.8%)
```

---

## The Vision

Decomposition is everything. The patterns are the vocabulary. The breathing loop is the syntax. The pages are the memory. The answer is the last exhale.

A small model learns not to be smarter, but to break hard problems into pieces it can already solve. The pattern library grows through curriculum. The breathing loop applies patterns one at a time. The pages carry results between breaths. The dynamic detach advances the frontier.

The model breathes. Each breath takes one bite. The bites accumulate. The patterns compose. The answer falls out.
