# Complex Planes — the addressing design (branch `complex-planes`, 2026-10-19)

**Authors:** Bryce + Claude (with Opus's review folded in). **Status:** design. Nothing has been trained, and no bars
have been pinned. The numbers below are zero-GPU toy reads of the addressing math alone (`scripts/toys/complex_addr_*.py`).

## 0. The thesis, credited part by part

Bryce: rebuild the transformer around complex planes. Binding (⊙) and unbinding (⊙ conj) are exact rotations. Composite
keys built from a hierarchy of planes give an effectively unlimited address space that can be searched without loss.

What each ingredient actually buys:

| Claim | Holds? | Where it really comes from |
|---|---|---|
| Exponentially many distinct addresses | **Yes** | The **digit system**: one register per level, each holding one of N symbols, gives N^L addresses. The same count holds for any discrete code. The phases do not create it. |
| Search through them is lossless | **Yes, in layout A** (below) | Cleanup snaps each register to an exact symbol. This is the neural → symbolic handoff in miniature. |
| Robust to an imperfect encoder | **Random noise: yes. Coherent noise: no.** | The phases' **noise margin**: 64 codes spread over 256 planes are nearly orthogonal, so random phase noise averages out. A lean toward a plausible sibling does NOT average out (§2). |
| Values attach to addresses exactly | **Yes** | **Binding**: value ⊙ address is stored, and ⊙ conj(address) returns the value exactly, with zero parameters. |
| "Orthogonality for free" | **No, approximate** | Random unit phasors in C^D overlap by about 1/√D, the same as random real vectors. Real space already has exponentially many nearly orthogonal directions (that is MLP superposition). |
| Many items in one vector | **No: capped** | Superposition crosstalk. The 10-16 clock read measured about 20–30 items per D = 256 register, for every key scheme. |

## 1. Four layouts under random phase noise (`complex_addr_layouts.py`)

D = 256 planes per register; 64 codes per level; 3 levels (262,144 addresses); 500 trials.

| Noise (rad) | **A: one register per level** | B: pure product + resonator | C: weighted prefix sum | D: 26 addresses in one vector |
|---|---|---|---|---|
| 0.0 | **1.000** | 0.096 | 0.992 | 0.964 |
| 0.5 | **1.000** | 0.040 | 0.966 | 0.952 |
| 1.0 | **1.000** | 0.008 | 0.726 | 0.646 |
| 1.5 | **1.000** | 0.000 | 0.242 | 0.230 |

- **B (a pure product L1⊙L2⊙L3 in one vector)** is a lock, not an index. Unbinding the root leaves L2⊙L3, which looks
  random, so there is no partial signal to search by. Reading it is a factorization problem. (The resonator here is a
  quick, untuned one, and 262k combinations at D = 256 is likely past its capacity.)
- **C (Gemini's prefix sum)** is searchable but leaky: the levels share one vector and crosstalk.
- **A** is exact. It searches level by level at 64 comparisons per level (192 in total, not 262,144). The price is
  WIDTH: L × D planes, linear in depth, against N^L addresses.
- **D** shows the storage cap is a separate limit. Address space is lossless; a memory holding many addresses in one
  vector is not.

## 2. Layout A under COHERENT noise (`complex_addr_sibling.py`; Opus's correction)

A real tree: 64 top codes × 8 children × 8 grandchildren = 4,096 valid paths. The encoder leans each level toward a
plausible sibling (another top code at level 1; a sibling under the same parent below) with strength α ~ U(0, αmax),
plus 0.5 rad of random noise. 2,000 trials.

| Sibling lean | independent snaps | greedy descent | **beam-2, joint score** | full joint read over valid paths |
|---|---|---|---|---|
| α ≤ 0.4 | 1.000 | 1.000 | 1.000 | 1.000 |
| α ≤ 0.6 | 0.582 | 0.582 | **0.811** | 0.806 |
| α ≤ 0.8 | 0.236 | 0.236 | **0.506** | 0.486 |
| α ≤ 1.0 | 0.136 | 0.136 | **0.302** | 0.303 |

**Readings:**
1. **Coherent error is a cliff, not a slope.** A level flips exactly when the lean passes the midpoint (α > 0.5); the
   independent column is (1 − p_flip)^3 to three decimals. More planes do not move the cliff, because the margin
   against a blend does not grow with D. The phases protect against random noise only.
2. **Redundancy is the only defense, and the tree supplies it.** A wrong snap at one level is out-voted when the levels
   below it only fit the true parent. Beam-2 with a joint score roughly doubles accuracy in the hard regimes. Greedy
   descent gains nothing over independent snaps: once the top snap is wrong it cannot be undone (the tree's old death).
3. **Beam-2 ≈ the full joint read.** Keeping two candidates per level captures the rescue; scoring every valid path
   adds nothing (the 10-16 tree race's beam-2 result, reproduced).
4. **Implication for the encoder:** what has to be trained is margin against SIBLINGS. That is the twin problem in
   address form: contrastive negatives drawn from the confusable sibling, not random negatives.

## 3. The addressing design

- **One plane group per level.** Each level of the hierarchy owns its own registers (a head group). Superposition is
  never used for the address itself.
- **The MLP is the per-level cleanup.** It soft-snaps each register toward its codebook, keeping the top 2 per level,
  not a hard argmax.
- **Read every register at the first look; the breaths refine.** Do NOT visit one level per breath (that is the
  six-way walk). "Coarse to fine" describes how confidence sharpens across breaths, not the order levels are visited.
  The final pick is held open until the levels agree under the tree's validity.
- **Superposition is reserved for the working set** (≤ ~20 items per register: the entities and quantities in play).
- **Values bind onto addresses** with ⊙ and unbind exactly; the cleanup restores what crosstalk erodes.

## 4. The complex transformer: where the planes go

| Part | Standard | Complex-plane version | Open question |
|---|---|---|---|
| Bridge in | Llama embedding, 2048 real | z = r·e^{iθ}, θ = W_θx, r = softplus(W_r x) | Keep the radius (two-channel law: angle = identity, radius = consolidation). |
| Position | RoPE | Llama already rotates 32 planes per head; ours is RoPE generalized to roles and entities | Share or separate position planes? |
| Score | q·k | Re(qᴴk) = Σ \|q\|\|k\| cos Δθ | Softmax on the real part. |
| Values | weighted sum | weighted phasor sum = superposition | Crosstalk grows with how many tokens a head mixes. |
| Bind / unbind | learned implicitly | ⊙ role / ⊙ conj(role), zero parameters | Which heads unbind which roles? |
| Norm | RMSNorm | normalize the radius, keep the phase | — |
| MLP | SwiGLU | **the cleanup memory** (soft snap to codebook, top 2) or modReLU | The deep surgery. A real nonlinearity applied separately to the real and imaginary parts breaks the phase algebra. |
| Out | linear | unbind by the role's conjugate → codebook match | Readable by construction. |

## 5. The toy ladder (bars to be pinned before any read)

- **Rung 0, synthetic:** toy v1's entity-tracking task (banked arms A / A4 / B / E) plus a NATIVE complex arm.
  Held-out combinations are the test.
- **Rung 1, GSM8K:** tag every token and bind related words. Gold comes mechanically from parse edges, the gold graph's
  quantity roles and the owner. The X-ray: unbind each token by each role and rank its true partner.
- **Controls:** (1) a real twin of equal real parameter count (C^D ≈ R^2D), the same data, learning-rate sweep and
  seeds; (2) a complex arm whose nonlinearity runs separately on the real and imaginary parts (does the phase algebra
  matter, or only the parameterization?).
- CPU only until a rung earns the card.
