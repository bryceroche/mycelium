# The Rotational Binding Bus — v5 Architecture & Diagnostic Contract

**Status:** design center of the architecture (Bryce's ruling, 2026-08-28:
"build everything around it"). Research lineage; the deployed stack
(conductor_v2) is untouched. Ledger entries 2026-08-28 carry the verdicts,
bars, and registrations; this doc is the standing contract.

## 1. Context & motivation

- **The breakthrough:** relational wiring between slots is not stored as
  static spatial positions (four position-probe designs died there) but as
  **phase rotations** in high-dimensional complex space
  (C^128 ≅ R^256, interleaved-real; FHRR).
- **Key empirical invariance:** the bus is the campaign's first
  **register-invariant** instrument — recovery holds across mint and wild,
  and v3 took the first outright wild crossover (`res` 72.0% vs the pointer
  incumbent's 69.2% on the 143 golds; wild res *exceeds* mint res — the
  strong form).
- **The move to v5:** one monolithic head → **4 dedicated role heads**
  (`arg1`, `arg2`, `res`, `op`) with **binding by construction**
  (structural-entry law: binding enters as structure, never as learned
  behavior).

## 2. The v5 architectural contract

The network does not learn how to bind; it only identifies content.
Binding is an architectural invariant it cannot violate.

| Layer | Responsibility |
|---|---|
| Neural (4 heads) | "Which entity is this?" → logits over codebook \|C\| |
| Codebook C | unit-L2 random complex codes (row norm 1; per-plane modulus varies — truth-maintained 2026-08-29), frozen |
| Structural (phasors) | "Where does it belong?" → fixed rotation e^{iθ_r}, frozen |
| Wire (superposition) | "Transmit relation" → Z_wire = Σ_r v_r ⊙ e^{iθ_r} |
| Cleanup (today) | cosine argmax vs codebook (`bind_read.py`) |
| Cleanup (registered future) | NL-Atlas Mahalanobis resolver — collisions become geometry |

- **Soft-pointer formulation:** head_r outputs logits ℓ_r ∈ R^|C|;
  p_r = softmax(ℓ_r); content vector v_r = Σ_k p_r(k)·c_k. The output space
  is bounded to the codebook hull; CE on ℓ_r is the native loss
  (v3's role-factored objective, now per-head).
- **Zero parameter interference:** four disjoint heads end hidden-capacity
  starvation (the v3 signature: `op` at .93 while args starved at ~.57).
  Precision (ledger addendum): with the native per-role CE there is literally
  zero cross-role gradient; any wire-level aux loss reaches all heads
  through the sum — parameters stay disjoint, gradient traffic is
  conditional on wire-loss.
- **The single-wire fence (NON-NEGOTIABLE):** the four bound vectors SUM
  onto one 256-float wire. Four separate outputs with no superposition is
  pointer heads in a trenchcoat — there is no bus without the wire.
- **Dimensionality convention:** D_real = 256 floats ≡ P_planes = 128
  complex FHRR channels. Code names these explicitly (D_real/P_planes;
  `complex_tensor.py` carries P = D//2).

## 3. The emergent diagnostic: phasor modulus A_r = ‖v_r‖

Codewords are unit-modulus, so a spread softmax suffers destructive phase
interference: ‖Σ_k p_r(k)·c_k‖ ≤ 1, with equality only at one-hot.
An uncertain head writes **dimmer** onto the wire.

- **Phase / amplitude separation:** phase θ_r encodes structural role
  (deterministic, frozen); amplitude ‖v_r‖ is an emergent, uncalibrated
  proxy for pointer concentration.
- **Why it may beat entropy (the honest subtlety):** given the FULL
  distribution p_r, A_r is deterministic — it adds nothing. The audit
  conditions on the scalar H(p_r): amplitude weights uncertainty by
  code-space geometry (mass split across *nearby* codes interferes less
  than mass split across *distant* codes), so A_r is a geometry-aware
  confidence scalar where entropy is geometry-blind. That difference is
  the entire hypothesis.
- **Pre-registered audit:** I(correct; A_r | H(p_r)) > 0 on banked reads.
- **THE GOODHART FENCE:** amplitude is strictly observational — NEVER
  supervised in any loss (a monitored signal in the loss learns concealment,
  not cure). Register with `diagnostic_register.py` before first use.
- **Downstream decoupling:** cleanup normalizes before resolution; the
  discrete pipeline never depends on amplitude.

## 4. Substrate decision rule (awaiting bindbus-joint)

Bars pinned 2026-08-28 before the fire:

- **Val ≥ 0.58 AND the wild res crossover survives (≥ parity with pointers)
  → v5 trains on the JOINT substrate** (the waist has plasticity to meet
  the rotational coordinates without the parse paying).
- **Otherwise → v5 trains FROZEN-PARSE** (the parse geometry is
  crystallized; the 4 soft-pointer heads train as a pure classification
  interface). Val < 0.57 additionally KILLS the joint line outright.
- Constitutional regardless of branch: **the trunk is frozen forever** —
  "frozen parse" refers to the trained head's parse parameters, never to
  the trunk, which no branch of this rule may touch.
