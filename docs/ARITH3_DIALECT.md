# THE ARITH3 DIALECT (v0 — de-facto extraction 2026-07-25)

The campaign's intermediate representation, written as law for the first
time. Seeded mechanically from 73,209 banked-good graphs (~650k factors);
**SPEC-TIGHTEN flags mark where the de-facto habit may be looser than the
intent — Bryce holds the tightening pass.** The verifier
(`mycelium/arith3_verifier.py`) enforces exactly this document; a change
here is a change there, same transaction.

## Node types (ftype)

| ftype | required fields | domains (de-facto) |
|---|---|---|
| `given` | var, value | var ∈ [0,24); value ∈ [0,999] (observed max 988) |
| `rel` | op, args, result | op ∈ {add, mul, sub}; **arity == 2 invariant** (235,737/235,737); args/result ∈ [0,24) |
| `mod` | var, k, result | k ∈ [2,9] observed; **law: k ≥ 1; k == 0 is the division-by-zero panic caught at compile time** (SPEC-TIGHTEN: is k=1 legal? de-facto never emitted) |
| `fdiv` | var, k, result | k ∈ [2,19] observed; same k-law as mod |
| `sel` | sel, args, result | sel ∈ {smaller, larger, even, odd} |
| `pct` | args, p | p carried; **no result field in any banked pct** (SPEC-TIGHTEN: is that the law or a habit?) |
| `macro` | name + per-name | FRAC_OF: a ∈ [1,9], k ∈ [2,12], x, result; OP_APPLY: op ∈ {add,sub}, k1,k2 ≥ 1, x, y, result |

## TIGHTEN-pass riders (awaiting Bryce)
- `loc` as a REQUIRED attribute on every node (source span provenance — #69).
- k=1 legality (empty specimen set — pure intent).
- pct's result-less form: law or habit?

## Loop law (the state-borne clause, #70)
Breath specialization, where it exists, is state-borne, never
weight-borne; per-breath adapters are a fence-gated last resort.

## Graph laws
- Non-empty (≥1 factor). All var indices ∈ [0, 24). Values digit-representable (≤ 3 digits).
- Macros expand before the solver (mg2; the key grades in primitives — constitutional, not verifier-enforced).
- The verifier constrains FORM only. It says "not legal arith3," never "doesn't match the problem." Semantic faithfulness belongs to the vote, the panel, and the key.

## Verdict class
A verifier rejection is **malformed-graph** (+ code + slot) — the third
death, earlier and cheaper than solver-unsat or budget-abstain.
