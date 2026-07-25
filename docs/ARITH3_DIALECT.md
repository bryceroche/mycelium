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

## THE TIGHTENED LAW (Bryce's pass, 2026-07-25 — rulings 1/1a/2/3)
- **k=1 is illegal** in mod/fdiv: *a multiplicative identity in a mod/fdiv
  factor carries no constraint — a factor that constrains nothing is not a
  degenerate relation but the absence of one, and the dialect does not
  permit vacuous factors.* (Census: 0/1.7M banked, 0/3,800 failures —
  zero retroactive cost; guards future emitters, the organ first.)
- **k=0 is illegal** likewise and more so: the divide-by-zero panic class
  relocates entirely from runtime to compile time (one wild specimen ever
  — E06's lone customer; the solver can no longer reach the panic from
  legal IR).
- **pct is a pure relation, result-less BY DESIGN** (affirmative): *pct
  constrains extant variables and produces no value; any "result" of a
  percentage computation is one of its args, bound elsewhere.* Flip-check
  ran against the eight: zero asked-for orphans — SEALED. (Texture note,
  not law: 5/8 specimens carry pct-only-bound non-query intermediates —
  lint candidate.)
- **loc is required on every node** (phase-in: mandatory for all NEW
  emitters; the parser's spans-from-birth satisfies it). Derived nodes
  carry `loc: derived(parents…)` with a NON-EMPTY parent set — *every
  node's provenance chain grounds out in source spans, transitively; a
  node that traces to nothing compiled from nothing.* (Verifier check
  E12-provenance.)
- **The canonicalizer's provenance contract** (written before its first
  line): constant-folding MERGES loc sets, never drops them — implied by
  false-merge-zero, now stated as law the builder inherits.

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
