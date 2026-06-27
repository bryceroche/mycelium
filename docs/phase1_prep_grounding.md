# Phase 1 — prep grounding (read before designing the schema)

**Status:** prep layer (2026-06-26). Feeds the fresh schema-design step.
**Big find:** we already *ran* a Phase-1 attempt (the v80–v85 / "classifier Phase-1" era,
pre-v98-pivot). The artifacts are in `.cache/` and they reframe the problem. Study these
*before* designing — don't start from a blank page or the web survey alone.

## 1. The prior schema (`.cache/gsm8k_factor_graphs_*.jsonl`)
An **arithmetic-DAG** factor graph per GSM8K problem. Record fields:
- `n_vars`, `n_factors`, `domain` = [0, 10000]
- `factor_types`: ops, vocab = **{mul, add, div, sub} ONLY** (mul 6190 / add 4776 / div 3695 / sub 2808 across train) — pure arithmetic, no comparison/equality.
- `factor_args`: `[a, b, result]` triples — membership + **directionality** (last elem = output var).
- `observed_mask` + `observed_values`: the givens (input quantities).
- `query_idx`: the answer variable. `var_descriptions`: NL phrase per variable. `gold_answer`, `question`, `gsm8k_idx`.
- Companion `gsm8k_phase1_classifier_*.jsonl`: per-variable parse annotations (text phrase → op/leaf_id/observed/value).

## 2. THE REFRAME (the load-bearing insight): GSM8K factor graphs are FORWARD-EXECUTABLE DAGs, not search-CSPs
The prior generator was an **LLM-parser** (`raw_response` + `parsed` in the rejected files),
and it **rejected on `answer_mismatch`** — it *executed* the DAG (givens → arithmetic →
query) and kept it only if the computed value equaled gold. Consequences:
- **Accepted graphs are execution-VERIFIED-correct parses** (~88% of GSM8K — train 4432/5000, val 899/1000, test 186/200). Rejection = the parse computed the wrong answer (~12%).
- **"Solving" a GSM8K graph is *forward arithmetic evaluation*** — no GAC, no backtracking, no search, no deducer. It's a calculator. So **for GSM8K the entire difficulty is CONSTRUCTION (parsing); the solving jaw is `eval()`.**
- => The factor-graph SOLVING machinery (deducer, search tier) **adds no value on forward-arithmetic** (GSM8K). It earns its keep only where solving needs genuine **unknowns / constraints / search** — i.e. **algebra (systems of equations), MATH-500**, not GSM8K-style forward word problems. (This is the scout's "kill condition," sharpened to its core.)

## 3. What the prior attempt actually showed (correcting the "GSM8K failed" memory)
- **Construction via LLM WORKS** (~88%, execution-verified). The bridge is feasible.
- The documented failure (breathing transformer ~0–1.7% on GSM8K) was the **breathing transformer as the parser/solver** hitting the reading-comprehension wall — NOT a failure of the factor-graph approach. The LLM-parse + execute pipeline was the working part; it just wasn't the *breathing transformer* doing it.

## 4. Registry mapping
- The 4 GSM8K ops **already map onto the existing `cage` predicate** (add/sub/mul/div with a result) — GSM8K needs **no new relations** to *represent*; and the "solve" is forward eval, so it needs no CSP at all.
- The relations that WOULD need adding are for the *interesting* (search-requiring) regime: `equals` / `linear_equation` (systems with unknowns), `compare_gt/lt` (inequalities — needs new ordering-propagator machinery in the core), and MathWorld's `transfer/ratio/part_whole` if we adopt that schema.

## 5. The value question — now concretely testable on existing data (the real probe)
Since the accepted graphs execute to gold by construction, the sharp test is **value over
LLM-direct**: modern LLMs solve GSM8K directly at ~90%+, but make *arithmetic* slips.
**Does LLM-parse → exact-eval beat LLM-direct by removing the arithmetic errors?** This is
runnable *now* on the 186 verified test graphs (execute them = 100% by construction; the real
comparison is parse-quality-on-fresh-problems vs LLM-direct). The architecture/min-weights and
end-to-end-differentiable questions sit on top of that.

## 6. Worked example (from the data)
> "For every 12 cans you recycle you receive \$0.50, and for every 5 kg of newspapers \$1.50.
> If your family collected 144 cans and 20 kg, how much money?"
- 13 vars (e.g. cans-per-group=12, cents-per-can-group=50, cans-collected=144, …),
- 6 factors: `div(144, 12)→can_groups`, `mul(can_groups, 50)→cents_from_cans`, … `div(total_cents, 100)→total_dollars`,
- observed = the 7 input quantities; query = total_dollars; gold = 12.0.
A clean forward DAG. *This is the worked mapping the schema design starts from.*

## Open questions for the FRESH schema-design step
1. **Adopt/refine the prior arithmetic-DAG schema, or switch to MathWorld's container-relation graph?** The arithmetic-DAG is execution-proven on GSM8K but is procedural (forward eval); MathWorld's relational form is more declarative (closer to a CSP) and may generalize better to algebra. Evaluate both against the *interesting* (search) regime, not just GSM8K.
2. **Where does solving stop being `eval()` and start needing the engine?** Pin the problem band (algebra / multiple unknowns / MATH-500) where the deducer + search tier actually earn their keep — that's where the schema must be a real factor graph, not a DAG.
3. **The value probe first:** does parse→exact-solve beat LLM-direct anywhere? If not, the two-phase bet is in question before we build a parser.
