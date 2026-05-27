"""v80 — regenerate ALL of L0-L5 with strict layered grammars (preserve L6 DAG).

v78c's L0-L4 collapsed to nearly identical entropy because they were all
free-flowing natural language. v80 forces a SMOOTH entropy ladder by giving
each layer a strict grammar that strips one specific kind of format
variability vs. the previous layer.

v80 v3 LOCK: shipped to prod training, killed at step 570. The
cross-checkpoint per-breath CE trajectory revealed a sharp bottleneck at L3:

    delta from step 50 → step 450 (smaller = layer learning slower)
    L0: 7.06 → 5.61  Δ=-1.45
    L1: 6.54 → 4.29  Δ=-2.25  (biggest drop — slot template memorizing)
    L2: 6.02 → 4.92  Δ=-1.10
    L3: 5.26 → 4.48  Δ=-0.79  ← THE BOTTLENECK
    L4: 4.52 → 3.61  Δ=-0.92
    L5: 4.33 → 3.05  Δ=-1.27
    L6: 3.77 → 2.38  Δ=-1.39

Diagnosis: L3 in v3 used unbounded snake_case verbs ("zola_red_hats",
"deduct_profit"). Inspection of 5 random v80 records found ZERO verb reuse
across problems — the model has to predict an essentially-unique identifier
per step per problem, blowing up the local vocab.

v80 v4 tried to fix this by bounding L3 to a fixed 50-verb list, but it
backfired: all the contraction collapsed into L2→L3 (cliff +30%) because
the bounded verbs lost their semantic content.

v80 v6 — compositional <OP_role>, codebook-aligned:

  L3 now uses compositional action names from a CLOSED 12-entry list
  aligned 1:1 with the model's per-head codebook (V78_HEAD_CODEBOOK_N=12).
  Each L3 action maps to a codebook cell which maps to an L4 OP:
      codebook cell k ↔ L3 verb #k ↔ L4 OP
      Codebook CLASSIFIES. L3 NAMES. L4 EXECUTES.

  LAYOUT — 4 ops × 3 roles each:
      ADD_total, ADD_combine, ADD_return,
      SUB_difference, SUB_remainder, SUB_cost,
      MUL_rate, MUL_groups, MUL_scale,
      DIV_split, DIV_rate, DIV_fraction.

  MOD/POW are dropped from L2/L4 too (measured frequency 5/14357 = 0.03%
  in v80 train data — cells would be undertrained). ADD/SUB/MUL/DIV cover
  99.97% of GSM8K.

  Cross-layer consistency: each L3 step's OP prefix MUST match the
  corresponding L4 step's OP. The L3→L4 transition then strips just the
  role suffix — one clean axis.

v80 v7 — thread the codebook through EVERY layer (L0, L1, L2, L3):

  v6's <OP_role> codebook cells were a supervision target ONLY at L3.
  Six of seven breaths had no signal pointing at the 12-cell codebook
  vocabulary. The hypothesis: when L0, L1, L2, L3 ALL contain the same
  <OP_role> tokens, the model's internal codebook representation will
  align with the 12-cell structure way more strongly than v6's single-
  layer signal.

  CHANGES (vs v6):

    L0:  After the fixed preamble and BEFORE the natural-language
         paraphrase, insert a sentence listing the UNIQUE <OP_role>s used:
         "The operations used are <X> and <Y>."     (2 ops)
         "The operations used are <X>, <Y>, and <Z>."  (3+ ops, Oxford)
         "The operations used are <X>."             (1 op)

    L1:  NEW slot key "OPS" between KNOWN and GOAL (6 keys total now):
         OPS: <X>, <Y>          (unique cells from L3, in step order)

    L2:  Replace "OP=<OP>" with "CELL=<OP_role>" (compositional).
         Each step's CELL token must match the corresponding L3 step's
         <OP_role> tag EXACTLY (positional, not set).

    L3:  unchanged from v6
    L4:  unchanged from v6 — still strips role to bare <OP>
    L5:  unchanged from v6
    L6:  unchanged from v6 — preserved upstream DAG

  Cross-layer rules (NEW positional/set checks):
    - set(L0 OP_roles) == set(L1 OPS) == set(unique L3 OP_roles)  (set eq)
    - L2 step k CELL token == L3 step k OP_role  (positional eq)
    - L3 step k OP prefix == L4 step k OP  (positional eq; carried from v6)

  4 of 7 layers (L0-L3) now contain codebook cells — the codebook is the
  central anchor every layer reinforces.

Strict grammars (each layer keeps the math, removes one freedom):

  L0 : loose paraphrase, fixed preamble
       "This is a {problem_type} problem involving {n} steps. {1-2 sent}"

  L1 : slot template, rigid keys
       TASK / STEPS / ENTITY / KNOWN / GOAL

  L2 : rigid step grammar, NO parentheticals
       "Step k: ... OP={op}. ARG={val}."  +  "ANSWER = {idx}"
       OP ∈ {ADD, SUB, MUL, DIV}   (v6: MOD/POW dropped)

  L3 : compositional <OP_role> action labels (v6 codebook-aligned)
       x_k := <OP_role>(args...)
       OP_role ∈ L3_OP_ROLES (12 entries; see constants)

  L4 : bounded OP-name function form
       x_k := <OP>(args...)   where OP ∈ {ADD,SUB,MUL,DIV}
       L3→L4 strips the role suffix (one axis).

  L5 : generic-variable equations, SINGLE-LINE, 1-INDEXED with underscore
       x_1 = expr ; x_2 = expr ; ... ; ANSWER = x_N

  L6 : single-line DAG (PRESERVED from existing v78c)
       x0 = expr ; x1 = expr ; answer = xN
       (zero-indexed, no underscore, answer lowercase)

Inputs:
  --src    v78c JSONL file with problem / gen_targets / answer / n_steps /
           layers={L0..L6}. We use ONLY problem / gen_targets / answer / n_steps
           and existing L6. We OVERWRITE L0..L5.

Outputs:
  --dst    JSONL with TOP-LEVEL `problem` + `answer` (matching the
           load_gsm8k_v77 loader), problem_type, n_steps, layers dict
           {L0..L6}, plus flat L0..L6 for inspection and passthrough
           fields (gen_targets, sympy_value, sympy_matches_gold).

Usage:
    # v7 smoke run: 100 problems
    .venv/bin/python scripts/v80_haiku_regenerate_all.py \\
        --src .cache/gsm8k_steps_v78c_train.jsonl \\
        --dst .cache/gsm8k_steps_v80_v7_smoke.jsonl \\
        --num 100 --concurrency 16

    # Full train + test (RUN AFTER SMOKE REVIEW)
    .venv/bin/python scripts/v80_haiku_regenerate_all.py \\
        --src .cache/gsm8k_steps_v78c_train.jsonl \\
        --dst .cache/gsm8k_steps_v80_train.jsonl \\
        --num 10000 --concurrency 16
    .venv/bin/python scripts/v80_haiku_regenerate_all.py \\
        --src .cache/gsm8k_steps_v78c_test.jsonl \\
        --dst .cache/gsm8k_steps_v80_test.jsonl \\
        --num 10000 --concurrency 16
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from anthropic import Anthropic
except ImportError:
    print("ERROR: anthropic SDK not installed. Run: .venv/bin/pip install anthropic", file=sys.stderr)
    sys.exit(1)

try:
    import sympy
    from sympy import sympify, Rational
except ImportError:
    print("ERROR: sympy not installed. Run: .venv/bin/pip install sympy", file=sys.stderr)
    sys.exit(1)


MODEL = "claude-haiku-4-5-20251001"
ANTHROPIC_KEY_FALLBACK = "/home/bryce/Desktop/keys/key1.txt"

_PRICE_IN_PER_MTOK = 1.0
_PRICE_OUT_PER_MTOK = 5.0


# ----------------------------------------------------------------------
# Closed vocabularies
# ----------------------------------------------------------------------
PROBLEM_TYPES = {
    "simple_arithmetic",
    "mixed_arithmetic",
    "percentages",
    "ratios",
    "fractions",
    "age_problem",
    "rate_problem",
    "geometry",
    "comparison",
    "distribution",
}

OPS = ("ADD", "SUB", "MUL", "DIV", "MOD", "POW")
OPS_SET = set(OPS)

# v6 codebook-aligned: L3 is `<OP_role>` from a CLOSED 12-entry list aligned
# 1:1 with the per-head codebook (V78_HEAD_CODEBOOK_N=12). Each L3 verb maps
# to a codebook cell:  codebook k ↔ L3 verb k ↔ L4 OP.
#     Codebook CLASSIFIES. L3 NAMES. L4 EXECUTES.
#
# Layout: 4 ops × 3 roles. MOD/POW dropped (measured frequency 5/14357 =
# 0.03% in v80 train) — wasted cells on rarely-used ops. ADD/SUB/MUL/DIV
# cover 99.97% of GSM8K. 3 roles per op cover common semantic patterns
# without exploding cardinality.
L3_OP_ROLES = [
    # ADD (3 cells)
    "ADD_total",       # sum to a running total
    "ADD_combine",     # combine groups/items
    "ADD_return",      # add back / restore
    # SUB (3 cells)
    "SUB_difference",  # compare two values
    "SUB_remainder",   # what's left after removal
    "SUB_cost",        # spend / pay / use up
    # MUL (3 cells)
    "MUL_rate",        # quantity × rate (hours × $/hr → cost)
    "MUL_groups",      # count × per-group (boxes × items/box)
    "MUL_scale",       # multiply by a factor (double, triple, ×percent)
    # DIV (3 cells)
    "DIV_split",       # divide equally among groups
    "DIV_rate",        # find per-unit rate
    "DIV_fraction",    # find fraction/percentage of total
]
assert len(L3_OP_ROLES) == 12, f"L3 op-roles must be 12 (codebook-aligned), got {len(L3_OP_ROLES)}"
assert len(set(L3_OP_ROLES)) == 12, "L3 op-roles must be unique"
L3_ACTION_SET = set(L3_OP_ROLES)
L3_OPS = ("ADD", "SUB", "MUL", "DIV")
L3_OPS_SET = set(L3_OPS)
for action in L3_OP_ROLES:
    assert "_" in action, f"L3 action {action!r} must have OP_role form"
    op, _role = action.split("_", 1)
    assert op in L3_OPS_SET, f"L3 action {action!r}: OP {op!r} not in {L3_OPS}"


# ----------------------------------------------------------------------
# The Haiku prompt.
# ----------------------------------------------------------------------
PROMPT_ALL_LAYERS = r"""You are filling in SIX strictly-templated supervision layers (L0, L1, L2, L3, L4, L5) for a math word problem. These layers train a neural net on a SMOOTH entropy ladder where each successive layer strips exactly one type of format variability.

The seventh layer (L6) is already given — your output for the earlier layers must be mathematically consistent with L6 (same arithmetic, same step count, same answer).

## Design rationale

GOAL: a smooth CE-loss ladder L0..L6 where each breath does a similar fraction of the remaining work. We measure CE-drop per layer and want roughly uniform %-drops, no cliffs.

## v3 LOCK trained to step 570 — diagnosis

Cross-checkpoint per-breath CE deltas (step 50 → step 450):
  L0: 7.06 → 5.61  Δ=-1.45
  L1: 6.54 → 4.29  Δ=-2.25  (biggest drop — slot template memorizing)
  L2: 6.02 → 4.92  Δ=-1.10
  L3: 5.26 → 4.48  Δ=-0.79  ← THE BOTTLENECK
  L4: 4.52 → 3.61  Δ=-0.92
  L5: 4.33 → 3.05  Δ=-1.27
  L6: 3.77 → 2.38  Δ=-1.39

Diagnosis: v3's L3 had per-problem-UNIQUE snake_case verbs ("deduct_profit",
"multiply_hamburgers"). ZERO verb reuse observed across 5 random records.
The model has to predict an essentially-unique identifier per step per
problem → vocab explosion → can't lock onto a pattern.

v4 attempt: bound L3 to a fixed 50-verb list. Backfired — all the
contraction collapsed into L2→L3 (cliff +30%) because the bounded verbs
lost their semantic content.

## v6 fix: compositional <OP_role>, codebook-aligned

L3 now uses a CLOSED 12-entry list of compositional `<OP_role>` action names.
The 12 entries align 1:1 with the model's 12-cell per-head codebook
(V78_HEAD_CODEBOOK_N=12). This unifies internal representation and supervision:

    codebook cell k ↔ L3 verb #k ↔ L4 OP
    The codebook CLASSIFIES. L3 NAMES. L4 EXECUTES.

LAYOUT: 4 ops × 3 roles each = 12. MOD/POW dropped (measured frequency
5/14357 = 0.03% in v80 train data; cells would be undertrained). ADD/SUB/MUL/
DIV cover 99.97% of GSM8K.

THE 12 LEGAL L3 ACTIONS:
    <ADD_total>       <ADD_combine>     <ADD_return>
    <SUB_difference>  <SUB_remainder>   <SUB_cost>
    <MUL_rate>        <MUL_groups>      <MUL_scale>
    <DIV_split>       <DIV_rate>        <DIV_fraction>

L2→L3 transition: NL action description → 12-entry compositional form.
L3→L4 transition: strip the role suffix → 4-entry bounded OP {ADD,SUB,MUL,DIV}.

THE PRINCIPLE: each transition strips EXACTLY ONE variability axis from the
previous layer. No bundled changes.

VOCABULARY CONTRACTION: the symbol vocabulary contracts MONOTONICALLY
L0 → L6. L3's 12-entry codebook-aligned vocab → L4's 4-entry bounded OP
vocabulary {ADD,SUB,MUL,DIV} → L5's pure arithmetic operators → L6's
identical operators with cleaner ANSWER case.

## v7 fix: thread the codebook through EVERY layer (L0, L1, L2, L3)

In v6 the codebook cells `<OP_role>` appeared ONLY at L3. Six of seven
breaths got no supervision signal pointing at the 12-cell codebook
vocabulary.

v7 makes the codebook the CENTRAL ANCHOR every layer reinforces. The same
`<OP_role>` tokens now appear in L0, L1, L2, AND L3 — 4 of 7 layers
contain codebook cells. When 4 layers' supervision contains the same
12-cell vocabulary at different abstraction levels, the model's internal
codebook representation aligns with the 12-cell structure way more
strongly than v6's single-layer signal.

L0 mentions the UNIQUE set of <OP_role>s used (as a sentence).
L1 lists them in a new OPS slot.
L2 has per-step CELL=<OP_role> tags that POSITIONALLY match L3.
L3 uses them as before.

Cross-layer constraint (NEW):
    set(L0 <OP_role>s) == set(L1 OPS) == set(unique L3 <OP_role>s)
    L2 step k CELL == L3 step k <OP_role>   (positional)
    L3 step k OP prefix == L4 step k OP      (positional; from v6)

THE SIX TEMPLATES YOU MUST FOLLOW EXACTLY
==========================================

L0 (loose paraphrase, fixed preamble + ops sentence + paraphrase) — v7:
    "This is a <PROBLEM_TYPE> problem involving <N> steps. The operations used are <OPS_SENTENCE>. <PARAPHRASE>"
  - <PROBLEM_TYPE> MUST be one of these EXACT strings:
      simple_arithmetic, mixed_arithmetic, percentages, ratios, fractions,
      age_problem, rate_problem, geometry, comparison, distribution
  - <N> = integer number of reasoning steps (matches the L6 DAG step count).
  - <OPS_SENTENCE> = list of UNIQUE <OP_role> cells used in L3 (set, not
    order-preserving — but each cell appears only once even if L3 uses it
    multiple times):
      1 op:    "<X>"
      2 ops:   "<X> and <Y>"
      3+ ops:  "<X>, <Y>, and <Z>"  (Oxford comma)
    Each <OP_role> must be from the 12-entry L3 list. The angle brackets
    `<` and `>` are LITERAL characters that must appear in the output.
    The sentence ends with a period: "The operations used are <X> and <Y>."
  - <PARAPHRASE> = 1-2 sentences in natural English summarising the problem
    and stating what we need to find. Must include the key numbers.

L1 (slot template, rigid keys) — v7 adds OPS as 5th line:
    TASK: <PROBLEM_TYPE>
    STEPS: <N>
    ENTITY: <ENTITY>
    KNOWN: <VAL> <UNIT>, <VAL> <UNIT>, ...
    OPS: <OP_role_1>, <OP_role_2>, ...
    GOAL: <TARGET> (<UNIT>)
  - EXACTLY these SIX lines, EXACTLY these keys, in this order
    (TASK, STEPS, ENTITY, KNOWN, OPS, GOAL).
  - TASK and STEPS values must match L0's values.
  - ENTITY = primary actor (name or definite noun like "the shopper").
  - KNOWN = comma-separated list of "VAL UNIT" tokens; if a value has a
    compound unit use "/" (e.g. "12 dollars/hour", "60 minutes/hour").
  - OPS = comma-separated list of UNIQUE <OP_role> cells used in L3 (same
    SET as L0's ops sentence). The angle brackets are LITERAL. Order: list
    cells in the order they first appear in L3 (no duplicates even if L3
    repeats a cell across steps).
    Example for L3 with [<MUL_rate>, <MUL_rate>, <SUB_difference>]:
      OPS: <MUL_rate>, <SUB_difference>
  - GOAL = "<short_target_phrase> (<unit>)" — the thing being asked for.
  - No extra keys, no commentary lines, no blank lines inside the block.

L2 (rigid step grammar, NO parentheticals) — v7 uses CELL=<OP_role>:
    Step 1: <SENTENCE>. CELL=<OP_role>. ARG=<VAL>.
    Step 2: <SENTENCE>. CELL=<OP_role>. ARG=<VAL>.
    ...
    Step N: <SENTENCE>. CELL=<OP_role>. ARG=<VAL>.
    ANSWER = <step_index>
  - One "Step k:" line per reasoning step, in order.
  - <SENTENCE> is a short imperative or declarative clause WITHOUT any "(...)"
    parentheses anywhere on the line. NO inline notes. NO commentary.
  - CELL=<OP_role> — the angle brackets `<` and `>` are LITERAL characters
    that MUST appear in the output. <OP_role> is one of the 12 codebook-
    aligned action names (same closed list as L3):
      <ADD_total>, <ADD_combine>, <ADD_return>,
      <SUB_difference>, <SUB_remainder>, <SUB_cost>,
      <MUL_rate>, <MUL_groups>, <MUL_scale>,
      <DIV_split>, <DIV_rate>, <DIV_fraction>.
    Exactly one CELL per step. The CELL token at step k MUST be IDENTICAL
    to the corresponding L3 step k's <OP_role> action name.
  - <VAL> is a plain decimal/integer numeric literal (e.g. 12, 0.5, 60).
    No units, no expressions.
  - Final line is LITERALLY "ANSWER = <k>" where <k> is the index (1..N)
    of the step whose output is the final answer.

L3 (compositional <OP_role> action labels — v6 CODEBOOK-ALIGNED):
    x_1 := <OP_role>(NUMBER_OR_VAR, ...)
    x_2 := <OP_role>(NUMBER_OR_VAR, ...)
    ...
    x_N := <OP_role>(NUMBER_OR_VAR, ...)
    ANSWER := x_K

  CRITICAL: The angle brackets `<` and `>` are LITERAL CHARACTERS of the L3
  format that MUST APPEAR in the output exactly as shown. Do NOT strip them.
  The compositional action name goes INSIDE the brackets, e.g. the literal
  output for the first step in the Weng example is `x_1 := <DIV_fraction>(50, 60)`.

  L3 ACTION must be EXACTLY one of these 12 strings (CLOSED VOCAB — aligned
  with the model's 12-cell per-head codebook):

    <ADD_total>        sum to a running total
    <ADD_combine>      combine groups / items
    <ADD_return>       add back / restore something taken away
    <SUB_difference>   compare two values to find their gap
    <SUB_remainder>    what's left after taking some away
    <SUB_cost>         spend / pay / use up part of a value
    <MUL_rate>         quantity × rate (hours × $/hr → cost)
    <MUL_groups>       count × items-per-group (boxes × items/box)
    <MUL_scale>        multiply by a factor (double, triple, ×percent)
    <DIV_split>        divide equally among groups (total / people)
    <DIV_rate>         find per-unit rate (distance / time → mph)
    <DIV_fraction>     find fraction/percentage of a total

  NO other action names. NO snake_case verbs of your own. NO action name
  outside this 12-entry list.

  CRITICAL CROSS-LAYER RULE: each L3 step's OP prefix MUST be IDENTICAL to
  the corresponding L4 step's OP. If L4 step 1 is <DIV>, then L3 step 1 must
  be one of {<DIV_split>, <DIV_rate>, <DIV_fraction>}. Disagreements fail
  validation. The OP also matches the L2 step's OP=<...> value.

  CHOOSING THE role: pick the role from the 3 options for that OP that best
  describes what the step OUTPUT semantically represents.

  - One `x_k := <OP_role>(args)` line per reasoning step, in order.
  - Args are comma-separated. Numeric literals OR earlier x_j references (j < k).
  - Final line is LITERALLY "ANSWER := x_<k>".

L4 (bounded OP-name function form):
    x_1 := <OP>(NUMBER_OR_VAR, ...)
    x_2 := <OP>(NUMBER_OR_VAR, ...)
    ...
    x_N := <OP>(NUMBER_OR_VAR, ...)
    ANSWER := x_K

  CRITICAL: angle brackets `<` and `>` are LITERAL FORMAT CHARACTERS in L4 too.
  Output them literally. Example: `x_1 := <DIV>(50, 60)` — keep the brackets.

  - One `x_k := <OP>(args)` line per reasoning step, in order.
  - <OP_k> MUST be one of the EXACT closed-vocab strings inside angle brackets:
      ADD, SUB, MUL, DIV
    (v6: MOD/POW removed — they are vanishingly rare in GSM8K. If the problem
    requires those, REWRITE it to use ADD/SUB/MUL/DIV instead.)
    NO other names. NO snake_case verbs. NO descriptive identifiers.
    Use the SAME OP as the matching L2 step (the OP=<...> value).
  - Args are comma-separated. They can be numeric literals OR earlier
    x_j references (j < k). NO inline parenthetical notes.
  - Final line is LITERALLY "ANSWER := x_<k>" where <k> is the index
    of the final-answer step.
  - L4 has the SAME structural skeleton as L3 (x_k := <NAME>(args)) but
    replaces L3's compositional <OP_role> labels with the bare OP. This is
    the bounded-vocab contraction step — strip the role, keep the OP.
  - REMINDER: L4 step k's OP MUST be the SAME OP that appears as the prefix
    of L3 step k's <OP_role> action name. If L3 step 1 is <DIV_fraction>, then
    L4 step 1 must be <DIV>.

L5 (generic-variable equations, SINGLE-LINE, ' ; ' separators):
    x_1 = <expr_1> ; x_2 = <expr_2> ; ... ; x_N = <expr_N> ; ANSWER = x_<k>
  - ENTIRE L5 is ONE single line — NO '\n' newlines anywhere.
  - Statements are separated by " ; " (space-semicolon-space).
  - Uses generic variable names x_1, x_2, x_3, ... (UNDERSCORE, 1-INDEXED).
    First variable is x_1, second is x_2, etc.
  - <expr_k> is a plain arithmetic expression with numeric literals,
    +-*/%, parentheses for grouping, and earlier x_j references (j < k).
  - The FINAL statement is LITERALLY "ANSWER = x_<k>" with ANSWER UPPERCASE.
  - L5 has the SAME equations as L4 (same arithmetic, same step count),
    just packed onto a single line with bare arithmetic operators (no
    "<OP>" function wrapper). L5→L6 strips the underscore + lowercases ANSWER.

KEY CONSTRAINTS
================

1. Numbers across L0..L5 MUST be consistent with L6's numeric literals.
2. <N> must equal the number of x-assignments in L6 (the L6 DAG has N steps).
3. The <PROBLEM_TYPE> in L0 and TASK in L1 MUST MATCH (same string).
4. NO inline parentheticals "(note...)" or commentary in L2/L5 or in
   L3/L4 args.
   Parentheses are allowed in L3/L4 as the call syntax "<NAME>(...)" and
   inside expressions in L5 as grouping (e.g., "(a + b) * c"), but never
   as inline notes.
5. KEEP LAYERS TIGHT: each layer one self-contained block, no preamble,
   no commentary.
6. Use plain decimal form for all numbers — no commas, no units inside
   equations, no "$".

OUTPUT FORMAT
=============

Output EXACTLY one JSON object, no preamble, no commentary, no markdown
fences. The JSON object has these EXACT keys:

  {
    "problem_type": "<one of the 10 problem_type strings>",
    "L0": "<full L0 single-line text>",
    "L1": "<full L1 block — newlines between key lines>",
    "L2": "<full L2 block — newlines between step lines>",
    "L3": "<full L3 block — newlines between step lines>",
    "L4": "<full L4 block — newlines between step lines>",
    "L5": "<full L5 single-line text — NO newlines, ';' separators>"
  }

Use real "\n" newline characters in the JSON string values for multi-line
layers (L1..L4). L0 and L5 are SINGLE LINES and must contain NO newlines.

EXAMPLE
=======

EXAMPLE INPUT
Problem: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Gold answer: 10
N steps: 2
L6 DAG: x0 = 50 / 60 ; x1 = x0 * 12 ; answer = x1

EXAMPLE OUTPUT
{"problem_type": "rate_problem", "L0": "This is a rate_problem problem involving 2 steps. The operations used are <DIV_fraction> and <MUL_rate>. Weng earns $12 an hour and worked 50 minutes; we need to find her earnings.", "L1": "TASK: rate_problem\nSTEPS: 2\nENTITY: Weng\nKNOWN: 12 dollars/hour, 50 minutes\nOPS: <DIV_fraction>, <MUL_rate>\nGOAL: earnings (dollars)", "L2": "Step 1: Convert minutes to hours. CELL=<DIV_fraction>. ARG=60.\nStep 2: Multiply hours by rate. CELL=<MUL_rate>. ARG=12.\nANSWER = 2", "L3": "x_1 := <DIV_fraction>(50, 60)\nx_2 := <MUL_rate>(x_1, 12)\nANSWER := x_2", "L4": "x_1 := <DIV>(50, 60)\nx_2 := <MUL>(x_1, 12)\nANSWER := x_2", "L5": "x_1 = 50 / 60 ; x_2 = x_1 * 12 ; ANSWER = x_2"}

Note in EXAMPLE OUTPUT how the same codebook cells (<DIV_fraction>, <MUL_rate>) thread through L0 (in the ops sentence), L1 (in OPS slot), L2 (in CELL= tags), and L3 (as action names). L4 strips the role to bare <OP>, and L5 strips the function wrapper to bare arithmetic operators. L3's `<DIV_fraction>` and L4's `<DIV>` share the same OP (DIV); L3's `<MUL_rate>` and L4's `<MUL>` share the same OP (MUL). The role suffix (`_fraction`, `_rate`) is one of the 3 roles available for that OP (DIV: split/rate/fraction; MUL: rate/groups/scale). L3→L4 strips the role; the OP is preserved.

EXAMPLE INPUT 2
Problem: Sam had 85 cookies. Sam doubled the collection, then gave 132 away. How many cookies does Sam have left?
Gold answer: 38
N steps: 2
L6 DAG: x0 = 85 * 2 ; x1 = x0 - 132 ; answer = x1

EXAMPLE OUTPUT 2
{"problem_type": "mixed_arithmetic", "L0": "This is a mixed_arithmetic problem involving 2 steps. The operations used are <MUL_scale> and <SUB_remainder>. Sam had 85 cookies, doubled them, then gave 132 away; find how many remain.", "L1": "TASK: mixed_arithmetic\nSTEPS: 2\nENTITY: Sam\nKNOWN: 85 cookies, 2 multiplier, 132 cookies\nOPS: <MUL_scale>, <SUB_remainder>\nGOAL: cookies_remaining (cookies)", "L2": "Step 1: Double the cookie count. CELL=<MUL_scale>. ARG=2.\nStep 2: Give some cookies away. CELL=<SUB_remainder>. ARG=132.\nANSWER = 2", "L3": "x_1 := <MUL_scale>(85, 2)\nx_2 := <SUB_remainder>(x_1, 132)\nANSWER := x_2", "L4": "x_1 := <MUL>(85, 2)\nx_2 := <SUB>(x_1, 132)\nANSWER := x_2", "L5": "x_1 = 85 * 2 ; x_2 = x_1 - 132 ; ANSWER = x_2"}

EXAMPLE INPUT 3
Problem: Caleb spent $10 on each of 4 ice cream pints. He also spent $1 on each of 4 yoghurts. How much more did he spend on ice cream than yoghurt?
Gold answer: 36
N steps: 3
L6 DAG: x0 = 10 * 4 ; x1 = 1 * 4 ; x2 = x0 - x1 ; answer = x2

EXAMPLE OUTPUT 3
{"problem_type": "comparison", "L0": "This is a comparison problem involving 3 steps. The operations used are <MUL_rate> and <SUB_difference>. Caleb bought 4 ice creams at $10 each and 4 yoghurts at $1 each; find how much more he spent on ice cream.", "L1": "TASK: comparison\nSTEPS: 3\nENTITY: Caleb\nKNOWN: 10 dollars/pint, 4 pints, 1 dollars/yoghurt, 4 yoghurts\nOPS: <MUL_rate>, <SUB_difference>\nGOAL: spending_difference (dollars)", "L2": "Step 1: Compute ice cream total. CELL=<MUL_rate>. ARG=4.\nStep 2: Compute yoghurt total. CELL=<MUL_rate>. ARG=4.\nStep 3: Subtract yoghurt from ice cream. CELL=<SUB_difference>. ARG=0.\nANSWER = 3", "L3": "x_1 := <MUL_rate>(10, 4)\nx_2 := <MUL_rate>(4, 1)\nx_3 := <SUB_difference>(x_1, x_2)\nANSWER := x_3", "L4": "x_1 := <MUL>(10, 4)\nx_2 := <MUL>(4, 1)\nx_3 := <SUB>(x_1, x_2)\nANSWER := x_3", "L5": "x_1 = 10 * 4 ; x_2 = 4 * 1 ; x_3 = x_1 - x_2 ; ANSWER = x_3"}

Note in EXAMPLE 3 (3 steps with one duplicate) how:
- L0 lists UNIQUE cells: "The operations used are <MUL_rate> and <SUB_difference>." (NOT "<MUL_rate> and <MUL_rate> and ...")
- L1 OPS slot lists UNIQUE cells in step-order: "OPS: <MUL_rate>, <SUB_difference>"
- L2 has the per-step CELL with possible repeats (CELL=<MUL_rate> for both steps 1 and 2)
- L3 matches L2 step-for-step (<MUL_rate>, <MUL_rate>, <SUB_difference>)

NOW DO THIS PROBLEM
====================
Problem: __PROBLEM__
Gold answer: __ANSWER__
N steps: __N_STEPS__
L6 DAG: __EXISTING_L6__

__STRICTER_REMINDER__OUTPUT (single JSON object, no markdown, no commentary):"""


# ----------------------------------------------------------------------
# Validation regexes
# ----------------------------------------------------------------------
# v7: L0 must contain the "The operations used are <X> ..." sentence after the
# preamble. We rely on a structural match for the preamble + the post-parse
# extractor `validate_L0_ops` to verify the <OP_role> tokens.
L0_RE = re.compile(
    r"^This is a \w+ problem involving \d+ steps?\. The operations used are .+"
)

# v7: L1 gains an OPS slot between KNOWN and GOAL (6 keys total).
L1_KEYS = ("TASK", "STEPS", "ENTITY", "KNOWN", "OPS", "GOAL")
L1_LINE_RE = re.compile(r"^(TASK|STEPS|ENTITY|KNOWN|OPS|GOAL):\s*.+$")

# v6 codebook-aligned: L3 uses compositional <OP_role> from a CLOSED 12-entry
# list (L3_OP_ROLES). 4 ops × 3 roles. Build alternation directly from the
# explicit list so we don't need to validate OP-role pair membership separately.
_L3_ACTION_ALT = "|".join(sorted(L3_OP_ROLES, key=lambda s: (-len(s), s)))  # longest first for safe matching
L3_STEP_RE = re.compile(
    rf"^x_\d+ := <(?P<action>{_L3_ACTION_ALT})>\([^()]*\)$"
)
L3_ANS_RE = re.compile(r"^ANSWER := x_\d+$")

# v7: L2 step grammar uses CELL=<OP_role> (compositional) instead of v6's OP=<OP>.
# Matches the same 12 codebook-aligned action names as L3.
L2_STEP_RE = re.compile(
    rf"^Step \d+: [^()\n]+\. CELL=<(?P<cell>{_L3_ACTION_ALT})>\. ARG=-?\d+(?:\.\d+)?\.$"
)
L2_ANS_RE = re.compile(r"^ANSWER = \d+$")

# v7: regex to extract <OP_role> tokens out of L0's ops sentence and L1's OPS slot.
# This finds ALL `<OP_role>` cell tokens anywhere in a text — we then check
# membership in L3_ACTION_SET.
L0_OP_ROLE_RE = re.compile(
    rf"<({_L3_ACTION_ALT})>"
)

# L4: bounded OP-name function form. v6: restricted to 4 OPs to match L3's
# vocab (codebook alignment). MOD/POW problems will fail validation and be
# dropped (~0.03% of GSM8K).
L4_STEP_RE = re.compile(r"^x_\d+ := <(ADD|SUB|MUL|DIV)>\([^()]*\)$")
L4_ANS_RE = re.compile(r"^ANSWER := x_\d+$")

# v4: L5 is a single-line statement list using " ; " separators, ZERO-INDEXED
# (no underscore), with ANSWER uppercase.
# Form: x_1 = <expr> ; x_2 = <expr> ; ... ; ANSWER = x_<k>
L5_FULL_RE = re.compile(
    r"^x_\d+\s*=\s*[^;]+(?:\s*;\s*x_\d+\s*=\s*[^;]+)*\s*;\s*ANSWER\s*=\s*x_\d+\s*$"
)
# Stmt RHS allows letters/digits/operators — but NOT underscore (so x_N is
# rejected). We accept lowercase x followed by digits as identifiers.
L5_STMT_EQ_RE = re.compile(r"^x_\d+\s*=\s*[0-9a-zA-Z_+\-*/%.() ]+$")
L5_STMT_ANS_RE = re.compile(r"^ANSWER\s*=\s*x_\d+$")


# ----------------------------------------------------------------------
# Validators per layer.
# ----------------------------------------------------------------------

def _split_lines(text: str) -> list[str]:
    """Strip and split, dropping blank lines."""
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def validate_L0(text: str) -> tuple[bool, str]:
    """v7: L0 must include 'The operations used are <X> ...' after the preamble,
    listing <OP_role>s from the 12-entry list."""
    text = text.strip()
    if not text.startswith("This is a "):
        return False, "L0 must start with 'This is a '"
    if "\n" in text:
        return False, "L0 must be a single line"
    if not L0_RE.match(text):
        return False, (
            "L0 does not match 'This is a <type> problem involving <N> steps. "
            "The operations used are <X>...'"
        )
    # v7: extract <OP_role>s and verify each is in the closed 12-list.
    op_roles = extract_l0_op_roles(text)
    if not op_roles:
        return False, "L0 ops sentence has no <OP_role> tokens"
    for tok in op_roles:
        if tok not in L3_ACTION_SET:
            return False, (
                f"L0 contains unknown <OP_role> token {tok!r} "
                f"(must be in {sorted(L3_OP_ROLES)})"
            )
    # No duplicates allowed in L0's ops sentence (it's the UNIQUE set).
    if len(op_roles) != len(set(op_roles)):
        return False, f"L0 ops sentence has duplicate <OP_role>s: {op_roles}"
    return True, ""


def extract_l0_op_roles(text: str) -> list[str]:
    """v7: extract <OP_role> tokens from L0's 'The operations used are ...' sentence.

    Returns the list of tokens (no <> brackets) in order of appearance.
    """
    # Locate the ops sentence; default to the substring after the preamble.
    m = re.search(
        r"The operations used are (.+?)\.",
        text,
    )
    if m is None:
        return []
    ops_segment = m.group(1)
    return L0_OP_ROLE_RE.findall(ops_segment)


def extract_l1_ops_slot(text: str) -> list[str] | None:
    """v7: return the parsed OPS slot from L1 as a list of <OP_role> tokens.

    Returns None if OPS slot is absent or malformed; an empty list is a valid
    'empty OPS slot' (which would itself fail validation upstream).
    """
    for ln in _split_lines(text):
        if ln.startswith("OPS:"):
            raw = ln.split(":", 1)[1].strip()
            return L0_OP_ROLE_RE.findall(raw)
    return None


def validate_L1(text: str, expected_problem_type: str | None = None) -> tuple[bool, str]:
    """v7: L1 has 6 lines (TASK, STEPS, ENTITY, KNOWN, OPS, GOAL)."""
    lines = _split_lines(text)
    if len(lines) != 6:
        return False, f"L1 must have exactly 6 lines, got {len(lines)}"
    seen = []
    for ln in lines:
        if not L1_LINE_RE.match(ln):
            return False, f"L1 line bad format: {ln!r}"
        key = ln.split(":", 1)[0].strip()
        seen.append(key)
    if seen != list(L1_KEYS):
        return False, f"L1 keys/order must be {list(L1_KEYS)}, got {seen}"
    # v7: OPS slot entries must all be in the 12-entry closed list, with no
    # duplicates (it lists UNIQUE cells used in L3).
    ops_tokens = extract_l1_ops_slot(text)
    if ops_tokens is None:
        return False, "L1 missing OPS slot"
    if not ops_tokens:
        return False, "L1 OPS slot is empty"
    for tok in ops_tokens:
        if tok not in L3_ACTION_SET:
            return False, (
                f"L1 OPS contains unknown <OP_role> token {tok!r} "
                f"(must be in {sorted(L3_OP_ROLES)})"
            )
    if len(ops_tokens) != len(set(ops_tokens)):
        return False, f"L1 OPS slot has duplicate <OP_role>s: {ops_tokens}"
    return True, ""


def validate_L2(text: str) -> tuple[bool, str]:
    """v7: L2 step grammar uses CELL=<OP_role> instead of v6's OP=<OP>."""
    lines = _split_lines(text)
    if len(lines) < 2:
        return False, "L2 needs at least 1 step + ANSWER"
    *step_lines, ans_line = lines
    for i, ln in enumerate(step_lines, start=1):
        # v7: parentheses ARE allowed as part of CELL=<...> tokens.
        # The disallowed thing is "inline notes" like "Step 1: foo (note). CELL=..."
        # The L2_STEP_RE rejects parentheses in the prose segment via [^()\n]+.
        if not L2_STEP_RE.match(ln):
            return False, (
                f"L2 step {i} bad format (need 'Step k: <prose>. CELL=<OP_role>. "
                f"ARG=<num>.'): {ln!r}"
            )
        # Check the step index matches.
        m = re.match(r"^Step (\d+):", ln)
        if m and int(m.group(1)) != i:
            return False, f"L2 step ordering wrong at line {i}: {ln!r}"
    if not L2_ANS_RE.match(ans_line):
        return False, f"L2 final line must be 'ANSWER = <k>': {ans_line!r}"
    # ANSWER index must be in range.
    k = int(ans_line.split("=", 1)[1].strip())
    n = len(step_lines)
    if not (1 <= k <= n):
        return False, f"L2 ANSWER index {k} out of range 1..{n}"
    return True, ""


def _l2_step_cells(text: str) -> list[str]:
    """v7: extract per-step CELL=<OP_role> tokens from L2 (assumes pre-validated)."""
    cells: list[str] = []
    for ln in _split_lines(text):
        m = L2_STEP_RE.match(ln)
        if m:
            cells.append(m.group("cell"))
    return cells


def _l3_step_actions(text: str) -> list[str]:
    """v7: extract per-step <OP_role> action tokens from L3 (assumes pre-validated)."""
    actions: list[str] = []
    for ln in _split_lines(text):
        m = L3_STEP_RE.match(ln)
        if m:
            actions.append(m.group("action"))
    return actions


def validate_L3(text: str) -> tuple[bool, str]:
    """v6 codebook-aligned: L3 uses compositional <OP_role> from L3_OP_ROLES (12).

    Structure (per line):  x_k := <OP_role>(args)
    where OP_role ∈ L3_OP_ROLES (the 12 codebook-aligned entries).
    Final line: 'ANSWER := x_<k>'.

    Args may only be numeric literals or earlier x_j references (j < k).
    """
    lines = _split_lines(text)
    if len(lines) < 2:
        return False, "L3 needs at least 1 step + ANSWER"
    *step_lines, ans_line = lines
    n = len(step_lines)
    for i, ln in enumerate(step_lines, start=1):
        m = L3_STEP_RE.match(ln)
        if not m:
            return False, (
                f"L3 step {i} bad format (need 'x_k := <OP_role>(args)' with OP_role "
                f"in {sorted(L3_OP_ROLES)}): {ln!r}"
            )
        action = m.group("action")
        if action not in L3_ACTION_SET:
            return False, f"L3 step {i} action {action!r} not in {sorted(L3_OP_ROLES)}: {ln!r}"
        op, role = action.split("_", 1)
        midx = re.match(r"^x_(\d+) :=", ln)
        if midx and int(midx.group(1)) != i:
            return False, f"L3 step indexing wrong at line {i}: {ln!r}"
        # Validate args: numeric literals or earlier x_j only.
        arg_str = ln[ln.index("(") + 1 : ln.rindex(")")].strip()
        if arg_str:
            for raw_arg in arg_str.split(","):
                a = raw_arg.strip()
                if not a:
                    return False, f"L3 step {i} has empty arg: {ln!r}"
                m2 = re.fullmatch(r"x_(\d+)", a)
                if m2:
                    j = int(m2.group(1))
                    if j >= i:
                        return False, f"L3 step {i} references x_{j} (not earlier): {ln!r}"
                    continue
                if re.fullmatch(r"-?\d+(?:\.\d+)?", a):
                    continue
                return False, f"L3 step {i} arg {a!r} is neither numeric nor x_j: {ln!r}"
    if not L3_ANS_RE.match(ans_line):
        return False, f"L3 final line must be 'ANSWER := x_<k>': {ans_line!r}"
    k = int(ans_line.split("x_", 1)[1].strip())
    if not (1 <= k <= n):
        return False, f"L3 ANSWER index {k} out of range 1..{n}"
    return True, ""


def _l3_step_ops(text: str) -> list[str]:
    """Extract the OP prefix from each L3 step (assumes pre-validated).
    v6: parse 'action' group then split out OP before underscore."""
    ops = []
    for ln in _split_lines(text):
        m = L3_STEP_RE.match(ln)
        if m:
            action = m.group("action")
            op = action.split("_", 1)[0]
            ops.append(op)
    return ops


def _l4_step_ops(text: str) -> list[str]:
    """Extract the OP from each L4 step (assumes pre-validated).
    v6: restricted to 4 OPs aligned with L3's vocab."""
    ops = []
    for ln in _split_lines(text):
        m = re.match(r"^x_\d+ := <(ADD|SUB|MUL|DIV)>", ln)
        if m:
            ops.append(m.group(1))
    return ops


def _has_inline_note_paren(line: str) -> bool:
    """Detect parenthetical notes (not arithmetic grouping).

    A safe heuristic: parse the RHS as a Python arithmetic expression (with
    identifiers as variables). If it fails to parse, it likely contains a
    free-form note.
    """
    # If there are NO parens at all, no problem.
    if "(" not in line and ")" not in line:
        return False
    # Extract RHS.
    if "=" not in line:
        return False
    rhs = line.split("=", 1)[1].strip()
    # Strip arithmetic expression chars and identifiers; whatever remains in
    # parens that's text-like is a note.
    # Try Python compile of the RHS — identifiers become Name nodes, which
    # is fine. If it doesn't compile, it has prose.
    try:
        import ast
        ast.parse(rhs, mode="eval")
        return False
    except SyntaxError:
        return True


def validate_L4(text: str) -> tuple[bool, str]:
    """L4 must be bounded OP-name function form.

    Structure (per line):  x_k := <OP>(args)
    where OP ∈ {ADD, SUB, MUL, DIV} (v6: MOD/POW dropped) and args are comma-separated
    numeric literals or earlier x_j references (j<k). Final line is
    'ANSWER := x_<k>'. No snake_case identifiers, no inline notes.
    """
    lines = _split_lines(text)
    if len(lines) < 2:
        return False, "L4 needs at least 1 step + ANSWER"
    *step_lines, ans_line = lines
    n = len(step_lines)
    for i, ln in enumerate(step_lines, start=1):
        if not L4_STEP_RE.match(ln):
            return False, f"L4 step {i} bad format (need 'x_k := <OP>(args)' with OP in {sorted(L3_OPS)}): {ln!r}"
        m = re.match(r"^x_(\d+) :=", ln)
        if m and int(m.group(1)) != i:
            return False, f"L4 step indexing wrong at line {i}: {ln!r}"
        # Validate args: numeric literals or earlier x_j only.
        arg_str = ln[ln.index("(") + 1 : ln.rindex(")")].strip()
        if arg_str:
            for raw_arg in arg_str.split(","):
                a = raw_arg.strip()
                if not a:
                    return False, f"L4 step {i} has empty arg: {ln!r}"
                # x_j with j < i
                m2 = re.fullmatch(r"x_(\d+)", a)
                if m2:
                    j = int(m2.group(1))
                    if j >= i:
                        return False, f"L4 step {i} references x_{j} (not earlier): {ln!r}"
                    continue
                # numeric literal (int or float, optional sign)
                if re.fullmatch(r"-?\d+(?:\.\d+)?", a):
                    continue
                return False, f"L4 step {i} arg {a!r} is neither numeric nor x_j: {ln!r}"
    if not L4_ANS_RE.match(ans_line):
        return False, f"L4 final line must be 'ANSWER := x_<k>': {ans_line!r}"
    k = int(ans_line.split("x_", 1)[1].strip())
    if not (1 <= k <= n):
        return False, f"L4 ANSWER index {k} out of range 1..{n}"
    return True, ""


def validate_L5(text: str) -> tuple[bool, str]:
    """v4: L5 must be a SINGLE LINE with ';' separators, ZERO-INDEXED vars.

    Form:  x0 = <expr> ; x1 = <expr> ; ... ; ANSWER = x<k>
    Uses xN (NO underscore, 0-indexed) and ANSWER (uppercase).
    """
    text = text.strip()
    if not text:
        return False, "L5 is empty"
    if "\n" in text:
        return False, "L5 must be a single line (no newlines); use ' ; ' separators"
    if ";" not in text:
        return False, "L5 must use ';' as statement separator"
    if not L5_FULL_RE.match(text):
        return False, "L5 does not match 'x_1 = ... ; ... ; ANSWER = x_<k>' single-line form"

    stmts = [p.strip() for p in text.split(";") if p.strip()]
    if len(stmts) < 2:
        return False, "L5 needs at least 1 equation + ANSWER statement"
    *eq_stmts, ans_stmt = stmts
    for i, st in enumerate(eq_stmts, start=1):
        if not L5_STMT_EQ_RE.match(st):
            return False, f"L5 statement {i} bad format: {st!r}"
        if _has_inline_note_paren(st):
            return False, f"L5 statement {i} contains parenthetical note: {st!r}"
        m = re.match(r"^x_(\d+)\s*=", st)
        if m is None:
            return False, f"L5 statement {i} must start with x_N: {st!r}"
        if int(m.group(1)) != i:
            return False, (
                f"L5 statement indexing wrong at position {i}: expected x_{i}, "
                f"got {st!r}"
            )
    if not L5_STMT_ANS_RE.match(ans_stmt):
        return False, f"L5 final statement must be 'ANSWER = x_<k>': {ans_stmt!r}"
    return True, ""


def validate_problem_type(pt: str) -> tuple[bool, str]:
    if pt not in PROBLEM_TYPES:
        return False, f"problem_type {pt!r} not in closed vocab {sorted(PROBLEM_TYPES)}"
    return True, ""


# ----------------------------------------------------------------------
# SymPy L6 verification (against the gold answer).
# ----------------------------------------------------------------------

def sympy_eval_L6(l6_text: str) -> float | None:
    """Eval a single-line DAG like 'x0 = 50 / 60 ; x1 = x0 * 12 ; answer = x1'.

    Returns the float numeric value of `answer`, or None on failure.
    """
    try:
        env: dict[str, object] = {}
        parts = [p.strip() for p in l6_text.split(";") if p.strip()]
        ans_val = None
        for p in parts:
            if "=" not in p:
                return None
            lhs, rhs = p.split("=", 1)
            lhs = lhs.strip()
            rhs = rhs.strip()
            # Eval RHS in env.
            val = sympify(rhs, locals={k: sympify(str(v)) for k, v in env.items()})
            # Substitute existing env into val.
            for k, v in env.items():
                if hasattr(val, "subs"):
                    val = val.subs(sympy.Symbol(k), sympify(str(v)))
            env[lhs] = val
            if lhs.lower() == "answer":
                ans_val = val
        if ans_val is None:
            return None
        return float(ans_val)
    except Exception:
        return None


def L6_matches_gold(l6_text: str, gold: float) -> bool:
    v = sympy_eval_L6(l6_text)
    if v is None:
        return False
    return abs(v - gold) < 1e-3


# ----------------------------------------------------------------------
# Cross-layer consistency: step count must match L6.
# ----------------------------------------------------------------------

def count_l6_steps(l6_text: str) -> int:
    parts = [p.strip() for p in l6_text.split(";") if p.strip() and not p.strip().lower().startswith("answer")]
    return len(parts)


def step_count_consistent(layers: dict[str, str], expected_n: int) -> tuple[bool, str]:
    # L2: number of "Step k:" lines
    l2_steps = [ln for ln in _split_lines(layers["L2"]) if ln.startswith("Step ")]
    if len(l2_steps) != expected_n:
        return False, f"L2 has {len(l2_steps)} steps, expected {expected_n}"
    # L3: number of "x_k :=" lines
    l3_steps = [ln for ln in _split_lines(layers["L3"]) if re.match(r"^x_\d+ :=", ln)]
    if len(l3_steps) != expected_n:
        return False, f"L3 has {len(l3_steps)} steps, expected {expected_n}"
    # L5: number of "x_N = ..." statements on the single line (not "ANSWER = ...").
    l5_stmts = [p.strip() for p in layers["L5"].split(";") if p.strip()]
    l5_steps = [st for st in l5_stmts if re.match(r"^x_\d+\s*=", st)]
    if len(l5_steps) != expected_n:
        return False, f"L5 has {len(l5_steps)} steps, expected {expected_n}"
    # L4: number of "x_k :=" lines (NOT the "ANSWER := x_k" line)
    l4_steps = [ln for ln in _split_lines(layers["L4"])
                if re.match(r"^x_\d+ :=", ln)]
    if len(l4_steps) != expected_n:
        return False, f"L4 has {len(l4_steps)} steps, expected {expected_n}"
    return True, ""


# ----------------------------------------------------------------------
# Parsing Haiku output (JSON object).
# ----------------------------------------------------------------------

def parse_haiku_json(text: str) -> dict | None:
    """Parse Haiku output as a JSON object. Tolerates markdown fences."""
    s = text.strip()
    # Strip ```json fences if present.
    if s.startswith("```"):
        # Find first newline and last fence.
        lines = s.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    # Try direct JSON.
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        # Try finding the first {...} block.
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    return obj


# ----------------------------------------------------------------------
# Full validation pipeline.
# ----------------------------------------------------------------------

def validate_full(obj: dict, expected_n: int) -> tuple[bool, str]:
    """Validate the parsed Haiku object. Returns (ok, error_message)."""
    required = ("problem_type", "L0", "L1", "L2", "L3", "L4", "L5")
    for k in required:
        if k not in obj:
            return False, f"missing key {k!r}"
        if not isinstance(obj[k], str):
            return False, f"key {k!r} must be string, got {type(obj[k]).__name__}"

    ok, err = validate_problem_type(obj["problem_type"])
    if not ok:
        return False, err

    pt = obj["problem_type"]
    for k, fn in (("L0", lambda t: validate_L0(t)),
                  ("L1", lambda t: validate_L1(t, pt)),
                  ("L2", validate_L2),
                  ("L3", validate_L3),
                  ("L4", validate_L4),
                  ("L5", validate_L5)):
        ok, err = fn(obj[k])
        if not ok:
            return False, f"{k}: {err}"

    # Cross-layer: TASK in L1 must match problem_type.
    l1_lines = _split_lines(obj["L1"])
    task_line = next((ln for ln in l1_lines if ln.startswith("TASK:")), "")
    task_val = task_line.split(":", 1)[1].strip() if ":" in task_line else ""
    if task_val != pt:
        return False, f"L1 TASK {task_val!r} != problem_type {pt!r}"
    steps_line = next((ln for ln in l1_lines if ln.startswith("STEPS:")), "")
    steps_val_str = steps_line.split(":", 1)[1].strip() if ":" in steps_line else ""
    try:
        steps_val = int(steps_val_str)
    except ValueError:
        return False, f"L1 STEPS must be integer, got {steps_val_str!r}"
    if steps_val != expected_n:
        return False, f"L1 STEPS={steps_val}, expected {expected_n}"

    # L0 step count.
    m = re.match(r"^This is a (\w+) problem involving (\d+) steps?\.", obj["L0"])
    if not m:
        return False, "L0 cannot extract type/step count"
    l0_type, l0_n = m.group(1), int(m.group(2))
    if l0_type != pt:
        return False, f"L0 type {l0_type!r} != problem_type {pt!r}"
    if l0_n != expected_n:
        return False, f"L0 step count {l0_n} != expected {expected_n}"

    # Step count consistency across L2/L3/L4/L5.
    layers = {k: obj[k] for k in ("L0", "L1", "L2", "L3", "L4", "L5")}
    ok, err = step_count_consistent(layers, expected_n)
    if not ok:
        return False, err

    # v6: L3 step k's OP MUST match L4 step k's OP.
    l3_ops = _l3_step_ops(obj["L3"])
    l4_ops = _l4_step_ops(obj["L4"])
    if len(l3_ops) != len(l4_ops):
        return False, f"L3 has {len(l3_ops)} OPs but L4 has {len(l4_ops)}"
    for i, (o3, o4) in enumerate(zip(l3_ops, l4_ops), start=1):
        if o3 != o4:
            return False, (
                f"L3/L4 OP mismatch at step {i}: L3 has <{o3}_role> but L4 "
                f"has <{o4}>"
            )

    # v7: L2 step k's CELL MUST match L3 step k's <OP_role> (positional).
    l2_cells = _l2_step_cells(obj["L2"])
    l3_actions = _l3_step_actions(obj["L3"])
    if len(l2_cells) != len(l3_actions):
        return False, (
            f"L2 has {len(l2_cells)} CELL tags but L3 has {len(l3_actions)} actions"
        )
    for i, (c2, c3) in enumerate(zip(l2_cells, l3_actions), start=1):
        if c2 != c3:
            return False, (
                f"L2/L3 CELL mismatch at step {i}: L2 has CELL=<{c2}> but L3 "
                f"has <{c3}>"
            )

    # v7: set(L0 ops sentence) == set(L1 OPS slot) == set(unique L3 actions).
    l0_ops_set = set(extract_l0_op_roles(obj["L0"]))
    l1_ops_tokens = extract_l1_ops_slot(obj["L1"]) or []
    l1_ops_set = set(l1_ops_tokens)
    l3_unique_set = set(l3_actions)
    if l0_ops_set != l3_unique_set:
        return False, (
            f"L0 ops set {sorted(l0_ops_set)} != L3 unique actions "
            f"{sorted(l3_unique_set)}"
        )
    if l1_ops_set != l3_unique_set:
        return False, (
            f"L1 OPS set {sorted(l1_ops_set)} != L3 unique actions "
            f"{sorted(l3_unique_set)}"
        )

    return True, ""


# ----------------------------------------------------------------------
# Haiku call with retries.
# ----------------------------------------------------------------------

def call_haiku_all_layers(
    client: Anthropic,
    record: dict,
    max_retries: int = 3,
) -> tuple[dict | None, dict | None, str | None]:
    """Call Haiku to regenerate L0-L5 for the given record.

    Returns (parsed_validated_obj, usage_dict, last_err_string).
    """
    problem = record["problem"]
    answer = record["answer"]
    existing_L6 = record["layers"]["L6"]
    n_steps = count_l6_steps(existing_L6)
    if n_steps < 1:
        return None, None, f"L6 has no steps: {existing_L6!r}"

    last_err = None
    in_tok = 0
    out_tok = 0
    for attempt in range(max_retries + 1):
        stricter = ""
        if attempt > 0:
            stricter = (
                "IMPORTANT REMINDER (your previous attempt failed validation: "
                f"{last_err or 'format violation'}). "
                "Follow the format EXACTLY. Output ONLY a single JSON object. "
                "Use real newlines inside multi-line layer strings (L1..L4). "
                "L0 MUST contain the sentence 'The operations used are <X> and "
                "<Y>.' (or comma+Oxford list for 3+ ops) after the preamble, "
                "listing the UNIQUE set of <OP_role>s used in L3. "
                "L1 MUST have EXACTLY 6 lines in this order: TASK, STEPS, "
                "ENTITY, KNOWN, OPS, GOAL. The OPS slot lists UNIQUE "
                "<OP_role>s in L3 step order. "
                "L2 step lines use 'CELL=<OP_role>' (NOT 'OP=<OP>'). The "
                "angle brackets ARE part of the CELL token. NO parenthetical "
                "notes in the prose part of L2 steps. "
                "L3 MUST use the compositional <OP_role> form in 'x_k := "
                "<OP_role>(args)'. OP_role is EXACTLY one of the 12 codebook-"
                "aligned action names: <ADD_total>, <ADD_combine>, <ADD_return>, "
                "<SUB_difference>, <SUB_remainder>, <SUB_cost>, <MUL_rate>, "
                "<MUL_groups>, <MUL_scale>, <DIV_split>, <DIV_rate>, "
                "<DIV_fraction>. NO other action names. NO custom snake_case "
                "verbs. NO multi-word names. NO leading or trailing underscores. "
                "The angle brackets `<` and `>` are LITERAL characters that "
                "MUST appear in the output. "
                "CROSS-LAYER: L2 step k's CELL MUST be IDENTICAL to L3 step "
                "k's <OP_role>. set(L0 ops sentence) == set(L1 OPS slot) == "
                "set(unique L3 OP_roles). L3 step k's OP prefix MUST be "
                "IDENTICAL to L4 step k's OP. "
                "L4 MUST use the closed OP vocabulary {ADD,SUB,MUL,DIV} "
                "in 'x_k := <OP>(args)' form — NO snake_case names. Args may "
                "only be numeric literals or earlier x_j references. "
                "L5 MUST be a SINGLE LINE with ' ; ' separators — NO newlines. "
                "L5 uses 1-INDEXED x_N vars (x_1, x_2, x_3; WITH underscore) and "
                "ANSWER uppercase. L5 statements MUST have NO inline parenthetical "
                "notes. Use exactly the closed problem_type and OP vocabularies.\n\n"
            )
        prompt = (PROMPT_ALL_LAYERS
                  .replace("__PROBLEM__", str(problem))
                  .replace("__ANSWER__", str(answer))
                  .replace("__N_STEPS__", str(n_steps))
                  .replace("__EXISTING_L6__", str(existing_L6))
                  .replace("__STRICTER_REMINDER__", stricter))
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            in_tok += response.usage.input_tokens
            out_tok += response.usage.output_tokens

            obj = parse_haiku_json(text)
            if obj is None:
                last_err = f"JSON parse failed: {text[:200]!r}"
            else:
                ok, err = validate_full(obj, expected_n=n_steps)
                if ok:
                    return obj, {"input_tokens": in_tok, "output_tokens": out_tok}, None
                last_err = err
        except Exception as e:
            last_err = f"API error: {e}"
        if attempt < max_retries:
            time.sleep(0.8 * (attempt + 1))
    return None, {"input_tokens": in_tok, "output_tokens": out_tok}, last_err


# ----------------------------------------------------------------------
# Per-problem processing.
# ----------------------------------------------------------------------

def _process_one(client: Anthropic, record: dict) -> dict:
    """Process one source record into a v80 output record (or a failure marker)."""
    problem = record["problem"]
    answer = record["answer"]
    existing_L6 = record["layers"]["L6"]
    gold = float(answer)

    # Data-integrity gate: skip if existing L6 doesn't SymPy-eval to gold.
    if not L6_matches_gold(existing_L6, gold):
        return {
            "__skipped__": True,
            "reason": "L6_not_matching_gold",
            "src_problem": problem[:120],
            "src_L6": existing_L6,
            "src_answer": answer,
        }

    obj, usage, err = call_haiku_all_layers(client, record)
    if obj is None:
        return {
            "__failed_api__": True,
            "error": err or "unknown",
            "usage": usage or {},
            "src_problem": problem[:120],
        }

    n_steps = count_l6_steps(existing_L6)
    out_rec = {
        # Loader (mycelium/l3_data.py::load_gsm8k_v77) reads `problem` and
        # `answer` at top level — keep those names exactly.
        "problem": problem,
        "answer": answer,
        "problem_type": obj["problem_type"],
        "n_steps": n_steps,
        # The loader reads layers from this dict (not the flat L0..L6 fields).
        "layers": {
            "L0": obj["L0"],
            "L1": obj["L1"],
            "L2": obj["L2"],
            "L3": obj["L3"],
            "L4": obj["L4"],
            "L5": obj["L5"],
            "L6": existing_L6,
        },
        # Top-level flat fields for inspection / spot-checking.
        "L0": obj["L0"],
        "L1": obj["L1"],
        "L2": obj["L2"],
        "L3": obj["L3"],
        "L4": obj["L4"],
        "L5": obj["L5"],
        "L6": existing_L6,
        # Passthrough useful fields from v78c.
        "gen_targets": record.get("gen_targets", []),
        "sympy_value": record.get("sympy_value"),
        "sympy_matches_gold": True,  # gated above
        "usage_v80": usage,
    }
    return out_rec


# ----------------------------------------------------------------------
# Stats / main.
# ----------------------------------------------------------------------

class _Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.kept = 0
        self.failed_api = 0
        self.failed_validation = 0
        self.skipped = 0
        self.in_tokens = 0
        self.out_tokens = 0
        self.processed = 0


def load_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    p = Path(ANTHROPIC_KEY_FALLBACK)
    if p.exists():
        return p.read_text().strip()
    raise RuntimeError(
        f"No API key: set ANTHROPIC_API_KEY env var or place key at {ANTHROPIC_KEY_FALLBACK}"
    )


def load_records(input_path: Path, num: int, start: int = 0) -> list[dict]:
    out = []
    with input_path.open("r") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            if i < start:
                continue
            out.append(json.loads(line))
            if len(out) >= num:
                break
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True,
                        help="Input v78c JSONL (we use problem/answer/n_steps/L6)")
    parser.add_argument("--dst", type=Path, required=True,
                        help="Output v80 JSONL")
    parser.add_argument("--num", type=int, default=100,
                        help="Number of problems to process (default 100)")
    parser.add_argument("--start", type=int, default=0,
                        help="Skip the first N problems in the input file")
    parser.add_argument("--concurrency", type=int, default=16,
                        help="Number of parallel Haiku API calls (default 16)")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max retry attempts per problem on validation failure")
    args = parser.parse_args()

    if not args.src.exists():
        print(f"ERROR: input file not found: {args.src}", file=sys.stderr)
        sys.exit(1)

    os.environ["ANTHROPIC_API_KEY"] = load_api_key()

    records = load_records(args.src, args.num, start=args.start)
    if not records:
        print(f"ERROR: no records found in {args.src}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(records)} records from {args.src} (start={args.start})")
    print(f"Regenerating L0-L5 via {MODEL} (concurrency={args.concurrency}, max_retries={args.max_retries})...")
    print(f"Output: {args.dst}")

    client = Anthropic()
    args.dst.parent.mkdir(parents=True, exist_ok=True)

    stats = _Stats()
    t0 = time.perf_counter()
    write_lock = threading.Lock()

    def _wrap(rec: dict) -> dict:
        return _process_one(client, rec)

    with args.dst.open("w") as out_f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(_wrap, ex): (i, ex) for i, ex in enumerate(records)}
            for fut in concurrent.futures.as_completed(futures):
                i, _ = futures[fut]
                try:
                    rec = fut.result()
                except Exception as e:
                    rec = {"__failed_api__": True, "error": f"future-raised: {e}"}

                with stats.lock:
                    stats.processed += 1
                    if rec.get("__failed_api__"):
                        stats.failed_api += 1
                        # Track validation vs network distinctly.
                        err = rec.get("error", "")
                        if any(kw in err for kw in ("API error", "future-raised", "JSON parse failed")):
                            pass
                        else:
                            stats.failed_validation += 1
                        usage = rec.get("usage") or {}
                        stats.in_tokens += int(usage.get("input_tokens", 0))
                        stats.out_tokens += int(usage.get("output_tokens", 0))
                        progress = stats.processed
                        if progress % 25 == 0 or progress == len(records):
                            _print_progress(stats, t0, len(records))
                        continue
                    if rec.get("__skipped__"):
                        stats.skipped += 1
                        progress = stats.processed
                        if progress % 25 == 0 or progress == len(records):
                            _print_progress(stats, t0, len(records))
                        continue
                    usage = rec.get("usage_v80") or {}
                    stats.in_tokens += int(usage.get("input_tokens", 0))
                    stats.out_tokens += int(usage.get("output_tokens", 0))

                with write_lock:
                    out_f.write(json.dumps(rec) + "\n")
                    out_f.flush()
                with stats.lock:
                    stats.kept += 1
                    if stats.processed % 25 == 0 or stats.processed == len(records):
                        _print_progress(stats, t0, len(records))

    elapsed = time.perf_counter() - t0
    in_cost = (stats.in_tokens / 1e6) * _PRICE_IN_PER_MTOK
    out_cost = (stats.out_tokens / 1e6) * _PRICE_OUT_PER_MTOK
    pass_rate = (stats.kept / max(stats.processed, 1)) * 100
    print(f"\n=== v80 regen done in {elapsed:.0f}s ===")
    print(f"Records written:    {stats.kept}/{len(records)}  ({pass_rate:.1f}% pass)")
    print(f"  api/format fail:    {stats.failed_api}")
    print(f"  data-integrity skipped: {stats.skipped}")
    print(f"Token usage:  in={stats.in_tokens:,}  out={stats.out_tokens:,}")
    print(f"Cost:         ${in_cost + out_cost:.2f}  (in=${in_cost:.2f} + out=${out_cost:.2f})")
    print(f"Output: {args.dst}")

    # v7: print 3 sample records verbatim, plus per-record constraint checks.
    _inspect_v7_constraints(args.dst, n_samples=3)


def _inspect_v7_constraints(dst_path: Path, n_samples: int = 3) -> None:
    """Print n_samples records verbatim and verify v7 invariants across all kept rows.

    Invariants (v6 carried forward):
      - L3 action ∈ L3_ACTION_SET (compositional <OP_role>, 12-entry codebook-aligned)
      - L3 step k's OP matches L4 step k's OP
      - L4 uses bounded OP vocab {ADD,SUB,MUL,DIV} (MOD/POW dropped)
      - L5 uses x_1/x_2/... (1-indexed, WITH underscore) + ANSWER uppercase, single line
      - L6 is unchanged shape (zero-indexed, lowercase answer)

    NEW v7 invariants:
      - L0 contains "The operations used are <X>..." sentence with <OP_role>s
      - L1 has 6 lines (OPS slot between KNOWN and GOAL)
      - L2 uses CELL=<OP_role> (not OP=<OP>)
      - L2 step k CELL == L3 step k <OP_role> (positional)
      - set(L0 ops) == set(L1 OPS) == set(unique L3 actions)
    """
    if not dst_path.exists():
        return
    records = []
    with dst_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("__failed_api__") or r.get("__skipped__"):
                continue
            records.append(r)
    if not records:
        print("\n[v7 inspect] no kept records to inspect.")
        return

    # Per-record constraint scan.
    n_total = len(records)
    n_l3_ok = 0
    n_l3l4_op_ok = 0
    n_l4_ok = 0
    n_l5_ok = 0
    n_l6_ok = 0
    # v7-specific counters.
    n_l0_ops_ok = 0   # L0 has the ops sentence with valid <OP_role>s
    n_l1_ops_ok = 0   # L1 has OPS slot as 5th line with valid <OP_role>s
    n_l2_cell_ok = 0  # L2 uses CELL=<OP_role>
    n_l2_l3_pos_ok = 0  # L2 step k CELL == L3 step k OP_role (positional)
    n_codebook_thread_ok = 0  # set(L0 ops) == set(L1 OPS) == set(unique L3 actions)
    bad = []
    op_role_counter: dict[str, int] = {}
    role_counter: dict[str, int] = {}
    for r in records:
        l0 = r["layers"]["L0"]
        l1 = r["layers"]["L1"]
        l2 = r["layers"]["L2"]
        l3 = r["layers"]["L3"]
        l4 = r["layers"]["L4"]
        l5 = r["layers"]["L5"]
        l6 = r["layers"]["L6"]

        l3_lines = [ln.strip() for ln in l3.splitlines() if ln.strip()]
        l3_actions_ok = True
        l3_actions_found = []
        for ln in l3_lines:
            mv = re.match(r"^x_\d+ := <([A-Za-z0-9_]+)>\(", ln)
            if mv:
                action = mv.group(1)
                l3_actions_found.append(action)
                if action not in L3_ACTION_SET:
                    l3_actions_ok = False
                    break
                op_role_counter[action] = op_role_counter.get(action, 0) + 1
                # Extract role for the role-frequency stat.
                # action = "<OP>_<role>" — split on the FIRST underscore.
                if "_" in action:
                    _op, _role = action.split("_", 1)
                    role_counter[_role] = role_counter.get(_role, 0) + 1
        if l3_actions_ok:
            n_l3_ok += 1
        else:
            bad.append(("L3", l3, l3_actions_found))

        # L3/L4 OP consistency check.
        try:
            l3_ops = _l3_step_ops(l3)
            l4_ops = _l4_step_ops(l4)
            if l3_ops and l3_ops == l4_ops:
                n_l3l4_op_ok += 1
            else:
                bad.append(("L3/L4 OP mismatch", f"L3 ops={l3_ops} L4 ops={l4_ops}", []))
        except Exception as e:
            bad.append(("L3/L4 OP exception", str(e), []))

        # L4 check: only bare 4-op vocab (v6: MOD/POW dropped).
        l4_lines = [ln.strip() for ln in l4.splitlines() if ln.strip()]
        l4_ops_ok = True
        for ln in l4_lines:
            if ln.startswith("ANSWER"):
                continue
            mv = re.match(r"^x_\d+ := <([A-Za-z0-9_]+)>\(", ln)
            if mv and mv.group(1) not in L3_OPS_SET:
                l4_ops_ok = False
                break
        if l4_ops_ok:
            n_l4_ok += 1
        else:
            bad.append(("L4", l4, []))

        # L5 v6 checks: 1-indexed underscore (x_1, x_2, ...), uppercase ANSWER, single line.
        l5_ok = (
            ("\n" not in l5)
            and (" ; ANSWER = x_" in l5)
            and bool(re.search(r"\bx_1\s*=", l5))
        )
        if l5_ok:
            n_l5_ok += 1
        else:
            bad.append(("L5", l5, []))

        # L6 invariant (must still be lowercase answer, zero-indexed, no x_).
        l6_ok = ("x_" not in l6) and ("answer = x" in l6)
        if l6_ok:
            n_l6_ok += 1
        else:
            bad.append(("L6", l6, []))

        # v7: L0 ops sentence check.
        l0_ops_list = extract_l0_op_roles(l0)
        if l0_ops_list and all(tok in L3_ACTION_SET for tok in l0_ops_list):
            n_l0_ops_ok += 1
        else:
            bad.append(("L0 ops", l0, l0_ops_list))

        # v7: L1 OPS slot check.
        l1_ops_list = extract_l1_ops_slot(l1)
        if l1_ops_list and all(tok in L3_ACTION_SET for tok in l1_ops_list):
            n_l1_ops_ok += 1
        else:
            bad.append(("L1 OPS slot", l1, l1_ops_list or []))

        # v7: L2 uses CELL=<OP_role>.
        l2_cells_list = _l2_step_cells(l2)
        # All step lines must produce a match (L2_STEP_RE non-empty + cell ∈ set).
        l2_step_line_count = sum(
            1 for ln in _split_lines(l2) if ln.startswith("Step ")
        )
        if (
            l2_step_line_count > 0
            and len(l2_cells_list) == l2_step_line_count
            and all(c in L3_ACTION_SET for c in l2_cells_list)
        ):
            n_l2_cell_ok += 1
        else:
            bad.append(("L2 CELL=<OP_role>", l2, l2_cells_list))

        # v7: positional L2 cell == L3 action.
        l3_actions_pos = _l3_step_actions(l3)
        if (
            l2_cells_list
            and l3_actions_pos
            and l2_cells_list == l3_actions_pos
        ):
            n_l2_l3_pos_ok += 1
        else:
            bad.append((
                "L2/L3 positional mismatch",
                f"L2 cells={l2_cells_list} L3 actions={l3_actions_pos}",
                [],
            ))

        # v7: codebook threading (set equality across L0/L1/L3).
        l0_set = set(l0_ops_list or [])
        l1_set = set(l1_ops_list or [])
        l3_set = set(l3_actions_pos or [])
        if l3_set and l0_set == l3_set and l1_set == l3_set:
            n_codebook_thread_ok += 1
        else:
            bad.append((
                "Codebook threading",
                f"L0={sorted(l0_set)} L1={sorted(l1_set)} L3={sorted(l3_set)}",
                [],
            ))

    print("\n=== v7 constraint check ===")
    print(f"L0 ops sentence (valid <OP_role>s): {n_l0_ops_ok}/{n_total}")
    print(f"L1 OPS slot (valid <OP_role>s):     {n_l1_ops_ok}/{n_total}")
    print(f"L2 uses CELL=<OP_role>:             {n_l2_cell_ok}/{n_total}")
    print(f"L2 step k CELL == L3 step k OP_role: {n_l2_l3_pos_ok}/{n_total}")
    print(f"Codebook threaded (L0/L1/L3 sets eq): {n_codebook_thread_ok}/{n_total}")
    print(f"L3 action ∈ 12-set:                 {n_l3_ok}/{n_total}")
    print(f"L3/L4 OP per-step match:            {n_l3l4_op_ok}/{n_total}")
    print(f"L4 OPs ∈ {{ADD,SUB,MUL,DIV}}:       {n_l4_ok}/{n_total}")
    print(f"L5 1-indexed (x_1...) + ANSWER UC + single-line: {n_l5_ok}/{n_total}")
    print(f"L6 unchanged shape:                 {n_l6_ok}/{n_total}")
    if op_role_counter:
        print("\n  L3 action-name frequency (across all records):")
        for action in L3_OP_ROLES:
            c = op_role_counter.get(action, 0)
            print(f"    {action:<24} {c}")
    if bad:
        print(f"\n  first violation example ({bad[0][0]}): {bad[0][1]!r}")
        if bad[0][2]:
            print(f"    actions found: {bad[0][2]}")

    print(f"\n=== {min(n_samples, len(records))} sample records (all 7 layers verbatim) ===")
    for i, r in enumerate(records[:n_samples], start=1):
        print(f"\n--- sample {i} (n_steps={r.get('n_steps', '?')}) ---")
        print(f"problem: {r['problem']}")
        print(f"answer:  {r['answer']}")
        for L in ("L0", "L1", "L2", "L3", "L4", "L5", "L6"):
            text = r["layers"][L]
            indented = "\n".join("    " + ln for ln in text.splitlines())
            print(f"  {L}:")
            print(indented)


def _print_progress(stats: _Stats, t0: float, total: int):
    elapsed = time.perf_counter() - t0
    in_cost = (stats.in_tokens / 1e6) * _PRICE_IN_PER_MTOK
    out_cost = (stats.out_tokens / 1e6) * _PRICE_OUT_PER_MTOK
    rate = stats.processed / max(elapsed, 1e-6)
    eta = (total - stats.processed) / max(rate, 1e-6)
    print(f"  [{stats.processed}/{total}]  kept={stats.kept}  "
          f"fail={stats.failed_api}  skip={stats.skipped}  "
          f"cost=${in_cost + out_cost:.2f}  "
          f"({elapsed:.0f}s, ETA {eta:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main()
