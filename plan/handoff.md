# Lambda Handoff: Track 2 as Problem Structure Parser

## The Insight

Track 2 doesn't compute answers — it **reads the problem and assembles a sympy program.** The IO tagger finds operation-bearing clauses. The classifier identifies the ROLE of each clause. Together they extract and structure the mathematical relationships that sympy solves.

This is fundamentally different from Track 1 (brute-force grab all LaTeX and try sympy). Track 2 understands problem STRUCTURE — which clauses are equations, which define variables, which set constraints, what's being asked for. That understanding comes from attention distillation training.

## Architecture

```
Problem Text
    │
    ▼
┌────────────────────┐
│ IO Tagger (0.5B)   │  Finds operation-bearing clauses
│ F1: 86.5%          │
└─────────┬──────────┘
          │ clauses
          ▼
┌────────────────────┐
│ Classifier (0.5B)  │  Labels each clause's ROLE:
│ Acc: 71.9%         │  EQUATION, CONSTRAINT, DEFINITION,
│                    │  OBJECTIVE, ASSIGNMENT
└─────────┬──────────┘
          │ (clause, role) pairs
          ▼
┌────────────────────┐
│ Expression         │  Extract LaTeX expressions from
│ Extractor          │  each clause: variables, equations,
│                    │  values, relationships
└─────────┬──────────┘
          │ sympy expressions
          ▼
┌────────────────────┐
│ Program Assembler  │  Build sympy program based on roles:
│                    │  - Collect all equations
│                    │  - Identify unknowns
│                    │  - Determine what to solve for
│                    │  - Assemble solve() call
└─────────┬──────────┘
          │ sympy program
          ▼
┌────────────────────┐
│ Sympy Executor     │  Execute the assembled program
│                    │  Return numerical answer
└─────────┬──────────┘
          │
          ▼
       Answer
```

## Clause Role Classification

The key change: instead of classifying operations (ADD, SUB, MUL, DIV), classify clause ROLES in the problem structure:

| Role | What it means | Example | Sympy action |
|---|---|---|---|
| EQUATION | An equation relating variables | "$m = r + 2$" | Add to equation list |
| CONSTRAINT | A numerical constraint | "total of 98 members" | Add constraint: expr = 98 |
| DEFINITION | Defines a function or variable | "Let $f(x) = x^2 + 1$" | Define sympy function |
| OBJECTIVE | What we're solving for | "Find the value of $m$" | Set as solve target |
| COMPUTATION | Direct computation | "What is $3 + \frac{1}{2}$?" | Evaluate directly |
| CONDITION | A condition or case | "when $x > 0$" | Add assumption |
| ASSIGNMENT | States a value | "$x = 5$" | Substitute value |

## Worked Example

Problem: "A rectangular band formation with $m$ band members in each of $r$ rows has a total of 98 band members. If $m = r + 2$, find $m$."

```
IO Tagger finds clauses:
  [1] "with $m$ band members in each of $r$ rows"
  [2] "has a total of 98 band members"
  [3] "$m = r + 2$"
  [4] "find $m$"

Classifier labels roles:
  [1] → EQUATION     (m * r is implied)
  [2] → CONSTRAINT   (total = 98)
  [3] → EQUATION     (m = r + 2)
  [4] → OBJECTIVE    (solve for m)

Expression Extractor:
  [1] → variables: m, r; relationship: m * r
  [2] → constraint: m * r = 98
  [3] → equation: m = r + 2
  [4] → target: m

Program Assembler builds:
  from sympy import symbols, solve
  m, r = symbols('m r')
  equations = [m * r - 98, m - (r + 2)]
  result = solve(equations, [m, r])
  answer = result[m]  # or pick positive solution

Sympy executes → m = 14 ✓
```

## Another Example

Problem: "If $f(x) = x^2 + 1$ and $g(x) = 2x - 1$, what is $f(g(3))$?"

```
IO Tagger finds clauses:
  [1] "$f(x) = x^2 + 1$"
  [2] "$g(x) = 2x - 1$"
  [3] "what is $f(g(3))$"

Classifier labels roles:
  [1] → DEFINITION   (defines f)
  [2] → DEFINITION   (defines g)
  [3] → OBJECTIVE    (evaluate f(g(3)))

Expression Extractor:
  [1] → f(x) = x**2 + 1
  [2] → g(x) = 2*x - 1
  [3] → evaluate: f(g(3))

Program Assembler builds:
  x = Symbol('x')
  f = x**2 + 1
  g = 2*x - 1
  g_of_3 = g.subs(x, 3)     # = 5
  result = f.subs(x, g_of_3)  # = 26

Sympy executes → 26 ✓
```

## Word Problem Example

Problem: "Sam earns $60 on days he works and loses $30 on days he doesn't. After 20 days he earned $660. How many days did he work?"

```
IO Tagger finds clauses:
  [1] "earns $60 on days he works"
  [2] "loses $30 on days he doesn't"
  [3] "After 20 days"
  [4] "earned $660"
  [5] "How many days did he work"

Classifier labels roles:
  [1] → EQUATION     (pay per work day = 60)
  [2] → EQUATION     (penalty per off day = 30)
  [3] → CONSTRAINT   (total days = 20)
  [4] → CONSTRAINT   (total earnings = 660)
  [5] → OBJECTIVE    (solve for work days)

Expression Extractor:
  [1] → 60 * w (w = work days)
  [2] → -30 * (total - w) (off days)
  [3] → w + off = 20
  [4] → earnings = 660
  [5] → target: w

Program Assembler builds:
  w = Symbol('w')
  earnings = 60 * w - 30 * (20 - w)
  result = solve(earnings - 660, w)

Sympy executes → w = 16 ✓
```

## Implementation Plan

### Step 1: Relabel classifier training data with ROLES

Take the existing 3,547 training pairs (clean, post-bug-fix).
Relabel from operation types (ADD, SUB, MUL, POW...) to clause roles (EQUATION, CONSTRAINT, DEFINITION, OBJECTIVE, etc.)

Heuristic relabeling:
```python
def classify_role(clause_text, operation_label):
    text = clause_text.lower()
    
    # OBJECTIVE: asking for something
    if any(w in text for w in ['find', 'what is', 'compute', 'calculate', 
                                'determine', 'how many', 'how much', 'evaluate']):
        return 'OBJECTIVE'
    
    # DEFINITION: defines a function
    if any(w in text for w in ['let ', 'define', 'given that']) or \
       re.search(r'[fg]\(x\)\s*=', clause_text):
        return 'DEFINITION'
    
    # ASSIGNMENT: states a specific value
    if re.search(r'[a-z]\s*=\s*\d', clause_text) and '(' not in clause_text:
        return 'ASSIGNMENT'
    
    # CONDITION: conditional
    if any(w in text for w in ['if ', 'when ', 'where ', 'such that', 'given']):
        return 'CONDITION'
    
    # CONSTRAINT: mentions a total, count, or specific number
    if any(w in text for w in ['total', 'sum', 'there are', 'has', 'costs']):
        return 'CONSTRAINT'
    
    # EQUATION: contains = with variables
    if '=' in clause_text and re.search(r'[a-zA-Z]', clause_text):
        return 'EQUATION'
    
    # COMPUTATION: pure math expression
    if re.search(r'\$[^$]+\$', clause_text) and operation_label != 'NO_OP':
        return 'COMPUTATION'
    
    return 'UNKNOWN'
```

### Step 2: Build Expression Extractor

For each clause, extract the mathematical content:

```python
from sympy.parsing.latex import parse_latex
from sympy import symbols, Symbol
import re

def extract_expressions(clause_text, role):
    """Extract sympy expressions from a classified clause."""
    
    # Pull all LaTeX from the clause
    latex_exprs = re.findall(r'\$([^$]+)\$', clause_text)
    latex_exprs += re.findall(r'\\\((.+?)\\\)', clause_text)
    
    parsed = []
    for latex in latex_exprs:
        try:
            expr = parse_latex(latex)
            parsed.append(expr)
        except:
            continue
    
    # Pull plain numbers
    numbers = re.findall(r'\b\d+\.?\d*\b', clause_text)
    
    return {
        'role': role,
        'expressions': parsed,
        'numbers': [float(n) for n in numbers],
        'raw_latex': latex_exprs,
        'text': clause_text
    }
```

### Step 3: Build Program Assembler

```python
from sympy import symbols, solve, Eq, Function

def assemble_program(extracted_clauses):
    """
    Assemble a sympy program from classified and extracted clauses.
    
    Returns a function that, when called, returns the answer.
    """
    equations = []
    definitions = {}
    constraints = []
    objective = None
    assignments = {}
    all_variables = set()
    
    for clause in extracted_clauses:
        role = clause['role']
        
        if role == 'DEFINITION':
            # Store function definitions
            for expr in clause['expressions']:
                if hasattr(expr, 'lhs') and hasattr(expr, 'rhs'):
                    definitions[str(expr.lhs)] = expr.rhs
        
        elif role == 'EQUATION':
            for expr in clause['expressions']:
                if hasattr(expr, 'lhs') and hasattr(expr, 'rhs'):
                    equations.append(Eq(expr.lhs, expr.rhs))
                else:
                    equations.append(expr)
                all_variables.update(expr.free_symbols)
        
        elif role == 'CONSTRAINT':
            for expr in clause['expressions']:
                all_variables.update(expr.free_symbols)
                constraints.append(expr)
            # Also check for plain numbers as constraints
            # "total of 98" → last equation = 98
        
        elif role == 'ASSIGNMENT':
            for expr in clause['expressions']:
                if hasattr(expr, 'lhs'):
                    assignments[expr.lhs] = expr.rhs
        
        elif role == 'OBJECTIVE':
            # What are we solving for?
            for expr in clause['expressions']:
                objective = expr
            # Or extract from text: "find m" → m
        
        elif role == 'COMPUTATION':
            # Direct evaluation
            for expr in clause['expressions']:
                try:
                    result = float(expr.evalf())
                    return result
                except:
                    pass
    
    # Apply assignments as substitutions
    for var, val in assignments.items():
        equations = [eq.subs(var, val) for eq in equations]
    
    # Apply definitions
    for func_name, func_expr in definitions.items():
        # Handle function composition etc.
        pass
    
    # Solve
    all_equations = equations + constraints
    if objective and all_equations:
        try:
            result = solve(all_equations, list(all_variables))
            if isinstance(result, list):
                return float(result[0]) if result else None
            elif isinstance(result, dict) and objective in result:
                return float(result[objective])
            else:
                return float(list(result.values())[0]) if result else None
        except:
            return None
    
    return None
```

### Step 4: Wire into E2E Pipeline

```python
def track2_solve(problem_text, io_tagger, classifier):
    """Full Track 2 pipeline."""
    
    # Step 1: Find clauses
    regions = io_tagger.predict(problem_text)
    
    # Step 2: Classify roles
    classified = []
    for region in regions:
        role = classifier.predict(region.text)
        classified.append((region, role))
    
    # Step 3: Extract expressions
    extracted = [extract_expressions(r.text, role) for r, role in classified]
    
    # Step 4: Assemble and execute
    answer = assemble_program(extracted)
    
    return answer
```

### Step 5: Retrain Classifier on Roles

- Relabel the 3,547 training pairs using the heuristic relabeler
- Retrain 0.5B classifier on ROLE labels instead of operation labels
- Eval on held-out set
- Run full E2E with role-based Track 2

### Step 6: Eval and Compare

Run on the same 500 MATH problems:
- Track 1 (frozen): 17.2% baseline
- Track 2 (new): ???
- Track 1 + Track 2 combined: ???

The goal: Track 2 > 17.2%. That validates the attention distillation architecture.

## What Makes This Different From Track 1

Track 1: Brute-force extract ALL LaTeX → try sympy → hope for the best
Track 2: UNDERSTAND problem structure → assemble targeted sympy program

Track 1 fails on:
- Word problems with no explicit equations
- Problems where the equation must be CONSTRUCTED from context
- Problems with multiple equations that need to be identified and combined
- Problems where not all LaTeX is relevant (distractors)

Track 2 handles these because the classifier learned (from attention distillation) which clauses play which roles. It knows "total of 98 members" is a CONSTRAINT even though there's no equation. It knows "find $m$" is the OBJECTIVE. It constructs the sympy program the way the 7B's attention structure organized the problem.

## Priority

1. Step 1 (relabel) — quick, heuristic
2. Step 3 (assembler) — the core new code
3. Step 4 (wire up) — integration
4. Step 5 (retrain) — if heuristic roles work, retrain for better accuracy
5. Step 6 (eval) — the moment of truth
