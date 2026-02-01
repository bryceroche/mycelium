"""Pattern registry with specialized prompts for each problem type."""
from dataclasses import dataclass, field
from typing import List, Optional, Callable


@dataclass
class Pattern:
    """A specialized problem-solving pattern."""
    name: str
    description: str
    prompt_template: str  # Has {problem} placeholder
    execution_type: str  # "steps", "sympy", "sympy_solve", "direct"
    examples: List[str] = field(default_factory=list)  # For embedding matching


# =============================================================================
# ARITHMETIC PATTERNS (execution_type="steps")
# =============================================================================

SEQUENTIAL = Pattern(
    name="sequential",
    description="Multi-step arithmetic problems - add, subtract, multiply, divide in sequence",
    execution_type="steps",
    prompt_template='''Break this problem into arithmetic steps. Output the actual math expressions.

Problem: {problem}

Output JSON:
{{
    "steps": [
        {{"description": "what this computes", "expr": "10 * 0.5", "result": "step1"}},
        {{"description": "next calculation", "expr": "step1 + 20", "result": "step2"}}
    ],
    "answer": "step2"
}}

RULES:
1. Each step must be ONE arithmetic operation: +, -, *, /
2. Use actual NUMBERS from the problem in "expr" - NO VARIABLES like x, y, n
3. Reference previous step results by name (step1, step2, etc.)
4. Use ** for exponents: "2 ** 3" not "2^3"

IMPORTANT:
- "80% more" means original + (original * 0.8), NOT original * 0.8
- Convert fractions to decimals: 1/2 → 0.5, 1/3 → 0.333333''',
    examples=[
        "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes muffins with 4. She sells the rest at $2 each. How much does she make?",
        "Tim has 10 apples. He gives 3 to Mary and buys 5 more. How many does he have?",
        "A store has 100 items. They sell 40 in morning and 25 in afternoon. How many left?",
        "Tom bought 3 notebooks at $4 each and 5 pens at $2 each. How much did he spend?",
        "A bus has 45 passengers. 12 get off and 8 get on. How many now?",
    ],
)

COMPLEMENT = Pattern(
    name="complement",
    description="Percentage complement - if X% did Y, find (100-X)% who didn't",
    execution_type="steps",
    prompt_template='''Solve this percentage complement problem.

Problem: {problem}

Output JSON:
{{
    "steps": [
        {{"description": "what this computes", "expr": "arithmetic expression", "result": "step1"}},
        ...
    ],
    "answer": "final_step_name"
}}

PERCENTAGE COMPLEMENT RULES:
1. If X% did something, then (100 - X)% did NOT do it
2. "40% got below B" means "60% got B and above"
3. Calculate the complement percentage first
4. Then apply to the total if needed

Use actual numbers. Each step ONE operation.''',
    examples=[
        "40% of students got below average. How many percent got average or above?",
        "30% of the cookies were eaten. What percent remain?",
        "If 15% of applicants were rejected, what percent were accepted?",
        "72% of voters supported the measure. What percent opposed it?",
        "If 88% of students passed the test, what percent failed?",
    ],
)

RATIO = Pattern(
    name="ratio",
    description="Ratio and proportion problems - split amounts or scale ratios",
    execution_type="steps",
    prompt_template='''Solve this ratio/proportion problem.

Problem: {problem}

Output JSON:
{{
    "steps": [
        {{"description": "what this computes", "expr": "arithmetic expression", "result": "step1"}},
        ...
    ],
    "answer": "final_step_name"
}}

RATIO RULES:
TYPE 1 - SPLIT RATIO (sharing in ratio X:Y):
1. Total parts = X + Y
2. Value per part = total_amount / total_parts
3. Each share = their_parts * value_per_part

TYPE 2 - KNOWN RATIO (if A:B = 2:3 and A=10, find B):
1. Scale factor = known_value / known_parts = 10 / 2 = 5
2. Other value = other_parts * scale_factor = 3 * 5 = 15

Use actual numbers. Each step ONE operation.''',
    examples=[
        "Boys to girls ratio is 2:3. If there are 10 boys, how many girls?",
        "The ratio of cats to dogs is 3:4. If there are 21 cats, how many dogs?",
        "A recipe calls for flour and sugar in a 5:2 ratio. If you use 10 cups of flour, how much sugar?",
        "Split $1400 in ratio 2:5. How much does Mike get?",
        "Paint mix requires red and white in 2:5 ratio. For 14 liters total, how much red paint?",
    ],
)

RATIO_CHAIN = Pattern(
    name="ratio_chain",
    description="Chained ratios - x/y and z/x, find z/y by multiplying",
    execution_type="steps",
    prompt_template='''Solve this ratio chain problem by multiplying/dividing ratios.

Problem: {problem}

Output JSON:
{{
    "steps": [
        {{"description": "what this computes", "expr": "arithmetic expression", "result": "step1"}},
        ...
    ],
    "answer": "final_step_name"
}}

RATIO CHAIN RULES:
1. Write down each given ratio as a fraction
2. To find a new ratio, multiply ratios that chain together
3. Cancel common terms: (x/y) * (z/x) = z/y (x cancels)

EXAMPLE: "If x/y = 2 and z/x = 4, what is z/y?"
- Chain: (z/x) * (x/y) = z/y
- Answer: 4 * 2 = 8

Use actual numbers. Each step ONE operation.''',
    examples=[
        "If x/y = 2 and z/x = 4, what is z/y?",
        "If a/b = 3 and c/a = 5, find c/b",
        "The ratio of cats to dogs is 3:1 and dogs to birds is 2:5. What is cats to birds?",
    ],
)

CONDITIONAL = Pattern(
    name="conditional",
    description="Threshold-based problems - different rates above/below a limit",
    execution_type="steps",
    prompt_template='''Solve this problem with conditional logic.

Problem: {problem}

Output JSON:
{{
    "steps": [
        {{"description": "what this computes", "expr": "arithmetic expression", "result": "step1"}},
        ...
    ],
    "answer": "final_step_name"
}}

CONDITIONAL LOGIC RULES:
1. First identify the THRESHOLD (e.g., "more than 8 hours")
2. Calculate the amount ABOVE threshold (e.g., 10 - 8 = 2 overtime hours)
3. Calculate the amount AT OR BELOW threshold (e.g., 8 regular hours)
4. Apply DIFFERENT RATES to each portion
5. Combine the results

Use actual numbers. Each step ONE operation.''',
    examples=[
        "Workers earn $10/hour for first 40 hours and $15/hour for overtime. How much for 50 hours?",
        "A phone plan charges $0.05/minute for the first 100 minutes and $0.03/minute after. Cost for 150 minutes?",
        "Tax is 10% on income up to $10,000 and 20% on income above. Tax on $15,000?",
        "A gym charges $30/month for up to 10 visits and $2 per extra visit. Cost for 15 visits?",
    ],
)

INVERSION = Pattern(
    name="inversion",
    description="Work backwards from result to find original value",
    execution_type="steps",
    prompt_template='''Solve this problem by working BACKWARDS to find an unknown.

Problem: {problem}

Output JSON:
{{
    "steps": [
        {{"description": "what this computes", "expr": "arithmetic expression", "result": "step1"}},
        ...
    ],
    "answer": "final_step_name"
}}

INVERSION RULES:
1. Identify what you KNOW (final result, amounts from sources)
2. Identify what you DON'T KNOW (the original/unknown)
3. Reverse the operations to find the unknown

EXAMPLES:
- "After 20% discount, price is $80" → original = 80 / 0.8 = 100
- "After adding 25% tax, total is $75" → original = 75 / 1.25 = 60
- "Spent half, has $15 left" → original = 15 * 2 = 30

Use actual numbers. Each step ONE operation.''',
    examples=[
        "After a 20% discount, the price was $80. What was the original price?",
        "After adding 25% tax, the total was $75. What was the pre-tax price?",
        "Maria spent half her money and then $10 more, leaving $15. How much did she start with?",
        "A number was tripled and then 10 was subtracted, giving 50. What was the original?",
        "After a 15% raise, salary became $5750. What was the original salary?",
    ],
)

COMPOSITION = Pattern(
    name="composition",
    description="Function composition f(g(x)) - evaluate from inside out",
    execution_type="steps",
    prompt_template='''Solve this function composition problem. Evaluate from INSIDE OUT.

Problem: {problem}

Output JSON:
{{
    "steps": [
        {{"description": "what this computes", "expr": "arithmetic expression", "result": "step1"}},
        ...
    ],
    "answer": "final_step_name"
}}

COMPOSITION RULES:
1. Identify the INNER function and its input value
2. Evaluate the INNER function first
3. Use that result as input to the OUTER function
4. Evaluate the OUTER function

EXAMPLE: "Let f(x) = 2x - 4 and g(x) = x^2 + 3. What is f(g(2))?"
Step 1: g(2) = 2**2 + 3 = 7
Step 2: f(7) = 2*7 - 4 = 10

Use ** for exponents. Each step ONE operation.''',
    examples=[
        "Let f(x) = 2x - 4 and g(x) = x^2 + 3. What is f(g(2))?",
        "If f(x) = 3x + 1 and g(x) = x - 2, find f(g(5))",
        "Given f(x) = x^2 and g(x) = x + 1, evaluate g(f(3))",
    ],
)

AGE_PROBLEM = Pattern(
    name="age_problem",
    description="Age word problems with past/future relationships",
    execution_type="steps",
    prompt_template='''Solve this age problem by setting up equations.

Problem: {problem}

Output JSON:
{{
    "steps": [
        {{"description": "what this computes", "expr": "arithmetic expression", "result": "step1"}},
        ...
    ],
    "answer": "final_step_name"
}}

AGE PROBLEM RULES:
1. Let x = person's current age (usually the younger person)
2. Set up equations based on relationships
3. Solve step by step

EXAMPLE: "Father is 5 times son's age. 3 years ago, sum of ages was 30. How old is son?"
- Equation: (s-3) + (5s-3) = 30 → 6s - 6 = 30 → s = 6

Use actual numbers. Each step ONE operation.''',
    examples=[
        "Father is 5 times son's age. 3 years ago, sum of ages was 30. How old is son?",
        "John is 5 years older than Mary. In 3 years, their ages will sum to 35. How old is Mary?",
        "A mother is 3 times as old as her daughter. In 10 years, she'll be twice as old. Find daughter's age.",
    ],
)

SUBSTITUTION = Pattern(
    name="substitution",
    description="Variable substitution - given a=2, b=3, evaluate expression",
    execution_type="steps",
    prompt_template='''Substitute the given values and compute the result.

Problem: {problem}

Output JSON:
{{
    "steps": [
        {{"description": "what this computes", "expr": "arithmetic expression", "result": "step1"}},
        ...
    ],
    "answer": "final_step_name"
}}

SUBSTITUTION RULES:
1. First, identify ALL given values (a=2, b=3, c=4, etc.)
2. Write out the expression with values substituted
3. Compute step by step, ONE operation at a time
4. Use ** for exponents

EXAMPLE: "If a=2, b=3, c=4, what is (b-c)^2 + a(b+c)?"
- (3-4)^2 + 2*(3+4) = 1 + 14 = 15

Use actual numbers. Each step ONE operation.''',
    examples=[
        "If a=2, b=3, c=4, what is (b-c)^2 + a(b+c)?",
        "Given x=5 and y=3, evaluate 2x^2 - 3y",
        "If $a = 2$ and $b = 5$, what is $3a + 2b$?",
    ],
)

# =============================================================================
# SYMBOLIC PATTERNS (execution_type="sympy" or "sympy_solve")
# =============================================================================

SYMBOLIC = Pattern(
    name="symbolic",
    description="Simplify, factor, or expand algebraic expressions using SymPy",
    execution_type="sympy",
    prompt_template='''Convert this algebra problem to a SymPy expression.

Problem: {problem}

Output JSON:
{{
    "operation": "simplify" | "factor" | "expand",
    "expression": "the expression in Python/SymPy syntax",
    "variable": "main variable (usually x)"
}}

CONVERSION RULES:
1. Use ** for exponents (NOT ^)
2. Use * for multiplication (even implicit: 2x → 2*x)
3. Fractions: a/b or Rational(a, b) for exact fractions
4. Square roots: sqrt(x)

EXAMPLES:
"Simplify (2s^5)/(s^3) - 6s^2"
→ {{"operation": "simplify", "expression": "(2*s**5)/(s**3) - 6*s**2", "variable": "s"}}

"Factor 30x^3 - 8x^2 + 20x"
→ {{"operation": "factor", "expression": "30*x**3 - 8*x**2 + 20*x", "variable": "x"}}''',
    examples=[
        "Simplify (2s^5)/(s^3) - 6s^2 + (7s^3)/s",
        "Factor 30x^3 - 8x^2 + 20x",
        "Expand (x+2)(x-3)",
        "Simplify (x^2 - 9)/(x - 3)",
    ],
)

EQUATION = Pattern(
    name="equation",
    description="Solve equations for a variable using SymPy",
    execution_type="sympy_solve",
    prompt_template='''Solve this equation for the variable.

Problem: {problem}

Output JSON:
{{
    "equation": "the equation in Python/SymPy syntax (use Eq(lhs, rhs))",
    "variable": "the variable to solve for (usually x)"
}}

CONVERSION RULES:
1. Use ** for exponents (NOT ^)
2. Use * for multiplication (even implicit: 2x → 2*x)
3. Use Eq(left, right) for equations

EXAMPLES:
"Solve x^2 - 9 = 0"
→ {{"equation": "Eq(x**2 - 9, 0)", "variable": "x"}}

"If 3^(x+8) = 9^(x+3), what is x?"
→ {{"equation": "Eq(3**(x+8), 9**(x+3))", "variable": "x"}}''',
    examples=[
        "Solve x^2 - 9 = 0",
        "Find x if 2x + 5 = 13",
        "If 3^(x+8) = 9^(x+3), what is x?",
        "Solve for y: 3y - 7 = 2y + 5",
    ],
)

RATIONALIZE = Pattern(
    name="rationalize",
    description="Rationalize denominators with square roots",
    execution_type="sympy",
    prompt_template='''Rationalize this expression's denominator.

Problem: {problem}

Output JSON:
{{
    "expression": "the expression in Python/SymPy syntax",
    "operation": "rationalize"
}}

CONVERSION RULES:
1. Use sqrt(n) for square roots
2. Use ** for exponents
3. Use Rational(a, b) for fractions

EXAMPLES:
"Rationalize 1/(2*sqrt(7))"
→ {{"expression": "1/(2*sqrt(7))", "operation": "rationalize"}}''',
    examples=[
        "Rationalize 1/(2*sqrt(7))",
        "Rationalize sqrt(2)/sqrt(3)",
        "Simplify 5/(3 + sqrt(2))",
    ],
)

LOGARITHM = Pattern(
    name="logarithm",
    description="Logarithm and 'what power of' problems",
    execution_type="steps",
    prompt_template='''Solve this logarithm or power problem.

Problem: {problem}

Output JSON:
{{
    "steps": [
        {{"description": "what this computes", "expr": "arithmetic expression", "result": "step1"}},
        ...
    ],
    "answer": "final_step_name"
}}

LOGARITHM RULES:
1. log_a(b) means "what power of a equals b?"
2. If a^x = b, then x = log(b) / log(a)
3. "What power of 4 equals 8?" means solve 4^x = 8

EXAMPLE: "Evaluate log_2(64)"
- 2^x = 64, x = log(64) / log(2) = 6

Use log() for natural log.''',
    examples=[
        "Evaluate log_2(64)",
        "What power of 4 equals 8?",
        "Find log_3(81)",
        "If 2^x = 32, what is x?",
    ],
)

COMPLEX = Pattern(
    name="complex",
    description="Complex number arithmetic (a+bi), where i^2=-1",
    execution_type="steps",
    prompt_template='''Solve this complex number problem. Remember i^2 = -1.

Problem: {problem}

Output JSON:
{{
    "steps": [
        {{"description": "compute real part", "expr": "expression", "result": "real"}},
        {{"description": "compute imaginary part", "expr": "expression", "result": "imag"}}
    ],
    "answer": "real + imag*i format"
}}

COMPLEX NUMBER RULES:
1. i^2 = -1
2. (a+bi)(c+di) = (ac-bd) + (ad+bc)i
3. Compute real and imaginary parts separately

EXAMPLE: "(2-2i)(5+5i)"
- Real: 2*5 - (-2)*5 = 10 + 10 = 20
- Imag: 2*5 + (-2)*5 = 0''',
    examples=[
        "Simplify (2-2i)(5+5i)",
        "Evaluate (1+2i)(6-3i)",
        "What is (3+4i) + (2-i)?",
        "Compute (5-3i)(2+i)",
    ],
)

SEQUENCE = Pattern(
    name="sequence",
    description="Arithmetic sequence - nth term, first negative, common difference",
    execution_type="steps",
    prompt_template='''Solve this arithmetic sequence problem.

Problem: {problem}

Output JSON:
{{
    "steps": [
        {{"description": "what this computes", "expr": "expression", "result": "step1"}}
    ],
    "answer": "final_step_name"
}}

ARITHMETIC SEQUENCE FORMULAS:
1. nth term: a_n = a_1 + (n-1)*d
2. First negative term: find smallest n where a_n < 0
3. d = second_term - first_term

EXAMPLE: "In 1000, 987, 974, ... which term is first negative?"
- a_1 = 1000, d = -13
- Solve: 1000 + (n-1)*(-13) < 0
- n > 77.9, so n = 78''',
    examples=[
        "In sequence 1000, 987, 974, ... find which term is first negative",
        "Find the 10th term of 3, 7, 11, 15, ...",
        "What is the common difference in 5, 12, 19, 26?",
    ],
)

VIETA = Pattern(
    name="vieta",
    description="Sum/product of roots using Vieta's formulas",
    execution_type="steps",
    prompt_template='''Solve using Vieta's formulas for quadratic equations.

Problem: {problem}

Output JSON:
{{
    "steps": [
        {{"description": "identify coefficients", "expr": "expression", "result": "step1"}}
    ],
    "answer": "final_step_name"
}}

VIETA'S FORMULAS for ax^2 + bx + c = 0:
- Sum of roots: x1 + x2 = -b/a
- Product of roots: x1 * x2 = c/a

Be careful: "sum of all possible values" may mean unique solutions only.''',
    examples=[
        "Sum of all values of x such that 2x(x-10) = -50",
        "Find the sum of the roots of x^2 - 5x + 6 = 0",
        "What is the product of the roots of 2x^2 + 3x - 5 = 0?",
    ],
)

SYSTEM = Pattern(
    name="system",
    description="System of equations with multiple unknowns",
    execution_type="sympy_solve",
    prompt_template='''Solve this system of equations.

Problem: {problem}

Output JSON:
{{
    "equations": ["Eq(a+b, 8)", "Eq(b+c, -3)", ...],
    "unknowns": ["a", "b", "c"],
    "find": "what expression to compute (e.g., a*b*c)"
}}

SYSTEM SOLVING:
1. Express all equations in Eq(lhs, rhs) form
2. List all unknowns
3. Specify what to find

EXAMPLE: "a+b=8, b+c=-3, a+c=-5, find abc"
→ {{"equations": ["Eq(a+b, 8)", "Eq(b+c, -3)", "Eq(a+c, -5)"], "unknowns": ["a", "b", "c"], "find": "a*b*c"}}''',
    examples=[
        "If a+b=8, b+c=-3, and a+c=-5, find abc",
        "Solve: 2x + y = 10, x - y = 1",
        "If x + 2y = 7 and 3x - y = 8, find x",
    ],
)

EXPONENT = Pattern(
    name="exponent",
    description="Exponent equations - solve for x in a^b = c^x",
    execution_type="steps",
    prompt_template='''Solve this exponent/power equation.

Problem: {problem}

Output JSON:
{{
    "steps": [
        {{"description": "what this computes", "expr": "expression", "result": "step1"}}
    ],
    "answer": "final_step_name"
}}

EXPONENT RULES:
1. Convert bases to same base: 4=2^2, 8=2^3, 9=3^2, 27=3^3
2. Use log for harder cases: x = log(value) / log(base)
3. (a^m)^n = a^(mn)

EXAMPLE: "If 2^8 = 4^x, what is x?"
- 4 = 2^2, so 4^x = 2^(2x)
- 2^8 = 2^(2x), so 8 = 2x, x = 4''',
    examples=[
        "If 2^8 = 4^x, what is x?",
        "(17^6 - 17^5) / 16 = 17^x, find x",
        "Solve 9^n = 27^2",
    ],
)

CIRCLE = Pattern(
    name="circle",
    description="Circle radius by completing the square",
    execution_type="direct",
    prompt_template='''Find the radius of this circle by completing the square.

Problem: {problem}

COMPLETE THE SQUARE:
- x^2 + 8x → (x+4)^2 - 16
- y^2 - 6y → (y-3)^2 - 9
- Then r^2 = sum of constants

Output JSON with just the numeric radius:
{{"answer": "5"}}''',
    examples=[
        "Find the radius of x^2 + 8x + y^2 - 6y = 0",
        "What is the radius of the circle x^2 + y^2 - 4x + 6y - 12 = 0?",
    ],
)

MIDPOINT = Pattern(
    name="midpoint",
    description="Midpoint formula problems",
    execution_type="direct",
    prompt_template='''Solve this midpoint problem.

Problem: {problem}

MIDPOINT FORMULA: midpoint = ((x1+x2)/2, (y1+y2)/2)
If midpoint M=(mx,my) and one point is (x1,y1), then:
- x2 = 2*mx - x1
- y2 = 2*my - y1

Output JSON with the coordinate pair:
{{"answer": "(15,-11)"}}''',
    examples=[
        "If the midpoint of (3,7) and B is (9,-2), what is B?",
        "Find the midpoint of (2,4) and (8,10)",
    ],
)

ALGEBRA = Pattern(
    name="algebra",
    description="Word problems requiring algebraic equation setup",
    execution_type="steps",
    prompt_template='''Solve this algebra problem by setting up and solving an equation.

Problem: {problem}

Output JSON:
{{
    "steps": [
        {{"description": "what this computes", "expr": "arithmetic expression", "result": "step1"}},
        ...
    ],
    "answer": "final_step_name"
}}

ALGEBRA RULES:
1. Identify the UNKNOWN (e.g., "previous income", "original price")
2. Set up the EQUATION relating before and after states
3. Solve step by step

Use actual numbers. Each step ONE operation.''',
    examples=[
        "Spent 40% on rent. Income increased by $600. Now rent is 25% of new income. Find previous income.",
        "A number increased by 20% equals 60. What was the original?",
        "Three times a number minus 7 equals 20. What is the number?",
    ],
)


# =============================================================================
# PATTERN REGISTRY
# =============================================================================

PATTERNS = {
    # Arithmetic (steps)
    "sequential": SEQUENTIAL,
    "complement": COMPLEMENT,
    "ratio": RATIO,
    "ratio_chain": RATIO_CHAIN,
    "conditional": CONDITIONAL,
    "inversion": INVERSION,
    "composition": COMPOSITION,
    "age_problem": AGE_PROBLEM,
    "substitution": SUBSTITUTION,
    "logarithm": LOGARITHM,
    "complex": COMPLEX,
    "sequence": SEQUENCE,
    "vieta": VIETA,
    "exponent": EXPONENT,
    "algebra": ALGEBRA,
    # SymPy
    "symbolic": SYMBOLIC,
    "equation": EQUATION,
    "rationalize": RATIONALIZE,
    "system": SYSTEM,
    # Direct
    "circle": CIRCLE,
    "midpoint": MIDPOINT,
}


def get_pattern(name: str) -> Optional[Pattern]:
    """Get a pattern by name."""
    return PATTERNS.get(name)
