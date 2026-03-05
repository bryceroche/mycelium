"""
LaTeX Preprocessor for C1 Inference

Standalone module for preprocessing LaTeX in math problem texts.
Import this at inference time to normalize problem text before tokenization.

Usage:
    from latex_preprocessor import preprocess_latex

    text = "What is $\\frac{1}{2}$ of $\\frac{3}{4}$?"
    cleaned = preprocess_latex(text)
    # Result: "What is $1/2$ of $3/4$?"
"""

import re


def preprocess_latex(text: str) -> str:
    """
    Normalize LaTeX notation to human-readable math.

    Conversions applied (in order):
    - \\frac{a}{b}, \\dfrac{a}{b}, \\tfrac{a}{b} -> a/b
    - \\binom{a}{b} -> binomial(a,b)
    - \\sqrt{a} -> sqrt(a)
    - \\sqrt[n]{a} -> root(a,n)
    - a^{b} -> a^b
    - a_{b} -> a_b
    - \\times -> ×
    - \\div -> ÷
    - \\cdot -> ·
    - \\pm -> ±
    - \\leq, \\le -> ≤
    - \\geq, \\ge -> ≥
    - \\neq, \\ne -> ≠
    - \\left( \\right) -> ( )
    - \\text{...}, \\mathrm{...}, \\mathbf{...} -> content only
    - Remaining \\commands -> command name without backslash
    """
    result = text

    # Remove \left and \right (keep delimiters)
    result = re.sub(r'\\left\s*([(\[{|])', r'\1', result)
    result = re.sub(r'\\right\s*([)\]}|])', r'\1', result)

    # Fractions: \frac{a}{b}, \dfrac{a}{b}, \tfrac{a}{b} -> a/b
    def replace_frac(match):
        content = match.group(1)
        depth = 0
        num_start = num_end = denom_start = denom_end = -1
        for i, c in enumerate(content):
            if c == '{':
                if depth == 0 and num_start == -1:
                    num_start = i + 1
                elif depth == 0 and denom_start == -1:
                    denom_start = i + 1
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0 and num_end == -1:
                    num_end = i
                elif depth == 0 and denom_end == -1:
                    denom_end = i
                    break

        if num_start != -1 and num_end != -1 and denom_start != -1 and denom_end != -1:
            num = content[num_start:num_end]
            denom = content[denom_start:denom_end]
            # Recursively process nested fractions
            num = preprocess_latex(num)
            denom = preprocess_latex(denom)
            # Add parens if complex
            if ' ' in num or '+' in num or '-' in num:
                num = f'({num})'
            if ' ' in denom or '+' in denom or '-' in denom:
                denom = f'({denom})'
            return f'{num}/{denom}'
        return match.group(0)

    # Multiple passes for nested fractions
    for _ in range(5):
        old = result
        result = re.sub(r'\\[dt]?frac(.{0,200})', replace_frac, result)
        if result == old:
            break

    # Binomial: \binom{a}{b} -> binomial(a,b)
    def replace_binom(match):
        content = match.group(1)
        depth = 0
        n_start = n_end = k_start = k_end = -1
        for i, c in enumerate(content):
            if c == '{':
                if depth == 0 and n_start == -1:
                    n_start = i + 1
                elif depth == 0 and k_start == -1:
                    k_start = i + 1
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0 and n_end == -1:
                    n_end = i
                elif depth == 0 and k_end == -1:
                    k_end = i
                    break

        if n_start != -1 and n_end != -1 and k_start != -1 and k_end != -1:
            n = content[n_start:n_end]
            k = content[k_start:k_end]
            return f'binomial({n},{k})'
        return match.group(0)

    result = re.sub(r'\\binom(.{0,100})', replace_binom, result)

    # Square roots
    result = re.sub(r'\\sqrt\[([^\]]+)\]\{([^}]+)\}', r'root(\2,\1)', result)
    result = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', result)

    # Exponents and subscripts: a^{b} -> a^b, a_{b} -> a_b
    result = re.sub(r'\^{([^}]+)}', r'^\1', result)
    result = re.sub(r'_{([^}]+)}', r'_\1', result)

    # Operators
    result = result.replace('\\times', '×')
    result = result.replace('\\div', '÷')
    result = result.replace('\\cdot', '·')
    result = result.replace('\\pm', '±')
    result = result.replace('\\mp', '∓')
    result = result.replace('\\leq', '≤')
    result = result.replace('\\geq', '≥')
    result = result.replace('\\neq', '≠')
    result = result.replace('\\le', '≤')
    result = result.replace('\\ge', '≥')
    result = result.replace('\\ne', '≠')
    result = result.replace('\\ldots', '...')
    result = result.replace('\\cdots', '...')
    result = result.replace('\\dots', '...')
    result = result.replace('\\infty', '∞')

    # Text commands: \text{...}, \mathrm{...}, etc -> just content
    result = re.sub(r'\\text\{([^}]*)\}', r'\1', result)
    result = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', result)
    result = re.sub(r'\\mathbf\{([^}]*)\}', r'\1', result)
    result = re.sub(r'\\textbf\{([^}]*)\}', r'\1', result)
    result = re.sub(r'\\textit\{([^}]*)\}', r'\1', result)
    result = re.sub(r'\\emph\{([^}]*)\}', r'\1', result)
    result = re.sub(r'\\mbox\{([^}]*)\}', r'\1', result)
    result = re.sub(r'\\hbox\{([^}]*)\}', r'\1', result)

    # Remove spacing/formatting commands
    result = re.sub(r'\\[,;:!]', ' ', result)  # \, \; \: \!
    result = re.sub(r'\\quad', ' ', result)
    result = re.sub(r'\\qquad', '  ', result)
    result = re.sub(r'\\hspace\{[^}]*\}', ' ', result)
    result = re.sub(r'\\vspace\{[^}]*\}', '', result)
    result = re.sub(r'\\\\', '\n', result)  # Line break

    # Remove display math delimiters
    result = re.sub(r'\\\[', '', result)
    result = re.sub(r'\\\]', '', result)
    result = re.sub(r'\\\(', '', result)
    result = re.sub(r'\\\)', '', result)

    # Remaining backslash commands -> command name only
    result = re.sub(r'\\([a-zA-Z]+)', r'\1', result)

    # Clean up whitespace
    result = re.sub(r'[ \t]+', ' ', result)
    result = re.sub(r'\n\s*\n', '\n', result)
    result = result.strip()

    return result


# Test examples
if __name__ == '__main__':
    test_cases = [
        (r"What is $\frac{1}{2}$ of $\frac{3}{4}$?", "What is $1/2$ of $3/4$?"),
        (r"Evaluate $\binom{5}{2}$", "Evaluate $binomial(5,2)$"),
        (r"Find $\sqrt{16}$ and $\sqrt[3]{27}$", "Find $sqrt(16)$ and $root(27,3)$"),
        (r"$x^{2} + y^{2} = z^{2}$", "$x^2 + y^2 = z^2$"),
        (r"$a \times b \div c$", "$a × b ÷ c$"),
        (r"$x \leq 5$ and $y \geq 3$", "$x ≤ 5$ and $y ≥ 3$"),
        (r"$\text{let } x = 5$", "$let  x = 5$"),
        (r"$\dfrac{a+b}{c-d}$", "$(a+b)/(c-d)$"),
    ]

    print("Testing LaTeX preprocessor:\n")
    for original, expected in test_cases:
        result = preprocess_latex(original)
        status = "✓" if expected in result or result == expected else "✗"
        print(f"{status} Input:    {original}")
        print(f"  Output:   {result}")
        print(f"  Expected: {expected}")
        print()
