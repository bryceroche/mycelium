"""
C1 Training Data Cleaning for v6

Two fixes:
1. LaTeX preprocessing - normalize LaTeX to readable math
2. Negative audit - remove mislabeled negatives
"""

import json
import re
from pathlib import Path
from collections import defaultdict


def preprocess_latex(text: str) -> str:
    """
    Normalize LaTeX notation to human-readable math.
    Apply transformations in order.
    """
    result = text

    # Remove \left and \right (keep the delimiters)
    result = re.sub(r'\\left\s*([(\[{|])', r'\1', result)
    result = re.sub(r'\\right\s*([)\]}|])', r'\1', result)

    # Fractions: \frac{a}{b}, \dfrac{a}{b}, \tfrac{a}{b} -> a/b
    # Handle nested braces carefully
    def replace_frac(match):
        # Extract numerator and denominator, handling nested braces
        content = match.group(1)
        # Find the numerator (first {...})
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

    # Match \frac, \dfrac, \tfrac followed by content
    frac_pattern = r'\\[dt]?frac(.{0,200})'
    for _ in range(5):  # Multiple passes for nested fractions
        old = result
        result = re.sub(frac_pattern, replace_frac, result)
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

    # Square root: \sqrt{a} -> sqrt(a), \sqrt[n]{a} -> root(a,n)
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

    # Text commands: \text{...}, \mathrm{...}, \mathbf{...} -> just content
    result = re.sub(r'\\text\{([^}]*)\}', r'\1', result)
    result = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', result)
    result = re.sub(r'\\mathbf\{([^}]*)\}', r'\1', result)
    result = re.sub(r'\\textbf\{([^}]*)\}', r'\1', result)
    result = re.sub(r'\\textit\{([^}]*)\}', r'\1', result)
    result = re.sub(r'\\emph\{([^}]*)\}', r'\1', result)
    result = re.sub(r'\\mbox\{([^}]*)\}', r'\1', result)
    result = re.sub(r'\\hbox\{([^}]*)\}', r'\1', result)

    # Remove common spacing/formatting commands
    result = re.sub(r'\\[,;:!]', ' ', result)  # \, \; \: \!
    result = re.sub(r'\\quad', ' ', result)
    result = re.sub(r'\\qquad', '  ', result)
    result = re.sub(r'\\hspace\{[^}]*\}', ' ', result)
    result = re.sub(r'\\vspace\{[^}]*\}', '', result)
    result = re.sub(r'\\\\', '\n', result)  # Line break

    # Remove display math delimiters but keep content
    result = re.sub(r'\\\[', '', result)
    result = re.sub(r'\\\]', '', result)
    result = re.sub(r'\\\(', '', result)
    result = re.sub(r'\\\)', '', result)

    # Handle inline math delimiters - keep the content
    # $...$ stays as is (tokenizer handles it)

    # Remove remaining backslash commands that don't affect meaning
    # Keep letters after backslash as-is (e.g., \pi -> pi, \alpha -> alpha)
    result = re.sub(r'\\([a-zA-Z]+)', r'\1', result)

    # Clean up extra whitespace
    result = re.sub(r'[ \t]+', ' ', result)
    result = re.sub(r'\n\s*\n', '\n', result)
    result = result.strip()

    return result


# Keywords and patterns for detecting math operations
OPERATION_KEYWORDS = [
    'add', 'added', 'adding', 'adds',
    'subtract', 'subtracted', 'subtracting', 'subtracts',
    'multiply', 'multiplied', 'multiplying', 'multiplies',
    'divide', 'divided', 'dividing', 'divides',
    'times', 'plus', 'minus',
    'split', 'splits', 'splitting',
    'share', 'shared', 'shares', 'sharing',
    'combine', 'combined', 'combines', 'combining',
    'total', 'totals', 'totaling',
    'sum', 'sums', 'summing',
    'difference',
    'product',
    'ratio', 'ratios',
    'percent', 'percentage',
    'fraction of', 'half of', 'third of', 'quarter of',
    'twice', 'double', 'triple',
    'each', 'per',
    'how many', 'how much',
]

OPERATOR_CHARS = ['+', '-', '×', '÷', '*', '/', '±']


def audit_negative(record: dict) -> dict:
    """
    Audit a negative (no boundaries) record for potential mislabeling.
    Returns audit info dict.
    """
    text = record.get('problem_text', '')
    text_lower = text.lower()

    flags = []

    # Count distinct numbers (integers and decimals)
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    distinct_numbers = len(set(numbers))
    if distinct_numbers >= 2:
        flags.append(f'has_{distinct_numbers}_numbers')

    # Check for operation keywords
    found_keywords = []
    for kw in OPERATION_KEYWORDS:
        if kw in text_lower:
            found_keywords.append(kw)
    if found_keywords:
        flags.append(f'keywords:{",".join(found_keywords[:3])}')

    # Count fraction expressions (after LaTeX preprocessing)
    cleaned = preprocess_latex(text)
    fractions = re.findall(r'\d+/\d+', cleaned)
    if len(fractions) >= 2:
        flags.append(f'has_{len(fractions)}_fractions')

    # Check for explicit operators
    found_ops = [op for op in OPERATOR_CHARS if op in cleaned]
    if len(found_ops) >= 1:
        # Only flag if combined with numbers
        if distinct_numbers >= 2:
            flags.append(f'operators:{",".join(found_ops)}')

    # Check for "of" pattern (often indicates multiplication)
    of_patterns = re.findall(r'\b(?:fraction|half|third|quarter|fifth|\d+/\d+)\s+of\b', text_lower)
    if of_patterns:
        flags.append('has_of_pattern')

    is_flagged = len(flags) >= 2  # Need at least 2 signals to flag

    return {
        'idx': record.get('idx', -1),
        'is_flagged': is_flagged,
        'flags': flags,
        'distinct_numbers': distinct_numbers,
        'n_keywords': len(found_keywords),
        'text_preview': text[:200],
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True)
    parser.add_argument('--report-file', type=str, required=True)
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.input_file}...")
    records = []
    with open(args.input_file) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records")

    # Separate positives and negatives
    positives = [r for r in records if r.get('has_boundaries', False)]
    negatives = [r for r in records if not r.get('has_boundaries', False)]

    print(f"  Positives: {len(positives)}")
    print(f"  Negatives: {len(negatives)}")

    # Step 1: Preprocess all LaTeX
    print("\nStep 1: LaTeX Preprocessing...")
    for i, r in enumerate(records):
        original = r['problem_text']
        cleaned = preprocess_latex(original)
        r['problem_text'] = cleaned
        r['original_text'] = original  # Keep original for reference

        if i < 3:
            print(f"\nExample {i+1}:")
            print(f"  Original: {original[:150]}...")
            print(f"  Cleaned:  {cleaned[:150]}...")

    print(f"\nPreprocessed {len(records)} records")

    # Step 2: Audit negatives
    print("\nStep 2: Auditing True Negatives...")
    audit_results = []
    flagged_negatives = []
    clean_negatives = []

    for i, r in enumerate(negatives):
        r['idx'] = i
        audit = audit_negative(r)
        audit_results.append(audit)

        if audit['is_flagged']:
            flagged_negatives.append(r)
        else:
            clean_negatives.append(r)

    print(f"\nAudit Results:")
    print(f"  Total negatives: {len(negatives)}")
    print(f"  Flagged (likely mislabeled): {len(flagged_negatives)}")
    print(f"  Clean (kept): {len(clean_negatives)}")

    # Show flagged examples
    print("\nFlagged Examples (first 10):")
    for audit in audit_results[:20]:
        if audit['is_flagged']:
            print(f"  Flags: {audit['flags']}")
            print(f"  Text: {audit['text_preview'][:150]}...")
            print()

    # Step 3: Build cleaned dataset
    print("\nStep 3: Building Cleaned Dataset...")
    cleaned_records = positives + clean_negatives

    print(f"\nFinal dataset:")
    print(f"  Positives: {len(positives)}")
    print(f"  Clean negatives: {len(clean_negatives)}")
    print(f"  Total: {len(cleaned_records)}")
    print(f"  Removed: {len(flagged_negatives)}")

    # Write output
    print(f"\nWriting cleaned data to {args.output_file}...")
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_file, 'w') as f:
        for r in cleaned_records:
            # Remove temporary fields
            if 'idx' in r:
                del r['idx']
            f.write(json.dumps(r) + '\n')

    # Write report
    print(f"Writing report to {args.report_file}...")

    # Aggregate flag statistics
    flag_counts = defaultdict(int)
    for audit in audit_results:
        for flag in audit['flags']:
            flag_type = flag.split(':')[0] if ':' in flag else flag
            flag_counts[flag_type] += 1

    report = {
        'original_total': len(records),
        'original_positives': len(positives),
        'original_negatives': len(negatives),
        'flagged_negatives': len(flagged_negatives),
        'clean_negatives': len(clean_negatives),
        'final_total': len(cleaned_records),
        'removal_rate': len(flagged_negatives) / len(negatives) if negatives else 0,
        'flag_statistics': dict(flag_counts),
        'flagged_examples': [
            {
                'flags': a['flags'],
                'text_preview': a['text_preview'],
            }
            for a in audit_results if a['is_flagged']
        ][:50],
        'latex_preprocessing': {
            'conversions_applied': [
                'frac{a}{b} -> a/b',
                'binom{a}{b} -> binomial(a,b)',
                'sqrt{a} -> sqrt(a)',
                'sqrt[n]{a} -> root(a,n)',
                'times -> ×',
                'div -> ÷',
                'cdot -> ·',
                'text{...} -> content',
                'removed formatting commands',
            ]
        }
    }

    with open(args.report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print("\nDone!")
    print(f"\nSummary:")
    print(f"  Input: {len(records)} records")
    print(f"  Output: {len(cleaned_records)} records")
    print(f"  Removed: {len(flagged_negatives)} mislabeled negatives ({100*len(flagged_negatives)/len(negatives):.1f}%)")


if __name__ == '__main__':
    main()
