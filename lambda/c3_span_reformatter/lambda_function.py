"""
Lambda Function: C3 Span Reformatter

Reformats existing C3 training pairs into span extraction format.
For each (problem_text, template, expression) triple:
1. Parse gold expression to find operands
2. Find each operand's span in problem text
3. Check WORD_TO_NUMBER mapping if not found directly
4. Mark operands that can't be found as "generated"

Input: S3 key to C3 training chunk
Output: Span extraction training pairs
"""

import json
import re
import boto3
from typing import List, Dict, Tuple, Optional, Set

s3 = boto3.client('s3')
BUCKET = 'mycelium-data'
OUTPUT_PREFIX = 'c3_span_training/chunks/'

# Word to number mapping
WORD_TO_NUMBER = {
    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
    'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
    'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
    'eighteen': '18', 'nineteen': '19', 'twenty': '20',
    'thirty': '30', 'forty': '40', 'fifty': '50', 'sixty': '60',
    'seventy': '70', 'eighty': '80', 'ninety': '90', 'hundred': '100',
    'thousand': '1000', 'million': '1000000',
    'half': '0.5', 'quarter': '0.25', 'third': '1/3',
    'first': '1', 'second': '2', 'twice': '2', 'double': '2', 'triple': '3',
}

# Common domain constants (orphaned values not in problem text)
DOMAIN_CONSTANTS = {
    '60': 'min/hr', '24': 'hr/day', '7': 'day/wk',
    '52': 'wk/yr', '12': 'mo/yr', '365': 'day/yr',
    '100': 'percent', '1000': 'kilo',
    '3.14159': 'pi', '2.71828': 'e',
}


def extract_operands(expr_str: str) -> Set[str]:
    """
    Extract operands (numbers and variables) from expression string.
    Uses regex to avoid sympy evaluation which collapses 15+23 to 38.
    Returns set of operand strings.
    """
    operands = set()

    # Function names to exclude
    func_names = {'sqrt', 'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
                  'log', 'ln', 'exp', 'factorial', 'binomial', 'abs',
                  'floor', 'ceil', 'mod', 'gcd', 'lcm'}

    # Extract numbers (integers, decimals, fractions like 1/3)
    # Match: 123, 12.5, -45, 1/3
    num_pattern = r'-?\d+(?:\.\d+)?(?:/\d+)?'
    nums = re.findall(num_pattern, expr_str)
    for num in nums:
        # Clean up: remove leading zeros except for "0" itself
        clean = num.lstrip('0') or '0'
        if clean.startswith('.'):
            clean = '0' + clean
        if clean.startswith('-0') and len(clean) > 2:
            clean = '-' + clean[2:].lstrip('0')
        operands.add(clean)

    # Extract variables (letters not part of function names)
    # First, remove function calls to avoid picking up function names
    expr_no_funcs = expr_str
    for func in func_names:
        expr_no_funcs = re.sub(r'\b' + func + r'\s*\(', '(', expr_no_funcs, flags=re.IGNORECASE)

    # Now extract single-letter or short variables
    var_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
    vars_found = re.findall(var_pattern, expr_no_funcs)
    for v in vars_found:
        if v.lower() not in func_names and len(v) <= 3:
            operands.add(v)

    return operands


def find_span_in_text(operand: str, text: str) -> Optional[Dict]:
    """
    Find the span of an operand in the problem text.
    Returns dict with span info or None if not found.
    """
    text_lower = text.lower()

    # Try direct match first (for numbers)
    if operand.lstrip('-').replace('.', '').isdigit():
        # Look for the number in text
        pattern = r'\b' + re.escape(operand) + r'\b'
        match = re.search(pattern, text)
        if match:
            return {
                'operand': operand,
                'span_start': match.start(),
                'span_end': match.end(),
                'span_text': match.group(),
                'source': 'direct'
            }

        # Try without leading zeros/signs
        clean_operand = operand.lstrip('0').lstrip('-') or '0'
        pattern = r'\b' + re.escape(clean_operand) + r'\b'
        match = re.search(pattern, text)
        if match:
            return {
                'operand': operand,
                'span_start': match.start(),
                'span_end': match.end(),
                'span_text': match.group(),
                'source': 'direct_cleaned'
            }

    # Try variable match (single letter variables)
    if len(operand) == 1 and operand.isalpha():
        # Look for $x$ or just x in context
        patterns = [
            r'\$' + re.escape(operand) + r'\$',  # $x$
            r'\b' + re.escape(operand) + r'\b',   # standalone x
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return {
                    'operand': operand,
                    'span_start': match.start(),
                    'span_end': match.end(),
                    'span_text': match.group(),
                    'source': 'variable'
                }

    # Try word-to-number mapping
    for word, num in WORD_TO_NUMBER.items():
        if num == operand or (operand.lstrip('-') == num):
            pattern = r'\b' + re.escape(word) + r'\b'
            match = re.search(pattern, text_lower)
            if match:
                # Get actual text (preserve case)
                start, end = match.start(), match.end()
                return {
                    'operand': operand,
                    'span_start': start,
                    'span_end': end,
                    'span_text': text[start:end],
                    'source': 'word_to_number',
                    'word': word
                }

    # Check if it's a domain constant
    clean_op = operand.lstrip('-')
    if clean_op in DOMAIN_CONSTANTS:
        return {
            'operand': operand,
            'span_start': -1,
            'span_end': -1,
            'span_text': '',
            'source': 'domain_constant',
            'constant_type': DOMAIN_CONSTANTS[clean_op]
        }

    return None


def reformat_pair(pair: Dict) -> Optional[Dict]:
    """
    Reformat a single C3 training pair into span extraction format.
    """
    input_text = pair.get('input', '')
    output_expr = pair.get('output', '')
    template = pair.get('template', '')

    # Extract problem text (remove [TEMPLATE: X] prefix)
    problem_text = re.sub(r'^\[TEMPLATE:\s*\w+\]\s*', '', input_text).strip()

    if not problem_text or not output_expr:
        return None

    # Extract operands from expression
    operands = extract_operands(output_expr)

    if not operands:
        return None

    # Find spans for each operand
    spans = []
    found_count = 0
    generated_count = 0

    for operand in operands:
        span_info = find_span_in_text(operand, problem_text)

        if span_info:
            spans.append(span_info)
            if span_info['source'] != 'domain_constant':
                found_count += 1
            else:
                generated_count += 1
        else:
            # Mark as generated (not found in text)
            spans.append({
                'operand': operand,
                'span_start': -1,
                'span_end': -1,
                'span_text': '',
                'source': 'generated'
            })
            generated_count += 1

    # Build output
    reformatted = {
        'problem_text': problem_text,
        'template': template,
        'expression': output_expr,
        'operands': list(operands),
        'spans': spans,
        'n_operands': len(operands),
        'n_found': found_count,
        'n_generated': generated_count,
        'all_found': generated_count == 0,
        # Preserve original metadata
        'problem_idx': pair.get('problem_idx'),
        'step_idx': pair.get('step_idx'),
    }

    return reformatted


def lambda_handler(event, context):
    """
    Process a chunk of C3 training data and reformat for span extraction.

    Event:
        chunk_key: S3 key to C3 training chunk
        chunk_idx: Index of this chunk (for output naming)
    """
    chunk_key = event.get('chunk_key')
    chunk_idx = event.get('chunk_idx', 0)

    if not chunk_key:
        return {'error': 'chunk_key required'}

    # Load chunk
    try:
        response = s3.get_object(Bucket=BUCKET, Key=chunk_key)
        content = response['Body'].read().decode('utf-8')

        # Handle both JSON array and JSONL formats
        if content.strip().startswith('['):
            pairs = json.loads(content)
        else:
            pairs = [json.loads(line) for line in content.strip().split('\n') if line.strip()]

    except Exception as e:
        return {'error': f'Failed to load {chunk_key}: {str(e)}'}

    # Process each pair
    reformatted_pairs = []
    stats = {
        'total': len(pairs),
        'success': 0,
        'failed': 0,
        'all_found': 0,
        'partial_found': 0,
        'none_found': 0,
        'by_template': {},
    }

    for pair in pairs:
        result = reformat_pair(pair)

        if result:
            reformatted_pairs.append(result)
            stats['success'] += 1

            # Track span finding stats
            if result['all_found']:
                stats['all_found'] += 1
            elif result['n_found'] > 0:
                stats['partial_found'] += 1
            else:
                stats['none_found'] += 1

            # Track by template
            template = result['template']
            if template not in stats['by_template']:
                stats['by_template'][template] = {'total': 0, 'all_found': 0}
            stats['by_template'][template]['total'] += 1
            if result['all_found']:
                stats['by_template'][template]['all_found'] += 1
        else:
            stats['failed'] += 1

    # Write results to S3
    output_key = f"{OUTPUT_PREFIX}chunk_{chunk_idx:04d}.jsonl"

    # Write as JSONL
    output_lines = [json.dumps(p) for p in reformatted_pairs]
    output_content = '\n'.join(output_lines)

    s3.put_object(
        Bucket=BUCKET,
        Key=output_key,
        Body=output_content,
        ContentType='application/json'
    )

    return {
        'success': True,
        'chunk_key': chunk_key,
        'chunk_idx': chunk_idx,
        'output_key': output_key,
        'stats': stats,
    }


# Local testing
if __name__ == '__main__':
    # Test cases
    test_pairs = [
        {
            'input': '[TEMPLATE: ADD] Tom has 15 apples. Jane has 23 apples. How many do they have together?',
            'output': '15+23',
            'template': 'ADD',
            'problem_idx': 0,
            'step_idx': 0,
        },
        {
            'input': '[TEMPLATE: MUL] A box contains twelve oranges. There are five boxes. How many oranges in total?',
            'output': '12*5',
            'template': 'MUL',
            'problem_idx': 1,
            'step_idx': 0,
        },
        {
            'input': '[TEMPLATE: DIV] The sum of two numbers is $30$. Let $x$ be one number.',
            'output': 'x-y',
            'template': 'DIV',
            'problem_idx': 2,
            'step_idx': 0,
        },
    ]

    print("Testing span reformatter:")
    for pair in test_pairs:
        result = reformat_pair(pair)
        if result:
            print(f"\nTemplate: {result['template']}")
            print(f"Expression: {result['expression']}")
            print(f"Operands: {result['operands']}")
            print(f"Found: {result['n_found']}/{result['n_operands']}")
            for span in result['spans']:
                print(f"  {span['operand']}: {span['source']} -> '{span['span_text']}'")
