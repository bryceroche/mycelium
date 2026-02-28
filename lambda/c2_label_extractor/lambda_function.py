"""
Lambda Function: C2 Label Extractor

Extracts multi-label operation types from MATH problem CoT text.
Pattern-based extraction - no sympy needed.

Input: S3 key to IAF chunk
Output: C2 training data with 15 operational labels per problem
"""

import json
import re
import boto3
from typing import Set, List, Dict

s3 = boto3.client('s3')
BUCKET = 'mycelium-data'
OUTPUT_PREFIX = 'c2_training/chunks/'

# 15 Operational Labels
LABELS = [
    "FACTORIAL", "LOG", "TRIG", "MOD",
    "SQRT", "CUBE", "FRAC_POW", "HIGH_POW", "SQUARE",
    "EQUATION", "DIV", "MUL", "ADD", "OTHER",
]

# Pattern-based extraction (fast, no dependencies)
PATTERNS = {
    'FACTORIAL': [r'\d+!', r'factorial', r'permutation', r'combination', r'\\binom', r'choose'],
    'LOG': [r'\\log', r'\\ln', r'\blog\b', r'\bln\b', r'logarithm', r'exponent'],
    'TRIG': [r'\\sin', r'\\cos', r'\\tan', r'\\cot', r'\\sec', r'\\csc',
             r'\bsin\b', r'\bcos\b', r'\btan\b', r'trigonometr', r'angle'],
    'MOD': [r'\\mod', r'\\pmod', r'\bmod\b', r'modulo', r'remainder when', r'divisible'],
    'SQRT': [r'\\sqrt', r'square root', r'√', r'radical'],
    'CUBE': [r'\^3\b', r'\^{3}', r'cubed', r'cube root', r'cubic'],
    'FRAC_POW': [r'\^\{?\\frac', r'\^\{?\d+/\d+\}?', r'root'],
    'HIGH_POW': [r'\^[4-9]\b', r'\^{[4-9]}', r'\^\d{2,}', r'\^{1[0-9]}', r'power'],
    'SQUARE': [r'\^2\b', r'\^{2}', r'squared', r'square of', r'quadratic'],
    'EQUATION': [r'\\le', r'\\ge', r'\\eq', r'solve', r'equation', r'solution'],
    'DIV': [r'\\frac', r'\\div', r'÷', r'divided by', r'ratio', r'fraction'],
    'MUL': [r'\\times', r'\\cdot', r'×', r'·', r'multiply', r'product'],
    'ADD': [r'\+', r'\\pm', r'\bsum\b', r'\btotal\b', r'\badd\b', r'plus', r'minus', r'subtract', r'difference'],
}


def extract_labels(text: str) -> List[str]:
    """Extract operation labels from CoT text using patterns."""
    labels = set()

    for label, patterns in PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                labels.add(label)
                break

    if not labels:
        labels.add('OTHER')

    return sorted(labels)


def lambda_handler(event, context):
    """
    Process a single IAF chunk and extract C2 training data.

    Event:
        chunk_key: S3 key to IAF chunk file
    """
    chunk_key = event.get('chunk_key')
    if not chunk_key:
        return {'error': 'chunk_key required'}

    # Load chunk
    try:
        response = s3.get_object(Bucket=BUCKET, Key=chunk_key)
        data = json.loads(response['Body'].read().decode('utf-8'))
    except Exception as e:
        return {'error': f'Failed to load {chunk_key}: {str(e)}'}

    # Process each problem
    results = []
    label_counts = {}

    for problem in data:
        cot = problem.get('generated_cot', '')
        problem_text = problem.get('problem_text', '')

        labels = extract_labels(cot)

        results.append({
            'problem_idx': problem.get('problem_idx'),
            'text': problem_text,
            'labels': labels,
            'level': problem.get('level'),
            'type': problem.get('type'),
        })

        for lbl in labels:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

    # Extract chunk number from key
    # e.g., "iaf_extraction/chunked/instance1_iaf_v3_gpu0_valid_chunk_042.json"
    import os
    chunk_name = os.path.basename(chunk_key).replace('.json', '')

    # Write results to S3
    output_key = f"{OUTPUT_PREFIX}{chunk_name}_c2.json"
    output = {
        'source_chunk': chunk_key,
        'n_problems': len(results),
        'label_counts': label_counts,
        'examples': results,
    }

    s3.put_object(
        Bucket=BUCKET,
        Key=output_key,
        Body=json.dumps(output),
        ContentType='application/json'
    )

    return {
        'success': True,
        'chunk_key': chunk_key,
        'n_problems': len(results),
        'label_counts': label_counts,
        'output_key': output_key,
    }


# Local testing
if __name__ == '__main__':
    # Test on a sample
    test_cot = """
    To solve this, we first find the square root of 16.
    \\sqrt{16} = 4
    Then we multiply by 3: 3 \\times 4 = 12
    Adding 5 gives us: 12 + 5 = 17
    """

    labels = extract_labels(test_cot)
    print(f"Labels: {labels}")
    # Expected: ['ADD', 'MUL', 'SQRT']
