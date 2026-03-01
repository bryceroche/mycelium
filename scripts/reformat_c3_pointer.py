#!/usr/bin/env python3
"""
Reformat C3 training data for pointer model.

Converts (problem_text, template, expression) triples into provenance labels.
Uses character positions from existing span data - token conversion happens during training.

No tokenizer needed here - just restructures the data with provenance labels.
"""

import json
import boto3

s3 = boto3.client('s3')
BUCKET = 'mycelium-data'

# Implicit value mappings
IMPLICIT_VALUES = {
    'half': 0.5, 'quarter': 0.25, 'third': 1/3,
    'double': 2, 'twice': 2, 'triple': 3, 'thrice': 3,
    'dozen': 12, 'score': 20, 'gross': 144,
    'percent': 100, 'century': 100,
}

# Domain constants
DOMAIN_CONSTANTS = {
    60: 'min/hr', 24: 'hr/day', 7: 'day/wk',
    52: 'wk/yr', 12: 'mo/yr', 365: 'day/yr',
    100: 'percent', 1000: 'kilo',
}

# Source type mapping
SOURCE_TYPE_MAP = {
    'direct': 'TEXT',
    'direct_cleaned': 'TEXT',
    'variable': 'TEXT',
    'word_to_number': 'IMPLICIT',
    'domain_constant': 'CONSTANT',
    'generated': 'PRIOR',
}


def determine_provenance(operand: str, span_info: dict) -> dict:
    """
    Determine the provenance label for an operand.
    Returns dict with source_type, char_position, value, etc.
    Token position conversion happens during training.
    """
    source = span_info.get('source', 'unknown')
    char_start = span_info.get('span_start', -1)
    char_end = span_info.get('span_end', -1)

    # Try to parse numeric value
    try:
        if '/' in str(operand):
            parts = str(operand).split('/')
            value = float(parts[0]) / float(parts[1])
        else:
            value = float(operand)
    except:
        value = operand  # Keep as string for variables

    # Determine source type
    source_type = SOURCE_TYPE_MAP.get(source, 'TEXT')

    result = {
        'operand': operand,
        'source_type': source_type,
        'value': value,
        'char_start': char_start,
        'char_end': char_end,
    }

    # Add type-specific info
    if source_type == 'IMPLICIT':
        word = span_info.get('word', '')
        result['word'] = word
        result['label'] = f'IMPLICIT_{word}'
    elif source_type == 'CONSTANT':
        const_type = span_info.get('constant_type', '')
        result['constant_type'] = const_type
        result['label'] = f'CONSTANT_{int(value) if isinstance(value, (int, float)) else value}'
    elif source_type == 'PRIOR':
        result['prior_index'] = 0  # Track A has no priors
        result['label'] = 'PRIOR_0'
    else:  # TEXT
        result['label'] = f'TEXT_CHAR_{char_start}'  # Will convert to token pos during training

    return result


def reformat_example(example: dict) -> dict:
    """
    Reformat a single training example for pointer model.
    """
    problem_text = example.get('problem_text', '')
    template = example.get('template', '')
    expression = example.get('expression', '')
    operands = example.get('operands', [])
    spans = example.get('spans', [])

    # Determine provenance for each operand
    provenance_labels = []
    for operand, span_info in zip(operands, spans):
        prov = determine_provenance(operand, span_info)
        provenance_labels.append(prov)

    return {
        'problem_text': problem_text,
        'template': template,
        'expression': expression,
        'operands': operands,
        'provenance': provenance_labels,
        'n_operands': len(operands),
        'source_types': [p['source_type'] for p in provenance_labels],
        'labels': [p['label'] for p in provenance_labels],
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", default="a,b", help="Which tracks to process: a, b, or a,b")
    args = parser.parse_args()

    tracks = args.tracks.split(',')
    all_examples = []

    # Load Track A
    if 'a' in tracks:
        print("Loading Track A data...")
        response = s3.get_object(Bucket=BUCKET, Key='c3_span_training/track_a.jsonl')
        content = response['Body'].read().decode('utf-8')
        track_a = [json.loads(line) for line in content.strip().split('\n') if line.strip()]
        print(f"  Track A: {len(track_a)} examples")
        all_examples.extend(track_a)

    # Load Track B
    if 'b' in tracks:
        print("Loading Track B data...")
        response = s3.get_object(Bucket=BUCKET, Key='c3_span_training/track_b.jsonl')
        content = response['Body'].read().decode('utf-8')
        track_b = [json.loads(line) for line in content.strip().split('\n') if line.strip()]
        print(f"  Track B: {len(track_b)} examples")
        all_examples.extend(track_b)

    examples = all_examples
    print(f"Total: {len(examples)} examples")

    # Reformat each example
    print("Reformatting for pointer model...")
    reformatted = []
    source_type_counts = {'TEXT': 0, 'PRIOR': 0, 'IMPLICIT': 0, 'CONSTANT': 0}

    for i, ex in enumerate(examples):
        try:
            ref = reformat_example(ex)
            reformatted.append(ref)

            # Count source types
            for st in ref['source_types']:
                source_type_counts[st] = source_type_counts.get(st, 0) + 1

        except Exception as e:
            print(f"  Error on example {i}: {e}")

    print(f"\nReformatted {len(reformatted)} examples")
    print(f"Source type distribution: {source_type_counts}")

    # Show samples
    print("\n--- Sample reformatted examples ---")
    for ex in reformatted[:5]:
        print(f"\nTemplate: {ex['template']}")
        print(f"Expression: {ex['expression']}")
        print(f"Operands: {ex['operands']}")
        print(f"Labels: {ex['labels']}")
        print(f"Provenance: {ex['provenance']}")

    # Upload to S3
    track_suffix = '_'.join(sorted(tracks))
    output_key = f'c3_pointer_training/track_{track_suffix}_pointer.jsonl'
    output_content = '\n'.join(json.dumps(ex) for ex in reformatted)
    s3.put_object(Bucket=BUCKET, Key=output_key, Body=output_content)
    print(f"\nUploaded to s3://{BUCKET}/{output_key}")

    return reformatted


if __name__ == '__main__':
    main()
