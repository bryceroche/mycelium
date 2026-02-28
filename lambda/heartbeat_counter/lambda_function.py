"""
Lambda Function: MATH Heartbeat Counter

Counts heartbeats (sustained low-IAF computation runs) in MATH problem traces.
Each heartbeat = one computation step by the teacher model.

Input: S3 key to IAF chunk
Output: Heartbeat counts per problem

Parameters:
  - threshold: 0.5 (IAF below this = computing)
  - min_run_length: 10 tokens (ignore micro-flips)
  - head: L22H4 (top computing head from analysis)
"""

import json
import boto3

s3 = boto3.client('s3')
BUCKET = 'mycelium-data'
OUTPUT_PREFIX = 'pulse_analysis/chunks/'

# Parameters - min_run_length=5 calibrated to match human step counts
THRESHOLD = 0.5
MIN_RUN_LENGTH = 5
HEAD = 'L22H4'


def count_heartbeats(iaf_traces: list) -> tuple:
    """
    Count heartbeats in IAF trace.

    Returns: (heartbeat_count, computing_ratio)
    """
    if not iaf_traces:
        return 0, 0.0

    # Extract head values
    values = [t.get(HEAD, 0.5) for t in iaf_traces]

    # Threshold to binary: 1 = reading (high IAF), 0 = computing (low IAF)
    binary = [1 if v >= THRESHOLD else 0 for v in values]

    # Count runs of 0s with length >= MIN_RUN_LENGTH
    run_len = 0
    heartbeats = 0

    for b in binary:
        if b == 0:
            run_len += 1
        else:
            if run_len >= MIN_RUN_LENGTH:
                heartbeats += 1
            run_len = 0

    # Check final run
    if run_len >= MIN_RUN_LENGTH:
        heartbeats += 1

    # Computing ratio = fraction of tokens in computing state
    computing_ratio = sum(1 for b in binary if b == 0) / len(binary)

    return heartbeats, computing_ratio


def lambda_handler(event, context):
    """
    Process a single IAF chunk and count heartbeats.

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
    total_heartbeats = 0

    for problem in data:
        iaf_traces = problem.get('iaf_traces', [])
        heartbeats, computing_ratio = count_heartbeats(iaf_traces)

        result = {
            'problem_idx': problem.get('problem_idx'),
            'heartbeat_count': heartbeats,
            'computing_ratio': round(computing_ratio, 4),
            'num_tokens': len(iaf_traces),
            'level': problem.get('level'),
            'type': problem.get('type'),
        }
        results.append(result)
        total_heartbeats += heartbeats

    # Extract chunk name
    import os
    chunk_name = os.path.basename(chunk_key).replace('.json', '')

    # Write results to S3
    output_key = f"{OUTPUT_PREFIX}{chunk_name}_heartbeats.json"
    output = {
        'source_chunk': chunk_key,
        'n_problems': len(results),
        'total_heartbeats': total_heartbeats,
        'mean_heartbeats': round(total_heartbeats / len(results), 2) if results else 0,
        'parameters': {
            'threshold': THRESHOLD,
            'min_run_length': MIN_RUN_LENGTH,
            'head': HEAD,
        },
        'results': results,
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
        'total_heartbeats': total_heartbeats,
        'mean_heartbeats': output['mean_heartbeats'],
        'output_key': output_key,
    }


# Local testing
if __name__ == '__main__':
    # Simulate a trace
    test_trace = [
        {'L22H4': 0.9},  # reading
        {'L22H4': 0.8},
        {'L22H4': 0.3},  # computing start
        {'L22H4': 0.2},
        {'L22H4': 0.1},
        {'L22H4': 0.2},
        {'L22H4': 0.1},
        {'L22H4': 0.3},
        {'L22H4': 0.2},
        {'L22H4': 0.1},
        {'L22H4': 0.2},
        {'L22H4': 0.1},  # 10 tokens computing
        {'L22H4': 0.9},  # reading again
        {'L22H4': 0.8},
    ]

    heartbeats, ratio = count_heartbeats(test_trace)
    print(f"Heartbeats: {heartbeats}, Computing ratio: {ratio:.2f}")
    # Expected: 1 heartbeat (10 consecutive low-IAF tokens)
