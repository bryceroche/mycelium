"""
Lambda Function: Heartbeat Calibration

Tests different min_run_length values on sample problems.
Returns heartbeat counts for each min_run value so we can compare with human counts.
"""

import json
import boto3

s3 = boto3.client('s3')
BUCKET = 'mycelium-data'
THRESHOLD = 0.5
HEAD = 'L22H4'


def count_heartbeats(iaf_traces, min_run):
    """Count heartbeats with given min_run_length."""
    if not iaf_traces:
        return 0
    values = [t.get(HEAD, 0.5) for t in iaf_traces]
    binary = [1 if v >= THRESHOLD else 0 for v in values]
    run_len = 0
    heartbeats = 0
    for b in binary:
        if b == 0:
            run_len += 1
        else:
            if run_len >= min_run:
                heartbeats += 1
            run_len = 0
    if run_len >= min_run:
        heartbeats += 1
    return heartbeats


def lambda_handler(event, context):
    """
    Calibrate heartbeat detection on sample problems.

    Event:
        chunk_key: S3 key to IAF chunk
        sample_indices: list of problem indices to test (default: first 2 per level)
        min_run_values: list of min_run values to test (default: [3, 5, 7, 10])
    """
    chunk_key = event.get('chunk_key', 'iaf_extraction/chunked/instance1_iaf_v3_gpu0_valid_chunk_000.json')
    min_run_values = event.get('min_run_values', [3, 5, 7, 10])

    # Load chunk
    try:
        response = s3.get_object(Bucket=BUCKET, Key=chunk_key)
        data = json.loads(response['Body'].read().decode('utf-8'))
    except Exception as e:
        return {'error': f'Failed to load {chunk_key}: {str(e)}'}

    # Group by level and pick 2 per level
    by_level = {}
    for p in data:
        level = p.get('level', 'Unknown')
        if level not in by_level:
            by_level[level] = []
        by_level[level].append(p)

    samples = []
    for level in ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']:
        if level in by_level:
            samples.extend(by_level[level][:2])

    # Run calibration
    results = []
    for p in samples:
        traces = p.get('iaf_traces', [])
        cot = p.get('generated_cot', '')

        # Count heartbeats for each min_run value
        hb_counts = {}
        for min_run in min_run_values:
            hb_counts[f'min_{min_run}'] = count_heartbeats(traces, min_run)

        results.append({
            'problem_idx': p.get('problem_idx'),
            'level': p.get('level'),
            'type': p.get('type'),
            'num_tokens': len(traces),
            'cot_preview': cot[:500],  # First 500 chars for manual review
            'problem_text': p.get('problem_text', '')[:300],
            **hb_counts
        })

    return {
        'success': True,
        'n_samples': len(results),
        'min_run_values': min_run_values,
        'results': results
    }


if __name__ == '__main__':
    # Local test
    result = lambda_handler({}, None)
    print(json.dumps(result, indent=2))
