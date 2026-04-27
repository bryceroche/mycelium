#!/usr/bin/env python3
"""Convert multi-step GSM8K into single-step training examples.

Each step of each problem becomes a standalone 1-cycle problem.
The model learns to parse and compute ONE operation on diverse GSM8K language.
Master this before chaining.

Usage:
  python scripts/generate_single_step_gsm8k.py
"""
import json, os

def convert_to_single_steps(input_path, output_path):
    """Convert multi-step JSONL to single-step JSONL."""
    samples = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line.strip()))

    single_steps = []
    for s in samples:
        problem = s['problem']
        targets = s['cycle_targets']
        gen_targets = s.get('cycle_gen_targets', [str(t) for t in targets])

        for i in range(len(targets)):
            # Each step becomes a standalone 1-cycle problem
            gen_target = gen_targets[i] if i < len(gen_targets) else str(targets[i])

            # Add #### marker if not present
            if '####' not in gen_target:
                gen_target = f"{gen_target} #### {targets[i]}"

            single_steps.append({
                'problem': problem,
                'cycle_targets': [targets[i]],
                'cycle_gen_targets': [gen_target],
                'final_answer': targets[i],
                'num_steps': 1,
                'original_step': i,
                'original_num_steps': len(targets),
            })

    with open(output_path, 'w') as f:
        for step in single_steps:
            f.write(json.dumps(step) + '\n')

    return len(single_steps)


if __name__ == '__main__':
    data_dir = 'data/per_cycle'

    # Training data
    train_in = os.path.join(data_dir, 'gsm8k_train.jsonl')
    train_out = os.path.join(data_dir, 'gsm8k_single_step_train.jsonl')
    n_train = convert_to_single_steps(train_in, train_out)
    print(f"Train: {n_train} single-step examples from {train_in}")

    # Eval data (keep multi-step for proper eval)
    eval_in = os.path.join(data_dir, 'gsm8k_eval.jsonl')
    eval_out = os.path.join(data_dir, 'gsm8k_single_step_eval.jsonl')
    n_eval = convert_to_single_steps(eval_in, eval_out)
    print(f"Eval: {n_eval} single-step examples from {eval_in}")
