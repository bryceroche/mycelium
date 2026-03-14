# Selector: Replaces Slot Tagger + Resolver
# Merges two components into one. Selection task, not generation.
# One model, one job: "which state table rows does this step need?"

# ============================================================
# THE TASK
# ============================================================
#
# Input: narration (from Narrator) + numbered state table (prior steps)
# Output: which state entries this step uses, by index
#
# This is MULTIPLE CHOICE, not generation. The answer space is
# the state table (typically 2-8 entries) plus LITERAL for constants.
#
# ============================================================
# INPUT/OUTPUT FORMAT (line-based, same convention as all models)
# ============================================================
#
# === SELECTOR INPUT ===
# Narration: Substitute x^2 + y^2 = 90 into the expansion
# State:
#   1: State the equation x^2 + y^2 = 90 → 90
#   2: State the equation xy = 27 → 27
#   3: Expand (x+y)^2 → x^2 + 2xy + y^2
# Select:
#
# === SELECTOR OUTPUT ===
# ARG_1 STATE 3
# ARG_2 STATE 1
# END
#
# === LITERAL EXAMPLE ===
# Narration: Multiply the radius by 2
# State:
#   1: Find the radius of the circle → 5
# Select:
#
# Output:
# ARG_1 STATE 1
# ARG_2 LITERAL 2
# END
#
# === SINGLE ARG EXAMPLE ===
# Narration: Evaluate the expression
# State:
#   1: State x^2 + y^2 = 90 → 90
#   2: After substitution → 90 + 54
# Select:
#
# Output:
# ARG_1 STATE 2
# END

# ============================================================
# EXTRACTION SPEC
# ============================================================
#
# Source: existing ~50K CoT-parsed examples (parsed_steps.jsonl)
# Each step has: narrator_text, operands with sources, result_value
#
# For each problem, walk steps in order, maintaining state table.

import json
import re
from pathlib import Path


def extract_selector_examples(parsed_steps_path: str, output_path: str):
    """
    Extract Selector training data from parsed CoT steps.
    
    For each step in each problem:
      - Build state table from all prior steps (narrator_text + result)
      - Identify which state entries this step's operands came from
      - Identify which operands are literals (not from any prior step)
      - Format as Selector input/output pair
    
    No SymPy needed. No normalization beyond the existing LaTeX cleanup.
    Should run in seconds.
    """
    examples = []
    
    with open(parsed_steps_path) as f:
        problems = [json.loads(line) for line in f]
    
    for problem in problems:
        state_table = []  # list of (narrator_text, result_value)
        
        for step_idx, step in enumerate(problem['steps']):
            narrator_text = step['narrator_text']  # from LoRA C output
            operands = step.get('operands', [])
            result_value = step.get('result', '')
            
            # Skip first step if it's GIVEN with no prior state
            # (GIVEN steps ADD to state table, they don't SELECT from it)
            # But GIVEN steps that reference problem text DO select
            
            if len(state_table) > 0 and len(operands) > 0:
                # Build input
                input_text = f"Narration: {narrator_text}\n"
                input_text += "State:\n"
                for i, (desc, val) in enumerate(state_table):
                    input_text += f"  {i+1}: {desc} → {val}\n"
                input_text += "Select:\n"
                
                # Build output: map each operand to state entry or LITERAL
                output_lines = []
                for arg_idx, operand in enumerate(operands):
                    arg_label = f"ARG_{arg_idx + 1}"
                    
                    if operand.get('source') == 'literal' or operand.get('is_literal'):
                        # Literal constant from narration
                        output_lines.append(
                            f"{arg_label} LITERAL {operand['value']}"
                        )
                    elif operand.get('source_step_id') is not None:
                        # Reference to prior step result
                        # Convert step_id to 1-indexed state table position
                        state_idx = operand['source_step_id'] + 1
                        if 1 <= state_idx <= len(state_table):
                            output_lines.append(
                                f"{arg_label} STATE {state_idx}"
                            )
                        else:
                            # Skip malformed references
                            continue
                    elif operand.get('source_step') is not None:
                        # Alternative field name for step reference
                        state_idx = operand['source_step'] + 1
                        if 1 <= state_idx <= len(state_table):
                            output_lines.append(
                                f"{arg_label} STATE {state_idx}"
                            )
                        else:
                            continue
                    else:
                        # Try to match by value against state table
                        matched = False
                        op_val = str(operand.get('value', ''))
                        for i, (desc, val) in enumerate(state_table):
                            if str(val) == op_val and op_val:
                                output_lines.append(
                                    f"{arg_label} STATE {i+1}"
                                )
                                matched = True
                                break
                        if not matched and op_val:
                            # Assume literal if no state match
                            output_lines.append(
                                f"{arg_label} LITERAL {op_val}"
                            )
                
                if output_lines:
                    output_lines.append("END")
                    output_text = "\n".join(output_lines)
                    
                    examples.append({
                        'input': input_text,
                        'output': output_text,
                        'problem_id': problem.get('problem_id', ''),
                        'step_idx': step_idx,
                    })
            
            # Add this step to state table for future steps
            state_table.append((narrator_text, result_value))
    
    # Write examples
    with open(output_path, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')
    
    print(f"Extracted {len(examples)} selector examples")
    print(f"  Problems: {len(problems)}")
    print(f"  Avg examples/problem: {len(examples)/len(problems):.1f}")
    
    # Sanity checks
    state_counts = []
    arg_counts = []
    literal_counts = []
    for ex in examples:
        state_lines = [l for l in ex['input'].split('\n') if l.strip().startswith(('1:', '2:', '3:', '4:', '5:', '6:', '7:', '8:', '9:'))]
        state_counts.append(len(state_lines))
        args = [l for l in ex['output'].split('\n') if l.startswith('ARG_')]
        arg_counts.append(len(args))
        lits = [l for l in ex['output'].split('\n') if 'LITERAL' in l]
        literal_counts.append(len(lits))
    
    print(f"  State table size: mean={sum(state_counts)/len(state_counts):.1f}, "
          f"max={max(state_counts)}")
    print(f"  Args per step: mean={sum(arg_counts)/len(arg_counts):.1f}, "
          f"max={max(arg_counts)}")
    print(f"  Literal rate: {sum(literal_counts)/sum(arg_counts)*100:.1f}%")
    
    return examples


def format_for_training(example: dict, tokenizer=None) -> dict:
    """
    Format a single example into the chat template for Qwen-0.5B training.
    
    CRITICAL: This exact function must be used at both training AND inference.
    Import it, don't rewrite it. (Bug #16)
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a parameter selector. Given a narration and a numbered "
                "state table of prior results, select which state entries this "
                "step needs. Output ARG_N STATE <number> or ARG_N LITERAL <value>. "
                "End with END."
            )
        },
        {
            "role": "user", 
            "content": example['input']
        },
        {
            "role": "assistant",
            "content": example['output']
        }
    ]
    return messages


# ============================================================
# INFERENCE
# ============================================================

def run_selector(model, tokenizer, narration: str, state_table: list) -> list:
    """
    Run selector at inference time.
    
    Args:
        narration: narrator output for this step
        state_table: list of (narrator_text, result_value) from prior steps
    
    Returns:
        list of (arg_idx, source_type, source_value) tuples
        source_type is 'STATE' or 'LITERAL'
        source_value is state_idx (int) or literal string
    """
    # Build input — MUST match training format exactly (Bug #16)
    input_text = f"Narration: {narration}\n"
    input_text += "State:\n"
    for i, (desc, val) in enumerate(state_table):
        input_text += f"  {i+1}: {desc} → {val}\n"
    input_text += "Select:\n"
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are a parameter selector. Given a narration and a numbered "
                "state table of prior results, select which state entries this "
                "step needs. Output ARG_N STATE <number> or ARG_N LITERAL <value>. "
                "End with END."
            )
        },
        {
            "role": "user",
            "content": input_text
        }
    ]
    
    # Generate with stop sequences
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,  # small — selection is short
        do_sample=False,
        # Stop at END or double newline
        eos_token_id=tokenizer.eos_token_id,
    )
    
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    )
    
    # Parse output
    bindings = []
    for line in response.strip().split('\n'):
        line = line.strip()
        if line == 'END':
            break
        
        match = re.match(r'ARG_(\d+)\s+(STATE|LITERAL)\s+(.+)', line)
        if match:
            arg_idx = int(match.group(1))
            source_type = match.group(2)
            source_value = match.group(3).strip()
            
            if source_type == 'STATE':
                source_value = int(source_value)
            
            bindings.append((arg_idx, source_type, source_value))
    
    return bindings


def resolve_bindings(bindings: list, state_table: list) -> dict:
    """
    Dereference STATE bindings to actual values.
    Returns dict for Translator input: {ARG_1: value, ARG_2: value, ...}
    """
    resolved = {}
    for arg_idx, source_type, source_value in bindings:
        key = f"ARG_{arg_idx}"
        if source_type == 'LITERAL':
            resolved[key] = source_value
        elif source_type == 'STATE':
            # 1-indexed to 0-indexed
            if 1 <= source_value <= len(state_table):
                _, value = state_table[source_value - 1]
                resolved[key] = value
            else:
                resolved[key] = f"<INVALID_STATE_{source_value}>"
    return resolved


# ============================================================
# TRANSLATOR INPUT FORMAT (updated for Selector output)
# ============================================================
#
# === TRANSLATOR (LoRA D) ===
# Input:
#     Narration: Substitute x^2 + y^2 = 90 into the expansion
#     Values:
#       ARG_1 = x^2 + 2xy + y^2
#       ARG_2 = 90
#     Expression:
# Output:
#     90 + 2*27
#
# Values come from resolve_bindings() — the Translator never
# sees state table indices, only concrete values.


# ============================================================
# TRAINING CONFIG
# ============================================================
#
# Same as all other LoRAs:
#   Base:       Qwen/Qwen2.5-0.5B (fresh, never continue previous)
#   LoRA:       r=16, alpha=32, dropout=0.05, Q/K/V/O
#   Epochs:     3
#   Batch:      4, grad_accum 8 (effective 32)
#   LR:         2e-4, cosine schedule
#   dtype:      float32 (A10G)
#   Split:      problem-level (never step-level)
#
# max_new_tokens=64 at inference (selection is short)
# Stop sequences: ["END", "\n\n", "Human:"]


if __name__ == '__main__':
    import sys
    
    input_path = sys.argv[1] if len(sys.argv) > 1 else 'parsed_steps.jsonl'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'selector_train.jsonl'
    
    examples = extract_selector_examples(input_path, output_path)
    
    # Print a few examples for verification
    print("\n=== SAMPLE EXAMPLES ===")
    for ex in examples[:3]:
        print(f"\nINPUT:\n{ex['input']}")
        print(f"OUTPUT:\n{ex['output']}")
        print("---")
