#!/usr/bin/env python3
"""Diagnose cycle 2: print correct vs wrong predictions with problem details."""
import sys
sys.path.insert(0, '.')
import json
import torch
import torch.nn.functional as F
from scripts.atom_lora import AtomLoRAModel, AnswerHead, AtomAdditiveLoRAManager

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
model = AtomLoRAModel()
model.compressor = model.compressor.to(device=device, dtype=torch.bfloat16)
model.hypernet = model.hypernet.to(device=device, dtype=torch.bfloat16)
model.atoms = model.atoms.to(device=device, dtype=torch.bfloat16)
model.confidence_head = model.confidence_head.to(device)
model.residual_gate = model.residual_gate.to(device=device, dtype=torch.bfloat16)
model.probe_head = model.probe_head.to(device)
model.message_generator = model.message_generator.to(device)
answer_head = AnswerHead(page_size=model.page_size).to(device)

# Load checkpoint
ckpt = torch.load('checkpoints/per_cycle_L4_best.pt', map_location='cpu')
for name in ['compressor', 'atoms', 'hypernet', 'confidence_head',
             'answer_head', 'residual_gate', 'message_generator']:
    if name in ckpt:
        obj = answer_head if name == 'answer_head' else getattr(model, name)
        own = obj.state_dict()
        for k, v in ckpt[name].items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
        obj.load_state_dict(own, strict=False)
print(f"Loaded checkpoint (accuracy={ckpt.get('accuracy', '?')}%)")

# Load eval data
eval_path = 'data/per_cycle/L4_eval.jsonl'
samples = []
with open(eval_path) as f:
    for line in f:
        line = line.strip()
        if line:
            samples.append(json.loads(line))
print(f"Loaded {len(samples)} eval problems\n")

model.eval()
answer_head.eval()

correct_examples = []
wrong_examples = []

with torch.no_grad():
    for s in samples:
        problem = s['problem']
        cycle_targets = s['cycle_targets']
        if len(cycle_targets) < 2:
            continue
        gold_c1 = cycle_targets[0]
        gold_c2 = cycle_targets[1]
        gold_final = s['final_answer']

        inputs = model.tokenizer(
            [problem], return_tensors='pt', padding=True,
            truncation=True, max_length=192,
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        state_pages = []
        messages = []
        mid_states_history = []

        for pass_num in range(3):
            # Text injection for cycle 2+
            if pass_num > 0 and len(state_pages) > 0:
                prev_pred = answer_head.decode(state_pages[-1].float())
                ctx = f"Step {pass_num} result: {int(prev_pred[0].item())}\n"
                aug = ctx + problem
                aug_inp = model.tokenizer(
                    [aug], return_tensors='pt', padding=True,
                    truncation=True, max_length=212,
                )
                eval_ids = aug_inp['input_ids'].to(device)
                eval_mask = aug_inp['attention_mask'].to(device)
            else:
                eval_ids = input_ids
                eval_mask = attention_mask

            page, _scales, mid_states, message = model.thinking_pass(
                eval_ids, eval_mask, state_pages, pass_num,
                prev_mid_states=mid_states_history if mid_states_history else None,
                messages=messages if messages else None,
            )
            state_pages.append(page)
            messages.append(message)
            mid_states_history.append(mid_states)

        # Decode predictions
        pred_c1 = int(answer_head.decode(state_pages[0].float())[0].item())
        pred_c2 = int(answer_head.decode(state_pages[1].float())[0].item())

        entry = {
            'problem': problem,
            'gold_c1': gold_c1, 'pred_c1': pred_c1, 'c1_correct': pred_c1 == gold_c1,
            'gold_c2': gold_c2, 'pred_c2': pred_c2, 'c2_correct': pred_c2 == gold_c2,
            'gold_final': gold_final,
            'gen_targets': s.get('cycle_gen_targets', []),
        }

        if pred_c2 == gold_c2:
            correct_examples.append(entry)
        else:
            wrong_examples.append(entry)

print(f"Cycle 2: {len(correct_examples)} correct, {len(wrong_examples)} wrong "
      f"({100*len(correct_examples)/(len(correct_examples)+len(wrong_examples)):.1f}%)")

print("\n" + "="*80)
print("CYCLE 2 CORRECT (what it CAN do):")
print("="*80)
for e in correct_examples[:15]:
    print(f"\n  Problem: {e['problem'][:90]}...")
    print(f"  C1: gold={e['gold_c1']}, pred={e['pred_c1']} {'✓' if e['c1_correct'] else '✗'}")
    print(f"  C2: gold={e['gold_c2']}, pred={e['pred_c2']} ✓")
    print(f"  Operation: {e['gold_c1']} → {e['gold_c2']} (delta={e['gold_c2']-e['gold_c1']})")

print("\n" + "="*80)
print("CYCLE 2 WRONG (what it CANNOT do):")
print("="*80)
for e in wrong_examples[:15]:
    print(f"\n  Problem: {e['problem'][:90]}...")
    print(f"  C1: gold={e['gold_c1']}, pred={e['pred_c1']} {'✓' if e['c1_correct'] else '✗'}")
    print(f"  C2: gold={e['gold_c2']}, pred={e['pred_c2']} ✗ (off by {abs(e['gold_c2']-e['pred_c2'])})")

# Summary stats
if correct_examples:
    c_c1 = [e['gold_c1'] for e in correct_examples]
    c_c2 = [e['gold_c2'] for e in correct_examples]
    print(f"\n\nCORRECT stats: c1 range [{min(c_c1)}-{max(c_c1)}], c2 range [{min(c_c2)}-{max(c_c2)}]")
    print(f"  c1 mean={sum(c_c1)/len(c_c1):.0f}, c2 mean={sum(c_c2)/len(c_c2):.0f}")

if wrong_examples:
    w_c1 = [e['gold_c1'] for e in wrong_examples]
    w_c2 = [e['gold_c2'] for e in wrong_examples]
    w_off = [abs(e['gold_c2'] - e['pred_c2']) for e in wrong_examples]
    print(f"\nWRONG stats: c1 range [{min(w_c1)}-{max(w_c1)}], c2 range [{min(w_c2)}-{max(w_c2)}]")
    print(f"  c1 mean={sum(w_c1)/len(w_c1):.0f}, c2 mean={sum(w_c2)/len(w_c2):.0f}")
    print(f"  off-by: mean={sum(w_off)/len(w_off):.0f}, median={sorted(w_off)[len(w_off)//2]}")
    # What does it predict instead?
    pred_c2s = [e['pred_c2'] for e in wrong_examples[:50]]
    gold_c1s = [e['gold_c1'] for e in wrong_examples[:50]]
    matches_c1 = sum(1 for p, g in zip(pred_c2s, gold_c1s) if p == g)
    print(f"  pred_c2 == gold_c1 (copying cycle 1): {matches_c1}/{min(50, len(wrong_examples))}")
