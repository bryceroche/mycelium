#!/usr/bin/env python3
"""Inspect cycle inputs/outputs for a few problems. Shows what each cycle sees and predicts."""
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
import glob
ckpt_path = 'checkpoints/per_cycle_L4.5_best.pt'
if not glob.glob(ckpt_path):
    ckpt_path = 'checkpoints/per_cycle_L4_best.pt'
    print(f"L4.5 checkpoint not found, using {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location='cpu')
for name in ['compressor', 'atoms', 'hypernet', 'confidence_head',
             'answer_head', 'residual_gate', 'message_generator']:
    if name in ckpt:
        obj = answer_head if name == 'answer_head' else getattr(model, name)
        own = obj.state_dict()
        for k, v in ckpt[name].items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
        obj.load_state_dict(own, strict=False)
print(f"Loaded checkpoint: {ckpt_path} (accuracy={ckpt.get('accuracy', '?')}%)")

# Load eval data
eval_path = 'data/per_cycle/L4.5_eval.jsonl'
if not glob.glob(eval_path):
    eval_path = 'data/per_cycle/L4_eval.jsonl'
samples = []
with open(eval_path) as f:
    for line in f:
        line = line.strip()
        if line:
            samples.append(json.loads(line))
print(f"Loaded {len(samples)} eval problems from {eval_path}\n")

model.eval()
answer_head.eval()

num_to_show = 5
num_passes = min(4, max(len(s['cycle_targets']) for s in samples[:num_to_show]) + 1)

with torch.no_grad():
    for idx in range(num_to_show):
        s = samples[idx]
        problem = s['problem']
        cycle_targets = s['cycle_targets']
        gen_targets = s.get('cycle_gen_targets', [str(ct) for ct in cycle_targets])

        print("=" * 90)
        print(f"PROBLEM {idx+1}: {problem}")
        print(f"  cycle_targets: {cycle_targets}")
        print(f"  final_answer:  {s['final_answer']}")
        print()

        inputs = model.tokenizer(
            [problem], return_tensors='pt', padding=True,
            truncation=True, max_length=192,
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        state_pages = []
        messages = []
        mid_states_history = []

        for pass_num in range(num_passes):
            # Text injection for cycle 2+
            if pass_num > 0 and len(state_pages) > 0:
                if pass_num > 0 and len(state_pages) >= 2:
                    prev_page = (state_pages[-1] - state_pages[-2]).float()
                else:
                    prev_page = state_pages[-1].float()
                prev_pred = answer_head.decode(prev_page)
                ctx = f"Step {pass_num} result: {int(prev_pred[0].item())}\n"
                aug = ctx + problem
                aug_inp = model.tokenizer(
                    [aug], return_tensors='pt', padding=True,
                    truncation=True, max_length=212,
                )
                eval_ids = aug_inp['input_ids'].to(device)
                eval_mask = aug_inp['attention_mask'].to(device)
                text_input = aug
            else:
                eval_ids = input_ids
                eval_mask = attention_mask
                text_input = problem

            page, scales, mid_states, message = model.thinking_pass(
                eval_ids, eval_mask, state_pages, pass_num,
                prev_mid_states=mid_states_history if mid_states_history else None,
                messages=messages if messages else None,
            )
            state_pages.append(page)
            messages.append(message)
            mid_states_history.append(mid_states)

            # Decode prediction from page or delta
            if pass_num > 0 and len(state_pages) >= 2:
                read_page = (state_pages[-1] - state_pages[-2]).float()
            else:
                read_page = page.float()
            pred = int(answer_head.decode(read_page)[0].item())

            # Gold target
            gold = cycle_targets[pass_num] if pass_num < len(cycle_targets) else "N/A"
            gold_gen = gen_targets[pass_num] if pass_num < len(gen_targets) else "N/A"
            correct = "✓" if pass_num < len(cycle_targets) and pred == cycle_targets[pass_num] else "✗"

            # Active atoms
            active = (scales.abs() > 0.1).sum().item()

            # Page stats
            page_norm = page.float().norm().item()
            if pass_num > 0:
                delta_norm = (state_pages[-1] - state_pages[-2]).float().norm().item()
            else:
                delta_norm = page_norm

            # Message stats
            msg_vals = message[0].tolist()
            msg_str = ", ".join(f"{v:.2f}" for v in msg_vals[:8])

            print(f"  CYCLE {pass_num + 1}:")
            if pass_num > 0:
                print(f"    Text input:   \"{ctx.strip()}\" + problem")
            else:
                print(f"    Text input:   problem only (no injection)")
            print(f"    Active atoms: {active}/64")
            print(f"    Page norm:    {page_norm:.2f}, delta norm: {delta_norm:.2f}")
            print(f"    Message[0:8]: [{msg_str}]")
            print(f"    Gold target:  {gold}")
            print(f"    Gold gen:     {gold_gen}")
            print(f"    Prediction:   {pred} {correct}")
            print()

        print()
