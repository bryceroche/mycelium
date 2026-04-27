#!/usr/bin/env python3
"""Three-way comparison: are atoms helping or hurting on single-step GSM8K?

Tests:
  1. scales in [-3, 3]     (current, full strength)
  2. scales in [-0.5, 0.5] (gentle, 6x weaker)
  3. scales = 0            (no atoms, pure vanilla Llama)

If gentle > full:    atoms too strong, interfering with computation
If zero > full:      atoms net negative, need fundamental rethink
If full > all:       atoms help, look elsewhere for the bottleneck

Usage:
  python scripts/diag_atom_strength.py
"""
import sys, json, re, torch
sys.path.insert(0, '.')
from scripts.atom_lora import AtomLoRAModel, AtomAdditiveLoRAManager
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
model = AtomLoRAModel()
model.atoms = model.atoms.to(device=device, dtype=torch.bfloat16)
model.controller = model.controller.to(device=device, dtype=torch.bfloat16)
model.confidence_head = model.confidence_head.to(device)
model.mobius = model.mobius.to(device)

ckpt = torch.load('checkpoints/per_cycle_gsm8k_best.pt', map_location='cpu')
for name in ['atoms', 'controller', 'confidence_head']:
    if name in ckpt:
        obj = getattr(model, name)
        own = obj.state_dict()
        for k, v in ckpt[name].items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
        obj.load_state_dict(own, strict=False)
print(f"Loaded checkpoint (accuracy={ckpt.get('accuracy', '?')}%)")

# Load single-step eval data
samples = []
with open('data/per_cycle/gsm8k_single_step_eval.jsonl') as f:
    for line in f:
        if line.strip():
            samples.append(json.loads(line.strip()))
print(f"Loaded {len(samples)} single-step eval problems\n")

model.eval()


def extract_answer(text):
    match = re.search(r'####\s*([-]?\d+)', text)
    if match:
        return int(match.group(1))
    nums = re.findall(r'[-]?\d+', text)
    return int(nums[-1]) if nums else None


def evaluate_with_scale_clamp(model, samples, device, clamp_val=None, max_problems=100):
    """Run eval with clamped atom scales.

    clamp_val=None: use full scales [-3, 3] (current behavior)
    clamp_val=0.5:  clamp to [-0.5, 0.5] (gentle)
    clamp_val=0.0:  zero all scales (no atoms)
    """
    correct = 0
    total = 0

    for idx in range(min(max_problems, len(samples))):
        s = samples[idx]
        problem = s['problem']
        target = s['final_answer']

        inputs = model.tokenizer([problem], return_tensors='pt', truncation=True, max_length=192)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            # Two-pass cycle 1
            (page_0, hp_0, page_1, hp_1,
             initial_scales, next_scales, focus_1, outputs_lora) = model.two_pass_cycle1(
                input_ids, attention_mask,
            )

            # Clamp scales for generation
            gen_scales = initial_scales.clone()
            if clamp_val is not None:
                if clamp_val == 0.0:
                    gen_scales = torch.zeros_like(gen_scales)
                else:
                    gen_scales = gen_scales.clamp(-clamp_val, clamp_val)

            # Generate with clamped scales
            if clamp_val == 0.0:
                # No LoRA at all
                gen_out = model.transformer.generate(
                    input_ids, attention_mask=attention_mask,
                    max_new_tokens=60, do_sample=False,
                )
            else:
                mgr = AtomAdditiveLoRAManager(model.transformer)
                mgr.apply(model.atoms, gen_scales)
                try:
                    gen_out = model.transformer.generate(
                        input_ids, attention_mask=attention_mask,
                        max_new_tokens=60, do_sample=False,
                    )
                finally:
                    mgr.remove()

            gen_text = model.tokenizer.decode(gen_out[0][input_ids.size(1):], skip_special_tokens=True)
            predicted = extract_answer(gen_text)

            if predicted == target:
                correct += 1
            total += 1

    return correct, total


print("=" * 60)
print("ATOM STRENGTH COMPARISON (single-step GSM8K)")
print("=" * 60)

# Test 1: Full strength [-3, 3]
print("\nTest 1: Full atom scales [-3, 3]...")
c1, t1 = evaluate_with_scale_clamp(model, samples, device, clamp_val=None, max_problems=100)
print(f"  Accuracy: {c1}/{t1} = {100*c1/t1:.1f}%")

# Test 2: Gentle [-0.5, 0.5]
print("\nTest 2: Gentle atom scales [-0.5, 0.5]...")
c2, t2 = evaluate_with_scale_clamp(model, samples, device, clamp_val=0.5, max_problems=100)
print(f"  Accuracy: {c2}/{t2} = {100*c2/t2:.1f}%")

# Test 3: No atoms (scales = 0)
print("\nTest 3: No atoms (scales = 0, pure vanilla Llama)...")
c3, t3 = evaluate_with_scale_clamp(model, samples, device, clamp_val=0.0, max_problems=100)
print(f"  Accuracy: {c3}/{t3} = {100*c3/t3:.1f}%")

print("\n" + "=" * 60)
print("RESULTS:")
print(f"  Full [-3,3]:     {100*c1/t1:.1f}%")
print(f"  Gentle [-0.5]:   {100*c2/t2:.1f}%")
print(f"  No atoms [0]:    {100*c3/t3:.1f}%")
print()

if c2 > c1:
    print("VERDICT: Atoms TOO STRONG → need layer envelope (parse loud, compute quiet)")
elif c3 > c1:
    print("VERDICT: Atoms NET NEGATIVE → fundamental rethink needed")
else:
    print("VERDICT: Atoms HELP → bottleneck is elsewhere (not atom strength)")
print("=" * 60)
