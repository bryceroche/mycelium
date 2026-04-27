#!/usr/bin/env python3
"""Diagnose GSM8K per-cycle: show inputs, outputs, generation for each cycle."""
import sys, re
sys.path.insert(0, '.')
import json, torch
from scripts.atom_lora import AtomLoRAModel, AnswerHead, AtomAdditiveLoRAManager
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AtomLoRAModel()
model.compressor = model.compressor.to(device=device, dtype=torch.bfloat16)
model.hypernet = model.hypernet.to(device=device, dtype=torch.bfloat16)
model.atoms = model.atoms.to(device=device, dtype=torch.bfloat16)
model.confidence_head = model.confidence_head.to(device)
model.residual_gate = model.residual_gate.to(device=device, dtype=torch.bfloat16)
model.probe_head = model.probe_head.to(device)
model.message_generator = model.message_generator.to(device)
model.ordinal_head = model.ordinal_head.to(device)
answer_head = AnswerHead(page_size=model.page_size).to(device)

ckpt_path = 'checkpoints/per_cycle_gsm8k_best.pt'
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
print(f"Loaded {ckpt_path} (accuracy={ckpt.get('accuracy', '?')}%)")

samples = []
with open('data/per_cycle/gsm8k_eval.jsonl') as f:
    for line in f:
        line = line.strip()
        if line:
            samples.append(json.loads(line))
print(f"Loaded {len(samples)} GSM8K eval problems\n")

model.eval()
answer_head.eval()

print("=" * 80)
print("GSM8K CYCLE INSPECTION")
print("=" * 80)

ah_correct_total = [0, 0]
gen_correct_total = [0, 0]
total_per_cycle = [0, 0]

with torch.no_grad():
    for idx in range(min(50, len(samples))):
        s = samples[idx]
        problem = s['problem']
        ct = s['cycle_targets']
        gt = s.get('cycle_gen_targets', [str(c) for c in ct])
        n_cycles = min(len(ct), 2)

        inputs = model.tokenizer([problem], return_tensors='pt', padding=True,
                                  truncation=True, max_length=256)
        input_ids = inputs['input_ids'].to(device)
        attn_mask = inputs['attention_mask'].to(device)

        state_pages = []
        messages = []
        mid_hist = []
        prev_preds = []

        show = idx < 8  # print detail for first 8

        if show:
            print(f"\nPROBLEM {idx+1}: {problem[:100]}...")
            print(f"  targets: {ct}  final: {s['final_answer']}")

        for pn in range(n_cycles):
            if pn > 0 and len(state_pages) > 0:
                lp_val = int(answer_head.decode(state_pages[-1].float())[0].item())
                prev_preds.append(lp_val)
                ctx = ''.join(f'Step {si+1} result: {pp}\n'
                              for si, pp in enumerate(prev_preds))
                aug = model.tokenizer([ctx + problem], return_tensors='pt',
                                       padding=True, truncation=True, max_length=300)
                eid = aug['input_ids'].to(device)
                em = aug['attention_mask'].to(device)
                pl = eid.size(1)
            else:
                eid = input_ids
                em = attn_mask
                pl = input_ids.size(1)

            page, sc, ms, msg, _raw_page = model.thinking_pass(
                eid, em, state_pages, pn,
                prev_mid_states=mid_hist if mid_hist else None,
                messages=messages if messages else None,
            )
            state_pages.append(page)
            messages.append(msg)
            mid_hist.append(ms)

            ah_pred = int(answer_head.decode(page.float())[0].item())
            page_norm = page.float().norm().item()

            # Generate
            manager = AtomAdditiveLoRAManager(model.transformer)
            manager.apply(model.atoms, sc)
            try:
                gen_out = model.transformer.generate(
                    input_ids=eid, attention_mask=em,
                    max_new_tokens=50, do_sample=False,
                    pad_token_id=model.tokenizer.pad_token_id or model.tokenizer.eos_token_id,
                )
                gen_text = model.tokenizer.decode(gen_out[0][pl:], skip_special_tokens=True)
            finally:
                manager.remove()

            # Extract number from generation
            eq_nums = re.findall(r'=\s*\$?([-]?[\d,]+\.?\d*)', gen_text)
            all_nums = re.findall(r'([-]?\d+)', gen_text)
            gen_num = None
            if eq_nums:
                try:
                    gen_num = int(float(eq_nums[-1].replace(',', '')))
                except ValueError:
                    pass
            if gen_num is None and all_nums:
                try:
                    gen_num = int(all_nums[-1])
                except ValueError:
                    pass

            gold = ct[pn]
            ah_ok = ah_pred == gold
            gen_ok = gen_num is not None and gen_num == gold

            if pn < 2:
                total_per_cycle[pn] += 1
                if ah_ok:
                    ah_correct_total[pn] += 1
                if gen_ok:
                    gen_correct_total[pn] += 1

            if show:
                print(f"  Cycle {pn+1}: gold={gold}  AH={ah_pred}({'Y' if ah_ok else 'N'})  "
                      f"GEN={gen_num}({'Y' if gen_ok else 'N'})  norm={page_norm:.2f}")
                print(f"    gold_gen: {gt[pn][:80] if pn < len(gt) else 'N/A'}")
                print(f"    model_gen: {gen_text[:100]}")

print("\n" + "=" * 80)
print("SUMMARY (50 problems)")
print("=" * 80)
for c in range(2):
    if total_per_cycle[c] > 0:
        ah_pct = 100 * ah_correct_total[c] / total_per_cycle[c]
        gen_pct = 100 * gen_correct_total[c] / total_per_cycle[c]
        print(f"  Cycle {c+1}: AH={ah_pct:.1f}%  GEN={gen_pct:.1f}%  (n={total_per_cycle[c]})")

# Page dynamics
print("\n" + "=" * 80)
print("PAGE DYNAMICS (20 problems)")
print("=" * 80)
with torch.no_grad():
    for idx in range(min(20, len(samples))):
        s = samples[idx]
        problem = s['problem']
        ct = s['cycle_targets']
        inputs = model.tokenizer([problem], return_tensors='pt', padding=True,
                                  truncation=True, max_length=256)
        input_ids = inputs['input_ids'].to(device)
        attn_mask = inputs['attention_mask'].to(device)
        state_pages = []
        messages = []
        mid_hist = []
        prev_preds = []
        for pn in range(min(len(ct), 2)):
            if pn > 0:
                lp_val = int(answer_head.decode(state_pages[-1].float())[0].item())
                prev_preds.append(lp_val)
                ctx = ''.join(f'Step {si+1} result: {pp}\n'
                              for si, pp in enumerate(prev_preds))
                aug = model.tokenizer([ctx + problem], return_tensors='pt',
                                       padding=True, truncation=True, max_length=300)
                eid = aug['input_ids'].to(device)
                em = aug['attention_mask'].to(device)
            else:
                eid = input_ids
                em = attn_mask
            page, sc, ms, msg, _raw_page = model.thinking_pass(
                eid, em, state_pages, pn,
                prev_mid_states=mid_hist if mid_hist else None,
                messages=messages if messages else None,
            )
            state_pages.append(page)
            messages.append(msg)
            mid_hist.append(ms)
        if len(state_pages) >= 2:
            p1 = state_pages[0].float()
            p2 = state_pages[1].float()
            cos12 = F.cosine_similarity(p1, p2, dim=-1).item()
            n1 = p1.norm().item()
            n2 = p2.norm().item()
            print(f"  P{idx+1}: cos(1,2)={cos12:.3f}  norms={n1:.2f},{n2:.2f}  targets={ct[:2]}")
