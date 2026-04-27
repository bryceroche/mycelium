#!/usr/bin/env python3
"""Debug script: print EVERYTHING for a few problems, cycle by cycle.
Shows exactly where the chain breaks: scales, pages, bypass, generation, extraction."""
import sys, json, re, torch
sys.path.insert(0, '.')
from scripts.atom_lora import AtomLoRAModel, AtomAdditiveLoRAManager
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

samples = []
with open('data/per_cycle/gsm8k_eval.jsonl') as f:
    for line in f:
        if line.strip():
            samples.append(json.loads(line.strip()))
print(f"Loaded {len(samples)} eval problems\n")

model.eval()

# 2 correct (idx 2=kitten, 7=toys), 3 wrong (idx 0=sally, 1=cesar, 6=cats)
debug_indices = [2, 7, 0, 1, 6]

for pidx in debug_indices:
    s = samples[pidx]
    problem = s['problem']
    ct = s['cycle_targets']
    final = s['final_answer']
    n_cycles = min(3, len(ct))

    print("=" * 80)
    print(f"PROBLEM {pidx+1}: {problem[:120]}")
    print(f"Targets: {ct}  Final: {final}")
    print("=" * 80)

    inputs = model.tokenizer([problem], return_tensors='pt', truncation=True, max_length=192)
    input_ids = inputs['input_ids'].to(device)
    attn_mask = inputs['attention_mask'].to(device)

    state_pages = []
    history_hiddens = []
    prev_preds = []
    all_scales = []
    prev_scales = None  # cycle 0: no LoRA

    for pn in range(n_cycles):
        # Build text injection
        if pn > 0 and prev_preds:
            ctx = ''.join(f'Step {i+1} result: {p}\n' for i, p in enumerate(prev_preds))
            full_text = ctx + problem
            aug = model.tokenizer([full_text], return_tensors='pt', truncation=True, max_length=256)
            eid = aug['input_ids'].to(device)
            em = aug['attention_mask'].to(device)
        else:
            ctx = ''
            full_text = problem
            eid = input_ids
            em = attn_mask

        print(f"\n--- CYCLE {pn+1} ---")
        print(f"  Text injection: {repr(ctx[:100]) if ctx else '(none)'}")
        print(f"  Full input ({eid.size(1)} tokens): {model.tokenizer.decode(eid[0][:80])}...")

        with torch.no_grad():
            page, next_scales, _mid, _msg, _raw, hidden_pool, focus, _bv = model.thinking_pass(
                eid, em, state_pages, pn,
                history_hiddens=history_hiddens,
                prev_scales=prev_scales,
            )

            # Scale analysis (next_scales = scales for NEXT cycle)
            top5_vals, top5_idx = next_scales[0].abs().topk(5)
            print(f"  Next scale norm: {next_scales[0].norm():.3f}")
            print(f"  Top 5 atoms: {list(zip(top5_idx.tolist(), [f'{v:.3f}' for v in top5_vals.tolist()]))}")
            if all_scales:
                cos = F.cosine_similarity(all_scales[-1], next_scales, dim=-1).item()
                print(f"  Scale cos with prev cycle: {cos:.4f}")
            all_scales.append(next_scales)

            # Page analysis
            print(f"  Page norm: {page.norm():.3f}")
            if state_pages:
                pcos = F.cosine_similarity(state_pages[-1], page, dim=-1).item()
                print(f"  Page cos with prev: {pcos:.4f}")

            state_pages.append(page)
            history_hiddens.append(hidden_pool)

            # Generate with CURRENT cycle's scales (prev_scales)
            if prev_scales is not None:
                mgr = AtomAdditiveLoRAManager(model.transformer)
                mgr.apply(model.atoms, prev_scales)
                try:
                    gen_out = model.transformer.generate(
                        eid, attention_mask=em, max_new_tokens=60, do_sample=False,
                    )
                finally:
                    mgr.remove()
            else:
                # Cycle 0: no LoRA
                gen_out = model.transformer.generate(
                    eid, attention_mask=em, max_new_tokens=60, do_sample=False,
                )

            prev_scales = next_scales  # advance for next cycle

            gen_text = model.tokenizer.decode(gen_out[0][eid.size(1):], skip_special_tokens=True)

            m = re.search(r'####\s*([-]?\d+)', gen_text)
            if not m:
                nums = re.findall(r'[-]?\d+', gen_text)
                extracted = int(nums[-1]) if nums else None
            else:
                extracted = int(m.group(1))

            target = ct[pn] if pn < len(ct) else '?'
            match = 'OK' if extracted == target else 'MISS'
            prev_preds.append(extracted if extracted else 0)

            print(f"  Generated: {repr(gen_text[:120])}")
            print(f"  Extracted: {extracted}  Target: {target}  {match}")

    final_match = 'CORRECT' if prev_preds and prev_preds[-1] == final else 'WRONG'
    print(f"\n  >>> FINAL: predicted={prev_preds[-1] if prev_preds else None}  answer={final}  {final_match}")
    print()
