#!/usr/bin/env python3
"""Debug script: print EVERYTHING for a few problems, cycle by cycle.
Shows exactly where the chain breaks: scales, pages, bypass, generation, extraction."""
import sys, json, re, torch
sys.path.insert(0, '.')
from scripts.atom_lora import AtomLoRAModel, AtomAdditiveLoRAManager, encode_text_context
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
model.mobius = model.mobius.to(device)
model.bypass = model.bypass.to(device)

ckpt = torch.load('checkpoints/per_cycle_gsm8k_best.pt', map_location='cpu')
for name in ['compressor','atoms','hypernet','confidence_head',
             'residual_gate','message_generator','ordinal_head','bypass']:
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
    messages = []
    bypass_vectors = []
    mid_hist = []
    prev_preds = []
    all_scales = []

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
            # Build text context
            text_ctx = encode_text_context(prev_preds, device=device).unsqueeze(0)

            page, scales, ms, msg, raw_page, hp, focus, bv = model.thinking_pass(
                eid, em, state_pages, pn,
                prev_mid_states=mid_hist if mid_hist else None,
                messages=messages if messages else None,
                bypass_vectors=bypass_vectors if bypass_vectors else None,
                text_context=text_ctx,
            )

            # Scale analysis
            top5_vals, top5_idx = scales[0].abs().topk(5)
            print(f"  Scale norm: {scales[0].norm():.3f}")
            print(f"  Top 5 atoms: {list(zip(top5_idx.tolist(), [f'{v:.3f}' for v in top5_vals.tolist()]))}")
            if all_scales:
                cos = F.cosine_similarity(all_scales[-1], scales, dim=-1).item()
                print(f"  Scale cos with prev cycle: {cos:.4f}")
            all_scales.append(scales)

            # Page analysis
            print(f"  Page norm: {page.norm():.3f}")
            if state_pages:
                pcos = F.cosine_similarity(state_pages[-1], page, dim=-1).item()
                print(f"  Page cos with prev: {pcos:.4f}")

            # Bypass analysis
            if bypass_vectors:
                bcos = F.cosine_similarity(bypass_vectors[-1], bv, dim=-1).item()
                print(f"  Bypass cos with prev: {bcos:.4f}")
            print(f"  Bypass norm: {bv.norm():.3f}")

            state_pages.append(page)
            messages.append(msg)
            bypass_vectors.append(bv)
            mid_hist.append(ms)

            # Generate
            if pn == 0 and len(state_pages) == 1:
                hyper_dtype = next(model.hypernet.parameters()).dtype
                zp = torch.zeros(1, model.page_size, device=device, dtype=hyper_dtype)
                gen_scales, _, _f = model.hypernet(
                    [zp], pass_num=0, return_pre_tanh=True,
                    messages=messages, bypass_summary=None,
                )
            else:
                bs = torch.stack(bypass_vectors).mean(dim=0) if bypass_vectors else None
                gen_scales, _, _f = model.hypernet(
                    state_pages, pass_num=pn, return_pre_tanh=True,
                    messages=messages, bypass_summary=bs,
                )

            mgr = AtomAdditiveLoRAManager(model.transformer)
            mgr.apply(model.atoms, gen_scales)
            try:
                gen_out = model.transformer.generate(
                    eid, attention_mask=em, max_new_tokens=60, do_sample=False,
                )
            finally:
                mgr.remove()

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
