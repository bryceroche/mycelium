#!/usr/bin/env python3
"""C3 Span Extractor Training - Horse Race: RoBERTa-base vs RoBERTa-large"""

import json, torch, torch.nn as nn, boto3, random, re, argparse, os
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

MAX_OPERANDS, MAX_SEQ_LEN = 4, 384

MODELS = {
    'roberta': 'deepset/roberta-base-squad2',
    'roberta-large': 'deepset/roberta-large-squad2',
}

class C3SlotExtractor(nn.Module):
    def __init__(self, backbone_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name, torch_dtype=torch.float32)
        hidden_size = self.backbone.config.hidden_size
        self.slot_embeddings = nn.Embedding(MAX_OPERANDS, hidden_size)
        self.start_head = nn.Linear(hidden_size, 1)
        self.end_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        device = hidden.device
        start_list, end_list = [], []
        for slot in range(MAX_OPERANDS):
            slot_emb = self.slot_embeddings(torch.tensor(slot, device=device))
            slot_hidden = hidden + slot_emb.unsqueeze(0).unsqueeze(0)
            start_list.append(self.start_head(slot_hidden))
            end_list.append(self.end_head(slot_hidden))
        start_logits = torch.cat(start_list, dim=-1)
        end_logits = torch.cat(end_list, dim=-1)
        mask = attention_mask.unsqueeze(-1)
        return start_logits.masked_fill(mask == 0, float('-inf')), end_logits.masked_fill(mask == 0, float('-inf'))

def char_to_token_position(char_pos, offset_mapping):
    for tok_idx, (tok_start, tok_end) in enumerate(offset_mapping):
        if tok_start <= char_pos < tok_end: return tok_idx
    for tok_idx, (tok_start, tok_end) in enumerate(offset_mapping):
        if tok_start >= char_pos: return max(0, tok_idx - 1)
    return len(offset_mapping) - 1

def build_input_with_priors(problem_text, template, provenance):
    prior_values = {prov.get('prior_index', 0): prov.get('value') for prov in provenance if prov.get('source_type') == 'PRIOR' and prov.get('value') is not None}
    parts = [f'[TEMPLATE: {template}]']
    prior_char_pos = {}
    for idx in sorted(prior_values.keys()):
        val = prior_values[idx]
        val_str = str(int(val)) if isinstance(val, float) and val == int(val) else str(val)
        current_pos = len(' '.join(parts)) + 1
        prior_char_pos[idx] = (current_pos + len(f'[PRIOR_{idx}: '), current_pos + len(f'[PRIOR_{idx}: ') + len(val_str))
        parts.append(f'[PRIOR_{idx}: {val_str}]')
    parts.append(problem_text)
    return ' '.join(parts), prior_char_pos

def reformat_example(ex, tokenizer, max_length=MAX_SEQ_LEN):
    # New format: input_text already has [TEMPLATE: ...] and spans array
    if 'input_text' in ex and 'spans' in ex:
        input_text = ex['input_text']
        spans = ex['spans']
        if not spans: return None
        enc = tokenizer(input_text, truncation=True, max_length=max_length, padding='max_length', return_tensors='pt', return_offsets_mapping=True)
        input_ids, attention_mask = enc['input_ids'].squeeze(0), enc['attention_mask'].squeeze(0)
        offset_mapping = enc['offset_mapping'].squeeze(0).tolist()
        start_targets, end_targets, operand_mask = torch.zeros(MAX_OPERANDS, dtype=torch.long), torch.zeros(MAX_OPERANDS, dtype=torch.long), torch.zeros(MAX_OPERANDS, dtype=torch.float)
        valid = 0
        for i, span in enumerate(spans[:MAX_OPERANDS]):
            cs, ce = span.get('span_start', -1), span.get('span_end', -1)
            if cs < 0 or ce < 0: continue
            s_tok, e_tok = char_to_token_position(cs, offset_mapping), char_to_token_position(ce - 1, offset_mapping)
            if s_tok >= max_length or e_tok >= max_length: continue
            if e_tok < s_tok: e_tok = s_tok
            start_targets[i], end_targets[i], operand_mask[i] = s_tok, e_tok, 1.0
            valid += 1
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'start_targets': start_targets, 'end_targets': end_targets, 'operand_mask': operand_mask} if valid else None
    # Old format: provenance array
    problem_text, template, provenance = ex['problem_text'], ex['template'], ex.get('provenance', [])
    if not provenance: return None
    input_text, prior_char_pos = build_input_with_priors(problem_text, template, provenance)
    prob_start = input_text.find(problem_text)
    if prob_start < 0: prob_start = len(input_text) - len(problem_text)
    enc = tokenizer(input_text, truncation=True, max_length=max_length, padding='max_length', return_tensors='pt', return_offsets_mapping=True)
    input_ids, attention_mask = enc['input_ids'].squeeze(0), enc['attention_mask'].squeeze(0)
    offset_mapping = enc['offset_mapping'].squeeze(0).tolist()
    start_targets, end_targets, operand_mask = torch.zeros(MAX_OPERANDS, dtype=torch.long), torch.zeros(MAX_OPERANDS, dtype=torch.long), torch.zeros(MAX_OPERANDS, dtype=torch.float)
    valid = 0
    for i, prov in enumerate(provenance[:MAX_OPERANDS]):
        st = prov.get('source_type', 'TEXT')
        if st == 'PRIOR':
            if prov.get('prior_index', 0) not in prior_char_pos: continue
            cs, ce = prior_char_pos[prov.get('prior_index', 0)]
        elif st == 'TEXT':
            cs, ce = prov.get('char_start', -1), prov.get('char_end', -1)
            if cs < 0: continue
            cs, ce = prob_start + cs, prob_start + ce
        elif st == 'IMPLICIT':
            word = prov.get('word', '')
            if not word: continue
            m = re.search(re.escape(word), problem_text, re.IGNORECASE)
            if not m: continue
            cs, ce = prob_start + m.start(), prob_start + m.end()
        else: continue
        s_tok, e_tok = char_to_token_position(cs, offset_mapping), char_to_token_position(ce - 1, offset_mapping)
        if s_tok >= max_length or e_tok >= max_length: continue
        if e_tok < s_tok: e_tok = s_tok
        start_targets[i], end_targets[i], operand_mask[i] = s_tok, e_tok, 1.0
        valid += 1
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'start_targets': start_targets, 'end_targets': end_targets, 'operand_mask': operand_mask} if valid else None

class DS(torch.utils.data.Dataset):
    def __init__(self, ex): self.ex = ex
    def __len__(self): return len(self.ex)
    def __getitem__(self, i): return self.ex[i]

def collate(batch): return {k: torch.stack([ex[k] for ex in batch]) for k in batch[0].keys()}

def compute_loss(sl, el, batch, device):
    st, et, om = batch['start_targets'].to(device), batch['end_targets'].to(device), batch['operand_mask'].to(device)
    sl, el = sl.transpose(1, 2), el.transpose(1, 2)
    loss, n = 0.0, 0
    for i in range(MAX_OPERANDS):
        m = om[:, i] == 1
        if m.sum() == 0: continue
        loss += nn.functional.cross_entropy(sl[:, i, :][m], st[:, i][m], reduction='sum') + nn.functional.cross_entropy(el[:, i, :][m], et[:, i][m], reduction='sum')
        n += m.sum().item()
    return loss / max(n * 2, 1)

def evaluate(model, dl, device):
    model.eval()
    ec = sc = endc = t = 0
    slot_stats = {i: {'exact': 0, 'total': 0} for i in range(MAX_OPERANDS)}
    with torch.no_grad():
        for b in dl:
            ids, mask, st, et, om = b['input_ids'].to(device), b['attention_mask'].to(device), b['start_targets'].to(device), b['end_targets'].to(device), b['operand_mask'].to(device)
            sl, el = model(ids, mask)
            sp, ep = sl.argmax(dim=1), el.argmax(dim=1)
            for i in range(MAX_OPERANDS):
                m = om[:, i] == 1
                if m.sum() == 0: continue
                sm, em = sp[:, i][m] == st[:, i][m], ep[:, i][m] == et[:, i][m]
                slot_exact = (sm & em).sum().item()
                slot_total = m.sum().item()
                slot_stats[i]['exact'] += slot_exact
                slot_stats[i]['total'] += slot_total
                sc += sm.sum().item(); endc += em.sum().item(); ec += slot_exact; t += slot_total
    # Per-slot breakdown
    for i in range(MAX_OPERANDS):
        if slot_stats[i]['total'] > 0:
            acc = slot_stats[i]['exact'] / slot_stats[i]['total']
            print(f"    Slot {i}: {acc:.3f} ({slot_stats[i]['exact']}/{slot_stats[i]['total']})")
    return {'exact_match': ec / max(t, 1), 'start_acc': sc / max(t, 1), 'end_acc': endc / max(t, 1)}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=['roberta', 'roberta-large'], required=True)
    p.add_argument("--data-path", default="s3://mycelium-data/c3_span_training/c3_train_with_priors.jsonl")
    p.add_argument("--output-dir", default="models/c3_horserace")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--slot-lr", type=float, default=1e-4, help="Learning rate for slot embeddings and heads")
    p.add_argument("--grad-accum-steps", type=int, default=4, help="Gradient accumulation steps for larger effective batch")
    args = p.parse_args()
    model_name, output_dir = MODELS[args.model], f"{args.output_dir}/{args.model}"
    lr, ws, ddp = int(os.environ.get("LOCAL_RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("WORLD_SIZE", 1)) > 1
    if ddp:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl"); torch.cuda.set_device(lr); device = torch.device(f"cuda:{lr}")
    else: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_proc = lr == 0
    if main_proc: print("=" * 70 + f"\nC3 SPAN EXTRACTOR - {args.model.upper()} + Slot Embeddings\n" + "=" * 70 + f"\nModel: {model_name}\nDevice: {device}, World size: {ws}")
    if args.data_path.startswith("s3://"): 
        s3 = boto3.client('s3'); r = s3.get_object(Bucket=args.data_path.split('/')[2], Key='/'.join(args.data_path.split('/')[3:]))
        raw = [json.loads(l) for l in r['Body'].iter_lines() if l]
    else:
        with open(args.data_path) as f: raw = [json.loads(l) for l in f if l.strip()]
    if main_proc: print(f"Loaded {len(raw)} raw examples")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    ref = [reformat_example(ex, tok) for ex in raw]; ref = [r for r in ref if r]
    if main_proc: print(f"Reformatted: {len(ref)} examples")
    random.seed(42); random.shuffle(ref); si = int(len(ref) * 0.9); tr, va = ref[:si], ref[si:]
    if main_proc: print(f"Train: {len(tr)}, Val: {len(va)}")
    tds, vds = DS(tr), DS(va)
    if ddp:
        from torch.utils.data.distributed import DistributedSampler
        ts = DistributedSampler(tds, shuffle=True); tl = torch.utils.data.DataLoader(tds, batch_size=args.batch_size, sampler=ts, num_workers=4, pin_memory=True, collate_fn=collate)
    else: ts = None; tl = torch.utils.data.DataLoader(tds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate)
    vl = torch.utils.data.DataLoader(vds, batch_size=args.batch_size * 2, num_workers=4, pin_memory=True, collate_fn=collate)
    model = C3SlotExtractor(model_name).to(device)
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[lr], find_unused_parameters=True)
    if main_proc: print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    # Two-group optimizer: higher LR for slot embeddings + heads, lower for backbone
    base_model = model.module if ddp else model
    backbone_params = list(base_model.backbone.parameters())
    slot_params = list(base_model.slot_embeddings.parameters()) + list(base_model.start_head.parameters()) + list(base_model.end_head.parameters())
    opt = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.learning_rate},
        {'params': slot_params, 'lr': args.slot_lr}
    ])
    if main_proc: print(f"Optimizer: backbone LR={args.learning_rate}, slot LR={args.slot_lr}, grad_accum={args.grad_accum_steps}")
    if main_proc: Path(output_dir).mkdir(parents=True, exist_ok=True)
    best = 0
    for ep in range(args.epochs):
        if ts: ts.set_epoch(ep)
        model.train(); tl_loss = 0; accum_loss = 0
        opt.zero_grad()
        for bi, b in enumerate(tl):
            ids, mask = b['input_ids'].to(device), b['attention_mask'].to(device)
            sl, el = model(ids, mask); loss = compute_loss(sl, el, b, device) / args.grad_accum_steps
            loss.backward(); accum_loss += loss.item() * args.grad_accum_steps
            if (bi + 1) % args.grad_accum_steps == 0 or (bi + 1) == len(tl):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad()
                tl_loss += accum_loss; accum_loss = 0
            if main_proc and (bi + 1) % 100 == 0: print(f"  Epoch {ep+1}, Batch {bi+1}/{len(tl)}, Loss: {loss.item() * args.grad_accum_steps:.4f}")
        if main_proc:
            em = model.module if ddp else model; m = evaluate(em, vl, device)
            print(f"\nEpoch {ep+1}:\n  Loss: {tl_loss/len(tl):.4f}\n  Exact: {m['exact_match']:.3f}, Start: {m['start_acc']:.3f}, End: {m['end_acc']:.3f}")
            if m['exact_match'] > best:
                best = m['exact_match']; sp = f"{output_dir}/best"; Path(sp).mkdir(parents=True, exist_ok=True)
                torch.save({'model_state_dict': (model.module if ddp else model).state_dict(), 'epoch': ep, 'metrics': m, 'model_type': args.model}, f"{sp}/model.pt")
                tok.save_pretrained(sp); print(f"  >>> Best: {best:.3f}")
    if main_proc: print(f"\nDONE - {args.model.upper()} Best exact match: {best:.3f}")
    if ddp: import torch.distributed as dist; dist.destroy_process_group()

if __name__ == "__main__": main()
