"""
Dual LoRA GSM8K (v22.2).

The real test. GSM8K has diverse multi-step word problems (3-10 steps),
large number ranges, and requires genuine multi-pass reasoning.

Key differences from L4:
- Problems loaded from HuggingFace openai/gsm8k dataset
- CoT targets: GSM8K's own reasoning traces (<<calc>> annotations stripped)
- 5 thinking passes (problems average 4.6 steps)
- Longer sequences (problems ~100 tokens, answers ~150 tokens)
- Answer extraction handles floats, negatives, commas

Warm start from L4 dual checkpoint (100.0%).
"""
import argparse
import re
import sys
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, '/home/ubuntu/mycelium')

from datasets import load_dataset
from scripts.train_dual_lora import (
    DualLoRAModel, PageConfidenceHead,
    DualAdditiveLoRAManager,
)
from src.contrastive_page_loss import per_page_contrastive_loss


def parse_final(answer_text):
    """Extract the final numeric answer after ####."""
    m = re.search(r'####\s*(.+)', answer_text)
    if not m:
        return None
    try:
        return float(m.group(1).strip().replace(',', ''))
    except ValueError:
        return None


def clean_cot(answer_text):
    """Strip <<calc=result>> annotations and #### line for clean CoT target."""
    # Remove <<...>> calculator annotations
    cleaned = re.sub(r'<<.*?>>', '', answer_text)
    # Remove the #### final answer line
    cleaned = re.sub(r'\n####.*$', '', cleaned, flags=re.MULTILINE)
    # Clean up whitespace
    cleaned = cleaned.strip()
    # Append "The answer is X." for consistent extraction
    final = parse_final(answer_text)
    if final is not None:
        final_str = str(int(final)) if final == int(final) else str(final)
        cleaned += f" The answer is {final_str}."
    return cleaned


def extract_answer(text):
    """Extract answer: 'The answer is X' first, then last number fallback."""
    # Stop at next "Problem:" (few-shot leakage)
    if "\nProblem:" in text:
        text = text.split("\nProblem:")[0]
    # Try "The answer is X"
    m = re.search(r'[Tt]he answer is\s*\$?(-?[\d,]+\.?\d*)', text)
    if m:
        try:
            v = float(m.group(1).replace(',', ''))
            return int(v) if v == int(v) else v
        except ValueError:
            pass
    # Fallback: last number
    nums = re.findall(r'-?[\d,]+\.?\d*', text)
    if nums:
        try:
            v = float(nums[-1].replace(',', ''))
            return int(v) if v == int(v) else v
        except ValueError:
            pass
    return None


class GSM8KDataset(Dataset):
    """GSM8K with clean CoT targets."""
    def __init__(self, split='train', max_samples=None):
        ds = load_dataset("openai/gsm8k", "main", split=split)
        self.samples = []
        for ex in ds:
            final = parse_final(ex['answer'])
            if final is None:
                continue
            cot = clean_cot(ex['answer'])
            final_int = int(final) if final == int(final) else final
            self.samples.append({
                'problem': ex['question'],
                'answer': cot,
                'final': final_int,
            })
            if max_samples and len(self.samples) >= max_samples:
                break
        print(f"Loaded {len(self.samples)} GSM8K problems (split={split})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def forward_train(model, problems, answers, finals_t, num_passes=5):
    """Forward pass with dual LoRA for GSM8K."""
    device = model.transformer.device

    inputs = model.tokenizer(
        problems, return_tensors='pt', padding=True, truncation=True, max_length=192,
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    answer_texts = [f" {a}" for a in answers]
    answer_inputs = model.tokenizer(
        answer_texts, return_tensors='pt', padding=True,
        truncation=True, max_length=256, add_special_tokens=False,
    )
    answer_ids = answer_inputs['input_ids'].to(device)

    batch_size = input_ids.size(0)
    embed_layer = model.transformer.model.embed_tokens
    problem_embeds = embed_layer(input_ids)

    state_pages = []
    blend_history = []
    strategy = torch.zeros(batch_size, model.strategy_size, device=device)

    for pass_num in range(num_passes):
        if pass_num == 0:
            outputs = model.transformer(
                inputs_embeds=problem_embeds, attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = list(outputs.hidden_states[1:])
            blend = torch.zeros(batch_size, 1, device=device)
        else:
            forward_mods, verify_mods, blend = model.hypernet(
                state_pages, strategy, pass_num=pass_num,
            )
            manager = DualAdditiveLoRAManager(model.transformer)
            manager.apply(forward_mods, verify_mods, blend)
            try:
                outputs = model.transformer(
                    inputs_embeds=problem_embeds, attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden_states = list(outputs.hidden_states[1:])
            finally:
                manager.remove()

        page_delta, strategy, _mid_states = model.compressor(hidden_states, pass_num)
        page = F.normalize(page_delta, dim=-1) * model.page_radius
        state_pages.append(page)
        blend_history.append(blend)

    # Answer loss — teacher-forced with dual LoRA
    forward_mods, verify_mods, final_blend = model.hypernet(
        state_pages, strategy, pass_num=num_passes,
    )
    manager = DualAdditiveLoRAManager(model.transformer)
    manager.apply(forward_mods, verify_mods, final_blend)
    try:
        answer_embeds = embed_layer(answer_ids)
        full_embeds = torch.cat([problem_embeds, answer_embeds], dim=1)
        outputs = model.transformer(inputs_embeds=full_embeds, use_cache=False)
    finally:
        manager.remove()
    prompt_len = input_ids.size(1)
    logits = outputs.logits[:, prompt_len - 1:-1, :]
    answer_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        answer_ids.reshape(-1),
        ignore_index=model.tokenizer.pad_token_id,
    )

    # Confidence loss
    confidence = model.confidence_head(state_pages, blend_history)
    conf_target = torch.ones_like(confidence)
    conf_loss = F.binary_cross_entropy(confidence, conf_target)

    # Contrastive loss — use final answer as group label
    # For GSM8K, many problems share the same answer, so contrastive is less
    # informative. Keep it but at lower weight.
    c_loss = per_page_contrastive_loss(
        state_pages, finals_t, cross_target=0.7, within_target=0.3,
    )

    # Diagnostics
    with torch.no_grad():
        last_page = state_pages[-1].float()
        normed = F.normalize(last_page, dim=-1)
        sim = normed @ normed.T
        B = sim.size(0)
        off_diag = sim - torch.eye(B, device=sim.device)
        page_cos_mean = off_diag.sum() / (B * (B - 1))
        blend_mean = final_blend.mean()

    return answer_loss, c_loss, conf_loss, page_cos_mean, blend_mean, confidence.mean()


def evaluate(model, eval_dataset, device, num_passes=5):
    """Evaluate on GSM8K with dual LoRA."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(eval_dataset), 4):
            batch_samples = [eval_dataset[j] for j in range(i, min(i + 4, len(eval_dataset)))]
            problems = [s['problem'] for s in batch_samples]
            gold_answers = [s['final'] for s in batch_samples]

            inputs = model.tokenizer(
                problems, return_tensors='pt', padding=True,
                truncation=True, max_length=192,
            )
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            batch_size = input_ids.size(0)

            state_pages = []
            blend_history = []
            strategy = torch.zeros(batch_size, model.strategy_size, device=device)

            for pass_num in range(num_passes):
                page, strategy, blend = model.thinking_pass(
                    input_ids, attention_mask, state_pages, strategy, pass_num,
                )
                state_pages.append(page)
                blend_history.append(blend)

            # Generate with dual LoRA
            forward_mods, verify_mods, blend = model.hypernet(
                state_pages, strategy, pass_num=len(state_pages),
            )
            manager = DualAdditiveLoRAManager(model.transformer)
            manager.apply(forward_mods, verify_mods, blend)
            try:
                generated = model.transformer.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    max_new_tokens=150, do_sample=False,
                    pad_token_id=model.tokenizer.pad_token_id,
                )
            finally:
                manager.remove()

            for j in range(batch_size):
                prompt_len = input_ids[j].size(0)
                gen_ids = generated[j][prompt_len:]
                gen_text = model.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                pred = extract_answer(gen_text)
                gold = gold_answers[j]
                # Compare: exact match for ints, tolerance for floats
                if pred is not None and gold is not None:
                    if isinstance(gold, float) and not gold == int(gold):
                        if abs(pred - gold) < 0.01:
                            correct += 1
                    elif pred == gold:
                        correct += 1
                total += 1

    acc = 100.0 * correct / total if total > 0 else 0.0
    return acc


def warm_start_dual(model, ckpt_path):
    """Warm-start dual model from a dual-LoRA checkpoint."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)

    # Compressor
    own = model.compressor.state_dict()
    loaded = 0
    for k, v in ckpt['compressor'].items():
        if k in own and own[k].shape == v.shape:
            own[k] = v
            loaded += 1
    model.compressor.load_state_dict(own, strict=False)
    print(f"  compressor: loaded {loaded}/{len(own)}")

    # Hypernet — load full dual state
    if 'hypernet' in ckpt:
        own_h = model.hypernet.state_dict()
        loaded_h = 0
        for k, v in ckpt['hypernet'].items():
            if k in own_h and own_h[k].shape == v.shape:
                own_h[k] = v
                loaded_h += 1
        model.hypernet.load_state_dict(own_h, strict=False)
        print(f"  hypernet: loaded {loaded_h}/{len(own_h)}")

    # Confidence head
    if 'confidence_head' in ckpt:
        own_c = model.confidence_head.state_dict()
        loaded_c = 0
        for k, v in ckpt['confidence_head'].items():
            if k in own_c and own_c[k].shape == v.shape:
                own_c[k] = v
                loaded_c += 1
        model.confidence_head.load_state_dict(own_c, strict=False)
        print(f"  confidence_head: loaded {loaded_c}/{len(own_c)}")
    else:
        print(f"  confidence head: fresh init")


def train(args):
    print("=" * 60)
    print("Dual LoRA GSM8K — The Real Test")
    print("=" * 60)
    print(f"num_passes={args.num_passes}  batch={args.batch_size}")
    print(f"lam={args.lam}  lam_conf={args.lam_conf}")
    print(f"Warm: {args.warm}")
    print("=" * 60)

    device = torch.device('cuda')
    model = DualLoRAModel()
    model.compressor = model.compressor.to(device=device, dtype=torch.bfloat16)
    model.hypernet = model.hypernet.to(device=device, dtype=torch.bfloat16)
    model.confidence_head = model.confidence_head.to(device)
    model.probe_head = model.probe_head.to(device)

    if args.warm:
        warm_start_dual(model, args.warm)

    train_dataset = GSM8KDataset(split='train')
    eval_dataset = GSM8KDataset(split='test')

    print(f"\nDataset: {len(train_dataset)} train, {len(eval_dataset)} eval")
    print(f"\nSample problems:")
    for i in range(3):
        s = train_dataset[i]
        print(f"  Q: {s['problem']}")
        print(f"  A: {s['answer']}")
        print(f"  Final: {s['final']}")
        print()

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda batch: {
            'problem': [s['problem'] for s in batch],
            'answer': [s['answer'] for s in batch],
            'final': [s['final'] for s in batch],
        },
    )

    optimizer = torch.optim.AdamW([
        {'params': list(model.compressor.parameters()), 'lr': 5e-5},
        # Forward templates
        {'params': (list(model.hypernet.A_forward.parameters())
                    + list(model.hypernet.B_forward.parameters())), 'lr': 5e-4},
        # Verify templates
        {'params': (list(model.hypernet.A_verify.parameters())
                    + list(model.hypernet.B_verify.parameters())), 'lr': 5e-4},
        # Shared hypernetwork
        {'params': (
            list(model.hypernet.page_project.parameters())
            + [model.hypernet.page_query]
            + list(model.hypernet.page_attn.parameters())
            + list(model.hypernet.page_norm.parameters())
            + list(model.hypernet.combine.parameters())
            + list(model.hypernet.pass_embed.parameters())
        ), 'lr': 5e-4},
        # Confidence head
        {'params': list(model.confidence_head.parameters()), 'lr': 1e-3},
    ])
    trainable = (
        list(model.compressor.parameters())
        + list(model.hypernet.parameters())
        + list(model.confidence_head.parameters())
    )
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")
    print(f"GPU mem after init: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Baseline
    base_acc = evaluate(model, eval_dataset, device, num_passes=args.num_passes)
    print(f"Baseline accuracy (before training): {base_acc:.1f}%\n")

    best = 0.0
    patience_counter = 0
    for epoch in range(args.epochs):
        model.train()
        ep_ans = ep_ctr = ep_conf = ep_cos = ep_blend = ep_confval = 0.0
        nb = 0
        t0 = time.time()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            problems = batch['problem']
            answers = batch['answer']
            finals_t = torch.tensor(
                [int(f) if isinstance(f, int) else int(f) for f in batch['final']],
                dtype=torch.long, device=device,
            )
            optimizer.zero_grad()
            ans_loss, c_loss, conf_loss, page_cos, blend_mean, conf_mean = forward_train(
                model, problems, answers, finals_t, num_passes=args.num_passes,
            )
            total_loss = ans_loss + args.lam * c_loss + args.lam_conf * conf_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            ep_ans += ans_loss.item()
            ep_ctr += c_loss.item()
            ep_conf += conf_loss.item()
            ep_cos += page_cos.item()
            ep_blend += blend_mean.item()
            ep_confval += conf_mean.item()
            nb += 1

        elapsed = time.time() - t0

        # Eval on test set
        acc = evaluate(model, eval_dataset, device, num_passes=args.num_passes)
        if acc > best:
            best = acc
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'compressor': model.compressor.state_dict(),
                'hypernet': model.hypernet.state_dict(),
                'confidence_head': model.confidence_head.state_dict(),
                'accuracy': acc,
                'level': 'GSM8K_dual',
            }, 'checkpoints/dual_lora_gsm8k_best.pt')
            print(f"  -> saved checkpoint (acc={acc:.1f}%)")
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch+1}: ans={ep_ans/nb:.4f} contr={ep_ctr/nb:.4f} "
            f"conf={ep_conf/nb:.4f} page_cos={ep_cos/nb:.4f} "
            f"blend={ep_blend/nb:.3f} conf_val={ep_confval/nb:.3f} | "
            f"Acc={acc:.1f}% best={best:.1f}% base={base_acc:.1f}% "
            f"[{elapsed:.0f}s, {nb/elapsed:.1f} it/s]"
        )
        if patience_counter >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} epochs")
            break

    print(f"\nFinal: {best:.1f}% (baseline {base_acc:.1f}%)")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--warm', type=str, default='checkpoints/dual_lora_L4_best.pt')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lam', type=float, default=0.02)
    p.add_argument('--lam_conf', type=float, default=0.1)
    p.add_argument('--num_passes', type=int, default=5)
    p.add_argument('--patience', type=int, default=5)
    args = p.parse_args()
    train(args)
