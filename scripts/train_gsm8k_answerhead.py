"""
GSM8K v21.1: pages + log-answer head. No generation at all.

Think → compress → think → compress → think → compress → LAST PAGE → Linear(64,1)
                                                                   → log10(|ans|+1)
                                                                   → Linear(64,1)
                                                                   → P(ans >= 0)

Loss: MSE on log-magnitude + BCE on sign. Single forward per answer loss
(just the 64-float head), no transformer forward for generation.

Warm-starts perceiver + hypernet + LoRA templates from two_step_pages_best.pt.
The log-answer head is fresh.

Eval: round if gold is integer, else float. Also report 1%-tolerance accuracy
(directional correctness even when integer-rounding fails for large answers).
"""
import argparse
import re
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, '/home/ubuntu/mycelium')

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.compressor_v3 import Compressor
from src.page_attention_hypernetwork import PageAttentionHypernetwork
from src.log_answer_head import LogAnswerHead
from src.additive_lora import AdditiveLoRAManager


FEW_SHOT = """Problem: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: Natalia sold 48/2 = 24 clips in May. Altogether she sold 48+24 = 72 clips. The answer is 72.

Problem: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer: Weng earns 12/60 = 0.2 per minute. Working 50 minutes she earned 0.2 x 50 = 10. The answer is 10.

Problem: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
Answer: Betty has 100/2 = 50. Her grandparents gave her 15*2 = 30. Total she has 50+15+30 = 95. She needs 100-95 = 5. The answer is 5.

"""


def parse_final(answer_text):
    m = re.search(r'####\s*(.+)', answer_text)
    if not m:
        return None
    try:
        return float(m.group(1).strip().replace(',', ''))
    except ValueError:
        return None


class GSM8KDataset(Dataset):
    def __init__(self, split='train', max_samples=None):
        ds = load_dataset("openai/gsm8k", "main", split=split)
        self.samples = []
        for ex in ds:
            final = parse_final(ex['answer'])
            if final is None:
                continue
            self.samples.append({
                'question': ex['question'],
                'final': final,
                'is_int': final == int(final),
            })
            if max_samples and len(self.samples) >= max_samples:
                break
        print(f"Loaded {len(self.samples)} GSM8K problems (split={split})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate(batch):
    return {
        'question': [s['question'] for s in batch],
        'final':    [s['final']    for s in batch],
        'is_int':   [s['is_int']   for s in batch],
    }


class PageAnswerModel(nn.Module):
    def __init__(self, model_name='unsloth/Llama-3.2-1B'):
        super().__init__()
        self.transformer = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map='auto',
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'

        for p in self.transformer.parameters():
            p.requires_grad = False

        self.d_model = self.transformer.config.hidden_size
        self.num_layers = self.transformer.config.num_hidden_layers
        num_kv_heads = self.transformer.config.num_key_value_heads
        head_dim = self.d_model // self.transformer.config.num_attention_heads
        d_kv = num_kv_heads * head_dim

        self.page_size = 64
        self.strategy_size = 64
        self.page_radius = self.page_size ** 0.5

        self.compressor = Compressor(
            num_transformer_layers=self.num_layers,
            d_transformer=self.d_model,
            d_perceiver=1024,
            num_queries=4,
            num_perceiver_layers=7,
            state_size=self.page_size,
            strategy_size=self.strategy_size,
        )
        self.hypernet = PageAttentionHypernetwork(
            d_model=self.d_model,
            d_kv=d_kv,
            page_size=self.page_size,
            strategy_size=self.strategy_size,
            rank=4,
            num_layers=self.num_layers,
        )
        self.answer_head = LogAnswerHead(page_size=self.page_size)

    def thinking_pass(self, input_ids, attention_mask, state_pages, strategy, pass_num):
        if len(state_pages) == 0:
            outputs = self.transformer(
                input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = list(outputs.hidden_states[1:])
        else:
            lora_mods = self.hypernet(state_pages, strategy)
            manager = AdditiveLoRAManager(self.transformer)
            manager.apply(lora_mods)
            try:
                outputs = self.transformer(
                    input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden_states = list(outputs.hidden_states[1:])
            finally:
                manager.remove()

        page_delta, new_strategy = self.compressor(hidden_states, pass_num)
        page = F.normalize(page_delta, dim=-1) * self.page_radius
        return page, new_strategy

    def run_thinking(self, questions, num_passes=3):
        prompts = [FEW_SHOT + f"Problem: {q}\nAnswer:" for q in questions]
        inputs = self.tokenizer(
            prompts, return_tensors='pt', padding=True,
            truncation=True, max_length=1024,
        )
        input_ids = inputs['input_ids'].to(self.transformer.device)
        attention_mask = inputs['attention_mask'].to(self.transformer.device)

        batch_size = input_ids.size(0)
        device = input_ids.device

        state_pages = []
        strategy = torch.zeros(batch_size, self.strategy_size, device=device)
        for pn in range(num_passes):
            page, strategy = self.thinking_pass(
                input_ids, attention_mask, state_pages, strategy, pn,
            )
            state_pages.append(page)
        return state_pages

    def forward_train(self, questions, golds, num_passes=3):
        state_pages = self.run_thinking(questions, num_passes=num_passes)
        last_page = state_pages[-1].to(torch.float32)
        gold_t = torch.tensor(golds, dtype=torch.float32, device=last_page.device)
        loss, mag_loss, sign_loss = self.answer_head.compute_loss(last_page, gold_t)
        return loss, mag_loss, sign_loss


def evaluate(model, eval_dataset, device, num_passes=3, batch_size=8):
    model.eval()
    exact = 0
    tol1 = 0  # within 1% of gold
    total = 0
    with torch.no_grad():
        for i in range(0, len(eval_dataset), batch_size):
            batch = eval_dataset.samples[i:i + batch_size]
            questions = [s['question'] for s in batch]
            golds = [s['final'] for s in batch]
            is_ints = [s['is_int'] for s in batch]

            state_pages = model.run_thinking(questions, num_passes=num_passes)
            last_page = state_pages[-1].to(torch.float32)
            preds = model.answer_head.decode(last_page).cpu().tolist()

            for pred, gold, is_int in zip(preds, golds, is_ints):
                if is_int:
                    pred_final = round(pred)
                    gold_final = int(gold)
                    if pred_final == gold_final:
                        exact += 1
                else:
                    if abs(pred - gold) < 1e-3:
                        exact += 1
                # 1% tolerance (directional correctness)
                denom = max(abs(gold), 1.0)
                if abs(pred - gold) / denom <= 0.01:
                    tol1 += 1
                total += 1
    return 100.0 * exact / total, 100.0 * tol1 / total


def warm_start(model, ckpt_path):
    print(f"Warm-starting from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    miss, unex = model.compressor.load_state_dict(ckpt['compressor'], strict=False)
    print(f"  compressor: missing={len(miss)} unexpected={len(unex)}")
    miss, unex = model.hypernet.load_state_dict(ckpt['hypernet'], strict=False)
    print(f"  hypernet: missing={len(miss)} unexpected={len(unex)}")


def train(args):
    print("=" * 60)
    print("GSM8K v21.1: pages + log-answer head (no generation)")
    print("=" * 60)
    print(f"Baseline (generation, 3-shot): 6.2%. Target: >6.2%.")
    print("=" * 60)

    device = torch.device('cuda')
    model = PageAnswerModel()
    model.compressor = model.compressor.to(device)
    model.hypernet = model.hypernet.to(device)
    model.answer_head = model.answer_head.to(device)

    if args.warm:
        warm_start(model, args.warm)

    train_dataset = GSM8KDataset(split='train')
    eval_dataset = GSM8KDataset(split='test', max_samples=500)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
    )

    optimizer = torch.optim.AdamW([
        {'params': list(model.compressor.parameters()), 'lr': 1e-4},
        {'params': list(model.hypernet.A_templates) + list(model.hypernet.B_templates), 'lr': 1e-3},
        {'params': (
            list(model.hypernet.page_project.parameters()) +
            [model.hypernet.page_query] +
            list(model.hypernet.page_attn.parameters()) +
            list(model.hypernet.page_norm.parameters()) +
            list(model.hypernet.combine.parameters())
        ), 'lr': 1e-3},
        {'params': list(model.answer_head.parameters()), 'lr': 1e-3},
    ])

    trainable = list(model.compressor.parameters()) + list(model.hypernet.parameters()) + list(model.answer_head.parameters())
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    accum = args.grad_accum
    best_exact = 0.0
    best_tol1 = 0.0
    for epoch in range(args.epochs):
        model.train()
        ep_loss = ep_mag = ep_sign = 0.0; nb = 0
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)):
            loss, ml, sl = model.forward_train(
                batch['question'], batch['final'], num_passes=args.passes,
            )
            (loss / accum).backward()
            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad()
            ep_loss += loss.item(); ep_mag += ml; ep_sign += sl; nb += 1

        exact, tol1 = evaluate(model, eval_dataset, device, num_passes=args.passes, batch_size=args.batch_size)
        if exact > best_exact:
            best_exact = exact
            torch.save({
                'epoch': epoch + 1,
                'compressor': model.compressor.state_dict(),
                'hypernet': model.hypernet.state_dict(),
                'answer_head': model.answer_head.state_dict(),
                'accuracy': exact,
                'tol1': tol1,
            }, 'checkpoints/gsm8k_answerhead_best.pt')
            print(f"  -> Saved checkpoint (exact={exact:.1f}%)")
        if tol1 > best_tol1:
            best_tol1 = tol1
        print(f"Epoch {epoch+1}: loss={ep_loss/nb:.4f} mag={ep_mag/nb:.4f} sign={ep_sign/nb:.4f} | exact={exact:.1f}% tol1%={tol1:.1f}% (best_exact={best_exact:.1f}%, best_tol={best_tol1:.1f}%, baseline=6.2%)")

    print(f"\nFinal: exact={best_exact:.1f}% tol1%={best_tol1:.1f}% (baseline: 6.2%)")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--warm', type=str, default='checkpoints/two_step_pages_best.pt')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--passes', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=12)
    p.add_argument('--grad_accum', type=int, default=1)
    args = p.parse_args()
    train(args)
