"""
GSM8K v21: page-based state + hybrid generation.

  Thinking: pages drive PageAttentionHypernetwork → LoRA → Llama (LoRA ON)
  Generate: pages → PageToTokens → prepend pseudo-tokens, LoRA OFF

Warm-starts perceiver + LoRA templates + page-attention hypernet from
two_step_pages_best.pt (the v21 checkpoint that just hit 86.2% on two-step).
PageToTokens is fresh.

Real GSM8K baseline (3-shot, 500 problems): 6.2%. Target: >6.2%.
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
from src.page_to_tokens import PageToTokens
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


def extract_answer(text):
    if "\nProblem:" in text:
        text = text.split("\nProblem:")[0]
    m = re.search(r'[Tt]he answer is\s*\$?(-?[\d,]+\.?\d*)', text)
    if m:
        try:
            v = float(m.group(1).replace(',', ''))
            return int(v) if v == int(v) else v
        except ValueError:
            pass
    nums = re.findall(r'-?[\d,]+\.?\d*', text)
    if not nums:
        return None
    try:
        v = float(nums[-1].replace(',', ''))
        return int(v) if v == int(v) else v
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
            answer_str = str(int(final)) if final == int(final) else str(final)
            # Full reasoning trace: strip <<calc>> annotations, replace ####
            # with "The answer is". Teaches the model to REASON, not just guess.
            trace = re.sub(r'<<[^>]*>>', '', ex['answer'])
            trace = trace.replace('####', 'The answer is').strip()
            if not trace.endswith('.'):
                trace = trace + '.'
            self.samples.append({
                'question': ex['question'],
                'answer': answer_str,
                'final': final,
                'completion': ' ' + trace,
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
        'question':   [s['question']   for s in batch],
        'completion': [s['completion'] for s in batch],
        'final':      [s['final']      for s in batch],
    }


class PageHybridModel(nn.Module):
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
        self.num_pseudo_tokens = 8

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
        self.page_to_tokens = PageToTokens(
            page_size=self.page_size,
            d_model=self.d_model,
            num_tokens=self.num_pseudo_tokens,
        )

    def get_input_embeddings(self):
        return self.transformer.get_input_embeddings()

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

    def _embed_with_pseudo(self, input_ids, attention_mask, pseudo_embeds):
        embed_layer = self.get_input_embeddings()
        token_embeds = embed_layer(input_ids).to(pseudo_embeds.dtype)
        inputs_embeds = torch.cat([pseudo_embeds, token_embeds], dim=1)
        bs, num_pt, _ = pseudo_embeds.shape
        pt_mask = torch.ones(bs, num_pt, dtype=attention_mask.dtype, device=attention_mask.device)
        full_mask = torch.cat([pt_mask, attention_mask], dim=1)
        return inputs_embeds, full_mask

    def compute_answer_loss(self, state_pages, prompt_ids, prompt_mask, comp_ids, comp_mask):
        pseudo_embeds = self.page_to_tokens(state_pages).to(torch.bfloat16)
        full_ids = torch.cat([prompt_ids, comp_ids], dim=1)
        full_mask = torch.cat([prompt_mask, comp_mask], dim=1)
        inputs_embeds, attn = self._embed_with_pseudo(full_ids, full_mask, pseudo_embeds)
        outputs = self.transformer(inputs_embeds=inputs_embeds, attention_mask=attn, use_cache=False)
        logits = outputs.logits

        num_pt = self.num_pseudo_tokens
        prompt_len = prompt_ids.size(1)
        comp_len = comp_ids.size(1)
        logit_start = num_pt + prompt_len - 1
        logit_end = logit_start + comp_len
        comp_logits = logits[:, logit_start:logit_end, :]

        targets = comp_ids.clone()
        targets[comp_mask == 0] = -100
        loss = F.cross_entropy(
            comp_logits.reshape(-1, comp_logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
        )
        return loss

    def forward_train(self, questions, completions, num_passes=3):
        prompts = [FEW_SHOT + f"Problem: {q}\nAnswer:" for q in questions]
        prompt_inputs = self.tokenizer(
            prompts, return_tensors='pt', padding=True,
            truncation=True, max_length=1024,
        )
        prompt_ids = prompt_inputs['input_ids'].to(self.transformer.device)
        prompt_mask = prompt_inputs['attention_mask'].to(self.transformer.device)

        comp_inputs = self.tokenizer(
            completions, return_tensors='pt', padding=True, add_special_tokens=False,
        )
        comp_ids = comp_inputs['input_ids'].to(self.transformer.device)
        comp_mask = comp_inputs['attention_mask'].to(self.transformer.device)

        batch_size = prompt_ids.size(0)
        device = prompt_ids.device

        state_pages = []
        strategy = torch.zeros(batch_size, self.strategy_size, device=device)
        for pass_num in range(num_passes):
            page, strategy = self.thinking_pass(
                prompt_ids, prompt_mask, state_pages, strategy, pass_num,
            )
            state_pages.append(page)

        # Final-only answer loss: 1 forward instead of num_passes.
        # Early pages still get gradient through hypernet → LoRA → later passes.
        loss = self.compute_answer_loss(state_pages, prompt_ids, prompt_mask, comp_ids, comp_mask)
        return loss


def evaluate(model, eval_dataset, device, num_passes=3):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(eval_dataset), 4):
            batch = eval_dataset.samples[i:i + 4]
            questions = [s['question'] for s in batch]
            golds = [s['final'] for s in batch]
            prompts = [FEW_SHOT + f"Problem: {q}\nAnswer:" for q in questions]
            inputs = model.tokenizer(
                prompts, return_tensors='pt', padding=True,
                truncation=True, max_length=1024,
            )
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            bs = input_ids.size(0)

            state_pages = []
            strategy = torch.zeros(bs, model.strategy_size, device=device)
            for pass_num in range(num_passes):
                page, strategy = model.thinking_pass(
                    input_ids, attention_mask, state_pages, strategy, pass_num,
                )
                state_pages.append(page)

            pseudo_embeds = model.page_to_tokens(state_pages).to(torch.bfloat16)
            inputs_embeds, attn = model._embed_with_pseudo(input_ids, attention_mask, pseudo_embeds)
            generated = model.transformer.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attn,
                max_new_tokens=120,
                do_sample=False,
                pad_token_id=model.tokenizer.pad_token_id,
            )
            for j in range(bs):
                gen_text = model.tokenizer.decode(generated[j], skip_special_tokens=True)
                pred = extract_answer(gen_text)
                gold = golds[j]
                if gold == int(gold):
                    gold = int(gold)
                if pred == gold:
                    correct += 1
                total += 1
    return 100.0 * correct / total if total > 0 else 0.0


def warm_start(model, ckpt_path):
    print(f"Warm-starting from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    miss, unex = model.compressor.load_state_dict(ckpt['compressor'], strict=False)
    print(f"  compressor: missing={len(miss)} unexpected={len(unex)}")
    miss, unex = model.hypernet.load_state_dict(ckpt['hypernet'], strict=False)
    print(f"  hypernet: missing={len(miss)} unexpected={len(unex)}")


def train(args):
    print("=" * 60)
    print("GSM8K v21: pages → LoRA (think) + pages → pseudo-tokens (gen)")
    print("=" * 60)
    print(f"Baseline: 6.2% (3-shot, 500 problems). Target: >6.2%.")
    print("=" * 60)

    device = torch.device('cuda')
    model = PageHybridModel()
    model.compressor = model.compressor.to(device)
    model.hypernet = model.hypernet.to(device)
    model.page_to_tokens = model.page_to_tokens.to(device)

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
        {'params': list(model.page_to_tokens.parameters()), 'lr': 1e-3},
    ])

    trainable = list(model.compressor.parameters()) + list(model.hypernet.parameters()) + list(model.page_to_tokens.parameters())
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    best = 0.0
    accum = args.grad_accum
    for epoch in range(args.epochs):
        model.train()
        ep_loss = 0.0; nb = 0
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)):
            loss = model.forward_train(batch['question'], batch['completion'], num_passes=args.passes)
            (loss / accum).backward()
            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad()
            ep_loss += loss.item(); nb += 1

        acc = evaluate(model, eval_dataset, device, num_passes=args.passes)
        if acc > best:
            best = acc
            torch.save({
                'epoch': epoch + 1,
                'compressor': model.compressor.state_dict(),
                'hypernet': model.hypernet.state_dict(),
                'page_to_tokens': model.page_to_tokens.state_dict(),
                'accuracy': acc,
            }, 'checkpoints/gsm8k_pages_best.pt')
            print(f"  -> Saved checkpoint (acc={acc:.1f}%)")
        print(f"Epoch {epoch+1}: loss={ep_loss/nb:.4f} | Acc={acc:.1f}% (best={best:.1f}%, baseline=6.2%)")

    print(f"\nFinal: {best:.1f}% (baseline: 6.2%)")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--warm', type=str, default='checkpoints/two_step_pages_best.pt')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--passes', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--grad_accum', type=int, default=2)
    args = p.parse_args()
    train(args)
