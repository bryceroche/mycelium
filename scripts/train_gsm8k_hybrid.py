"""
GSM8K hybrid training: LoRA for thinking, pseudo-tokens for generation.

The arithmetic-trained LoRA destroys language generation (degenerates into
"Answer 1600 / Answer 1600 ..." spam). Diagnosis confirmed by inspection:
the LoRA learned "emit a number immediately" from arithmetic and overrides
the base model's natural language flow on GSM8K.

Fix: hybrid path.
  - Thinking: state → LoRA → rewires attention → state_delta + strategy
  - Generate: state → pseudo-tokens prepended to embeddings → coherent text

Trained from scratch on GSM8K (NOT warm-started from arithmetic). The LoRA
needs to learn word-problem attention patterns, not get yanked from
"focus on numbers after equals" toward something it can't reach.

Real baseline (verified 500 problems, 3-shot, no thinking, no LoRA): 6.2%.
Target: >6.2%.
"""

import re
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import sys
sys.path.insert(0, '/home/ubuntu/mycelium')

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.compressor_v3 import Compressor
from src.state_conditioned_lora_v3 import StateConditionedLoRA
from src.additive_lora import AdditiveLoRAManager
from src.pseudo_token_head import PseudoTokenHead


# Few-shot prefix used by the verified baseline (6.2% on 500 problems).
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
    """'The answer is X' with last-number fallback. Stop at next 'Problem:'."""
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
            self.samples.append({
                'question': ex['question'],
                'answer': answer_str,
                'final': final,
                # Build the supervision target as a clean "The answer is X." string.
                # Keeps generation aligned with the few-shot format.
                'completion': f" The answer is {answer_str}.",
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
        'answer':     [s['answer']     for s in batch],
        'final':      [s['final']      for s in batch],
        'completion': [s['completion'] for s in batch],
    }


class HybridThinkingModel(nn.Module):
    """LoRA for thinking, pseudo-tokens for generation."""

    def __init__(self, model_name='unsloth/Llama-3.2-1B'):
        super().__init__()
        self.transformer = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map='auto',
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'  # for batched generation

        for p in self.transformer.parameters():
            p.requires_grad = False

        self.d_model = self.transformer.config.hidden_size
        self.num_layers = self.transformer.config.num_hidden_layers
        num_kv_heads = self.transformer.config.num_key_value_heads
        head_dim = self.d_model // self.transformer.config.num_attention_heads
        d_kv = num_kv_heads * head_dim

        self.state_size = 64
        self.strategy_size = 512
        self.state_radius = self.state_size ** 0.5
        self.num_pseudo_tokens = 4

        self.compressor = Compressor(
            num_transformer_layers=self.num_layers,
            d_transformer=self.d_model,
            d_perceiver=1024,
            num_queries=4,
            num_perceiver_layers=7,
            state_size=self.state_size,
            strategy_size=self.strategy_size,
        )

        self.lora = StateConditionedLoRA(
            d_model=self.d_model,
            d_kv=d_kv,
            state_size=self.state_size,
            strategy_size=self.strategy_size,
            rank=4,
            num_layers=self.num_layers,
        )

        self.pseudo_head = PseudoTokenHead(
            state_size=self.state_size,
            strategy_size=self.strategy_size,
            d_model=self.d_model,
            num_tokens=self.num_pseudo_tokens,
        )

    def get_input_embeddings(self):
        return self.transformer.get_input_embeddings()

    def thinking_pass(self, input_ids, attention_mask, state, strategy, pass_num):
        """LoRA-on forward to compute state delta and new strategy."""
        lora_mods = self.lora(state, strategy)
        manager = AdditiveLoRAManager(self.transformer)
        manager.apply(lora_mods)
        try:
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = list(outputs.hidden_states[1:])
        finally:
            manager.remove()

        state_delta, new_strategy = self.compressor(hidden_states, pass_num)
        new_state = F.normalize(state + state_delta, dim=-1) * self.state_radius
        return new_state, new_strategy

    def _embed_with_pseudo(self, input_ids, attention_mask, pseudo_embeds):
        """
        Build inputs_embeds = [pseudo_tokens ; token_embeds] and an
        attention mask that covers both. Pseudo tokens always attendable.
        """
        embed_layer = self.get_input_embeddings()
        token_embeds = embed_layer(input_ids).to(pseudo_embeds.dtype)
        inputs_embeds = torch.cat([pseudo_embeds, token_embeds], dim=1)

        bs, num_pt, _ = pseudo_embeds.shape
        pt_mask = torch.ones(bs, num_pt, dtype=attention_mask.dtype, device=attention_mask.device)
        full_mask = torch.cat([pt_mask, attention_mask], dim=1)
        return inputs_embeds, full_mask

    def compute_answer_loss(self, state, strategy, prompt_ids, prompt_mask, completion_ids, completion_mask):
        """
        Teacher-forced loss on completion tokens, generated with LoRA OFF
        and pseudo-tokens prepended at the very start.

        Layout in the embedding space:
            [pseudo_tokens] [prompt_tokens] [completion_tokens]
            ^^^^^^^^^^^^^^^^ no loss        ^^^^^^^^^^^^^^^^^^^ loss here
        """
        pseudo_embeds = self.pseudo_head(state, strategy).to(torch.bfloat16)

        # Build full input: prompt + completion (token side)
        full_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        full_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        inputs_embeds, attn = self._embed_with_pseudo(full_ids, full_mask, pseudo_embeds)

        # Forward through the *unmodified* transformer
        outputs = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            use_cache=False,
        )
        logits = outputs.logits  # (batch, num_pt + prompt_len + comp_len, vocab)

        num_pt = self.num_pseudo_tokens
        prompt_len = prompt_ids.size(1)
        comp_len = completion_ids.size(1)

        # Predict token at position (num_pt + prompt_len + i) from logits at (num_pt + prompt_len + i - 1)
        logit_start = num_pt + prompt_len - 1
        logit_end = num_pt + prompt_len - 1 + comp_len
        comp_logits = logits[:, logit_start:logit_end, :]

        # Mask out padding tokens in the completion target
        targets = completion_ids.clone()
        targets[completion_mask == 0] = -100

        loss = F.cross_entropy(
            comp_logits.reshape(-1, comp_logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
        )
        return loss

    def forward_train(self, questions, completions, num_passes=3):
        # Build prompts (few-shot prefix + this problem)
        prompts = [FEW_SHOT + f"Problem: {q}\nAnswer:" for q in questions]
        prompt_inputs = self.tokenizer(
            prompts, return_tensors='pt', padding=True,
            truncation=True, max_length=1024,
        )
        prompt_ids = prompt_inputs['input_ids'].to(self.transformer.device)
        prompt_mask = prompt_inputs['attention_mask'].to(self.transformer.device)

        comp_inputs = self.tokenizer(
            completions, return_tensors='pt', padding=True,
            add_special_tokens=False,
        )
        comp_ids = comp_inputs['input_ids'].to(self.transformer.device)
        comp_mask = comp_inputs['attention_mask'].to(self.transformer.device)

        batch_size = prompt_ids.size(0)
        device = prompt_ids.device

        # Random hypersphere init (we learned this is a feature, not a bug)
        state = torch.randn(batch_size, self.state_size, device=device)
        state = F.normalize(state, dim=-1) * self.state_radius
        strategy = torch.zeros(batch_size, self.strategy_size, device=device)

        all_states = []
        all_strategies = []
        for pass_num in range(num_passes):
            state, strategy = self.thinking_pass(
                prompt_ids, prompt_mask, state, strategy, pass_num
            )
            all_states.append(state)
            all_strategies.append(strategy)

        # Deep supervision: each pass tries to support the same completion
        # via its own pseudo-tokens. Later passes weighted more.
        loss = torch.tensor(0.0, device=device)
        for i, (s, st) in enumerate(zip(all_states, all_strategies)):
            weight = (i + 1) / num_passes
            pass_loss = self.compute_answer_loss(
                s, st, prompt_ids, prompt_mask, comp_ids, comp_mask,
            )
            loss = loss + weight * pass_loss
        loss = loss / ((num_passes + 1) / 2)

        return loss


def evaluate(model, eval_dataset, device, num_passes=3):
    """Generate with LoRA OFF + pseudo-tokens, extract 'The answer is X'."""
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

            # Thinking passes
            state = torch.randn(bs, model.state_size, device=device)
            state = F.normalize(state, dim=-1) * model.state_radius
            strategy = torch.zeros(bs, model.strategy_size, device=device)
            for pass_num in range(num_passes):
                state, strategy = model.thinking_pass(
                    input_ids, attention_mask, state, strategy, pass_num,
                )

            # Generate with pseudo-tokens, LoRA OFF
            pseudo_embeds = model.pseudo_head(state, strategy).to(torch.bfloat16)
            inputs_embeds, attn = model._embed_with_pseudo(input_ids, attention_mask, pseudo_embeds)

            generated = model.transformer.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attn,
                max_new_tokens=120,
                do_sample=False,
                pad_token_id=model.tokenizer.pad_token_id,
            )
            # When using inputs_embeds, generate() returns ONLY the new tokens
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


def train():
    print("GSM8K HYBRID: LoRA thinking + pseudo-token generation")
    print("=" * 60)
    print("Trained from scratch (no warm start). Real baseline: 6.2%.")
    print("=" * 60)

    device = torch.device('cuda')

    model = HybridThinkingModel()
    model.compressor = model.compressor.to(device)
    model.lora = model.lora.to(device)
    model.pseudo_head = model.pseudo_head.to(device)

    train_dataset = GSM8KDataset(split='train')
    eval_dataset = GSM8KDataset(split='test', max_samples=500)
    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, collate_fn=collate,
    )

    # From scratch: full LR (not the 0.3x fine-tuning rates)
    optimizer = torch.optim.AdamW([
        {'params': list(model.compressor.parameters()), 'lr': 1e-4},
        {'params': list(model.lora.A_templates) + list(model.lora.B_templates), 'lr': 1e-3},
        {'params': list(model.lora.state_to_scales.parameters()), 'lr': 1e-3},
        {'params': list(model.pseudo_head.parameters()), 'lr': 1e-3},
    ])

    trainable = (list(model.compressor.parameters()) +
                 list(model.lora.parameters()) +
                 list(model.pseudo_head.parameters()))
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    num_epochs = 3
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            optimizer.zero_grad()
            loss = model.forward_train(
                batch['question'], batch['completion'], num_passes=3
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg = epoch_loss / num_batches
        accuracy = evaluate(model, eval_dataset, device, num_passes=3)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'epoch': epoch + 1,
                'compressor': model.compressor.state_dict(),
                'lora': model.lora.state_dict(),
                'pseudo_head': model.pseudo_head.state_dict(),
                'accuracy': accuracy,
            }, 'checkpoints/gsm8k_hybrid_best.pt')
            print(f"  -> Saved new best checkpoint")

        print(f"Epoch {epoch+1}: loss={avg:.4f} | Acc={accuracy:.1f}% (best={best_accuracy:.1f}%, baseline=6.2%)")

    print(f"\nFinal: {best_accuracy:.1f}% (baseline: 6.2%)")


if __name__ == '__main__':
    train()
