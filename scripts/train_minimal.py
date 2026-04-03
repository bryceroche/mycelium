#!/usr/bin/env python3
"""Minimal curriculum training with chat template."""

import os, json, random, re
from datetime import datetime

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

CONFIG = {
    "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "num_layers": 22,
    "d_model": 2048,
    "state_size": 64,
    "num_tokens": 4,
    "num_queries": 4,
    "num_perceiver_layers": 3,
    "d_perceiver": 256,
    "max_passes": 3,
    "batch_size": 8,
    "learning_rate": 1e-4,
    "epochs_per_level": 3,
}


class SimplePerceiver(nn.Module):
    def __init__(self, num_layers=22, d_model=2048, d_perceiver=256, 
                 num_queries=4, num_perceiver_layers=3, state_size=64):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, d_perceiver) * 0.02)
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.input_project = nn.Linear(d_model, d_perceiver)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(d_perceiver, d_perceiver), nn.GELU(), nn.Linear(d_perceiver, d_perceiver))
            for _ in range(num_perceiver_layers)
        ])
        self.project_out = nn.Linear(d_perceiver, state_size // num_queries)
    
    def forward(self, all_layer_hidden):
        weights = torch.softmax(self.layer_weights, dim=0)
        stacked = torch.stack([h.float() for h in all_layer_hidden], dim=0)
        combined = (stacked * weights.view(-1, 1, 1, 1)).sum(dim=0)
        pooled = combined.mean(dim=1)
        x = self.input_project(pooled)
        for layer in self.layers:
            x = x + layer(x)
        batch_size = pooled.size(0)
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        x_expanded = x.unsqueeze(1) + queries
        out = self.project_out(x_expanded)
        return out.flatten(start_dim=1)


class StateInjector(nn.Module):
    def __init__(self, state_size=64, d_model=2048, num_tokens=4):
        super().__init__()
        self.num_tokens = num_tokens
        self.chunk_size = state_size // num_tokens
        self.project = nn.Linear(self.chunk_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(num_tokens, d_model) * 0.02)
    
    def forward(self, state, dtype=torch.bfloat16):
        chunks = state.float().reshape(-1, self.num_tokens, self.chunk_size)
        tokens = self.project(chunks) + self.pos_embed
        return tokens.to(dtype)


class ThinkingSystem(nn.Module):
    def __init__(self, config):
        super().__init__()
        print(f"Loading {config['model_name']}...")
        self.transformer = AutoModelForCausalLM.from_pretrained(
            config['model_name'], torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.perceiver = SimplePerceiver(
            num_layers=config['num_layers'], d_model=config['d_model'],
            d_perceiver=config['d_perceiver'], num_queries=config['num_queries'],
            num_perceiver_layers=config['num_perceiver_layers'], state_size=config['state_size'])
        
        self.injector = StateInjector(state_size=config['state_size'],
            d_model=config['d_model'], num_tokens=config['num_tokens'])
        
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.state_size = config['state_size']
        self.max_passes = config['max_passes']
        
        for p in self.transformer.parameters():
            p.requires_grad = False
        print(f"Trainable: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    @property
    def device(self):
        return next(self.transformer.parameters()).device
    
    def format_prompt(self, question):
        """Format question with chat template."""
        messages = [{"role": "user", "content": question}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    def think(self, question, max_passes=None):
        if max_passes is None:
            max_passes = self.max_passes
        
        prompt = self.format_prompt(question)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        embeds = self.transformer.get_input_embeddings()(inputs.input_ids)
        
        state = torch.zeros(1, self.state_size, device=self.device)
        
        for pass_num in range(max_passes):
            state_tokens = self.injector(state, dtype=embeds.dtype)
            combined = torch.cat([state_tokens, embeds], dim=1)
            
            with torch.no_grad():
                outputs = self.transformer(inputs_embeds=combined, output_hidden_states=True)
            
            hidden = list(outputs.hidden_states[1:])
            delta = self.perceiver(hidden)
            state = state + self.alpha * delta
        
        return state
    
    def generate(self, question, state, max_tokens=32):
        prompt = self.format_prompt(question)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        embeds = self.transformer.get_input_embeddings()(inputs.input_ids)
        state_tokens = self.injector(state, dtype=embeds.dtype)
        combined = torch.cat([state_tokens, embeds], dim=1)
        
        state_mask = torch.ones(1, state_tokens.size(1), device=self.device)
        full_mask = torch.cat([state_mask, inputs.attention_mask], dim=1)
        
        outputs = self.transformer.generate(
            inputs_embeds=combined, attention_mask=full_mask,
            max_new_tokens=max_tokens, do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the answer part after the question
        if "assistant" in response.lower():
            response = response.split("assistant")[-1]
        return response


def extract_answer(text):
    """Extract numerical answer from response."""
    # Find numbers in text
    nums = re.findall(r'\d+', text)
    if nums:
        return nums[-1]  # Last number is usually the answer
    return None


def check_answer(pred, gold):
    if pred is None: return False
    try: return abs(float(pred) - float(gold)) < 0.01
    except: return False


def load_data(filepath):
    with open(filepath) as f:
        return [json.loads(line) for line in f]


def train_level(model, data, config, level_name):
    print(f"\n{'='*50}\nTraining: {level_name}\n{'='*50}")
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=config['learning_rate'])
    
    random.shuffle(data)
    n_train = int(0.9 * len(data))
    train_data, eval_data = data[:n_train], data[n_train:]
    print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")
    
    best_acc = 0.0
    
    for epoch in range(config['epochs_per_level']):
        model.train()
        random.shuffle(train_data)
        
        correct, total = 0, 0
        pbar = tqdm(range(0, len(train_data), config['batch_size']), desc=f"E{epoch+1}")
        
        for i in pbar:
            batch = train_data[i:i+config['batch_size']]
            optimizer.zero_grad()
            batch_loss = 0.0
            
            for item in batch:
                try:
                    state = model.think(item['question'])
                    batch_loss += 0.01 * (state ** 2).mean()
                    
                    with torch.no_grad():
                        output = model.generate(item['question'], state)
                        pred = extract_answer(output)
                        if check_answer(pred, item['answer']):
                            correct += 1
                    total += 1
                except Exception as e:
                    print(f"Err: {e}")
            
            if batch_loss > 0:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            pbar.set_postfix(acc=f"{correct/max(total,1):.1%}")
        
        # Eval
        model.eval()
        eval_correct = 0
        with torch.no_grad():
            for item in eval_data[:50]:
                try:
                    state = model.think(item['question'])
                    output = model.generate(item['question'], state)
                    if check_answer(extract_answer(output), item['answer']):
                        eval_correct += 1
                except: pass
        
        eval_acc = eval_correct / min(50, len(eval_data))
        print(f"Epoch {epoch+1}: train={correct/max(total,1):.1%}, eval={eval_acc:.1%}")
        
        if eval_acc > best_acc:
            best_acc = eval_acc
            torch.save(model.state_dict(), f"checkpoints/{level_name}_best.pt")
    
    return best_acc


def main():
    print("="*50)
    print("Mycelium v18 Curriculum Training")
    print(f"Started: {datetime.now()}")
    print("="*50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    os.makedirs("checkpoints", exist_ok=True)
    
    model = ThinkingSystem(CONFIG).to(device)
    
    # Quick sanity check
    print("\nSanity check:")
    model.eval()
    with torch.no_grad():
        state = model.think("What is 2 + 3?")
        out = model.generate("What is 2 + 3?", state)
        print(f"  Q: What is 2 + 3?")
        print(f"  A: {out[:80]}")
        print(f"  Extracted: {extract_answer(out)}")
    
    levels = [
        ("L0_single", "data/curriculum/level_0_single_step.jsonl"),
        ("L1_two", "data/curriculum/level_1_two_step.jsonl"),
        ("L2_three", "data/curriculum/level_2_three_step.jsonl"),
    ]
    
    for name, path in levels:
        if not os.path.exists(path):
            print(f"Skip {name}")
            continue
        best = train_level(model, load_data(path), CONFIG, name)
        print(f"{name} done. Best: {best:.1%}")
        torch.save(model.state_dict(), f"checkpoints/{name}_final.pt")
    
    print(f"\nDone at {datetime.now()}")


if __name__ == "__main__":
    main()
