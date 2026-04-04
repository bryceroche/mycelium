"""
Train with Strategy Channel v3: Use existing hooks-based architecture + strategy.

Changes from baseline (53%):
- Compressor outputs state_delta (64) + strategy (512)
- Hypernetwork takes state (64) + strategy (512) = 576 floats
- Hooks stay the same (proven to work)

This is the minimal change to add the strategy channel.
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import sys
sys.path.insert(0, '/home/ubuntu/mycelium')

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.compressor_v3 import Compressor
from src.state_conditioned_lora_v3 import StateConditionedLoRA
from src.lora_hooks import apply_lora, remove_lora


class TwoStepArithmeticDataset(Dataset):
    def __init__(self, num_samples=10000, seed=42):
        random.seed(seed)
        self.samples = []
        
        for _ in range(num_samples):
            a = random.randint(2, 50)
            b = random.randint(2, 50)
            c = random.randint(2, 50)
            
            if random.random() < 0.5:
                op1 = '+'
                intermediate = a + b
            else:
                a = b * random.randint(2, 10)
                op1 = '/'
                intermediate = a // b
            
            if random.random() < 0.5:
                op2 = '+'
                final = intermediate + c
            else:
                op2 = '-'
                final = intermediate - c
            
            problem = f"({a} {op1} {b}) {op2} {c} ="
            self.samples.append({
                'problem': problem,
                'intermediate': intermediate,
                'final': final,
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class ThinkingModelV3(nn.Module):
    """ThinkingModel with strategy channel."""
    
    def __init__(self, model_name='meta-llama/Llama-3.2-1B'):
        super().__init__()
        
        self.transformer = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map='auto',
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Freeze transformer
        for param in self.transformer.parameters():
            param.requires_grad = False
        
        # Config
        self.d_model = self.transformer.config.hidden_size
        self.num_layers = self.transformer.config.num_hidden_layers
        num_kv_heads = self.transformer.config.num_key_value_heads
        head_dim = self.d_model // self.transformer.config.num_attention_heads
        d_kv = num_kv_heads * head_dim
        
        self.state_size = 64
        self.strategy_size = 512
        self.state_radius = (self.state_size ** 0.5)
        
        # Compressor v3: outputs state_delta + strategy
        self.compressor = Compressor(
            num_transformer_layers=self.num_layers,
            d_transformer=self.d_model,
            d_perceiver=1024,
            num_queries=4,
            num_perceiver_layers=7,
            state_size=self.state_size,
            strategy_size=self.strategy_size,
        )
        
        # Hypernetwork v3: takes state + strategy
        self.lora = StateConditionedLoRA(
            d_model=self.d_model,
            d_kv=d_kv,
            state_size=self.state_size,
            strategy_size=self.strategy_size,
            rank=4,
            num_layers=self.num_layers,
        )
        
        # Probe head
        self.probe_head = nn.Linear(self.state_size, 1)
    
    def thinking_pass(self, input_ids, attention_mask, state, strategy, pass_num):
        """Single thinking pass with hooks-based LoRA."""
        
        # Apply LoRA with state + strategy
        lora_mods = self.lora(state, strategy)
        
        # Register hooks
        from src.lora_hooks import LoRAHookManager
        manager = LoRAHookManager(self.transformer)
        manager.apply(lora_mods)
        
        try:
            # Forward pass
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = list(outputs.hidden_states[1:])  # skip embedding
        finally:
            manager.remove()
        
        # Compress -> state_delta + new_strategy
        state_delta, new_strategy = self.compressor(hidden_states, pass_num)
        
        # Accumulate state on hypersphere
        new_state = state + state_delta
        new_state = F.normalize(new_state, dim=-1) * self.state_radius
        
        return new_state, new_strategy
    
    def forward(self, problems, num_passes=2):
        # Tokenize
        inputs = self.tokenizer(
            problems, return_tensors='pt', padding=True,
            truncation=True, max_length=128,
        )
        input_ids = inputs['input_ids'].to(self.transformer.device)
        attention_mask = inputs['attention_mask'].to(self.transformer.device)
        
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Initialize
        state = torch.zeros(batch_size, self.state_size, device=device)
        strategy = torch.zeros(batch_size, self.strategy_size, device=device)
        
        all_states = []
        
        for pass_num in range(num_passes):
            state, strategy = self.thinking_pass(
                input_ids, attention_mask, state, strategy, pass_num
            )
            all_states.append(state)
        
        return state, all_states


def train():
    print("Training with Strategy Channel v3 (Hooks + Strategy)")
    print("=" * 60)
    print("Architecture:")
    print("  - Hooks-based LoRA (proven to work)")
    print("  - Compressor outputs state (64) + strategy (512)")
    print("  - Hypernetwork takes 576 floats (2.25 per scale)")
    print("=" * 60)
    
    device = torch.device('cuda')
    
    model = ThinkingModelV3()
    model.compressor = model.compressor.to(device)
    model.lora = model.lora.to(device)
    model.probe_head = model.probe_head.to(device)
    
    train_dataset = TwoStepArithmeticDataset(num_samples=5000, seed=42)
    eval_dataset = TwoStepArithmeticDataset(num_samples=500, seed=123)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    trainable = list(model.compressor.parameters()) +                 list(model.lora.parameters()) +                 list(model.probe_head.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")
    
    num_epochs = 10
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            problems = batch['problem']
            # Normalize targets to [-1, 1] range (like baseline)
            intermediates = torch.tensor(
                [float(s) / 1000.0 for s in batch['intermediate']],
                dtype=torch.float32, device=device
            )
            finals = torch.tensor(
                [float(s) / 1000.0 for s in batch['final']],
                dtype=torch.float32, device=device
            )
            
            optimizer.zero_grad()
            
            _, all_states = model(problems, num_passes=2)
            
            pred1 = model.probe_head(all_states[0]).squeeze(-1)
            pred2 = model.probe_head(all_states[1]).squeeze(-1)
            
            loss = F.mse_loss(pred1, intermediates) + F.mse_loss(pred2, finals)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        # Eval
        model.eval()
        correct_b1 = correct_b2 = correct_both = total = 0
        
        with torch.no_grad():
            for i in range(0, len(eval_dataset), 8):
                batch_samples = [eval_dataset[j] for j in range(i, min(i+8, len(eval_dataset)))]
                problems = [s['problem'] for s in batch_samples]
                # Scale targets for comparison
                intermediates = torch.tensor([s['intermediate'] for s in batch_samples], device=device)
                finals = torch.tensor([s['final'] for s in batch_samples], device=device)
                
                _, all_states = model(problems, num_passes=2)
                
                # Scale predictions back and round
                pred1 = (model.probe_head(all_states[0]).squeeze(-1) * 1000).round()
                pred2 = (model.probe_head(all_states[1]).squeeze(-1) * 1000).round()
                
                b1 = (pred1 == intermediates).cpu().numpy()
                b2 = (pred2 == finals).cpu().numpy()
                
                correct_b1 += b1.sum()
                correct_b2 += b2.sum()
                correct_both += (b1 & b2).sum()
                total += len(batch_samples)
        
        acc_b1 = 100 * correct_b1 / total
        acc_b2 = 100 * correct_b2 / total
        acc_both = 100 * correct_both / total
        
        if acc_both > best_accuracy:
            best_accuracy = acc_both
        
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f} | B1={acc_b1:.1f}% B2={acc_b2:.1f}% Both={acc_both:.1f}% (best={best_accuracy:.1f}%)")
    
    print(f"\nFinal: {best_accuracy:.1f}% (baseline: 53%)")


if __name__ == '__main__':
    train()
