"""
Robust evaluation of three-step checkpoint.
5 random seeds x 500 problems. Same protocol as two-step eval.
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.insert(0, '/home/ubuntu/mycelium')

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.compressor_v3 import Compressor
from src.state_conditioned_lora_v3 import StateConditionedLoRA
from src.additive_lora import AdditiveLoRAManager


class ThreeStepArithmeticDataset:
    def __init__(self, num_samples=500, seed=42):
        random.seed(seed)
        self.samples = []
        for _ in range(num_samples):
            a = random.randint(2, 50)
            b = random.randint(2, 50)
            c = random.randint(2, 50)
            d = random.randint(2, 50)
            if random.random() < 0.5:
                op1 = '+'; v1 = a + b
            else:
                a = b * random.randint(2, 10); op1 = '/'; v1 = a // b
            if random.random() < 0.5:
                op2 = '+'; v2 = v1 + c
            else:
                op2 = '-'; v2 = v1 - c
            if random.random() < 0.5:
                op3 = '+'; v3 = v2 + d
            else:
                op3 = '-'; v3 = v2 - d
            problem = f"(({a} {op1} {b}) {op2} {c}) {op3} {d} ="
            self.samples.append({'problem': problem, 'v3': v3})


def build_model(checkpoint_path, device):
    model_name = 'unsloth/Llama-3.2-1B'
    transformer = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map='auto',
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    for param in transformer.parameters():
        param.requires_grad = False

    d_model = transformer.config.hidden_size
    num_layers = transformer.config.num_hidden_layers
    num_kv_heads = transformer.config.num_key_value_heads
    head_dim = d_model // transformer.config.num_attention_heads
    d_kv = num_kv_heads * head_dim

    compressor = Compressor(
        num_transformer_layers=num_layers, d_transformer=d_model,
        d_perceiver=1024, num_queries=4, num_perceiver_layers=7,
        state_size=64, strategy_size=512,
    ).to(device)
    lora = StateConditionedLoRA(
        d_model=d_model, d_kv=d_kv, state_size=64, strategy_size=512,
        rank=4, num_layers=num_layers,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    compressor.load_state_dict(ckpt['compressor'])
    lora.load_state_dict(ckpt['lora'])
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}, reported accuracy {ckpt['accuracy']:.1f}%")
    return transformer, tokenizer, compressor, lora


def thinking_pass(transformer, compressor, lora, input_ids, attention_mask, state, strategy, pass_num):
    lora_mods = lora(state, strategy)
    manager = AdditiveLoRAManager(transformer)
    manager.apply(lora_mods)
    try:
        outputs = transformer(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = list(outputs.hidden_states[1:])
    finally:
        manager.remove()
    state_delta, new_strategy = compressor(hidden_states, pass_num)
    state_radius = 64 ** 0.5
    new_state = F.normalize(state + state_delta, dim=-1) * state_radius
    return new_state, new_strategy


def evaluate_seed(transformer, tokenizer, compressor, lora, eval_data, device, init_seed):
    torch.manual_seed(init_seed)
    correct = total = 0
    for i in range(0, len(eval_data.samples), 8):
        batch = eval_data.samples[i:i+8]
        problems = [s['problem'] for s in batch]
        golds = [s['v3'] for s in batch]

        inputs = tokenizer(problems, return_tensors='pt', padding=True, truncation=True, max_length=128)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        bs = input_ids.size(0)

        state = torch.randn(bs, 64, device=device)
        state = F.normalize(state, dim=-1) * (64 ** 0.5)
        strategy = torch.zeros(bs, 512, device=device)

        for pass_num in range(3):
            state, strategy = thinking_pass(
                transformer, compressor, lora, input_ids, attention_mask, state, strategy, pass_num
            )

        lora_mods = lora(state, strategy)
        manager = AdditiveLoRAManager(transformer)
        manager.apply(lora_mods)
        try:
            generated = transformer.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.pad_token_id,
            )
        finally:
            manager.remove()

        for j in range(bs):
            prompt_len = input_ids[j].size(0)
            gen_text = tokenizer.decode(generated[j][prompt_len:], skip_special_tokens=True).strip().rstrip('.')
            try:
                pred = int(gen_text.split()[0]) if gen_text else None
            except (ValueError, IndexError):
                pred = None
            if pred == golds[j]:
                correct += 1
            total += 1

    return 100.0 * correct / total if total > 0 else 0.0


def main():
    print("=" * 60)
    print("ROBUST EVALUATION: three_step_best checkpoint")
    print("5 seeds x 500 problems = 2500 total evaluations")
    print("=" * 60)

    device = torch.device('cuda')
    transformer, tokenizer, compressor, lora = build_model(
        '/home/ubuntu/mycelium/checkpoints/three_step_best.pt', device
    )
    eval_data = ThreeStepArithmeticDataset(num_samples=500, seed=999)

    seeds = [42, 123, 456, 789, 1337]
    accuracies = []
    for seed in seeds:
        with torch.no_grad():
            acc = evaluate_seed(transformer, tokenizer, compressor, lora, eval_data, device, seed)
        accuracies.append(acc)
        print(f"  Seed {seed:>5d}: {acc:.1f}%")

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    print(f"\n{'=' * 60}")
    print(f"RESULTS:")
    print(f"  Per-seed: {', '.join(f'{a:.1f}%' for a in accuracies)}")
    print(f"  Mean:     {mean_acc:.1f}%")
    print(f"  Std:      {std_acc:.1f}%")
    print(f"  Range:    {min(accuracies):.1f}% - {max(accuracies):.1f}%")
    print(f"  Effective per-step: {(mean_acc/100)**(1/3) * 100:.1f}% (cube root of three-step)")
    print(f"{'=' * 60}")

    if std_acc > 15:
        print("WARNING: High variance — model is init-sensitive")
    elif std_acc > 5:
        print("NOTE: Moderate variance")
    else:
        print("GOOD: Low variance — model is robust to init")


if __name__ == '__main__':
    main()
