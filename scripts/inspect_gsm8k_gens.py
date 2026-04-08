"""
Inspect what the GSM8K-trained model actually generates.
Loads gsm8k_best.pt and prints prompt → raw generation → extracted → gold
for the first 30 test problems. Goal: catch extraction bugs or format issues.
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, '/home/ubuntu/mycelium')

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.compressor_v3 import Compressor
from src.state_conditioned_lora_v3 import StateConditionedLoRA
from src.additive_lora import AdditiveLoRAManager


def parse_final(answer_text):
    m = re.search(r'####\s*(.+)', answer_text)
    if not m:
        return None
    try:
        return float(m.group(1).strip().replace(',', ''))
    except ValueError:
        return None


def extract_pred(gen_text):
    """Current extraction logic from train_gsm8k.py."""
    num_match = re.search(r'-?\d+\.?\d*', gen_text)
    if num_match:
        try:
            pred = float(num_match.group())
            if pred == int(pred):
                pred = int(pred)
            return pred
        except ValueError:
            pass
    return None


def extract_pred_alt(gen_text):
    """Alternative: last number in text (often the final answer)."""
    nums = re.findall(r'-?\d+\.?\d*', gen_text)
    if not nums:
        return None
    try:
        pred = float(nums[-1])
        if pred == int(pred):
            pred = int(pred)
        return pred
    except ValueError:
        return None


def main():
    device = torch.device('cuda')

    model_name = 'unsloth/Llama-3.2-1B'
    transformer = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map='auto',
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    for p in transformer.parameters():
        p.requires_grad = False

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

    ckpt = torch.load('/home/ubuntu/mycelium/checkpoints/gsm8k_best.pt',
                      map_location=device, weights_only=True)
    compressor.load_state_dict(ckpt['compressor'])
    lora.load_state_dict(ckpt['lora'])
    print(f"Loaded gsm8k_best.pt: epoch {ckpt['epoch']}, accuracy {ckpt['accuracy']:.1f}%")

    ds = load_dataset("openai/gsm8k", "main", split='test[:30]')

    state_size, strategy_size = 64, 512
    state_radius = state_size ** 0.5

    correct_first = correct_last = 0
    total = 0

    torch.manual_seed(42)
    with torch.no_grad():
        for i, ex in enumerate(ds):
            q = ex['question']
            gold = parse_final(ex['answer'])
            if gold is None:
                continue
            if gold == int(gold):
                gold = int(gold)

            prompt = f"Problem: {q}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=256)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            state = torch.randn(1, state_size, device=device)
            state = F.normalize(state, dim=-1) * state_radius
            strategy = torch.zeros(1, strategy_size, device=device)

            for pass_num in range(3):
                lora_mods = lora(state, strategy)
                manager = AdditiveLoRAManager(transformer)
                manager.apply(lora_mods)
                try:
                    outputs = transformer(
                        input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                    hs = list(outputs.hidden_states[1:])
                finally:
                    manager.remove()
                state_delta, strategy = compressor(hs, pass_num)
                state = F.normalize(state + state_delta, dim=-1) * state_radius

            lora_mods = lora(state, strategy)
            manager = AdditiveLoRAManager(transformer)
            manager.apply(lora_mods)
            try:
                generated = transformer.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    max_new_tokens=80, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            finally:
                manager.remove()

            prompt_len = input_ids.size(1)
            gen_text = tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True)

            pred_first = extract_pred(gen_text)
            pred_last = extract_pred_alt(gen_text)

            ok_first = pred_first == gold
            ok_last = pred_last == gold
            correct_first += int(ok_first)
            correct_last += int(ok_last)
            total += 1

            print(f"\n=== {i} | gold={gold} | first={pred_first} {'✓' if ok_first else '✗'} | last={pred_last} {'✓' if ok_last else '✗'} ===")
            print(f"Q: {q[:200]}")
            print(f"GEN: {repr(gen_text[:300])}")

    print(f"\n{'=' * 60}")
    print(f"First-number extraction: {correct_first}/{total} = {100*correct_first/total:.1f}%")
    print(f"Last-number extraction:  {correct_last}/{total} = {100*correct_last/total:.1f}%")


if __name__ == '__main__':
    main()
