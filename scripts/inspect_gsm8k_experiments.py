"""
Two experiments on the GSM8K checkpoint:
  A) LoRA OFF during generation (thinking passes still apply LoRA)
  B) Few-shot prompting (3 GSM8K examples in prompt) — both with and without LoRA

Run on the same 30 test problems for direct comparison.
"""

import re
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/home/ubuntu/mycelium')

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.compressor_v3 import Compressor
from src.state_conditioned_lora_v3 import StateConditionedLoRA
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


def extract_first(text):
    m = re.search(r'-?\d+\.?\d*', text)
    if not m:
        return None
    try:
        v = float(m.group())
        return int(v) if v == int(v) else v
    except ValueError:
        return None


def extract_last(text):
    nums = re.findall(r'-?\d+\.?\d*', text)
    if not nums:
        return None
    try:
        v = float(nums[-1])
        return int(v) if v == int(v) else v
    except ValueError:
        return None


def extract_after_marker(text):
    """Look for 'The answer is X' pattern."""
    m = re.search(r'[Tt]he answer is\s*\$?(-?\d+\.?\d*)', text)
    if m:
        try:
            v = float(m.group(1))
            return int(v) if v == int(v) else v
        except ValueError:
            pass
    return extract_last(text)


def think(transformer, compressor, lora, input_ids, attention_mask, num_passes=3):
    """Run thinking passes with LoRA, return final state and strategy."""
    bs = input_ids.size(0)
    state = torch.randn(bs, 64, device=input_ids.device)
    state = F.normalize(state, dim=-1) * (64 ** 0.5)
    strategy = torch.zeros(bs, 512, device=input_ids.device)

    for pass_num in range(num_passes):
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
        state = F.normalize(state + state_delta, dim=-1) * (64 ** 0.5)

    return state, strategy


def generate(transformer, tokenizer, lora, input_ids, attention_mask, state, strategy,
             use_lora=True, max_new_tokens=120):
    if use_lora:
        lora_mods = lora(state, strategy)
        manager = AdditiveLoRAManager(transformer)
        manager.apply(lora_mods)
    try:
        gen = transformer.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    finally:
        if use_lora:
            manager.remove()
    return tokenizer.decode(gen[0][input_ids.size(1):], skip_special_tokens=True)


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

    # Counters: (config) -> {first, last, marker}
    results = {
        'A_zero_shot_lora_on':  [0, 0, 0],
        'A_zero_shot_lora_off': [0, 0, 0],
        'B_few_shot_lora_on':   [0, 0, 0],
        'B_few_shot_lora_off':  [0, 0, 0],
        'BASELINE_few_shot_no_thinking_no_lora': [0, 0, 0],
    }
    total = 0
    samples_to_print = []

    torch.manual_seed(42)
    with torch.no_grad():
        for i, ex in enumerate(ds):
            q = ex['question']
            gold = parse_final(ex['answer'])
            if gold is None:
                continue
            if gold == int(gold):
                gold = int(gold)
            total += 1

            # Build prompts
            prompt_zs = f"Problem: {q}\nAnswer:"
            prompt_fs = FEW_SHOT + f"Problem: {q}\nAnswer:"

            for tag, prompt in [('A', prompt_zs), ('B', prompt_fs)]:
                inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024)
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)

                state, strategy = think(transformer, compressor, lora, input_ids, attention_mask)

                for use_lora in (True, False):
                    text = generate(transformer, tokenizer, lora, input_ids, attention_mask,
                                    state, strategy, use_lora=use_lora,
                                    max_new_tokens=120)
                    key = f"{tag}_{'few_shot' if tag=='B' else 'zero_shot'}_lora_{'on' if use_lora else 'off'}"
                    f, l, m = extract_first(text), extract_last(text), extract_after_marker(text)
                    results[key][0] += int(f == gold)
                    results[key][1] += int(l == gold)
                    results[key][2] += int(m == gold)
                    if i < 6 and tag == 'B' and not use_lora:
                        samples_to_print.append((i, gold, text[:300]))

            # Pure baseline: few-shot, no thinking, no LoRA
            inputs = tokenizer(prompt_fs, return_tensors='pt', truncation=True, max_length=1024)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            gen = transformer.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=120, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            text = tokenizer.decode(gen[0][input_ids.size(1):], skip_special_tokens=True)
            f, l, m = extract_first(text), extract_last(text), extract_after_marker(text)
            results['BASELINE_few_shot_no_thinking_no_lora'][0] += int(f == gold)
            results['BASELINE_few_shot_no_thinking_no_lora'][1] += int(l == gold)
            results['BASELINE_few_shot_no_thinking_no_lora'][2] += int(m == gold)

    print(f"\n{'=' * 70}")
    print(f"RESULTS over {total} problems (first / last / 'answer is' extraction)")
    print(f"{'=' * 70}")
    for key, (f, l, m) in results.items():
        print(f"  {key:50s}  first={f}/{total}={100*f/total:4.1f}%  "
              f"last={l}/{total}={100*l/total:4.1f}%  marker={m}/{total}={100*m/total:4.1f}%")

    print(f"\n{'=' * 70}")
    print("SAMPLE GENERATIONS (few-shot, LoRA OFF, with thinking passes)")
    print(f"{'=' * 70}")
    for i, gold, text in samples_to_print:
        print(f"\n--- Problem {i} | gold={gold} ---")
        print(repr(text))


if __name__ == '__main__':
    main()
