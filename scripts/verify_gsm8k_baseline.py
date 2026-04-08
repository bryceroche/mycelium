"""
Verify the GSM8K base model baseline on 500 problems with proper few-shot prompting.
30 problems was too small. Find the real floor we need to beat.
"""

import re
import torch
import sys
sys.path.insert(0, '/home/ubuntu/mycelium')

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def extract_marker(text):
    """Look for 'The answer is X' pattern, falls back to last number."""
    # Stop at next "Problem:" if model continues generating examples
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
    if nums:
        try:
            v = float(nums[-1].replace(',', ''))
            return int(v) if v == int(v) else v
        except ValueError:
            pass
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
    transformer.eval()

    ds = load_dataset("openai/gsm8k", "main", split='test[:500]')
    print(f"Loaded {len(ds)} test problems")

    correct = total = 0
    batch_size = 8
    prompts = []
    golds = []

    for ex in ds:
        gold = parse_final(ex['answer'])
        if gold is None:
            continue
        if gold == int(gold):
            gold = int(gold)
        prompts.append(FEW_SHOT + f"Problem: {ex['question']}\nAnswer:")
        golds.append(gold)

    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_golds = golds[i:i + batch_size]

            inputs = tokenizer(
                batch_prompts, return_tensors='pt', padding=True,
                truncation=True, max_length=1024,
            )
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            gen = transformer.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=120, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

            for j in range(len(batch_prompts)):
                prompt_len = (input_ids[j] != tokenizer.pad_token_id).sum().item()
                # Find where this prompt ends in the generation (account for left padding)
                # Easier: decode only the new tokens after the full input length
                full_len = input_ids.size(1)
                gen_text = tokenizer.decode(gen[j][full_len:], skip_special_tokens=True)
                pred = extract_marker(gen_text)
                if pred == batch_golds[j]:
                    correct += 1
                total += 1

            if (i // batch_size) % 5 == 0:
                print(f"  {total}/{len(prompts)}: {100 * correct / total:.1f}%")

    print(f"\n{'=' * 60}")
    print(f"BASELINE: few-shot, no thinking, no LoRA")
    print(f"Accuracy: {correct}/{total} = {100 * correct / total:.1f}%")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
