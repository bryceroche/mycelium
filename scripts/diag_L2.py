"""Quick diagnostic: what does the model generate on L2 word ops?"""
import sys, torch
sys.path.insert(0, '/home/ubuntu/mycelium')

from scripts.train_stepping_stones import L2WordOpsDataset
from scripts.train_two_step_pages import PageThinkingModel
from scripts.train_two_step_contrastive import warm_start
from src.additive_lora import AdditiveLoRAManager


def main():
    device = torch.device('cuda')
    model = PageThinkingModel()
    model.compressor = model.compressor.to(device=device, dtype=torch.bfloat16)
    model.hypernet = model.hypernet.to(device=device, dtype=torch.bfloat16)
    model.probe_head = model.probe_head.to(device)

    ckpt_path = 'checkpoints/stepping_stones_L2_best.pt'
    warm_start(model, ckpt_path)
    model.eval()

    dataset = L2WordOpsDataset(num_samples=500, seed=999)

    correct = 0
    total = 0
    print(f"\n{'='*70}")
    print(f"L2 DIAGNOSTIC — 20 problems, 3 fixed passes")
    print(f"{'='*70}\n")

    with torch.no_grad():
        for i in range(20):
            s = dataset[i]
            problem = s['problem']
            gold = s['final']

            inputs = model.tokenizer(
                [problem], return_tensors='pt', padding=True,
                truncation=True, max_length=128,
            )
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            state_pages = []
            strategy = torch.zeros(1, model.strategy_size, device=device)
            for pass_num in range(3):
                page, strategy = model.thinking_pass(
                    input_ids, attention_mask, state_pages, strategy, pass_num,
                )
                state_pages.append(page)

            lora_mods = model.hypernet(state_pages, strategy, pass_num=3)
            manager = AdditiveLoRAManager(model.transformer)
            manager.apply(lora_mods)
            try:
                generated = model.transformer.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    max_new_tokens=50, do_sample=False,
                    pad_token_id=model.tokenizer.pad_token_id,
                )
            finally:
                manager.remove()

            prompt_len = input_ids[0].size(0)
            gen_ids = generated[0][prompt_len:]
            gen_text = model.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            import re
            m = re.search(r'The answer is (\d+)', gen_text)
            if m:
                pred = int(m.group(1))
            else:
                numbers = re.findall(r'\b(\d+)\b', gen_text)
                pred = int(numbers[-1]) if numbers else None

            mark = "OK" if pred == gold else "XX"
            if pred == gold:
                correct += 1
            total += 1

            print(f"[{mark}] {problem}")
            print(f"     Gold: {gold}  |  Extracted: {pred}")
            print(f"     Generated: '{gen_text}'")
            print()


if __name__ == '__main__':
    main()
