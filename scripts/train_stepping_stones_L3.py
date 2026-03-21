"""
Stepping Stones L3: named quantities (v21.5).

Problems like "Jamie had 56 cookies and gave 2 away. How many does Jamie have now?"
CoT targets: "Jamie had 56 cookies. Jamie gave 2 away. 56 - 2 = 54. The answer is 54."

Fixed passes (3) with pass-conditioned hypernetwork + target-cos contrastive.
Warm start from L2 CoT best checkpoint (53.4%).
"""
import argparse
import random
import sys
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, '/home/ubuntu/mycelium')

from scripts.train_two_step_pages import PageThinkingModel
from scripts.train_two_step_contrastive import warm_start
from scripts.train_stepping_stones import forward_train, evaluate, extract_answer
from src.contrastive_page_loss import per_page_contrastive_loss
from src.additive_lora import AdditiveLoRAManager


NAMES = [
    'Jamie', 'Sarah', 'Mike', 'Emma', 'Alex', 'Lisa', 'Tom', 'Anna',
    'Ben', 'Kate', 'Sam', 'Mia', 'Jack', 'Zoe', 'Noah', 'Lily',
    'Ryan', 'Ella', 'Dan', 'Sophia',
]

OBJECTS = [
    'cookies', 'apples', 'marbles', 'stickers', 'pencils', 'books',
    'cards', 'coins', 'shells', 'flowers', 'balloons', 'crayons',
    'rocks', 'stamps', 'buttons', 'beads', 'toy cars', 'ribbons',
]

# Each template: (problem_fmt, cot_fmt, op_fn)
# {name}, {obj}, {a}, {b} are filled in; op_fn(a, b) gives the answer
TEMPLATES_ADD = [
    (
        "{name} had {a} {obj}. {name} found {b} more. How many {obj} does {name} have now?",
        "{name} had {a} {obj}. {name} found {b} more. {a} + {b} = {result}. The answer is {result}.",
        lambda a, b: a + b,
    ),
    (
        "{name} had {a} {obj}. {friend} gave {name} {b} more. How many {obj} does {name} have now?",
        "{name} had {a} {obj}. {friend} gave {name} {b} more. {a} + {b} = {result}. The answer is {result}.",
        lambda a, b: a + b,
    ),
    (
        "{name} collected {a} {obj} in the morning and {b} {obj} in the afternoon. How many {obj} did {name} collect in total?",
        "{name} collected {a} {obj} in the morning and {b} in the afternoon. {a} + {b} = {result}. The answer is {result}.",
        lambda a, b: a + b,
    ),
    (
        "{name} has {a} {obj}. {name} buys {b} more. How many {obj} does {name} have now?",
        "{name} has {a} {obj}. {name} buys {b} more. {a} + {b} = {result}. The answer is {result}.",
        lambda a, b: a + b,
    ),
]

TEMPLATES_SUB = [
    (
        "{name} had {a} {obj} and gave {b} away. How many {obj} does {name} have now?",
        "{name} had {a} {obj}. {name} gave {b} away. {a} - {b} = {result}. The answer is {result}.",
        lambda a, b: a - b,
    ),
    (
        "{name} had {a} {obj}. {name} ate {b} of them. How many {obj} are left?",
        "{name} had {a} {obj}. {name} ate {b} of them. {a} - {b} = {result}. The answer is {result}.",
        lambda a, b: a - b,
    ),
    (
        "{name} had {a} {obj}. {name} lost {b}. How many {obj} does {name} have now?",
        "{name} had {a} {obj}. {name} lost {b}. {a} - {b} = {result}. The answer is {result}.",
        lambda a, b: a - b,
    ),
    (
        "{name} started with {a} {obj} and used {b}. How many {obj} are left?",
        "{name} started with {a} {obj} and used {b}. {a} - {b} = {result}. The answer is {result}.",
        lambda a, b: a - b,
    ),
]

TEMPLATES_TWOSTEP = [
    (
        "{name} had {a} {obj}. {name} gave {b} to {friend} and found {c} more. How many {obj} does {name} have now?",
        "{name} had {a} {obj}. {name} gave {b} away. {a} - {b} = {mid}. {name} found {c} more. {mid} + {c} = {result}. The answer is {result}.",
        lambda a, b, c: a - b + c,
    ),
    (
        "{name} had {a} {obj}. {friend} gave {name} {b} more, then {name} lost {c}. How many {obj} does {name} have now?",
        "{name} had {a} {obj}. {friend} gave {name} {b} more. {a} + {b} = {mid}. {name} lost {c}. {mid} - {c} = {result}. The answer is {result}.",
        lambda a, b, c: a + b - c,
    ),
    (
        "{name} collected {a} {obj}. {name} gave {b} to {friend} and then collected {c} more. How many {obj} does {name} have now?",
        "{name} collected {a} {obj}. {name} gave {b} away. {a} - {b} = {mid}. {name} collected {c} more. {mid} + {c} = {result}. The answer is {result}.",
        lambda a, b, c: a - b + c,
    ),
]


class L3NamedQtyDataset(Dataset):
    """
    L3: named quantities with narrative context.
    Mix of 1-step (add/sub) and 2-step problems.
    Answers constrained to [1, 200].
    CoT targets matching base model's natural completion style.
    """
    def __init__(self, num_samples=20000, seed=42):
        rng = random.Random(seed)
        self.samples = []
        attempts = 0
        while len(self.samples) < num_samples:
            attempts += 1
            if attempts > num_samples * 50:
                break

            name = rng.choice(NAMES)
            friend = rng.choice([n for n in NAMES if n != name])
            obj = rng.choice(OBJECTS)

            # 60% one-step, 40% two-step
            if rng.random() < 0.6:
                # One-step
                if rng.random() < 0.5:
                    tmpl = rng.choice(TEMPLATES_ADD)
                    a = rng.randint(1, 150)
                    b = rng.randint(1, 100)
                    result = tmpl[2](a, b)
                else:
                    tmpl = rng.choice(TEMPLATES_SUB)
                    a = rng.randint(5, 200)
                    b = rng.randint(1, a - 1)
                    result = tmpl[2](a, b)

                if 1 <= result <= 200:
                    problem = tmpl[0].format(
                        name=name, friend=friend, obj=obj, a=a, b=b,
                    )
                    cot = tmpl[1].format(
                        name=name, friend=friend, obj=obj, a=a, b=b, result=result,
                    )
                    self.samples.append({
                        'problem': problem,
                        'answer': cot,
                        'final': result,
                    })
            else:
                # Two-step
                tmpl = rng.choice(TEMPLATES_TWOSTEP)
                a = rng.randint(10, 150)
                b = rng.randint(1, min(a - 1, 80))
                c = rng.randint(1, 80)

                if tmpl[2].__code__.co_varnames[:3] == ('a', 'b', 'c'):
                    result = tmpl[2](a, b, c)
                else:
                    result = tmpl[2](a, b, c)

                # Compute mid for CoT
                # Determine mid based on template pattern
                prob_text = tmpl[0]
                if 'gave' in prob_text and 'found' in prob_text:
                    mid = a - b
                elif 'gave' in prob_text and 'collected' in prob_text:
                    mid = a - b
                else:
                    mid = a + b

                if 1 <= result <= 200 and mid >= 0:
                    problem = tmpl[0].format(
                        name=name, friend=friend, obj=obj, a=a, b=b, c=c,
                    )
                    cot = tmpl[1].format(
                        name=name, friend=friend, obj=obj, a=a, b=b, c=c,
                        mid=mid, result=result,
                    )
                    self.samples.append({
                        'problem': problem,
                        'answer': cot,
                        'final': result,
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def train(args):
    print("=" * 60)
    print("Stepping Stones L3 — named quantities")
    print("=" * 60)
    print(f"target_cos=0.7  within=0.3  lam={args.lam}  num_passes={args.num_passes}")
    print(f"batch={args.batch_size}  train_n={args.num_train}  patience={args.patience}")
    print(f"Warm: {args.warm}")
    print("Pass-conditioned hypernetwork + target-cos contrastive")
    print("=" * 60)

    device = torch.device('cuda')
    model = PageThinkingModel()
    model.compressor = model.compressor.to(device=device, dtype=torch.bfloat16)
    model.hypernet = model.hypernet.to(device=device, dtype=torch.bfloat16)
    model.probe_head = model.probe_head.to(device)

    if args.warm:
        warm_start(model, args.warm)

    train_dataset = L3NamedQtyDataset(num_samples=args.num_train, seed=42)
    eval_dataset = L3NamedQtyDataset(num_samples=500, seed=123)
    print(f"\nSample problems:")
    for i in range(5):
        s = train_dataset[i]
        print(f"  Q: {s['problem']}")
        print(f"  A: {s['answer']}")
        print()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW([
        {'params': list(model.compressor.parameters()), 'lr': 5e-5},
        {'params': list(model.hypernet.A_templates) + list(model.hypernet.B_templates), 'lr': 5e-4},
        {'params': (
            list(model.hypernet.page_project.parameters())
            + [model.hypernet.page_query]
            + list(model.hypernet.page_attn.parameters())
            + list(model.hypernet.page_norm.parameters())
            + list(model.hypernet.combine.parameters())
            + list(model.hypernet.pass_embed.parameters())
        ), 'lr': 5e-4},
    ])
    trainable = list(model.compressor.parameters()) + list(model.hypernet.parameters())
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")
    print(f"GPU mem after init: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Baseline
    base_acc = evaluate(model, eval_dataset, device, num_passes=args.num_passes)
    print(f"Baseline accuracy (before training): {base_acc:.1f}%\n")

    lam = args.lam
    best = 0.0
    patience_counter = 0
    for epoch in range(args.epochs):
        model.train()
        ep_total = ep_ans = ep_ctr = ep_cos = ep_xcross = 0.0
        nb = 0
        t0 = time.time()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            problems = batch['problem']
            answers = batch['answer']
            finals_t = torch.tensor(
                [int(s) for s in batch['final']],
                dtype=torch.long, device=device,
            )
            optimizer.zero_grad()
            ans_loss, c_loss, page_cos, cross_cos = forward_train(
                model, problems, answers, finals_t, num_passes=args.num_passes,
            )
            total_loss = ans_loss + lam * c_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            ep_total += total_loss.item()
            ep_ans += ans_loss.item()
            ep_ctr += c_loss.item()
            ep_cos += page_cos.item()
            ep_xcross += cross_cos.item()
            nb += 1

        elapsed = time.time() - t0
        acc = evaluate(model, eval_dataset, device, num_passes=args.num_passes)
        if acc > best:
            best = acc
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'compressor': model.compressor.state_dict(),
                'hypernet': model.hypernet.state_dict(),
                'accuracy': acc,
                'level': 'L3',
            }, 'checkpoints/stepping_stones_L3_best.pt')
            print(f"  -> saved checkpoint (acc={acc:.1f}%)")
        else:
            patience_counter += 1
        print(
            f"Epoch {epoch+1}: total={ep_total/nb:.4f} ans={ep_ans/nb:.4f} "
            f"contrastive={ep_ctr/nb:.4f} page_cos={ep_cos/nb:.4f} "
            f"p2v3={ep_xcross/nb:.4f} | "
            f"Acc={acc:.1f}% (best={best:.1f}%, base={base_acc:.1f}%) "
            f"[{elapsed:.0f}s, {nb/elapsed:.1f} it/s]"
        )
        if patience_counter >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} epochs")
            break

    print(f"\nFinal: {best:.1f}% (baseline {base_acc:.1f}%)")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--warm', type=str, default='checkpoints/stepping_stones_L2_best.pt')
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lam', type=float, default=0.05)
    p.add_argument('--num_train', type=int, default=20000)
    p.add_argument('--num_passes', type=int, default=3)
    p.add_argument('--patience', type=int, default=3)
    args = p.parse_args()
    train(args)
