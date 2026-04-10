"""
Dual LoRA L4: two-step word problems (v22.1).

Problems like "A store has 59 cookies. They sell 27 on Monday and 14 on
Tuesday. How many are left?"

All problems are genuinely two-step — every problem requires two sequential
arithmetic operations with narrative context. This is where verification
should shine because errors compound across two operations.

CoT targets: "A store has 59 cookies. They sell 27 on Monday. 59 - 27 = 32.
They sell 14 on Tuesday. 32 - 14 = 18. The answer is 18."

Dual LoRA, warm start from L3 dual checkpoint (96.0%).
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

from scripts.train_dual_lora import (
    DualLoRAModel, forward_train, evaluate, warm_start_dual,
)
from scripts.train_stepping_stones import extract_answer


NAMES = [
    'Jamie', 'Sarah', 'Mike', 'Emma', 'Alex', 'Lisa', 'Tom', 'Anna',
    'Ben', 'Kate', 'Sam', 'Mia', 'Jack', 'Zoe', 'Noah', 'Lily',
    'Ryan', 'Ella', 'Dan', 'Sophia',
]

PLACES = [
    'a store', 'a bakery', 'a farm', 'a school', 'a library',
    'a garden', 'a shop', 'a market', 'a cafe', 'a toy store',
]

OBJECTS = [
    'cookies', 'apples', 'books', 'flowers', 'cupcakes', 'oranges',
    'pencils', 'stickers', 'balloons', 'muffins', 'cards', 'toys',
    'donuts', 'sandwiches', 'tickets', 'bottles', 'candles', 'stamps',
]

# Verbs for gaining items
GAIN_VERBS = [
    ('received', 'from a supplier'),
    ('baked', 'more'),
    ('bought', 'more'),
    ('found', 'more'),
    ('got', 'as a delivery'),
    ('made', 'more'),
]

# Verbs for losing items
LOSE_VERBS = [
    ('sold', ''),
    ('gave away', ''),
    ('used', ''),
    ('donated', ''),
    ('threw out', ''),
    ('lost', ''),
]

# Time markers for step 1 and step 2
TIME_PAIRS = [
    ('on Monday', 'on Tuesday'),
    ('in the morning', 'in the afternoon'),
    ('on Saturday', 'on Sunday'),
    ('in the first hour', 'in the second hour'),
    ('before lunch', 'after lunch'),
    ('yesterday', 'today'),
    ('in January', 'in February'),
    ('on the first day', 'on the second day'),
]


def _make_sub_sub(rng):
    """Sub then sub: start - a - b. Like the store/cookies example."""
    place = rng.choice(PLACES)
    obj = rng.choice(OBJECTS)
    t1, t2 = rng.choice(TIME_PAIRS)
    v1, _ = rng.choice(LOSE_VERBS)
    v2, _ = rng.choice(LOSE_VERBS)

    start = rng.randint(20, 200)
    a = rng.randint(1, start // 2)
    mid = start - a
    b = rng.randint(1, mid - 1)
    result = mid - b

    problem = (
        f"{place.capitalize()} had {start} {obj}. "
        f"They {v1} {a} {t1} and {v2} {b} {t2}. "
        f"How many {obj} are left?"
    )
    cot = (
        f"{place.capitalize()} had {start} {obj}. "
        f"They {v1} {a} {t1}. {start} - {a} = {mid}. "
        f"They {v2} {b} {t2}. {mid} - {b} = {result}. "
        f"The answer is {result}."
    )
    return problem, cot, result


def _make_add_add(rng):
    """Add then add: start + a + b."""
    name = rng.choice(NAMES)
    obj = rng.choice(OBJECTS)
    t1, t2 = rng.choice(TIME_PAIRS)
    v1, s1 = rng.choice(GAIN_VERBS)
    v2, s2 = rng.choice(GAIN_VERBS)

    start = rng.randint(1, 80)
    a = rng.randint(1, 80)
    mid = start + a
    b = rng.randint(1, 80)
    result = mid + b

    problem = (
        f"{name} had {start} {obj}. "
        f"{name} {v1} {a} {s1} {t1} and {v2} {b} {s2} {t2}. "
        f"How many {obj} does {name} have now?"
    )
    cot = (
        f"{name} had {start} {obj}. "
        f"{name} {v1} {a} {s1} {t1}. {start} + {a} = {mid}. "
        f"{name} {v2} {b} {s2} {t2}. {mid} + {b} = {result}. "
        f"The answer is {result}."
    )
    return problem, cot, result


def _make_sub_add(rng):
    """Sub then add: start - a + b."""
    name = rng.choice(NAMES)
    obj = rng.choice(OBJECTS)
    t1, t2 = rng.choice(TIME_PAIRS)
    v_lose, _ = rng.choice(LOSE_VERBS)
    v_gain, s_gain = rng.choice(GAIN_VERBS)

    start = rng.randint(10, 150)
    a = rng.randint(1, start - 1)
    mid = start - a
    b = rng.randint(1, 100)
    result = mid + b

    problem = (
        f"{name} had {start} {obj}. "
        f"{name} {v_lose} {a} {t1}, then {v_gain} {b} {s_gain} {t2}. "
        f"How many {obj} does {name} have now?"
    )
    cot = (
        f"{name} had {start} {obj}. "
        f"{name} {v_lose} {a} {t1}. {start} - {a} = {mid}. "
        f"{name} {v_gain} {b} {s_gain} {t2}. {mid} + {b} = {result}. "
        f"The answer is {result}."
    )
    return problem, cot, result


def _make_add_sub(rng):
    """Add then sub: start + a - b."""
    name = rng.choice(NAMES)
    obj = rng.choice(OBJECTS)
    t1, t2 = rng.choice(TIME_PAIRS)
    v_gain, s_gain = rng.choice(GAIN_VERBS)
    v_lose, _ = rng.choice(LOSE_VERBS)

    start = rng.randint(1, 100)
    a = rng.randint(1, 100)
    mid = start + a
    b = rng.randint(1, mid - 1)
    result = mid - b

    problem = (
        f"{name} had {start} {obj}. "
        f"{name} {v_gain} {a} {s_gain} {t1}, then {v_lose} {b} {t2}. "
        f"How many {obj} does {name} have now?"
    )
    cot = (
        f"{name} had {start} {obj}. "
        f"{name} {v_gain} {a} {s_gain} {t1}. {start} + {a} = {mid}. "
        f"{name} {v_lose} {b} {t2}. {mid} - {b} = {result}. "
        f"The answer is {result}."
    )
    return problem, cot, result


def _make_person_transfer(rng):
    """Person-to-person: A gives to B, then B gives to C or uses some."""
    names = rng.sample(NAMES, 3)
    obj = rng.choice(OBJECTS)

    start = rng.randint(20, 150)
    a = rng.randint(1, start // 2)
    mid = start - a
    b = rng.randint(1, mid - 1)
    result = mid - b

    problem = (
        f"{names[0]} had {start} {obj}. "
        f"{names[0]} gave {a} to {names[1]} and then gave {b} to {names[2]}. "
        f"How many {obj} does {names[0]} have now?"
    )
    cot = (
        f"{names[0]} had {start} {obj}. "
        f"{names[0]} gave {a} to {names[1]}. {start} - {a} = {mid}. "
        f"{names[0]} gave {b} to {names[2]}. {mid} - {b} = {result}. "
        f"The answer is {result}."
    )
    return problem, cot, result


def _make_group_event(rng):
    """Group event: people arrive and leave."""
    place = rng.choice(PLACES)
    t1, t2 = rng.choice(TIME_PAIRS)

    # Mix: some arrive, some leave, or both arrive, etc.
    pattern = rng.choice(['arrive_arrive', 'arrive_leave', 'leave_leave'])

    start = rng.randint(5, 100)

    if pattern == 'arrive_arrive':
        a = rng.randint(1, 60)
        mid = start + a
        b = rng.randint(1, 60)
        result = mid + b
        problem = (
            f"There were {start} people at {place}. "
            f"{a} more people arrived {t1} and {b} more arrived {t2}. "
            f"How many people are at {place} now?"
        )
        cot = (
            f"There were {start} people at {place}. "
            f"{a} more arrived {t1}. {start} + {a} = {mid}. "
            f"{b} more arrived {t2}. {mid} + {b} = {result}. "
            f"The answer is {result}."
        )
    elif pattern == 'arrive_leave':
        a = rng.randint(1, 80)
        mid = start + a
        b = rng.randint(1, mid - 1)
        result = mid - b
        problem = (
            f"There were {start} people at {place}. "
            f"{a} more people arrived {t1}, but {b} left {t2}. "
            f"How many people are at {place} now?"
        )
        cot = (
            f"There were {start} people at {place}. "
            f"{a} arrived {t1}. {start} + {a} = {mid}. "
            f"{b} left {t2}. {mid} - {b} = {result}. "
            f"The answer is {result}."
        )
    else:  # leave_leave
        a = rng.randint(1, start // 2)
        mid = start - a
        b = rng.randint(1, mid - 1)
        result = mid - b
        problem = (
            f"There were {start} people at {place}. "
            f"{a} people left {t1} and {b} more left {t2}. "
            f"How many people are at {place} now?"
        )
        cot = (
            f"There were {start} people at {place}. "
            f"{a} left {t1}. {start} - {a} = {mid}. "
            f"{b} left {t2}. {mid} - {b} = {result}. "
            f"The answer is {result}."
        )

    return problem, cot, result


GENERATORS = [
    _make_sub_sub,
    _make_add_add,
    _make_sub_add,
    _make_add_sub,
    _make_person_transfer,
    _make_group_event,
]


class L4TwoStepWordDataset(Dataset):
    """
    L4: two-step word problems with narrative context.
    Every problem requires exactly two sequential arithmetic operations.
    Numbers in [1, 200], results in [1, 300].
    CoT targets matching base model's natural completion style.
    """
    def __init__(self, num_samples=20000, seed=42):
        rng = random.Random(seed)
        self.samples = []
        attempts = 0
        while len(self.samples) < num_samples:
            attempts += 1
            if attempts > num_samples * 50:
                raise RuntimeError(
                    f"Could not generate {num_samples} samples after {attempts} attempts"
                )

            gen = rng.choice(GENERATORS)
            problem, cot, result = gen(rng)

            if 1 <= result <= 300:
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
    print("Dual LoRA L4 — Two-Step Word Problems")
    print("=" * 60)
    print(f"num_passes={args.num_passes}  batch={args.batch_size}")
    print(f"lam={args.lam}  lam_conf={args.lam_conf}")
    print(f"Warm: {args.warm}")
    print("=" * 60)

    device = torch.device('cuda')
    model = DualLoRAModel()
    model.compressor = model.compressor.to(device=device, dtype=torch.bfloat16)
    model.hypernet = model.hypernet.to(device=device, dtype=torch.bfloat16)
    model.confidence_head = model.confidence_head.to(device)
    model.probe_head = model.probe_head.to(device)

    if args.warm:
        warm_start_dual(model, args.warm)

    train_dataset = L4TwoStepWordDataset(num_samples=args.num_train, seed=42)
    eval_dataset = L4TwoStepWordDataset(num_samples=500, seed=123)
    print(f"\nDataset: {len(train_dataset)} train, {len(eval_dataset)} eval")
    print(f"\nSample problems:")
    for i in range(5):
        s = train_dataset[i]
        print(f"  Q: {s['problem']}")
        print(f"  A: {s['answer']}")
        print()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW([
        {'params': list(model.compressor.parameters()), 'lr': 5e-5},
        # Forward templates (warm-started, lower LR)
        {'params': (list(model.hypernet.A_forward.parameters())
                    + list(model.hypernet.B_forward.parameters())), 'lr': 5e-4},
        # Verify templates (warm-started from L3 dual, same LR)
        {'params': (list(model.hypernet.A_verify.parameters())
                    + list(model.hypernet.B_verify.parameters())), 'lr': 5e-4},
        # Shared hypernetwork
        {'params': (
            list(model.hypernet.page_project.parameters())
            + [model.hypernet.page_query]
            + list(model.hypernet.page_attn.parameters())
            + list(model.hypernet.page_norm.parameters())
            + list(model.hypernet.combine.parameters())
            + list(model.hypernet.pass_embed.parameters())
        ), 'lr': 5e-4},
        # Confidence head
        {'params': list(model.confidence_head.parameters()), 'lr': 1e-3},
    ])
    trainable = (
        list(model.compressor.parameters())
        + list(model.hypernet.parameters())
        + list(model.confidence_head.parameters())
    )
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")
    print(f"GPU mem after init: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Baseline
    base_acc, _ = evaluate(model, eval_dataset, device, num_passes=args.num_passes)
    print(f"Baseline accuracy (before training): {base_acc:.1f}%\n")

    best = 0.0
    patience_counter = 0
    for epoch in range(args.epochs):
        model.train()
        ep_ans = ep_ctr = ep_conf = ep_cos = ep_blend = ep_confval = 0.0
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
            ans_loss, c_loss, conf_loss, page_cos, blend_mean, conf_mean = forward_train(
                model, problems, answers, finals_t, num_passes=args.num_passes,
            )
            total_loss = ans_loss + args.lam * c_loss + args.lam_conf * conf_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            ep_ans += ans_loss.item()
            ep_ctr += c_loss.item()
            ep_conf += conf_loss.item()
            ep_cos += page_cos.item()
            ep_blend += blend_mean.item()
            ep_confval += conf_mean.item()
            nb += 1

        elapsed = time.time() - t0

        # Eval with fixed passes
        acc_fixed, _ = evaluate(
            model, eval_dataset, device, num_passes=args.num_passes,
        )

        acc = acc_fixed
        if acc > best:
            best = acc
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'compressor': model.compressor.state_dict(),
                'hypernet': model.hypernet.state_dict(),
                'confidence_head': model.confidence_head.state_dict(),
                'accuracy': acc,
                'level': 'L4_dual',
            }, 'checkpoints/dual_lora_L4_best.pt')
            print(f"  -> saved checkpoint (acc={acc:.1f}%)")
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch+1}: ans={ep_ans/nb:.4f} contr={ep_ctr/nb:.4f} "
            f"conf={ep_conf/nb:.4f} page_cos={ep_cos/nb:.4f} "
            f"blend={ep_blend/nb:.3f} conf_val={ep_confval/nb:.3f} | "
            f"Acc={acc:.1f}% best={best:.1f}% base={base_acc:.1f}% "
            f"[{elapsed:.0f}s, {nb/elapsed:.1f} it/s]"
        )
        if patience_counter >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} epochs")
            break

    print(f"\nFinal: {best:.1f}% (baseline {base_acc:.1f}%)")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--warm', type=str, default='checkpoints/dual_lora_L3_best.pt')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=24)
    p.add_argument('--lam', type=float, default=0.05)
    p.add_argument('--lam_conf', type=float, default=0.1)
    p.add_argument('--num_train', type=int, default=20000)
    p.add_argument('--num_passes', type=int, default=3)
    p.add_argument('--patience', type=int, default=3)
    args = p.parse_args()
    train(args)
