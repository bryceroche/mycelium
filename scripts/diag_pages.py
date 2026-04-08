"""
Page content diagnostic: do the 86.2% two-step pages actually encode the answer?

Method:
  1. Load two_step_pages_best.pt (86.2%).
  2. Generate N two-step arithmetic problems.
  3. Run thinking forward (2 passes) per problem → last_page (64 floats).
  4. Group problems by gold answer.
  5. Within-group cos sim (same answer) vs between-group cos sim (different).

If same >> different, the answer is encoded in the pages.
If same ≈ different, the 86.2% came from LoRA rewiring the generation path,
not from page content.
"""
import random
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/ubuntu/mycelium')

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.compressor_v3 import Compressor
from src.page_attention_hypernetwork import PageAttentionHypernetwork
from src.additive_lora import AdditiveLoRAManager

CKPT = 'checkpoints/two_step_pages_best.pt'
STRATEGY_SIZE = 512  # original training config of the 86.2% checkpoint
NUM_PROBLEMS = 200
NUM_PASSES = 2


def gen_problems(n, seed=0):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        a = rng.randint(2, 50); b = rng.randint(2, 50); c = rng.randint(2, 50)
        if rng.random() < 0.5:
            op1 = '+'; mid = a + b
        else:
            a = b * rng.randint(2, 10); op1 = '/'; mid = a // b
        if rng.random() < 0.5:
            op2 = '+'; final = mid + c
        else:
            op2 = '-'; final = mid - c
        out.append({'problem': f"({a} {op1} {b}) {op2} {c} =", 'final': final})
    return out


def main():
    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained('unsloth/Llama-3.2-1B')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    transformer = AutoModelForCausalLM.from_pretrained(
        'unsloth/Llama-3.2-1B', torch_dtype=torch.bfloat16, device_map='auto',
    )
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
        state_size=64, strategy_size=STRATEGY_SIZE,
    ).to(device)
    hypernet = PageAttentionHypernetwork(
        d_model=d_model, d_kv=d_kv, page_size=64,
        strategy_size=STRATEGY_SIZE, rank=4, num_layers=num_layers,
    ).to(device)

    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    compressor.load_state_dict(ck['compressor'])
    hypernet.load_state_dict(ck['hypernet'])
    print(f"Loaded {CKPT} (epoch={ck['epoch']} acc={ck['accuracy']}%)")
    compressor.eval(); hypernet.eval()
    page_radius = 64 ** 0.5

    problems = gen_problems(NUM_PROBLEMS, seed=0)

    last_pages = []  # (B, 64)
    finals = []
    BS = 16
    with torch.no_grad():
        for i in range(0, len(problems), BS):
            batch = problems[i:i+BS]
            texts = [p['problem'] for p in batch]
            inputs = tokenizer(texts, return_tensors='pt', padding=True,
                               truncation=True, max_length=128)
            input_ids = inputs['input_ids'].to(device)
            attn = inputs['attention_mask'].to(device)
            bsz = input_ids.size(0)
            state_pages = []
            strategy = torch.zeros(bsz, STRATEGY_SIZE, device=device)
            for pn in range(NUM_PASSES):
                if pn == 0:
                    out = transformer(input_ids=input_ids, attention_mask=attn,
                                      output_hidden_states=True)
                    hs = list(out.hidden_states[1:])
                else:
                    mods = hypernet(state_pages, strategy)
                    mgr = AdditiveLoRAManager(transformer)
                    mgr.apply(mods)
                    try:
                        out = transformer(input_ids=input_ids, attention_mask=attn,
                                          output_hidden_states=True)
                        hs = list(out.hidden_states[1:])
                    finally:
                        mgr.remove()
                delta, strategy = compressor(hs, pn)
                page = F.normalize(delta, dim=-1) * page_radius
                state_pages.append(page)
            last_pages.append(state_pages[-1].float().cpu())
            finals.extend([p['final'] for p in batch])

    last = torch.cat(last_pages, dim=0)  # (N, 64)
    finals_t = torch.tensor(finals)

    # Normalize for cosine
    n = F.normalize(last, dim=-1)
    # Pairwise cosine similarity
    sim = n @ n.T                                     # (N, N)

    # Masks: upper triangle only (no self)
    N = last.size(0)
    ii, jj = torch.triu_indices(N, N, offset=1)
    pair_sim = sim[ii, jj]
    pair_same = (finals_t[ii] == finals_t[jj])

    same_sim = pair_sim[pair_same]
    diff_sim = pair_sim[~pair_same]

    # Per-page variance across samples (sanity — are pages varying at all?)
    page_std_per_dim = last.std(dim=0)  # (64,)

    print("\n=== DIAGNOSTIC RESULTS ===")
    print(f"N problems: {N}  unique answers: {len(torch.unique(finals_t))}")
    print(f"Same-answer pairs:    {int(pair_same.sum())}")
    print(f"Different-answer pairs: {int((~pair_same).sum())}")
    print()
    print(f"Last-page cosine similarity:")
    print(f"  same answer  : mean={same_sim.mean():.4f}  std={same_sim.std():.4f}  n={len(same_sim)}")
    print(f"  diff answer  : mean={diff_sim.mean():.4f}  std={diff_sim.std():.4f}  n={len(diff_sim)}")
    print(f"  delta        : {same_sim.mean() - diff_sim.mean():+.4f}")
    print()
    print(f"Last-page per-dim std across {N} problems:")
    print(f"  mean std: {page_std_per_dim.mean():.4f}   max: {page_std_per_dim.max():.4f}   min: {page_std_per_dim.min():.4f}")
    print(f"  num dims with std<0.01: {int((page_std_per_dim < 0.01).sum())}/64")
    print()

    # A few specific answer groups
    from collections import defaultdict
    by_ans = defaultdict(list)
    for i, f in enumerate(finals):
        by_ans[f].append(i)
    top = sorted(by_ans.items(), key=lambda kv: -len(kv[1]))[:3]
    for ans, idxs in top:
        if len(idxs) < 2: continue
        sub = n[idxs]
        sm = sub @ sub.T
        m = sm[torch.triu_indices(len(idxs), len(idxs), offset=1).unbind()].mean()
        print(f"  answer={ans}  n={len(idxs)}  within-group cos sim={m:.4f}")

    print()
    if same_sim.mean() - diff_sim.mean() > 0.05:
        print("VERDICT: Pages encode the answer (same >> diff).")
    elif same_sim.mean() - diff_sim.mean() > 0.01:
        print("VERDICT: Weak signal — pages partially encode the answer.")
    else:
        print("VERDICT: Pages do NOT encode the answer. 86.2% came from LoRA→generation path.")


if __name__ == '__main__':
    main()
