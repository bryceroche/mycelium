"""
Three-page diagnostic on the 83.4% three-step contrastive checkpoint.

Questions:
1. Are pages differentiated across problems? (fixed-point collapse check)
2. Do page 1, 2, 3 encode different things?
3. Are page 1s more similar (all "parse") while page 3s are more varied (different answers)?
4. Do pages form answer-keyed equivalence classes?
"""
import random, sys, torch, torch.nn.functional as F
from collections import defaultdict
sys.path.insert(0, "/home/ubuntu/mycelium")
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.compressor_v3 import Compressor
from src.page_attention_hypernetwork import PageAttentionHypernetwork
from src.additive_lora import AdditiveLoRAManager


def gen_three_step(n, seed=0):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        a = rng.randint(2, 50); b = rng.randint(2, 50)
        c = rng.randint(2, 50); d = rng.randint(2, 50)
        if rng.random() < 0.5:
            op1, v1 = "+", a + b
        else:
            a = b * rng.randint(2, 10); op1, v1 = "/", a // b
        if rng.random() < 0.5:
            op2, v2 = "+", v1 + c
        else:
            op2, v2 = "-", v1 - c
        if rng.random() < 0.5:
            op3, v3 = "+", v2 + d
        else:
            op3, v3 = "-", v2 - d
        out.append({
            "problem": f"(({a} {op1} {b}) {op2} {c}) {op3} {d} =",
            "v1": v1, "v2": v2, "final": v3,
        })
    return out


def main():
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    transformer = AutoModelForCausalLM.from_pretrained(
        "unsloth/Llama-3.2-1B", torch_dtype=torch.bfloat16, device_map="auto")
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
        state_size=64, strategy_size=512).to(device=device, dtype=torch.bfloat16)
    hypernet = PageAttentionHypernetwork(
        d_model=d_model, d_kv=d_kv, page_size=64,
        strategy_size=512, rank=4, num_layers=num_layers).to(device=device, dtype=torch.bfloat16)

    ck = torch.load("checkpoints/three_step_contrastive_best.pt", map_location="cpu", weights_only=False)
    compressor.load_state_dict(ck["compressor"])
    hypernet.load_state_dict(ck["hypernet"])
    print(f"Loaded checkpoint (epoch={ck.get('epoch','?')} acc={ck.get('accuracy','?')}%)")
    compressor.eval(); hypernet.eval()
    page_radius = 64 ** 0.5

    problems = gen_three_step(200, seed=99)

    # Collect all three pages per problem
    all_pages = [[], [], []]  # all_pages[pass_num] = list of (B, 64)
    v1s, v2s, finals = [], [], []
    BS = 16

    with torch.no_grad():
        for i in range(0, len(problems), BS):
            batch = problems[i:i+BS]
            texts = [p["problem"] for p in batch]
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(device)
            attn_mask = inputs["attention_mask"].to(device)
            bsz = input_ids.size(0)
            embed_layer = transformer.model.embed_tokens
            problem_embeds = embed_layer(input_ids)
            state_pages = []
            strategy = torch.zeros(bsz, 512, device=device)
            for pn in range(3):
                if pn == 0:
                    out = transformer(inputs_embeds=problem_embeds, attention_mask=attn_mask, output_hidden_states=True)
                    hs = list(out.hidden_states[1:])
                else:
                    mods = hypernet(state_pages, strategy)
                    mgr = AdditiveLoRAManager(transformer)
                    mgr.apply(mods)
                    try:
                        out = transformer(inputs_embeds=problem_embeds, attention_mask=attn_mask, output_hidden_states=True)
                        hs = list(out.hidden_states[1:])
                    finally:
                        mgr.remove()
                delta, strategy = compressor(hs, pn)
                page = F.normalize(delta, dim=-1) * page_radius
                state_pages.append(page)
                all_pages[pn].append(page.float().cpu())

            v1s.extend([p["v1"] for p in batch])
            v2s.extend([p["v2"] for p in batch])
            finals.extend([p["final"] for p in batch])

    # Stack into (N, 64) per page
    pages = [torch.cat(ap, dim=0) for ap in all_pages]
    v1_t = torch.tensor(v1s)
    v2_t = torch.tensor(v2s)
    finals_t = torch.tensor(finals)
    N = pages[0].size(0)

    print(f"\n{'='*60}")
    print(f"THREE-PAGE DIAGNOSTIC — {N} problems, 3 pages each")
    print(f"{'='*60}")

    # Per-page analysis
    for pn in range(3):
        p = pages[pn]
        n = F.normalize(p, dim=-1)
        sim = n @ n.T
        ii, jj = torch.triu_indices(N, N, offset=1)
        pair_sim = sim[ii, jj]

        # Group by the relevant intermediate value
        if pn == 0:
            group_t = v1_t  # page 1 should encode step 1 result
            group_name = "v1 (step 1 result)"
        elif pn == 1:
            group_t = v2_t  # page 2 should encode step 2 result
            group_name = "v2 (step 2 result)"
        else:
            group_t = finals_t  # page 3 should encode final answer
            group_name = "final answer"

        pair_same = (group_t[ii] == group_t[jj])
        same_sim = pair_sim[pair_same]
        diff_sim = pair_sim[~pair_same]
        page_std = p.std(dim=0)

        print(f"\n--- Page {pn+1} (grouped by {group_name}) ---")
        print(f"  Overall mean cos: {pair_sim.mean():.4f}")
        if len(same_sim) > 0:
            print(f"  Same-group cos:   {same_sim.mean():.4f}  (n={len(same_sim)})")
        print(f"  Diff-group cos:   {diff_sim.mean():.4f}  (n={len(diff_sim)})")
        if len(same_sim) > 0:
            print(f"  Delta:            {same_sim.mean() - diff_sim.mean():+.4f}")
        print(f"  Per-dim std:      mean={page_std.mean():.4f}  dead(<0.01)={int((page_std < 0.01).sum())}/64")

    # Cross-page similarity: are pages 1, 2, 3 different from each other?
    print(f"\n--- Cross-page similarity ---")
    for i in range(3):
        for j in range(i+1, 3):
            ni = F.normalize(pages[i], dim=-1)
            nj = F.normalize(pages[j], dim=-1)
            # Per-problem cross-page cos
            cross = (ni * nj).sum(dim=-1)
            print(f"  Page {i+1} vs Page {j+1}: mean cos={cross.mean():.4f}  std={cross.std():.4f}")

    # Also check: does page 1 correlate with v1, page 2 with v2, page 3 with final?
    # (cross-check: does page 3 correlate with v1? it shouldn't as much)
    print(f"\n--- Cross-check: each page grouped by each intermediate ---")
    labels = [("v1", v1_t), ("v2", v2_t), ("final", finals_t)]
    for pn in range(3):
        n = F.normalize(pages[pn], dim=-1)
        sim = n @ n.T
        ii, jj = torch.triu_indices(N, N, offset=1)
        pair_sim = sim[ii, jj]
        deltas = []
        for lname, lt in labels:
            pair_same = (lt[ii] == lt[jj])
            if pair_same.sum() > 0:
                d = pair_sim[pair_same].mean() - pair_sim[~pair_same].mean()
            else:
                d = float('nan')
            deltas.append(d)
        print(f"  Page {pn+1} delta by: v1={deltas[0]:+.4f}  v2={deltas[1]:+.4f}  final={deltas[2]:+.4f}")

    # Top answer groups for page 3
    print(f"\n--- Page 3 answer groups (top 5) ---")
    n3 = F.normalize(pages[2], dim=-1)
    by_ans = defaultdict(list)
    for i, f in enumerate(finals):
        by_ans[f].append(i)
    top = sorted(by_ans.items(), key=lambda kv: -len(kv[1]))[:5]
    for ans, idxs in top:
        if len(idxs) < 2:
            continue
        sub = n3[idxs]
        sm = sub @ sub.T
        m = sm[torch.triu_indices(len(idxs), len(idxs), offset=1).unbind()].mean()
        print(f"  answer={ans:>4d}  n={len(idxs)}  within-group cos={m:.4f}")

    # Verdict
    print(f"\n{'='*60}")
    n3 = F.normalize(pages[2], dim=-1)
    sim3 = n3 @ n3.T
    ii, jj = torch.triu_indices(N, N, offset=1)
    same3 = (finals_t[ii] == finals_t[jj])
    s3 = sim3[ii, jj]
    delta3 = s3[same3].mean() - s3[~same3].mean() if same3.sum() > 0 else 0
    if delta3 > 0.05:
        print("VERDICT: Pages are DIFFERENTIATED. Architecture is working as designed.")
    elif delta3 > 0.01:
        print("VERDICT: Weak differentiation. Contrastive partially working.")
    else:
        print("VERDICT: Pages are CONSTANT. Fixed-point collapse persists.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
