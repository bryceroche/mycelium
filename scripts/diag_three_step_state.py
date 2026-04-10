"""Diagnostic: are the 73.6% three-step STATES constant or differentiated?
Uses the old v20.2 architecture (StateConditionedLoRA, not page-attention)."""
import random, sys, torch, torch.nn.functional as F
from collections import defaultdict
sys.path.insert(0, "/home/ubuntu/mycelium")
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.compressor_v3 import Compressor
from src.state_conditioned_lora_v3 import StateConditionedLoRA
from src.additive_lora import AdditiveLoRAManager

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
        state_size=64, strategy_size=512).to(device)
    lora = StateConditionedLoRA(
        d_model=d_model, d_kv=d_kv, state_size=64, strategy_size=512,
        rank=4, num_layers=num_layers).to(device)

    ck = torch.load("checkpoints/three_step_best.pt", map_location="cpu", weights_only=False)
    compressor.load_state_dict(ck["compressor"])
    lora.load_state_dict(ck["lora"])
    print(f"Loaded three_step_best.pt (epoch={ck['epoch']} acc={ck['accuracy']}%)")
    compressor.eval(); lora.eval()
    state_radius = 64 ** 0.5

    rng = random.Random(0)
    problems_data = []
    for _ in range(200):
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
        problems_data.append({"problem": f"(({a} {op1} {b}) {op2} {c}) {op3} {d} =", "final": v3})

    final_states = []
    finals = []
    BS = 16
    with torch.no_grad():
        for i in range(0, 200, BS):
            batch = problems_data[i:i+BS]
            texts = [p["problem"] for p in batch]
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(device)
            attn = inputs["attention_mask"].to(device)
            bsz = input_ids.size(0)
            state = torch.randn(bsz, 64, device=device)
            state = F.normalize(state, dim=-1) * state_radius
            strategy = torch.zeros(bsz, 512, device=device)
            for pn in range(3):
                mods = lora(state, strategy)
                mgr = AdditiveLoRAManager(transformer)
                mgr.apply(mods)
                try:
                    out = transformer(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
                    hs = list(out.hidden_states[1:])
                finally:
                    mgr.remove()
                delta, strategy = compressor(hs, pn)
                state = F.normalize(state + delta, dim=-1) * state_radius
            final_states.append(state.float().cpu())
            finals.extend([p["final"] for p in batch])

    last = torch.cat(final_states, dim=0)
    finals_t = torch.tensor(finals)
    n = F.normalize(last, dim=-1)
    sim = n @ n.T
    N = last.size(0)
    ii, jj = torch.triu_indices(N, N, offset=1)
    pair_sim = sim[ii, jj]
    pair_same = (finals_t[ii] == finals_t[jj])
    same_sim = pair_sim[pair_same]
    diff_sim = pair_sim[~pair_same]
    page_std = last.std(dim=0)

    print()
    print("=== THREE-STEP STATE DIAGNOSTIC (non-contrastive 73.6%) ===")
    print(f"N={N}  unique answers: {len(torch.unique(finals_t))}")
    print(f"Same-answer pairs: {int(pair_same.sum())}  Diff-answer pairs: {int((~pair_same).sum())}")
    print()
    print("Final state cosine similarity:")
    if len(same_sim) > 0:
        print(f"  same answer : mean={same_sim.mean():.4f}  std={same_sim.std():.4f}  n={len(same_sim)}")
    print(f"  diff answer : mean={diff_sim.mean():.4f}  std={diff_sim.std():.4f}  n={len(diff_sim)}")
    if len(same_sim) > 0:
        print(f"  delta       : {same_sim.mean() - diff_sim.mean():+.4f}")
    print()
    print(f"Per-dim std: mean={page_std.mean():.4f}  max={page_std.max():.4f}  min={page_std.min():.4f}")
    print(f"Dead dims (std<0.01): {int((page_std < 0.01).sum())}/64")

    by_ans = defaultdict(list)
    for i, f in enumerate(finals):
        by_ans[f].append(i)
    top = sorted(by_ans.items(), key=lambda kv: -len(kv[1]))[:5]
    print()
    print("Top answer groups:")
    for ans, idxs in top:
        if len(idxs) < 2:
            continue
        sub = n[idxs]
        sm = sub @ sub.T
        m = sm[torch.triu_indices(len(idxs), len(idxs), offset=1).unbind()].mean()
        print(f"  answer={ans:>4d}  n={len(idxs)}  within-group cos={m:.4f}")

    print()
    if len(same_sim) > 0 and same_sim.mean() - diff_sim.mean() < 0.005:
        print("VERDICT: States are CONSTANT (fixed-point collapse). Contrastive IS needed.")
    elif len(same_sim) > 0 and same_sim.mean() - diff_sim.mean() > 0.05:
        print("VERDICT: States are DIFFERENTIATED. Contrastive is UNNECESSARY for three-step.")
    else:
        print("VERDICT: Weak differentiation.")

if __name__ == "__main__":
    main()
