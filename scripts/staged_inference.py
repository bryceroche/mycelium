#!/usr/bin/env python3
"""
Staged Batch Inference Pipeline

Each model runs ALL problems before unloading:
  Stage 1: C1-A → scaffolds for all problems
  Stage 2: Canonicalizer → telegrams for all problems
  Stage 3: Energy Landscape → scores for all problems

Problems tagged with integer problem_id throughout.
"""

import torch
import json
import gc
from dataclasses import dataclass, field
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


@dataclass
class Problem:
    problem_id: int
    text: str
    scaffold: List[str] = field(default_factory=list)
    telegrams: List[str] = field(default_factory=list)
    energy: float = 0.0


class StagedInference:
    def __init__(self, device="cuda"):
        self.device = device
        self.problems: Dict[int, Problem] = {}

    def load_problems(self, texts: List[str]):
        """Load problems with integer IDs."""
        self.problems = {
            i: Problem(problem_id=i, text=text)
            for i, text in enumerate(texts)
        }
        print(f"Loaded {len(self.problems)} problems")

    # ─────────────────────────────────────────────────────────
    # Stage 1: C1-A Scaffolds
    # ─────────────────────────────────────────────────────────

    def stage1_c1a_scaffolds(self, model_path: str = None):
        """Run C1-A on all problems → scaffolds."""
        print("\n=== Stage 1: C1-A Scaffolds ===")

        base = "Qwen/Qwen2.5-0.5B"
        tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=torch.float32, trust_remote_code=True
        )
        if model_path:
            model = PeftModel.from_pretrained(model, model_path)
        model.eval().to(self.device)

        C1A_CLASSES = ["SETUP", "SUBSTITUTE", "SIMPLIFY", "SOLVE", "COMPUTE", "THEOREM", "OTHER"]

        for pid, prob in self.problems.items():
            prompt = f"Problem: {prob.text}\nPredict the solution scaffold:\n"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id)

            pred = tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):].strip().upper()

            types = []
            for token in pred.replace(",", " ").replace("[", "").replace("]", "").split():
                if token in C1A_CLASSES:
                    types.append(token)
                    if len(types) >= 15:
                        break
            if not types:
                types = ["SETUP", "COMPUTE", "COMPUTE"]

            prob.scaffold = types

        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()

        print(f"  Generated scaffolds for {len(self.problems)} problems")
        avg_steps = sum(len(p.scaffold) for p in self.problems.values()) / len(self.problems)
        print(f"  Avg scaffold length: {avg_steps:.1f}")

    # ─────────────────────────────────────────────────────────
    # Stage 2: Canonicalizer Telegrams
    # ─────────────────────────────────────────────────────────

    def stage2_canonicalizer(self, model_path: str = "models/canonicalizer_v2"):
        """Run canonicalizer on all problems → telegrams (step-at-a-time)."""
        print("\n=== Stage 2: Canonicalizer ===")

        base = "Qwen/Qwen2.5-0.5B"
        tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        newline_token = tokenizer.encode("\n", add_special_tokens=False)[0]

        model = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=torch.float32, trust_remote_code=True
        )
        model = PeftModel.from_pretrained(model, model_path)
        model.eval().to(self.device)

        for pid, prob in self.problems.items():
            telegrams = []

            for step_i, scaffold_type in enumerate(prob.scaffold):
                prev_str = "\n".join(telegrams) if telegrams else "(none)"
                prompt = f"""Problem: {prob.text}
Structure: {scaffold_type} (step {step_i+1} of {len(prob.scaffold)})
Previous steps:
{prev_str}
Next instruction:"""

                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(self.device)

                with torch.no_grad():
                    out = model.generate(
                        **inputs, max_new_tokens=30, do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=newline_token
                    )

                output = tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):]
                telegram = output.split("\n")[0].strip()
                telegrams.append(telegram)

            prob.telegrams = telegrams

        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()

        print(f"  Generated telegrams for {len(self.problems)} problems")
        total_steps = sum(len(p.telegrams) for p in self.problems.values())
        print(f"  Total telegram steps: {total_steps}")

    # ─────────────────────────────────────────────────────────
    # Stage 3: Energy Landscape Scoring
    # ─────────────────────────────────────────────────────────

    def stage3_energy_scoring(self, model_path: str = "models/energy_landscape_v1"):
        """Score all telegram sequences with energy landscape."""
        print("\n=== Stage 3: Energy Scoring ===")

        import math
        import torch.nn as nn
        import torch.nn.functional as F

        class TelegramTokenizer:
            def __init__(self):
                self.token2id = {}
                self.max_len = 64
            def load(self, path):
                with open(path) as f:
                    d = json.load(f)
                self.token2id = d["token2id"]
                self.max_len = d["max_len"]
            def _tokenize(self, text):
                tokens, cur = [], ""
                for ch in text:
                    if ch in " \t":
                        if cur: tokens.append(cur); cur = ""
                    elif ch in "()[]{}=+*/-^,;":
                        if cur: tokens.append(cur); cur = ""
                        tokens.append(ch)
                    else:
                        cur += ch
                if cur: tokens.append(cur)
                return tokens
            def encode_padded(self, text):
                ids = [self.token2id.get(t, 1) for t in self._tokenize(text)[:self.max_len]]
                return torch.tensor(ids + [0]*(self.max_len - len(ids)), dtype=torch.long)

        class InstructionEncoder(nn.Module):
            def __init__(self, vocab_size=512, embed_dim=64, hidden_dim=128, output_dim=64, max_len=64):
                super().__init__()
                self.tok_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                self.pos_emb = nn.Embedding(max_len, embed_dim)
                self.conv1 = nn.Conv1d(embed_dim, hidden_dim, 3, padding=1)
                self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
                self.proj = nn.Linear(hidden_dim, output_dim)
            def forward(self, x):
                B, L = x.shape
                pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
                h = self.tok_emb(x) + self.pos_emb(pos)
                h = h.permute(0, 2, 1)
                h = F.gelu(self.conv1(h))
                h = F.gelu(self.conv2(h))
                mask = (x != 0).float().unsqueeze(1)
                h = (h * mask).sum(2) / mask.sum(2).clamp(min=1)
                return self.proj(h)

        class NodeEnergy(nn.Module):
            def __init__(self, dim=64, hidden=128):
                super().__init__()
                self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                         nn.Linear(hidden, hidden//2), nn.GELU(),
                                         nn.Linear(hidden//2, 1))
            def forward(self, h): return self.mlp(h)

        class PairEnergy(nn.Module):
            def __init__(self, dim=64, hidden=128):
                super().__init__()
                self.mlp = nn.Sequential(nn.Linear(dim*2, hidden), nn.GELU(),
                                         nn.Linear(hidden, hidden//2), nn.GELU(),
                                         nn.Linear(hidden//2, 1))

        class EnergyLandscape(nn.Module):
            def __init__(self, vocab_size=512, embed_dim=64, hidden_dim=128, output_dim=64, max_len=64):
                super().__init__()
                self.encoder = InstructionEncoder(vocab_size, embed_dim, hidden_dim, output_dim, max_len)
                self.node_energy = NodeEnergy(output_dim, hidden_dim)
                self.pair_energy = PairEnergy(output_dim, hidden_dim)
            def total_energy(self, embeddings):
                n = embeddings.shape[0]
                node_e = self.node_energy(embeddings).sum()
                if n < 2:
                    pair_e = 0
                else:
                    idx_i, idx_j = torch.triu_indices(n, n, offset=1, device=embeddings.device)
                    h_i, h_j = embeddings[idx_i], embeddings[idx_j]
                    fwd = self.pair_energy.mlp(torch.cat([h_i, h_j], -1))
                    bwd = self.pair_energy.mlp(torch.cat([h_j, h_i], -1))
                    pair_e = ((fwd - bwd) / 2).sum()
                return (1.0 / math.sqrt(2 * math.pi * max(n, 1))) * (node_e + pair_e)

        tokenizer = TelegramTokenizer()
        tokenizer.load(f"{model_path}/tokenizer.json")

        with open(f"{model_path}/config.json") as f:
            config = json.load(f)

        model = EnergyLandscape(vocab_size=config["vocab_size"], embed_dim=64, hidden_dim=128, output_dim=64, max_len=64)
        model.load_state_dict(torch.load(f"{model_path}/energy_landscape.pt", weights_only=True))
        model.eval().to(self.device)

        for pid, prob in self.problems.items():
            if not prob.telegrams:
                prob.energy = float("inf")
                continue

            tokens = torch.stack([tokenizer.encode_padded(t) for t in prob.telegrams]).to(self.device)

            with torch.no_grad():
                embeddings = model.encoder(tokens)
                prob.energy = model.total_energy(embeddings).item()

        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()

        energies = [p.energy for p in self.problems.values()]
        print(f"  Scored {len(self.problems)} problems")
        print(f"  Energy range: [{min(energies):.3f}, {max(energies):.3f}]")
        print(f"  Mean energy: {sum(energies)/len(energies):.3f}")

    # ─────────────────────────────────────────────────────────
    # Full Pipeline
    # ─────────────────────────────────────────────────────────

    def run_all_stages(self, c1a_path=None, canon_path="models/canonicalizer_v2",
                       energy_path="models/energy_landscape_v1"):
        """Run complete pipeline."""
        self.stage1_c1a_scaffolds(c1a_path)
        self.stage2_canonicalizer(canon_path)
        self.stage3_energy_scoring(energy_path)

    def get_results(self) -> List[Dict]:
        """Get results sorted by problem_id."""
        return [
            {
                "problem_id": p.problem_id,
                "text": p.text,
                "scaffold": p.scaffold,
                "telegrams": p.telegrams,
                "energy": p.energy,
            }
            for p in sorted(self.problems.values(), key=lambda x: x.problem_id)
        ]

    def print_summary(self, n: int = 5):
        """Print summary of first n problems."""
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)

        for p in sorted(self.problems.values(), key=lambda x: x.problem_id)[:n]:
            print(f"\n[Problem {p.problem_id}] {p.text[:50]}...")
            print(f"  Scaffold ({len(p.scaffold)}): {p.scaffold}")
            print(f"  Telegrams:")
            for t in p.telegrams[:5]:
                print(f"    {t}")
            if len(p.telegrams) > 5:
                print(f"    ... ({len(p.telegrams) - 5} more)")
            print(f"  Energy: {p.energy:.4f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="JSONL with problem texts")
    parser.add_argument("--output", default="inference_results.jsonl")
    parser.add_argument("--n-problems", type=int, default=50)
    args = parser.parse_args()

    if args.input:
        problems = []
        with open(args.input) as f:
            for line in f:
                d = json.loads(line)
                problems.append(d.get("problem_text") or d.get("text"))
        problems = problems[:args.n_problems]
    else:
        problems = [
            "What is 2 + 3 * 4?",
            "If x = 5, what is 2x + 3?",
            "If x^2 + y^2 = 90 and xy = 27, find x + y.",
            "Solve for x: 2x^2 - 5x + 2 = 0",
            "Find the area of a triangle with sides 3, 4, and 5.",
        ]

    pipeline = StagedInference(device="cuda")
    pipeline.load_problems(problems)
    pipeline.run_all_stages()
    pipeline.print_summary()

    results = pipeline.get_results()
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
