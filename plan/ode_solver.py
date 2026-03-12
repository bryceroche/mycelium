"""
ODE Solver for Mycelium v7.

The hiker navigating downhill on the energy landscape.

Takes rough telegram embeddings from the canonicalizer,
follows the energy gradient downhill via adaptive Runge-Kutta,
and produces refined embeddings that decode to precise SymPy-parseable telegrams.

Dynamics:   dh/dt = -∇E(h)  (gradient descent on learned energy)
Solver:     dopri5 (adaptive Runge-Kutta, via torchdiffeq)
Bounds:     tanh * 0.1 (prevents explosion)
π-norm:     at input, state, and energy levels

Integration time from C1-B belief propagation depth:
    1-2 steps  → t_max=1.0 (simple, fast)
    3-4 steps  → t_max=2.0 (medium)
    5+ steps   → t_max=3.0 (complex)

Usage:
    python ode_solver.py \
        --energy-model models/energy_landscape_v1 \
        --test  # runs sanity check on a few sequences
"""

import argparse
import json
import math
import os
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────
# Energy gradient dynamics
# ─────────────────────────────────────────────────────────────

class EnergyGradientDynamics(nn.Module):
    """
    dh/dt = -∇E(h) with tanh * 0.1 bounds.

    The ODE right-hand side. Given current state h (a sequence of
    instruction embeddings), compute the energy gradient and return
    the bounded update direction.
    """

    def __init__(self, energy_model, bound_scale: float = 0.1):
        super().__init__()
        self.energy_model = energy_model
        self.bound_scale = bound_scale

    def forward(self, t, h):
        """
        t: current integration time (scalar, unused but required by ODE interface)
        h: (n_steps, repr_dim) — current instruction embeddings

        returns: dh/dt = -tanh(∇E) * bound_scale
        """
        # Enable gradient computation for h
        h_grad = h.detach().requires_grad_(True)

        # Compute total energy
        energy = self.energy_model.total_energy(h_grad)

        # Compute gradient of energy w.r.t. embeddings
        grad = torch.autograd.grad(
            energy, h_grad, create_graph=False, retain_graph=False
        )[0]

        # Bounded update: tanh prevents explosion, scale controls step size
        dh_dt = -torch.tanh(grad) * self.bound_scale

        return dh_dt


# ─────────────────────────────────────────────────────────────
# ODE Solver
# ─────────────────────────────────────────────────────────────

class ODERefinement:
    """
    Refine rough telegram embeddings by descending the energy landscape.

    Uses adaptive Runge-Kutta (dopri5) for efficient integration.
    Falls back to fixed-step Euler if torchdiffeq is unavailable.
    """

    def __init__(self, energy_model, bound_scale: float = 0.1,
                 rtol: float = 1e-3, atol: float = 1e-4,
                 max_steps: int = 100):
        self.energy_model = energy_model
        self.dynamics = EnergyGradientDynamics(energy_model, bound_scale)
        self.rtol = rtol
        self.atol = atol
        self.max_steps = max_steps

        # Try to use torchdiffeq for adaptive stepping
        try:
            from torchdiffeq import odeint
            self.odeint = odeint
            self.solver_type = "dopri5"
            print("ODE solver: dopri5 (torchdiffeq)")
        except ImportError:
            self.odeint = None
            self.solver_type = "euler"
            print("ODE solver: fixed-step Euler (install torchdiffeq for dopri5)")

    def integration_time(self, n_steps: int, bp_depth: Optional[float] = None) -> float:
        """
        Determine integration time from problem complexity.

        If C1-B provides bp_depth, use it directly.
        Otherwise estimate from step count.
        """
        if bp_depth is not None:
            return bp_depth

        if n_steps <= 2:
            return 1.0
        elif n_steps <= 4:
            return 2.0
        else:
            return 3.0

    def pi_normalize(self, h: torch.Tensor) -> torch.Tensor:
        """
        π-normalize embeddings for scale invariance.
        Projects onto hypersphere of radius sqrt(2π * dim).
        """
        dim = h.shape[-1]
        target_norm = math.sqrt(2 * math.pi * dim)
        current_norm = h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return h * (target_norm / current_norm)

    def refine_dopri5(self, h0: torch.Tensor, t_max: float) -> Tuple[torch.Tensor, dict]:
        """Refine using adaptive Runge-Kutta (dopri5)."""
        t_span = torch.tensor([0.0, t_max], device=h0.device)

        # Flatten for ODE solver (it expects a single tensor)
        shape = h0.shape
        h0_flat = h0.reshape(-1)

        def dynamics_flat(t, h_flat):
            h = h_flat.reshape(shape)
            dh = self.dynamics(t, h)
            return dh.reshape(-1)

        trajectory = self.odeint(
            dynamics_flat, h0_flat, t_span,
            method='dopri5',
            rtol=self.rtol,
            atol=self.atol,
            options={'max_num_steps': self.max_steps},
        )

        h_refined = trajectory[-1].reshape(shape)

        info = {
            "solver": "dopri5",
            "t_max": t_max,
        }

        return h_refined, info

    def refine_euler(self, h0: torch.Tensor, t_max: float,
                     n_euler_steps: int = 50) -> Tuple[torch.Tensor, dict]:
        """Fallback: fixed-step Euler integration."""
        dt = t_max / n_euler_steps
        h = h0.clone()

        energies = []

        for step in range(n_euler_steps):
            t = torch.tensor(step * dt)

            # Compute energy before update (for monitoring)
            with torch.no_grad():
                e = self.energy_model.total_energy(h).item()
                energies.append(e)

            # Euler step
            dh = self.dynamics(t, h)
            h = h + dt * dh

            # π-normalize after each step to stay on manifold
            h = self.pi_normalize(h)

        # Final energy
        with torch.no_grad():
            final_e = self.energy_model.total_energy(h).item()
            energies.append(final_e)

        info = {
            "solver": "euler",
            "t_max": t_max,
            "n_steps": n_euler_steps,
            "energy_trajectory": energies,
            "energy_drop": energies[0] - energies[-1],
            "converged": energies[-1] < energies[0],
        }

        return h, info

    def refine(self, h0: torch.Tensor, n_steps: int,
               bp_depth: Optional[float] = None) -> Tuple[torch.Tensor, dict]:
        """
        Main entry point: refine a sequence of instruction embeddings.

        h0: (n_steps, repr_dim) — rough instruction embeddings
        n_steps: number of instructions (for integration time)
        bp_depth: optional C1-B belief propagation depth

        Returns: (h_refined, info_dict)
        """
        # π-normalize input
        h0 = self.pi_normalize(h0)

        # Determine integration time
        t_max = self.integration_time(n_steps, bp_depth)

        # Compute initial energy
        with torch.no_grad():
            initial_energy = self.energy_model.total_energy(h0).item()

        # Run ODE
        if self.odeint is not None:
            try:
                h_refined, info = self.refine_dopri5(h0, t_max)
            except Exception as e:
                # dopri5 can fail on stiff dynamics — fall back to Euler
                print(f"  dopri5 failed ({e}), falling back to Euler")
                h_refined, info = self.refine_euler(h0, t_max)
        else:
            h_refined, info = self.refine_euler(h0, t_max)

        # π-normalize output
        h_refined = self.pi_normalize(h_refined)

        # Compute final energy
        with torch.no_grad():
            final_energy = self.energy_model.total_energy(h_refined).item()

        info["initial_energy"] = initial_energy
        info["final_energy"] = final_energy
        info["energy_drop"] = initial_energy - final_energy
        info["converged"] = final_energy < initial_energy

        return h_refined, info


# ─────────────────────────────────────────────────────────────
# Embedding ↔ Telegram decoder
# ─────────────────────────────────────────────────────────────

class TelegramDecoder:
    """
    Decode refined embeddings back to telegram strings.

    Two strategies:
    1. Nearest-neighbor: find closest known instruction in a codebook
    2. Perturbation: apply small modifications to the original telegram
       guided by the embedding shift direction

    For MVP, we use nearest-neighbor against the training corpus.
    """

    def __init__(self, energy_model, tokenizer):
        self.energy_model = energy_model
        self.tokenizer = tokenizer
        self.codebook = {}  # telegram_str → embedding
        self.codebook_list = []  # [(telegram_str, embedding)]

    def build_codebook(self, telegrams: List[str], batch_size: int = 256):
        """Encode all known valid telegrams into the codebook."""
        device = next(self.energy_model.parameters()).device

        for i in range(0, len(telegrams), batch_size):
            batch = telegrams[i:i + batch_size]
            token_ids = torch.stack([
                self.tokenizer.encode_padded(t) for t in batch
            ]).to(device)

            with torch.no_grad():
                embeddings = self.energy_model.encoder(token_ids)

            for telegram, emb in zip(batch, embeddings):
                self.codebook[telegram] = emb.cpu()
                self.codebook_list.append((telegram, emb.cpu()))

        print(f"Codebook: {len(self.codebook_list)} entries")

    def decode_nearest(self, h_refined: torch.Tensor, top_k: int = 5) -> List[Dict]:
        """
        Find nearest codebook entries to each refined embedding.

        h_refined: (n_steps, repr_dim)
        Returns: list of {telegram, distance, rank} per step
        """
        if not self.codebook_list:
            raise ValueError("Codebook is empty — call build_codebook first")

        # Stack all codebook embeddings
        codebook_embs = torch.stack([emb for _, emb in self.codebook_list])
        codebook_strs = [s for s, _ in self.codebook_list]

        results = []
        for i in range(h_refined.shape[0]):
            h = h_refined[i].cpu().unsqueeze(0)  # (1, dim)

            # Cosine similarity
            sims = F.cosine_similarity(h, codebook_embs, dim=-1)  # (codebook_size,)
            topk_vals, topk_idx = sims.topk(top_k)

            candidates = []
            for rank, (sim, idx) in enumerate(zip(topk_vals, topk_idx)):
                candidates.append({
                    "telegram": codebook_strs[idx.item()],
                    "similarity": sim.item(),
                    "rank": rank,
                })

            results.append(candidates)

        return results

    def decode_perturbation(self, original_telegrams: List[str],
                            h_original: torch.Tensor,
                            h_refined: torch.Tensor,
                            verb_locked: bool = True) -> List[str]:
        """
        Decode by perturbing original telegrams based on embedding shift.

        The shift direction h_refined - h_original tells us what changed.
        We find codebook entries near the refined embedding that share
        the same verb as the original (verb is locked by scaffold).

        Falls back to nearest-neighbor if no same-verb match is found.
        """
        results = []

        for i, orig in enumerate(original_telegrams):
            orig_verb = orig.split()[0] if orig.split() else ""

            h = h_refined[i].cpu().unsqueeze(0)
            codebook_embs = torch.stack([emb for _, emb in self.codebook_list])
            sims = F.cosine_similarity(h, codebook_embs, dim=-1)

            # Sort by similarity
            sorted_idx = sims.argsort(descending=True)

            # Find best match with same verb
            found = False
            for idx in sorted_idx[:50]:  # search top 50
                candidate = self.codebook_list[idx.item()][0]
                cand_verb = candidate.split()[0] if candidate.split() else ""

                if not verb_locked or cand_verb == orig_verb:
                    results.append(candidate)
                    found = True
                    break

            if not found:
                # No same-verb match — keep original
                results.append(orig)

        return results


# ─────────────────────────────────────────────────────────────
# Full refinement pipeline
# ─────────────────────────────────────────────────────────────

class TelegramRefiner:
    """
    Complete rough → precise refinement pipeline.

    1. Encode rough telegrams to embeddings
    2. ODE descends energy landscape
    3. Decode refined embeddings to precise telegrams
    4. Parse with SymPy oracle
    """

    def __init__(self, energy_model, tokenizer, decoder: TelegramDecoder,
                 bound_scale: float = 0.1):
        self.energy_model = energy_model
        self.tokenizer = tokenizer
        self.decoder = decoder
        self.ode = ODERefinement(energy_model, bound_scale=bound_scale)
        self.device = next(energy_model.parameters()).device

    def encode_telegrams(self, telegrams: List[str]) -> torch.Tensor:
        """Encode telegram strings to embeddings."""
        token_ids = torch.stack([
            self.tokenizer.encode_padded(t) for t in telegrams
        ]).to(self.device)

        with torch.no_grad():
            embeddings = self.energy_model.encoder(token_ids)

        return embeddings

    def refine(self, telegrams: List[str],
               bp_depth: Optional[float] = None,
               decode_strategy: str = "perturbation") -> Dict:
        """
        Full refinement: rough telegrams → refined telegrams.

        Returns dict with:
            - refined_telegrams: list of refined strings
            - initial_energy: energy before ODE
            - final_energy: energy after ODE
            - energy_drop: improvement
            - ode_info: solver details
        """
        n_steps = len(telegrams)

        # 1. Encode
        h0 = self.encode_telegrams(telegrams)

        # 2. ODE refinement
        h_refined, ode_info = self.ode.refine(h0, n_steps, bp_depth)

        # 3. Decode
        if decode_strategy == "nearest":
            candidates = self.decoder.decode_nearest(h_refined, top_k=3)
            refined_telegrams = [c[0]["telegram"] for c in candidates]
        elif decode_strategy == "perturbation":
            refined_telegrams = self.decoder.decode_perturbation(
                telegrams, h0, h_refined, verb_locked=True
            )
        else:
            raise ValueError(f"Unknown decode strategy: {decode_strategy}")

        return {
            "original_telegrams": telegrams,
            "refined_telegrams": refined_telegrams,
            "initial_energy": ode_info["initial_energy"],
            "final_energy": ode_info["final_energy"],
            "energy_drop": ode_info["energy_drop"],
            "converged": ode_info["converged"],
            "ode_info": ode_info,
        }


# ─────────────────────────────────────────────────────────────
# Test / sanity check
# ─────────────────────────────────────────────────────────────

def run_sanity_check(energy_model_path: str):
    """Test ODE refinement on a few known sequences."""
    from train_energy_landscape import EnergyLandscape, TelegramTokenizer

    # Load model
    config_path = os.path.join(energy_model_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    tokenizer = TelegramTokenizer()
    tokenizer.load(os.path.join(energy_model_path, "tokenizer.json"))

    model = EnergyLandscape(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        hidden_dim=config["hidden_dim"],
        repr_dim=config["repr_dim"],
        max_len=config["max_len"],
        pair_weight=config.get("pair_weight", 1.0),
    )
    model.load_state_dict(
        torch.load(os.path.join(energy_model_path, "energy_landscape.pt"),
                    map_location="cpu")
    )
    model.eval()

    device = "cpu"
    model = model.to(device)

    # Build a small codebook from test sequences
    test_telegrams = [
        "GIVEN x^2+y^2=90",
        "GIVEN xy=27",
        "EXPAND (x+y)^2",
        "SUBS _prev x^2+y^2 90",
        "EVAL _prev",
        "EVAL 90+54",
        "EVAL 144",
        "SOLVE x^2-5x+6 x",
        "SIMPLIFY _prev",
        "ANSWER _prev",
        "GIVEN a=1 r=1/2",
        "APPLY geometric_sum a r",
        "SUBS _prev a 1 r 1/2",
    ]

    decoder = TelegramDecoder(model, tokenizer)
    decoder.build_codebook(test_telegrams)

    refiner = TelegramRefiner(model, tokenizer, decoder, bound_scale=0.1)

    # Test 1: Correct sequence — energy should be low, minimal change
    print("\n═══ Test 1: Correct sequence (should stay similar) ═══")
    correct = ["GIVEN x^2+y^2=90", "EXPAND (x+y)^2", "SUBS _prev x^2+y^2 90", "EVAL _prev"]
    result = refiner.refine(correct)
    print(f"  Initial energy: {result['initial_energy']:.4f}")
    print(f"  Final energy:   {result['final_energy']:.4f}")
    print(f"  Energy drop:    {result['energy_drop']:.4f}")
    print(f"  Converged:      {result['converged']}")
    for orig, refined in zip(result['original_telegrams'], result['refined_telegrams']):
        changed = " ← CHANGED" if orig != refined else ""
        print(f"    {orig:35s} → {refined}{changed}")

    # Test 2: Shuffled sequence — energy should drop significantly
    print("\n═══ Test 2: Shuffled sequence (should change toward correct) ═══")
    shuffled = ["EVAL _prev", "SUBS _prev x^2+y^2 90", "GIVEN x^2+y^2=90", "EXPAND (x+y)^2"]
    result = refiner.refine(shuffled)
    print(f"  Initial energy: {result['initial_energy']:.4f}")
    print(f"  Final energy:   {result['final_energy']:.4f}")
    print(f"  Energy drop:    {result['energy_drop']:.4f}")
    print(f"  Converged:      {result['converged']}")
    for orig, refined in zip(result['original_telegrams'], result['refined_telegrams']):
        changed = " ← CHANGED" if orig != refined else ""
        print(f"    {orig:35s} → {refined}{changed}")

    # Test 3: Wrong verb — should find better match
    print("\n═══ Test 3: Wrong verb (SOLVE instead of EXPAND) ═══")
    wrong_verb = ["GIVEN x^2+y^2=90", "SOLVE (x+y)^2", "SUBS _prev x^2+y^2 90", "EVAL _prev"]
    result = refiner.refine(wrong_verb, decode_strategy="nearest")
    print(f"  Initial energy: {result['initial_energy']:.4f}")
    print(f"  Final energy:   {result['final_energy']:.4f}")
    print(f"  Energy drop:    {result['energy_drop']:.4f}")
    for orig, refined in zip(result['original_telegrams'], result['refined_telegrams']):
        changed = " ← CHANGED" if orig != refined else ""
        print(f"    {orig:35s} → {refined}{changed}")

    # Energy comparison summary
    print("\n═══ Energy comparison ═══")
    sequences = {
        "correct": ["GIVEN x^2+y^2=90", "EXPAND (x+y)^2", "SUBS _prev x^2+y^2 90", "EVAL _prev"],
        "shuffled": ["EVAL _prev", "SUBS _prev x^2+y^2 90", "GIVEN x^2+y^2=90", "EXPAND (x+y)^2"],
        "wrong_verb": ["GIVEN x^2+y^2=90", "SOLVE (x+y)^2", "SUBS _prev x^2+y^2 90", "EVAL _prev"],
        "reversed": ["EVAL _prev", "SUBS _prev x^2+y^2 90", "EXPAND (x+y)^2", "GIVEN x^2+y^2=90"],
    }

    for name, seq in sequences.items():
        h = refiner.encode_telegrams(seq)
        h = refiner.ode.pi_normalize(h)
        with torch.no_grad():
            e = model.total_energy(h).item()
        print(f"  {name:15s}: {e:.4f}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ODE refinement on energy landscape")
    parser.add_argument("--energy-model", required=True,
                        help="Path to energy landscape model directory")
    parser.add_argument("--test", action="store_true",
                        help="Run sanity check")
    parser.add_argument("--bound-scale", type=float, default=0.1)

    args = parser.parse_args()

    if args.test:
        run_sanity_check(args.energy_model)


if __name__ == "__main__":
    main()
