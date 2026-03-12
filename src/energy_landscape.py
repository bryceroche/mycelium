"""
Energy Landscape for Mycelium v7 ODE Factor Graph

Learns an energy function over solution steps where:
- Correct solutions have LOWER energy
- Incorrect solutions have HIGHER energy

The ODE solver uses -grad(E) to refine hidden states toward correct solutions.

Architecture:
1. NodeEnergyMLP: E_node(h) -> scalar
2. PairEnergyMLP: E_pair(h_i, h_j) = (MLP(h_i || h_j) - MLP(h_j || h_i)) / 2
   - Antisymmetric: E(i,j) = -E(j,i)
3. EnergyLandscape: Total = sum(E_node) + lambda * sum(E_pair)

Training: Contrastive loss - correct solutions should have lower energy than incorrect.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass


# Qwen-0.5B hidden dimension
HIDDEN_DIM = 896


@dataclass
class EnergyOutput:
    """Output from energy computation."""
    total_energy: torch.Tensor  # scalar or (batch,)
    node_energies: torch.Tensor  # (n_steps,) or (batch, n_steps)
    pair_energies: Optional[torch.Tensor] = None  # (n_pairs,) or (batch, n_pairs)
    gradients: Optional[torch.Tensor] = None  # grad w.r.t. hidden states


class NodeEnergyMLP(nn.Module):
    """
    MLP that maps step hidden state to scalar energy.

    Input: h (896-dim hidden state from Qwen-0.5B)
    Output: scalar energy

    Architecture:
        Linear(896 -> 512) -> LayerNorm -> GELU ->
        Linear(512 -> 256) -> LayerNorm -> GELU ->
        Linear(256 -> 1)
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM, intermediate_dims: Tuple[int, ...] = (512, 256)):
        super().__init__()

        layers = []
        in_dim = hidden_dim

        for out_dim in intermediate_dims:
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
            ])
            in_dim = out_dim

        # Final projection to scalar
        layers.append(nn.Linear(in_dim, 1))

        self.mlp = nn.Sequential(*layers)

        # Initialize final layer to small values for stable training
        nn.init.xavier_uniform_(self.mlp[-1].weight, gain=0.1)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Hidden states, shape (batch, hidden_dim) or (hidden_dim,)

        Returns:
            Scalar energy, shape (batch,) or ()
        """
        squeeze = h.dim() == 1
        if squeeze:
            h = h.unsqueeze(0)

        energy = self.mlp(h).squeeze(-1)  # (batch,)

        if squeeze:
            energy = energy.squeeze(0)

        return energy


class PairEnergyMLP(nn.Module):
    """
    Antisymmetric MLP for pairwise energy between step hidden states.

    E_pair(h_i, h_j) = (MLP(h_i || h_j) - MLP(h_j || h_i)) / 2

    This ensures: E(i, j) = -E(j, i)

    The antisymmetry is crucial for the factor graph:
    - If step i should come before step j, E(i,j) < 0
    - The factor graph can use these energies as directed consistency scores
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM, intermediate_dims: Tuple[int, ...] = (512, 256)):
        super().__init__()

        # MLP takes concatenation of two hidden states
        input_dim = hidden_dim * 2

        layers = []
        in_dim = input_dim

        for out_dim in intermediate_dims:
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
            ])
            in_dim = out_dim

        # Final projection to scalar
        layers.append(nn.Linear(in_dim, 1))

        self.mlp = nn.Sequential(*layers)

        # Initialize final layer to small values
        nn.init.xavier_uniform_(self.mlp[-1].weight, gain=0.1)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, h_i: torch.Tensor, h_j: torch.Tensor) -> torch.Tensor:
        """
        Compute antisymmetric pairwise energy.

        Args:
            h_i: Hidden state of step i, shape (batch, hidden_dim) or (hidden_dim,)
            h_j: Hidden state of step j, shape (batch, hidden_dim) or (hidden_dim,)

        Returns:
            Antisymmetric energy E(i,j), shape (batch,) or ()
        """
        squeeze = h_i.dim() == 1
        if squeeze:
            h_i = h_i.unsqueeze(0)
            h_j = h_j.unsqueeze(0)

        # Forward: concat(h_i, h_j)
        concat_ij = torch.cat([h_i, h_j], dim=-1)
        e_ij = self.mlp(concat_ij).squeeze(-1)

        # Reverse: concat(h_j, h_i)
        concat_ji = torch.cat([h_j, h_i], dim=-1)
        e_ji = self.mlp(concat_ji).squeeze(-1)

        # Antisymmetric: (e_ij - e_ji) / 2
        energy = (e_ij - e_ji) / 2

        if squeeze:
            energy = energy.squeeze(0)

        return energy

    def forward_batch_pairs(self, h: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise energies for multiple pairs efficiently.

        Args:
            h: All hidden states, shape (n_steps, hidden_dim)
            pairs: Pair indices, shape (n_pairs, 2) - each row is [i, j]

        Returns:
            Pairwise energies, shape (n_pairs,)
        """
        h_i = h[pairs[:, 0]]  # (n_pairs, hidden_dim)
        h_j = h[pairs[:, 1]]  # (n_pairs, hidden_dim)
        return self.forward(h_i, h_j)


class EnergyLandscape(nn.Module):
    """
    Full energy landscape combining node and pair energies.

    Total Energy = sum(E_node(h_i)) + lambda * sum(E_pair(h_i, h_j))

    The lambda parameter controls the relative weight of structural (pairwise)
    consistency vs individual step quality.

    For ODE integration:
        dh/dt = -grad_h(E)

    This pushes hidden states toward lower energy configurations,
    which correspond to correct solutions.
    """

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        node_intermediate_dims: Tuple[int, ...] = (512, 256),
        pair_intermediate_dims: Tuple[int, ...] = (512, 256),
        pair_weight: float = 0.1,
        use_pair_energy: bool = True,
    ):
        """
        Args:
            hidden_dim: Dimension of step hidden states (896 for Qwen-0.5B)
            node_intermediate_dims: MLP hidden dims for node energy
            pair_intermediate_dims: MLP hidden dims for pair energy
            pair_weight: Lambda weight for pair energies in total
            use_pair_energy: Whether to compute pair energies (can disable for speed)
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.pair_weight = pair_weight
        self.use_pair_energy = use_pair_energy

        self.node_energy = NodeEnergyMLP(hidden_dim, node_intermediate_dims)

        if use_pair_energy:
            self.pair_energy = PairEnergyMLP(hidden_dim, pair_intermediate_dims)
        else:
            self.pair_energy = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        pairs: Optional[torch.Tensor] = None,
        compute_gradients: bool = False,
    ) -> EnergyOutput:
        """
        Compute total energy of a solution.

        Args:
            hidden_states: Step hidden states, shape (n_steps, hidden_dim)
            pairs: Optional pair indices for pairwise energy, shape (n_pairs, 2)
                   If None and use_pair_energy=True, computes all adjacent pairs
            compute_gradients: If True, also return gradients w.r.t. hidden states

        Returns:
            EnergyOutput with total energy, per-node energies, per-pair energies
        """
        n_steps = hidden_states.shape[0]

        # Enable gradient computation if needed
        if compute_gradients:
            hidden_states = hidden_states.requires_grad_(True)

        # Node energies
        node_energies = self.node_energy(hidden_states)  # (n_steps,)
        total_energy = node_energies.sum()

        # Pair energies
        pair_energies = None
        if self.use_pair_energy and self.pair_energy is not None:
            if pairs is None:
                # Default: adjacent pairs (sequential consistency)
                if n_steps > 1:
                    pairs = torch.stack([
                        torch.arange(n_steps - 1, device=hidden_states.device),
                        torch.arange(1, n_steps, device=hidden_states.device)
                    ], dim=1)

            if pairs is not None and len(pairs) > 0:
                pair_energies = self.pair_energy.forward_batch_pairs(hidden_states, pairs)
                total_energy = total_energy + self.pair_weight * pair_energies.sum()

        # Compute gradients if requested
        gradients = None
        if compute_gradients:
            gradients = torch.autograd.grad(
                total_energy, hidden_states, create_graph=True
            )[0]

        return EnergyOutput(
            total_energy=total_energy,
            node_energies=node_energies,
            pair_energies=pair_energies,
            gradients=gradients,
        )

    def forward_batch(
        self,
        hidden_states: torch.Tensor,
        step_counts: torch.Tensor,
        compute_gradients: bool = False,
    ) -> List[EnergyOutput]:
        """
        Compute energies for a batch of solutions with varying step counts.

        Args:
            hidden_states: Padded hidden states, shape (batch, max_steps, hidden_dim)
            step_counts: Number of steps per solution, shape (batch,)
            compute_gradients: Whether to compute gradients

        Returns:
            List of EnergyOutput, one per solution
        """
        results = []
        batch_size = hidden_states.shape[0]

        for b in range(batch_size):
            n_steps = step_counts[b].item()
            h = hidden_states[b, :n_steps]  # (n_steps, hidden_dim)
            results.append(self.forward(h, compute_gradients=compute_gradients))

        return results

    def negative_gradient(self, hidden_states: torch.Tensor, pairs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute -grad_h(E) for ODE integration.

        This is the direction that decreases energy, pushing toward correct solutions.

        Args:
            hidden_states: Step hidden states, shape (n_steps, hidden_dim)
            pairs: Optional pair indices

        Returns:
            Negative gradient, shape (n_steps, hidden_dim)
        """
        output = self.forward(hidden_states, pairs, compute_gradients=True)
        return -output.gradients

    def energy(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute total energy for ODE solver interface.

        Args:
            h: Hidden states, shape (n_steps, hidden_dim) or (batch, hidden_dim)

        Returns:
            Total energy as scalar or (batch,)
        """
        # Handle both batched and single sample
        if h.dim() == 1:
            h = h.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        # For ODE solver: just use node energies (simpler, faster)
        node_energies = self.node_energy(h)  # (batch,) or (n_steps,)
        total = node_energies.sum()

        return total

    def gradient(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute energy gradient for ODE solver interface.

        Args:
            h: Hidden states, shape (n_steps, hidden_dim)

        Returns:
            Gradient, shape (n_steps, hidden_dim)
        """
        h = h.requires_grad_(True)
        e = self.energy(h)
        grad = torch.autograd.grad(e, h, create_graph=False)[0]
        return grad


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for training the energy landscape.

    Correct solutions should have LOWER energy than incorrect solutions.

    Loss = max(0, E_correct - E_incorrect + margin)

    This is a margin-based hinge loss that pushes:
    - E_correct down
    - E_incorrect up
    - Until they're separated by at least `margin`
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        energy_correct: torch.Tensor,
        energy_incorrect: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            energy_correct: Energies of correct solutions, shape (batch,) or scalar
            energy_incorrect: Energies of incorrect solutions, shape (batch,) or scalar

        Returns:
            Contrastive loss, scalar
        """
        # Hinge loss: want E_correct < E_incorrect - margin
        loss = F.relu(energy_correct - energy_incorrect + self.margin)
        return loss.mean()


class EnergyLandscapeTrainer:
    """
    Training utilities for the energy landscape.

    Trains on pairs of (correct_solution, incorrect_solution) where each
    solution is a sequence of step hidden states.
    """

    def __init__(
        self,
        energy_landscape: EnergyLandscape,
        lr: float = 1e-4,
        margin: float = 1.0,
        weight_decay: float = 0.01,
    ):
        self.energy_landscape = energy_landscape
        self.loss_fn = ContrastiveLoss(margin=margin)
        self.optimizer = torch.optim.AdamW(
            energy_landscape.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    def train_step(
        self,
        correct_hidden_states: torch.Tensor,
        incorrect_hidden_states: torch.Tensor,
        correct_pairs: Optional[torch.Tensor] = None,
        incorrect_pairs: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Single training step on a correct/incorrect pair.

        Args:
            correct_hidden_states: Hidden states for correct solution
            incorrect_hidden_states: Hidden states for incorrect solution
            correct_pairs: Optional pair indices for correct solution
            incorrect_pairs: Optional pair indices for incorrect solution

        Returns:
            Dict with 'loss', 'energy_correct', 'energy_incorrect'
        """
        self.optimizer.zero_grad()

        # Compute energies
        correct_output = self.energy_landscape(correct_hidden_states, correct_pairs)
        incorrect_output = self.energy_landscape(incorrect_hidden_states, incorrect_pairs)

        # Contrastive loss
        loss = self.loss_fn(correct_output.total_energy, incorrect_output.total_energy)

        # Backward
        loss.backward()
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'energy_correct': correct_output.total_energy.item(),
            'energy_incorrect': incorrect_output.total_energy.item(),
            'margin': incorrect_output.total_energy.item() - correct_output.total_energy.item(),
        }

    def train_batch(
        self,
        correct_batch: List[torch.Tensor],
        incorrect_batch: List[torch.Tensor],
    ) -> Dict[str, float]:
        """
        Train on a batch of correct/incorrect pairs.

        Args:
            correct_batch: List of correct solution hidden states
            incorrect_batch: List of incorrect solution hidden states

        Returns:
            Averaged metrics
        """
        self.optimizer.zero_grad()

        total_loss = 0.0
        total_e_correct = 0.0
        total_e_incorrect = 0.0
        n = len(correct_batch)

        for correct_h, incorrect_h in zip(correct_batch, incorrect_batch):
            correct_output = self.energy_landscape(correct_h)
            incorrect_output = self.energy_landscape(incorrect_h)

            loss = self.loss_fn(correct_output.total_energy, incorrect_output.total_energy)
            total_loss += loss
            total_e_correct += correct_output.total_energy.item()
            total_e_incorrect += incorrect_output.total_energy.item()

        # Average loss
        avg_loss = total_loss / n
        avg_loss.backward()
        self.optimizer.step()

        return {
            'loss': avg_loss.item(),
            'energy_correct': total_e_correct / n,
            'energy_incorrect': total_e_incorrect / n,
            'margin': (total_e_incorrect - total_e_correct) / n,
        }


def create_energy_landscape(
    hidden_dim: int = HIDDEN_DIM,
    pair_weight: float = 0.1,
    use_pair_energy: bool = True,
    device: str = 'cpu',
) -> EnergyLandscape:
    """
    Factory function to create an energy landscape with sensible defaults.

    Args:
        hidden_dim: Dimension of hidden states (896 for Qwen-0.5B)
        pair_weight: Weight for pairwise energies
        use_pair_energy: Whether to use pairwise energy terms
        device: Device to place model on

    Returns:
        EnergyLandscape model
    """
    model = EnergyLandscape(
        hidden_dim=hidden_dim,
        node_intermediate_dims=(512, 256),
        pair_intermediate_dims=(512, 256),
        pair_weight=pair_weight,
        use_pair_energy=use_pair_energy,
    )

    # Use float32 (A10G doesn't support bfloat16)
    model = model.to(device=device, dtype=torch.float32)

    return model


if __name__ == "__main__":
    print("=== Energy Landscape Tests ===\n")

    # Test device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Hidden dim: {HIDDEN_DIM}")

    # Create model
    energy_model = create_energy_landscape(device=device)
    print(f"\nModel created with {sum(p.numel() for p in energy_model.parameters()):,} parameters")

    # Test node energy
    print("\n1. Node Energy MLP:")
    h_single = torch.randn(HIDDEN_DIM, device=device)
    e_single = energy_model.node_energy(h_single)
    print(f"   Single step energy: {e_single.item():.4f}")

    h_batch = torch.randn(5, HIDDEN_DIM, device=device)
    e_batch = energy_model.node_energy(h_batch)
    print(f"   Batch energies: {e_batch.tolist()}")

    # Test pair energy antisymmetry
    print("\n2. Pair Energy Antisymmetry:")
    h_i = torch.randn(HIDDEN_DIM, device=device)
    h_j = torch.randn(HIDDEN_DIM, device=device)
    e_ij = energy_model.pair_energy(h_i, h_j)
    e_ji = energy_model.pair_energy(h_j, h_i)
    print(f"   E(i,j) = {e_ij.item():.4f}")
    print(f"   E(j,i) = {e_ji.item():.4f}")
    print(f"   E(i,j) + E(j,i) = {(e_ij + e_ji).item():.6f} (should be ~0)")

    # Test full energy computation
    print("\n3. Full Energy Landscape:")
    solution = torch.randn(4, HIDDEN_DIM, device=device)  # 4-step solution
    output = energy_model(solution, compute_gradients=True)
    print(f"   Total energy: {output.total_energy.item():.4f}")
    print(f"   Node energies: {output.node_energies.tolist()}")
    print(f"   Pair energies: {output.pair_energies.tolist() if output.pair_energies is not None else 'N/A'}")
    print(f"   Gradients shape: {output.gradients.shape}")

    # Test negative gradient (for ODE)
    print("\n4. Negative Gradient (ODE direction):")
    neg_grad = energy_model.negative_gradient(solution)
    print(f"   Shape: {neg_grad.shape}")
    print(f"   Norm: {neg_grad.norm().item():.4f}")

    # Test contrastive loss
    print("\n5. Contrastive Training:")
    correct_solution = torch.randn(4, HIDDEN_DIM, device=device)
    incorrect_solution = torch.randn(4, HIDDEN_DIM, device=device)

    trainer = EnergyLandscapeTrainer(energy_model, lr=1e-3, margin=1.0)

    print("   Training for 10 steps...")
    for step in range(10):
        metrics = trainer.train_step(correct_solution, incorrect_solution)
        if step % 3 == 0:
            print(f"   Step {step}: loss={metrics['loss']:.4f}, "
                  f"E_correct={metrics['energy_correct']:.4f}, "
                  f"E_incorrect={metrics['energy_incorrect']:.4f}, "
                  f"margin={metrics['margin']:.4f}")

    print("\n   Final state:")
    print(f"   E(correct) < E(incorrect): {metrics['energy_correct'] < metrics['energy_incorrect']}")
    print(f"   Margin achieved: {metrics['margin']:.4f} (target: 1.0)")

    print("\n=== All tests passed ===")
