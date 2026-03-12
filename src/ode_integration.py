"""
ODE Integration for Mycelium v7

Wires the trained energy landscape to the ODE solver for inference.
Takes generator hidden states, refines them via gradient descent on E(h),
then decodes to SymPy telegrams for oracle execution.

Pipeline:
    Generator hidden states (896-dim)
    → ODE refinement on energy landscape
    → Decode to telegram sequence
    → Oracle executes telegrams
    → Compare final answer to gold
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import json

from .energy_landscape import EnergyLandscape, create_energy_landscape, HIDDEN_DIM
from .ode_solver import ODESolver, ODEResult, create_ode_solver


@dataclass
class RefinementResult:
    """Result from ODE refinement + oracle execution."""
    # ODE refinement
    refined_hidden: torch.Tensor
    initial_energy: float
    final_energy: float
    converged: bool

    # Telegram execution
    telegrams: List[str]
    execution_success: bool
    execution_rate: float
    final_answer: Any

    # Verification
    correct: Optional[bool] = None
    gold_answer: Optional[str] = None


class ODEIntegration:
    """
    Full integration: energy landscape + ODE solver + oracle.

    Usage:
        integration = ODEIntegration.from_checkpoint(
            's3://mycelium-data/models/energy_landscape_v1/energy_landscape.pt'
        )
        result = integration.refine_and_execute(
            hidden_states,  # (n_steps, 896)
            telegrams,      # List[str] from generator
            gold_answer,    # Optional for verification
            bp_depth=3
        )
    """

    def __init__(
        self,
        energy_landscape: EnergyLandscape,
        ode_solver: ODESolver,
        device: torch.device = None
    ):
        self.energy_landscape = energy_landscape
        self.ode_solver = ode_solver
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move energy landscape to device
        self.energy_landscape = self.energy_landscape.to(self.device)
        self.energy_landscape.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        adaptive: bool = False,
        scale: float = 0.1,
        epsilon: float = 1e-4,
        max_steps: int = 100,
        device: torch.device = None
    ) -> 'ODEIntegration':
        """
        Load from checkpoint file (local path or S3).

        Args:
            checkpoint_path: Path to energy_landscape.pt
            adaptive: Use adaptive ODE solver with restarts
            scale: ODE update scale
            epsilon: Convergence threshold
            max_steps: Max ODE integration steps
            device: Device for computation
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load checkpoint
        if checkpoint_path.startswith('s3://'):
            import subprocess
            import io
            result = subprocess.run(
                ['aws', 's3', 'cp', checkpoint_path, '-'],
                capture_output=True
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to download {checkpoint_path}")
            checkpoint = torch.load(io.BytesIO(result.stdout), map_location=device, weights_only=False)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Create energy landscape
        energy_landscape = create_energy_landscape(
            hidden_dim=HIDDEN_DIM,
            pair_weight=0.1,
            use_pair_energy=True,
            device=str(device)
        )

        # Load state dict - handle architecture mismatches
        loaded = False
        if isinstance(checkpoint, dict):
            state_dict = None
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'energy_net' in checkpoint:
                state_dict = checkpoint['energy_net']
            else:
                state_dict = checkpoint

            # Check for architecture match
            if state_dict:
                try:
                    energy_landscape.load_state_dict(state_dict)
                    loaded = True
                    print(f"Loaded energy landscape from {checkpoint_path}")
                except RuntimeError as e:
                    print(f"Warning: Architecture mismatch, using random initialization")
                    print(f"  (checkpoint was trained with different dimensions)")
                    loaded = False

        if not loaded:
            print("Using randomly initialized energy landscape")

        # Create ODE solver
        ode_solver = create_ode_solver(
            energy_landscape=energy_landscape,
            adaptive=adaptive,
            scale=scale,
            epsilon=epsilon,
            max_steps=max_steps,
            device=device
        )

        return cls(energy_landscape, ode_solver, device)

    @classmethod
    def create_untrained(
        cls,
        adaptive: bool = False,
        scale: float = 0.1,
        device: torch.device = None
    ) -> 'ODEIntegration':
        """Create with untrained (random) energy landscape for testing."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        energy_landscape = create_energy_landscape(
            hidden_dim=HIDDEN_DIM,
            pair_weight=0.1,
            use_pair_energy=True,
            device=str(device)
        )

        ode_solver = create_ode_solver(
            energy_landscape=energy_landscape,
            adaptive=adaptive,
            scale=scale,
            device=device
        )

        return cls(energy_landscape, ode_solver, device)

    def compute_energy(self, hidden_states: torch.Tensor) -> float:
        """Compute total energy of hidden states."""
        hidden_states = hidden_states.to(self.device, dtype=torch.float32)
        with torch.no_grad():
            output = self.energy_landscape(hidden_states)
            return output.total_energy.item()

    def refine_hidden_states(
        self,
        hidden_states: torch.Tensor,
        bp_depth: int = 3,
        return_trajectory: bool = False
    ) -> ODEResult:
        """
        Refine hidden states via ODE gradient descent on energy landscape.

        Args:
            hidden_states: Generator hidden states (n_steps, 896)
            bp_depth: Belief propagation depth (controls integration time)
            return_trajectory: Return full trajectory for visualization

        Returns:
            ODEResult with refined hidden states
        """
        hidden_states = hidden_states.to(self.device, dtype=torch.float32)

        # Flatten to (n_steps * hidden_dim,) for ODE
        # The energy landscape expects (n_steps, hidden_dim)
        # But ODE solver expects (batch, hidden_dim)
        # So we process step-by-step or reshape

        n_steps = hidden_states.shape[0]

        # Process all steps as a batch
        result = self.ode_solver.refine(
            hidden_states,
            bp_depth=bp_depth,
            return_trajectory=return_trajectory
        )

        return result

    def refine_and_execute(
        self,
        hidden_states: torch.Tensor,
        telegrams: List[str],
        gold_answer: Optional[str] = None,
        bp_depth: int = 3,
        timeout: int = 5
    ) -> RefinementResult:
        """
        Full pipeline: refine hidden states, execute telegrams, verify.

        Args:
            hidden_states: Generator hidden states (n_steps, 896)
            telegrams: Generated telegram sequence
            gold_answer: Optional gold answer for verification
            bp_depth: ODE integration depth
            timeout: SymPy execution timeout

        Returns:
            RefinementResult with all outputs
        """
        # Import oracle here to avoid circular imports
        from .oracle import execute_sequence, compare_answers, parse_telegram_expr

        # 1. Refine hidden states
        ode_result = self.refine_hidden_states(hidden_states, bp_depth)

        # 2. Execute telegram sequence
        exec_result = execute_sequence(telegrams, timeout=timeout)

        # 3. Parse final answer for comparison
        final_answer = None
        if exec_result['final_answer']:
            try:
                final_answer = parse_telegram_expr(exec_result['final_answer'])
            except Exception:
                final_answer = exec_result['final_answer']

        # 4. Compare to gold if provided
        correct = None
        if gold_answer is not None and final_answer is not None:
            correct = compare_answers(final_answer, gold_answer, timeout=timeout)

        return RefinementResult(
            refined_hidden=ode_result.refined,
            initial_energy=ode_result.initial_energy,
            final_energy=ode_result.final_energy,
            converged=ode_result.converged,
            telegrams=telegrams,
            execution_success=exec_result['success'],
            execution_rate=exec_result['execution_rate'],
            final_answer=final_answer,
            correct=correct,
            gold_answer=gold_answer
        )

    def batch_refine(
        self,
        hidden_states_list: List[torch.Tensor],
        bp_depths: List[int]
    ) -> List[ODEResult]:
        """
        Refine a batch of hidden state sequences.

        Args:
            hidden_states_list: List of (n_steps, 896) tensors
            bp_depths: BP depth for each sequence

        Returns:
            List of ODEResult
        """
        results = []
        for h, depth in zip(hidden_states_list, bp_depths):
            results.append(self.refine_hidden_states(h, bp_depth=depth))
        return results


class EnergyGuidedDecoder:
    """
    Use energy landscape to guide telegram generation.

    At each decoding step, use energy gradient to bias token selection
    toward lower-energy (more correct) continuations.
    """

    def __init__(
        self,
        energy_landscape: EnergyLandscape,
        temperature: float = 1.0,
        energy_weight: float = 0.1
    ):
        self.energy_landscape = energy_landscape
        self.temperature = temperature
        self.energy_weight = energy_weight

    def score_candidates(
        self,
        current_hidden: torch.Tensor,
        candidate_hiddens: torch.Tensor
    ) -> torch.Tensor:
        """
        Score candidate next-step hidden states by energy.

        Args:
            current_hidden: Current hidden state (896,)
            candidate_hiddens: Candidate next hiddens (n_candidates, 896)

        Returns:
            Energy-based scores (n_candidates,) - lower is better
        """
        device = current_hidden.device
        n_candidates = candidate_hiddens.shape[0]

        with torch.no_grad():
            # Compute node energy for each candidate
            energies = self.energy_landscape.node_energy(candidate_hiddens)

            # Compute pair energy with current state
            if self.energy_landscape.use_pair_energy:
                current_expanded = current_hidden.unsqueeze(0).expand(n_candidates, -1)
                pair_energies = self.energy_landscape.pair_energy(
                    current_expanded, candidate_hiddens
                )
                energies = energies + self.energy_landscape.pair_weight * pair_energies

            return energies


def load_integration(
    checkpoint_path: str = 's3://mycelium-data/models/energy_landscape_v1/energy_landscape.pt',
    adaptive: bool = False
) -> ODEIntegration:
    """Convenience function to load ODE integration."""
    return ODEIntegration.from_checkpoint(checkpoint_path, adaptive=adaptive)


if __name__ == "__main__":
    print("=== ODE Integration Tests ===\n")

    # Test with untrained energy landscape
    print("1. Creating untrained integration...")
    integration = ODEIntegration.create_untrained(adaptive=False)
    print(f"   Device: {integration.device}")

    # Test energy computation
    print("\n2. Testing energy computation...")
    dummy_hidden = torch.randn(4, HIDDEN_DIM)  # 4-step solution
    energy = integration.compute_energy(dummy_hidden)
    print(f"   Energy: {energy:.4f}")

    # Test ODE refinement
    print("\n3. Testing ODE refinement...")
    result = integration.refine_hidden_states(dummy_hidden, bp_depth=3)
    print(f"   Initial energy: {result.initial_energy:.4f}")
    print(f"   Final energy: {result.final_energy:.4f}")
    print(f"   Converged: {result.converged}")
    print(f"   Energy reduction: {(1 - result.final_energy/result.initial_energy)*100:.1f}%")

    # Test full pipeline with mock telegrams
    print("\n4. Testing full pipeline...")
    mock_telegrams = [
        "GIVEN x^2 + y^2 = 25",
        "GIVEN x + y = 7",
        "SOLVE _prev x",
        "ANSWER _prev"
    ]

    try:
        full_result = integration.refine_and_execute(
            dummy_hidden,
            mock_telegrams,
            gold_answer="3",  # One possible solution
            bp_depth=3
        )
        print(f"   Execution success: {full_result.execution_success}")
        print(f"   Execution rate: {full_result.execution_rate:.1%}")
        print(f"   Final answer: {full_result.final_answer}")
        print(f"   Correct: {full_result.correct}")
    except ImportError as e:
        print(f"   Skipped (oracle not available): {e}")

    print("\n=== Tests complete ===")
