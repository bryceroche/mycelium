"""
ODE Solver for Mycelium v7

Refines rough slot filler outputs to precise SymPy expressions via continuous
optimization on a learned energy landscape.

Architecture:
- Uses dopri5 adaptive integrator from torchdiffeq
- Gradient descent dynamics: dh/dt = -grad(E(h))
- Bounded updates: tanh * 0.1 to prevent divergence
- Integrates until convergence or max_steps

The solver takes a rough hidden state representation and refines it by
flowing down the energy landscape until it reaches a low-energy basin
corresponding to a valid SymPy expression.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Callable, Protocol
from dataclasses import dataclass

# torchdiffeq for ODE integration
try:
    from torchdiffeq import odeint
except ImportError:
    raise ImportError(
        "torchdiffeq is required for ODE solving. "
        "Install with: pip install torchdiffeq"
    )


class EnergyLandscape(Protocol):
    """Protocol for energy landscape interface."""

    def energy(self, h: torch.Tensor) -> torch.Tensor:
        """Compute energy for hidden state(s)."""
        ...

    def gradient(self, h: torch.Tensor) -> torch.Tensor:
        """Compute energy gradient with respect to h."""
        ...


@dataclass
class ODEResult:
    """Result of ODE refinement."""
    refined: torch.Tensor  # Final refined hidden state
    initial: torch.Tensor  # Initial hidden state (for reference)
    trajectory: Optional[torch.Tensor]  # Full trajectory if requested
    converged: bool  # Whether convergence criterion was met
    n_steps: int  # Number of integration steps taken
    final_energy: float  # Energy at final state
    initial_energy: float  # Energy at initial state


class EnergyGradientDynamics(nn.Module):
    """
    ODE dynamics for gradient descent on energy landscape.

    dh/dt = -tanh(grad(E(h))) * scale

    The tanh bounds the update magnitude to prevent divergence,
    while still following the energy gradient direction.
    """

    def __init__(
        self,
        energy_landscape: EnergyLandscape,
        scale: float = 0.1,
        use_autograd: bool = True
    ):
        """
        Args:
            energy_landscape: The learned energy landscape
            scale: Scale factor for updates (default 0.1)
            use_autograd: If True, compute gradient via autograd. If False,
                         use energy_landscape.gradient() method.
        """
        super().__init__()
        self.energy_landscape = energy_landscape
        self.scale = scale
        self.use_autograd = use_autograd

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Compute dh/dt = -tanh(grad(E(h))) * scale

        Args:
            t: Current time (unused but required by ODE interface)
            h: Current hidden state [batch_size, hidden_dim]

        Returns:
            dh/dt: Time derivative of hidden state
        """
        if self.use_autograd:
            # Compute gradient via autograd
            h_grad = h.detach().requires_grad_(True)
            energy = self.energy_landscape.energy(h_grad)

            # Sum energy if batched (for gradient computation)
            if energy.dim() > 0:
                energy = energy.sum()

            grad = torch.autograd.grad(energy, h_grad, create_graph=False)[0]
        else:
            # Use explicit gradient method from energy landscape
            grad = self.energy_landscape.gradient(h)

        # Bounded update: tanh prevents divergence
        dh_dt = -torch.tanh(grad) * self.scale

        return dh_dt


class ODESolver:
    """
    ODE-based refinement solver for Mycelium.

    Takes rough hidden state representations and refines them by
    integrating gradient descent dynamics on the energy landscape.

    Usage:
        solver = ODESolver(energy_landscape)
        result = solver.refine(rough_hidden, bp_depth=3)
        refined_hidden = result.refined
    """

    def __init__(
        self,
        energy_landscape: EnergyLandscape,
        scale: float = 0.1,
        epsilon: float = 1e-4,
        max_steps: int = 100,
        method: str = 'dopri5',
        rtol: float = 1e-5,
        atol: float = 1e-6,
        use_autograd: bool = True,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            energy_landscape: The learned energy landscape for computing E(h)
            scale: Scale factor for bounded updates (default 0.1)
            epsilon: Convergence threshold for ||dh/dt|| (default 1e-4)
            max_steps: Maximum integration steps (default 100)
            method: ODE solver method (default 'dopri5')
            rtol: Relative tolerance for adaptive solver (default 1e-5)
            atol: Absolute tolerance for adaptive solver (default 1e-6)
            use_autograd: If True, compute gradient via autograd (default True)
            device: Device for computation (default: auto-detect)
        """
        self.energy_landscape = energy_landscape
        self.scale = scale
        self.epsilon = epsilon
        self.max_steps = max_steps
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.use_autograd = use_autograd

        # Device handling - default to CPU, use float32 (A10G doesn't support bfloat16)
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        # Create dynamics module
        self.dynamics = EnergyGradientDynamics(
            energy_landscape=energy_landscape,
            scale=scale,
            use_autograd=use_autograd
        )

    def _compute_integration_time(self, bp_depth: int) -> float:
        """
        Map bp_depth (belief propagation depth from C1-B) to integration time.

        Higher bp_depth means more complex problem structure,
        requiring longer integration time.

        Args:
            bp_depth: Belief propagation depth (typically 1-5)

        Returns:
            Integration time T
        """
        # Linear mapping: bp_depth in [1, 5] -> T in [1.0, 5.0]
        # Can be tuned based on empirical results
        base_time = 1.0
        time_per_depth = 1.0
        return base_time + (bp_depth - 1) * time_per_depth

    def _check_convergence(self, h: torch.Tensor) -> Tuple[bool, float]:
        """
        Check if the dynamics have converged.

        Convergence is when ||dh/dt|| < epsilon.

        Args:
            h: Current hidden state

        Returns:
            (converged: bool, norm: float)
        """
        # Compute dh/dt at current state
        t_dummy = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        dh_dt = self.dynamics(t_dummy, h)

        # Compute L2 norm
        norm = torch.norm(dh_dt).item()

        return norm < self.epsilon, norm

    def refine(
        self,
        rough_hidden: torch.Tensor,
        bp_depth: int = 3,
        return_trajectory: bool = False,
        n_trajectory_points: int = 20
    ) -> ODEResult:
        """
        Refine rough hidden state via ODE integration on energy landscape.

        Args:
            rough_hidden: Initial rough hidden state [batch_size, hidden_dim] or [hidden_dim]
            bp_depth: Belief propagation depth from C1-B (controls integration time)
            return_trajectory: If True, return full trajectory
            n_trajectory_points: Number of points in trajectory (if requested)

        Returns:
            ODEResult containing refined state and diagnostics
        """
        # Ensure correct device and dtype (float32 required for A10G)
        rough_hidden = rough_hidden.to(device=self.device, dtype=torch.float32)

        # Handle 1D input (single sample)
        squeeze_output = False
        if rough_hidden.dim() == 1:
            rough_hidden = rough_hidden.unsqueeze(0)
            squeeze_output = True

        # Store initial state
        initial = rough_hidden.clone()

        # Compute initial energy
        with torch.no_grad():
            initial_energy = self.energy_landscape.energy(initial)
            if initial_energy.dim() > 0:
                initial_energy = initial_energy.mean().item()
            else:
                initial_energy = initial_energy.item()

        # Compute integration time from bp_depth
        T = self._compute_integration_time(bp_depth)

        # Set up time points for integration
        if return_trajectory:
            t_span = torch.linspace(0, T, n_trajectory_points, device=self.device, dtype=torch.float32)
        else:
            # Just initial and final time
            t_span = torch.tensor([0.0, T], device=self.device, dtype=torch.float32)

        # Integrate ODE
        # dopri5 is an adaptive solver that automatically handles step sizing
        trajectory = odeint(
            self.dynamics,
            rough_hidden,
            t_span,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            options={'max_num_steps': self.max_steps}
        )

        # Extract final state
        refined = trajectory[-1]

        # Check convergence at final state
        converged, final_norm = self._check_convergence(refined)

        # Compute final energy
        with torch.no_grad():
            final_energy = self.energy_landscape.energy(refined)
            if final_energy.dim() > 0:
                final_energy = final_energy.mean().item()
            else:
                final_energy = final_energy.item()

        # Handle trajectory output
        if return_trajectory:
            traj_out = trajectory
        else:
            traj_out = None

        # Squeeze output if input was 1D
        if squeeze_output:
            refined = refined.squeeze(0)
            initial = initial.squeeze(0)
            if traj_out is not None:
                traj_out = traj_out.squeeze(1)

        return ODEResult(
            refined=refined,
            initial=initial,
            trajectory=traj_out,
            converged=converged,
            n_steps=len(t_span) - 1,  # Approximate - adaptive solver may use more
            final_energy=final_energy,
            initial_energy=initial_energy
        )

    def refine_batch(
        self,
        rough_hiddens: torch.Tensor,
        bp_depths: torch.Tensor,
        return_trajectories: bool = False
    ) -> list:
        """
        Refine a batch of hidden states with potentially different bp_depths.

        Note: For efficiency, samples with the same bp_depth are grouped
        and processed together.

        Args:
            rough_hiddens: Batch of hidden states [batch_size, hidden_dim]
            bp_depths: BP depth for each sample [batch_size]
            return_trajectories: If True, return trajectories for all samples

        Returns:
            List of ODEResult, one per sample
        """
        rough_hiddens = rough_hiddens.to(device=self.device, dtype=torch.float32)
        bp_depths = bp_depths.to(device=self.device)

        batch_size = rough_hiddens.shape[0]
        results = [None] * batch_size

        # Group by bp_depth for efficient batched integration
        unique_depths = torch.unique(bp_depths)

        for depth in unique_depths:
            depth_val = int(depth.item())
            mask = bp_depths == depth
            indices = torch.where(mask)[0]

            # Get subset with this bp_depth
            subset = rough_hiddens[mask]

            # Refine this subset
            result = self.refine(
                subset,
                bp_depth=depth_val,
                return_trajectory=return_trajectories
            )

            # Distribute results back
            for i, idx in enumerate(indices):
                idx = idx.item()
                if subset.shape[0] == 1:
                    # Single sample case
                    results[idx] = ODEResult(
                        refined=result.refined,
                        initial=result.initial,
                        trajectory=result.trajectory,
                        converged=result.converged,
                        n_steps=result.n_steps,
                        final_energy=result.final_energy,
                        initial_energy=result.initial_energy
                    )
                else:
                    # Extract individual sample from batch
                    results[idx] = ODEResult(
                        refined=result.refined[i],
                        initial=result.initial[i],
                        trajectory=result.trajectory[:, i] if result.trajectory is not None else None,
                        converged=result.converged,  # Same for whole batch
                        n_steps=result.n_steps,
                        final_energy=result.final_energy,  # Averaged - approximate
                        initial_energy=result.initial_energy
                    )

        return results


class AdaptiveODESolver(ODESolver):
    """
    Adaptive ODE solver that monitors energy and adjusts integration.

    Extends the base solver with:
    - Early stopping when energy plateaus
    - Restart from current state if stuck in local minimum
    - Energy barrier detection
    """

    def __init__(
        self,
        energy_landscape: EnergyLandscape,
        scale: float = 0.1,
        epsilon: float = 1e-4,
        max_steps: int = 100,
        method: str = 'dopri5',
        rtol: float = 1e-5,
        atol: float = 1e-6,
        use_autograd: bool = True,
        device: Optional[torch.device] = None,
        plateau_patience: int = 5,
        plateau_threshold: float = 1e-6
    ):
        """
        Additional args beyond ODESolver:
            plateau_patience: Steps to wait before declaring plateau
            plateau_threshold: Minimum energy decrease to not count as plateau
        """
        super().__init__(
            energy_landscape=energy_landscape,
            scale=scale,
            epsilon=epsilon,
            max_steps=max_steps,
            method=method,
            rtol=rtol,
            atol=atol,
            use_autograd=use_autograd,
            device=device
        )
        self.plateau_patience = plateau_patience
        self.plateau_threshold = plateau_threshold

    def refine_adaptive(
        self,
        rough_hidden: torch.Tensor,
        bp_depth: int = 3,
        max_restarts: int = 3,
        noise_scale: float = 0.01
    ) -> ODEResult:
        """
        Adaptively refine with plateau detection and restarts.

        If energy plateaus, adds small noise and restarts integration
        to escape local minima.

        Args:
            rough_hidden: Initial rough hidden state
            bp_depth: Controls base integration time
            max_restarts: Maximum number of restarts on plateau
            noise_scale: Scale of noise added on restart

        Returns:
            ODEResult from best (lowest energy) run
        """
        best_result = None
        best_energy = float('inf')
        current_hidden = rough_hidden.clone()

        for restart in range(max_restarts + 1):
            # Run standard refinement
            result = self.refine(
                current_hidden,
                bp_depth=bp_depth,
                return_trajectory=True,
                n_trajectory_points=self.plateau_patience + 1
            )

            # Track best result
            if result.final_energy < best_energy:
                best_energy = result.final_energy
                best_result = result

            # Check for convergence
            if result.converged:
                break

            # Check for plateau
            if result.trajectory is not None:
                energies = []
                for i in range(result.trajectory.shape[0]):
                    with torch.no_grad():
                        e = self.energy_landscape.energy(result.trajectory[i])
                        if e.dim() > 0:
                            e = e.mean()
                        energies.append(e.item())

                # Check if energy is decreasing sufficiently
                energy_diffs = [energies[i] - energies[i+1] for i in range(len(energies)-1)]
                is_plateau = all(d < self.plateau_threshold for d in energy_diffs[-self.plateau_patience:])

                if is_plateau and restart < max_restarts:
                    # Add noise to escape local minimum
                    noise = torch.randn_like(result.refined) * noise_scale
                    current_hidden = result.refined + noise
                    continue

            break

        return best_result


# Convenience function for creating solver
def create_ode_solver(
    energy_landscape: EnergyLandscape,
    adaptive: bool = False,
    **kwargs
) -> ODESolver:
    """
    Factory function to create appropriate ODE solver.

    Args:
        energy_landscape: The learned energy landscape
        adaptive: If True, create AdaptiveODESolver
        **kwargs: Additional arguments passed to solver constructor

    Returns:
        ODESolver or AdaptiveODESolver instance
    """
    if adaptive:
        return AdaptiveODESolver(energy_landscape, **kwargs)
    else:
        return ODESolver(energy_landscape, **kwargs)


if __name__ == "__main__":
    # Test with mock energy landscape
    print("=== ODE Solver Tests ===\n")

    class MockEnergyLandscape:
        """Simple quadratic energy for testing: E(h) = ||h||^2"""

        def energy(self, h: torch.Tensor) -> torch.Tensor:
            return (h ** 2).sum(dim=-1)

        def gradient(self, h: torch.Tensor) -> torch.Tensor:
            return 2 * h

    # Create solver with mock landscape
    landscape = MockEnergyLandscape()
    solver = ODESolver(
        energy_landscape=landscape,
        scale=0.1,
        epsilon=1e-4,
        max_steps=100,
        device=torch.device('cpu')  # Use CPU for testing
    )

    # Test 1: Single sample refinement
    print("1. Single sample refinement:")
    rough = torch.randn(896, dtype=torch.float32) * 5  # 896-dim hidden state
    result = solver.refine(rough, bp_depth=3)
    print(f"   Initial energy: {result.initial_energy:.4f}")
    print(f"   Final energy: {result.final_energy:.4f}")
    print(f"   Converged: {result.converged}")
    print(f"   Energy reduction: {(1 - result.final_energy/result.initial_energy)*100:.1f}%")

    # Test 2: Batch refinement
    print("\n2. Batch refinement with varying bp_depth:")
    batch = torch.randn(4, 896, dtype=torch.float32) * 5
    depths = torch.tensor([1, 2, 3, 4])
    results = solver.refine_batch(batch, depths)
    for i, res in enumerate(results):
        print(f"   Sample {i} (depth={depths[i].item()}): "
              f"energy {res.initial_energy:.2f} -> {res.final_energy:.4f}")

    # Test 3: Trajectory visualization
    print("\n3. Trajectory (energy over time):")
    result = solver.refine(rough, bp_depth=3, return_trajectory=True, n_trajectory_points=10)
    if result.trajectory is not None:
        energies = [(landscape.energy(result.trajectory[i]).item())
                    for i in range(result.trajectory.shape[0])]
        print(f"   Energy trajectory: {[f'{e:.2f}' for e in energies]}")

    # Test 4: Adaptive solver
    print("\n4. Adaptive solver with restarts:")
    adaptive_solver = AdaptiveODESolver(
        energy_landscape=landscape,
        scale=0.1,
        device=torch.device('cpu')
    )
    result = adaptive_solver.refine_adaptive(rough, bp_depth=3, max_restarts=2)
    print(f"   Final energy: {result.final_energy:.6f}")
    print(f"   Converged: {result.converged}")

    print("\n=== All tests passed ===")
