"""
Entropy-aware thinking loop with four-quadrant stopping logic.

This module provides inference-time thinking with entropy flow monitoring.
It tracks page deltas, surprise per cycle, and uses EntropyFlowConfidence
to get confidence + smoothness signals for intelligent stopping decisions.

Four-quadrant decision matrix:
    High conf + smooth  -> STOP (good answer)
    High conf + choppy  -> SUSPICIOUS (log warning)
    Low conf + smooth   -> CONTINUE (on track, needs more cycles)
    Low conf + choppy   -> RETRY with perturbed atoms (if retries left)

Components used from src/entropy_flow.py:
    - EntropyTracker (page_entropy, page_delta_norm, atom_entropy)
    - SurpriseDetector (compute_surprise, compute_surprise_batch)
    - EntropyFlowConfidence (GRU-based, outputs confidence + smoothness)
    - compute_smoothness_target
"""

import os
import sys
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, '/home/ubuntu/mycelium')

from src.entropy_flow import (
    EntropyTracker,
    SurpriseDetector,
    EntropyFlowConfidence,
    compute_smoothness_target,
)


# ---------------------------------------------------------------------------
# Data classes for return values
# ---------------------------------------------------------------------------
@dataclass
class ThinkingResult:
    """Result of entropy-aware thinking loop."""
    state_pages: List[torch.Tensor]           # All pages from thinking
    atom_scales_history: List[torch.Tensor]   # Atom scales per pass
    surprises: List[torch.Tensor]             # Surprise scores per cycle
    final_confidence: torch.Tensor            # Final confidence (B, 1)
    final_smoothness: torch.Tensor            # Final smoothness (B, 1)
    num_passes: int                           # How many passes were taken
    stopped_early: bool                       # True if stopped before max_passes
    stop_reason: str                          # Why we stopped


@dataclass
class QuadrantDecision:
    """Decision from four-quadrant logic."""
    action: str                # 'STOP', 'CONTINUE', 'RETRY', 'SUSPICIOUS'
    confidence: float          # Confidence value
    smoothness: float          # Smoothness value
    reason: str                # Human-readable explanation


# ---------------------------------------------------------------------------
# Four-quadrant decision logic
# ---------------------------------------------------------------------------
def quadrant_decision(
    confidence: torch.Tensor,
    smoothness: torch.Tensor,
    conf_threshold: float = 0.85,
    smooth_threshold: float = 0.7,
    choppy_threshold: float = 0.3,
) -> QuadrantDecision:
    """
    Determine action based on confidence and smoothness (four-quadrant logic).

    Args:
        confidence: (B, 1) or scalar - confidence in current answer
        smoothness: (B, 1) or scalar - smoothness of thinking flow
        conf_threshold: threshold above which we consider "high confidence"
        smooth_threshold: threshold above which we consider "smooth"
        choppy_threshold: threshold below which we consider "choppy"

    Returns:
        QuadrantDecision with action and explanation
    """
    # Handle tensor inputs
    if isinstance(confidence, torch.Tensor):
        conf_val = confidence.mean().item()
    else:
        conf_val = confidence

    if isinstance(smoothness, torch.Tensor):
        smooth_val = smoothness.mean().item()
    else:
        smooth_val = smoothness

    high_conf = conf_val > conf_threshold
    smooth = smooth_val > smooth_threshold
    choppy = smooth_val < choppy_threshold

    if high_conf and smooth:
        return QuadrantDecision(
            action='STOP',
            confidence=conf_val,
            smoothness=smooth_val,
            reason=f"High confidence ({conf_val:.2f}) + smooth flow ({smooth_val:.2f}). Good answer."
        )
    elif high_conf and choppy:
        return QuadrantDecision(
            action='SUSPICIOUS',
            confidence=conf_val,
            smoothness=smooth_val,
            reason=f"High confidence ({conf_val:.2f}) but choppy ({smooth_val:.2f}). Might be wrong."
        )
    elif not high_conf and smooth:
        return QuadrantDecision(
            action='CONTINUE',
            confidence=conf_val,
            smoothness=smooth_val,
            reason=f"Low confidence ({conf_val:.2f}) but smooth ({smooth_val:.2f}). On track, needs more."
        )
    else:  # low conf and choppy
        return QuadrantDecision(
            action='RETRY',
            confidence=conf_val,
            smoothness=smooth_val,
            reason=f"Low confidence ({conf_val:.2f}) and choppy ({smooth_val:.2f}). Lost the thread."
        )


# ---------------------------------------------------------------------------
# Atom perturbation for retry logic
# ---------------------------------------------------------------------------
def perturb_atom_scales(
    atom_scales: torch.Tensor,
    noise_scale: float = 0.2,
) -> torch.Tensor:
    """
    Perturb atom scales with noise for retry attempts.

    Args:
        atom_scales: (B, num_atoms) - current atom scales
        noise_scale: standard deviation of noise to add

    Returns:
        perturbed_scales: (B, num_atoms) - tanh-bounded perturbed scales
    """
    noise = torch.randn_like(atom_scales) * noise_scale
    # Add noise and re-apply tanh to keep in [-1, 1]
    return torch.tanh(atom_scales + noise)


# ---------------------------------------------------------------------------
# Main thinking loop with entropy tracking
# ---------------------------------------------------------------------------
def think_with_entropy_tracking(
    model,  # AtomLoRAModel instance
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    confidence_head: EntropyFlowConfidence,
    max_passes: int = 12,
    conf_threshold: float = 0.85,
    smooth_threshold: float = 0.7,
    surprise_threshold: float = 2.0,
    max_retries: int = 2,
    verbose: bool = False,
) -> ThinkingResult:
    """
    Run thinking passes with entropy monitoring and four-quadrant stopping logic.

    Args:
        model: AtomLoRAModel with thinking_pass method
        input_ids: (B, seq_len) tokenized problem
        attention_mask: (B, seq_len) attention mask
        confidence_head: EntropyFlowConfidence module for confidence/smoothness
        max_passes: Maximum number of thinking passes
        conf_threshold: Confidence threshold for stopping
        smooth_threshold: Smoothness threshold for "smooth" classification
        surprise_threshold: Surprise threshold for retry trigger
        max_retries: Maximum retries per pass when surprise is high
        verbose: Whether to print debug info

    Returns:
        ThinkingResult with pages, scales, surprises, and final metrics
    """
    batch_size = input_ids.size(0)
    device = input_ids.device

    state_pages: List[torch.Tensor] = []
    atom_scales_history: List[torch.Tensor] = []
    page_deltas: List[torch.Tensor] = []
    mid_states_history: List[torch.Tensor] = []

    surprise_detector = SurpriseDetector()
    stop_reason = "max_passes"
    stopped_early = False

    for pass_num in range(max_passes):
        # Run one thinking pass
        page, atom_scales, current_mid_states = model.thinking_pass(
            input_ids=input_ids,
            attention_mask=attention_mask,
            state_pages=state_pages,
            pass_num=pass_num,
            max_passes=max_passes,
            prev_mid_states=mid_states_history if mid_states_history else None,
        )

        # Track page delta (information change)
        if len(state_pages) > 0:
            delta = EntropyTracker.page_delta_norm(page, state_pages[-1])
        else:
            delta = page.norm(dim=-1)  # First pass: delta from zero
        page_deltas.append(delta)

        # Compute surprise for this cycle
        surprises = surprise_detector.compute_surprise(page_deltas)
        current_surprise = surprises[-1]
        mean_surprise = current_surprise.mean().item()

        if verbose:
            print(f"Pass {pass_num}: delta={delta.mean().item():.4f}, "
                  f"surprise={mean_surprise:.4f}")

        # Retry logic: if surprise is high, perturb atoms and retry
        retry_count = 0
        while mean_surprise > surprise_threshold and retry_count < max_retries and len(state_pages) > 0:
            if verbose:
                print(f"  Retry {retry_count + 1}: surprise {mean_surprise:.2f} > {surprise_threshold}")

            # Perturb atom scales
            atom_scales_perturbed = perturb_atom_scales(atom_scales)

            # Re-run the pass with perturbed atoms
            page_retry, _, _ = _retry_thinking_pass(
                model, input_ids, attention_mask, state_pages,
                pass_num, atom_scales_perturbed, mid_states_history,
            )

            # Check if retry is smoother
            delta_retry = EntropyTracker.page_delta_norm(page_retry, state_pages[-1])

            # Compute surprise for retry
            page_deltas_retry = page_deltas[:-1] + [delta_retry]
            surprises_retry = surprise_detector.compute_surprise(page_deltas_retry)
            surprise_retry = surprises_retry[-1].mean().item()

            if surprise_retry < mean_surprise:
                # Retry was smoother - accept it
                page = page_retry
                atom_scales = atom_scales_perturbed
                page_deltas[-1] = delta_retry
                mean_surprise = surprise_retry
                if verbose:
                    print(f"    Accepted: surprise {surprise_retry:.4f} < {mean_surprise:.4f}")
            else:
                if verbose:
                    print(f"    Rejected: surprise {surprise_retry:.4f} >= {mean_surprise:.4f}")

            retry_count += 1

        # Store results
        state_pages.append(page)
        atom_scales_history.append(atom_scales)
        mid_states_history.append(current_mid_states)

        # Check confidence and smoothness (need at least 2 pages for flow detection)
        if pass_num >= 1:
            with torch.no_grad():
                confidence, smoothness = confidence_head(state_pages)

            decision = quadrant_decision(
                confidence, smoothness,
                conf_threshold=conf_threshold,
                smooth_threshold=smooth_threshold,
            )

            if verbose:
                print(f"  Decision: {decision.action} - {decision.reason}")

            if decision.action == 'STOP':
                stopped_early = True
                stop_reason = "confident_and_smooth"
                break
            elif decision.action == 'SUSPICIOUS':
                # Log warning but continue (might still be right)
                print(f"WARNING: Pass {pass_num} - {decision.reason}")
                # Could optionally stop here or continue
                # For now, continue but mark it
                stop_reason = "suspicious_but_continued"

    # Final confidence and smoothness
    with torch.no_grad():
        final_confidence, final_smoothness = confidence_head(state_pages)

    # Compute final surprises
    final_surprises = SurpriseDetector.compute_surprise_batch(page_deltas)

    return ThinkingResult(
        state_pages=state_pages,
        atom_scales_history=atom_scales_history,
        surprises=final_surprises,
        final_confidence=final_confidence,
        final_smoothness=final_smoothness,
        num_passes=len(state_pages),
        stopped_early=stopped_early,
        stop_reason=stop_reason,
    )


def _retry_thinking_pass(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    state_pages: List[torch.Tensor],
    pass_num: int,
    perturbed_scales: torch.Tensor,
    mid_states_history: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Helper to run a thinking pass with pre-computed perturbed atom scales.

    This bypasses the hypernetwork and directly uses the perturbed scales.
    Used during retry logic when surprise is high.
    """
    from scripts.atom_lora import AtomAdditiveLoRAManager

    batch_size = input_ids.size(0)

    # Apply perturbed atom LoRA via monkey-patching
    manager = AtomAdditiveLoRAManager(model.transformer)
    manager.apply(model.atoms, perturbed_scales)

    try:
        outputs = model.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = list(outputs.hidden_states[1:])
    finally:
        manager.remove()

    # Compress to get page
    page_delta, _strategy, current_mid_states = model.compressor(
        hidden_states, pass_num,
        prev_mid_states=mid_states_history if mid_states_history else None,
    )

    # Normalize on hypersphere
    page = F.normalize(page_delta, dim=-1) * model.page_radius

    # Add Fourier structural identity
    page = model.fourier_page.apply(page, pass_num)

    return page, perturbed_scales, current_mid_states


# ---------------------------------------------------------------------------
# Diagnostic: visualize entropy flow
# ---------------------------------------------------------------------------
def visualize_entropy_flow(
    state_pages: List[torch.Tensor],
    atom_scales_history: List[torch.Tensor],
    output_path: str = 'entropy_flow.png',
) -> None:
    """
    Plot entropy flow across cycles for a single problem.

    Shows: page deltas, atom entropy, surprise, cumulative smoothness.

    Args:
        state_pages: List of (batch, 64) tensors
        atom_scales_history: List of (batch, num_atoms) tensors
        output_path: Where to save the plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for visualization")
        return

    fig, axes = plt.subplots(4, 1, figsize=(10, 12))

    # Take first sample in batch for visualization
    pages = [p[0] if p.dim() > 1 else p for p in state_pages]
    scales = [s[0] if s.dim() > 1 else s for s in atom_scales_history]

    # 1. Page deltas (information change per cycle)
    deltas = []
    for i in range(1, len(pages)):
        delta = (pages[i] - pages[i-1]).norm().item()
        deltas.append(delta)

    if deltas:
        axes[0].bar(range(len(deltas)), deltas)
        axes[0].set_title('Page Delta Per Cycle (information change)')
        axes[0].set_ylabel('||page_k - page_{k-1}||')

    # 2. Atom entropy (complexity of thinking)
    atom_ents = []
    for scale in scales:
        probs = F.softmax(scale.abs(), dim=-1)
        ent = -(probs * torch.log(probs + 1e-8)).sum().item()
        atom_ents.append(ent)

    axes[1].plot(atom_ents, 'o-')
    axes[1].set_title('Atom Entropy Per Cycle (thinking complexity)')
    axes[1].set_ylabel('H(atom_scales)')

    # 3. Surprise (deviation from expected delta)
    if len(deltas) > 1:
        surprises = SurpriseDetector.compute_surprise_batch(
            [torch.tensor([d]) for d in deltas]
        )
        surprise_vals = [s.item() for s in surprises]
        colors = ['red' if s > 2.0 else 'green' for s in surprise_vals]
        axes[2].bar(range(len(surprise_vals)), surprise_vals, color=colors)
        axes[2].axhline(y=2.0, color='red', linestyle='--', label='threshold')
        axes[2].set_title('Surprise Per Cycle (red = unexpected)')
        axes[2].set_ylabel('|actual - expected| / std')
        axes[2].legend()

    # 4. Cumulative smoothness
    if len(deltas) > 1:
        smooth_vals = []
        for i in range(1, len(deltas) + 1):
            stacked = torch.tensor(deltas[:i])
            mean_val = stacked.mean()
            if mean_val > 1e-8 and len(deltas[:i]) > 1:
                cv = stacked.std() / mean_val
                smooth_vals.append(max(0, 1 - cv.item()))
            else:
                smooth_vals.append(1.0)

        axes[3].plot(smooth_vals, 'o-', color='blue')
        axes[3].set_title('Cumulative Smoothness (higher = steadier flow)')
        axes[3].set_ylabel('1 - CV(deltas)')
        axes[3].set_ylim(0, 1)

    for ax in axes:
        ax.set_xlabel('Cycle')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Simple test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("Testing thinking_with_entropy.py")
    print("=" * 60)

    # Test 1: EntropyFlowConfidence forward pass
    print("\n--- Test 1: EntropyFlowConfidence forward pass ---")
    batch_size = 4
    page_size = 64
    num_pages = 5

    confidence_head = EntropyFlowConfidence(page_size=page_size, hidden=128)

    # Create mock pages
    mock_pages = [torch.randn(batch_size, page_size) for _ in range(num_pages)]
    confidence, smoothness = confidence_head(mock_pages)

    print(f"  Input: {num_pages} pages of shape ({batch_size}, {page_size})")
    print(f"  Confidence shape: {confidence.shape}, range: [{confidence.min():.3f}, {confidence.max():.3f}]")
    print(f"  Smoothness shape: {smoothness.shape}, range: [{smoothness.min():.3f}, {smoothness.max():.3f}]")
    assert confidence.shape == (batch_size, 1), f"Expected ({batch_size}, 1), got {confidence.shape}"
    assert smoothness.shape == (batch_size, 1), f"Expected ({batch_size}, 1), got {smoothness.shape}"
    print("  PASSED")

    # Test 2: compute_smoothness_target
    print("\n--- Test 2: compute_smoothness_target ---")

    # Smooth deltas (all equal)
    smooth_deltas = [torch.ones(batch_size) for _ in range(5)]
    smooth_target = compute_smoothness_target(smooth_deltas)
    print(f"  Smooth deltas (all 1.0): smoothness = {smooth_target.mean():.4f} (expected: ~1.0)")
    assert smooth_target.mean() > 0.99, "Equal deltas should give smoothness ~1.0"

    # Choppy deltas (high variance)
    choppy_deltas = [
        torch.ones(batch_size) * 0.5,
        torch.ones(batch_size) * 5.0,
        torch.ones(batch_size) * 0.5,
    ]
    choppy_target = compute_smoothness_target(choppy_deltas)
    print(f"  Choppy deltas (0.5, 5.0, 0.5): smoothness = {choppy_target.mean():.4f} (expected: <0.5)")
    assert choppy_target.mean() < 0.5, "Choppy deltas should give low smoothness"
    print("  PASSED")

    # Test 3: Four-quadrant decision logic
    print("\n--- Test 3: Four-quadrant decision logic ---")

    test_cases = [
        (0.9, 0.8, 'STOP'),       # High conf + smooth
        (0.9, 0.2, 'SUSPICIOUS'),  # High conf + choppy
        (0.5, 0.8, 'CONTINUE'),   # Low conf + smooth
        (0.5, 0.2, 'RETRY'),      # Low conf + choppy
    ]

    for conf, smooth, expected_action in test_cases:
        decision = quadrant_decision(
            confidence=torch.tensor([[conf]]),
            smoothness=torch.tensor([[smooth]]),
        )
        status = "OK" if decision.action == expected_action else "FAIL"
        print(f"  conf={conf:.1f}, smooth={smooth:.1f} -> {decision.action:12s} [{status}]")
        print(f"    {decision.reason}")
        assert decision.action == expected_action, f"Expected {expected_action}, got {decision.action}"

    print("  PASSED")

    # Test 4: Atom perturbation
    print("\n--- Test 4: Atom perturbation ---")
    atom_scales = torch.randn(batch_size, 64).tanh()
    perturbed = perturb_atom_scales(atom_scales, noise_scale=0.2)

    print(f"  Original range: [{atom_scales.min():.3f}, {atom_scales.max():.3f}]")
    print(f"  Perturbed range: [{perturbed.min():.3f}, {perturbed.max():.3f}]")
    print(f"  Mean abs diff: {(perturbed - atom_scales).abs().mean():.4f}")

    # Verify perturbed is still in [-1, 1] (tanh bounded)
    assert perturbed.min() >= -1.0 and perturbed.max() <= 1.0, "Perturbed scales should be in [-1, 1]"
    # Verify perturbation actually changed values
    assert (perturbed - atom_scales).abs().mean() > 0.01, "Perturbation should change values"
    print("  PASSED")

    # Test 5: SurpriseDetector integration
    print("\n--- Test 5: SurpriseDetector integration ---")
    detector = SurpriseDetector()

    # Simulate deltas with one surprising jump
    deltas = [
        torch.ones(batch_size) * 1.0,   # normal
        torch.ones(batch_size) * 1.1,   # normal
        torch.ones(batch_size) * 5.0,   # SURPRISE!
        torch.ones(batch_size) * 1.0,   # back to normal
    ]

    surprises = detector.compute_surprise(deltas)
    print(f"  Cycle 0 surprise: {surprises[0].mean():.4f} (expected: 0.0)")
    print(f"  Cycle 1 surprise: {surprises[1].mean():.4f} (expected: low)")
    print(f"  Cycle 2 surprise: {surprises[2].mean():.4f} (expected: HIGH)")
    print(f"  Cycle 3 surprise: {surprises[3].mean():.4f}")

    assert surprises[0].mean() == 0.0, "First cycle should have 0 surprise"
    assert surprises[2].mean() > surprises[1].mean(), "Jump cycle should have higher surprise"
    print("  PASSED")

    # Test 6: EntropyTracker methods
    print("\n--- Test 6: EntropyTracker methods ---")

    page = torch.randn(batch_size, page_size)
    page_prev = torch.randn(batch_size, page_size)
    atom_scales = torch.randn(batch_size, 64)

    page_ent = EntropyTracker.page_entropy(page)
    delta_norm = EntropyTracker.page_delta_norm(page, page_prev)
    atom_ent = EntropyTracker.atom_entropy(atom_scales)

    print(f"  page_entropy: {page_ent.mean():.4f} (shape: {page_ent.shape})")
    print(f"  page_delta_norm: {delta_norm.mean():.4f} (shape: {delta_norm.shape})")
    print(f"  atom_entropy: {atom_ent.mean():.4f} (shape: {atom_ent.shape})")

    assert page_ent.shape == (batch_size,)
    assert delta_norm.shape == (batch_size,)
    assert atom_ent.shape == (batch_size,)
    print("  PASSED")

    # Test 7: ThinkingResult dataclass
    print("\n--- Test 7: ThinkingResult dataclass ---")
    result = ThinkingResult(
        state_pages=mock_pages,
        atom_scales_history=[torch.randn(batch_size, 64) for _ in range(num_pages)],
        surprises=[torch.zeros(batch_size) for _ in range(num_pages - 1)],
        final_confidence=confidence,
        final_smoothness=smoothness,
        num_passes=num_pages,
        stopped_early=False,
        stop_reason="max_passes",
    )
    print(f"  num_passes: {result.num_passes}")
    print(f"  stopped_early: {result.stopped_early}")
    print(f"  stop_reason: {result.stop_reason}")
    print("  PASSED")

    # Test 8: Gradient flow through confidence head
    print("\n--- Test 8: Gradient flow through confidence head ---")
    grad_pages = [torch.randn(batch_size, page_size, requires_grad=True) for _ in range(3)]
    conf_grad, smooth_grad = confidence_head(grad_pages)
    loss = conf_grad.sum() + smooth_grad.sum()
    loss.backward()

    has_grad = all(p.grad is not None and p.grad.abs().sum() > 0 for p in grad_pages)
    print(f"  All pages have gradients: {has_grad}")
    assert has_grad, "Gradients should flow to input pages"
    print("  PASSED")

    print("\n" + "=" * 60)
    print("All tests PASSED!")
    print("=" * 60)

    # Print summary of four-quadrant logic
    print("\n" + "=" * 60)
    print("FOUR-QUADRANT STOPPING LOGIC SUMMARY")
    print("=" * 60)
    print("""
                    Smooth thinking          Choppy thinking
                    (smoothness > 0.7)       (smoothness < 0.3)
                    ---------------------    ---------------------
High confidence     STOP                     SUSPICIOUS
(conf > 0.85)       Good answer.             Confident but unreliable.
                    Trust it.                Log warning.

Low confidence      CONTINUE                 RETRY
(conf < 0.85)       On track but not done.   Lost the thread.
                    More cycles needed.      Perturb atoms, try again.
""")
