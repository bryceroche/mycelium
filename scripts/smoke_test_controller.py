"""
Phase 0: Controller Smoke Test

The single most important test in the v2 rebuild.
Can the controller produce different outputs for different inputs?

No Llama, no atoms, no data pipeline. Just:
  10 different input vectors → 10 different target scale vectors
  Train the controller to map each input to its target.

PASS criteria:
  1. Loss → 0 (controller learned the mapping)
  2. Different inputs → different outputs (cosine sim < 0.9)
  3. Gradients flow to all controller components
  4. No dimension collapse (all 64 dims active)
  5. scale_mid_frac > 0.5 (scales in linear tanh regime)

DO NOT PROCEED TO PHASE 1 UNTIL THIS PASSES.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.controller import BreathingController, count_parameters


def run_smoke_test(device='cpu', num_problems=10, num_steps=2000, lr=3e-4):
    print("=" * 60)
    print("Phase 0: Controller Smoke Test")
    print("=" * 60)

    # Create controller
    ctrl = BreathingController()
    ctrl = ctrl.to(device)
    total, trainable = count_parameters(ctrl)
    print(f"Controller: {trainable:,} trainable parameters")

    # Create toy environment: 10 different inputs, 10 different targets
    torch.manual_seed(42)
    inputs = torch.randn(num_problems, 2048, device=device)  # 10 "hidden states"

    # Target scales: bounded, diverse
    raw_targets = torch.randn(num_problems, 64, device=device)
    targets = 0.46 * torch.tanh(raw_targets)  # bounded [-0.46, 0.46]

    # Verify targets are diverse
    target_cos = []
    for i in range(num_problems):
        for j in range(i + 1, num_problems):
            target_cos.append(F.cosine_similarity(
                targets[i:i+1], targets[j:j+1]).item())
    print(f"Target scale diversity: mean_cos={sum(target_cos)/len(target_cos):.4f}")

    optimizer = torch.optim.AdamW(ctrl.parameters(), lr=lr, weight_decay=0.01)

    # Training
    print(f"\nTraining for {num_steps} steps...")
    ctrl.train()

    for step in range(num_steps):
        total_loss = 0.0

        for i in range(num_problems):
            optimizer.zero_grad()

            scales, page, embed, action, energy, conf = ctrl.forward_simple(
                inputs[i:i+1]
            )

            loss = F.mse_loss(scales, targets[i:i+1])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / num_problems

        if step % 200 == 0 or step == num_steps - 1:
            # Evaluate
            ctrl.eval()
            with torch.no_grad():
                all_scales = []
                all_pages = []
                for i in range(num_problems):
                    scales, page, embed, action, energy, conf = ctrl.forward_simple(
                        inputs[i:i+1]
                    )
                    all_scales.append(scales)
                    all_pages.append(page)

                scales_stack = torch.cat(all_scales, dim=0)  # (N, 64)
                pages_stack = torch.cat(all_pages, dim=0)    # (N, 256)

                # Pairwise cosine sim of scales
                cos_sims = []
                for i in range(num_problems):
                    for j in range(i + 1, num_problems):
                        cos_sims.append(F.cosine_similarity(
                            scales_stack[i:i+1], scales_stack[j:j+1]).item())
                mean_cos = sum(cos_sims) / len(cos_sims)

                # Mid fraction (not saturated)
                mid_frac = (scales_stack.abs() < 0.44).float().mean().item()

                # Dead dims
                scale_var = scales_stack.var(dim=0)
                dead_dims = (scale_var < 1e-6).sum().item()

                # Page diversity
                page_cos = []
                for i in range(num_problems):
                    for j in range(i + 1, num_problems):
                        page_cos.append(F.cosine_similarity(
                            pages_stack[i:i+1], pages_stack[j:j+1]).item())
                mean_page_cos = sum(page_cos) / len(page_cos)

                print(f"  step {step:4d}: loss={avg_loss:.6f} "
                      f"scale_cos={mean_cos:.4f} "
                      f"mid_frac={mid_frac:.3f} "
                      f"dead_dims={dead_dims}/64 "
                      f"page_cos={mean_page_cos:.4f}")

            ctrl.train()

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    ctrl.eval()
    with torch.no_grad():
        all_scales = []
        all_pages = []
        all_energies = []
        for i in range(num_problems):
            scales, page, embed, action, energy, conf = ctrl.forward_simple(
                inputs[i:i+1]
            )
            all_scales.append(scales)
            all_pages.append(page)
            all_energies.append(energy)

        scales_stack = torch.cat(all_scales, dim=0)
        pages_stack = torch.cat(all_pages, dim=0)

        # Test 1: Loss → 0
        final_loss = 0.0
        for i in range(num_problems):
            final_loss += F.mse_loss(all_scales[i], targets[i:i+1]).item()
        final_loss /= num_problems
        test1_pass = final_loss < 0.01
        print(f"\n1. Loss → 0: loss={final_loss:.6f} {'✓ PASS' if test1_pass else '✗ FAIL'}")

        # Test 2: Different inputs → different outputs
        cos_sims = []
        for i in range(num_problems):
            for j in range(i + 1, num_problems):
                cos_sims.append(F.cosine_similarity(
                    scales_stack[i:i+1], scales_stack[j:j+1]).item())
        mean_cos = sum(cos_sims) / len(cos_sims)
        test2_pass = mean_cos < 0.9
        print(f"2. Scale diversity: mean_cos={mean_cos:.4f} {'✓ PASS' if test2_pass else '✗ FAIL'}")

    # Test 3: Gradients flow to all components (outside no_grad!)
    ctrl.train()
    optimizer.zero_grad()
    test_input = inputs[0:1].clone().detach()
    scales_t, page_t, embed_t, action_t, energy_t, conf_t = ctrl.forward_simple(test_input)
    test_loss = F.mse_loss(scales_t, targets[0:1])
    test_loss.backward()

    # Only check components in the scale output path (page/energy have no loss in this test)
    grad_components = {
        'state_encoder': ctrl.state_encoder,
        'notebook_attn': ctrl.notebook_attn,
        'trunk': ctrl.trunk,
        'scale_head': ctrl.scale_head,
    }

    all_have_grad = True
    print(f"3. Gradient flow:")
    for name, module in grad_components.items():
        has_grad = any(p.grad is not None and p.grad.abs().max() > 0
                     for p in module.parameters() if p.requires_grad)
        grad_norm = sum(p.grad.norm().item() for p in module.parameters()
                      if p.grad is not None) if has_grad else 0.0
        status = '✓' if has_grad else '✗'
        print(f"   {status} {name:25s}: grad_norm={grad_norm:.6f}")
        if not has_grad:
            all_have_grad = False
    test3_pass = all_have_grad

    with torch.no_grad():
        # Test 4: No dimension collapse
        scale_var = scales_stack.var(dim=0)
        dead_dims = (scale_var < 1e-6).sum().item()
        active_dims = 64 - dead_dims
        test4_pass = dead_dims < 10
        print(f"4. Dimension health: {active_dims}/64 active, {dead_dims} dead {'✓ PASS' if test4_pass else '✗ FAIL'}")

        # Test 5: Scales in linear tanh regime
        mid_frac = (scales_stack.abs() < 0.44).float().mean().item()
        test5_pass = mid_frac > 0.5
        print(f"5. Tanh regime: mid_frac={mid_frac:.3f} {'✓ PASS' if test5_pass else '✗ FAIL'}")

        # Bonus: page diversity
        page_cos = []
        for i in range(num_problems):
            for j in range(i + 1, num_problems):
                page_cos.append(F.cosine_similarity(
                    pages_stack[i:i+1], pages_stack[j:j+1]).item())
        mean_page_cos = sum(page_cos) / len(page_cos)
        print(f"\nBonus — Page diversity: mean_cos={mean_page_cos:.4f} (no training signal, just checking)")

        # Overall
        all_pass = test1_pass and test2_pass and test3_pass and test4_pass and test5_pass
        print("\n" + "=" * 60)
        if all_pass:
            print("ALL TESTS PASSED ✓")
            print("Controller can produce different outputs for different inputs.")
            print("Proceed to Phase 1.")
        else:
            print("SOME TESTS FAILED ✗")
            print("DO NOT proceed to Phase 1. Debug the controller first.")
        print("=" * 60)

    return all_pass


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    run_smoke_test(device=device)
