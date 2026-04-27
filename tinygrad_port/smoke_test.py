#!/usr/bin/env python3
"""Smoke test all tinygrad port components on CUDA."""
import os
os.environ["CUDA"] = "1"
from tinygrad import Tensor

print("=== Testing Perceiver ===")
from tinygrad_port.perceiver import Perceiver
perc = Perceiver(page_size=64, d_transformer=2048, num_perceiver_layers=7, num_queries=4, use_wavelet=False)
hidden_states = [Tensor.randn(2, 32, 2048) for _ in range(16)]
page_delta = perc(hidden_states, pass_num=0)
print(f"  Page delta: {page_delta.shape} (expected: 2, 64)")

print()
print("=== Testing Losses ===")
from tinygrad_port.losses import answer_head_loss, isotropic_regularizer, per_cycle_target_weight
from tinygrad_port.answer_head import AnswerHead

ah = AnswerHead(page_size=64)
page = Tensor.randn(2, 64)
gold = Tensor([42, 117])
loss = answer_head_loss(ah, page, gold, cycle_num=0)
print(f"  AH loss: {loss.numpy():.4f}")

raw_pages = Tensor.randn(8, 64)
iso = isotropic_regularizer(raw_pages)
print(f"  Iso reg: {iso.numpy():.4f}")

w = per_cycle_target_weight(0.04, 0, 3)
print(f"  Fade weight at 4%: {w:.4f} (should be ~1.0)")
w = per_cycle_target_weight(0.85, 0, 3)
print(f"  Fade weight at 85%: {w:.4f} (should be ~0.2)")

print()
print("=== Testing Data Loading ===")
from tinygrad_port.data import PerCycleDataset, batch_iterator
ds = PerCycleDataset("data/per_cycle/gsm8k_train.jsonl", max_passes=3)
print(f"  Loaded {len(ds)} GSM8K problems")
batch = ds.get_batch([0, 1, 2, 3])
print(f"  Batch keys: {list(batch.keys())}")
ct = batch["cycle_targets"]
print(f"  cycle_targets shape: {ct.shape}")
print(f"  Problem 1: {batch['problems'][0][:60]}...")

print()
print("=== Backward Pass Test ===")
from tinygrad_port.hypernetwork import AtomHypernetwork
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters

hyper = AtomHypernetwork(page_size=64, num_atoms=64)
params = get_parameters(hyper)
opt = AdamW(params, lr=1e-3)

pages = [Tensor.randn(2, 64) for _ in range(3)]
scales = hyper(pages, pass_num=1)
loss = scales.square().mean()
opt.zero_grad()
loss.backward()
opt.step()
print(f"  Backward + step OK, loss={loss.numpy():.6f}")

grad_norms = [p.grad.square().sum().sqrt().numpy() if p.grad is not None else 0.0 for p in params[:5]]
print(f"  First 5 grad norms: {grad_norms}")

print()
print("=" * 60)
print("ALL TINYGRAD SMOKE TESTS PASSED ✓")
print("=" * 60)
