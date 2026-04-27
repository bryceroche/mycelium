#!/usr/bin/env python3
"""Test LoRA injection into Llama forward pass in tinygrad."""
import os
os.environ["CUDA"] = "1"
from tinygrad import Tensor
Tensor.training = True

print("=== Testing LoRA + Llama Integration ===")
print()

# Use a TINY Llama config for testing (not real weights)
from tinygrad_port.llama import Llama
from tinygrad_port.lora import LoRAAtoms

# Small config for fast testing
DIM = 256
N_LAYERS = 2
N_HEADS = 4
N_KV_HEADS = 2
VOCAB = 1000
HIDDEN = 512

print("Creating small Llama (2 layers, 256 dim)...")
llama = Llama(dim=DIM, n_layers=N_LAYERS, n_heads=N_HEADS,
              n_kv_heads=N_KV_HEADS, vocab_size=VOCAB,
              hidden_dim=HIDDEN, max_seq_len=128)

print("Creating LoRA atoms (64 atoms, rank 6)...")
atoms = LoRAAtoms(num_atoms=64, rank=6, num_layers=N_LAYERS,
                  d_model=DIM, d_kv=N_KV_HEADS * (DIM // N_HEADS))

# Test 1: Forward WITHOUT LoRA
print()
print("Test 1: Forward WITHOUT LoRA")
tokens = Tensor([[1, 2, 3, 4, 5]])  # batch=1, seq=5
logits_base, hidden_base = llama(tokens, output_hidden_states=True)
print(f"  Logits shape: {logits_base.shape} (expected: 1, 5, {VOCAB})")
print(f"  Hidden states: {len(hidden_base)} layers")

# Test 2: Forward WITH LoRA
print()
print("Test 2: Forward WITH LoRA")
atom_scales = Tensor.randn(1, 64) * 0.1  # small scales
logits_lora, hidden_lora = llama(tokens, output_hidden_states=True,
                                  lora_atoms=atoms, atom_scales=atom_scales)
print(f"  Logits shape: {logits_lora.shape}")
print(f"  Hidden states: {len(hidden_lora)} layers")

# Test 3: LoRA changes the output
print()
print("Test 3: LoRA changes the output")
diff = (logits_lora - logits_base).abs().mean().numpy()
print(f"  Mean logit difference: {diff:.6f} (should be > 0)")
assert diff > 0, "LoRA should change the output!"
print(f"  LoRA IS modifying attention ✓")

# Test 4: Backward through LoRA
print()
print("Test 4: Backward through LoRA")
from tinygrad.nn.state import get_parameters

loss = logits_lora.sum()
loss.backward()

atom_params = get_parameters(atoms)
with_grad = sum(1 for p in atom_params if p.grad is not None)
print(f"  Atom params with grad: {with_grad}/{len(atom_params)}")
assert with_grad > 0, "LoRA atoms should get gradients!"
print(f"  Gradients flow through LoRA → Llama → loss ✓")

# Test 5: Different scales → different outputs
print()
print("Test 5: Different scales → different outputs")
scales_a = Tensor.randn(1, 64) * 0.3
scales_b = Tensor.randn(1, 64) * 0.3
logits_a, _ = llama(tokens, lora_atoms=atoms, atom_scales=scales_a)
logits_b, _ = llama(tokens, lora_atoms=atoms, atom_scales=scales_b)
diff_ab = (logits_a - logits_b).abs().mean().numpy()
print(f"  Logit difference between scale configs: {diff_ab:.6f}")
assert diff_ab > 0, "Different scales should produce different outputs!"
print(f"  Atom scales differentiate attention ✓")

print()
print("=" * 60)
print("LORA + LLAMA INTEGRATION: ALL TESTS PASSED ✓")
print("=" * 60)
