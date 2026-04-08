"""Tiny smoke test for the hybrid pipeline. 1 train batch + 1 eval batch."""
import sys
sys.path.insert(0, '/home/ubuntu/mycelium')
import torch
from scripts.train_gsm8k_hybrid import HybridThinkingModel, GSM8KDataset, collate, evaluate

device = torch.device('cuda')
model = HybridThinkingModel()
model.compressor = model.compressor.to(device)
model.lora = model.lora.to(device)
model.pseudo_head = model.pseudo_head.to(device)

train_ds = GSM8KDataset(split='train', max_samples=8)
eval_ds = GSM8KDataset(split='test', max_samples=8)

# 1 train step
batch = collate([train_ds[i] for i in range(4)])
loss = model.forward_train(batch['question'], batch['completion'], num_passes=3)
print(f"Train loss (untrained): {loss.item():.4f}")
loss.backward()
print("Backward OK")

# 1 eval pass
acc = evaluate(model, eval_ds, device, num_passes=3)
print(f"Eval accuracy on 8 samples (untrained): {acc:.1f}%")
print("SMOKE TEST PASSED")
