# Morning Handoff: L3 + Dual LoRA Verification

## Where We Are

```
Task              Base Model    Breathing       Key Fix
──────────────────────────────────────────────────────────
L0 single-step    70%           100%            answer loss
L1 two-step       0%            94.8% ±0.0%     contrastive (target-cos 0.7)
L2 three-step     0%            83.4%           target-cosine
L2 word ops       0.6%          53.4%           CoT targets
L3 named qty      ???           ???             NEXT
```

Key discoveries from yesterday:
- Pages were constant (fixed-point collapse) → contrastive loss fixed it → 94.8%
- Pages were copied across passes → pass-conditioned hypernetwork fixed it → p2v3=0.30
- Terse answer targets caused number-spam → CoT targets fixed it → 53.4% on word ops
- High page_cos on easy problems is FINE — similar problems should have similar pages
- The model uses ~1 effective pass for everything so far — multi-pass hasn't been needed yet

---

## Priority 1: L3 Training (Named Quantities)

Train on L3 problems: "Jamie had 56 cookies and gave 2 away. How many are left?"

### Recipe (proven on L2)

```
Data:           20K L3 problems, positive integer answers in [1, 200]
Targets:        CoT format matching base model's natural style:
                "Jamie had 56 cookies. He gave 2 away. 56 - 2 = 54. The answer is 54."
Passes:         Fixed at 3 (same as L2, confidence head comes later)
Loss:           generation_loss + 0.05 * target_cos_contrastive(target=0.7) per page
Architecture:   Pass-conditioned hypernetwork, frozen Llama, additive LoRA
Early stopping: patience=2, save best by eval accuracy
Max epochs:     6 (overfitting starts at epoch 4 typically)
Warm start:     From L2 CoT best checkpoint (53.4%)
```

### What to Monitor
```
accuracy:    target >30% (L3 is harder than L2 — narrative parsing)
page_cos:    will likely be high (0.95+) — that's fine for similar problems
p2v3:        should stay <0.5 (pass conditioning working)
ans_loss:    watch for train/eval divergence (overfitting signal)
generations: print 10 samples — is the model producing CoT reasoning or number-spam?
```

### If L3 Fails (<20%)
Likely cause: frozen Llama can't parse narrative context ("Jamie had 56 cookies and gave 2 away" is harder than "56 minus 2"). Consider unfreezing Llama at 1e-7 for L3 specifically.

---

## Priority 2: Dual LoRA Verification Architecture

After L3 baseline is established, add the verification LoRA. This gives the model a REASON to use multiple passes — not just compute but also check its work.

### Why This Matters Now

The model uses ~1 effective pass for everything. It computes and stops. There's no incentive to think again. The verification LoRA provides that incentive: "I computed 72, let me verify... 48/2=24 ✓, 24+48=72 ✓, confident."

Even on easy problems, verification catches the 2.6% per-step errors:
```
Without verification: 97.4% per-step
With verification:    potentially 99%+ per-step (errors caught before output)
```

### Architecture

Two sets of LoRA templates sharing one hypernetwork. A learned blend weight transitions from computation to verification:

```python
class DualLoRA(nn.Module):
    def __init__(self, d_model=2048, rank=4, num_layers=16):
        # Forward templates (computation-focused attention)
        self.A_forward = nn.ParameterList([...])  # rank 4 per Q,K,V,O
        self.B_forward = nn.ParameterList([...])
        
        # Verification templates (consistency-checking attention)
        self.A_verify = nn.ParameterList([...])   # rank 4 per Q,K,V,O
        self.B_verify = nn.ParameterList([...])
        
        # ~1.1M params each, ~2.2M total (< 0.2% of model)


class BlendedHypernetwork(nn.Module):
    def __init__(self, page_summary_dim, strategy_size, pass_embed_dim, num_scales=256):
        # Outputs: forward_scales (256) + verify_scales (256) + blend (1)
        self.scales_net = nn.Sequential(
            nn.Linear(page_summary_dim + strategy_size + pass_embed_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256 + 256 + 1),
        )
    
    def forward(self, page_summary, strategy, pass_embed):
        out = self.scales_net(torch.cat([page_summary, strategy, pass_embed], dim=-1))
        forward_scales = torch.tanh(out[:, :256])
        verify_scales = torch.tanh(out[:, 256:512])
        blend = torch.sigmoid(out[:, 512:513])  # 0=forward, 1=verify
        return forward_scales, verify_scales, blend
```

### Blended Application (additive, no hooks)

```python
# At each layer, for each projection:
q_forward = (hidden @ A_forward.T) * forward_scales @ B_forward
q_verify = (hidden @ A_verify.T) * verify_scales @ B_verify
q_lora = (1 - blend) * q_forward + blend * q_verify

q = layer.q_proj(hidden) + q_lora
```

### Expected Blend Trajectory

```
Pass 1: blend ≈ 0.1  (mostly computing)
Pass 2: blend ≈ 0.3  (computing + starting to check)
Pass 3: blend ≈ 0.7  (mostly verifying)
Pass 4: blend ≈ 0.9  (pure verification → confidence high → stop)
```

The blend is LEARNED, not hardcoded. The model discovers when to transition from computing to verifying based on what produces correct answers.

### Confidence Head Uses Blend

The confidence head reads pages AND blend history. It learns: "don't stop until verification has happened (blend > 0.5)."

```python
class PageConfidenceHead(nn.Module):
    def __init__(self, page_size=64, hidden=128):
        super().__init__()
        self.page_project = nn.Linear(page_size, hidden)
        self.blend_project = nn.Linear(1, hidden)
        self.attn = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, hidden))
        self.output = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, state_pages, blend_history):
        pages = torch.stack(state_pages, dim=1)
        pages_proj = self.page_project(pages)
        blends = torch.stack(blend_history, dim=1)
        blend_proj = self.blend_project(blends)
        pages_proj = pages_proj + blend_proj
        
        batch_size = pages.size(0)
        q = self.query.unsqueeze(0).expand(batch_size, -1, -1)
        attended, _ = self.attn(query=q, key=pages_proj, value=pages_proj)
        return self.output(attended.squeeze(1))
```

### Training the Confidence Head

No efficiency penalty. Correctness signal only:

```python
# At each pass, generate answer from current pages
# confidence_target = 1.0 if answer is correct, 0.0 if wrong
# The head learns: "stop when you'd get it right"
# It naturally discovers: answers are more likely correct AFTER verification
```

### Dynamic Passes at Inference

```python
min_passes = 1
max_passes = 12
confidence_threshold = 0.85

# Easy problem: compute (1 pass) + verify (1 pass) = 2 passes → stop
# Hard problem: compute (4 passes) + verify (3 passes) = 7 passes → stop
```

---

## Priority 3: L4 and L5 (After Dual LoRA Works)

Once dual LoRA + confidence head is working on L3:

```
L4: Easy 2-step word problems (small numbers, [1-200])
    "A store has 59 cookies. They sell 27 on Monday and 14 on Tuesday. How many are left?"
    Expected: 4-6 passes (parse + compute step 1 + compute step 2 + verify)

L5: Full GSM8K
    Expected: 6-12 passes
    This is the milestone that matters
```

Same recipe at each level: CoT targets, 20K problems, early stopping, warm start from previous level.

---

## Implementation Order

```
Step 1: Generate 20K L3 problems with CoT targets
Step 2: Train L3 with current architecture (establish baseline)
Step 3: Implement DualLoRA + BlendedHypernetwork
Step 4: Implement confidence head with correctness training
Step 5: Retrain L3 with dual LoRA + confidence head
Step 6: Compare: does dual LoRA + dynamic passes beat fixed 3 passes?
Step 7: If yes → L4, L5 with full architecture
```

---

## Key Lessons (Don't Repeat These Mistakes)

```
1. CoT targets, not terse numbers. Train on full reasoning traces.
2. High page_cos on easy problems is FINE. Don't force differentiation.
3. Early stopping at epoch 2-4. Overfitting is the recurring enemy.
4. 20K+ problems per level. Small datasets → memorization.
5. Print 10 generations before trusting accuracy numbers. Format bugs hide everywhere.
6. The model finds shortcuts. Every unconstrained degree of freedom will be exploited.
7. Pass-conditioned hypernetwork prevents page copying. Keep it.
8. Contrastive loss prevents fixed-point collapse. Keep it as insurance.
9. Don't force multi-pass on problems that don't need it.
10. Verify baseline on same eval set with same prompt format (Bug #38).
```

---

## Infrastructure Notes

```
VM:             AWS g5.xlarge (A10G 24GB), currently stopped
SSH:            ssh -i secrets/portugal_key.pem ubuntu@<check AWS console for IP>
Project dir:    ~/mycelium
Always use:     tmux (session loss = context loss)
Logs:           ~/mycelium/logs/
Checkpoints:    ~/mycelium/*.pt
Monitor:        tail -f ~/mycelium/logs/<current_run>.log
```
