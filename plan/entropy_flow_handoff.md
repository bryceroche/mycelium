# Handoff: Entropy Flow + Surprise Detection + Smoothness-Aware Confidence

## One-Sentence Summary

Track the entropy of thinking across cycles. Measure surprise (unexpected entropy drops) per cycle. Add a smoothness signal to the confidence head — the model knows not just "am I confident?" but "was my thinking smooth?" Choppy thinking with high confidence is suspicious. Smooth thinking with low confidence just needs more cycles.

---

## The Core Idea

Electricity follows the path of least resistance. Good thinking follows the path of smoothest entropy reduction. Each cycle should reduce uncertainty by a consistent amount — steady flow toward the answer, not sparks and shorts.

```
Good thinking (smooth current):
  Uncertainty: 100% → 80% → 60% → 40% → 20% → stop
  Entropy drops evenly. Each cycle does its fair share.

Bad thinking (resistance, sparks):
  Uncertainty: 100% → 95% → 30% → 80% → 15% → stop???
  Cycle 3 did too much. Cycle 4 regressed. Choppy flow.
  Might get the right answer. Might not. Don't trust it.
```

---

## Three Components

### 1. Per-Cycle Entropy Measurement

Track the information content of each cycle's output relative to the previous cycle.

```python
class EntropyTracker:
    """Tracks entropy flow across thinking cycles."""
    
    @staticmethod
    def page_entropy(page):
        """
        Estimate entropy of a page using its value distribution.
        Higher entropy = more information = more uncertainty.
        """
        # Normalize page to a probability-like distribution
        probs = F.softmax(page.abs(), dim=-1)  # (batch, 64)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)  # (batch,)
        return entropy
    
    @staticmethod
    def page_delta_norm(page_current, page_previous):
        """
        How much did the page change? Large change = lots of new information.
        """
        return (page_current - page_previous).norm(dim=-1)  # (batch,)
    
    @staticmethod
    def atom_entropy(atom_scales):
        """
        Entropy of atom activation pattern.
        High entropy = many atoms active (complex thinking).
        Low entropy = few atoms active (simple thinking).
        """
        probs = F.softmax(atom_scales.abs(), dim=-1)  # (batch, 64)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)  # (batch,)
        return entropy
```

### 2. Surprise Detection

Surprise = how much a cycle's entropy drop deviates from the running average. High surprise means the cycle encountered something unexpected.

```python
class SurpriseDetector:
    """Detects unexpected entropy changes per cycle."""
    
    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.running_delta = None
        self.running_var = None
    
    def compute_surprise(self, page_deltas):
        """
        page_deltas: list of (batch,) tensors — one per cycle
        returns: list of (batch,) surprise scores
        """
        surprises = []
        
        for i, delta in enumerate(page_deltas):
            if i == 0:
                # First cycle — no expectation yet
                surprises.append(torch.zeros_like(delta))
                self.running_delta = delta.mean().item()
                self.running_var = delta.var().item() + 1e-8
            else:
                # Surprise = |actual - expected| / std
                expected = self.running_delta
                std = math.sqrt(self.running_var)
                surprise = ((delta - expected).abs() / (std + 1e-8))
                surprises.append(surprise)
                
                # Update running stats
                current_mean = delta.mean().item()
                current_var = delta.var().item()
                self.running_delta = (self.momentum * self.running_delta 
                                      + (1 - self.momentum) * current_mean)
                self.running_var = (self.momentum * self.running_var 
                                    + (1 - self.momentum) * current_var)
        
        return surprises
    
    @staticmethod
    def compute_surprise_batch(page_deltas):
        """
        Simpler batch version — surprise relative to mean of previous deltas.
        No running stats needed.
        """
        surprises = []
        for i, delta in enumerate(page_deltas):
            if i == 0:
                surprises.append(torch.zeros_like(delta))
            else:
                # Expected = mean of all previous deltas
                prev_deltas = torch.stack(page_deltas[:i], dim=0)  # (i, batch)
                expected = prev_deltas.mean(dim=0)  # (batch,)
                std = prev_deltas.std(dim=0).clamp(min=1e-8)
                surprise = (delta - expected).abs() / std
                surprises.append(surprise)
        
        return surprises
```

### 3. Smoothness-Aware Confidence Head

The confidence head now tracks the FLOW of pages using a GRU, outputting both confidence and smoothness.

```python
class EntropyFlowConfidence(nn.Module):
    """
    Confidence head that tracks entropy flow across thinking cycles.
    Outputs: confidence (am I done?) and smoothness (was my thinking steady?).
    """
    def __init__(self, page_size=64, hidden=128):
        super().__init__()
        
        # Project pages to hidden dim
        self.page_project = nn.Linear(page_size, hidden)
        
        # GRU tracks the DYNAMICS of page changes
        # Learns to detect smooth vs choppy entropy reduction
        self.flow_gru = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        
        # Output: confidence + smoothness
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        self.smoothness_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, state_pages):
        """
        state_pages: list of (batch, 64) tensors
        returns: confidence (batch, 1), smoothness (batch, 1)
        """
        pages = torch.stack(state_pages, dim=1)  # (batch, num_pages, 64)
        pages_proj = self.page_project(pages)     # (batch, num_pages, hidden)
        
        # GRU processes pages sequentially — learns flow dynamics
        flow_output, _ = self.flow_gru(pages_proj)  # (batch, num_pages, hidden)
        last_flow = flow_output[:, -1, :]            # (batch, hidden)
        
        confidence = self.confidence_head(last_flow)   # (batch, 1)
        smoothness = self.smoothness_head(last_flow)   # (batch, 1)
        
        return confidence, smoothness
```

---

## Integration Into Thinking Loop

```python
def think_with_entropy_tracking(self, problem, max_passes=12, 
                                  conf_threshold=0.85, smooth_threshold=0.7,
                                  surprise_threshold=2.0, max_retries=2):
    state_pages = []
    atom_scales_history = []
    page_deltas = []
    surprise_detector = SurpriseDetector()
    
    for pass_num in range(max_passes):
        # Generate atom scales and think
        atom_scales = self.hypernetwork(state_pages, pass_num)
        page = self.think_one_pass(problem, state_pages, atom_scales, pass_num)
        
        # Track page delta (information change)
        if len(state_pages) > 0:
            delta = (page - state_pages[-1]).norm(dim=-1)
        else:
            delta = page.norm(dim=-1)
        page_deltas.append(delta)
        
        # Compute surprise for this cycle
        surprises = surprise_detector.compute_surprise(page_deltas)
        current_surprise = surprises[-1]
        
        # If surprise is high — this cycle hit resistance
        # Retry with different atoms (perturbation)
        retry_count = 0
        while current_surprise.mean() > surprise_threshold and retry_count < max_retries:
            # Perturb atom scales and retry
            noise = torch.randn_like(atom_scales) * 0.2
            atom_scales_retry = torch.tanh(atom_scales + noise)
            page_retry = self.think_one_pass(problem, state_pages, atom_scales_retry, pass_num)
            
            # Check if retry is smoother
            delta_retry = (page_retry - state_pages[-1]).norm(dim=-1) if state_pages else page_retry.norm(dim=-1)
            surprises_retry = surprise_detector.compute_surprise(page_deltas[:-1] + [delta_retry])
            
            if surprises_retry[-1].mean() < current_surprise.mean():
                # Retry was smoother — accept it
                page = page_retry
                atom_scales = atom_scales_retry
                page_deltas[-1] = delta_retry
                current_surprise = surprises_retry[-1]
            
            retry_count += 1
        
        state_pages.append(page)
        atom_scales_history.append(atom_scales)
        
        # Check confidence AND smoothness
        if pass_num >= 1:  # need at least 2 pages for flow detection
            confidence, smoothness = self.confidence_head(state_pages)
            
            if confidence > conf_threshold and smoothness > smooth_threshold:
                break  # confident AND smooth — good answer
            
            # Log suspicious states
            if confidence > conf_threshold and smoothness < 0.3:
                print(f"WARNING: confident ({confidence:.2f}) but choppy ({smoothness:.2f})")
    
    return state_pages, atom_scales_history, surprises
```

---

## Training the Smoothness Head

The smoothness head is trained alongside the confidence head. The smoothness target measures how evenly the page deltas are distributed:

```python
def compute_smoothness_target(page_deltas):
    """
    Smoothness = 1 - coefficient of variation of page deltas.
    Even deltas (smooth flow) → high smoothness.
    Uneven deltas (choppy flow) → low smoothness.
    """
    deltas = torch.stack(page_deltas, dim=1)  # (batch, num_passes)
    mean_delta = deltas.mean(dim=1, keepdim=True).clamp(min=1e-8)
    std_delta = deltas.std(dim=1, keepdim=True)
    
    cv = std_delta / mean_delta  # coefficient of variation
    smoothness = torch.clamp(1.0 - cv, 0.0, 1.0)  # (batch, 1)
    return smoothness


def train_confidence_and_smoothness(model, problem, gold_answer, max_passes=5):
    state_pages = []
    page_deltas = []
    conf_loss = 0.0
    smooth_loss = 0.0
    
    for pass_num in range(max_passes):
        page = model.think_one_pass(problem, state_pages, pass_num)
        
        # Track delta
        if state_pages:
            delta = (page - state_pages[-1]).norm(dim=-1)
        else:
            delta = page.norm(dim=-1)
        page_deltas.append(delta)
        
        state_pages.append(page)
        
        if pass_num >= 1:
            # Confidence target: would current pages produce right answer?
            with torch.no_grad():
                answer = model.generate_from_pages(state_pages, problem)
                is_correct = float(extract_number(answer) == gold_answer)
            
            # Smoothness target: how even are the page deltas so far?
            smooth_target = compute_smoothness_target(page_deltas)
            
            # Predict both
            pred_conf, pred_smooth = model.confidence_head(state_pages)
            
            conf_loss += F.binary_cross_entropy(pred_conf, torch.tensor([[is_correct]]))
            smooth_loss += F.mse_loss(pred_smooth, smooth_target)
    
    return (conf_loss + smooth_loss) / max_passes
```

---

## The Four Decision Quadrants

```
                    Smooth thinking          Choppy thinking
                    ─────────────────────    ─────────────────────
High confidence     STOP — good answer       SUSPICIOUS — might be
                    Smooth flow led to       confidently wrong.
                    a reliable conclusion.   Log warning. Optionally
                    Trust it.                retry with different atoms.

Low confidence      CONTINUE — on track      RETRY — lost the thread.
                    but not done yet.        Load from last save point
                    Smooth progress,         (page cache). Try different
                    just needs more cycles.  atom configuration. If
                                             retries exhausted, continue
                                             with best attempt.
```

---

## Diagnostics: Entropy Flow Visualization

```python
def visualize_entropy_flow(state_pages, atom_scales_history):
    """
    Plot the entropy flow across cycles for a single problem.
    Shows: page deltas, atom entropy, surprise, smoothness.
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    
    # 1. Page deltas (information change per cycle)
    deltas = []
    for i in range(1, len(state_pages)):
        delta = (state_pages[i] - state_pages[i-1]).norm().item()
        deltas.append(delta)
    axes[0].bar(range(len(deltas)), deltas)
    axes[0].set_title('Page Delta Per Cycle (information change)')
    axes[0].set_ylabel('||page_k - page_{k-1}||')
    
    # 2. Atom entropy (complexity of thinking)
    atom_ents = []
    for scales in atom_scales_history:
        probs = F.softmax(scales.abs(), dim=-1)
        ent = -(probs * torch.log(probs + 1e-8)).sum().item()
        atom_ents.append(ent)
    axes[1].plot(atom_ents, 'o-')
    axes[1].set_title('Atom Entropy Per Cycle (thinking complexity)')
    axes[1].set_ylabel('H(atom_scales)')
    
    # 3. Surprise (deviation from expected delta)
    surprises = SurpriseDetector.compute_surprise_batch(
        [torch.tensor([d]) for d in deltas]
    )
    surprise_vals = [s.item() for s in surprises]
    axes[2].bar(range(len(surprise_vals)), surprise_vals, 
                color=['red' if s > 2.0 else 'green' for s in surprise_vals])
    axes[2].axhline(y=2.0, color='red', linestyle='--', label='threshold')
    axes[2].set_title('Surprise Per Cycle (red = unexpected)')
    axes[2].set_ylabel('|actual - expected| / std')
    
    # 4. Cumulative smoothness
    smooth_vals = []
    for i in range(1, len(deltas) + 1):
        partial_deltas = [torch.tensor([d]) for d in deltas[:i]]
        stacked = torch.tensor(deltas[:i])
        cv = stacked.std() / (stacked.mean() + 1e-8)
        smooth_vals.append(max(0, 1 - cv.item()))
    axes[3].plot(smooth_vals, 'o-', color='blue')
    axes[3].set_title('Cumulative Smoothness (higher = steadier flow)')
    axes[3].set_ylabel('1 - CV(deltas)')
    axes[3].set_ylim(0, 1)
    
    for ax in axes:
        ax.set_xlabel('Cycle')
    
    plt.tight_layout()
    plt.savefig('entropy_flow.png', dpi=150)
    print("Saved entropy_flow.png")
```

---

## Integration Into Loss

```python
total_loss = (generation_loss                       # main signal
              + 0.05 * contrastive_loss             # prevents page collapse
              + 0.3 * answer_head_loss              # digit extraction
              + 0.1 * confidence_loss               # when to stop
              + 0.1 * smoothness_loss)              # quality of thinking flow
```

The smoothness loss teaches the model that smooth thinking is BETTER thinking. Over time, the model learns to produce even entropy drops — steady progress toward the answer, not chaotic jumps.

---

## Parameter Cost

```
EntropyTracker:     0 params (pure computation)
SurpriseDetector:   0 params (running statistics)
GRU flow detector:  ~200K params (2-layer GRU, hidden=128)
Smoothness head:    ~8K params (Linear(128, 64) + Linear(64, 1))
Confidence head:    ~8K params (same structure)

Total new: ~216K (negligible)
```

---

## Connection to Other Components

```
Surprise detection → triggers MCMC retry (inference-time)
Smoothness signal → complements confidence for stopping decision
Entropy flow → diagnostic for post-training analysis
Page cache → surprise tells us WHEN to use the save point
Atom spectrogram → surprise correlates with unusual atom patterns
```

The entropy flow is the unifying measurement. It connects the confidence head, the MCMC retry mechanism, the page cache save points, and the atom activation patterns into one coherent framework: the model should think smoothly, and when it doesn't, something is wrong.

---

## What NOT to Do

```
- Do NOT use surprise to reject pages during TRAINING. Training should always accept.
  Surprise-based retry is for INFERENCE only. Training uses standard gradient.
  
- Do NOT make the smoothness loss too strong. 0.1 weight is a starting point.
  Too strong → model learns to make even deltas regardless of correctness.
  The generation loss must still dominate.

- Do NOT threshold surprise too aggressively during inference.
  threshold=2.0 (2 standard deviations) is generous. Start there.
  Lower threshold = more retries = slower but more robust.

- Do NOT use the GRU confidence head until it's properly trained.
  Train on L3 first where we have ground truth, then deploy on GSM8K.
```
