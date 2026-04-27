# Handoff: Split Generation + Base KV Cache

## One-Sentence Summary

Split each cycle's generation into two phases — atoms-ON for equation SETUP ("48 / 2 =") and atoms-OFF for arithmetic COMPUTE ("24") — using a cached base KV to make the split FASTER than the current single-pass approach while producing correct arithmetic.

---

## The Problem: Atoms Help Parsing But Corrupt Arithmetic

The diagnostic proved it:

```
Vanilla Llama arithmetic:  9/12 correct ("12 * 20 = 240" ✓)
Atom-modified arithmetic:  "12 * 20 = 195" ✗ (even at gentle 0.4 strength)

Category breakdown at 20.3% accuracy:
  32% WRONG ARITHMETIC — right operation, wrong answer
  24% wrong step — correct math, different part of problem
  14% correct
  20% other errors
  10% copying

The #1 failure is arithmetic corruption by atoms.
The atoms STEER correctly but INTERFERE with computation.
```

The atoms transform Llama from a multiple-choice bot (1.9%) into a math solver (20.3%). But they corrupt the actual arithmetic. The parsing needs atoms. The arithmetic needs vanilla Llama.

## The Solution: Split Generation

Each cycle generates in two phases:

```
PHASE A — SETUP (atoms ON):
  Llama reads problem with atom-modified attention.
  Atoms steer: "this is a division problem, focus on 48 and half"
  Generates the equation setup: "She sold half as many. 48 / 2 ="
  STOP at the "=" sign.

PHASE B — COMPUTE (atoms OFF):
  Continue generation WITHOUT atoms.
  Vanilla Llama sees "48 / 2 =" and completes: "24"
  Llama's arithmetic circuits run undisturbed.
  Complete: "24 #### 24</s>"

Combined: "She sold half as many. 48 / 2 = 24 #### 24</s>"
```

Atoms write the equation. Vanilla Llama solves it. No interference.

```
BEFORE (atoms on for everything):
  Atoms help:  find "48" and "half" in the text        ✓ (parsing)
  Atoms hurt:  corrupt "48 / 2 = ???" computation      ✗ (arithmetic)
  Result:      "48 / 2 = 28" (wrong)

AFTER (split):
  Phase A:     atoms find "48" and "half" → write "48 / 2 ="   ✓ (parsing)
  Phase B:     vanilla Llama computes "48 / 2 = 24"            ✓ (arithmetic)
  Result:      "48 / 2 = 24" (correct!)
```

---

## Base KV Cache: Making It FAST

The naive split doubles the number of forward passes. But with KV caching, the split is actually FASTER than the current approach.

### What's Cacheable

The problem text tokens are the SAME across all passes. Their base KV (without LoRA) is computed once and reused:

```python
# The key insight: separate BASE from LORA in K,V computation
# 
# With atoms:     K = W_k @ x + scale * (x @ B.T) @ A.T
#                 K = BASE_K  + LORA_DELTA_K
#
# Without atoms:  K = W_k @ x
#                 K = BASE_K
#
# BASE_K is IDENTICAL in both cases.
# Cache it once. Reuse everywhere.
```

### Three Types of KV Cache

```
1. BASE KV CACHE (computed once, reused everywhere):
   K_base = W_k @ x for all problem text tokens
   V_base = W_v @ x for all problem text tokens
   Computed at cycle 0 (vanilla comprehension pass).
   Reused by ALL subsequent passes.
   
2. LORA DELTA (cheap to compute per cycle):
   K_delta = scale * (x @ B.T) @ A.T
   This is rank-6 — MUCH cheaper than the full W_k @ x (rank-2048).
   Only needed for setup phases (atoms ON).

3. NEW TOKEN KV (computed fresh for generated tokens):
   Only the text injection tokens and new generated tokens
   need fresh KV computation. ~15-20 tokens per cycle.
```

### Implementation

```python
class KVCache:
    """
    Stores base KV for problem text tokens.
    Supports adding LoRA deltas and extending with new tokens.
    """
    def __init__(self):
        self.base_k = {}  # layer_idx → (batch, num_heads, seq_len, head_dim)
        self.base_v = {}
        self.problem_seq_len = 0
    
    def compute_and_cache_base(self, model, problem_ids):
        """
        Run vanilla Llama on problem text. Cache all base K,V.
        Called ONCE per problem.
        """
        with torch.no_grad():  # base KV doesn't need grad
            for layer_idx, layer in enumerate(model.llama.layers):
                x = layer.input  # input to this layer
                self.base_k[layer_idx] = layer.self_attn.k_proj(x)
                self.base_v[layer_idx] = layer.self_attn.v_proj(x)
        
        self.problem_seq_len = problem_ids.size(1)
    
    def get_kv_with_lora(self, layer_idx, atom_scales, lora_atoms):
        """
        Get K,V with LoRA modification. Reuses cached base.
        Only computes the LoRA DELTA (rank-6, cheap).
        """
        base_k = self.base_k[layer_idx]
        base_v = self.base_v[layer_idx]
        
        # Compute only the LoRA delta (rank-6, not rank-2048)
        delta_k = compute_lora_delta(base_k, atom_scales, lora_atoms, 'k')
        delta_v = compute_lora_delta(base_v, atom_scales, lora_atoms, 'v')
        
        return base_k + delta_k, base_v + delta_v
    
    def get_kv_vanilla(self, layer_idx):
        """
        Get K,V without any LoRA. Just the cached base.
        FREE — no computation needed.
        """
        return self.base_k[layer_idx], self.base_v[layer_idx]
```

### The Full Cycle with Caching

```python
def generate_split_cycle(model, problem_ids, kv_cache, atom_scales, 
                          text_injection, tokenizer):
    """
    One breathing cycle with split generation + KV cache.
    
    Phase A: atoms ON, generate equation setup (stop at "=")
    Phase B: atoms OFF, complete the arithmetic
    
    Uses cached base KV for problem text tokens.
    """
    
    # Prepare full input: text_injection + problem_text
    full_input = tokenize(text_injection + problem_text)
    new_tokens = full_input[kv_cache.problem_seq_len:]  # only injection tokens are new
    
    # === PHASE A: SETUP (atoms ON) ===
    # Problem text tokens: use cached base KV + LoRA delta (cheap)
    # New tokens (text injection): compute full KV
    
    setup_tokens = []
    for step in range(max_setup_tokens):
        logits = model.llama.forward_one_token(
            kv_cache=kv_cache,
            atom_scales=atom_scales,  # atoms ON
            use_lora=True,
        )
        
        next_token = logits.argmax(-1)
        setup_tokens.append(next_token)
        
        # Stop when we generate "="
        if next_token == tokenizer.encode("=")[0]:
            break
    
    # === PHASE B: COMPUTE (atoms OFF) ===
    # Continue from where setup left off
    # Use vanilla KV cache for problem tokens (FREE)
    # No LoRA delta — Llama computes naturally
    
    compute_tokens = []
    for step in range(max_compute_tokens):
        logits = model.llama.forward_one_token(
            kv_cache=kv_cache,
            atom_scales=None,  # atoms OFF — vanilla Llama
            use_lora=False,
        )
        
        next_token = logits.argmax(-1)
        compute_tokens.append(next_token)
        
        # Stop at EOS
        if next_token == tokenizer.eos_token_id:
            break
    
    # Combine: setup + compute
    full_generation = setup_tokens + compute_tokens
    # "She sold half as many. 48 / 2 = " + "24 #### 24</s>"
    
    return full_generation
```

### Cost Analysis

```
CURRENT APPROACH (no split, no cache):
  Two-pass cycle 1:     2 full Llama forwards
  Cycle 1 generation:   1 full forward  
  Cycle 2 generation:   1 full forward
  Cycle 3 generation:   1 full forward
  ─────────────────────────────────────
  Total for 3 steps:    5 full forwards = 5.0 cost units

SPLIT WITH BASE KV CACHE:
  Cycle 0 vanilla:      1 full forward (compute + cache base KV)   = 1.0
  Cycle 1 setup:        lora delta on cached base + new tokens     = 0.3
  Cycle 1 compute:      reuse vanilla cache + ~3 answer tokens     = 0.1
  Cycle 2 setup:        lora delta + text injection tokens         = 0.35
  Cycle 2 compute:      reuse vanilla cache + ~3 answer tokens     = 0.1
  Cycle 3 setup:        lora delta + text injection tokens         = 0.4
  Cycle 3 compute:      reuse vanilla cache + ~3 answer tokens     = 0.1
  ──────────────────────────────────────────────────────────────────
  Total for 3 steps:                                                = 2.35

  FASTER than current (2.35 vs 5.0) AND better arithmetic!
```

```
For 5-step GSM8K:
  Current:          7 passes × 1.0 = 7.0 cost units
  Split + cache:    1.0 + 5 × (0.35 + 0.1) = 3.25 cost units
  
  2x faster AND correct arithmetic.
```

---

## The Breathing Loop (Updated)

```
CYCLE 0: COMPREHEND
  Vanilla Llama reads problem (no atoms)
  → cache base KV for all problem tokens
  → controller reads hidden states → produces initial scales
  
CYCLE 1: BREATHE
  Phase A — SETUP (atoms ON):
    Llama reads with atoms (using cached base KV + LoRA delta)
    Generates: "She sold half as many clips. 48 / 2 ="
    Controller reads hidden states → page + next scales
    
  Phase B — COMPUTE (atoms OFF):
    Vanilla Llama continues (using cached vanilla KV)
    Generates: "24 #### 24</s>"
    Clean arithmetic — no atom interference
    
  → Extract "24" → text injection for cycle 2

CYCLE 2: BREATHE AGAIN
  Phase A — SETUP (atoms ON, different scales):
    Reads: "Step 1 result: 24\n[problem]"
    Generates: "Total clips: 48 + 24 ="
    
  Phase B — COMPUTE (atoms OFF):
    Generates: "72 #### 72</s>"
    
  → Extract "72" → final answer

Each breath: atoms SET UP the equation, vanilla Llama SOLVES it.
Parse loudly, compute quietly. The dance, implemented correctly.
```

---

## Training

The generation targets should put the equation on its own clear line to help the "=" stopping point:

```
Target format:
  "She sold half as many clips in May.
   48 / 2 = 24
   #### 24</s>"

Line 1: natural sentence (atoms help generate this — parsing)
Line 2: standalone equation (vanilla Llama completes the "= X" — computation)
Line 3: answer marker (extraction)
```

Training loss applies to the FULL generation but the split happens during generation:

```python
def train_step(model, problem_ids, targets, kv_cache):
    # Phase A: generate setup WITH atoms, get logits
    setup_logits = generate_setup(model, problem_ids, kv_cache, 
                                   atom_scales, stop_at="=")
    
    # Phase B: generate compute WITHOUT atoms, get logits
    compute_logits = generate_compute(model, kv_cache, 
                                       use_lora=False, stop_at_eos=True)
    
    # Combine logits
    all_logits = torch.cat([setup_logits, compute_logits], dim=1)
    
    # Single gen_loss on full generation
    gen_loss = F.cross_entropy(all_logits, targets)
    
    # Gradient flows:
    # Through setup_logits → atoms (learn to write equations)
    # Through compute_logits → vanilla Llama only (learn arithmetic)
    # The atoms DON'T get gradient from arithmetic errors!
    
    return gen_loss
```

Key insight for training: the atoms ONLY get gradient from Phase A (equation setup). They DON'T get gradient from Phase B (arithmetic completion). This means atoms can't learn to interfere with arithmetic — they only learn to set up equations.

```
Phase A gradient → atoms:      "write better equations" ✓
Phase B gradient → Llama only: "compute better arithmetic" ✓
                   NOT atoms:   atoms can't corrupt arithmetic even via gradient
```

---

## Differentiability

The split is fully differentiable:

```
Phase A: atom_scales → LoRA → Llama → setup_logits → gen_loss
         Gradient flows to atoms and controller ✓

Phase B: vanilla Llama → compute_logits → gen_loss  
         Gradient flows to... nothing trainable!
         Llama is frozen. No atoms applied.
         
         BUT — the CONTEXT from Phase A feeds Phase B.
         The "48 / 2 =" tokens from Phase A are in the KV cache
         that Phase B reads. Gradient flows backward:
         
         Phase B loss → attention over Phase A tokens → Phase A logits → atoms
         
         "The equation you set up led to wrong completion — change the setup"
```

The atoms learn from BOTH phases:
- Phase A: direct gradient on setup tokens
- Phase B: indirect gradient through attention over setup tokens in the KV cache

---

## Stopping at "="

The Phase A / Phase B split happens at the "=" token. How to detect it during generation:

```python
EQUALS_TOKEN_IDS = tokenizer.encode("=")  # might be multiple token IDs

def should_stop_setup(generated_token):
    """Stop Phase A when we hit '=' — Phase B takes over for the answer."""
    return generated_token in EQUALS_TOKEN_IDS
```

If the model generates "48 / 2 =" → stop, switch to Phase B.
If the model generates "She sold half" without reaching "=" after 20 tokens → force stop, Phase B generates the rest.

Max setup tokens: 25 (enough for natural sentence + equation start).
Max compute tokens: 10 (enough for the answer + #### + marker + EOS).

---

## Expected Impact

```
BEFORE (atoms corrupt arithmetic):
  32% wrong arithmetic ("12*20=195")
  20.3% single-step accuracy
  
AFTER (split generation):
  Atoms set up equations (correct parsing, proven at 20%)
  Vanilla Llama computes (correct arithmetic, proven 9/12 = 75%)
  
  Expected: 20% find correct equation × 75% compute correctly = 15%
  MINIMUM. The setup accuracy should improve with training,
  and more equations land in Llama's sweet spot.
  
  Optimistic: 30-40% single-step if setup quality improves.
  The arithmetic ceiling is removed — Llama computes undisturbed.
```

---

## What to Monitor

```
1. Phase A quality: does the model generate correct equation setups?
   "48 / 2 =" (correct operation + correct operands)
   If setup is wrong, Phase B can't help

2. Phase B accuracy: does vanilla Llama complete correctly?
   Given "48 / 2 =", does it generate "24"?
   Should be ~75% (proven in standalone test)

3. The "=" detection: does the model learn to generate "=" reliably?
   If it rambles without producing "=", Phase B never starts
   The training target format (equation on its own line) teaches this

4. Combined accuracy: setup correct AND compute correct
   This is the final single-step accuracy
   Should exceed 20.3% (current ceiling)

5. Speed: actual passes vs theoretical
   2.35 units for 3 steps (vs 5.0 currently)
   Verify the KV cache actually saves time
```

---

## What NOT to Do

```
- Do NOT apply atoms during Phase B.
  The whole point is clean arithmetic. Atoms OFF during computation.
  
- Do NOT generate past EOS in Phase B.
  Phase B generates: "24 #### 24</s>" and STOPS.
  No rambling. The EOS is the hard stop.

- Do NOT cache LoRA-modified KV for reuse.
  Each cycle has DIFFERENT atom scales → different LoRA deltas.
  Only BASE KV is reusable across cycles.
  LoRA deltas are cheap to recompute per cycle.

- Do NOT backpropagate Phase B errors to atoms.
  Phase B uses frozen vanilla Llama. No atoms.
  If Phase B computes wrong ("48 / 2 = 28"), that's Llama's limit.
  The atoms can't fix Llama's arithmetic — they can only set up better equations.
  
- Do NOT skip the vanilla comprehension pass (cycle 0).
  The base KV cache depends on it.
  It also provides the controller's initial comprehension.
  One full pass upfront saves many partial passes later.
```
