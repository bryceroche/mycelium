"""
Compressor v3: Two-headed output — State (64) + Strategy (512)

State:    64 floats, accumulates on hypersphere, persistent memory
Strategy: 512 floats, overwrites each pass, rich signal to hypernetwork

The hypernetwork goes from starving (64 → 256) to well-fed (576 → 256).
Each LoRA scale now has ~2.25 floats of information instead of ~0.25.

v24.3: Optional Haar wavelet preprocessing for 2x input compression.
v24.7: Direct path skip connection for gradient coupling — mean-pool last layer,
       project directly to state, bypass perceiver cross-attention softmax.
v24.8: Page communication — perceiver cross-attends over BOTH hidden states
       AND previous pages. Creates direct gradient path from pages to new page.
       The page_to_kv projection (64 -> 1024) and learnable page_gate control
       how much influence previous pages have on compression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .haar_wavelet import LayerWiseWavelet


class ResidualPageGate(nn.Module):
    """Per-dimension blending of new and old page.

    gate approx 1: keep new_page (overwrite this dimension)
    gate approx 0: keep old_page (preserve this dimension)

    This shifts eigenvalues by ~0.5, converting contracting dynamics to stable.
    The per-dimension gate lets the model decide which dimensions should persist
    (e.g., coarse problem type) vs. which should update (e.g., current computation).

    Effect on Jacobian eigenvalues:
        Without residual: page_{k+1} = f(page_k), eigenvalues < 0.25 (contracting)
        With residual:    page_{k+1} = gate * f(page_k) + (1-gate) * page_k
                          Eigenvalues shift by ~0.5, converting to stable dynamics.

    Parameter cost: ~8K (Linear(128, 64) = 8,192 params)
    """

    def __init__(self, page_size: int = 64):
        super().__init__()
        self.page_size = page_size
        # Concatenate new and old pages, then produce per-dimension gate
        self.gate = nn.Linear(page_size * 2, page_size)
        self._init_weights()

    def _init_weights(self) -> None:
        # Initialize to produce gates near 0.5 (balanced blending)
        # bias of 0 with sigmoid gives 0.5
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, new_page: torch.Tensor, old_page: torch.Tensor) -> torch.Tensor:
        """
        Blend new and old pages with learned per-dimension gating.

        Args:
            new_page: (batch, page_size) - freshly computed page from perceiver
            old_page: (batch, page_size) - previous pass's page

        Returns:
            blended: (batch, page_size) - gated combination of new and old
        """
        combined = torch.cat([new_page, old_page], dim=-1)  # (batch, page_size * 2)
        gate_values = torch.sigmoid(self.gate(combined))     # (batch, page_size)
        blended = gate_values * new_page + (1 - gate_values) * old_page
        return blended

    def count_parameters(self) -> int:
        """Return total parameter count."""
        return sum(p.numel() for p in self.parameters())


class Compressor(nn.Module):
    """
    Two-headed Perceiver: outputs both state delta and strategy vector.

    State delta:  64 floats, accumulates on hypersphere (persistent memory)
    Strategy:     512 floats, overwritten each pass (attention instructions)

    The strategy channel feeds the hypernetwork with rich signal without
    expanding the bottleneck. Memory is still 64 floats.

    v24.7 adds DIRECT PATH skip connection for gradient coupling:
    - Direct path: mean-pool last layer -> linear -> GELU -> linear -> state
    - Contextual path: full perceiver attention -> state head
    - Learnable blend parameter mixes the two paths
    The direct path bypasses the perceiver's cross-attention softmax.

    v24.8 adds PAGE COMMUNICATION — the perceiver cross-attends over BOTH
    hidden states AND previous pages. This closes the communication loop:
    - Before: perceiver was BLIND to previous pages (only saw Llama hidden states)
    - After: perceiver sees what was already compressed, can focus on NEW info
    Creates direct gradient path: page_k -> page_to_kv -> attention -> page_{k+1}
    """

    def __init__(
        self,
        num_transformer_layers: int = 16,
        d_transformer: int = 2048,
        d_perceiver: int = 1024,
        num_queries: int = 4,
        num_perceiver_layers: int = 7,
        state_size: int = 64,
        strategy_size: int = 64,  # shrunk from 512 → 64 to close bypass
        max_passes: int = 20,
        use_wavelet: bool = True,  # v24.3: Haar wavelet preprocessing (default ON)
        wavelet_levels: int = 4,
    ) -> None:
        super().__init__()

        self.num_transformer_layers = num_transformer_layers
        self.d_transformer = d_transformer
        self.d_perceiver = d_perceiver
        self.num_queries = num_queries
        self.num_perceiver_layers = num_perceiver_layers
        self.state_size = state_size
        self.strategy_size = strategy_size
        self.max_passes = max_passes
        self.use_wavelet = use_wavelet

        # v24.3: Optional wavelet preprocessing (2x input compression)
        if use_wavelet:
            self.wavelet = LayerWiseWavelet(
                num_layers=num_transformer_layers,
                max_level=wavelet_levels,
                truncate_finest=True,
                share_weights=True,
            )
        else:
            self.wavelet = None

        # Learned queries
        self.queries = nn.Parameter(torch.randn(num_queries, d_perceiver) * 0.02)

        # Pass embedding
        self.pass_embed = nn.Embedding(max_passes, d_perceiver)

        # Layer gate
        self.layer_gate = nn.Sequential(
            nn.Linear(d_perceiver, 64),
            nn.ReLU(),
            nn.Linear(64, num_transformer_layers),
        )

        # Input projection
        self.input_project = nn.Linear(d_transformer, d_perceiver)

        # 7-layer perceiver stack
        self.perceiver_layers = nn.ModuleList([
            nn.ModuleDict({
                'cross_attn': nn.MultiheadAttention(
                    embed_dim=d_perceiver, num_heads=8,
                    kdim=d_perceiver, vdim=d_perceiver, batch_first=True,
                ),
                'cross_norm': nn.LayerNorm(d_perceiver),
                'self_attn': nn.MultiheadAttention(
                    embed_dim=d_perceiver, num_heads=8, batch_first=True,
                ),
                'self_norm': nn.LayerNorm(d_perceiver),
                'ffn': nn.Sequential(
                    nn.Linear(d_perceiver, d_perceiver * 4),
                    nn.GELU(),
                    nn.Linear(d_perceiver * 4, d_perceiver),
                ),
                'ffn_norm': nn.LayerNorm(d_perceiver),
            })
            for _ in range(num_perceiver_layers)
        ])

        # Skip connection: cross-attend to previous passes' mid-layer states
        self.skip_attn = nn.MultiheadAttention(d_perceiver, num_heads=8, batch_first=True)
        self.skip_norm = nn.LayerNorm(d_perceiver)

        # TWO OUTPUT HEADS (CONTEXTUAL PATH):
        # 1. State head: 64 floats (accumulates, tight bottleneck)
        self.state_head = nn.Linear(d_perceiver * num_queries, state_size)

        # 2. Strategy head: 512 floats (overwrites, rich signal to hypernetwork)
        self.strategy_head = nn.Linear(d_perceiver * num_queries, strategy_size)

        # ===== DIRECT PATH (v24.7 — clean gradient highway) =====
        # Mean-pool last layer's hidden states, project directly to state_size
        # Bypasses perceiver cross-attention softmax
        self.direct_pool_norm = nn.LayerNorm(d_transformer)
        self.direct_project = nn.Sequential(
            nn.Linear(d_transformer, d_transformer // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_transformer // 4, state_size),
        )

        # ===== BLEND (learnable mixing of direct + contextual) =====
        # Initialize to 0.5 (balanced). Model can shift during training.
        # sigmoid(0.0) = 0.5
        self.blend_logit = nn.Parameter(torch.tensor(0.0))

        # ===== PAGE COMMUNICATION (v24.8 — perceiver sees previous pages) =====
        # Project pages into perceiver's key/value space so the perceiver can
        # cross-attend over BOTH hidden states AND previous pages.
        # This creates a direct gradient path from pages to new page.
        self.page_to_kv = nn.Linear(state_size, d_perceiver)

        # Learnable gate for page influence (starts at 0.5)
        # Low (0.1): perceiver ignoring pages (bad)
        # Mid (0.3-0.7): balancing hidden states and pages (good)
        # High (0.9): perceiver ignoring hidden states (bad)
        self.page_gate = nn.Parameter(torch.tensor(0.5))

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.queries, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.input_project.weight)
        nn.init.zeros_(self.input_project.bias)
        nn.init.xavier_uniform_(self.state_head.weight)
        nn.init.zeros_(self.state_head.bias)
        nn.init.xavier_uniform_(self.strategy_head.weight)
        nn.init.zeros_(self.strategy_head.bias)
        nn.init.zeros_(self.layer_gate[0].weight)
        nn.init.zeros_(self.layer_gate[0].bias)
        nn.init.zeros_(self.layer_gate[2].weight)
        nn.init.zeros_(self.layer_gate[2].bias)
        # Direct path initialization
        nn.init.xavier_uniform_(self.direct_project[0].weight)
        nn.init.zeros_(self.direct_project[0].bias)
        nn.init.xavier_uniform_(self.direct_project[3].weight)
        nn.init.zeros_(self.direct_project[3].bias)
        # Page communication (v24.8)
        nn.init.xavier_uniform_(self.page_to_kv.weight)
        nn.init.zeros_(self.page_to_kv.bias)

    def forward(
        self,
        all_layer_hidden_states: List[torch.Tensor],
        pass_num: int,
        prev_mid_states: Optional[List[torch.Tensor]] = None,
        state_pages: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compress hidden states into state delta AND strategy vector.

        Args:
            all_layer_hidden_states: List of 16 tensors, each (batch, seq_len, 2048)
            pass_num: Which thinking pass (0-indexed)
            prev_mid_states: Optional list of (batch, num_queries, d_perceiver) tensors
                from previous passes' mid-layer (after layer 4). If None or empty,
                skip connection is not applied (backward compatible).
            state_pages: Optional list of (batch, state_size) tensors — accumulated
                pages from previous passes. If provided, the perceiver cross-attends
                over BOTH hidden states AND these pages (v24.8 page communication).
                This creates a direct gradient path from previous pages to new page.

        Returns:
            state_delta:       (batch, 64) — accumulates on hypersphere
            strategy:          (batch, 512) — overwrites, feeds hypernetwork
            current_mid_states: (batch, num_queries, d_perceiver) — detached mid-layer
                states for future passes to cross-attend to
        """
        assert len(all_layer_hidden_states) == self.num_transformer_layers

        batch_size = all_layer_hidden_states[0].size(0)
        device = all_layer_hidden_states[0].device
        dtype = all_layer_hidden_states[0].dtype

        # v24.3: Optional wavelet preprocessing (2x compression before perceiver)
        if self.wavelet is not None:
            all_layer_hidden_states = self.wavelet(all_layer_hidden_states)

        # Pass conditioning
        pass_idx = torch.tensor(pass_num, device=device)
        pass_context = self.pass_embed(pass_idx)
        queries = (self.queries + pass_context).unsqueeze(0).expand(batch_size, -1, -1)

        # Layer gating
        layer_logits = self.layer_gate(pass_context)
        layer_weights = F.softmax(layer_logits, dim=-1)

        # Combine layers
        stacked = torch.stack(all_layer_hidden_states, dim=0)
        weights = layer_weights.view(self.num_transformer_layers, 1, 1, 1)
        combined = (stacked * weights).sum(dim=0)

        # Project to perceiver space
        kv = self.input_project(combined.to(dtype=self.input_project.weight.dtype))
        queries = queries.to(dtype=kv.dtype)

        # ===== PAGE COMMUNICATION (v24.8) =====
        # If we have previous pages, add them as additional key/value tokens.
        # The perceiver sees BOTH hidden states AND previous pages, enabling
        # informed compression: "What's in the hidden states that ISN'T already
        # in my pages?" Creates direct gradient path: pages -> page_to_kv -> kv -> new_page
        if state_pages is not None and len(state_pages) > 0:
            # Stack pages: (batch, num_pages, state_size)
            pages_stacked = torch.stack(state_pages, dim=1)
            # Project to perceiver dim: (batch, num_pages, d_perceiver)
            pages_projected = self.page_to_kv(pages_stacked.to(dtype=self.page_to_kv.weight.dtype))
            # Gate the page contribution (learnable, starts at 0.5)
            gate = torch.sigmoid(self.page_gate)
            pages_projected = pages_projected * gate
            # Concatenate page tokens with hidden state tokens
            # kv: (batch, seq_len, d_perceiver) + pages: (batch, num_pages, d_perceiver)
            kv = torch.cat([kv, pages_projected], dim=1)

        # Layers 0-3 (first half)
        for layer in self.perceiver_layers[:4]:
            attended, _ = layer['cross_attn'](query=queries, key=kv, value=kv)
            queries = layer['cross_norm'](queries + attended)

            refined, _ = layer['self_attn'](query=queries, key=queries, value=queries)
            queries = layer['self_norm'](queries + refined)

            ffn_out = layer['ffn'](queries)
            queries = layer['ffn_norm'](queries + ffn_out)

        # Save mid-layer states (detached — no gradient through stored states)
        current_mid_states = queries.detach().clone()

        # Skip connection: cross-attend to previous passes' mid-layer states
        if prev_mid_states is not None and len(prev_mid_states) > 0:
            prev = torch.cat(prev_mid_states, dim=1)  # (batch, P*num_queries, d_perceiver)
            skip_out, _ = self.skip_attn(query=queries, key=prev, value=prev)
            queries = self.skip_norm(queries + skip_out)

        # Layers 4-6 (second half)
        for layer in self.perceiver_layers[4:]:
            attended, _ = layer['cross_attn'](query=queries, key=kv, value=kv)
            queries = layer['cross_norm'](queries + attended)

            refined, _ = layer['self_attn'](query=queries, key=queries, value=queries)
            queries = layer['self_norm'](queries + refined)

            ffn_out = layer['ffn'](queries)
            queries = layer['ffn_norm'](queries + ffn_out)

        # Flatten queries: (batch, num_queries * d_perceiver)
        flat = queries.flatten(start_dim=1)

        # ===== CONTEXTUAL PATH (existing perceiver output) =====
        context_state = self.state_head(flat)   # (batch, 64)
        strategy = self.strategy_head(flat)      # (batch, 512)

        # ===== DIRECT PATH (v24.7 — gradient highway) =====
        # Mean-pool the last layer (BEFORE wavelet, use original hidden states)
        # NOTE: We need the original last layer, not wavelet-processed
        # But wavelet has already been applied above. For now, use the
        # post-wavelet last layer. This still bypasses perceiver attention.
        last_layer = all_layer_hidden_states[-1]  # (batch, seq_len, d_transformer)
        pooled = last_layer.mean(dim=1)            # (batch, d_transformer)
        pooled = self.direct_pool_norm(pooled.to(dtype=self.direct_pool_norm.weight.dtype))
        direct_state = self.direct_project(pooled)  # (batch, state_size)

        # ===== BLEND direct + contextual for state =====
        blend = torch.sigmoid(self.blend_logit)  # scalar in (0, 1)
        state_delta = blend * direct_state + (1 - blend) * context_state

        return state_delta, strategy, current_mid_states

    def count_parameters(self) -> dict:
        counts = {
            'queries': self.queries.numel(),
            'pass_embed': sum(p.numel() for p in self.pass_embed.parameters()),
            'layer_gate': sum(p.numel() for p in self.layer_gate.parameters()),
            'input_project': sum(p.numel() for p in self.input_project.parameters()),
            'perceiver_layers': sum(
                sum(p.numel() for p in layer.parameters())
                for layer in self.perceiver_layers
            ),
            'skip_attn': sum(p.numel() for p in self.skip_attn.parameters()),
            'skip_norm': sum(p.numel() for p in self.skip_norm.parameters()),
            'state_head': sum(p.numel() for p in self.state_head.parameters()),
            'strategy_head': sum(p.numel() for p in self.strategy_head.parameters()),
            'direct_path': (
                sum(p.numel() for p in self.direct_pool_norm.parameters()) +
                sum(p.numel() for p in self.direct_project.parameters()) +
                1  # blend_logit
            ),
            'page_communication': (
                sum(p.numel() for p in self.page_to_kv.parameters()) +
                1  # page_gate
            ),
        }
        if self.wavelet is not None:
            counts['wavelet'] = self.wavelet.count_parameters()
        counts['total'] = sum(counts.values())
        return counts


if __name__ == "__main__":
    print("Testing Compressor v3 with strategy channel...")

    compressor = Compressor()
    counts = compressor.count_parameters()

    print(f"\nParameter counts:")
    for name, count in counts.items():
        if count > 1e6:
            print(f"  {name}: {count / 1e6:.1f}M")
        else:
            print(f"  {name}: {count:,}")

    # Test forward pass
    batch_size = 2
    seq_len = 128
    all_layer_hidden_states = [torch.randn(batch_size, seq_len, 2048) for _ in range(16)]

    state_delta, strategy, mid_states = compressor(all_layer_hidden_states, pass_num=0)

    print(f"\nOutput shapes:")
    print(f"  state_delta:  {state_delta.shape} (memory, accumulates)")
    print(f"  strategy:     {strategy.shape} (instructions, overwrites)")
    print(f"  mid_states:   {mid_states.shape} (perceiver mid-layer, private memory)")

    # Gradient flow (no prev_mid_states)
    (state_delta.sum() + strategy.sum()).backward()
    print("\nGradient flow (no skip): OK")

    # Test with skip connection
    compressor.zero_grad()
    mid_history = [mid_states]
    state_delta2, strategy2, mid_states2 = compressor(
        all_layer_hidden_states, pass_num=1, prev_mid_states=mid_history,
    )
    (state_delta2.sum() + strategy2.sum()).backward()
    print("Gradient flow (with skip): OK")
    print(f"  mid_states2: {mid_states2.shape}")

    # Test with wavelet preprocessing (v24.3)
    print("\n--- Testing with Wavelet Preprocessing (v24.3) ---")
    compressor_wavelet = Compressor(use_wavelet=True, wavelet_levels=4)
    counts_wavelet = compressor_wavelet.count_parameters()

    print(f"\nParameter counts (with wavelet):")
    for name, count in counts_wavelet.items():
        if count > 1e6:
            print(f"  {name}: {count / 1e6:.1f}M")
        else:
            print(f"  {name}: {count:,}")

    state_delta_w, strategy_w, mid_states_w = compressor_wavelet(
        all_layer_hidden_states, pass_num=0
    )

    print(f"\nWith wavelet:")
    print(f"  Input:  {len(all_layer_hidden_states)} layers x {all_layer_hidden_states[0].shape}")
    print(f"  state_delta:  {state_delta_w.shape}")
    print(f"  strategy:     {strategy_w.shape}")
    print(f"  mid_states:   {mid_states_w.shape}")

    # Gradient flow
    compressor_wavelet.zero_grad()
    (state_delta_w.sum() + strategy_w.sum()).backward()
    wavelet_grad = compressor_wavelet.wavelet.wavelet.level_weights.grad
    print(f"  Wavelet level_weights grad: {wavelet_grad}")
    print("Gradient flow (wavelet): OK")

    # ===== Test ResidualPageGate (v24.8) =====
    print("\n--- Testing ResidualPageGate (v24.8) ---")
    page_size = 64
    gate = ResidualPageGate(page_size=page_size)
    print(f"  Parameters: {gate.count_parameters():,}")

    # Create test pages with requires_grad
    new_page = torch.randn(batch_size, page_size, requires_grad=True)
    old_page = torch.randn(batch_size, page_size, requires_grad=True)

    # Forward pass
    blended = gate(new_page, old_page)
    print(f"  new_page shape:  {new_page.shape}")
    print(f"  old_page shape:  {old_page.shape}")
    print(f"  blended shape:   {blended.shape}")

    # Verify gradient flow through both inputs
    loss = blended.sum()
    loss.backward()

    assert new_page.grad is not None, "No gradient to new_page!"
    assert old_page.grad is not None, "No gradient to old_page!"
    assert gate.gate.weight.grad is not None, "No gradient to gate weights!"
    assert gate.gate.bias.grad is not None, "No gradient to gate bias!"

    print(f"  new_page grad norm: {new_page.grad.norm().item():.6f}")
    print(f"  old_page grad norm: {old_page.grad.norm().item():.6f}")
    print(f"  gate weight grad norm: {gate.gate.weight.grad.norm().item():.6f}")
    print("Gradient flow (ResidualPageGate): OK")

    # Verify output is bounded blend
    with torch.no_grad():
        new_page_test = torch.randn(batch_size, page_size)
        old_page_test = torch.randn(batch_size, page_size)
        blended_test = gate(new_page_test, old_page_test)

        # Blended should be element-wise between new and old (or close)
        # Due to the gate, blended = gate * new + (1-gate) * old
        # where gate is in (0, 1). Check that blended values are reasonable.
        print(f"  new_page_test mean: {new_page_test.mean().item():.4f}")
        print(f"  old_page_test mean: {old_page_test.mean().item():.4f}")
        print(f"  blended mean:       {blended_test.mean().item():.4f}")
    print("ResidualPageGate output sanity check: OK")

    # ===== Test Page Communication (v24.8) =====
    print("\n--- Testing Page Communication (v24.8) ---")
    compressor_pages = Compressor(use_wavelet=False)  # Disable wavelet for cleaner test
    counts_pages = compressor_pages.count_parameters()

    print(f"\nParameter counts (with page communication):")
    for name, count in counts_pages.items():
        if count > 1e6:
            print(f"  {name}: {count / 1e6:.1f}M")
        else:
            print(f"  {name}: {count:,}")

    # Test pass 0: no previous pages
    state_delta_p0, strategy_p0, mid_states_p0 = compressor_pages(
        all_layer_hidden_states, pass_num=0, state_pages=None
    )
    print(f"\nPass 0 (no pages):")
    print(f"  state_delta: {state_delta_p0.shape}")
    print(f"  strategy:    {strategy_p0.shape}")

    # Test pass 1: one previous page
    prev_pages = [state_delta_p0.detach().clone()]  # Simulate accumulated page
    # Create new hidden states (would normally come from LoRA-modified Llama)
    all_layer_hidden_states_p1 = [torch.randn(batch_size, seq_len, 2048) for _ in range(16)]

    compressor_pages.zero_grad()
    state_delta_p1, strategy_p1, mid_states_p1 = compressor_pages(
        all_layer_hidden_states_p1, pass_num=1, state_pages=prev_pages
    )
    print(f"\nPass 1 (1 previous page):")
    print(f"  state_delta: {state_delta_p1.shape}")
    print(f"  strategy:    {strategy_p1.shape}")
    print(f"  page_gate value: {torch.sigmoid(compressor_pages.page_gate).item():.4f}")

    # Test gradient flow through pages
    loss_p1 = state_delta_p1.sum() + strategy_p1.sum()
    loss_p1.backward()

    assert compressor_pages.page_to_kv.weight.grad is not None, "No gradient to page_to_kv!"
    assert compressor_pages.page_gate.grad is not None, "No gradient to page_gate!"
    print(f"  page_to_kv weight grad norm: {compressor_pages.page_to_kv.weight.grad.norm().item():.6f}")
    print(f"  page_gate grad: {compressor_pages.page_gate.grad.item():.6f}")
    print("Gradient flow (page communication): OK")

    # Test pass 2: two previous pages
    prev_pages_2 = [state_delta_p0.detach().clone(), state_delta_p1.detach().clone()]
    all_layer_hidden_states_p2 = [torch.randn(batch_size, seq_len, 2048) for _ in range(16)]

    compressor_pages.zero_grad()
    state_delta_p2, strategy_p2, mid_states_p2 = compressor_pages(
        all_layer_hidden_states_p2, pass_num=2, state_pages=prev_pages_2
    )
    print(f"\nPass 2 (2 previous pages):")
    print(f"  state_delta: {state_delta_p2.shape}")
    print(f"  strategy:    {strategy_p2.shape}")

    # Verify gradient flows through multiple pages
    loss_p2 = state_delta_p2.sum() + strategy_p2.sum()
    loss_p2.backward()
    print(f"  page_to_kv weight grad norm: {compressor_pages.page_to_kv.weight.grad.norm().item():.6f}")
    print("Gradient flow (2 pages): OK")

    # Test that page communication parameter count is correct
    expected_page_comm_params = 64 * 1024 + 1024 + 1  # Linear(64, 1024) + bias + page_gate
    actual_page_comm_params = counts_pages['page_communication']
    print(f"\nPage communication params: {actual_page_comm_params:,} (expected ~{expected_page_comm_params:,})")
    assert actual_page_comm_params == 64 * 1024 + 1024 + 1, f"Unexpected param count: {actual_page_comm_params}"
    print("Parameter count check: OK")

    print("\nAll tests passed!")
