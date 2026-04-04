"""
StateConditionedLoRA v3: With optional strategy channel (512 floats).

When strategy is provided: 576 floats → 256 scales (2.25 floats/scale)
When strategy is None: 64 floats → 256 scales (0.25 floats/scale, backward compat)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class StateConditionedLoRA(nn.Module):
    def __init__(
        self,
        d_model: int = 2048,
        d_kv: int = 512,
        state_size: int = 64,
        strategy_size: int = 512,  # NEW
        rank: int = 4,
        num_layers: int = 16,
        num_projections: int = 4,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_kv = d_kv
        self.state_size = state_size
        self.strategy_size = strategy_size
        self.rank = rank
        self.num_layers = num_layers
        self.num_projections = num_projections
        
        self.proj_dims = [d_model, d_kv, d_kv, d_model]
        
        self.A_templates = nn.ParameterList([
            nn.Parameter(torch.randn(num_layers, d_model, rank) * 0.01)
            for _ in range(num_projections)
        ])
        self.B_templates = nn.ParameterList([
            nn.Parameter(torch.randn(num_layers, rank, self.proj_dims[i]) * 0.01)
            for i in range(num_projections)
        ])
        
        # Hypernetwork with strategy channel
        num_scales = num_layers * num_projections * rank
        input_size = state_size + strategy_size
        self.state_to_scales = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_scales),
            nn.Tanh(),
        )
        
        self.proj_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    
    def forward(
        self,
        state: torch.Tensor,
        strategy: Optional[torch.Tensor] = None,
    ) -> Dict[int, Dict[str, Dict[str, torch.Tensor]]]:
        batch_size = state.size(0)
        
        # If no strategy provided, use zeros
        if strategy is None:
            strategy = torch.zeros(batch_size, self.strategy_size, device=state.device, dtype=state.dtype)
        
        combined = torch.cat([state, strategy], dim=-1)
        all_scales = self.state_to_scales(combined)
        
        all_scales = all_scales.reshape(
            batch_size, self.num_layers, self.num_projections, self.rank
        )
        
        lora_mods: Dict[int, Dict[str, Dict[str, torch.Tensor]]] = {}
        
        for layer_idx in range(self.num_layers):
            lora_mods[layer_idx] = {}
            for proj_idx, proj_name in enumerate(self.proj_names):
                scales = all_scales[:, layer_idx, proj_idx, :]
                A = self.A_templates[proj_idx][layer_idx]
                B = self.B_templates[proj_idx][layer_idx]
                lora_mods[layer_idx][proj_name] = {'A': A, 'B': B, 'scales': scales}
        
        return lora_mods


if __name__ == '__main__':
    print("Testing StateConditionedLoRA v3...")
    lora = StateConditionedLoRA()
    
    state = torch.randn(2, 64)
    strategy = torch.randn(2, 512)
    
    mods = lora(state, strategy)
    print(f"Output: {len(mods)} layers")
    print(f"Scales shape: {mods[0]['q_proj']['scales'].shape}")
    
    # Test backward compat (no strategy)
    mods2 = lora(state)
    print(f"Without strategy: {len(mods2)} layers")
    
    print("OK")
