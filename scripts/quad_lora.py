"""
QuadLoRA: Four-Mode LoRA Architecture (v23).

Four sets of LoRA templates — parse, compute, verify, answer — blended via
4-way softmax. Each mode specializes for a different cognitive operation:

  PARSE:   Read the problem, extract quantities and relationships
  COMPUTE: Apply operations to extracted quantities
  VERIFY:  Check that the solution is internally consistent
  ANSWER:  Shape hidden states for clean answer extraction

Evolution from DualLoRA (v22): forward->compute, verify->verify, +parse, +answer.
Includes AnswerHead for direct digit-based answer extraction from last page.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

import os
import sys
# Support both local dev and remote VM paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, '/home/ubuntu/mycelium')

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.compressor_v3 import Compressor


# ---------------------------------------------------------------------------
# Mode names, used throughout
# ---------------------------------------------------------------------------
MODE_NAMES = ['parse', 'compute', 'verify', 'answer']
NUM_MODES = len(MODE_NAMES)


# ---------------------------------------------------------------------------
# QuadAdditiveLoRAManager
# ---------------------------------------------------------------------------
class QuadAdditiveLoRAManager:
    """
    Manages inline blended quad LoRA (parse + compute + verify + answer)
    on Llama attention projections via monkey-patching.

    At each projection layer:
        lora_parse   = (x @ A_parse)   * parse_scales   @ B_parse
        lora_compute = (x @ A_compute) * compute_scales @ B_compute
        lora_verify  = (x @ A_verify)  * verify_scales  @ B_verify
        lora_answer  = (x @ A_answer)  * answer_scales  @ B_answer
        lora_out     = blend[0]*parse + blend[1]*compute + blend[2]*verify + blend[3]*answer
        output       = W @ x + lora_out
    """

    PROJECTION_NAMES = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

    def __init__(self, model: nn.Module):
        self.model = model
        self._original_forwards: Dict[int, Dict[str, callable]] = {}
        self._active = False

        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            self.num_layers = len(model.model.layers)
        else:
            self.num_layers = 16

    def _get_projection(self, layer_idx: int, proj_name: str) -> nn.Module:
        return getattr(self.model.model.layers[layer_idx].self_attn, proj_name)

    @staticmethod
    def _make_quad_lora_forward(
        original_forward: callable,
        mode_params: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        blend: torch.Tensor,
    ) -> callable:
        """
        Create a patched forward that blends four LoRA terms.

        mode_params: list of (A, B, scales) for each mode in order
                     [parse, compute, verify, answer]
        blend: (B, 4) — softmax weights summing to 1
        """
        def forward(x: torch.Tensor) -> torch.Tensor:
            base_output = original_forward(x)
            dtype = x.dtype
            device = x.device

            lora_out = None
            for mode_idx, (A, B, scales) in enumerate(mode_params):
                a = A.to(dtype=dtype, device=device)
                b = B.to(dtype=dtype, device=device)
                s = scales.to(dtype=dtype, device=device)
                mode_lora = (x @ a * s.unsqueeze(1)) @ b  # (B, seq, proj_dim)

                # blend[:, mode_idx] is (B,) -> (B, 1, 1) for broadcasting
                w = blend[:, mode_idx].to(dtype=dtype, device=device)
                w = w.unsqueeze(1).unsqueeze(2)
                weighted = w * mode_lora

                if lora_out is None:
                    lora_out = weighted
                else:
                    lora_out = lora_out + weighted

            return base_output + lora_out

        return forward

    def apply(
        self,
        all_mods: Dict[str, Dict[int, Dict[str, Dict[str, torch.Tensor]]]],
        blend: torch.Tensor,
    ) -> None:
        """
        Apply blended quad LoRA.

        Args:
            all_mods: {mode_name: {layer_idx: {proj_name: {A, B, scales}}}}
            blend: (B, 4) softmax blend weights
        """
        if self._active:
            raise RuntimeError(
                "Quad LoRA already applied. Call remove() before applying again."
            )

        self._original_forwards = {}

        # Get layer indices from first mode
        first_mode = MODE_NAMES[0]
        layer_indices = sorted(all_mods[first_mode].keys())

        for layer_idx in layer_indices:
            self._original_forwards[layer_idx] = {}

            for proj_name in self.PROJECTION_NAMES:
                if proj_name not in all_mods[first_mode][layer_idx]:
                    continue

                proj_module = self._get_projection(layer_idx, proj_name)
                self._original_forwards[layer_idx][proj_name] = proj_module.forward

                # Gather (A, B, scales) for each mode
                mode_params = []
                for mode in MODE_NAMES:
                    mod = all_mods[mode][layer_idx][proj_name]
                    mode_params.append((mod['A'], mod['B'], mod['scales']))

                proj_module.forward = self._make_quad_lora_forward(
                    proj_module.forward, mode_params, blend,
                )

        self._active = True

    def remove(self) -> None:
        if not self._active:
            return
        for layer_idx, layer_forwards in self._original_forwards.items():
            for proj_name, original_forward in layer_forwards.items():
                proj_module = self._get_projection(layer_idx, proj_name)
                proj_module.forward = original_forward
        self._original_forwards = {}
        self._active = False

    def is_active(self) -> bool:
        return self._active

    def __del__(self):
        if self._active:
            self.remove()


# ---------------------------------------------------------------------------
# QuadHypernetwork
# ---------------------------------------------------------------------------
class QuadHypernetwork(nn.Module):
    """
    Hypernetwork with four LoRA template sets and 4-way softmax blend.

    Same page cross-attention mechanism as DualPageHypernetwork but outputs:
    - 4 x num_scales scales (tanh, num_scales per mode)
    - 4 blend logits -> softmax -> 4-way blend weights

    Templates: A and B for each of parse, compute, verify, answer.
    GQA-aware: K,V projections use d_kv (512) not d_model (2048).
    """

    def __init__(
        self,
        d_model: int = 2048,
        d_kv: int = 512,
        page_size: int = 64,
        strategy_size: int = 64,
        rank: int = 4,
        num_layers: int = 16,
        num_projections: int = 4,
        attn_dim: int = 256,
        num_query_heads: int = 4,
        num_attn_heads: int = 4,
        max_passes: int = 10,
        pass_embed_dim: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_kv = d_kv
        self.page_size = page_size
        self.strategy_size = strategy_size
        self.rank = rank
        self.num_layers = num_layers
        self.num_projections = num_projections
        self.num_query_heads = num_query_heads
        self.attn_dim = attn_dim

        # GQA: K,V use d_kv, Q,O use d_model
        self.proj_dims = [d_model, d_kv, d_kv, d_model]
        self.proj_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

        num_scales = num_layers * num_projections * rank
        self._num_scales = num_scales

        # --- Four sets of templates ---
        # parse: language-heavy, comprehension-focused
        self.A_parse = nn.ParameterList([
            nn.Parameter(torch.randn(num_layers, d_model, rank) * 0.01)
            for _ in range(num_projections)
        ])
        self.B_parse = nn.ParameterList([
            nn.Parameter(torch.randn(num_layers, rank, self.proj_dims[i]) * 0.01)
            for i in range(num_projections)
        ])

        # compute: math-heavy, execution-focused
        self.A_compute = nn.ParameterList([
            nn.Parameter(torch.randn(num_layers, d_model, rank) * 0.01)
            for _ in range(num_projections)
        ])
        self.B_compute = nn.ParameterList([
            nn.Parameter(torch.randn(num_layers, rank, self.proj_dims[i]) * 0.01)
            for i in range(num_projections)
        ])

        # verify: broad, relational, consistency-checking
        self.A_verify = nn.ParameterList([
            nn.Parameter(torch.randn(num_layers, d_model, rank) * 0.01)
            for _ in range(num_projections)
        ])
        self.B_verify = nn.ParameterList([
            nn.Parameter(torch.randn(num_layers, rank, self.proj_dims[i]) * 0.01)
            for i in range(num_projections)
        ])

        # answer: extraction-focused, answer-oriented
        self.A_answer = nn.ParameterList([
            nn.Parameter(torch.randn(num_layers, d_model, rank) * 0.01)
            for _ in range(num_projections)
        ])
        self.B_answer = nn.ParameterList([
            nn.Parameter(torch.randn(num_layers, rank, self.proj_dims[i]) * 0.01)
            for i in range(num_projections)
        ])

        # Collect references for convenience
        self._template_lists = {
            'parse':   (self.A_parse,   self.B_parse),
            'compute': (self.A_compute, self.B_compute),
            'verify':  (self.A_verify,  self.B_verify),
            'answer':  (self.A_answer,  self.B_answer),
        }

        # --- Page cross-attention (shared across all modes) ---
        self.page_project = nn.Linear(page_size, attn_dim)
        self.page_query = nn.Parameter(torch.randn(num_query_heads, attn_dim) * 0.02)
        self.page_attn = nn.MultiheadAttention(
            attn_dim, num_heads=num_attn_heads, batch_first=True,
        )
        self.page_norm = nn.LayerNorm(attn_dim)

        # --- Pass embedding ---
        self.pass_embed = nn.Embedding(max_passes, pass_embed_dim)
        self.pass_embed_dim = pass_embed_dim

        # --- Combine -> 4 * num_scales + 4 blend logits ---
        combined_dim = num_query_heads * attn_dim + strategy_size + pass_embed_dim
        output_dim = NUM_MODES * num_scales + NUM_MODES  # 4*256 + 4 = 1028
        self.combine = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.GELU(),
            nn.Linear(512, output_dim),
        )

    def compute_scales_and_blend(
        self,
        state_pages: List[torch.Tensor],
        strategy: torch.Tensor,
        pass_num: int = 0,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Returns:
            scales_dict: {mode_name: (B, num_scales)} tanh-bounded
            blend: (B, 4) softmax-bounded, sums to 1
        """
        batch_size = strategy.size(0)
        device = strategy.device
        dtype = strategy.dtype

        pass_t = torch.tensor([pass_num], device=device)
        pass_emb = self.pass_embed(pass_t).expand(batch_size, -1)

        if len(state_pages) == 0:
            scales_dict = {
                mode: torch.zeros(batch_size, self._num_scales, device=device, dtype=dtype)
                for mode in MODE_NAMES
            }
            # Uniform blend when no pages
            blend = torch.ones(batch_size, NUM_MODES, device=device, dtype=dtype) / NUM_MODES
            return scales_dict, blend

        pages = torch.stack(state_pages, dim=1)          # (B, P, page_size)
        pages_proj = self.page_project(pages)             # (B, P, attn_dim)
        queries = self.page_query.unsqueeze(0).expand(batch_size, -1, -1)
        attended, _ = self.page_attn(query=queries, key=pages_proj, value=pages_proj)
        attended = self.page_norm(attended)
        page_summary = attended.flatten(start_dim=1)      # (B, num_query_heads * attn_dim)

        combined = torch.cat([page_summary, strategy, pass_emb], dim=-1)
        out = self.combine(combined)

        # Split: 4 scale blocks + 4 blend logits
        ns = self._num_scales
        scales_dict = {}
        for i, mode in enumerate(MODE_NAMES):
            scales_dict[mode] = torch.tanh(out[:, i * ns:(i + 1) * ns])

        blend_logits = out[:, NUM_MODES * ns:]            # (B, 4)
        blend = F.softmax(blend_logits, dim=-1)           # (B, 4)

        return scales_dict, blend

    def forward(
        self,
        state_pages: List[torch.Tensor],
        strategy: torch.Tensor,
        pass_num: int = 0,
    ) -> Tuple[Dict[str, Dict[int, Dict[str, Dict[str, torch.Tensor]]]], torch.Tensor]:
        """
        Returns:
            all_mods: {mode_name: {layer_idx: {proj_name: {A, B, scales}}}}
            blend: (B, 4) softmax blend weights
        """
        scales_dict, blend = self.compute_scales_and_blend(
            state_pages, strategy, pass_num,
        )
        batch_size = blend.size(0)

        # Reshape each mode's scales to (B, num_layers, num_projections, rank)
        reshaped_scales = {}
        for mode in MODE_NAMES:
            reshaped_scales[mode] = scales_dict[mode].reshape(
                batch_size, self.num_layers, self.num_projections, self.rank,
            )

        all_mods = {}
        for mode in MODE_NAMES:
            A_list, B_list = self._template_lists[mode]
            mods = {}
            for layer_idx in range(self.num_layers):
                mods[layer_idx] = {}
                for proj_idx, proj_name in enumerate(self.proj_names):
                    mods[layer_idx][proj_name] = {
                        'A': A_list[proj_idx][layer_idx],     # (d_model, rank)
                        'B': B_list[proj_idx][layer_idx],     # (rank, proj_dim)
                        'scales': reshaped_scales[mode][:, layer_idx, proj_idx, :],
                    }
            all_mods[mode] = mods

        return all_mods, blend

    def warm_start_from_dual(self, dual_state: dict) -> Tuple[int, int]:
        """
        Warm-start from a dual LoRA checkpoint.

        Mapping:
            dual forward -> quad compute
            dual verify  -> quad verify
            parse, answer -> small random init (already done by __init__)

        Shared components (page_attn, combine, pass_embed) loaded where shapes match.
        """
        loaded = 0
        skipped = 0
        own_state = self.state_dict()

        for key, value in dual_state.items():
            mapped_key = key
            # Map dual forward templates -> quad compute templates
            if key.startswith('A_forward.'):
                mapped_key = key.replace('A_forward.', 'A_compute.')
            elif key.startswith('B_forward.'):
                mapped_key = key.replace('B_forward.', 'B_compute.')
            # Verify templates map directly
            elif key.startswith('A_verify.'):
                mapped_key = key  # same name
            elif key.startswith('B_verify.'):
                mapped_key = key  # same name

            if mapped_key in own_state and own_state[mapped_key].shape == value.shape:
                own_state[mapped_key] = value
                loaded += 1
            elif key in own_state and own_state[key].shape == value.shape:
                # Shared components: page_project, page_query, page_attn, page_norm, pass_embed
                own_state[key] = value
                loaded += 1
            else:
                skipped += 1

        self.load_state_dict(own_state, strict=False)
        print(f"  quad hypernet warm start from dual: loaded {loaded}, skipped {skipped}")
        return loaded, skipped


# ---------------------------------------------------------------------------
# AnswerHead
# ---------------------------------------------------------------------------
class AnswerHead(nn.Module):
    """
    Reads the last page (64 floats) and predicts the answer as digits.

    Three sub-heads:
    - sign_head:   Linear(page_size, 2)           -> positive or negative
    - length_head: Linear(page_size, max_digits)   -> how many digits (1-indexed)
    - digit_heads: max_digits x Linear(page_size, 10) -> 0-9 per position

    The ANSWER LoRA mode shapes hidden states so the perceiver compresses them
    into pages that this head can read as digits.
    """

    def __init__(self, page_size: int = 64, max_digits: int = 6):
        super().__init__()
        self.page_size = page_size
        self.max_digits = max_digits

        self.sign_head = nn.Linear(page_size, 2)
        self.length_head = nn.Linear(page_size, max_digits)
        self.digit_heads = nn.ModuleList([
            nn.Linear(page_size, 10) for _ in range(max_digits)
        ])

    def forward(
        self, last_page: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            last_page: (B, page_size) float tensor

        Returns:
            sign_logits:   (B, 2)
            length_logits: (B, max_digits)
            digit_logits:  list of max_digits (B, 10) tensors
        """
        page = last_page.float()
        sign_logits = self.sign_head(page)
        length_logits = self.length_head(page)
        digit_logits = [head(page) for head in self.digit_heads]
        return sign_logits, length_logits, digit_logits

    @torch.no_grad()
    def decode(self, last_page: torch.Tensor) -> torch.Tensor:
        """
        Decode predicted answer as integer tensor (B,).

        Args:
            last_page: (B, page_size)

        Returns:
            answers: (B,) integer tensor
        """
        sign_logits, length_logits, digit_logits = self.forward(last_page)
        batch_size = last_page.size(0)

        # Predicted number of digits (1-indexed: argmax 0 -> 1 digit)
        num_digits = length_logits.argmax(dim=-1) + 1   # (B,)
        is_negative = sign_logits.argmax(dim=-1) == 1    # (B,)

        answers = torch.zeros(batch_size, dtype=torch.long, device=last_page.device)
        for i in range(self.max_digits):
            digit = digit_logits[i].argmax(dim=-1)       # (B,)
            answers = answers * 10 + digit

        # Mask to predicted length: keep only the first num_digits digits
        # e.g. if num_digits=3 and max_digits=6, we predicted 6 digits but
        # only the first 3 matter. The digit heads are ordered left-to-right
        # (most significant first).
        # To mask: divide by 10^(max_digits - num_digits)
        trim_power = (self.max_digits - num_digits).clamp(min=0)
        divisor = (10 ** trim_power).long()
        answers = answers // divisor

        # Apply sign
        answers = torch.where(is_negative, -answers, answers)
        return answers


def answer_head_loss(
    answer_head: AnswerHead,
    last_page: torch.Tensor,
    gold_answers: torch.Tensor,
) -> torch.Tensor:
    """
    Compute answer head loss from gold integer answers.

    Args:
        answer_head: AnswerHead module
        last_page: (B, page_size)
        gold_answers: (B,) integer tensor of gold answers

    Returns:
        loss: scalar tensor
    """
    sign_logits, length_logits, digit_logits = answer_head(last_page)
    device = last_page.device
    batch_size = last_page.size(0)

    # Parse gold answers into sign, length, digits
    gold_abs = gold_answers.abs()
    gold_sign = (gold_answers < 0).long()                  # 0=positive, 1=negative

    # Convert absolute values to digit strings
    gold_strings = [str(v.item()) for v in gold_abs]
    max_digits = answer_head.max_digits

    gold_length = torch.tensor(
        [len(s) - 1 for s in gold_strings],                # 0-indexed for CE
        dtype=torch.long, device=device,
    )
    gold_length = gold_length.clamp(max=max_digits - 1)

    # Pad digit strings to max_digits (left-pad with 0)
    gold_digit_matrix = torch.zeros(
        batch_size, max_digits, dtype=torch.long, device=device,
    )
    for b, s in enumerate(gold_strings):
        s = s[:max_digits]  # truncate if longer than max_digits
        for i, ch in enumerate(s):
            gold_digit_matrix[b, i] = int(ch)

    # Losses
    loss = F.cross_entropy(sign_logits, gold_sign)
    loss = loss + F.cross_entropy(length_logits, gold_length)

    for i in range(max_digits):
        # Only supervise digit positions that exist in the gold answer
        # Create mask: digit position i exists if i < len(gold_string)
        mask = torch.tensor(
            [1.0 if i < len(gold_strings[b]) else 0.0 for b in range(batch_size)],
            device=device,
        )
        if mask.sum() > 0:
            digit_loss = F.cross_entropy(
                digit_logits[i], gold_digit_matrix[:, i], reduction='none',
            )
            loss = loss + (digit_loss * mask).sum() / mask.sum()

    return loss


# ---------------------------------------------------------------------------
# QuadConfidenceHead
# ---------------------------------------------------------------------------
class QuadConfidenceHead(nn.Module):
    """
    Confidence head that reads pages + 4-way blend history.

    Evolution of PageConfidenceHead: blend_project takes 4 inputs (not 1)
    so it can see the full mode distribution trajectory and learn patterns
    like 'don't stop if no VERIFY pass has happened yet.'
    """

    def __init__(self, page_size: int = 64, hidden: int = 128, num_heads: int = 4):
        super().__init__()
        self.page_project = nn.Linear(page_size, hidden)
        self.blend_project = nn.Linear(NUM_MODES, hidden)  # 4-way blend, not 1
        self.attn = nn.MultiheadAttention(hidden, num_heads=num_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, hidden) * 0.02)
        self.output = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        state_pages: List[torch.Tensor],
        blend_history: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            state_pages: list of (B, page_size) tensors
            blend_history: list of (B, 4) tensors

        Returns:
            confidence: (B, 1) in [0, 1]
        """
        pages = torch.stack(state_pages, dim=1).float()      # (B, P, page_size)
        pages_proj = self.page_project(pages)                 # (B, P, hidden)

        blends = torch.stack(blend_history, dim=1).float()    # (B, P, 4)
        blend_proj = self.blend_project(blends)               # (B, P, hidden)
        pages_proj = pages_proj + blend_proj

        batch_size = pages.size(0)
        q = self.query.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 1, hidden)
        attended, _ = self.attn(query=q, key=pages_proj, value=pages_proj)
        return self.output(attended.squeeze(1))                # (B, 1)


# ---------------------------------------------------------------------------
# QuadLoRAModel
# ---------------------------------------------------------------------------
class QuadLoRAModel(nn.Module):
    """
    Four-mode LoRA model with parse/compute/verify/answer specialization.

    Same base as DualLoRAModel:
    - Llama 3.2 1B frozen
    - 7-layer perceiver compressor
    - Page-based state accumulation

    But uses QuadHypernetwork (4 template sets, 4-way softmax blend)
    and adds an AnswerHead for direct digit extraction.
    """

    def __init__(self, model_name: str = 'unsloth/Llama-3.2-1B'):
        super().__init__()

        # --- Frozen transformer ---
        self.transformer = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map='auto',
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        for p in self.transformer.parameters():
            p.requires_grad = False

        # --- Dimensions ---
        self.d_model = self.transformer.config.hidden_size         # 2048
        self.num_layers = self.transformer.config.num_hidden_layers  # 16
        num_kv_heads = self.transformer.config.num_key_value_heads   # 8
        head_dim = self.d_model // self.transformer.config.num_attention_heads
        d_kv = num_kv_heads * head_dim                              # 512

        self.page_size = 64
        self.strategy_size = 64
        self.page_radius = self.page_size ** 0.5

        # --- Compressor (same as dual) ---
        self.compressor = Compressor(
            num_transformer_layers=self.num_layers,
            d_transformer=self.d_model,
            d_perceiver=1024,
            num_queries=4,
            num_perceiver_layers=7,
            state_size=self.page_size,
            strategy_size=self.strategy_size,
        )

        # --- Quad hypernetwork ---
        self.hypernet = QuadHypernetwork(
            d_model=self.d_model,
            d_kv=d_kv,
            page_size=self.page_size,
            strategy_size=self.strategy_size,
            rank=4,
            num_layers=self.num_layers,
        )

        # --- Confidence head (4-way blend aware) ---
        self.confidence_head = QuadConfidenceHead(
            page_size=self.page_size, hidden=128, num_heads=4,
        )

        # --- Answer head (digit extraction from last page) ---
        self.answer_head = AnswerHead(
            page_size=self.page_size, max_digits=6,
        )

        # --- Probe head (same as dual, for intermediate value supervision) ---
        self.probe_head = nn.Linear(self.page_size, 1)

    def thinking_pass(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        state_pages: List[torch.Tensor],
        strategy: torch.Tensor,
        pass_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One thinking pass with quad LoRA.

        Returns:
            page: (B, page_size) normalized on hypersphere
            new_strategy: (B, strategy_size)
            blend: (B, 4) softmax blend weights
        """
        batch_size = input_ids.size(0)

        if len(state_pages) == 0:
            # First pass: no LoRA
            outputs = self.transformer(
                input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = list(outputs.hidden_states[1:])
            blend = torch.ones(
                batch_size, NUM_MODES, device=input_ids.device, dtype=strategy.dtype,
            ) / NUM_MODES
        else:
            all_mods, blend = self.hypernet(state_pages, strategy, pass_num=pass_num)
            manager = QuadAdditiveLoRAManager(self.transformer)
            manager.apply(all_mods, blend)
            try:
                outputs = self.transformer(
                    input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden_states = list(outputs.hidden_states[1:])
            finally:
                manager.remove()

        page_delta, new_strategy = self.compressor(hidden_states, pass_num)
        page = F.normalize(page_delta, dim=-1) * self.page_radius
        return page, new_strategy, blend


# ---------------------------------------------------------------------------
# Warm-start from dual checkpoint
# ---------------------------------------------------------------------------
def warm_start_quad_from_dual(
    quad_model: QuadLoRAModel,
    dual_checkpoint_path: str,
) -> None:
    """
    Warm-start a QuadLoRAModel from a dual LoRA checkpoint.

    Mapping:
    - compressor: loaded directly (same architecture)
    - confidence_head: fresh init (different blend_project input size: 4 vs 1)
    - hypernet:
        - dual forward templates -> quad compute templates
        - dual verify templates  -> quad verify templates
        - parse templates: small random init (from __init__)
        - answer templates: small random init (from __init__)
        - shared components (page_attn, page_project, etc.): loaded where shapes match
    - answer_head: fresh init (new component)
    - probe_head: loaded if present

    Args:
        quad_model: QuadLoRAModel to warm-start
        dual_checkpoint_path: path string OR already-loaded checkpoint dict
    """
    if isinstance(dual_checkpoint_path, dict):
        ckpt = dual_checkpoint_path
    else:
        ckpt = torch.load(dual_checkpoint_path, map_location='cpu', weights_only=True)

    # --- Compressor (same architecture, direct load) ---
    if 'compressor' in ckpt:
        own = quad_model.compressor.state_dict()
        loaded = 0
        for k, v in ckpt['compressor'].items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
                loaded += 1
        quad_model.compressor.load_state_dict(own, strict=False)
        print(f"  compressor: loaded {loaded}/{len(own)}")

    # --- Hypernet (map dual -> quad) ---
    if 'hypernet' in ckpt:
        quad_model.hypernet.warm_start_from_dual(ckpt['hypernet'])

    # --- Probe head ---
    if 'probe_head' in ckpt:
        own = quad_model.probe_head.state_dict()
        loaded = 0
        for k, v in ckpt['probe_head'].items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
                loaded += 1
        quad_model.probe_head.load_state_dict(own, strict=False)
        print(f"  probe_head: loaded {loaded}/{len(own)}")
    else:
        print(f"  probe_head: fresh init (not in checkpoint)")

    # --- Components with fresh init ---
    print(f"  confidence_head: fresh init (4-way blend, different input dim)")
    print(f"  answer_head: fresh init (new component)")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("QuadLoRA Architecture Self-Test")
    print("=" * 60)

    # --- Test QuadHypernetwork standalone ---
    print("\n--- QuadHypernetwork ---")
    d_model, d_kv, rank, num_layers = 64, 16, 4, 2
    h = QuadHypernetwork(
        d_model=d_model, d_kv=d_kv, page_size=64, strategy_size=64,
        rank=rank, num_layers=num_layers,
    )
    pages = [torch.randn(2, 64) for _ in range(3)]
    strategy = torch.randn(2, 64)

    all_mods, blend = h(pages, strategy, pass_num=1)
    print(f"  modes: {list(all_mods.keys())}")
    print(f"  layers per mode: {len(all_mods['parse'])}")
    print(f"  parse q_scales: {all_mods['parse'][0]['q_proj']['scales'].shape}")
    print(f"  blend shape: {blend.shape}")
    print(f"  blend values: {blend[0].tolist()}")
    print(f"  blend sum: {blend[0].sum().item():.6f}")
    assert abs(blend[0].sum().item() - 1.0) < 1e-5, "blend should sum to 1"

    # Empty pages
    all_mods_0, blend_0 = h([], strategy)
    print(f"  empty: parse sum={all_mods_0['parse'][0]['q_proj']['scales'].abs().sum():.4f}")
    print(f"  empty blend: {blend_0[0].tolist()}")

    # Param count
    total = sum(p.numel() for p in h.parameters())
    for mode in MODE_NAMES:
        A_list, B_list = h._template_lists[mode]
        mp = sum(p.numel() for p in list(A_list.parameters()) + list(B_list.parameters()))
        print(f"  {mode} templates: {mp:,}")
    shared = total - sum(
        sum(p.numel() for p in list(A.parameters()) + list(B.parameters()))
        for A, B in h._template_lists.values()
    )
    print(f"  shared (attn+combine+pass): {shared:,}")
    print(f"  total: {total:,}")

    # --- Test AnswerHead ---
    print("\n--- AnswerHead ---")
    ah = AnswerHead(page_size=64, max_digits=6)
    page = torch.randn(4, 64)
    sign, length, digits = ah(page)
    print(f"  sign: {sign.shape}, length: {length.shape}, digits: {len(digits)}x{digits[0].shape}")

    decoded = ah.decode(page)
    print(f"  decoded: {decoded.tolist()}")

    # Test loss
    gold = torch.tensor([42, -137, 0, 99999])
    loss = answer_head_loss(ah, page, gold)
    print(f"  loss on random page: {loss.item():.4f}")
    loss.backward()
    print(f"  gradient flows: {ah.sign_head.weight.grad is not None}")
    print(f"  answer_head params: {sum(p.numel() for p in ah.parameters()):,}")

    # --- Test QuadConfidenceHead ---
    print("\n--- QuadConfidenceHead ---")
    ch = QuadConfidenceHead(page_size=64, hidden=128, num_heads=4)
    pages_list = [torch.randn(2, 64) for _ in range(3)]
    blend_hist = [torch.randn(2, 4).softmax(dim=-1) for _ in range(3)]
    conf = ch(pages_list, blend_hist)
    print(f"  confidence: {conf.shape}, values: {conf.squeeze().tolist()}")
    print(f"  confidence_head params: {sum(p.numel() for p in ch.parameters()):,}")

    # --- Test QuadAdditiveLoRAManager ---
    print("\n--- QuadAdditiveLoRAManager ---")

    class MockSelfAttn(nn.Module):
        def __init__(self, d, dk):
            super().__init__()
            self.q_proj = nn.Linear(d, d, bias=False)
            self.k_proj = nn.Linear(d, dk, bias=False)
            self.v_proj = nn.Linear(d, dk, bias=False)
            self.o_proj = nn.Linear(d, d, bias=False)

    class MockLayer(nn.Module):
        def __init__(self, d, dk):
            super().__init__()
            self.self_attn = MockSelfAttn(d, dk)

    class MockLlama(nn.Module):
        def __init__(self, d=64, dk=16, n=2):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([MockLayer(d, dk) for _ in range(n)])

    d, dk, B_size, S = 64, 16, 2, 10
    mock_model = MockLlama(d, dk, num_layers)
    x = torch.randn(B_size, S, d)

    proj_dims_map = {'q_proj': d, 'k_proj': dk, 'v_proj': dk, 'o_proj': d}

    # Build all_mods manually for mock
    all_mods_mock = {}
    for mode in MODE_NAMES:
        mods = {}
        for li in range(num_layers):
            mods[li] = {}
            for pn, pd in proj_dims_map.items():
                mods[li][pn] = {
                    'A': torch.randn(d, rank) * 0.1,
                    'B': torch.randn(rank, pd) * 0.1,
                    'scales': torch.randn(B_size, rank).tanh(),
                }
        all_mods_mock[mode] = mods

    # Uniform blend
    blend_uniform = torch.ones(B_size, 4) / 4.0
    baseline_out = mock_model.model.layers[0].self_attn.q_proj(x)

    mgr = QuadAdditiveLoRAManager(mock_model)
    mgr.apply(all_mods_mock, blend_uniform)
    out_with_lora = mock_model.model.layers[0].self_attn.q_proj(x)
    mgr.remove()

    diff = (out_with_lora - baseline_out).norm()
    print(f"  baseline norm: {baseline_out.norm():.4f}")
    print(f"  lora diff from baseline: {diff:.4f}")
    assert diff > 0, "LoRA should change output"

    # Gradient flow through blend
    blend_leaf = torch.ones(B_size, 4, requires_grad=True)
    blend_g = blend_leaf / 4.0
    blend_g.retain_grad()
    x_g = torch.randn(B_size, S, d, requires_grad=True)
    mgr.apply(all_mods_mock, blend_g)
    out = mock_model.model.layers[0].self_attn.q_proj(x_g)
    out.sum().backward()
    mgr.remove()
    assert blend_leaf.grad is not None, "blend leaf should receive gradients"
    assert x_g.grad is not None, "input should receive gradients"
    print(f"  blend grad: {blend_leaf.grad[0].tolist()}")
    print(f"  gradients flow: OK")

    # Test blend extremes: pure mode should equal that mode's LoRA only
    for mode_idx, mode_name in enumerate(MODE_NAMES):
        pure_blend = torch.zeros(B_size, 4)
        pure_blend[:, mode_idx] = 1.0
        mgr.apply(all_mods_mock, pure_blend)
        out_pure = mock_model.model.layers[0].self_attn.q_proj(x)
        mgr.remove()
        # Verify it's different from uniform
        diff_from_uniform = (out_pure - out_with_lora).norm()
        print(f"  pure {mode_name} diff from uniform: {diff_from_uniform:.4f}")

    print("\n--- All QuadLoRA tests passed! ---")
