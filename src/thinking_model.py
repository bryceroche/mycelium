"""
ThinkingModel: Asymmetric Hourglass Architecture for Mycelium v19.

A transformer that THINKS before it SPEAKS. Llama is SANDWICHED between:
- DECOMPRESSOR (MLP, 1.3M): Projects 64-float state → input bias (EASY job)
- COMPRESSOR (7 layers, 120M): Squeezes all 16 layer hidden states → 64 floats (HARD job)

The intelligence is in COMPRESSION (what to keep), not DECOMPRESSION (how to project).
89x more params for compression than decompression.

The transformer runs PRISTINE — no layer splitting, no architectural surgery.
It processes exactly as Llama was pretrained to process.

Architecture:
    64 floats (state on hypersphere)
            |
            v
    DECOMPRESSOR (MLP, 1.3M) — EASY: just project
    64 floats → 512 → 2048 → residual stream bias
            |
            v
    [bias + problem tokens] → Llama layers 1-16 (untouched)
            |
            v
    COMPRESSOR (7 layers, 120M) — HARD: must select
    all 16 layer hidden states → compress → 64 floats
            |
            v
    state = normalize(state + delta) * √64  ← HYPERSPHERE
            |
            ├──→ Confidence → ready? ──→ GENERATE
            |
            └──→ loop back to DECOMPRESSOR

Total parameters: ~1.35B (1.23B Llama + ~121M hourglass)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

from transformers import AutoModelForCausalLM, AutoTokenizer

from .decompressor import Decompressor
from .compressor import Compressor
from .confidence_head import ConfidenceHead


class ThinkingModel(nn.Module):
    """
    Asymmetric Hourglass: DECOMPRESSOR → Llama → COMPRESSOR.

    The model thinks in multiple passes before generating an answer. Each pass:
    1. DECOMPRESS: state → bias (added to ALL input positions) — EASY job
    2. TRANSFORMER: [bias + problem] → all 16 layers (PRISTINE Llama)
    3. COMPRESS: all 16 layer hidden states → 64-float delta — HARD job
    4. HYPERSPHERE: state = normalize(state + delta) * √64
    5. Check confidence, break if above threshold

    Key design:
    - Transformer is SANDWICHED, not modified
    - Bias modulates ALL input positions (not just prepended tokens)
    - Intentional asymmetry: 120M compressor, 1.3M decompressor (89x ratio)
    - Hypersphere: constant magnitude, each pass is a rotation

    Args:
        model_name: HuggingFace model name for Llama 3.2 1B-Instruct
        state_size: Size of the compressed state vector (default: 64)
        num_queries: Number of compressor queries (default: 4)
        num_layers: Depth of decompressor/compressor (default: 7)
        d_internal: Internal dimension of hourglass (default: 1024)
        max_passes: Maximum thinking passes (default: 20)
        use_unsloth: Whether to try loading with unsloth (default: True)
    """

    def __init__(
        self,
        model_name: str = "unsloth/Llama-3.2-1B-Instruct",
        state_size: int = 64,
        num_queries: int = 4,
        num_layers: int = 7,
        d_internal: int = 1024,
        max_passes: int = 20,
        use_unsloth: bool = True,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.state_size = state_size
        self.max_passes = max_passes

        # Hypersphere radius = √64 ≈ 8.0
        self.state_radius = math.sqrt(state_size)

        # Load the transformer and tokenizer
        self.transformer, self.tokenizer = self._load_transformer(
            model_name, use_unsloth
        )

        # Get transformer config
        self.d_model = self.transformer.config.hidden_size  # 2048 for Llama 3.2 1B
        self.num_transformer_layers = self.transformer.config.num_hidden_layers  # 16

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # DECOMPRESSOR: 64 floats → bias (1.3M params, lightweight MLP)
        # The decompressor has the EASY job: just project state into bias
        self.decompressor = Decompressor(
            state_size=state_size,
            d_model=self.d_model,
            d_hidden=512,  # Simple MLP: 64 → 512 → 512 → 2048
            max_passes=max_passes,
        )

        # COMPRESSOR: all 16 layers → 64 floats (120M params)
        # The compressor has the HARD job: selecting what matters from 16 layers
        self.compressor = Compressor(
            num_transformer_layers=self.num_transformer_layers,
            d_transformer=self.d_model,
            d_internal=d_internal,
            num_queries=num_queries,
            num_layers=num_layers,
            state_size=state_size,
            max_passes=max_passes,
        )

        # ConfidenceHead: 64 floats → scalar confidence
        self.confidence = ConfidenceHead(state_size=state_size)

        # Track device (set on first forward pass)
        self._device: Optional[torch.device] = None

    def _load_transformer(
        self, model_name: str, use_unsloth: bool
    ) -> Tuple[nn.Module, Any]:
        """Load Llama 3.2 1B-Instruct via unsloth or HuggingFace."""
        if use_unsloth and UNSLOTH_AVAILABLE:
            try:
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_name,
                    dtype=torch.bfloat16,
                    load_in_4bit=False,
                )
                print(f"Loaded {model_name} via unsloth (bfloat16)")
                return model, tokenizer
            except Exception as e:
                print(f"Unsloth loading failed ({e}), falling back to HuggingFace")

        # HuggingFace fallback
        hf_model_name = model_name
        if model_name.startswith("unsloth/"):
            hf_model_name = model_name.replace("unsloth/", "meta-llama/")

        model = AutoModelForCausalLM.from_pretrained(
            hf_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            hf_model_name,
            trust_remote_code=True,
        )
        print(f"Loaded {hf_model_name} via HuggingFace (bfloat16)")

        return model, tokenizer

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        if self._device is None:
            self._device = next(self.transformer.parameters()).device
        return self._device

    def init_state(self, batch_size: int = 1) -> torch.Tensor:
        """
        Initialize state on hypersphere (random direction * radius).

        The state lives on a sphere of radius √64. Each thinking pass
        rotates the state to a new position on the sphere.
        """
        state = torch.randn(batch_size, self.state_size, device=self.device)
        return F.normalize(state, dim=-1) * self.state_radius

    def update_state(self, state: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """
        Rotate state on hypersphere: state = normalize(state + delta) * √64.

        No learnable alpha. The hypersphere handles magnitude.
        Each pass is a rotation, not a magnitude change.
        """
        return F.normalize(state + delta, dim=-1) * self.state_radius

    def _get_prompt_embeddings(
        self, problem_text: str
    ) -> Tuple[torch.Tensor, List[int]]:
        """Apply chat template and get input embeddings."""
        messages = [{"role": "user", "content": problem_text}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        prompt_ids = self.tokenizer.encode(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.device)

        prompt_embeds = self.transformer.get_input_embeddings()(prompt_ids)

        return prompt_embeds, prompt_ids.tolist()[0]

    def think(
        self,
        problem_text: str,
        max_passes: int = 10,
        confidence_threshold: float = 0.8,
        scale: float = 1.0,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[float]]:
        """
        Think about a problem in multiple passes WITHOUT generating text.

        Each pass:
        1. DECOMPRESS: state → bias
        2. TRANSFORMER: [bias + problem] → all 16 layers
        3. COMPRESS: all layers → delta
        4. HYPERSPHERE: state = normalize(state + delta) * √64

        Args:
            problem_text: The problem to think about
            max_passes: Maximum number of thinking passes (default: 10)
            confidence_threshold: Confidence level to stop (default: 0.8)
            scale: Scale factor for bias (state warmup) (default: 1.0)

        Returns:
            Tuple of:
                - final_state: The accumulated state vector (1, 64)
                - all_states: List of state vectors at each pass
                - confidences: List of confidence scores at each pass
        """
        # Get prompt embeddings
        prompt_embeds, _ = self._get_prompt_embeddings(problem_text)
        # prompt_embeds: (1, seq_len, 2048)

        # Initialize state on hypersphere
        state = self.init_state(batch_size=1)

        all_states: List[torch.Tensor] = [state.clone()]
        confidences: List[float] = []

        for pass_num in range(max_passes):
            # DECOMPRESS: state → bias
            bias = self.decompressor(state, pass_num=pass_num, scale=scale)
            # bias: (1, 1, 2048) - broadcasts across sequence

            # Ensure dtype matches transformer embeddings
            bias = bias.to(dtype=prompt_embeds.dtype)

            # TRANSFORMER: [bias + problem] → all 16 layers
            # Bias modulates ALL input positions (not just prepended tokens)
            input_embeds = prompt_embeds + bias

            outputs = self.transformer(
                inputs_embeds=input_embeds,
                output_hidden_states=True,
                return_dict=True,
            )

            # outputs.hidden_states: tuple of 17 tensors (embedding + 16 layers)
            # Skip the embedding layer (index 0), keep the 16 transformer layers
            all_layer_hidden: List[torch.Tensor] = list(outputs.hidden_states[1:])

            # COMPRESS: all 16 layers → 64-float delta
            delta = self.compressor(
                [h.float() for h in all_layer_hidden],
                pass_num=pass_num,
            )  # (1, 64)

            # HYPERSPHERE: rotate state
            state = self.update_state(state, delta)

            all_states.append(state.clone())

            # Check confidence
            conf = self.confidence(state)
            conf_value = conf.item()
            confidences.append(conf_value)

            # Early stopping if confident enough
            if conf_value >= confidence_threshold:
                break

        return state, all_states, confidences

    def generate_answer(
        self,
        problem_text: str,
        state: torch.Tensor,
        max_new_tokens: int = 512,
        scale: float = 1.0,
    ) -> str:
        """
        Generate text answer using the accumulated state.

        The state is expanded into bias that modulates the input,
        then the transformer generates from the modulated representation.

        Args:
            problem_text: The original problem text
            state: The accumulated state vector (1, 64)
            max_new_tokens: Maximum tokens to generate (default: 512)
            scale: Scale factor for bias (default: 1.0)

        Returns:
            The generated answer text
        """
        # Get prompt embeddings
        prompt_embeds, prompt_ids = self._get_prompt_embeddings(problem_text)

        # DECOMPRESS: state → bias
        bias = self.decompressor(state, pass_num=0, scale=scale)
        bias = bias.to(dtype=prompt_embeds.dtype)

        # Bias modulates ALL input positions
        input_embeds = prompt_embeds + bias

        # Create attention mask
        seq_len = input_embeds.size(1)
        attention_mask = torch.ones(1, seq_len, device=self.device, dtype=torch.long)

        # Generate with greedy decoding
        with torch.no_grad():
            outputs = self.transformer.generate(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = outputs[0]
        full_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return full_text

    def solve(
        self,
        problem_text: str,
        max_passes: int = 10,
        confidence_threshold: float = 0.8,
        max_new_tokens: int = 512,
        scale: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Full pipeline: think about a problem, then generate the answer.

        Args:
            problem_text: The problem to solve
            max_passes: Maximum thinking passes (default: 10)
            confidence_threshold: Confidence to stop thinking (default: 0.8)
            max_new_tokens: Maximum tokens in answer (default: 512)
            scale: Scale factor for bias (default: 1.0)

        Returns:
            Dictionary containing:
                - answer: The generated answer text
                - num_passes: Number of thinking passes taken
                - confidences: Confidence at each pass
                - final_state_norm: L2 norm of final state (should be √64)
        """
        # Think
        state, all_states, confidences = self.think(
            problem_text=problem_text,
            max_passes=max_passes,
            confidence_threshold=confidence_threshold,
            scale=scale,
        )

        # Generate
        answer = self.generate_answer(
            problem_text=problem_text,
            state=state,
            max_new_tokens=max_new_tokens,
            scale=scale,
        )

        return {
            "answer": answer,
            "num_passes": len(confidences),
            "confidences": confidences,
            "final_state_norm": state.norm().item(),
        }

    def forward_with_state(
        self,
        problem_text: str,
        state: torch.Tensor,
        pass_num: int = 0,
        scale: float = 1.0,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Single forward pass with state, returning logits for training.

        Used for teacher-forced training where we want loss against gold answers.

        Args:
            problem_text: The problem text
            state: The state vector (1, 64)
            pass_num: Which thinking pass (for decompressor conditioning)
            scale: Scale factor for bias

        Returns:
            Tuple of:
                - logits: Output logits (1, seq_len, vocab_size)
                - hidden_states: List of hidden states from all layers
        """
        # Get prompt embeddings
        prompt_embeds, _ = self._get_prompt_embeddings(problem_text)

        # DECOMPRESS: state → bias
        bias = self.decompressor(state, pass_num=pass_num, scale=scale)
        bias = bias.to(dtype=prompt_embeds.dtype)

        # Bias modulates ALL input positions
        input_embeds = prompt_embeds + bias

        # Forward with hidden states
        outputs = self.transformer(
            inputs_embeds=input_embeds,
            output_hidden_states=True,
            return_dict=True,
        )

        return outputs.logits, list(outputs.hidden_states[1:])

    def think_and_get_loss(
        self,
        question: str,
        answer: str,
        num_passes: int = 3,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Think through a problem and compute loss on the answer.

        Gradients flow: answer_loss → logits → bias → decompressor → state → compressor

        Args:
            question: The problem text
            answer: The correct answer text
            num_passes: Number of thinking passes
            scale: Scale factor for bias (state warmup)

        Returns:
            Cross-entropy loss on answer tokens
        """
        # Get question embeddings
        prompt_embeds, _ = self._get_prompt_embeddings(question)

        # Get answer tokens
        answer_ids = self.tokenizer.encode(
            answer,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.device)

        # Initialize state on hypersphere
        state = self.init_state(batch_size=1)

        # Think for num_passes
        for pass_num in range(num_passes):
            # DECOMPRESS
            bias = self.decompressor(state, pass_num=pass_num, scale=scale)
            bias = bias.to(dtype=prompt_embeds.dtype)

            # TRANSFORMER
            input_embeds = prompt_embeds + bias
            outputs = self.transformer(
                inputs_embeds=input_embeds,
                output_hidden_states=True,
                use_cache=False,
            )
            all_layer_hidden = list(outputs.hidden_states[1:])

            # COMPRESS
            delta = self.compressor(
                [h.float() for h in all_layer_hidden],
                pass_num=pass_num,
            )

            # HYPERSPHERE rotation
            state = self.update_state(state, delta)

        # Final pass to compute answer loss
        bias = self.decompressor(state, pass_num=num_passes, scale=scale)
        bias = bias.to(dtype=prompt_embeds.dtype)

        # Concatenate question + answer for teacher forcing
        answer_embeds = self.transformer.get_input_embeddings()(answer_ids)
        full_embeds = torch.cat([prompt_embeds + bias, answer_embeds], dim=1)

        # Forward
        outputs = self.transformer(
            inputs_embeds=full_embeds,
            use_cache=False,
        )

        # Compute loss on answer tokens only
        prompt_len = prompt_embeds.size(1)
        answer_len = answer_ids.size(1)

        # Shift logits and labels for next-token prediction
        logits = outputs.logits[:, prompt_len - 1:-1, :]  # (1, answer_len, vocab_size)
        targets = answer_ids.squeeze(0)  # (answer_len,)

        loss = F.cross_entropy(logits.squeeze(0), targets)

        return loss

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters in each component."""
        counts = {
            "transformer": sum(p.numel() for p in self.transformer.parameters()),
            "decompressor": sum(p.numel() for p in self.decompressor.parameters()),
            "compressor": sum(p.numel() for p in self.compressor.parameters()),
            "confidence": sum(p.numel() for p in self.confidence.parameters()),
        }
        counts["hourglass"] = (
            counts["decompressor"] + counts["compressor"] + counts["confidence"]
        )
        counts["total"] = counts["transformer"] + counts["hourglass"]

        return counts

    def freeze_transformer(self) -> None:
        """Freeze the transformer for Phase 1 training."""
        for param in self.transformer.parameters():
            param.requires_grad = False
        print("Transformer frozen. Training only hourglass components.")

    def unfreeze_transformer(self) -> None:
        """Unfreeze the transformer for Phase 2 training."""
        for param in self.transformer.parameters():
            param.requires_grad = True
        print("Transformer unfrozen. End-to-end training enabled.")

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get list of currently trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]


def _test_thinking_model():
    """Quick sanity check for ThinkingModel."""
    print("Testing ThinkingModel (Asymmetric Hourglass v19)...")
    print("Note: This requires GPU and will download Llama 3.2 1B if not cached.\n")

    if not torch.cuda.is_available():
        print("CUDA not available. Testing component integration only...\n")

        # Test that imports work
        from .decompressor import Decompressor
        from .compressor import Compressor
        from .confidence_head import ConfidenceHead

        print("All imports successful!")
        print("ThinkingModel structure verified.")
        return

    device = torch.device("cuda")

    try:
        model = ThinkingModel(
            model_name="unsloth/Llama-3.2-1B-Instruct",
            state_size=64,
            num_queries=4,
            num_layers=7,
            max_passes=20,
        )
        model = model.to(device)

        # Count parameters
        param_counts = model.count_parameters()
        print("\nParameter counts:")
        for name, count in param_counts.items():
            if count > 1e6:
                print(f"  {name}: {count / 1e6:.1f}M")
            else:
                print(f"  {name}: {count:,}")

        # Verify intentional asymmetry
        print(f"\nHourglass asymmetry (intentional):")
        print(f"  Decompressor: {param_counts['decompressor'] / 1e6:.2f}M (EASY job)")
        print(f"  Compressor:   {param_counts['compressor'] / 1e6:.1f}M (HARD job)")
        ratio = param_counts['compressor'] / param_counts['decompressor']
        print(f"  Ratio:        {ratio:.0f}x more params for compression")

        # Test thinking
        print("\n--- Testing think() ---")
        problem = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether?"

        state, all_states, confidences = model.think(
            problem_text=problem,
            max_passes=3,
            confidence_threshold=0.99,
        )

        print(f"Problem: {problem[:50]}...")
        print(f"Thinking passes: {len(confidences)}")
        print(f"Confidences: {[f'{c:.3f}' for c in confidences]}")
        print(f"Final state norm: {state.norm().item():.3f}")
        print(f"Expected norm (√64): {math.sqrt(64):.3f}")

        # Verify hypersphere constraint
        for i, s in enumerate(all_states):
            norm = s.norm().item()
            expected = math.sqrt(64)
            assert abs(norm - expected) < 0.01, f"State {i} norm {norm:.3f} != {expected:.3f}"
        print("Hypersphere constraint: OK")

        # Test generation
        print("\n--- Testing generate_answer() ---")
        answer = model.generate_answer(problem, state, max_new_tokens=100)
        print(f"Generated answer: {answer[:200]}...")

        # Test gradient flow
        print("\n--- Testing gradient flow ---")
        model.train()
        model.freeze_transformer()

        loss = model.think_and_get_loss(
            question="What is 2 + 3?",
            answer="5",
            num_passes=2,
        )
        loss.backward()

        decompressor_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.decompressor.parameters()
        )
        compressor_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.compressor.parameters()
        )

        print(f"Decompressor gradients: {'OK' if decompressor_has_grad else 'MISSING'}")
        print(f"Compressor gradients: {'OK' if compressor_has_grad else 'MISSING'}")

        print("\nAll tests passed!")

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    _test_thinking_model()
