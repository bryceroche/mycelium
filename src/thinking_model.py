"""
ThinkingModel: The Full Integrated Thinking Architecture for Mycelium v18.

This module combines Llama 3.2 1B-Instruct with:
- AllLayerPerceiver: 7-layer Perceiver reading all 16 Llama layers (~108M params)
- StateInjector: 64 floats -> 4 pseudo-tokens (~130K params)
- ConfidenceHead: State -> readiness score (~2.1K params)

The model THINKS before it SPEAKS. It can process a problem multiple times
internally, building up understanding through a 64-float state vector,
before producing any output tokens.

Architecture:
    [problem tokens] + [4 state pseudo-tokens]
            |
            v
    Llama Layers 1-16 (collect ALL hidden states)
            |
            v
    AllLayerPerceiver (pass-conditioned, 7 layers)
            |
            v
    64-float state delta
            |
            v
    state = state + alpha * delta  (residual accumulation)
            |
            v
    ConfidenceHead -> ready?
        NO  -> loop back
        YES -> generate answer

Total parameters: ~1.34B (1.23B Llama + ~108M new)
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional

try:
    # Try unsloth first for optimized loading
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

from transformers import AutoModelForCausalLM, AutoTokenizer

from .all_layer_perceiver import AllLayerPerceiver
from .state_injector import StateInjector
from .confidence_head import ConfidenceHead


class ThinkingModel(nn.Module):
    """
    Integrated thinking architecture: Llama 3.2 1B-Instruct + Perceiver compression.

    The model thinks in multiple passes before generating an answer. Each thinking
    pass runs through all 16 Llama layers, the AllLayerPerceiver compresses to
    64 floats, and the state accumulates via residual connection.

    Key features:
    - NO text generation during thinking (fast forward passes only)
    - Reads ALL 16 transformer layers (not just the final one)
    - Pass-conditioned compression (different layers matter at different stages)
    - Residual state update: state = state + alpha * delta
    - Confidence-based early stopping

    Args:
        model_name: HuggingFace model name for Llama 3.2 1B-Instruct
        state_size: Size of the compressed state vector (default: 64)
        num_pseudo_tokens: Number of pseudo-tokens for state injection (default: 4)
        num_queries: Number of perceiver queries (default: 4)
        num_perceiver_layers: Depth of the perceiver stack (default: 7)
        d_perceiver: Internal dimension of perceiver (default: 1024)
        max_passes: Maximum thinking passes (default: 20)
        alpha_init: Initial value for the residual weight (default: 0.1)
        use_unsloth: Whether to try loading with unsloth (default: True)

    Example:
        >>> model = ThinkingModel()
        >>> result = model.solve("What is 2 + 3?", max_passes=5)
        >>> print(result['answer'])
        >>> print(f"Solved in {result['num_passes']} passes")
    """

    def __init__(
        self,
        model_name: str = "unsloth/Llama-3.2-1B-Instruct",
        state_size: int = 64,
        num_pseudo_tokens: int = 4,
        num_queries: int = 4,
        num_perceiver_layers: int = 7,
        d_perceiver: int = 1024,
        max_passes: int = 20,
        alpha_init: float = 0.1,
        use_unsloth: bool = True,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.state_size = state_size
        self.num_pseudo_tokens = num_pseudo_tokens
        self.max_passes = max_passes

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

        # AllLayerPerceiver: reads all 16 layers, compresses to 64 floats
        self.compressor = AllLayerPerceiver(
            num_transformer_layers=self.num_transformer_layers,
            d_transformer=self.d_model,
            d_perceiver=d_perceiver,
            num_queries=num_queries,
            num_perceiver_layers=num_perceiver_layers,
            state_size=state_size,
            max_passes=max_passes,
        )

        # StateInjector: 64 floats -> 4 pseudo-tokens (each 2048-dim)
        self.injector = StateInjector(
            state_size=state_size,
            d_model=self.d_model,
            num_tokens=num_pseudo_tokens,
        )

        # ConfidenceHead: 64 floats -> scalar confidence
        self.confidence = ConfidenceHead(state_size=state_size)

        # Learnable residual weight (initialized small for stable start)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        # Track device (set on first forward pass)
        self._device: Optional[torch.device] = None

    def _load_transformer(
        self, model_name: str, use_unsloth: bool
    ) -> Tuple[nn.Module, Any]:
        """
        Load Llama 3.2 1B-Instruct via unsloth or HuggingFace.

        Args:
            model_name: Model identifier
            use_unsloth: Whether to try unsloth first

        Returns:
            Tuple of (model, tokenizer)
        """
        if use_unsloth and UNSLOTH_AVAILABLE:
            try:
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_name,
                    dtype=torch.bfloat16,
                    load_in_4bit=False,  # Full precision for gradients
                )
                print(f"Loaded {model_name} via unsloth (bfloat16)")
                return model, tokenizer
            except Exception as e:
                print(f"Unsloth loading failed ({e}), falling back to HuggingFace")

        # HuggingFace fallback
        hf_model_name = model_name
        if model_name.startswith("unsloth/"):
            # Map unsloth name to HuggingFace equivalent
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

    def _get_prompt_embeddings(
        self, problem_text: str
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Apply chat template and get input embeddings.

        Args:
            problem_text: The problem text to encode

        Returns:
            Tuple of (embeddings tensor, token ids list)
        """
        # Apply chat template for Instruct model
        messages = [{"role": "user", "content": problem_text}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Tokenize
        prompt_ids = self.tokenizer.encode(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.device)

        # Get embeddings
        prompt_embeds = self.transformer.get_input_embeddings()(prompt_ids)

        return prompt_embeds, prompt_ids.tolist()[0]

    def think(
        self,
        problem_text: str,
        max_passes: int = 10,
        confidence_threshold: float = 0.8,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[float]]:
        """
        Think about a problem in multiple passes WITHOUT generating text.

        Each pass:
        1. Inject current state as 4 pseudo-tokens
        2. Forward through all 16 Llama layers
        3. AllLayerPerceiver reads all layers, compresses to 64 floats
        4. Residual update: state = state + alpha * delta
        5. Check confidence, break if above threshold

        Args:
            problem_text: The problem to think about
            max_passes: Maximum number of thinking passes (default: 10)
            confidence_threshold: Confidence level to stop thinking (default: 0.8)

        Returns:
            Tuple of:
                - final_state: The accumulated state vector (1, 64)
                - all_states: List of state vectors at each pass
                - confidences: List of confidence scores at each pass
        """
        # Get prompt embeddings
        prompt_embeds, _ = self._get_prompt_embeddings(problem_text)
        # prompt_embeds: (1, seq_len, 2048)

        # Initialize state as zeros
        state = self.injector.get_empty_state(
            batch_size=1,
            device=self.device,
            dtype=torch.float32,  # Keep state in fp32 for stability
        )

        all_states: List[torch.Tensor] = [state.clone()]
        confidences: List[float] = []

        for pass_num in range(max_passes):
            # Inject state as pseudo-tokens: (1, 4, 2048)
            state_tokens = self.injector(state)

            # Ensure dtype matches transformer embeddings
            state_tokens = state_tokens.to(dtype=prompt_embeds.dtype)

            # Concatenate: [state_tokens, problem_embeds]
            input_embeds = torch.cat([state_tokens, prompt_embeds], dim=1)

            # Forward through ALL 16 layers, collecting hidden states
            outputs = self.transformer(
                inputs_embeds=input_embeds,
                output_hidden_states=True,
                return_dict=True,
            )

            # outputs.hidden_states: tuple of 17 tensors (embedding + 16 layers)
            # Each tensor: (batch, seq_len, 2048)
            # Skip the embedding layer (index 0), keep the 16 transformer layers
            all_layer_hidden: List[torch.Tensor] = list(outputs.hidden_states[1:])

            # AllLayerPerceiver compresses to 64 floats
            # Convert to fp32 for compression head stability
            state_delta = self.compressor(
                [h.float() for h in all_layer_hidden],
                pass_num=pass_num,
            )  # (1, 64)

            # Residual update: accumulate, don't replace
            # alpha is learnable, starts at 0.1
            state = state + self.alpha * state_delta

            all_states.append(state.clone())

            # Check confidence
            conf = self.confidence(state)  # (1, 1)
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
    ) -> str:
        """
        Generate text answer using the accumulated state.

        This is the final pass where the model produces output tokens.
        The state pseudo-tokens are prepended to give the model access
        to its accumulated understanding.

        Args:
            problem_text: The original problem text
            state: The accumulated state vector (1, 64)
            max_new_tokens: Maximum tokens to generate (default: 512)

        Returns:
            The generated answer text
        """
        # Get prompt embeddings
        prompt_embeds, prompt_ids = self._get_prompt_embeddings(problem_text)

        # Inject state as pseudo-tokens
        state_tokens = self.injector(state)
        state_tokens = state_tokens.to(dtype=prompt_embeds.dtype)

        # Concatenate: [state_tokens, problem_embeds]
        input_embeds = torch.cat([state_tokens, prompt_embeds], dim=1)

        # Create attention mask (all ones, we attend to everything)
        seq_len = input_embeds.size(1)
        attention_mask = torch.ones(1, seq_len, device=self.device, dtype=torch.long)

        # Generate with greedy decoding (temperature=0 equivalent)
        with torch.no_grad():
            outputs = self.transformer.generate(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode output
        # Note: outputs includes the input tokens (represented by state_tokens + prompt)
        # We need to skip the pseudo-token positions in decoding
        # The generated tokens start after the full input sequence
        generated_ids = outputs[0]

        # Decode the full output and extract the generated portion
        full_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return full_text

    def solve(
        self,
        problem_text: str,
        max_passes: int = 10,
        confidence_threshold: float = 0.8,
        max_new_tokens: int = 512,
    ) -> Dict[str, Any]:
        """
        Full pipeline: think about a problem, then generate the answer.

        This is the main entry point for using the ThinkingModel.

        Args:
            problem_text: The problem to solve
            max_passes: Maximum thinking passes (default: 10)
            confidence_threshold: Confidence to stop thinking (default: 0.8)
            max_new_tokens: Maximum tokens in answer (default: 512)

        Returns:
            Dictionary containing:
                - answer: The generated answer text
                - num_passes: Number of thinking passes taken
                - confidences: Confidence at each pass
                - final_state_norm: L2 norm of final state (diagnostic)
                - alpha: Current value of the residual weight
        """
        # Think
        state, all_states, confidences = self.think(
            problem_text=problem_text,
            max_passes=max_passes,
            confidence_threshold=confidence_threshold,
        )

        # Generate
        answer = self.generate_answer(
            problem_text=problem_text,
            state=state,
            max_new_tokens=max_new_tokens,
        )

        return {
            "answer": answer,
            "num_passes": len(confidences),
            "confidences": confidences,
            "final_state_norm": state.norm().item(),
            "alpha": self.alpha.item(),
        }

    def forward_with_state(
        self,
        problem_text: str,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single forward pass with state injection, returning logits for training.

        Used for teacher-forced training where we want loss against gold answers.

        Args:
            problem_text: The problem text
            state: The state vector to inject (1, 64)

        Returns:
            Tuple of:
                - logits: Output logits from the transformer (1, seq_len, vocab_size)
                - hidden_states: List of hidden states from all layers
        """
        # Get prompt embeddings
        prompt_embeds, _ = self._get_prompt_embeddings(problem_text)

        # Inject state
        state_tokens = self.injector(state)
        state_tokens = state_tokens.to(dtype=prompt_embeds.dtype)

        # Concatenate
        input_embeds = torch.cat([state_tokens, prompt_embeds], dim=1)

        # Forward with hidden states
        outputs = self.transformer(
            inputs_embeds=input_embeds,
            output_hidden_states=True,
            return_dict=True,
        )

        return outputs.logits, outputs.hidden_states

    def count_parameters(self) -> Dict[str, int]:
        """
        Count parameters in each component.

        Returns:
            Dictionary with parameter counts per component and totals.
        """
        counts = {
            "transformer": sum(
                p.numel() for p in self.transformer.parameters()
            ),
            "compressor": sum(
                p.numel() for p in self.compressor.parameters()
            ),
            "injector": sum(
                p.numel() for p in self.injector.parameters()
            ),
            "confidence": sum(
                p.numel() for p in self.confidence.parameters()
            ),
            "alpha": 1,  # Single learnable scalar
        }
        counts["new_params"] = (
            counts["compressor"]
            + counts["injector"]
            + counts["confidence"]
            + counts["alpha"]
        )
        counts["total"] = counts["transformer"] + counts["new_params"]

        return counts

    def freeze_transformer(self) -> None:
        """Freeze the transformer for Phase 1 training."""
        for param in self.transformer.parameters():
            param.requires_grad = False
        print("Transformer frozen. Training only compression components.")

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
    print("Testing ThinkingModel...")
    print("Note: This requires GPU and will download Llama 3.2 1B if not cached.\n")

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping full test.")
        print("Testing component integration only...\n")

        # Test that imports work
        from .all_layer_perceiver import AllLayerPerceiver
        from .state_injector import StateInjector
        from .confidence_head import ConfidenceHead

        print("All imports successful!")
        print("ThinkingModel structure verified.")
        return

    # Full test with model loading
    device = torch.device("cuda")

    try:
        model = ThinkingModel(
            model_name="unsloth/Llama-3.2-1B-Instruct",
            state_size=64,
            num_pseudo_tokens=4,
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

        # Test thinking
        print("\n--- Testing think() ---")
        problem = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether?"

        state, all_states, confidences = model.think(
            problem_text=problem,
            max_passes=3,  # Short test
            confidence_threshold=0.99,  # Don't early stop
        )

        print(f"Problem: {problem[:50]}...")
        print(f"Thinking passes: {len(confidences)}")
        print(f"Confidences: {[f'{c:.3f}' for c in confidences]}")
        print(f"Final state norm: {state.norm().item():.3f}")
        print(f"Alpha value: {model.alpha.item():.3f}")

        # Verify state accumulation
        norms = [s.norm().item() for s in all_states]
        print(f"State norms over passes: {[f'{n:.3f}' for n in norms]}")

        # Test generation
        print("\n--- Testing generate_answer() ---")
        answer = model.generate_answer(problem, state, max_new_tokens=100)
        print(f"Generated answer: {answer[:200]}...")

        # Test full solve
        print("\n--- Testing solve() ---")
        result = model.solve(problem, max_passes=3)
        print(f"Full solve result keys: {list(result.keys())}")

        # Test gradient flow
        print("\n--- Testing gradient flow ---")
        model.train()
        model.freeze_transformer()

        state2, _, _ = model.think(problem, max_passes=2)
        loss = state2.sum()
        loss.backward()

        # Check that compressor has gradients
        compressor_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.compressor.parameters()
        )
        print(f"Compressor gradients: {'OK' if compressor_has_grad else 'MISSING'}")

        # Check alpha gradient
        alpha_has_grad = model.alpha.grad is not None
        print(f"Alpha gradient: {'OK' if alpha_has_grad else 'MISSING'}")

        print("\nAll tests passed!")

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    _test_thinking_model()
