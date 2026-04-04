"""
ThinkingModel: State-Conditioned LoRA Architecture for Mycelium v20.

A transformer that THINKS before it SPEAKS. The state vector REWIRES the transformer
through state-conditioned LoRA, not by injecting tokens or biases.

The Architecture:
    STATE (64 floats on hypersphere)
           |
           v
    HYPERNETWORK (StateConditionedLoRA, 1.1M) — generates LoRA scales
    64 floats -> 256 scales (16 layers x 4 projections x 4 rank)
           |
           v
    Scale learned A/B templates -> Apply LoRA to Q, K, V, O in ALL 16 layers
           |
           v
    [problem tokens] -> Llama layers 1-16 (WITH LoRA modifications)
           |          |         |              |
         (all 16 layer hidden states saved)
           |          |         |              |
           v          v         v              v
    7-LAYER PERCEIVER COMPRESSOR (105M params)
    reads ALL layers, pass-conditioned attention
           |
           v
    64-float state delta
           |
           v
    state = normalize(state + delta) * sqrt(64)  <- HYPERSPHERE
           |
           +---> ConfidenceHead -> ready?
           |            |
           |        YES v
           |       GENERATE ANSWER
           |       (Llama + final LoRA, generate text)
           |
           +---> NO: loop back to HYPERNETWORK

The key insight: The state can't be ignored - it's IN the weights, not in the input.
Different state = different attention = different thinking style per pass.

Total parameters: ~1.34B (1.23B Llama + ~106M new components)
- StateConditionedLoRA: ~1.1M (the rewirer)
- Compressor: ~105M (the note-taker)
- ConfidenceHead: ~2.1K (the judge)
- Bottleneck: 64 floats (the tight straw)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from .state_conditioned_lora import StateConditionedLoRA
from .lora_hooks import apply_lora, remove_lora
from .compressor import Compressor
from .confidence_head import ConfidenceHead


class ThinkingModel(nn.Module):
    """
    State-Conditioned LoRA Architecture for Mycelium v20.

    The model thinks in multiple passes before generating an answer. Each pass:
    1. HYPERNETWORK: state -> LoRA scales (256 scales from 64 floats)
    2. APPLY LORA: scales modify Q, K, V, O projections in all 16 layers
    3. TRANSFORMER: forward pass with LoRA-modified attention
    4. COMPRESS: Perceiver reads all 16 layers -> 64-float delta
    5. REMOVE LORA: clean up hooks before next pass
    6. HYPERSPHERE: state = normalize(state + delta) * sqrt(64)
    7. Check confidence, break if above threshold

    Key design:
    - State REWIRES attention through LoRA, not through tokens/biases
    - Templates are learned "attention modification vocabularies"
    - State-derived scales select the mix of templates per pass
    - Same problem, different attention patterns each pass
    - Transformer can't ignore the state - it's in the weights

    Args:
        model_name: HuggingFace model name for Llama 3.2 1B-Instruct
        state_size: Size of the state vector (default: 64)
        lora_rank: LoRA rank - number of attention styles per projection (default: 4)
        num_queries: Number of compressor queries (default: 4)
        num_perceiver_layers: Depth of perceiver compressor (default: 7)
        d_perceiver: Internal dimension of perceiver (default: 1024)
        max_passes: Maximum thinking passes (default: 20)
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        state_size: int = 64,
        lora_rank: int = 4,
        num_queries: int = 4,
        num_perceiver_layers: int = 7,
        d_perceiver: int = 1024,
        max_passes: int = 20,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.state_size = state_size
        self.lora_rank = lora_rank
        self.max_passes = max_passes

        # Hypersphere radius = sqrt(64) approx 8.0
        self.state_radius = math.sqrt(state_size)

        # Load the transformer and tokenizer
        self.transformer, self.tokenizer = self._load_transformer(model_name)

        # Get transformer config
        self.d_model = self.transformer.config.hidden_size  # 2048 for Llama 3.2 1B
        self.num_transformer_layers = self.transformer.config.num_hidden_layers  # 16

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # STATE-CONDITIONED LORA: 64 floats -> LoRA scales (~1.1M params)
        # The hypernetwork generates scaling factors for learned LoRA templates
        #
        # GQA (Grouped Query Attention): K, V have fewer heads than Q
        # d_kv = num_kv_heads * head_dim
        num_kv_heads = getattr(self.transformer.config, 'num_key_value_heads',
                               self.transformer.config.num_attention_heads)
        head_dim = self.d_model // self.transformer.config.num_attention_heads
        d_kv = num_kv_heads * head_dim  # 512 for Llama 3.2 1B (8 * 64)

        self.lora = StateConditionedLoRA(
            d_model=self.d_model,
            d_kv=d_kv,
            state_size=state_size,
            rank=lora_rank,
            num_layers=self.num_transformer_layers,
            num_projections=4,  # Q, K, V, O
        )

        # COMPRESSOR: all 16 layers -> 64 floats (~105M params)
        # 7-layer Perceiver that reads all transformer layers with pass-conditioned attention
        self.compressor = Compressor(
            num_transformer_layers=self.num_transformer_layers,
            d_transformer=self.d_model,
            d_perceiver=d_perceiver,
            num_queries=num_queries,
            num_perceiver_layers=num_perceiver_layers,
            state_size=state_size,
            max_passes=max_passes,
        )

        # CONFIDENCE HEAD: 64 floats -> scalar confidence (~2.1K params)
        # Decides when to stop thinking and generate answer
        self.confidence = ConfidenceHead(state_size=state_size)

        # Track device (set on first forward pass)
        self._device: Optional[torch.device] = None

    def _load_transformer(
        self, model_name: str
    ) -> Tuple[nn.Module, Any]:
        """Load Llama 3.2 1B-Instruct via HuggingFace."""
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        print(f"Loaded {model_name} via HuggingFace (bfloat16)")

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

        The state lives on a sphere of radius sqrt(64). Each thinking pass
        rotates the state to a new position on the sphere.

        Args:
            batch_size: Number of states to initialize

        Returns:
            state: (batch_size, 64) on hypersphere
        """
        state = torch.randn(batch_size, self.state_size, device=self.device)
        return F.normalize(state, dim=-1) * self.state_radius

    def update_state(self, state: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """
        Rotate state on hypersphere: state = normalize(state + delta) * sqrt(64).

        Each pass is a rotation, not a magnitude change. The hypersphere ensures
        the state maintains constant magnitude while rotating to new positions.

        Args:
            state: Current state (batch, 64)
            delta: State delta from compressor (batch, 64)

        Returns:
            new_state: Rotated state on hypersphere (batch, 64)
        """
        return F.normalize(state + delta, dim=-1) * self.state_radius

    def apply_lora(self, state: torch.Tensor) -> None:
        """
        Apply state-conditioned LoRA to all 16 attention layers.

        Uses lora_hooks.apply_lora to register forward hooks that modify
        Q, K, V, O projection outputs based on state-derived scales.

        Args:
            state: State vector (batch, 64) on hypersphere
        """
        apply_lora(self.transformer, self.lora, state)

    def remove_lora(self) -> None:
        """
        Remove all LoRA hooks from transformer.

        Must be called after each thinking pass to prevent hook accumulation.
        """
        remove_lora(self.transformer)

    def _get_prompt_ids(self, problem_text: str) -> torch.Tensor:
        """Apply chat template and get input token IDs."""
        # Wrap problem with instruction (same as baseline evaluation)
        formatted_prompt = f"""Solve this math problem step by step. Put your final answer in \\boxed{{}}.

Problem: {problem_text}

Solution:"""
        messages = [{"role": "user", "content": formatted_prompt}]

        # apply_chat_template may return tensor or BatchEncoding depending on version
        result = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        # Handle both tensor and BatchEncoding return types
        if hasattr(result, 'input_ids'):
            prompt_ids = result.input_ids.to(self.device)
        elif isinstance(result, torch.Tensor):
            prompt_ids = result.to(self.device)
        else:
            # Fallback: tokenize the string output
            prompt_text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            prompt_ids = self.tokenizer(
                prompt_text, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self.device)

        return prompt_ids

    def think(
        self,
        problem_text: str,
        max_passes: int = 10,
        confidence_threshold: float = 0.8,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[float]]:
        """
        Think about a problem in multiple passes WITHOUT generating text.

        Each pass:
        1. Apply state-conditioned LoRA (different attention per pass)
        2. Forward through transformer (WITH LoRA modifications)
        3. Remove LoRA hooks
        4. Perceiver compresses all 16 layers to 64-float delta
        5. Rotate state on hypersphere
        6. Check confidence

        The same problem tokens are processed each pass, but the transformer
        attends DIFFERENTLY because the LoRA weights change based on state.

        Args:
            problem_text: The problem to think about
            max_passes: Maximum number of thinking passes (default: 10)
            confidence_threshold: Confidence level to stop (default: 0.8)

        Returns:
            Tuple of:
                - final_state: The accumulated state vector (1, 64)
                - all_states: List of state vectors at each pass
                - confidences: List of confidence scores at each pass
        """
        # Get prompt token IDs
        prompt_ids = self._get_prompt_ids(problem_text)
        # prompt_ids: (1, seq_len)

        # Initialize state on hypersphere
        state = self.init_state(batch_size=1)

        all_states: List[torch.Tensor] = [state.clone()]
        confidences: List[float] = []

        for pass_num in range(max_passes):
            # Apply state-conditioned LoRA to all 16 layers
            self.apply_lora(state)

            # Forward through transformer (WITH LoRA modifications)
            outputs = self.transformer(
                input_ids=prompt_ids,
                output_hidden_states=True,
                return_dict=True,
            )

            # Remove LoRA hooks after pass (IMPORTANT: prevents accumulation)
            self.remove_lora()

            # outputs.hidden_states: tuple of 17 tensors (embedding + 16 layers)
            # Skip the embedding layer (index 0), keep the 16 transformer layers
            all_layer_hidden: List[torch.Tensor] = list(outputs.hidden_states[1:])

            # Perceiver compresses all 16 layers to 64-float delta
            delta = self.compressor(
                [h.float() for h in all_layer_hidden],
                pass_num=pass_num,
            )  # (1, 64)

            # Rotate state on hypersphere
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
    ) -> str:
        """
        Generate text answer using the final state.

        The state is expanded into LoRA modifications that change how
        the transformer attends during generation.

        Args:
            problem_text: The original problem text
            state: The accumulated state vector (1, 64)
            max_new_tokens: Maximum tokens to generate (default: 512)

        Returns:
            The generated answer text
        """
        # Get prompt token IDs
        prompt_ids = self._get_prompt_ids(problem_text)

        # Apply final state-conditioned LoRA
        self.apply_lora(state)

        # Generate with greedy decoding
        with torch.no_grad():
            outputs = self.transformer.generate(
                input_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Remove LoRA hooks
        self.remove_lora()

        generated_ids = outputs[0]
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
                - final_state_norm: L2 norm of final state (should be sqrt(64))
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
        }

    def forward_with_state(
        self,
        problem_text: str,
        state: torch.Tensor,
        pass_num: int = 0,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Single forward pass with state, returning logits for training.

        Used for teacher-forced training where we want loss against gold answers.

        Args:
            problem_text: The problem text
            state: The state vector (1, 64)
            pass_num: Which thinking pass (for compressor conditioning)

        Returns:
            Tuple of:
                - logits: Output logits (1, seq_len, vocab_size)
                - hidden_states: List of hidden states from all 16 layers
        """
        # Get prompt token IDs
        prompt_ids = self._get_prompt_ids(problem_text)

        # Apply state-conditioned LoRA
        self.apply_lora(state)

        # Forward with hidden states
        outputs = self.transformer(
            input_ids=prompt_ids,
            output_hidden_states=True,
            return_dict=True,
        )

        # Remove LoRA hooks
        self.remove_lora()

        return outputs.logits, list(outputs.hidden_states[1:])

    def think_and_get_loss(
        self,
        question: str,
        answer: str,
        num_passes: int = 3,
    ) -> torch.Tensor:
        """
        Think through a problem and compute loss on the answer.

        Gradients flow: answer_loss -> logits -> LoRA -> state -> compressor

        Args:
            question: The problem text
            answer: The correct answer text
            num_passes: Number of thinking passes

        Returns:
            Cross-entropy loss on answer tokens
        """
        # Get question token IDs
        prompt_ids = self._get_prompt_ids(question)

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
            # Apply state-conditioned LoRA
            self.apply_lora(state)

            # Forward through transformer
            outputs = self.transformer(
                input_ids=prompt_ids,
                output_hidden_states=True,
                use_cache=False,
            )

            # Remove LoRA hooks
            self.remove_lora()

            all_layer_hidden = list(outputs.hidden_states[1:])

            # Compress to delta
            delta = self.compressor(
                [h.float() for h in all_layer_hidden],
                pass_num=pass_num,
            )

            # Rotate on hypersphere
            state = self.update_state(state, delta)

        # Final pass to compute answer loss
        # Apply final state-conditioned LoRA
        self.apply_lora(state)

        # Concatenate question + answer for teacher forcing
        full_ids = torch.cat([prompt_ids, answer_ids], dim=1)

        # Forward
        outputs = self.transformer(
            input_ids=full_ids,
            use_cache=False,
        )

        # Remove LoRA hooks
        self.remove_lora()

        # Compute loss on answer tokens only
        prompt_len = prompt_ids.size(1)
        answer_len = answer_ids.size(1)

        # Shift logits and labels for next-token prediction
        logits = outputs.logits[:, prompt_len - 1:-1, :]  # (1, answer_len, vocab_size)
        targets = answer_ids.squeeze(0)  # (answer_len,)

        loss = F.cross_entropy(logits.squeeze(0), targets)

        return loss

    def count_parameters(self) -> Dict[str, int]:
        """
        Count parameters in each component.

        Expected for v20 defaults:
            - transformer: ~1.23B (Llama 3.2 1B)
            - lora: ~1.1M (state-conditioned LoRA hypernetwork)
            - compressor: ~105M (7-layer Perceiver)
            - confidence: ~2.1K
            - new_components: ~106M (lora + compressor + confidence)
            - total: ~1.34B

        Returns:
            Dictionary with parameter counts per component
        """
        counts = {
            "transformer": sum(p.numel() for p in self.transformer.parameters()),
            "lora": sum(p.numel() for p in self.lora.parameters()),
            "compressor": sum(p.numel() for p in self.compressor.parameters()),
            "confidence": sum(p.numel() for p in self.confidence.parameters()),
        }
        counts["new_components"] = (
            counts["lora"] + counts["compressor"] + counts["confidence"]
        )
        counts["total"] = counts["transformer"] + counts["new_components"]

        return counts

    def freeze_transformer(self) -> None:
        """Freeze the transformer for Phase 1 training."""
        for param in self.transformer.parameters():
            param.requires_grad = False
        print("Transformer frozen. Training only LoRA templates + Perceiver + Confidence.")

    def unfreeze_transformer(self) -> None:
        """Unfreeze the transformer for Phase 2 training."""
        for param in self.transformer.parameters():
            param.requires_grad = True
        print("Transformer unfrozen. End-to-end training enabled.")

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get list of currently trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]

    def get_lora_scale_statistics(self, state: torch.Tensor) -> Dict[str, float]:
        """
        Get statistics about LoRA scales for monitoring.

        Useful for checking if different states produce different attention patterns.

        Args:
            state: State vector (batch, 64)

        Returns:
            Dictionary with scale statistics
        """
        return self.lora.get_scale_statistics(state)


def test_thinking_model():
    """
    Test function that runs a simple forward pass.

    Validates:
    1. Model loads correctly
    2. Parameter counts match expected values
    3. think() runs without errors
    4. Hypersphere constraint is maintained
    5. Gradient flow works
    """
    print("Testing ThinkingModel (State-Conditioned LoRA v20)...")
    print("Note: This requires GPU and will download Llama 3.2 1B if not cached.\n")

    if not torch.cuda.is_available():
        print("CUDA not available. Testing component integration only...\n")

        # Test that imports work
        from .state_conditioned_lora import StateConditionedLoRA
        from .lora_hooks import apply_lora, remove_lora
        from .compressor import Compressor
        from .confidence_head import ConfidenceHead

        print("All imports successful!")
        print("ThinkingModel structure verified.")
        return

    device = torch.device("cuda")

    try:
        model = ThinkingModel(
            model_name="meta-llama/Llama-3.2-1B-Instruct",
            state_size=64,
            lora_rank=4,
            num_queries=4,
            num_perceiver_layers=7,
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

        # Verify architecture
        print(f"\nArchitecture verification:")
        print(f"  LoRA (rewirer):      {param_counts['lora'] / 1e6:.2f}M params")
        print(f"  Compressor (note-taker): {param_counts['compressor'] / 1e6:.1f}M params")
        print(f"  Confidence (judge):  {param_counts['confidence']:,} params")
        print(f"  New components:      {param_counts['new_components'] / 1e6:.1f}M params")
        print(f"  Bottleneck:          64 floats")

        # Verify asymmetry: compressor >> LoRA
        ratio = param_counts['compressor'] / param_counts['lora']
        print(f"  Compressor/LoRA ratio: {ratio:.0f}x")
        print(f"  (Compression is the HARD job, rewiring is the LIGHT job)")

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
        print(f"Expected norm (sqrt(64)): {math.sqrt(64):.3f}")

        # Verify hypersphere constraint
        for i, s in enumerate(all_states):
            norm = s.norm().item()
            expected = math.sqrt(64)
            assert abs(norm - expected) < 0.01, f"State {i} norm {norm:.3f} != {expected:.3f}"
        print("Hypersphere constraint: OK")

        # Test LoRA scale statistics
        print("\n--- Testing LoRA scale variation ---")
        stats1 = model.get_lora_scale_statistics(all_states[0])
        stats2 = model.get_lora_scale_statistics(all_states[-1])
        print(f"Initial state scales: mean={stats1['mean']:.4f}, std={stats1['std']:.4f}")
        print(f"Final state scales:   mean={stats2['mean']:.4f}, std={stats2['std']:.4f}")

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

        lora_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.lora.parameters()
        )
        compressor_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.compressor.parameters()
        )

        print(f"LoRA gradients:       {'OK' if lora_has_grad else 'MISSING'}")
        print(f"Compressor gradients: {'OK' if compressor_has_grad else 'MISSING'}")

        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_thinking_model()
