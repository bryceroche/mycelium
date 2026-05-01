"""
Baked Llama: frozen math-mode LLM with KV cache and soft token injection.

L1 + L2 LoRA permanently absorbed into weights. No runtime modification.
KV cache valid for entire problem lifetime. Soft tokens from controller
injected as virtual prefix for each generation cycle.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class BakedLlama(nn.Module):
    """Frozen Llama with baked L1+L2 LoRA, KV cache, and soft token injection."""

    def __init__(self, model_name='unsloth/Llama-3.2-1B'):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        # Freeze everything
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

        self.d_model = self.model.config.hidden_size  # 2048
        self.num_layers = len(self.model.model.layers)

        # KV cache state (per problem)
        self._cached_kv = None
        self._cached_seq_len = 0

    def bake_lora(self, atoms, scale=0.46):
        """Permanently bake LoRA atoms into Llama's weights.

        Args:
            atoms: LoRAAtoms module with A/B parameter dicts
            scale: fixed scale for all atoms (0.46 = universal blend)
        """
        with torch.no_grad():
            for layer_idx in range(self.num_layers):
                for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    proj = getattr(
                        self.model.model.layers[layer_idx].self_attn,
                        proj_name,
                    )
                    A = atoms.A[proj_name][:, layer_idx]  # (64, d_model, rank)
                    B = atoms.B[proj_name][:, layer_idx]  # (64, rank, proj_dim)
                    delta = sum(scale * (A[i] @ B[i]) for i in range(A.shape[0]))
                    proj.weight.data += delta.T.to(dtype=proj.weight.dtype)

    def encode_problem(self, input_ids, attention_mask=None):
        """Encode problem ONCE. Cache KV. Return hidden states for controller.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len) optional

        Returns:
            hidden_states: list of (batch, seq_len, 2048) per layer
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=True,
            )
        self._cached_kv = outputs.past_key_values
        self._cached_seq_len = input_ids.size(1)
        return list(outputs.hidden_states)

    def generate_with_soft_tokens(self, soft_tokens, max_new_tokens=64,
                                   teacher_target_ids=None):
        """Generate using cached KV with soft token prefix.

        Soft tokens are prepended to the generation as virtual tokens.
        Llama attends to them alongside the cached problem tokens.

        Args:
            soft_tokens: (batch, N_soft, 2048) from controller's page projection
            max_new_tokens: max generation length
            teacher_target_ids: (batch, target_len) for teacher-forced training.
                If provided, returns logits instead of generating.

        Returns:
            If teacher_target_ids: gen_logits (batch, target_len, vocab)
            If not: generated token ids
        """
        if self._cached_kv is None:
            raise RuntimeError("Call encode_problem first")

        batch_size = soft_tokens.size(0)

        if teacher_target_ids is not None:
            # Teacher-forced: soft tokens + target tokens → logits
            target_embeds = self.model.model.embed_tokens(teacher_target_ids)
            # Concatenate soft tokens + target embeddings
            full_embeds = torch.cat([soft_tokens, target_embeds], dim=1)

            # Build attention mask: cached problem tokens + soft tokens + target tokens
            soft_len = soft_tokens.size(1)
            target_len = teacher_target_ids.size(1)
            # KV cache covers the problem tokens, so attention mask only needs new tokens
            new_attn = torch.ones(
                batch_size, soft_len + target_len,
                device=soft_tokens.device, dtype=torch.long,
            )
            # Full attention: cached_seq_len + soft_len + target_len
            full_attn = torch.ones(
                batch_size, self._cached_seq_len + soft_len + target_len,
                device=soft_tokens.device, dtype=torch.long,
            )

            outputs = self.model(
                inputs_embeds=full_embeds,
                attention_mask=full_attn,
                past_key_values=self._cached_kv,
                use_cache=False,
                output_hidden_states=False,
            )

            # Extract logits for target tokens only (skip soft token positions)
            gen_logits = outputs.logits[:, soft_len - 1:-1, :]
            return gen_logits

        else:
            # Free generation with soft tokens as prefix
            with torch.no_grad():
                # First pass: process soft tokens with cached KV
                soft_out = self.model(
                    inputs_embeds=soft_tokens,
                    past_key_values=self._cached_kv,
                    use_cache=True,
                )
                extended_kv = soft_out.past_key_values

                # Generate from extended KV
                # Start with the last token's prediction
                next_token = soft_out.logits[:, -1:, :].argmax(dim=-1)

                generated = [next_token]
                for _ in range(max_new_tokens - 1):
                    out = self.model(
                        input_ids=next_token,
                        past_key_values=extended_kv,
                        use_cache=True,
                    )
                    extended_kv = out.past_key_values
                    next_token = out.logits[:, -1:, :].argmax(dim=-1)
                    generated.append(next_token)

                    # Stop at EOS
                    if (next_token == self.tokenizer.eos_token_id).all():
                        break

                return torch.cat(generated, dim=1)

    def extend_cache(self, generated_ids):
        """Extend KV cache with generated tokens for next cycle.

        Args:
            generated_ids: (batch, gen_len) tokens from this cycle's generation
        """
        if self._cached_kv is None:
            raise RuntimeError("Call encode_problem first")

        with torch.no_grad():
            outputs = self.model(
                input_ids=generated_ids,
                past_key_values=self._cached_kv,
                use_cache=True,
            )
        self._cached_kv = outputs.past_key_values
        self._cached_seq_len += generated_ids.size(1)

    def reset_cache(self):
        """Clear KV cache (new problem)."""
        self._cached_kv = None
        self._cached_seq_len = 0


if __name__ == "__main__":
    print("BakedLlama — testing")
    llama = BakedLlama()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    llama = llama.to(device)

    # Test encode
    text = "Sam had 48 cookies and found 12 more. How many cookies does Sam have now?"
    toks = llama.tokenizer(text, return_tensors='pt').to(device)
    hidden = llama.encode_problem(toks['input_ids'])
    print(f"Encoded: {len(hidden)} layers, shape {hidden[0].shape}")
    print(f"KV cached: seq_len={llama._cached_seq_len}")

    # Test generate with soft tokens
    soft = torch.randn(1, 4, llama.d_model, device=device, dtype=torch.bfloat16)
    gen = llama.generate_with_soft_tokens(soft, max_new_tokens=30)
    print(f"Generated: {llama.tokenizer.decode(gen[0], skip_special_tokens=True)[:80]}")
    print("OK")
