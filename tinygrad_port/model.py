"""Full Mycelium breathing model assembled from components."""
from tinygrad import Tensor, dtypes
from tinygrad.nn import Linear
from tinygrad.nn.state import get_parameters
import math

from tinygrad_port.llama import Llama, load_llama_weights, freeze_llama
from tinygrad_port.perceiver import Perceiver
from tinygrad_port.hypernetwork import AtomHypernetwork
from tinygrad_port.lora import LoRAAtoms, AtomAdditiveLoRAInjector
from tinygrad_port.answer_head import AnswerHead, AtomConfidenceHead


class CycleMessageGenerator:
    """32-float direct signal from Llama's last layer, bypassing perceiver."""
    def __init__(self, d_model=2048, message_dim=32):
        self.message_dim = message_dim
        # LayerNorm + projection
        self.ln_weight = Tensor.ones(d_model).requires_grad_()
        self.ln_bias = Tensor.zeros(d_model).requires_grad_()
        self.w1 = (Tensor.randn(512, d_model) * (2 / d_model) ** 0.5).requires_grad_()
        self.b1 = Tensor.zeros(512).requires_grad_()
        self.w2 = (Tensor.randn(message_dim, 512) * (2 / 512) ** 0.5).requires_grad_()
        self.b2 = Tensor.zeros(message_dim).requires_grad_()

    def __call__(self, last_layer_hidden):
        # Mean pool over sequence
        pooled = last_layer_hidden.mean(axis=1).cast(dtypes.float32)
        # LayerNorm
        mean = pooled.mean(axis=-1, keepdim=True)
        var = pooled.var(axis=-1, keepdim=True)
        x = (pooled - mean) / (var + 1e-6).sqrt() * self.ln_weight + self.ln_bias
        # Project
        x = (x @ self.w1.T + self.b1).gelu()
        x = x @ self.w2.T + self.b2
        return x


class IsotropicRegularizer:
    """Zero-param regularizer: forces raw pages toward isotropic Gaussian."""
    def __init__(self, target_var=1.0, corr_weight=0.1):
        self.target_var = target_var
        self.corr_weight = corr_weight

    def __call__(self, pages_batch):
        if pages_batch.shape[0] < 4:
            return Tensor(0.0)
        dim_means = pages_batch.mean(axis=0)
        mean_loss = (dim_means * dim_means).mean()
        dim_vars = pages_batch.var(axis=0)
        var_loss = ((dim_vars - self.target_var) * (dim_vars - self.target_var)).mean()
        normalized = (pages_batch - dim_means) / (dim_vars.sqrt() + 1e-8)
        B = pages_batch.shape[0]
        correlation = normalized.T.matmul(normalized) / B
        D = pages_batch.shape[1]
        identity = Tensor.eye(D)
        off_diag = correlation - identity
        corr_loss = (off_diag * off_diag).mean()
        return mean_loss + var_loss + self.corr_weight * corr_loss


class AtomLoRAModel:
    """Full Mycelium breathing model.

    Components:
      - Llama 3.2 1B (frozen)
      - 7-layer Perceiver (105M)
      - 64 LoRA Atoms (82M)
      - Atom Hypernetwork (101M)
      - Answer Head (0.9M)
      - Confidence Head (2.5M)
      - Message Generator (1.1M)
      - Isotropic Regularizer (0 params)
    """
    def __init__(self, llama_weights_path=None):
        self.page_size = 64
        self.page_radius = math.sqrt(64)  # 8.0
        self.num_atoms = 64

        # Llama (frozen)
        self.llama = Llama()
        if llama_weights_path:
            load_llama_weights(self.llama, llama_weights_path)
        freeze_llama(self.llama)

        # Perceiver compressor
        self.compressor = Perceiver(page_size=self.page_size)

        # LoRA atoms
        self.atoms = LoRAAtoms(num_atoms=64, rank=6)
        self.lora_injector = AtomAdditiveLoRAInjector(self.atoms)

        # Hypernetwork (100M brain)
        self.hypernet = AtomHypernetwork(
            page_size=self.page_size, num_atoms=64,
            attn_dim=1024, num_query_heads=8, num_attn_heads=16,
            num_attn_layers=6, message_dim=32, max_messages=12,
        )

        # Answer head (per-cycle, 0.9M)
        self.answer_head = AnswerHead(page_size=self.page_size)

        # Confidence head (2.5M)
        self.confidence_head = AtomConfidenceHead(page_size=self.page_size)

        # Message generator (1.1M)
        self.message_generator = CycleMessageGenerator(d_model=2048, message_dim=32)

        # Isotropic regularizer (0 params)
        self.isotropic_reg = IsotropicRegularizer()

    def get_trainable_parameters(self):
        """Get all trainable parameters (excludes frozen Llama)."""
        params = []
        for component in [self.compressor, self.atoms, self.hypernet,
                          self.answer_head, self.confidence_head,
                          self.message_generator]:
            params.extend(get_parameters(component))
        return params

    def think_one_pass(self, input_ids, state_pages, pass_num,
                       messages=None):
        """One breathing cycle: expand (Llama) → collapse (perceiver) → record (notebook).

        Returns: (page, atom_scales, message, raw_page_delta)
        """
        batch_size = input_ids.shape[0]

        # Get atom scales from hypernetwork
        if len(state_pages) == 0:
            zero_page = Tensor.zeros(batch_size, self.page_size)
            atom_scales = self.hypernet([zero_page], pass_num=0, messages=messages)
        else:
            atom_scales = self.hypernet(state_pages, pass_num=pass_num, messages=messages)

        # Apply LoRA and run Llama
        self.lora_injector.set_scales(atom_scales)
        # NOTE: In practice, LoRA injection into Llama forward requires
        # modifying the attention computation inline. This is a simplified
        # version — the full implementation hooks into each layer.
        logits, hidden_states = self.llama(input_ids, output_hidden_states=True)
        self.lora_injector.clear_scales()

        # Perceiver compresses to page
        page_delta = self.compressor(hidden_states, pass_num)

        # Normalize to hypersphere
        page = page_delta / (page_delta.square().sum(axis=-1, keepdim=True).sqrt() + 1e-8)
        page = page * self.page_radius

        # Generate message (direct bypass)
        message = self.message_generator(hidden_states[-1])

        return page, atom_scales, message, page_delta
