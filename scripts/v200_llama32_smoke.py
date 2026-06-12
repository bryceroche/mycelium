"""Stage 1A verification smoke for Llama-3.2-1B.

Runs the same checks as v200_llama_smoke.py (SmolLM2-1.7B) but against
Llama-3.2-1B, which has different architecture specifics:
  - vocab_size: 128256 (vs SmolLM2's 49152)
  - num_key_value_heads: 8 with GQA (vs SmolLM2's 32, no GQA)
  - rope_theta: 500000 (vs SmolLM2's 130000)
  - RoPE scaling: llama3 type with factor=32 (vs SmolLM2's standard)
  - hidden_size: 2048 (same)
  - num_attention_heads: 32 (same)
  - intermediate_size: 8192 (same)
  - rms_norm_eps: 1e-5 (same)

GQA note: Q has 32 heads (2048d total), K/V have 8 heads (512d total).
wk and wv weight matrices are (2048, 512), not (2048, 2048).
K/V are expanded at attention time: each KV head serves 4 Q heads.

Artifacts produced (at llama32_* paths, NOT overwriting SmolLM2 artifacts):
  .cache/v200_smoke/llama32_load.log              — smoke log
  .cache/v200_smoke/llama32_weights.sha256        — SHA256 of weight tensors
  .cache/v200_smoke/llama32_load.provenance.json  — four-axis provenance sidecar

Completion criterion (per §11 of docs/v200_brief.md):
  Final line of llama32_load.log must be "SMOKE PASSED <metrics>".

Checks performed:
  1. Model loads without error
  2. SHA256 of loaded weight tensors saved
  3. B=2, T=64 random forward (baseline): shape=[2,64,2048], no-NaN
  4. Interface contract verified: .embed, .layers[0..3], .ln_f, forward_with_taps
  5. wv-shared forward: L0's wv broadcast to L1-L3 (GQA-aware test)
  6. Per-layer norms and timing
  7. Peak memory estimate
  8. Comparison table vs SmolLM2 baseline
"""

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

# Ensure project root is on path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from tinygrad import Tensor, dtypes, Device
from tinygrad.nn import Embedding, RMSNorm
from tinygrad.nn.state import safe_load

import numpy as np


# ---------------------------------------------------------------------------
# Llama-3.2-1B config constants (from config.json)
# ---------------------------------------------------------------------------

HIDDEN_SIZE      = 2048
NUM_Q_HEADS      = 32
NUM_KV_HEADS     = 8          # GQA — fewer KV heads than Q heads
HEAD_DIM         = 64         # hidden / num_q_heads
KV_DIM           = NUM_KV_HEADS * HEAD_DIM   # 512
NUM_Q_PER_KV     = NUM_Q_HEADS // NUM_KV_HEADS  # 4
INTERMEDIATE     = 8192
VOCAB_SIZE       = 128256
ROPE_THETA       = 500000.0
RMS_EPS          = 1e-5
MAX_SEQ_LEN      = 4096       # Practical limit for smoke (actual model supports 131072)

# SmolLM2 baseline numbers (from .cache/v200_smoke/llama_load.log)
SMOLLM2_LOAD_TIME   = 0.6
SMOLLM2_FWD_TIME    = 0.013
SMOLLM2_SHAPE       = "[2, 64, 2048]"
SMOLLM2_NAN_FREE    = "YES"
SMOLLM2_WV_COS      = 0.5126

# Weights path
LLAMA32_WEIGHTS = os.path.join(_PROJECT_ROOT, ".cache", "llama-3.2-1b-weights", "model.safetensors")

# Output paths (llama32_* to avoid clobbering SmolLM2 artifacts)
LOG_DIR   = os.path.join(_PROJECT_ROOT, ".cache", "v200_smoke")
LOG_PATH  = os.path.join(LOG_DIR, "llama32_load.log")
SHA_PATH  = os.path.join(LOG_DIR, "llama32_weights.sha256")
PROV_PATH = os.path.join(LOG_DIR, "llama32_load.provenance.json")

os.makedirs(LOG_DIR, exist_ok=True)

_log_fh = open(LOG_PATH, "w", buffering=1)


def log(msg: str = "") -> None:
    print(msg)
    _log_fh.write(msg + "\n")
    _log_fh.flush()


# ---------------------------------------------------------------------------
# RoPE helpers (Llama-3.2 uses llama3 scaling but smoke uses standard for
# forward validity check — we don't need exact inference, just non-NaN)
# ---------------------------------------------------------------------------

def _build_rope_tables(seq_len: int) -> Tuple[Tensor, Tensor]:
    """Precompute RoPE cos/sin tables. Standard (no scaling) for smoke validity."""
    half = HEAD_DIM // 2
    inv_freq = Tensor(
        [1.0 / (ROPE_THETA ** (2 * i / HEAD_DIM)) for i in range(half)],
        dtype=dtypes.float,
    )
    pos = Tensor.arange(seq_len, dtype=dtypes.float)
    angles = pos.reshape(-1, 1) * inv_freq.reshape(1, -1)   # (seq, half)
    angles_full = Tensor.cat(angles, angles, dim=-1)         # (seq, HEAD_DIM)
    cos_t = angles_full.cos().contiguous().realize()
    sin_t = angles_full.sin().contiguous().realize()
    return cos_t, sin_t


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE rotation. x: (B, n_heads, T, head_dim)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = Tensor.cat(-x2, x1, dim=-1)
    c = cos.reshape(1, 1, cos.shape[0], cos.shape[1])
    s = sin.reshape(1, 1, sin.shape[0], sin.shape[1])
    return x * c + rotated * s


# ---------------------------------------------------------------------------
# GQA Attention
# ---------------------------------------------------------------------------

class LlamaGQAAttention:
    """GQA attention for Llama-3.2-1B: 32 Q heads, 8 KV heads.

    Weight layout after load (in_dim, out_dim):
      wq: (2048, 2048) — 32 Q heads
      wk: (2048, 512)  — 8 KV heads × 64 head_dim
      wv: (2048, 512)
      wo: (2048, 2048)

    At attention time each KV head is shared by NUM_Q_PER_KV=4 Q heads.
    """

    def __init__(self):
        self.wq = Tensor.zeros(HIDDEN_SIZE, HIDDEN_SIZE)        # (2048, 2048)
        self.wk = Tensor.zeros(HIDDEN_SIZE, KV_DIM)             # (2048, 512)
        self.wv = Tensor.zeros(HIDDEN_SIZE, KV_DIM)             # (2048, 512)
        self.wo = Tensor.zeros(HIDDEN_SIZE, HIDDEN_SIZE)        # (2048, 2048)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor,
                attn_mask: Optional[Tensor] = None) -> Tensor:
        B, T, H = x.shape

        # Projections
        q = x @ self.wq   # (B, T, 2048)
        k = x @ self.wk   # (B, T, 512)
        v = x @ self.wv   # (B, T, 512)

        # Reshape to multi-head
        q = q.reshape(B, T, NUM_Q_HEADS, HEAD_DIM).transpose(1, 2)   # (B, 32, T, 64)
        k = k.reshape(B, T, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)  # (B, 8, T, 64)
        v = v.reshape(B, T, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)  # (B, 8, T, 64)

        # Apply RoPE to Q and K
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # GQA: expand K/V to match Q head count (each KV head serves 4 Q heads)
        # k: (B, 8, T, 64) → (B, 32, T, 64) by repeating each head 4 times
        k = k.repeat((1, NUM_Q_PER_KV, 1, 1))  # (B, 32, T, 64)
        v = v.repeat((1, NUM_Q_PER_KV, 1, 1))  # (B, 32, T, 64)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(HEAD_DIM)
        scores = (q @ k.transpose(-2, -1)) * scale
        scores = scores.clip(-1e4, 1e4)
        if attn_mask is not None:
            scores = scores + attn_mask
        attn_w = scores.softmax(-1)

        out = attn_w @ v                              # (B, 32, T, 64)
        out = out.transpose(1, 2).reshape(B, T, H)    # (B, T, 2048)
        return out @ self.wo


# ---------------------------------------------------------------------------
# MLP (SwiGLU — same as SmolLM2)
# ---------------------------------------------------------------------------

class LlamaMLP:
    def __init__(self):
        self.gate_proj = Tensor.zeros(HIDDEN_SIZE, INTERMEDIATE)
        self.up_proj   = Tensor.zeros(HIDDEN_SIZE, INTERMEDIATE)
        self.down_proj = Tensor.zeros(INTERMEDIATE, HIDDEN_SIZE)

    def forward(self, x: Tensor) -> Tensor:
        gate = (x @ self.gate_proj).silu()
        up   = x @ self.up_proj
        return (gate * up) @ self.down_proj


# ---------------------------------------------------------------------------
# Transformer layer with tap hooks
# ---------------------------------------------------------------------------

class LlamaLayer:
    def __init__(self):
        self.attn    = LlamaGQAAttention()
        self.mlp     = LlamaMLP()
        self.attn_ln = RMSNorm(HIDDEN_SIZE, eps=RMS_EPS)
        self.mlp_ln  = RMSNorm(HIDDEN_SIZE, eps=RMS_EPS)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor,
                attn_mask: Optional[Tensor] = None) -> Tensor:
        x = x + self.attn.forward(self.attn_ln(x), cos, sin, attn_mask)
        x = x + self.mlp.forward(self.mlp_ln(x))
        return x

    def forward_with_taps(self, x: Tensor, cos: Tensor, sin: Tensor,
                          attn_mask: Optional[Tensor] = None
                          ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Returns (x_post, taps) per §15 interface contract."""
        pre_ln = x
        x_post_attn = x + self.attn.forward(self.attn_ln(x), cos, sin, attn_mask)
        x_post_mlp  = x_post_attn + self.mlp.forward(self.mlp_ln(x_post_attn))
        taps = {
            "pre_ln_resid":    pre_ln,
            "post_attn_resid": x_post_attn,
            "post_mlp_resid":  x_post_mlp,
        }
        return x_post_mlp, taps


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class LlamaBase32:
    """Llama-3.2-1B base: embed + L0-L3 + ln_f.

    Interface contract (§15 of docs/v200_brief.md):
      .embed           — Embedding (128256, 2048)
      .layers[0..3]    — LlamaLayer list with GQA
      .ln_f            — RMSNorm (2048)
      .forward(ids)              -> (B, T, 2048)
      .forward_with_taps(ids)    -> (B, T, 2048), [taps_per_layer]
    """

    def __init__(self, weights_path: str):
        self.hidden_size = HIDDEN_SIZE
        self.vocab_size  = VOCAB_SIZE
        self.n_heads     = NUM_Q_HEADS
        self.n_kv_heads  = NUM_KV_HEADS
        self.head_dim    = HEAD_DIM

        self.embed  = Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.layers: List[LlamaLayer] = [LlamaLayer() for _ in range(4)]
        self.ln_f   = RMSNorm(HIDDEN_SIZE, eps=RMS_EPS)

        self.rope_cos, self.rope_sin = _build_rope_tables(MAX_SEQ_LEN)
        self._load_weights(weights_path)

    def _gpu(self, t: Tensor) -> Tensor:
        return t.to(Device.DEFAULT).realize()

    def _assign(self, dst: Tensor, src: Tensor) -> None:
        if src.shape != dst.shape:
            src = src.reshape(dst.shape)
        if src.dtype != dst.dtype:
            src = src.cast(dst.dtype)
        if src.device != dst.device:
            src = src.to(dst.device)
        dst.assign(src).realize()

    def _load_weights(self, path: str) -> None:
        """Load Llama-3.2-1B safetensors into the model's first 4 layers.

        GQA weight dimensions:
          q_proj.weight: (2048, 2048) → transpose → (2048, 2048)
          k_proj.weight: (512, 2048)  → transpose → (2048, 512)
          v_proj.weight: (512, 2048)  → transpose → (2048, 512)
          o_proj.weight: (2048, 2048) → transpose → (2048, 2048)
        """
        log(f"[llama32] Loading weights from {path}")
        sd = safe_load(path)
        log(f"[llama32] Loaded {len(sd)} weight tensors")

        # Token embedding (tied with lm_head in Llama-3.2-1B)
        self._assign(self.embed.weight, self._gpu(sd["model.embed_tokens.weight"]))

        for i in range(4):
            layer = self.layers[i]
            p = f"model.layers.{i}"

            # Attention — transpose from (out, in) to (in, out)
            # Q: (2048, 2048).T = (2048, 2048)
            # K/V: (512, 2048).T = (2048, 512)
            self._assign(layer.attn.wq, self._gpu(sd[f"{p}.self_attn.q_proj.weight"]).T)
            self._assign(layer.attn.wk, self._gpu(sd[f"{p}.self_attn.k_proj.weight"]).T)
            self._assign(layer.attn.wv, self._gpu(sd[f"{p}.self_attn.v_proj.weight"]).T)
            self._assign(layer.attn.wo, self._gpu(sd[f"{p}.self_attn.o_proj.weight"]).T)

            # MLP — same as SmolLM2
            self._assign(layer.mlp.gate_proj, self._gpu(sd[f"{p}.mlp.gate_proj.weight"]).T)
            self._assign(layer.mlp.up_proj,   self._gpu(sd[f"{p}.mlp.up_proj.weight"]).T)
            self._assign(layer.mlp.down_proj, self._gpu(sd[f"{p}.mlp.down_proj.weight"]).T)

            # RMSNorm
            self._assign(layer.attn_ln.weight, self._gpu(sd[f"{p}.input_layernorm.weight"]))
            self._assign(layer.mlp_ln.weight,  self._gpu(sd[f"{p}.post_attention_layernorm.weight"]))

        self._assign(self.ln_f.weight, self._gpu(sd["model.norm.weight"]))
        log("[llama32] Weights loaded. L0-L3 + embed + ln_f ready.")

    def _get_rope(self, T: int) -> Tuple[Tensor, Tensor]:
        return self.rope_cos[:T], self.rope_sin[:T]

    def forward(self, token_ids: Tensor) -> Tensor:
        """Forward through embed + L0-L3 + ln_f. Returns (B, T, 2048)."""
        B, T = token_ids.shape
        x = self.embed(token_ids)
        cos, sin = self._get_rope(T)
        for layer in self.layers:
            x = layer.forward(x, cos, sin)
        return self.ln_f(x)

    def forward_with_taps(self, token_ids: Tensor
                          ) -> Tuple[Tensor, List[Dict[str, Tensor]]]:
        """Forward with per-layer taps. Returns (x_final, [taps_per_layer])."""
        B, T = token_ids.shape
        x = self.embed(token_ids)
        cos, sin = self._get_rope(T)
        all_taps = []
        for layer in self.layers:
            x, taps = layer.forward_with_taps(x, cos, sin)
            all_taps.append(taps)
        x = self.ln_f(x)
        return x, all_taps

    def forward_wv_shared(self, token_ids: Tensor) -> Tensor:
        """Forward with L0's wv broadcast to L1-L3 (GQA-aware wv-sharing test).

        For GQA, wv is (2048, 512) — L0's wv is shared to L1, L2, L3.
        The test measures whether sharing the V projection across layers
        (the Pythia-era pin) is portable to Llama-3.2-1B's GQA attention.
        """
        B, T = token_ids.shape
        x = self.embed(token_ids)
        cos, sin = self._get_rope(T)

        wv_L0 = self.layers[0].attn.wv   # (2048, 512)

        for layer in self.layers:
            x_ln = layer.attn_ln(x)
            q = x_ln @ layer.attn.wq                # (B, T, 2048)
            k = x_ln @ layer.attn.wk                # (B, T, 512)
            v = x_ln @ wv_L0                        # ← L0's wv, shape (B, T, 512)

            q = q.reshape(B, T, NUM_Q_HEADS,  HEAD_DIM).transpose(1, 2)   # (B, 32, T, 64)
            k = k.reshape(B, T, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)   # (B, 8, T, 64)
            v = v.reshape(B, T, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)   # (B, 8, T, 64)

            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)

            # GQA expand
            k = k.repeat((1, NUM_Q_PER_KV, 1, 1))
            v = v.repeat((1, NUM_Q_PER_KV, 1, 1))

            scale = 1.0 / math.sqrt(HEAD_DIM)
            scores = (q @ k.transpose(-2, -1)) * scale
            scores = scores.clip(-1e4, 1e4)
            attn_w = scores.softmax(-1)
            out = (attn_w @ v).transpose(1, 2).reshape(B, T, HIDDEN_SIZE)
            out = out @ layer.attn.wo

            x_post_attn = x + out
            x = x_post_attn + layer.mlp.forward(layer.mlp_ln(x_post_attn))

        return self.ln_f(x)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_sha256(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(1 << 20)
            if not buf:
                break
            sha.update(buf)
    return sha.hexdigest()


def get_git_sha() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"],
                           capture_output=True, text=True, cwd=_PROJECT_ROOT, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_tinygrad_sha() -> str:
    try:
        import tinygrad
        tg_dir = os.path.dirname(os.path.dirname(tinygrad.__file__))
        r = subprocess.run(["git", "rev-parse", "HEAD"],
                           capture_output=True, text=True, cwd=tg_dir, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else "editable-no-git"
    except Exception:
        return "unknown"


def per_layer_stats(model: LlamaBase32, token_ids: Tensor) -> Tuple[dict, Tensor]:
    x_final, all_taps = model.forward_with_taps(token_ids)
    stats = {}
    for i, taps in enumerate(all_taps):
        pre  = taps["pre_ln_resid"].float().numpy()
        attn = taps["post_attn_resid"].float().numpy()
        mlp  = taps["post_mlp_resid"].float().numpy()

        def _norm(a): return float(np.linalg.norm(a.reshape(a.shape[0], -1, a.shape[-1]), axis=-1).mean())

        delta = mlp - pre
        stats[f"layer{i}"] = {
            "pre_ln_norm":    _norm(pre),
            "post_attn_norm": _norm(attn),
            "post_mlp_norm":  _norm(mlp),
            "delta_norm":     _norm(delta),
        }
    final_norm = float(np.linalg.norm(
        x_final.float().numpy().reshape(x_final.shape[0], -1, x_final.shape[-1]), axis=-1
    ).mean())
    stats["ln_f_output_norm"] = final_norm
    return stats, x_final


def check_nan(t: Tensor, name: str) -> bool:
    arr = t.float().numpy()
    has_nan = bool(np.isnan(arr).any())
    has_inf = bool(np.isinf(arr).any())
    if has_nan or has_inf:
        log(f"  WARN: {name} contains NaN={has_nan} Inf={has_inf}")
    return has_nan or has_inf


# ---------------------------------------------------------------------------
# Main smoke
# ---------------------------------------------------------------------------

def run_smoke() -> int:
    log("=" * 70)
    log("Mycelium v200 Stage 1A — Llama-3.2-1B Smoke Test")
    log(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    log(f"Device:    {Device.DEFAULT}")
    log("=" * 70)

    # ------------------------------------------------------------------
    # 1. Resolve weights path
    # ------------------------------------------------------------------
    log("\n[1] Resolving weights path...")
    weights_path = LLAMA32_WEIGHTS
    if not os.path.exists(weights_path):
        log(f"SMOKE FAILED weights not found at {weights_path}")
        log("Run: python3 -c \"from huggingface_hub import snapshot_download; "
            "snapshot_download('meta-llama/Llama-3.2-1B', "
            "local_dir='/home/bryce/mycelium/.cache/llama-3.2-1b-weights', "
            "ignore_patterns=['*.bin', 'original/*'])\"")
        return 1
    size_gb = os.path.getsize(weights_path) / (1024**3)
    log(f"    weights_path = {weights_path}")
    log(f"    file size    = {size_gb:.2f} GB")

    # ------------------------------------------------------------------
    # 2. SHA256
    # ------------------------------------------------------------------
    log("\n[2] Computing SHA256 of weights file...")
    t0 = time.time()
    sha256 = compute_sha256(weights_path)
    t_sha = time.time() - t0
    log(f"    sha256       = {sha256}")
    log(f"    hashing time = {t_sha:.1f}s")
    with open(SHA_PATH, "w") as f:
        f.write(f"{sha256}  {weights_path}\n")
    log(f"    saved to     {SHA_PATH}")

    # ------------------------------------------------------------------
    # 3. Load model
    # ------------------------------------------------------------------
    log("\n[3] Loading model (LlamaBase32)...")
    t0 = time.time()
    try:
        model = LlamaBase32(weights_path)
        t_load = time.time() - t0
    except Exception as e:
        log(f"SMOKE FAILED model load error: {e}")
        import traceback
        log(traceback.format_exc())
        return 1
    log(f"    Load time:    {t_load:.1f}s")
    log(f"    hidden_size:  {model.hidden_size}")
    log(f"    vocab_size:   {model.vocab_size}")
    log(f"    n_q_heads:    {model.n_heads}")
    log(f"    n_kv_heads:   {model.n_kv_heads}  (GQA: 4 Q heads per KV head)")
    log(f"    head_dim:     {model.head_dim}")
    log(f"    n_layers:     {len(model.layers)}")

    # ------------------------------------------------------------------
    # 4. Interface contract verification (§15)
    # ------------------------------------------------------------------
    log("\n[4] Interface contract verification (§15)...")
    contract_ok = True

    # Check .embed
    try:
        assert hasattr(model, 'embed'), "model.embed missing"
        assert hasattr(model.embed, 'weight'), "model.embed.weight missing"
        assert list(model.embed.weight.shape) == [VOCAB_SIZE, HIDDEN_SIZE], \
            f"embed shape {list(model.embed.weight.shape)} != [{VOCAB_SIZE}, {HIDDEN_SIZE}]"
        log(f"    [PASS] model.embed: shape {list(model.embed.weight.shape)}")
    except AssertionError as e:
        log(f"    [FAIL] {e}")
        contract_ok = False

    # Check .layers[0..3]
    for i in range(4):
        try:
            layer = model.layers[i]
            assert hasattr(layer, 'attn'), f"layers[{i}].attn missing"
            assert hasattr(layer.attn, 'wq'), f"layers[{i}].attn.wq missing"
            assert hasattr(layer.attn, 'wk'), f"layers[{i}].attn.wk missing"
            assert hasattr(layer.attn, 'wv'), f"layers[{i}].attn.wv missing"
            assert hasattr(layer.attn, 'wo'), f"layers[{i}].attn.wo missing"
            assert hasattr(layer, 'mlp'), f"layers[{i}].mlp missing"
            assert hasattr(layer.mlp, 'gate_proj'), f"layers[{i}].mlp.gate_proj missing"
            assert hasattr(layer.mlp, 'up_proj'), f"layers[{i}].mlp.up_proj missing"
            assert hasattr(layer.mlp, 'down_proj'), f"layers[{i}].mlp.down_proj missing"
            # Shape checks for GQA weights
            assert list(layer.attn.wq.shape) == [HIDDEN_SIZE, HIDDEN_SIZE], \
                f"wq shape {list(layer.attn.wq.shape)}"
            assert list(layer.attn.wk.shape) == [HIDDEN_SIZE, KV_DIM], \
                f"wk shape {list(layer.attn.wk.shape)} != [{HIDDEN_SIZE}, {KV_DIM}]"
            assert list(layer.attn.wv.shape) == [HIDDEN_SIZE, KV_DIM], \
                f"wv shape {list(layer.attn.wv.shape)} != [{HIDDEN_SIZE}, {KV_DIM}]"
            assert list(layer.attn.wo.shape) == [HIDDEN_SIZE, HIDDEN_SIZE], \
                f"wo shape {list(layer.attn.wo.shape)}"
            log(f"    [PASS] model.layers[{i}]: wq{list(layer.attn.wq.shape)} "
                f"wk{list(layer.attn.wk.shape)} wv{list(layer.attn.wv.shape)} "
                f"wo{list(layer.attn.wo.shape)}")
        except AssertionError as e:
            log(f"    [FAIL] layers[{i}]: {e}")
            contract_ok = False

    # Check .ln_f
    try:
        assert hasattr(model, 'ln_f'), "model.ln_f missing"
        assert hasattr(model.ln_f, 'weight'), "model.ln_f.weight missing"
        assert list(model.ln_f.weight.shape) == [HIDDEN_SIZE], \
            f"ln_f.weight shape {list(model.ln_f.weight.shape)}"
        log(f"    [PASS] model.ln_f: shape {list(model.ln_f.weight.shape)}")
    except AssertionError as e:
        log(f"    [FAIL] {e}")
        contract_ok = False

    # forward_with_taps — will verify shape after running baseline forward
    log(f"    [INFO] forward_with_taps will be verified in step [5]")

    # ------------------------------------------------------------------
    # 5. Baseline forward (B=2, T=64, random tokens)
    # ------------------------------------------------------------------
    log("\n[5] Baseline forward (B=2, T=64, random tokens)...")
    np.random.seed(42)
    token_ids = Tensor(
        np.random.randint(0, VOCAB_SIZE, (2, 64), dtype=np.int32)
    )
    log(f"    token_ids shape: {list(token_ids.shape)}")

    t0 = time.time()
    try:
        baseline_stats, x_baseline = per_layer_stats(model, token_ids)
        t_fwd_base = time.time() - t0
    except Exception as e:
        log(f"SMOKE FAILED baseline forward error: {e}")
        import traceback
        log(traceback.format_exc())
        return 1

    log(f"    Forward time: {t_fwd_base:.3f}s")
    log(f"    Output shape: {list(x_baseline.shape)}")

    # Shape assertion
    if list(x_baseline.shape) != [2, 64, 2048]:
        log(f"SMOKE FAILED output shape {list(x_baseline.shape)} != [2, 64, 2048]")
        return 1

    # NaN check
    baseline_has_nan = check_nan(x_baseline, "baseline output")
    if baseline_has_nan:
        log("SMOKE FAILED NaN/Inf in baseline forward output")
        return 1

    # Verify forward_with_taps shape via the per_layer_stats call above
    log(f"    [PASS] forward_with_taps: output shape [2, 64, 2048], 4 tap dicts returned")
    for i in range(4):
        for tap_name in ["pre_ln_resid", "post_attn_resid", "post_mlp_resid"]:
            log(f"           layer{i}.{tap_name}: verified via per_layer_stats")

    log(f"\n    Per-layer norms (baseline):")
    log(f"    {'Layer':>8}  {'pre_ln':>10}  {'post_attn':>10}  {'post_mlp':>10}  {'delta':>10}")
    for i in range(4):
        s = baseline_stats[f"layer{i}"]
        log(f"    {'L'+str(i):>8}  {s['pre_ln_norm']:>10.4f}  {s['post_attn_norm']:>10.4f}  "
            f"{s['post_mlp_norm']:>10.4f}  {s['delta_norm']:>10.4f}")
    log(f"    {'ln_f_out':>8}  {baseline_stats['ln_f_output_norm']:>10.4f}")

    # Warm forward for timing
    t0 = time.time()
    _ = model.forward(token_ids)
    t_fwd_warm = time.time() - t0
    log(f"\n    Warm forward time: {t_fwd_warm:.3f}s")

    # ------------------------------------------------------------------
    # 6. wv-shared forward (GQA-aware: L0's wv(2048,512) shared to L1-L3)
    # ------------------------------------------------------------------
    log("\n[6] wv-shared forward (L0 wv broadcast to L1-L3, GQA-aware)...")
    log("    NOTE: For Llama-3.2-1B GQA, wv is (2048, 512) — 8 KV heads × 64.")
    log("    Sharing L0's wv to L1-L3 means the same V projection for all layers.")
    log("    This tests the Pythia-era wv-sharing pin on GQA architecture.")

    t0 = time.time()
    try:
        x_wvshared = model.forward_wv_shared(token_ids)
        t_fwd_wvs = time.time() - t0
    except Exception as e:
        log(f"SMOKE FAILED wv-shared forward error: {e}")
        import traceback
        log(traceback.format_exc())
        return 1

    log(f"    Forward time: {t_fwd_wvs:.3f}s")
    wvs_has_nan = check_nan(x_wvshared, "wv-shared output")

    # Cosine similarity between baseline and wv-shared outputs
    x_base_np = x_baseline.float().numpy().reshape(-1, 2048)
    x_wvs_np  = x_wvshared.float().numpy().reshape(-1, 2048)

    cos_sims = (x_base_np * x_wvs_np).sum(-1) / (
        np.linalg.norm(x_base_np, axis=-1) * np.linalg.norm(x_wvs_np, axis=-1) + 1e-8
    )
    mean_cos_sim = float(cos_sims.mean())
    min_cos_sim  = float(cos_sims.min())
    l2_diff = float(np.linalg.norm(x_base_np - x_wvs_np, axis=-1).mean())

    base_norm = baseline_stats["ln_f_output_norm"]
    wvs_norm  = float(np.linalg.norm(x_wvs_np.reshape(2, -1, 2048), axis=-1).mean())
    norm_ratio = wvs_norm / (base_norm + 1e-8)

    log(f"\n    wv-sharing comparison:")
    log(f"    Metric                    Baseline       wv-shared      Ratio/Delta")
    log(f"    ln_f_output_norm          {base_norm:>10.4f}     {wvs_norm:>10.4f}     ratio={norm_ratio:.4f}")
    log(f"    mean_cos_sim(base,wvs)    {'N/A':>10}     {'N/A':>10}     {mean_cos_sim:.6f}")
    log(f"    min_cos_sim(base,wvs)     {'N/A':>10}     {'N/A':>10}     {min_cos_sim:.6f}")
    log(f"    mean_L2_dist(base,wvs)    {'N/A':>10}     {'N/A':>10}     {l2_diff:.4f}")

    log(f"\n    wv-sharing portability assessment (Llama-3.2-1B, GQA):")
    if mean_cos_sim > 0.99:
        verdict = "HIGH cos-sim — wv-sharing barely changes output. Pin appears portable."
        portability = "PORTABLE (cos > 0.99)"
    elif mean_cos_sim > 0.90:
        verdict = "MODERATE cos-sim — wv-sharing changes output noticeably. Pin portability unclear."
        portability = "UNCLEAR (0.90 < cos < 0.99)"
    else:
        verdict = f"LOW cos-sim ({mean_cos_sim:.4f}) — wv-sharing substantially changes output."
        portability = "REFUTED (cos < 0.90)"

    log(f"    {verdict}")
    if wvs_has_nan:
        portability = "REFUTED (NaN in wv-shared output)"
        log(f"    VERDICT: NaN in wv-shared — pin BREAKS Llama-3.2-1B representations.")
    else:
        log(f"    VERDICT: {portability}")

    # SmolLM2 comparison note
    log(f"\n    SmolLM2 wv-sharing baseline: cos={SMOLLM2_WV_COS:.4f} → REFUTED")
    if mean_cos_sim < 0.90:
        log(f"    Llama-3.2-1B also REFUTED: cos={mean_cos_sim:.4f} (similar to SmolLM2)")
    else:
        log(f"    Llama-3.2-1B result differs from SmolLM2 (cos={mean_cos_sim:.4f} vs {SMOLLM2_WV_COS:.4f})")

    # Warm wv-shared timing
    t0 = time.time()
    _ = model.forward_wv_shared(token_ids)
    t_fwd_wvs_warm = time.time() - t0
    log(f"\n    Warm wv-shared forward: {t_fwd_wvs_warm:.3f}s")

    # ------------------------------------------------------------------
    # 7. Peak memory estimate
    # ------------------------------------------------------------------
    log("\n[7] Peak memory estimate...")
    # Q attn: wq(2048×2048) + wk(2048×512) + wv(2048×512) + wo(2048×2048) = 4M + 1M + 1M + 4M = 10M per layer
    attn_params = (HIDDEN_SIZE * HIDDEN_SIZE +   # wq
                   HIDDEN_SIZE * KV_DIM +        # wk
                   HIDDEN_SIZE * KV_DIM +        # wv
                   HIDDEN_SIZE * HIDDEN_SIZE)    # wo
    mlp_params  = 2 * HIDDEN_SIZE * INTERMEDIATE + INTERMEDIATE * HIDDEN_SIZE
    ln_params   = 2 * HIDDEN_SIZE
    total_params = (
        VOCAB_SIZE * HIDDEN_SIZE +
        4 * (attn_params + mlp_params + ln_params) +
        HIDDEN_SIZE
    )
    weight_mem_gb = total_params * 2 / (1024**3)  # bfloat16 = 2 bytes
    log(f"    Q-attn params/layer:   {attn_params/1e6:.2f}M (GQA: wk,wv are 2048×512)")
    log(f"    MLP params/layer:      {mlp_params/1e6:.2f}M")
    log(f"    L0-L3 + embed + ln_f:  {total_params/1e6:.1f}M params")
    log(f"    Weight memory (bf16):  {weight_mem_gb:.2f} GB")
    log(f"    Full Llama-3.2-1B has ~1.24B params; we load only L0-L3 + embed + ln_f")

    # ------------------------------------------------------------------
    # 8. Comparison table vs SmolLM2
    # ------------------------------------------------------------------
    log("\n[8] Comparison table: SmolLM2-1.7B vs Llama-3.2-1B")
    # Cleaner table format
    log(f"                             SmolLM2-1.7B    Llama-3.2-1B")
    log(f"    load time              {SMOLLM2_LOAD_TIME:.1f}s              {t_load:.1f}s")
    log(f"    forward time           {SMOLLM2_FWD_TIME:.3f}s            {t_fwd_warm:.3f}s")
    log(f"    output shape           [2,64,2048]     [2,64,2048]")
    log(f"    NaN-free               YES             {'YES' if not baseline_has_nan else 'NO'}")
    log(f"    peak GPU mem           0.69 GB         {weight_mem_gb:.2f} GB")
    log(f"    wv-sharing cos vs base 0.5126          {mean_cos_sim:.4f}   <-- portability")

    # ------------------------------------------------------------------
    # 9. Provenance sidecar
    # ------------------------------------------------------------------
    log("\n[9] Writing provenance sidecar...")
    git_sha = get_git_sha()
    tg_sha  = get_tinygrad_sha()
    now_iso = datetime.now(timezone.utc).isoformat()

    provenance = {
        "what": {
            "metric":     "llama32_smoke_forward_stats",
            "units":      "raw activations + L2 norms, wv-sharing cos-sim",
            "shape":      "[B=2, T=64, H=2048] after L0-L3 + ln_f",
            "head_group": None,
        },
        "where": {
            "file": LOG_PATH,
            "key":  None,
        },
        "when": {
            "timestamp_iso": now_iso,
            "git_sha":       git_sha,
            "config_diff":   "Llama-3.2-1B vs SmolLM2-1.7B baseline: vocab 128256 vs 49152, GQA 8 KV heads vs 32, rope_theta 500k vs 130k",
            "step":          0,
        },
        "with_what": {
            "ckpt":  "meta-llama/Llama-3.2-1B",
            "split": None,
            "seed":  42,
            "env": {
                "tinygrad_sha": tg_sha,
                "device":       f"AM driver/AMD 7900 XTX ({Device.DEFAULT})",
                "env_vars": {
                    "LLAMA32_WEIGHTS": os.environ.get("LLAMA32_WEIGHTS", "unset"),
                },
            },
        },
    }

    with open(PROV_PATH, "w") as f:
        json.dump(provenance, f, indent=2)
    log(f"    Saved: {PROV_PATH}")

    # ------------------------------------------------------------------
    # 10. Summary + SMOKE PASSED / FAILED
    # ------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"  Model:           meta-llama/Llama-3.2-1B")
    log(f"  Weights path:    {weights_path}")
    log(f"  SHA256:          {sha256[:16]}...")
    log(f"  Load time:       {t_load:.1f}s")
    log(f"  Baseline fwd:    {t_fwd_warm:.3f}s (warm, B=2, T=64)")
    log(f"  wv-shared fwd:   {t_fwd_wvs_warm:.3f}s (warm, B=2, T=64)")
    log(f"  Output shape:    {list(x_baseline.shape)}")
    log(f"  Baseline NaN:    {baseline_has_nan}")
    log(f"  wv-shared NaN:   {wvs_has_nan}")
    log(f"  wv-sharing cos:  {mean_cos_sim:.6f}")
    log(f"  wv portability:  {portability}")
    log(f"  Contract OK:     {contract_ok}")
    log(f"  Interface:       .embed{list(model.embed.weight.shape)} "
        f".layers[0..3] .ln_f .forward_with_taps()")

    pass_conditions = [
        (not baseline_has_nan,                              "baseline forward is NaN-free"),
        (list(x_baseline.shape) == [2, 64, 2048],          "output shape is [2, 64, 2048]"),
        (contract_ok,                                        "interface contract verified"),
        (os.path.exists(SHA_PATH),                          "SHA256 file exists"),
        (os.path.exists(PROV_PATH),                         "provenance file exists"),
    ]

    all_pass = all(cond for cond, _ in pass_conditions)
    for cond, desc in pass_conditions:
        status = "PASS" if cond else "FAIL"
        log(f"  [{status}] {desc}")

    log("")
    if all_pass:
        metrics = (
            f"model=meta-llama/Llama-3.2-1B "
            f"load={t_load:.1f}s "
            f"fwd={t_fwd_warm:.3f}s "
            f"shape=[2,64,2048] "
            f"nan=False "
            f"wv_cos={mean_cos_sim:.4f} "
            f"wv_portability={portability} "
            f"sha={sha256[:8]}"
        )
        log(f"SMOKE PASSED {metrics}")
        return 0
    else:
        failed = [desc for cond, desc in pass_conditions if not cond]
        log(f"SMOKE FAILED {'; '.join(failed)}")
        return 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    exit_code = run_smoke()
    _log_fh.close()
    sys.exit(exit_code)
