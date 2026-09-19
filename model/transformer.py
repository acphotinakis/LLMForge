"""
GPT-style decoder-only Transformer model for Research LLM.

Features:
  - Rotary Position Embeddings (RoPE)
  - Grouped-Query Attention (GQA) — set n_kv_heads < n_heads for MQA/GQA
  - Pre-LayerNorm with RMSNorm (more stable than post-LN)
  - SwiGLU feed-forward network
  - Optional Flash Attention via F.scaled_dot_product_attention (PyTorch 2.0+)
  - Weight tying between token embedding and output projection
  - Configurable dropout, bias, and context length

Architecture reference: GPT-2 / LLaMA hybrid design.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


# ============================================================
# 1. Configuration dataclass
# ============================================================

@dataclass
class ModelConfig:
    """Full specification of the Transformer architecture."""
    vocab_size: int = 32_000
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: Optional[int] = None   # None → same as n_heads (standard MHA)
    d_model: int = 768
    d_ff: Optional[int] = None          # None → 4 * d_model
    context_length: int = 1024
    dropout: float = 0.1
    bias: bool = False
    rope_base: float = 10_000.0
    norm_eps: float = 1.0e-5
    tie_embeddings: bool = True

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.d_ff is None:
            # SwiGLU convention: ~2/3 * 4 * d_model, rounded to multiple of 256
            raw = int(2 / 3 * 4 * self.d_model)
            self.d_ff = (raw + 255) // 256 * 256
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def n_params(self) -> int:
        """Approximate parameter count (excluding position embeddings)."""
        # Embedding + output
        embed = self.vocab_size * self.d_model
        # Each transformer block
        attn = self.d_model * (self.n_heads + 2 * self.n_kv_heads) * self.head_dim
        ff = 3 * self.d_model * self.d_ff  # SwiGLU has 3 matrices
        ln = 2 * self.d_model  # two layernorms per block
        block = attn + ff + ln
        total = embed + self.n_layers * block + self.d_model  # final norm
        if self.tie_embeddings:
            total -= self.vocab_size * self.d_model
        return total

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_config(cls, cfg: object) -> "ModelConfig":
        """Construct from a utils.Config object."""
        mc = cfg.model
        return cls(
            vocab_size=mc.vocab_size,
            n_layers=mc.n_layers,
            n_heads=mc.n_heads,
            n_kv_heads=mc.get("n_kv_heads", None),
            d_model=mc.d_model,
            d_ff=mc.get("d_ff", None),
            context_length=mc.context_length,
            dropout=mc.dropout,
            bias=mc.bias,
            rope_base=mc.rope_base,
            norm_eps=mc.norm_eps,
            tie_embeddings=mc.tie_embeddings,
        )


# ============================================================
# 2. Building blocks
# ============================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no bias, no learned mean shift)."""

    def __init__(self, dim: int, eps: float = 1.0e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE).

    Precomputes cos/sin tables up to ``max_seq_len``.
    During forward, slices the table to the actual sequence length.
    """

    def __init__(self, head_dim: int, base: float = 10_000.0, max_seq_len: int = 8192):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        # theta_i = base^(-2i / d)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cache", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cache", emb.sin()[None, None, :, :], persistent=False)
        self._cached_seq_len = seq_len

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self._cached_seq_len:
            self._build_cache(seq_len)
        return (
            self.cos_cache[:, :, :seq_len, :].to(x.dtype),
            self.sin_cache[:, :, :seq_len, :].to(x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


# ============================================================
# 3. Attention (with GQA + RoPE)
# ============================================================

class CausalSelfAttention(nn.Module):
    """
    Multi-head (or Grouped-Query) causal self-attention with RoPE.

    If ``n_kv_heads < n_heads``, uses GQA: KV heads are shared across
    groups of Q heads, reducing KV cache memory at inference.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_groups = cfg.n_heads // cfg.n_kv_heads  # how many Q heads per KV head
        self.scale = self.head_dim ** -0.5

        # Projections
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=cfg.bias)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=cfg.bias)

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # RoPE
        self.rope = RotaryEmbedding(cfg.head_dim, base=cfg.rope_base, max_seq_len=cfg.context_length * 2)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape

        # Project
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)    # (B, n_h, T, hd)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2) # (B, n_kv, T, hd)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2) # (B, n_kv, T, hd)

        # Apply RoPE
        cos, sin = self.rope(q, T)
        q, k = apply_rope(q, k, cos, sin)

        # Expand KV for GQA: repeat KV heads n_groups times
        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)

        # Scaled dot-product attention (uses Flash Attention if PyTorch 2.0+)
        dropout_p = self.attn_dropout.p if self.training else 0.0
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=dropout_p,
            is_causal=(mask is None),
        )

        # Reshape and project out
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.resid_dropout(self.o_proj(y))


# ============================================================
# 4. Feed-Forward (SwiGLU)
# ============================================================

class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    SwiGLU(x) = swish(xW_gate) ⊗ (xW_up)
    Output = SwiGLU(x) @ W_down

    Uses 3 projections instead of 2, reducing d_ff slightly vs. vanilla FFN
    to keep parameter count comparable.
    """

    def __init__(self, d_model: int, d_ff: int, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.up_proj   = nn.Linear(d_model, d_ff, bias=bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ============================================================
# 5. Transformer block
# ============================================================

class TransformerBlock(nn.Module):
    """A single decoder-only Transformer block (pre-norm)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn  = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.ffn   = SwiGLUFFN(cfg.d_model, cfg.d_ff, bias=cfg.bias, dropout=cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================
# 6. Full GPT model
# ============================================================

class GPTModel(nn.Module):
    """
    Decoder-only GPT-style language model.

    Inputs:  token IDs  (B, T)
    Outputs: logits     (B, T, vocab_size)

    The model uses:
      - Learned token embeddings (no absolute position embeddings — RoPE handles position)
      - N × TransformerBlock
      - Final RMSNorm
      - Linear head (weight-tied with embedding by default)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb  = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop_emb = nn.Dropout(cfg.dropout)
        self.blocks   = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_out = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.lm_head  = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        # Initialise weights
        self.apply(self._init_weights)
        # Scale residual projections (GPT-2 trick)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

        logger.info(
            f"GPTModel initialised: {cfg.n_layers}L / {cfg.n_heads}H / "
            f"d={cfg.d_model} / ctx={cfg.context_length} / "
            f"~{cfg.n_params / 1e6:.1f}M params"
        )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids:  (B, T) token indices.
            labels:     (B, T) target token indices for LM loss.  If None,
                        only logits are returned.
            mask:       Optional attention mask.  None → use causal mask.

        Returns:
            (logits, loss) where loss is None if labels is None.
        """
        B, T = input_ids.shape
        assert T <= self.cfg.context_length, (
            f"Sequence length {T} exceeds model context length {self.cfg.context_length}"
        )

        x = self.drop_emb(self.tok_emb(input_ids))  # (B, T, d_model)

        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.norm_out(x)               # (B, T, d_model)
        logits = self.lm_head(x)           # (B, T, vocab_size)

        loss = None
        if labels is not None:
            # Flatten for cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Auto-regressive text generation.

        Args:
            input_ids:         (1, T) prompt token IDs.
            max_new_tokens:    Maximum tokens to generate.
            temperature:       Sampling temperature (1.0 = neutral).
            top_k:             Keep only top-k logits before sampling.
            top_p:             Nucleus sampling threshold.
            repetition_penalty: > 1.0 penalises repeated tokens.
            eos_token_id:      Stop generation when this token is produced.

        Returns:
            (1, T + new_tokens) tensor of token IDs.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to context window
            ctx = input_ids if input_ids.size(1) <= self.cfg.context_length \
                else input_ids[:, -self.cfg.context_length:]

            logits, _ = self.forward(ctx)
            logits = logits[:, -1, :]  # (1, vocab_size) — last token prediction

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(input_ids[0].tolist()):
                    logits[0, token_id] /= repetition_penalty

            # Temperature
            logits = logits / max(temperature, 1e-8)

            # Top-k filtering
            if top_k is not None and top_k > 0:
                top_k_clamped = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, top_k_clamped)
                min_val = values[:, -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < min_val, float("-inf"))

            # Top-p (nucleus) filtering
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative prob above threshold
                sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[sorted_indices_to_remove] = float("-inf")
                logits = torch.scatter(logits, 1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        return input_ids

    # ------------------------------------------------------------------ #
    #  Utilities                                                           #
    # ------------------------------------------------------------------ #

    def num_parameters(self, only_trainable: bool = True) -> int:
        """Count model parameters."""
        return sum(
            p.numel() for p in self.parameters()
            if not only_trainable or p.requires_grad
        )

    def configure_optimiser(
        self,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        beta1: float = 0.9,
        beta2: float = 0.95,
        eps: float = 1e-8,
        device_type: str = "cuda",
    ) -> torch.optim.Optimizer:
        """
        Set up AdamW with weight decay applied only to weight matrices,
        not to biases, norms, or embeddings (standard practice).
        """
        decay_params, no_decay_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # Decay: weight matrices only
            if param.ndim >= 2 and not any(
                nd in name for nd in ["norm", "embedding", "emb"]
            ):
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        optim_groups = [
            {"params": decay_params,    "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        # Use fused AdamW if available (faster on CUDA)
        fused_available = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        use_fused = fused_available and device_type == "cuda"
        kwargs = {"fused": True} if use_fused else {}
        if use_fused:
            logger.info("Using fused AdamW")

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=eps,
            **kwargs,
        )
        return optimizer


# ============================================================
# 7. Factory function
# ============================================================

def build_model(cfg: object, vocab_size: Optional[int] = None) -> GPTModel:
    """
    Build a ``GPTModel`` from a Config object.

    Args:
        cfg:        Top-level Config (``cfg.model`` must exist).
        vocab_size: If provided, overrides ``cfg.model.vocab_size``.

    Returns:
        Initialised ``GPTModel``.
    """
    model_cfg = ModelConfig.from_config(cfg)
    if vocab_size is not None:
        model_cfg.vocab_size = vocab_size
    return GPTModel(model_cfg)
