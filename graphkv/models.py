"""Decoder forward passes that work with a per-layer, variable-length KV cache.

Graph-based compression gives every layer its own cache length, which the
stock Hugging Face ``generate()``/``Cache`` machinery cannot represent (it
builds one attention mask for all layers, and its cache API changes between
releases). The adapters below therefore reuse the Hugging Face *modules*
(embeddings, projections, norms, MLPs, rotary embedding) of a loaded model,
and only re-implement the attention-over-cache part. The token positions
always use the true token index, so absolute position embeddings (GPT-2) and
RoPE (Llama) stay correct after the cache has been shortened.

``tests/test_models.py`` checks that the logits match the Hugging Face
implementation exactly when no compression is applied.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .cache import KVCache

SUPPORTED_MODEL_TYPES = ("gpt2", "llama", "mistral", "qwen2")


def cached_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Scaled dot-product attention of new queries over the cache.

    Args:
        query: ``[1, num_heads, q_len, head_dim]`` for the ``q_len`` new tokens.
        key, value: ``[1, num_kv_heads, cache_len, head_dim]``; the last
            ``q_len`` entries are the new tokens themselves.
        scale: softmax temperature applied to ``q @ k^T``.
        mask: optional ``[num_kv_heads, cache_len]`` per-entry mask, either
            boolean (``True`` = real entry, ``False`` = padding) or an additive
            float bias (``-inf`` for padding; see :meth:`KVCache.attention_mask`).

    Every older cache entry (sink, centroid or raw token) lies in the past of
    all new tokens and is fully visible. The new tokens attend to each other
    causally.
    """
    num_heads, q_len = query.shape[1], query.shape[2]
    num_kv_heads, cache_len = key.shape[1], key.shape[2]
    if num_kv_heads != num_heads:  # grouped-query attention, same layout as HF repeat_kv
        n_rep = num_heads // num_kv_heads
        key = key.repeat_interleave(n_rep, dim=1)
        value = value.repeat_interleave(n_rep, dim=1)
        if mask is not None:
            mask = mask.repeat_interleave(n_rep, dim=0)
    if mask is not None:
        mask = mask[None, :, None, :]
        if mask.is_floating_point():
            mask = mask.to(query.dtype)
    if q_len > 1:
        causal = torch.ones(q_len, cache_len, dtype=torch.bool, device=query.device)
        causal[:, cache_len - q_len :] = torch.ones(q_len, q_len, dtype=torch.bool, device=query.device).tril()
        if mask is None:
            mask = causal
        elif mask.dtype == torch.bool:
            mask = mask & causal
        else:
            mask = torch.where(causal, mask, torch.tensor(float("-inf"), dtype=mask.dtype, device=mask.device))
    return F.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=scale)


class DecoderAdapter:
    """Common interface: ``logits = adapter(input_ids, cache)``."""

    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    max_positions: Optional[int]

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.config = model.config

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def new_cache(self, track_weights: bool = False, proportional_attention: bool = False) -> KVCache:
        return KVCache(self.num_layers, self.num_kv_heads, track_weights, proportional_attention)

    def _positions(self, cache: KVCache, q_len: int) -> torch.Tensor:
        start = cache.num_tokens
        if self.max_positions is not None and start + q_len > self.max_positions:
            raise ValueError(
                f"sequence would reach position {start + q_len}, but the model supports "
                f"at most {self.max_positions} positions"
            )
        return torch.arange(start, start + q_len, device=self.device)

    def forward(self, input_ids: torch.Tensor, cache: KVCache, last_only: bool = False) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, input_ids: torch.Tensor, cache: KVCache, last_only: bool = False) -> torch.Tensor:
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError("input_ids must have shape [1, seq_len]")
        if self.model.training:
            raise RuntimeError("call model.eval() first; dropout is not applied by the adapter")
        return self.forward(input_ids, cache, last_only)


class GPT2Adapter(DecoderAdapter):
    """GPT-2 (learned absolute position embeddings, fused QKV ``Conv1D``)."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model)
        cfg = model.config
        if getattr(cfg, "add_cross_attention", False):
            raise ValueError("GPT-2 models with cross-attention are not supported")
        tr = model.transformer
        self.blocks = tr.h
        self.num_layers = len(self.blocks)
        self.num_heads = self.num_kv_heads = cfg.n_head
        self.embed_dim = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.max_positions = cfg.n_positions
        self.scales = []
        for i in range(self.num_layers):
            scale = self.head_dim**-0.5 if getattr(cfg, "scale_attn_weights", True) else 1.0
            if getattr(cfg, "scale_attn_by_inverse_layer_idx", False):
                scale /= float(i + 1)
            self.scales.append(scale)

    def forward(self, input_ids: torch.Tensor, cache: KVCache, last_only: bool = False) -> torch.Tensor:
        tr = self.model.transformer
        q_len = input_ids.shape[1]
        positions = self._positions(cache, q_len)
        h = tr.wte(input_ids) + tr.wpe(positions).unsqueeze(0)
        for i, block in enumerate(self.blocks):
            qkv = block.attn.c_attn(block.ln_1(h))
            q, k, v = (t.view(1, q_len, self.num_heads, self.head_dim).transpose(1, 2) for t in qkv.split(self.embed_dim, dim=2))
            keys, values = cache.update(i, k, v)
            attn = cached_attention(q, keys, values, self.scales[i], cache.attention_mask(i))
            h = h + block.attn.c_proj(attn.transpose(1, 2).reshape(1, q_len, self.embed_dim))
            h = h + block.mlp(block.ln_2(h))
        cache.num_tokens += q_len
        if last_only:
            h = h[:, -1:]
        return self.model.lm_head(tr.ln_f(h))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class LlamaAdapter(DecoderAdapter):
    """Llama-style decoders with RoPE and grouped-query attention.

    Covers Llama / TinyLlama and the architecturally identical Mistral and
    Qwen2 families. Cached keys are stored *after* the rotary embedding (as in
    Hugging Face), so merged centroids are averages of rotated keys.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model)
        cfg = model.config
        base = model.model
        self.layers = base.layers
        self.num_layers = len(self.layers)
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.max_positions = None  # RoPE has no hard limit; quality may drop past the training context
        attn = self.layers[0].self_attn
        # Model-level rotary module in recent transformers; older releases keep an
        # identical copy in every attention layer. Both map (x, position_ids) -> (cos, sin).
        self.rotary = getattr(base, "rotary_emb", None) or getattr(attn, "rotary_emb", None)
        if self.rotary is None:
            raise RuntimeError("could not find the rotary embedding module; please install transformers>=4.45")
        for name in ("q_norm", "k_norm"):
            if getattr(attn, name, None) is not None:
                raise ValueError(f"attention with {name} (e.g. Qwen3) is not supported")
        self.scale = float(getattr(attn, "scaling", self.head_dim**-0.5))
        self.sliding_window = self._sliding_window(cfg)

    @staticmethod
    def _sliding_window(cfg) -> Optional[int]:
        window = getattr(cfg, "sliding_window", None)
        if window is None:
            return None
        if cfg.model_type == "qwen2":
            layer_types = getattr(cfg, "layer_types", None)
            uses = any(t == "sliding_attention" for t in layer_types) if layer_types else getattr(cfg, "use_sliding_window", False)
            return window if uses else None
        return window

    def forward(self, input_ids: torch.Tensor, cache: KVCache, last_only: bool = False) -> torch.Tensor:
        base = self.model.model
        q_len = input_ids.shape[1]
        positions = self._positions(cache, q_len)
        if self.sliding_window is not None and cache.num_tokens + q_len > self.sliding_window:
            raise ValueError(
                f"sequence exceeds the model's sliding window ({self.sliding_window}); not supported"
            )
        h = base.embed_tokens(input_ids)
        cos, sin = self.rotary(h, positions.unsqueeze(0))
        cos, sin = cos.unsqueeze(1).to(h.dtype), sin.unsqueeze(1).to(h.dtype)
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            x = layer.input_layernorm(h)
            q = attn.q_proj(x).view(1, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = attn.k_proj(x).view(1, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = attn.v_proj(x).view(1, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
            keys, values = cache.update(i, k, v)
            out = cached_attention(q, keys, values, self.scale, cache.attention_mask(i))
            h = h + attn.o_proj(out.transpose(1, 2).reshape(1, q_len, self.num_heads * self.head_dim))
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        cache.num_tokens += q_len
        if last_only:
            h = h[:, -1:]
        return self.model.lm_head(base.norm(h))


def make_adapter(model: torch.nn.Module) -> DecoderAdapter:
    """Wrap a Hugging Face causal LM in the matching adapter."""
    model_type = getattr(model.config, "model_type", None)
    if model_type == "gpt2":
        return GPT2Adapter(model)
    if model_type in ("llama", "mistral", "qwen2"):
        return LlamaAdapter(model)
    raise ValueError(f"model_type '{model_type}' is not supported (supported: {', '.join(SUPPORTED_MODEL_TYPES)})")
