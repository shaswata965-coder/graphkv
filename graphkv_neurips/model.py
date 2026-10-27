"""RoPE decoder forward over an explicit KV context.

Graph-KV needs three things that the stock Hugging Face ``forward``/``generate``
path does not give portably: a block of new tokens must attend to an
*arbitrary* set of cached KV entries (the KV caches of its source chunks), it
must get *arbitrary* RoPE positions (shared position ranges), and APE needs a
modified softmax over the context. The official code gets the first two from
FlashAttention-2's bottom-right causal alignment (it passes no mask), so it
only runs on CUDA with ``flash-attn`` and transformers 4.50.

This module reuses the Hugging Face *modules* of a loaded model (embeddings,
projections, norms, MLPs, rotary embedding) and implements only the attention
over ``[context, new tokens]`` itself: every context entry is visible, the new
tokens attend to each other causally. ``tests/test_gkv_model.py`` checks that
the logits equal Hugging Face's for plain causal decoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

SUPPORTED_MODEL_TYPES = ("llama", "mistral", "qwen2", "qwen3")

# RoPE variants whose rotation angle is linear in the position, so that
# R(p + d) = R(d) R(p) and cached keys can be moved to new positions exactly.
_SHIFTABLE_ROPE = (None, "default", "linear", "llama3")


@dataclass
class Segment:
    """KV entries of a run of tokens.

    ``keys[l]`` / ``values[l]`` have shape ``[1, num_kv_heads, n, head_dim]``;
    keys are stored after RoPE at ``positions`` (shape ``[n]``), as in Hugging Face.
    """

    keys: List[torch.Tensor]
    values: List[torch.Tensor]
    positions: torch.Tensor

    def __len__(self) -> int:
        return int(self.positions.numel())

    @staticmethod
    def cat(segments: Sequence["Segment"]) -> "Segment":
        segments = [s for s in segments if len(s) > 0]
        if not segments:
            raise ValueError("cannot concatenate an empty list of segments")
        if len(segments) == 1:
            return segments[0]
        num_layers = len(segments[0].keys)
        return Segment(
            keys=[torch.cat([s.keys[l] for s in segments], dim=2) for l in range(num_layers)],
            values=[torch.cat([s.values[l] for s in segments], dim=2) for l in range(num_layers)],
            positions=torch.cat([s.positions for s in segments]),
        )

    def slice(self, start: int, stop: int) -> "Segment":
        return Segment(
            keys=[k[:, :, start:stop] for k in self.keys],
            values=[v[:, :, start:stop] for v in self.values],
            positions=self.positions[start:stop],
        )

    def nbytes(self) -> int:
        return sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in zip(self.keys, self.values))


class KVBuffer:
    """Pre-allocated, growable KV cache for decoding one sequence on top of a context."""

    def __init__(self, decoder: "RopeDecoder", capacity: int, context: Optional[Segment] = None) -> None:
        n = len(context) if context is not None else 0
        if capacity < n:
            raise ValueError("capacity is smaller than the context")
        shape = (1, decoder.num_kv_heads, capacity, decoder.head_dim)
        kw = dict(dtype=decoder.dtype, device=decoder.device)
        self.keys = [torch.zeros(shape, **kw) for _ in range(decoder.num_layers)]
        self.values = [torch.zeros(shape, **kw) for _ in range(decoder.num_layers)]
        self.positions = torch.zeros(capacity, dtype=torch.long, device=decoder.device)
        self.length = 0
        if context is not None and n:
            for l in range(len(self.keys)):
                self.keys[l][:, :, :n] = context.keys[l]
                self.values[l][:, :, :n] = context.values[l]
            self.positions[:n] = context.positions
            self.length = n

    def write(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        stop = self.length + k.shape[2]
        if stop > self.keys[layer].shape[2]:
            raise RuntimeError("KV buffer is full")
        self.keys[layer][:, :, self.length : stop] = k
        self.values[layer][:, :, self.length : stop] = v
        return self.keys[layer][:, :, :stop], self.values[layer][:, :, :stop]

    def commit(self, positions: torch.Tensor) -> None:
        n = positions.numel()
        self.positions[self.length : self.length + n] = positions.reshape(-1)
        self.length += n

    def segment(self) -> Segment:
        n = self.length
        return Segment([k[:, :, :n] for k in self.keys], [v[:, :, :n] for v in self.values], self.positions[:n])


@dataclass
class APEConfig:
    """Adaptive Parallel Encoding (Yang et al., ICLR 2025) attention for query/answer tokens.

    Context entries (``context_mask`` True) get logits divided by ``temperature``;
    their log-sum-exp is multiplied by ``scale * temperature`` before it is
    combined with the log-sum-exp over the remaining entries (prefix and the
    query/answer tokens), exactly as in ``ape/ape_llama.py``.
    """

    context_mask: torch.Tensor  # [context_len] bool
    temperature: float = 0.9
    scale: float = 0.9


def _supports_gqa() -> bool:
    try:
        q = torch.zeros(1, 2, 1, 4)
        kv = torch.zeros(1, 1, 1, 4)
        F.scaled_dot_product_attention(q, kv, kv, enable_gqa=True)
        return True
    except TypeError:  # torch < 2.5
        return False


_HAS_GQA = _supports_gqa()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RopeDecoder:
    """Llama-family decoder (Llama, Mistral, Qwen2, Qwen3) driven with explicit positions and context."""

    def __init__(self, model: torch.nn.Module) -> None:
        cfg = model.config
        if cfg.model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(f"model_type '{cfg.model_type}' is not supported (supported: {', '.join(SUPPORTED_MODEL_TYPES)})")
        if model.training:
            raise RuntimeError("call model.eval() first; dropout is not applied here")
        self.model = model
        self.config = cfg
        base = model.model
        self.base = base
        self.layers = base.layers
        self.num_layers = len(self.layers)
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        attn = self.layers[0].self_attn
        self.rotary = getattr(base, "rotary_emb", None) or getattr(attn, "rotary_emb", None)
        if self.rotary is None:
            raise RuntimeError("could not find the rotary embedding module; please install transformers>=4.45")
        self.scale = float(getattr(attn, "scaling", self.head_dim**-0.5))
        self.qk_norm = getattr(attn, "q_norm", None) is not None
        self.sliding_window = self._sliding_window(cfg)
        rope = getattr(cfg, "rope_scaling", None) or getattr(cfg, "rope_parameters", None) or {}
        self.rope_type = rope.get("rope_type", rope.get("type")) if isinstance(rope, dict) else None

    @staticmethod
    def _sliding_window(cfg) -> Optional[int]:
        window = getattr(cfg, "sliding_window", None)
        if window is None:
            return None
        if cfg.model_type in ("qwen2", "qwen3"):
            layer_types = getattr(cfg, "layer_types", None)
            uses = any(t == "sliding_attention" for t in layer_types) if layer_types else getattr(cfg, "use_sliding_window", False)
            return window if uses else None
        return window

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    # ------------------------------------------------------------------ positions

    def _cos_sin(self, positions: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """``positions`` [B, n] -> cos, sin of shape [B, 1, n, head_dim]."""
        probe = torch.empty(1, dtype=dtype, device=self.device)
        cos, sin = self.rotary(probe, positions.to(self.device))
        return cos.unsqueeze(1), sin.unsqueeze(1)

    def move(self, segment: Segment, new_positions: torch.Tensor) -> Segment:
        """Re-rotate cached keys from ``segment.positions`` to ``new_positions`` (values are position-free).

        RoPE angles are linear in the position, so rotating by the difference
        gives the keys the tokens would have had at the new positions.
        """
        new_positions = new_positions.to(segment.positions.device)
        if torch.equal(new_positions, segment.positions):
            return segment
        if self.rope_type not in _SHIFTABLE_ROPE:
            raise ValueError(f"moving cached keys is not supported for rope_type '{self.rope_type}'")
        delta = (new_positions - segment.positions).unsqueeze(0)
        cos, sin = self._cos_sin(delta, torch.float32)
        keys = []
        for k in segment.keys:
            k32 = k.float()
            keys.append((k32 * cos + _rotate_half(k32) * sin).to(k.dtype))
        return Segment(keys=keys, values=list(segment.values), positions=new_positions.clone())

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        context: Optional[Segment] = None,
        *,
        cache: Optional[KVBuffer] = None,
        logits: str = "last",
        lengths: Optional[Sequence[int]] = None,
        ape: Optional[APEConfig] = None,
        probe: Optional[Callable[[int, torch.Tensor, torch.Tensor], None]] = None,
    ) -> Tuple[Optional[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Run ``input_ids`` [B, n] at ``positions`` [B, n] on top of a shared ``context``.

        ``probe(layer, q, keys)`` is called in every layer with the rotated
        queries and all keys the new tokens attend to (for attention analyses).

        All rows attend to every context entry and causally to their own new
        tokens. Rows may be right-padded; ``lengths`` selects the last real
        token for ``logits="last"`` (padding never influences real tokens
        because it comes after them). With ``cache`` (batch size 1) the context
        is the buffer's content and the new KV entries are appended to it.

        Returns ``(logits, new_keys, new_values)`` with per-layer tensors
        ``[B, num_kv_heads, n, head_dim]``; ``logits`` is ``[B, vocab]`` for
        ``"last"``, ``[B, n, vocab]`` for ``"all"`` and ``None`` for ``"none"``.
        """
        if input_ids.dim() != 2 or positions.shape != input_ids.shape:
            raise ValueError("input_ids and positions must both have shape [batch, seq_len]")
        if cache is not None and (context is not None or input_ids.shape[0] != 1):
            raise ValueError("cache takes a single sequence and replaces context")
        bsz, q_len = input_ids.shape
        ctx_len = cache.length if cache is not None else (len(context) if context is not None else 0)
        if self.sliding_window is not None and ctx_len + q_len > self.sliding_window:
            raise ValueError(f"context exceeds the model's sliding window ({self.sliding_window}); not supported")
        device = self.device
        input_ids = input_ids.to(device)
        h = self.base.embed_tokens(input_ids)
        cos, sin = self._cos_sin(positions, h.dtype)
        causal = torch.ones(q_len, q_len, dtype=torch.bool, device=device).tril()
        mask = torch.cat([torch.ones(q_len, ctx_len, dtype=torch.bool, device=device), causal], dim=1)
        new_keys: List[torch.Tensor] = []
        new_values: List[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            x = layer.input_layernorm(h)
            q = attn.q_proj(x).view(bsz, q_len, self.num_heads, self.head_dim)
            k = attn.k_proj(x).view(bsz, q_len, self.num_kv_heads, self.head_dim)
            v = attn.v_proj(x).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            if self.qk_norm:
                q, k = attn.q_norm(q), attn.k_norm(k)
            q, k = q.transpose(1, 2), k.transpose(1, 2)
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
            new_keys.append(k)
            new_values.append(v)
            if cache is not None:
                keys, values = cache.write(i, k, v)
            elif context is not None:
                keys = torch.cat([context.keys[i].to(k.dtype).expand(bsz, -1, -1, -1), k], dim=2)
                values = torch.cat([context.values[i].to(v.dtype).expand(bsz, -1, -1, -1), v], dim=2)
            else:
                keys, values = k, v
            if probe is not None:
                probe(i, q, keys)
            if ape is not None:
                out = self._ape_attention(q, keys, values, ctx_len, ape)
            else:
                out = self._attention(q, keys, values, mask)
            h = h + attn.o_proj(out.transpose(1, 2).reshape(bsz, q_len, self.num_heads * self.head_dim))
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        if cache is not None:
            cache.commit(positions.to(cache.positions.device))
        if logits == "none":
            return None, new_keys, new_values
        if logits == "last":
            if lengths is None:
                h = h[:, -1]
            else:
                idx = torch.as_tensor([int(n) - 1 for n in lengths], device=device)
                h = h[torch.arange(bsz, device=device), idx]
        elif isinstance(logits, int) and not isinstance(logits, bool) and logits > 0:
            h = h[:, -logits:]  # the last ``logits`` positions
        elif logits != "all":
            raise ValueError("logits must be 'last', 'all', 'none' or a positive int")
        return self.model.lm_head(self.base.norm(h)), new_keys, new_values

    def _repeat_kv(self, t: torch.Tensor) -> torch.Tensor:
        if self.num_kv_heads == self.num_heads:
            return t
        return t.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

    def _sdpa(self, q, keys, values, **kw) -> torch.Tensor:
        if self.num_kv_heads != self.num_heads and _HAS_GQA:
            return F.scaled_dot_product_attention(q, keys, values, scale=self.scale, enable_gqa=True, **kw)
        return F.scaled_dot_product_attention(q, self._repeat_kv(keys), self._repeat_kv(values), scale=self.scale, **kw)

    def _attention(self, q, keys, values, mask, max_scores: int = 1 << 26) -> torch.Tensor:
        q_len, k_len = q.shape[2], keys.shape[2]
        if q_len == 1:  # one new token sees everything
            return self._sdpa(q, keys, values)
        if q_len == k_len:  # no context: plain causal attention, memory-efficient kernel
            return self._sdpa(q, keys, values, is_causal=True)
        # context + causal block: process query rows in slices so the score matrix stays bounded
        rows = max(1, max_scores // max(1, q.shape[0] * q.shape[1] * k_len))
        if rows >= q_len:
            return self._sdpa(q, keys, values, attn_mask=mask)
        outs = [self._sdpa(q[:, :, s : s + rows], keys, values, attn_mask=mask[s : s + rows]) for s in range(0, q_len, rows)]
        return torch.cat(outs, dim=2)

    def _ape_attention(self, q, keys, values, ctx_len: int, ape: APEConfig) -> torch.Tensor:
        """Two-part softmax of APE: context with temperature, the rest causal, merged via scaled LSE."""
        keys, values = self._repeat_kv(keys), self._repeat_kv(values)
        q_len = q.shape[2]
        scores = torch.matmul(q.float(), keys.float().transpose(-1, -2)) * self.scale
        is_ctx = torch.zeros(keys.shape[2], dtype=torch.bool, device=q.device)
        is_ctx[: ape.context_mask.numel()] = ape.context_mask.to(q.device)
        visible = torch.ones(q_len, keys.shape[2], dtype=torch.bool, device=q.device)
        visible[:, ctx_len:] = torch.ones(q_len, q_len, dtype=torch.bool, device=q.device).tril()
        neg = torch.finfo(torch.float32).min
        ctx_scores = torch.where(is_ctx & visible, scores / ape.temperature, neg)
        oth_scores = torch.where(~is_ctx & visible, scores, neg)
        lse_ctx = torch.logsumexp(ctx_scores, dim=-1, keepdim=True)
        lse_oth = torch.logsumexp(oth_scores, dim=-1, keepdim=True)
        out_ctx = torch.matmul(torch.softmax(ctx_scores, dim=-1), values.float())
        out_oth = torch.matmul(torch.softmax(oth_scores, dim=-1), values.float())
        w = torch.softmax(torch.cat([lse_ctx * (ape.scale * ape.temperature), lse_oth], dim=-1), dim=-1)
        return (w[..., :1] * out_ctx + w[..., 1:] * out_oth).to(q.dtype)

    # ------------------------------------------------------------------ helpers

    @torch.no_grad()
    def encode(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        context: Optional[Segment] = None,
        *,
        chunk_size: Optional[int] = None,
        logits: str = "none",
    ) -> Tuple[Optional[torch.Tensor], Segment]:
        """Encode one block ``[1, n]`` and return its KV as a :class:`Segment`.

        With ``chunk_size`` the block is fed in pieces (each piece attends to the
        context and to the earlier pieces), which bounds the attention-score
        memory for long blocks without changing the result.
        """
        if input_ids.shape[0] != 1:
            raise ValueError("encode() takes a single block; use encode_batch() for several")
        n = input_ids.shape[1]
        step = chunk_size or n
        pieces: List[Segment] = []
        out_logits = None
        for start in range(0, n, step):
            stop = min(n, start + step)
            ctx = Segment.cat(([context] if context is not None else []) + pieces) if (context is not None or pieces) else None
            want = logits if stop == n else "none"
            out_logits, ks, vs = self.forward(input_ids[:, start:stop], positions[:, start:stop], ctx, logits=want)
            pieces.append(Segment(ks, vs, positions[0, start:stop].to(self.device)))
        return out_logits, Segment.cat(pieces)

    @torch.no_grad()
    def encode_batch(
        self,
        blocks: Sequence[Sequence[int]],
        starts: Sequence[int],
        context: Optional[Segment] = None,
        *,
        pad_id: int = 0,
        max_batch_tokens: int = 16384,
    ) -> List[Segment]:
        """Encode independent blocks that share one ``context``; block ``i`` occupies positions ``starts[i] + [0, len)``.

        Blocks are right-padded and batched (up to ``max_batch_tokens`` padded
        tokens per forward pass); the result is identical to encoding each block alone.
        """
        order = sorted(range(len(blocks)), key=lambda i: len(blocks[i]))
        out: List[Optional[Segment]] = [None] * len(blocks)
        group: List[int] = []

        def flush() -> None:
            if not group:
                return
            width = max(len(blocks[i]) for i in group)
            ids = torch.full((len(group), width), pad_id, dtype=torch.long)
            pos = torch.zeros((len(group), width), dtype=torch.long)
            for row, i in enumerate(group):
                ids[row, : len(blocks[i])] = torch.as_tensor(list(blocks[i]), dtype=torch.long)
                pos[row] = starts[i] + torch.arange(width)
            _, ks, vs = self.forward(ids, pos, context, logits="none")
            for row, i in enumerate(group):
                n = len(blocks[i])
                out[i] = Segment([k[row : row + 1, :, :n] for k in ks], [v[row : row + 1, :, :n] for v in vs], pos[row, :n].to(self.device))
            group.clear()

        for i in order:
            if len(blocks[i]) == 0:
                raise ValueError("cannot encode an empty block")
            width = max([len(blocks[j]) for j in group] + [len(blocks[i])])
            if group and width * (len(group) + 1) > max_batch_tokens:
                flush()
            group.append(i)
        flush()
        return out  # type: ignore[return-value]

    def kv_bytes_per_token(self) -> int:
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * torch.tensor([], dtype=self.dtype).element_size()
