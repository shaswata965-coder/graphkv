"""Attention statistics used to build graphs: chunk boundaries and chunk relevance.

* :func:`cross_boundary_attention` -- for candidate cut points ``b`` in a text,
  the attention mass that the ``window`` tokens after ``b`` put on the tokens
  before ``b`` (sink tokens excluded), averaged over heads and probe layers.
  A low value marks an *attention shift*: cutting there loses the least
  information when the two sides are encoded independently.
* :func:`query_attention_mass` -- how much attention the question puts on each
  context entry when it reads the parallel (independent) chunk encodings; used
  to pick the top-k chunks without an external retriever.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from .model import RopeDecoder, Segment


def default_probe_layers(num_layers: int) -> List[int]:
    """About five layers spread over the middle half of the network."""
    lo, hi = num_layers // 4, max(num_layers // 4, (3 * num_layers) // 4)
    step = max(1, (hi - lo) // 4)
    return list(range(lo, hi + 1, step))[:5]


def _probabilities(decoder: RopeDecoder, q: torch.Tensor, keys: torch.Tensor, ctx_len: int, max_scores: int = 1 << 25):
    """Yield ``(row_start, probs)`` with probs ``[rows, K]`` averaged over heads (bottom-right causal)."""
    heads = q.shape[1]
    k = keys.repeat_interleave(heads // keys.shape[1], dim=1).float()
    n, K = q.shape[2], k.shape[2]
    rows = max(1, max_scores // max(1, heads * K))
    cols = torch.arange(K, device=q.device)
    for r0 in range(0, n, rows):
        qb = q[:, :, r0 : r0 + rows].float()
        scores = torch.matmul(qb, k.transpose(-1, -2)) * decoder.scale
        limit = ctx_len + torch.arange(r0, r0 + qb.shape[2], device=q.device)
        scores = scores.masked_fill(cols[None, :] > limit[:, None], float("-inf"))
        yield r0, torch.softmax(scores, dim=-1).mean(dim=1)[0]


@torch.no_grad()
def cross_boundary_attention(
    decoder: RopeDecoder,
    ids: Sequence[int],
    candidates: Sequence[int],
    *,
    layers: Optional[Sequence[int]] = None,
    window: int = 16,
    sink: int = 4,
) -> torch.Tensor:
    """Score ``[len(candidates)]``: mean attention from ``ids[b:b+window]`` to ``ids[sink:b]``."""
    layers = list(layers) if layers is not None else default_probe_layers(decoder.num_layers)
    cand = torch.as_tensor(list(candidates), dtype=torch.long, device=decoder.device)
    if (cand <= sink).any() or (cand >= len(ids)).any():
        raise ValueError("candidates must lie in (sink, len(ids))")
    total = torch.zeros(len(cand), dtype=torch.float32, device=decoder.device)

    def probe(layer: int, q: torch.Tensor, keys: torch.Tensor) -> None:
        if layer not in layers:
            return
        for r0, p in _probabilities(decoder, q, keys, 0):
            csum = torch.cumsum(p, dim=-1)  # csum[t, x] = sum_{s <= x} p[t, s]
            before_b = csum[:, cand - 1]
            before_sink = csum[:, sink - 1 : sink] if sink > 0 else torch.zeros_like(before_b[:, :1])
            t = torch.arange(r0, r0 + p.shape[0], device=p.device)[:, None]
            inside = (t >= cand[None, :]) & (t < cand[None, :] + window)
            total.add_(((before_b - before_sink) * inside).sum(dim=0))

    x = torch.as_tensor([list(ids)])
    decoder.forward(x, torch.arange(len(ids)).unsqueeze(0), logits="none", probe=probe)
    rows = torch.clamp(len(ids) - cand, max=window).float()
    return (total / (rows * len(layers))).cpu()


@torch.no_grad()
def query_attention_mass(
    decoder: RopeDecoder,
    context: Segment,
    query_ids: Sequence[int],
    query_start: int,
    *,
    layers: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Attention mass ``[len(context)]`` from the question tokens, averaged over heads, tokens and probe layers."""
    layers = list(layers) if layers is not None else default_probe_layers(decoder.num_layers)
    mass = torch.zeros(len(context), dtype=torch.float32, device=decoder.device)
    n_ctx = len(context)

    def probe(layer: int, q: torch.Tensor, keys: torch.Tensor) -> None:
        if layer not in layers:
            return
        for _, p in _probabilities(decoder, q, keys, n_ctx):
            mass.add_(p[:, :n_ctx].sum(dim=0))

    pos = query_start + torch.arange(len(query_ids)).unsqueeze(0)
    decoder.forward(torch.as_tensor([list(query_ids)]), pos, context, logits="none", probe=probe)
    return (mass / (len(query_ids) * len(layers))).cpu()


def chunk_totals(mass: torch.Tensor, lengths: Sequence[int], offset: int = 0) -> List[float]:
    """Sum a per-entry mass over consecutive chunks of ``lengths`` that start at ``offset``."""
    out, cursor = [], offset
    for n in lengths:
        out.append(float(mass[cursor : cursor + n].sum()))
        cursor += n
    return out
