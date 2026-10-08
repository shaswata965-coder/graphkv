"""Per-layer KV cache and the graph-based compression step (Algorithm 1)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import torch

from .clustering import cluster_assignment, epsilon_adjacency, label_propagation, merge_clusters

GRANULARITIES = ("head", "token")


@dataclass(frozen=True)
class GraphKVConfig:
    """Hyper-parameters of the graph-based KV-cache compression.

    Attributes:
        epsilon: distance threshold ``epsilon`` of Eq. 1. Two cached entries
            are connected when the L2 distance of their key vectors is below it.
        interval: compression interval ``M``. The cache is compressed after
            every ``M`` decoded tokens.
        num_sink: number of leading attention-sink tokens ``N_sink`` that are
            never merged (the paper uses 4).
        max_iters: number of label-propagation iterations ``T``. ``None`` runs
            to convergence (exact connected components).
        granularity: what a graph node is.

            * ``"head"`` (default): Algorithm 1 applied to the 4-D cache
              ``[batch, kv_heads, seq, head_dim]``. ``cdist`` then yields one
              distance matrix per KV head, so every (layer, KV head) is
              clustered independently on its ``head_dim``-dimensional keys.
              Heads can end up with different numbers of entries.
            * ``"token"``: one graph per layer, each node is a token whose
              feature is its key across all KV heads (``kv_heads * head_dim``
              values). All heads share one clustering. Distances are about
              ``sqrt(kv_heads)`` times larger, so comparable compression needs
              a larger ``epsilon``.
        weighted_merge: off by default (paper behaviour: plain mean of the
            current entries, Eq. 2). When on, each entry is weighted by the
            number of original tokens it already represents, so repeated
            merges give the exact mean over original tokens. This is an
            optional extension, not part of the paper.
        proportional_attention: off by default (paper behaviour). When on,
            attention logits of every cache entry get ``+log(n)`` where ``n``
            is the number of original tokens the entry represents, so a
            centroid receives the attention mass of all tokens it replaced
            (as in "proportional attention" of Token Merging, Bolya et al.
            2023). Together with ``weighted_merge`` this is exact when the
            merged entries are identical. Optional extension, not in the paper.
    """

    epsilon: float
    interval: int = 16
    num_sink: int = 4
    max_iters: Optional[int] = None
    granularity: str = "head"
    weighted_merge: bool = False
    proportional_attention: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.epsilon) or self.epsilon < 0:
            raise ValueError(f"epsilon must be a finite value >= 0, got {self.epsilon}")
        if self.interval < 1:
            raise ValueError(f"interval (M) must be >= 1, got {self.interval}")
        if self.num_sink < 0:
            raise ValueError(f"num_sink must be >= 0, got {self.num_sink}")
        if self.max_iters is not None and self.max_iters < 0:
            raise ValueError(f"max_iters (T) must be >= 0 or None, got {self.max_iters}")
        if self.granularity not in GRANULARITIES:
            raise ValueError(f"granularity must be one of {GRANULARITIES}, got '{self.granularity}'")

    def to_dict(self) -> dict:
        return asdict(self)


class KVCache:
    """Key/value cache whose layers (and heads) may hold different numbers of entries.

    Per layer, keys and values are ``[1, num_kv_heads, length, head_dim]``
    tensors. An *entry* is one key + one value vector of one KV head. When
    heads of a layer hold different numbers of entries (per-head clustering),
    the shorter heads are padded and ``valid[layer]`` (``[num_kv_heads,
    length]``, ``None`` when nothing is padded) marks the real entries.
    Padding only exists to keep tensors rectangular; memory statistics count
    real entries only, as a ragged/paged layout would store them.

    ``num_tokens`` counts every token the model has consumed and is used as
    the absolute position of the next token, independent of how far the cache
    has been compressed. Only batch size 1 is supported, because each
    sequence ends up with its own cluster structure.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        track_weights: bool = False,
        proportional_attention: bool = False,
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.keys: List[Optional[torch.Tensor]] = [None] * num_layers
        self.values: List[Optional[torch.Tensor]] = [None] * num_layers
        self.valid: List[Optional[torch.Tensor]] = [None] * num_layers
        # Number of original tokens represented by each entry, ``[heads, length]``
        # (only kept when ``track_weights`` is set, for the weighted-merge extension).
        self.weights: List[Optional[torch.Tensor]] = [None] * num_layers
        self.proportional_attention = proportional_attention
        self.track_weights = track_weights or proportional_attention
        self.num_tokens = 0

    def update(self, layer: int, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new entries to ``layer`` and return its full keys and values."""
        if key.shape[0] != 1:
            raise ValueError("KVCache only supports batch size 1")
        heads, q_len = key.shape[1], key.shape[2]
        if self.keys[layer] is None:
            self.keys[layer], self.values[layer] = key, value
        else:
            self.keys[layer] = torch.cat([self.keys[layer], key], dim=2)
            self.values[layer] = torch.cat([self.values[layer], value], dim=2)
        if self.valid[layer] is not None:
            new = torch.ones(heads, q_len, dtype=torch.bool, device=key.device)
            self.valid[layer] = torch.cat([self.valid[layer], new], dim=1)
        if self.track_weights:
            ones = torch.ones(heads, q_len, device=key.device, dtype=torch.float32)
            w = self.weights[layer]
            self.weights[layer] = ones if w is None else torch.cat([w, ones], dim=1)
        return self.keys[layer], self.values[layer]

    def attention_mask(self, layer: int) -> Optional[torch.Tensor]:
        """Per-entry attention mask for ``layer``, ``[num_kv_heads, length]`` or ``None``.

        Boolean validity mask by default; with proportional attention an
        additive float bias ``log(tokens per entry)`` (``-inf`` on padding).
        """
        valid = self.valid[layer]
        if not self.proportional_attention:
            return valid
        bias = self.weights[layer].log()
        return bias if valid is None else bias.masked_fill(~valid, float("-inf"))

    def layer_entries(self, layer: int) -> int:
        """Real entries of one layer, summed over its KV heads."""
        k = self.keys[layer]
        if k is None:
            return 0
        if self.valid[layer] is None:
            return int(k.shape[1] * k.shape[2])
        return int(self.valid[layer].sum())

    def layer_lengths(self) -> List[float]:
        """Average number of entries per KV head, for every layer."""
        return [self.layer_entries(i) / self.num_kv_heads for i in range(self.num_layers)]

    def num_entries(self) -> int:
        """Real entries summed over all layers and KV heads."""
        return sum(self.layer_entries(i) for i in range(self.num_layers))

    def uncompressed_entries(self) -> int:
        """Entries a standard (uncompressed) cache would hold at this point."""
        return self.num_tokens * self.num_layers * self.num_kv_heads

    def compression_ratio(self) -> float:
        """Fraction of entries saved relative to a standard cache, in ``[0, 1)``."""
        full = self.uncompressed_entries()
        return 0.0 if full == 0 else 1.0 - self.num_entries() / full

    def bytes_per_entry(self) -> int:
        """Bytes needed to store one entry (key + value of one head)."""
        k = next((k for k in self.keys if k is not None), None)
        return 0 if k is None else 2 * k.shape[3] * k.element_size()

    def nbytes(self) -> int:
        """Bytes of the real entries (excluding padding)."""
        return self.num_entries() * self.bytes_per_entry()

    def allocated_nbytes(self) -> int:
        """Bytes actually allocated, including padding."""
        return sum(t.numel() * t.element_size() for t in self.keys + self.values if t is not None)


@dataclass
class CompressionEvent:
    """Bookkeeping for one call of :func:`compress_cache`."""

    step: int
    num_tokens: int
    entries_before: int
    entries_after: int
    uncompressed_entries: int
    layer_lengths_after: List[float]
    seconds: float = 0.0

    @property
    def compression_ratio(self) -> float:
        """Saving relative to a standard cache right after this event."""
        full = self.uncompressed_entries
        return 0.0 if full == 0 else 1.0 - self.entries_after / full


def compress_layer(
    key: torch.Tensor,
    value: torch.Tensor,
    config: GraphKVConfig,
    valid: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Compress one layer's cache (Algorithm 1, lines 2-13).

    Args:
        key, value: ``[1, num_kv_heads, length, head_dim]``.
        config: compression hyper-parameters.
        valid: optional ``[num_kv_heads, length]`` mask of real entries.
        weights: optional ``[num_kv_heads, length]`` number of original tokens
            per entry (tracked for the weighted-merge and proportional-attention
            extensions). If given, the returned weights are updated to match.

    Returns:
        ``(key, value, valid, weights)`` after compression. Each head keeps
        its ``min(N_sink, length)`` sink entries followed by one centroid per
        cluster; heads with fewer clusters are padded (see :class:`KVCache`).
    """
    if key.shape[0] != 1:
        raise ValueError("compress_layer only supports batch size 1")
    _, num_heads, length, head_dim = key.shape
    num_sink = min(config.num_sink, length)
    num_prunable = length - num_sink
    if num_prunable <= 1:
        return key, value, valid, weights

    # Line 3: split into the sink zone and the prunable zone.
    k_prune, v_prune = key[0, :, num_sink:], value[0, :, num_sink:]  # [heads, P, D]
    node_valid = None if valid is None else valid[:, num_sink:]
    node_weights = weights[:, num_sink:] if (config.weighted_merge and weights is not None) else None

    if config.granularity == "token":
        if node_valid is not None:
            raise ValueError("token granularity expects an unpadded cache")
        # One node per token: [heads, P, D] -> [1, P, heads * D].
        k_nodes = k_prune.transpose(0, 1).reshape(1, num_prunable, num_heads * head_dim)
        v_nodes = v_prune.transpose(0, 1).reshape(1, num_prunable, num_heads * head_dim)
        if node_weights is not None:
            node_weights = node_weights[:1]  # identical across heads
    else:
        k_nodes, v_nodes = k_prune, v_prune

    # Lines 4-8: epsilon-graph on keys + label propagation.
    adj = epsilon_adjacency(k_nodes, config.epsilon, node_valid)
    labels = label_propagation(adj, config.max_iters)

    # Lines 9-12: one centroid per cluster for keys and values.
    assignment, num_clusters = cluster_assignment(labels, node_valid)
    num_nodes = num_prunable if node_valid is None else node_valid.sum(dim=-1)
    if bool((num_clusters == num_nodes).all()):
        return key, value, valid, weights  # nothing merged
    k_cent, cluster_weight = merge_clusters(k_nodes, assignment, num_clusters, node_weights)
    v_cent, _ = merge_clusters(v_nodes, assignment, num_clusters, node_weights)
    m = k_cent.shape[-2]

    if config.granularity == "token":
        k_cent = k_cent[0].reshape(m, num_heads, head_dim).transpose(0, 1)
        v_cent = v_cent[0].reshape(m, num_heads, head_dim).transpose(0, 1)
        cluster_weight = cluster_weight.expand(num_heads, m)

    # Line 13: K' = [K_sink, K_centroid], V' = [V_sink, V_centroid].
    new_key = torch.cat([key[:, :, :num_sink], k_cent.unsqueeze(0)], dim=2)
    new_value = torch.cat([value[:, :, :num_sink], v_cent.unsqueeze(0)], dim=2)

    new_valid = None
    if config.granularity == "head" and bool((num_clusters < m).any()):
        centroid_valid = torch.arange(m, device=key.device).unsqueeze(0) < num_clusters.unsqueeze(1)
        sink_valid = torch.ones(num_heads, num_sink, dtype=torch.bool, device=key.device)
        new_valid = torch.cat([sink_valid, centroid_valid], dim=1)

    new_weights = None
    if weights is not None:
        # Original tokens per centroid, whether or not the mean itself was weighted.
        if node_weights is None:
            w = weights[:, num_sink:]
            w = w[:1] if config.granularity == "token" else w
            _, cluster_weight = merge_clusters(w.unsqueeze(-1), assignment, num_clusters, w)
            if config.granularity == "token":
                cluster_weight = cluster_weight.expand(num_heads, m)
        new_weights = torch.cat([weights[:, :num_sink].float(), cluster_weight.float()], dim=1)
    return new_key, new_value, new_valid, new_weights


@torch.no_grad()
def compress_cache(cache: KVCache, config: GraphKVConfig, step: int = 0) -> CompressionEvent:
    """Compress every layer of ``cache`` in place (Algorithm 1, lines 1-15)."""
    before = cache.num_entries()
    for layer in range(cache.num_layers):
        if cache.keys[layer] is None:
            continue
        cache.keys[layer], cache.values[layer], cache.valid[layer], cache.weights[layer] = compress_layer(
            cache.keys[layer], cache.values[layer], config, cache.valid[layer], cache.weights[layer]
        )
    return CompressionEvent(
        step=step,
        num_tokens=cache.num_tokens,
        entries_before=before,
        entries_after=cache.num_entries(),
        uncompressed_entries=cache.uncompressed_entries(),
        layer_lengths_after=cache.layer_lengths(),
    )
