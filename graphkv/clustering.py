"""Graph construction, label propagation and centroid merging.

These are the numerical building blocks of the paper's method (Sec. II-A.2,
Eqs. 1-2 and Algorithm 1, lines 4-12). They operate on batches of independent
graphs: node features have shape ``[..., N, F]`` (any number of leading
"group" dimensions, e.g. one group per attention head) and an optional
``valid`` mask ``[..., N]`` marks padding nodes that must be ignored. They
know nothing about transformers, which keeps them easy to test in isolation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def pairwise_sq_distances(x: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distances between all nodes (``[..., N, F]`` -> ``[..., N, N]``).

    Computed in float32 via the Gram-matrix identity on mean-centred features.
    Centring does not change any distance, but it removes the large shared
    offset that transformer key vectors carry, which would otherwise make the
    ``|a|^2 + |b|^2 - 2ab`` form lose precision. Unlike ``torch.cdist`` in
    exact mode, this only needs a matmul, so it behaves identically on CPU,
    CUDA and Apple MPS.
    """
    x = x.float()
    x = x - x.mean(dim=-2, keepdim=True)
    sq_norms = (x * x).sum(dim=-1)
    d2 = sq_norms.unsqueeze(-1) + sq_norms.unsqueeze(-2) - 2.0 * (x @ x.transpose(-1, -2))
    return d2.clamp_min_(0.0)


def epsilon_adjacency(x: torch.Tensor, epsilon: float, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Boolean adjacency matrix of the epsilon-graph (Eq. 1).

    ``A[u, v] = ||x_u - x_v||_2 < epsilon``. The diagonal is always set, so
    every node is its own neighbour (this is what makes ``L_i`` part of the
    min in Algorithm 1, line 7). Invalid (padding) nodes get no edges.
    """
    n = x.shape[-2]
    eye = torch.eye(n, dtype=torch.bool, device=x.device)
    if epsilon <= 0:
        return eye.expand(*x.shape[:-2], n, n).clone()
    adj = pairwise_sq_distances(x) < float(epsilon) ** 2
    if valid is not None:
        adj &= valid.unsqueeze(-1) & valid.unsqueeze(-2)
    return adj | eye


def label_propagation(adj: torch.Tensor, max_iters: Optional[int] = None) -> torch.Tensor:
    """Min-label propagation over boolean adjacency matrices ``[..., N, N]``.

    Every node starts with its own index as label. One iteration performs the
    synchronous update of Algorithm 1, line 7::

        L_i <- min({L_i} U {L_j | A_ij = 1})

    * ``max_iters=T`` runs exactly ``T`` such iterations (stopping early only
      once a fixed point is reached, which does not change the result). With a
      small ``T`` labels travel at most ``T`` hops, so a large connected
      component can be split into several clusters.
    * ``max_iters=None`` runs to convergence, i.e. returns the exact connected
      components of the graph (the paper's description in Sec. II-A.2). Here a
      pointer-jumping step ``L <- L[L]`` is interleaved to converge in a
      logarithmic number of rounds; it yields the same fixed point (every
      node labelled with the smallest index in its component).

    Returns a ``LongTensor`` of shape ``[..., N]``; ``labels[..., i]`` is the
    smallest node index reached by node ``i`` and identifies its cluster.
    """
    n = adj.shape[-1]
    # int32 labels halve the memory traffic of the [.., N, N] masked min (~3x faster on CPU).
    labels = torch.arange(n, device=adj.device, dtype=torch.int32).expand(adj.shape[:-1]).contiguous()
    if n == 0:
        return labels.long()
    sentinel = torch.tensor(n, device=adj.device, dtype=torch.int32)
    exact = max_iters is None
    iters = n if exact else int(max_iters)
    for _ in range(iters):
        new = torch.where(adj, labels.unsqueeze(-2), sentinel).amin(dim=-1)
        if exact:
            # Pointer jumping: follow labels until they stop changing.
            while True:
                jumped = torch.gather(new, -1, new.long())
                if torch.equal(jumped, new):
                    break
                new = jumped
        if torch.equal(new, labels):
            break
        labels = new
    return labels.long()


def cluster_assignment(
    labels: torch.Tensor, valid: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Turn labels into dense cluster indices (Algorithm 1, line 9).

    Args:
        labels: ``[..., N]`` labels from :func:`label_propagation`.
        valid: optional ``[..., N]`` mask; invalid nodes belong to no cluster.

    Returns:
        ``(assignment [..., N], num_clusters [...])``. ``assignment`` is in
        ``0 .. num_clusters-1`` (``-1`` for invalid nodes), with clusters
        numbered by increasing label, i.e. by their earliest token.
    """
    if valid is None:
        valid = torch.ones_like(labels, dtype=torch.bool)
    n = labels.shape[-1]
    # used[..., l] = some valid node carries label l. A dense comparison (same
    # O(N^2) size as the adjacency) instead of scatter_reduce, for MPS support.
    hits = labels.unsqueeze(-1) == torch.arange(n, device=labels.device)
    used = (hits & valid.unsqueeze(-1)).any(dim=-2).long()
    rank = used.cumsum(dim=-1) - 1
    assignment = torch.where(valid, rank.gather(-1, labels), torch.full_like(labels, -1))
    return assignment, used.sum(dim=-1)


def merge_clusters(
    x: torch.Tensor,
    assignment: torch.Tensor,
    num_clusters: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Replace every cluster by its centroid (Eq. 2).

    Args:
        x: ``[..., N, F]`` features to average (keys or values).
        assignment, num_clusters: output of :func:`cluster_assignment`.
        weights: optional ``[..., N]`` non-negative weights. ``None`` gives the
            paper's plain mean ``1/|C| * sum_{i in C} x_i``. Passing the number
            of original tokens each entry already represents makes the
            centroid the exact mean over original tokens.

    Returns:
        ``(centroids [..., M, F], cluster_weight [..., M])`` where ``M`` is the
        largest cluster count over the leading dimensions. Groups with fewer
        clusters are zero-padded (``cluster_weight == 0`` there).
    """
    m = int(num_clusters.max()) if num_clusters.numel() else 0
    ids = torch.arange(m, device=x.device).unsqueeze(-1)
    # One-hot matmul instead of index_add_: deterministic on every backend.
    one_hot = (assignment.unsqueeze(-2) == ids).float()  # [..., M, N]
    if weights is not None:
        one_hot = one_hot * weights.float().unsqueeze(-2)
    cluster_weight = one_hot.sum(dim=-1)
    centroids = (one_hot @ x.float()) / cluster_weight.clamp_min(1e-12).unsqueeze(-1)
    return centroids.to(x.dtype), cluster_weight
