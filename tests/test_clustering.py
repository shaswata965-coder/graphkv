import itertools

import pytest
import torch

from graphkv.clustering import (
    cluster_assignment,
    epsilon_adjacency,
    label_propagation,
    merge_clusters,
    pairwise_sq_distances,
)


def reference_components(adj: torch.Tensor) -> list:
    """Smallest node index of each node's connected component (BFS)."""
    n = adj.shape[0]
    label = [-1] * n
    for start in range(n):
        if label[start] != -1:
            continue
        stack, members = [start], []
        label[start] = start
        while stack:
            u = stack.pop()
            members.append(u)
            for v in torch.nonzero(adj[u]).flatten().tolist():
                if label[v] == -1:
                    label[v] = start
                    stack.append(v)
    return label


def reference_fixed_iters(adj: torch.Tensor, iters: int) -> list:
    """Algorithm 1, lines 6-8, written as literally as possible."""
    n = adj.shape[0]
    labels = list(range(n))
    for _ in range(iters):
        labels = [min([labels[i]] + [labels[j] for j in range(n) if adj[i, j]]) for i in range(n)]
    return labels


def test_distances_match_float64_cdist():
    torch.manual_seed(0)
    # Large shared offset, as in real key vectors, to stress numerical precision.
    x = torch.randn(3, 50, 64) * 3 + 200.0
    exact = torch.cdist(x.double(), x.double()) ** 2
    assert torch.allclose(pairwise_sq_distances(x).double(), exact, rtol=1e-4, atol=1e-2)


def test_adjacency_is_strict_threshold_and_reflexive():
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    adj = epsilon_adjacency(x, 1.0)  # distance exactly 1 is NOT < 1
    assert adj.tolist() == [[True, False, False], [False, True, False], [False, False, True]]
    adj = epsilon_adjacency(x, 1.5)
    assert adj.tolist() == [[True, True, False], [True, True, False], [False, False, True]]
    assert epsilon_adjacency(x, 0.0).equal(torch.eye(3, dtype=torch.bool))


def test_adjacency_ignores_invalid_nodes():
    x = torch.zeros(4, 2)  # all identical
    valid = torch.tensor([True, True, False, True])
    adj = epsilon_adjacency(x, 1.0, valid)
    assert adj[2].tolist() == [False, False, True, False]
    assert adj[:, 2].tolist() == [False, False, True, False]
    assert adj[0].tolist() == [True, True, False, True]


@pytest.mark.parametrize("seed,p", itertools.product(range(5), [0.02, 0.05, 0.2]))
def test_converged_propagation_equals_connected_components(seed, p):
    g = torch.Generator().manual_seed(seed)
    n = 60
    upper = torch.rand(n, n, generator=g) < p
    adj = (upper | upper.T) | torch.eye(n, dtype=torch.bool)
    assert label_propagation(adj).tolist() == reference_components(adj)


def test_chain_graph_converges():
    # Path 0-1-...-n-1 in reverse index order: worst case for plain propagation.
    n = 40
    adj = torch.eye(n, dtype=torch.bool)
    for i in range(n - 1):
        adj[i, i + 1] = adj[i + 1, i] = True
    assert label_propagation(adj).tolist() == [0] * n


@pytest.mark.parametrize("iters", [0, 1, 2, 3, 10])
def test_fixed_iterations_match_algorithm_1(iters):
    g = torch.Generator().manual_seed(1)
    n = 30
    upper = torch.rand(n, n, generator=g) < 0.08
    adj = (upper | upper.T) | torch.eye(n, dtype=torch.bool)
    assert label_propagation(adj, max_iters=iters).tolist() == reference_fixed_iters(adj, iters)


def test_batched_propagation_matches_per_graph():
    g = torch.Generator().manual_seed(2)
    upper = torch.rand(4, 25, 25, generator=g) < 0.06
    adj = (upper | upper.transpose(1, 2)) | torch.eye(25, dtype=torch.bool)
    batched = label_propagation(adj)
    for h in range(4):
        assert batched[h].tolist() == reference_components(adj[h])


def test_cluster_assignment_orders_by_first_member_and_skips_invalid():
    labels = torch.tensor([[0, 1, 0, 3, 1, 5], [0, 0, 2, 2, 4, 4]])
    valid = torch.tensor([[True] * 6, [True, True, True, True, False, False]])
    assignment, counts = cluster_assignment(labels, valid)
    assert assignment.tolist() == [[0, 1, 0, 2, 1, 3], [0, 0, 1, 1, -1, -1]]
    assert counts.tolist() == [4, 2]


def test_merge_clusters_means_and_padding():
    x = torch.tensor([[[1.0], [3.0], [10.0]], [[2.0], [4.0], [6.0]]])
    assignment = torch.tensor([[0, 0, 1], [0, 0, 0]])
    counts = torch.tensor([2, 1])
    cent, w = merge_clusters(x, assignment, counts)
    assert cent[0].flatten().tolist() == [2.0, 10.0]
    assert cent[1].flatten().tolist() == [4.0, 0.0]  # second slot is padding
    assert w.tolist() == [[2.0, 1.0], [3.0, 0.0]]


def test_weighted_merge_equals_mean_of_original_tokens():
    torch.manual_seed(0)
    tokens = torch.randn(7, 4)
    # First merge tokens 0-2 and 3-4, then merge everything.
    a = torch.tensor([0, 0, 0, 1, 1, 2, 3])
    cent, w = merge_clusters(tokens, a, torch.tensor(4))
    final, total = merge_clusters(cent, torch.zeros(4, dtype=torch.long), torch.tensor(1), weights=w)
    assert torch.allclose(final[0], tokens.mean(dim=0), atol=1e-6)
    assert total.tolist() == [7.0]
