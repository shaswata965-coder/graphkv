import math

import pytest
import torch

from graphkv.cache import GraphKVConfig, KVCache, compress_cache, compress_layer
from graphkv.models import cached_attention


def make_kv(heads=2, length=12, dim=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(1, heads, length, dim, generator=g)
    v = torch.randn(1, heads, length, dim, generator=g)
    return k, v


def test_config_validation():
    with pytest.raises(ValueError):
        GraphKVConfig(epsilon=-1)
    with pytest.raises(ValueError):
        GraphKVConfig(epsilon=1, interval=0)
    with pytest.raises(ValueError):
        GraphKVConfig(epsilon=1, granularity="layer")
    with pytest.raises(ValueError):
        GraphKVConfig(epsilon=float("nan"))


@pytest.mark.parametrize("granularity", ["head", "token"])
def test_zero_epsilon_is_identity(granularity):
    k, v = make_kv()
    out = compress_layer(k, v, GraphKVConfig(epsilon=0.0, granularity=granularity))
    assert out[0] is k and out[1] is v and out[2] is None


@pytest.mark.parametrize("granularity", ["head", "token"])
def test_huge_epsilon_keeps_sinks_and_one_centroid(granularity):
    k, v = make_kv(length=12)
    nk, nv, valid, _ = compress_layer(k, v, GraphKVConfig(epsilon=1e6, num_sink=4, granularity=granularity))
    assert nk.shape == (1, 2, 5, 4) and valid is None
    assert torch.equal(nk[:, :, :4], k[:, :, :4])  # sinks untouched
    assert torch.allclose(nk[:, :, 4], k[:, :, 4:].mean(dim=2), atol=1e-6)
    assert torch.allclose(nv[:, :, 4], v[:, :, 4:].mean(dim=2), atol=1e-6)


def test_head_granularity_clusters_heads_independently():
    # Head 0: tokens come in identical pairs. Head 1: all tokens far apart.
    base = torch.arange(8, dtype=torch.float32).repeat_interleave(2) * 100.0
    k = torch.zeros(1, 2, 16, 3)
    k[0, 0, :, 0] = base
    k[0, 1, :, 0] = torch.arange(16, dtype=torch.float32) * 100.0
    v = torch.randn(1, 2, 16, 3)
    nk, nv, valid, _ = compress_layer(k, v, GraphKVConfig(epsilon=1.0, num_sink=2))
    # Prunable tokens 2..15 -> head 0 has 7 pairs, head 1 keeps 14 entries.
    assert nk.shape[2] == 2 + 14
    assert valid[0].tolist() == [True] * 9 + [False] * 7
    assert valid[1].all()
    assert torch.allclose(nv[0, 0, 2], v[0, 0, 2:4].mean(dim=0))
    assert torch.equal(nk[0, 1], k[0, 1])


def test_token_granularity_needs_all_heads_close():
    k = torch.zeros(1, 2, 6, 1)
    k[0, 0, :, 0] = torch.tensor([0.0, 0.0, 0.0, 0.1, 50.0, 50.1])
    k[0, 1, :, 0] = torch.tensor([0.0, 0.0, 0.0, 0.1, 0.0, 99.0])
    v = torch.randn(1, 2, 6, 1)
    nk, _, valid, _ = compress_layer(k, v, GraphKVConfig(epsilon=1.0, num_sink=2, granularity="token"))
    # Tokens 2,3 merge; token 4 is close to them in head 1 but not in head 0; 5 is alone.
    assert nk.shape[2] == 2 + 3 and valid is None


def test_repeated_compression_of_padded_cache():
    torch.manual_seed(0)
    cfg = GraphKVConfig(epsilon=2.5, num_sink=4)
    cache = KVCache(num_layers=1, num_kv_heads=3)
    for step in range(1, 61):
        k = torch.randn(1, 3, 1, 4)
        k[0, 0] *= 0.1  # head 0 merges aggressively, others less
        cache.update(0, k, torch.randn(1, 3, 1, 4))
        cache.num_tokens += 1
        if step % 8 == 0:
            event = compress_cache(cache, cfg, step)
            assert event.entries_after <= event.entries_before
            valid = cache.valid[0]
            if valid is not None:
                assert valid[:, :4].all()  # sinks always real
                assert cache.keys[0].shape[2] == valid.shape[1]
    assert cache.layer_lengths()[0] < 60
    assert 0.0 < cache.compression_ratio() < 1.0


def test_weighted_merge_tracks_token_counts():
    cfg = GraphKVConfig(epsilon=1e6, num_sink=1, weighted_merge=True)
    cache = KVCache(num_layers=1, num_kv_heads=1, track_weights=True)
    values = []
    for step in range(1, 9):
        v = torch.full((1, 1, 1, 1), float(step))
        values.append(float(step))
        cache.update(0, torch.zeros(1, 1, 1, 1), v)
        cache.num_tokens += 1
        if step % 3 == 0:
            compress_cache(cache, cfg, step)
    compress_cache(cache, cfg, 9)
    # Sink is token 1; the centroid must be the exact mean of tokens 2..8.
    assert cache.values[0][0, 0, 1, 0].item() == pytest.approx(sum(values[1:]) / 7)
    assert cache.weights[0].tolist() == [[1.0, 7.0]]


def test_masked_attention_equals_per_head_attention():
    torch.manual_seed(0)
    heads, kv_heads, length, dim = 4, 2, 9, 8
    q = torch.randn(1, heads, 1, dim)
    k = torch.randn(1, kv_heads, length, dim)
    v = torch.randn(1, kv_heads, length, dim)
    valid = torch.ones(kv_heads, length, dtype=torch.bool)
    valid[1, 3:6] = False  # head 1 has 3 padding slots in the middle
    out = cached_attention(q, k, v, 0.5, valid)
    for h in range(heads):
        kv = h // (heads // kv_heads)
        keep = valid[kv]
        ref = torch.softmax(q[0, h] @ k[0, kv, keep].T * 0.5, dim=-1) @ v[0, kv, keep]
        assert torch.allclose(out[0, h], ref, atol=1e-6)


def test_compression_ratio_accounting():
    cache = KVCache(num_layers=2, num_kv_heads=2)
    k, v = make_kv(length=10)
    for layer in range(2):
        cache.update(layer, k, v)
    cache.num_tokens = 10
    assert cache.compression_ratio() == 0.0
    event = compress_cache(cache, GraphKVConfig(epsilon=1e6, num_sink=4))
    # Each layer/head keeps 4 sinks + 1 centroid = 5 of 10 entries.
    assert event.compression_ratio == pytest.approx(0.5)
    assert cache.nbytes() == 2 * 2 * 5 * 2 * 4 * 4


@pytest.mark.parametrize("granularity", ["head", "token"])
def test_proportional_attention_is_exact_for_duplicate_entries(granularity):
    """Merging identical entries with size-proportional attention changes nothing."""
    torch.manual_seed(0)
    heads, dim = 2, 8
    distinct_k, distinct_v = torch.randn(1, heads, 6, dim) * 5, torch.randn(1, heads, 6, dim)
    repeats = torch.tensor([1, 1, 3, 1, 4, 2])  # tokens 2, 4, 5 are repeated
    k, v = distinct_k.repeat_interleave(repeats, dim=2), distinct_v.repeat_interleave(repeats, dim=2)
    query = torch.randn(1, 4, 1, dim)  # 4 query heads -> GQA over 2 KV heads

    cfg = GraphKVConfig(epsilon=1e-3, num_sink=2, granularity=granularity, weighted_merge=True, proportional_attention=True)
    cache = KVCache(num_layers=1, num_kv_heads=heads, proportional_attention=True)
    cache.update(0, k, v)
    cache.num_tokens = k.shape[2]
    before = cached_attention(query, cache.keys[0], cache.values[0], 0.3, cache.attention_mask(0))
    compress_cache(cache, cfg)
    assert cache.keys[0].shape[2] == 6
    assert cache.weights[0][0].tolist() == [1.0, 1.0, 3.0, 1.0, 4.0, 2.0]
    after = cached_attention(query, cache.keys[0], cache.values[0], 0.3, cache.attention_mask(0))
    assert torch.allclose(before, after, atol=1e-5)

    # Without the size bias the same merge does change the output.
    plain = cached_attention(query, cache.keys[0], cache.values[0], 0.3, None)
    assert not torch.allclose(before, plain, atol=1e-3)


def test_token_counts_accumulate_without_weighted_merge():
    cfg = GraphKVConfig(epsilon=1e6, num_sink=1, proportional_attention=True)
    cache = KVCache(num_layers=1, num_kv_heads=1, proportional_attention=True)
    for step in range(1, 9):
        cache.update(0, torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1))
        cache.num_tokens += 1
        if step % 3 == 0:
            compress_cache(cache, cfg, step)
    compress_cache(cache, cfg, 9)
    assert cache.weights[0].tolist() == [[1.0, 7.0]]
    assert cache.attention_mask(0)[0, 1].item() == pytest.approx(torch.log(torch.tensor(7.0)).item())


def test_proportional_mask_marks_padding():
    torch.manual_seed(0)
    k = torch.zeros(1, 2, 8, 1)
    k[0, 0, :, 0] = torch.tensor([0, 1, 2, 2, 2, 2, 9, 9.0])  # head 0 merges, head 1 does not
    k[0, 1, :, 0] = torch.arange(8.0) * 10
    cache = KVCache(num_layers=1, num_kv_heads=2, proportional_attention=True)
    cache.update(0, k, torch.randn(1, 2, 8, 1))
    cache.num_tokens = 8
    compress_cache(cache, GraphKVConfig(epsilon=0.5, num_sink=2, proportional_attention=True))
    mask = cache.attention_mask(0)
    assert mask.shape == (2, 8)
    assert mask[0].tolist()[:4] == pytest.approx([0.0, 0.0, math.log(4), math.log(2)])
    assert mask[0, 4:].tolist() == [float("-inf")] * 4
    assert mask[1].tolist() == [0.0] * 8
