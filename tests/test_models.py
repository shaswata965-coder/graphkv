import pytest
import torch

from graphkv.cache import GraphKVConfig, compress_cache
from graphkv.models import make_adapter
from graphkv.selftest import TINY_CONFIGS, parity_error, tiny_model


@pytest.mark.parametrize("model_type", list(TINY_CONFIGS))
def test_adapter_matches_huggingface(model_type):
    err = parity_error(tiny_model(model_type))
    assert err["full"] < 1e-4, err
    assert err["incremental"] < 1e-4, err


def test_gpt2_inverse_layer_scaling_matches_huggingface():
    err = parity_error(tiny_model("gpt2", scale_attn_by_inverse_layer_idx=True))
    assert err["incremental"] < 1e-4, err


@pytest.mark.parametrize("model_type", ["gpt2", "llama"])
def test_positions_survive_compression(model_type):
    """After merging, new tokens must still get their true absolute position."""
    model = tiny_model(model_type)
    adapter = make_adapter(model)
    ids = torch.randint(0, model.config.vocab_size, (1, 20), generator=torch.Generator().manual_seed(0))
    cache = adapter.new_cache()
    with torch.inference_mode():
        adapter(ids[:, :16], cache)
        compress_cache(cache, GraphKVConfig(epsilon=1e6, num_sink=4))
        assert cache.layer_lengths() == [5.0] * adapter.num_layers
        assert cache.num_tokens == 16
        adapter(ids[:, 16:17], cache)
    assert cache.num_tokens == 17


def test_gpt2_rejects_sequences_past_position_limit():
    model = tiny_model("gpt2", n_positions=8)
    adapter = make_adapter(model)
    with pytest.raises(ValueError, match="positions"):
        with torch.inference_mode():
            adapter(torch.zeros(1, 9, dtype=torch.long), adapter.new_cache())


@pytest.mark.parametrize("model_type", ["gpt2", "llama"])
def test_padding_entries_have_no_effect(model_type):
    """Invalid (padding) cache slots must be invisible to attention."""
    model = tiny_model(model_type)
    adapter = make_adapter(model)
    ids = torch.randint(0, model.config.vocab_size, (1, 20), generator=torch.Generator().manual_seed(3))
    with torch.inference_mode():
        clean = adapter.new_cache()
        adapter(ids[:, :16], clean)
        padded = adapter.new_cache()
        adapter(ids[:, :16], padded)
        for layer in range(adapter.num_layers):
            k, v = padded.keys[layer], padded.values[layer]
            heads = k.shape[1]
            junk = torch.full((1, heads, 3, k.shape[3]), 1e3)
            padded.keys[layer] = torch.cat([k[:, :, :8], junk, k[:, :, 8:]], dim=2)
            padded.values[layer] = torch.cat([v[:, :, :8], junk, v[:, :, 8:]], dim=2)
            valid = torch.ones(heads, 19, dtype=torch.bool)
            valid[:, 8:11] = False
            padded.valid[layer] = valid
        expected = adapter(ids[:, 16:20], clean)
        got = adapter(ids[:, 16:20], padded)
    assert torch.allclose(got, expected, atol=1e-5)


def test_decoding_on_compressed_per_head_cache():
    model = tiny_model("llama")
    adapter = make_adapter(model)
    ids = torch.randint(0, model.config.vocab_size, (1, 30), generator=torch.Generator().manual_seed(3))
    cache = adapter.new_cache()
    with torch.inference_mode():
        adapter(ids[:, :24], cache)
        compress_cache(cache, GraphKVConfig(epsilon=0.5, num_sink=2))
        assert any(v is not None for v in cache.valid), "test needs a padded cache"
        logits = adapter(ids[:, 24:26], cache)
    assert torch.isfinite(logits).all()
    assert all(v is None or v.shape[1] == k.shape[2] for v, k in zip(cache.valid, cache.keys))


def test_unsupported_model_type():
    class Dummy(torch.nn.Module):
        class config:  # noqa: N801
            model_type = "bert"

    with pytest.raises(ValueError, match="not supported"):
        make_adapter(Dummy())
