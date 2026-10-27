"""The decoder used by Graph-KV reproduces Hugging Face exactly for ordinary causal decoding."""

import pytest
import torch

from graphkv_neurips.model import APEConfig, KVBuffer, RopeDecoder, Segment
from graphkv_neurips.testing import TINY_CONFIGS, tiny_model


def _ids(n, seed=0):
    return torch.randint(0, 256, (1, n), generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("model_type", TINY_CONFIGS)
def test_logits_match_huggingface(model_type):
    model = tiny_model(model_type)
    dec = RopeDecoder(model)
    ids = _ids(30)
    with torch.no_grad():
        ref = model(ids).logits
        full, _, _ = dec.forward(ids, torch.arange(30).unsqueeze(0), logits="all")
        _, seg = dec.encode(ids[:, :20], torch.arange(20).unsqueeze(0))
        tail, _, _ = dec.forward(ids[:, 20:], torch.arange(20, 30).unsqueeze(0), seg, logits="all")
    assert (full - ref).abs().max() < 1e-5
    assert (tail - ref[:, 20:]).abs().max() < 1e-5


@pytest.mark.parametrize("model_type", TINY_CONFIGS)
def test_moving_cached_keys_equals_encoding_at_new_positions(model_type):
    dec = RopeDecoder(tiny_model(model_type))
    ids = _ids(12, 1)
    with torch.no_grad():
        _, at0 = dec.encode(ids, torch.arange(12).unsqueeze(0))
        _, at100 = dec.encode(ids, 100 + torch.arange(12).unsqueeze(0))
    moved = dec.move(at0, 100 + torch.arange(12))
    for a, b in zip(moved.keys, at100.keys):
        assert (a - b).abs().max() < 1e-5
    # values carry no position; they differ only by float rounding inherited from earlier layers
    assert torch.equal(moved.values[0], at100.values[0])
    for a, b in zip(moved.values, at100.values):
        assert (a - b).abs().max() < 1e-5


def test_chunked_encoding_and_batching_change_nothing():
    dec = RopeDecoder(tiny_model("qwen2"))
    blocks = [_ids(n, s)[0].tolist() for n, s in ((9, 1), (4, 2), (13, 3))]
    with torch.no_grad():
        _, ctx = dec.encode(_ids(5, 9), torch.arange(5).unsqueeze(0))
        batched = dec.encode_batch(blocks, [5, 5, 5], ctx, max_batch_tokens=20)
        for b, seg in zip(blocks, batched):
            ids = torch.as_tensor([b])
            _, alone = dec.encode(ids, 5 + torch.arange(len(b)).unsqueeze(0), ctx)
            _, pieces = dec.encode(ids, 5 + torch.arange(len(b)).unsqueeze(0), ctx, chunk_size=3)
            for x, y, z in zip(seg.keys, alone.keys, pieces.keys):
                assert (x - y).abs().max() < 1e-5 and (y - z).abs().max() < 1e-5


def test_kv_buffer_decoding_matches_huggingface_generate():
    model = tiny_model("llama")
    dec = RopeDecoder(model)
    ids = _ids(10, 4)
    with torch.no_grad():
        ref = model.generate(ids, max_new_tokens=8, do_sample=False, pad_token_id=0)[0, 10:].tolist()
        cache = KVBuffer(dec, 32)
        logits, _, _ = dec.forward(ids, torch.arange(10).unsqueeze(0), cache=cache)
        out = []
        for step in range(8):
            nxt = int(logits[0].argmax())
            out.append(nxt)
            logits, _, _ = dec.forward(torch.tensor([[nxt]]), torch.tensor([[10 + step]]), cache=cache)
    assert out == ref
    assert cache.length == 18


def test_ape_with_unit_temperature_and_scale_is_ordinary_attention():
    """With T = S = 1 the two-part softmax of APE merges back into one softmax."""
    dec = RopeDecoder(tiny_model("llama"))
    with torch.no_grad():
        _, ctx = dec.encode(_ids(15, 5), torch.arange(15).unsqueeze(0))
        q = _ids(4, 6)
        pos = 20 + torch.arange(4).unsqueeze(0)
        mask = torch.zeros(15, dtype=torch.bool)
        mask[5:] = True
        plain, _, _ = dec.forward(q, pos, ctx, logits="all")
        ape, _, _ = dec.forward(q, pos, ctx, logits="all", ape=APEConfig(mask, 1.0, 1.0))
        sharp, _, _ = dec.forward(q, pos, ctx, logits="all", ape=APEConfig(mask, 0.9, 0.9))
    assert (plain - ape).abs().max() < 1e-4
    assert (plain - sharp).abs().max() > 1e-4


def test_segment_cat_and_slice_roundtrip():
    dec = RopeDecoder(tiny_model("llama"))
    with torch.no_grad():
        _, seg = dec.encode(_ids(10, 7), torch.arange(10).unsqueeze(0))
    joined = Segment.cat([seg.slice(0, 4), seg.slice(4, 10)])
    assert torch.equal(joined.positions, seg.positions)
    assert all(torch.equal(a, b) for a, b in zip(joined.keys, seg.keys))
