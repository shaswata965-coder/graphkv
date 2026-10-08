import math

import pytest
import torch

from graphkv.cache import GraphKVConfig
from graphkv.engine import GraphKVEngine
from graphkv.models import make_adapter
from graphkv.selftest import tiny_model


@pytest.fixture(scope="module")
def adapter():
    return make_adapter(tiny_model("llama"))


def prompt(n=20, seed=0, vocab=97):
    return torch.randint(0, vocab, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def test_teacher_forced_nll_matches_single_forward(adapter):
    ids = prompt(30)
    result = GraphKVEngine(adapter).score(ids[:18], ids[18:])
    with torch.inference_mode():
        logits = adapter.model(input_ids=torch.tensor([ids])).logits[0].float()
    log_probs = torch.log_softmax(logits, dim=-1)
    expected = [-float(log_probs[t - 1, ids[t]]) for t in range(18, 30)]
    assert result.token_nll == pytest.approx(expected, abs=1e-4)
    assert result.perplexity == pytest.approx(math.exp(sum(expected) / len(expected)), rel=1e-4)
    assert result.events == []


def test_zero_epsilon_reproduces_baseline(adapter):
    ids = prompt(20)
    base = GraphKVEngine(adapter).generate(ids, 25)
    same = GraphKVEngine(adapter, GraphKVConfig(epsilon=0.0, interval=4)).generate(ids, 25)
    assert same.tokens == base.tokens
    assert same.token_nll == pytest.approx(base.token_nll, abs=1e-5)
    assert len(same.events) == 6  # compression after decode steps 4, 8, ..., 24
    assert same.max_compression == 0.0


def test_compression_steps_and_bookkeeping(adapter):
    ids = prompt(20)
    cfg = GraphKVConfig(epsilon=1e6, interval=8, num_sink=4)
    res = GraphKVEngine(adapter, cfg).generate(ids, 50)
    assert len(res.tokens) == 50
    # 49 tokens are fed back (the 50th is only sampled), compressions at 8..48.
    assert [e.step for e in res.events] == [8, 16, 24, 32, 40, 48]
    assert [e.num_tokens for e in res.events] == [28, 36, 44, 52, 60, 68]
    # After each event every layer/head holds 4 sinks + 1 centroid.
    assert all(e.layer_lengths_after == [5.0] * adapter.num_layers for e in res.events)
    assert res.final_layer_lengths == [6.0] * adapter.num_layers  # + token 49
    assert res.max_compression == pytest.approx(1 - 5 / 68)
    assert res.final_compression == pytest.approx(1 - 6 / 69)
    assert res.peak_kv_bytes >= res.final_kv_bytes


def test_stop_tokens_end_generation(adapter):
    ids = prompt(20)
    base = GraphKVEngine(adapter).generate(ids, 30)
    stop = base.tokens[5]
    res = GraphKVEngine(adapter).generate(ids, 30, stop_token_ids=[stop])
    assert res.tokens == base.tokens[: base.tokens.index(stop) + 1]


def test_seeded_sampling_is_reproducible(adapter):
    ids = prompt(20)
    eng = GraphKVEngine(adapter, GraphKVConfig(epsilon=0.5, interval=4))
    a = eng.generate(ids, 20, do_sample=True, top_k=20, top_p=0.9, generator=torch.Generator().manual_seed(7))
    b = eng.generate(ids, 20, do_sample=True, top_k=20, top_p=0.9, generator=torch.Generator().manual_seed(7))
    assert a.tokens == b.tokens


@pytest.mark.parametrize("granularity", ["head", "token"])
def test_compressed_generation_runs_for_all_granularities(adapter, granularity):
    ids = prompt(40)
    cfg = GraphKVConfig(epsilon=0.5 if granularity == "head" else 0.8, interval=8, granularity=granularity, max_iters=3)
    res = GraphKVEngine(adapter, cfg).generate(ids, 30)
    assert len(res.tokens) == 30
    assert all(math.isfinite(x) for x in res.token_nll)
    assert 0.0 <= res.final_compression < 1.0
