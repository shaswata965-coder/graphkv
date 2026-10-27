"""Properties of Graph-KV stated in Sec. 3 of the paper, checked on tiny random models."""

import itertools

import pytest
import torch

from graphkv_neurips.graph import GraphPlan, encode_graph
from graphkv_neurips.methods import Method, Prompt, decode, generate, parse_method, prepare
from graphkv_neurips.model import RopeDecoder
from graphkv_neurips.testing import tiny_model


def _tokens(n, seed):
    return torch.randint(3, 256, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def _hf_kv(model, ids, positions, mask=None):
    with torch.no_grad():
        out = model(torch.as_tensor([ids]), position_ids=torch.as_tensor([positions]), attention_mask=mask, use_cache=True)
    cache = out.past_key_values
    layers = range(model.config.num_hidden_layers)
    if hasattr(cache, "layers"):
        return [cache.layers[l].keys for l in layers], [cache.layers[l].values for l in layers]
    return [cache.key_cache[l] for l in layers], [cache.value_cache[l] for l in layers]


@pytest.mark.parametrize("prefix_len", [0, 5])
def test_fig2_target_equals_reading_source_then_target(prefix_len):
    """Fig. 2: attending doc.1 to the KV of doc.2 is reading doc.2 followed by doc.1."""
    model = tiny_model("llama")
    dec = RopeDecoder(model)
    doc2, doc1, prefix = _tokens(9, 1), _tokens(6, 2), _tokens(prefix_len, 3)
    plan = GraphPlan.from_edges(2, [(0, 1)])
    enc = encode_graph(dec, prefix, [doc2, doc1], plan)
    # sequential reading of [doc2, doc1] starting where the shared source range starts
    keys, values = _hf_kv(model, doc2 + doc1, list(range(prefix_len, prefix_len + 15)))
    target = slice(prefix_len + 9, prefix_len + 15)  # prefix, doc2 (source), doc1 (target)
    for l in range(2):
        assert (enc.segment.keys[l][:, :, target] - keys[l][:, :, 9:]).abs().max() < 1e-5
        assert (enc.segment.values[l][:, :, target] - values[l][:, :, 9:]).abs().max() < 1e-5
    assert enc.segment.positions.tolist() == list(range(prefix_len + 15))
    assert enc.query_start == prefix_len + 15


def test_graph_structured_block_mask():
    """Fig. 1: Graph-KV equals one forward pass with a graph-structured block mask and shared positions."""
    model = tiny_model("qwen2")
    dec = RopeDecoder(model)
    d2, d3, d1 = _tokens(7, 4), _tokens(5, 5), _tokens(6, 6)
    n = 18
    allow = torch.zeros(n, n, dtype=torch.bool)
    allow[:7, :7] = torch.ones(7, 7, dtype=torch.bool).tril()
    allow[7:12, 7:12] = torch.ones(5, 5, dtype=torch.bool).tril()
    allow[12:, :12] = True
    allow[12:, 12:] = torch.ones(6, 6, dtype=torch.bool).tril()
    mask = torch.where(allow, 0.0, torch.finfo(torch.float32).min)[None, None]
    positions = list(range(7)) + list(range(5)) + list(range(7, 13))  # sources share [0, L), target [L, ...)
    try:
        keys, values = _hf_kv(model, d2 + d3 + d1, positions, mask)
    except Exception as err:  # noqa: BLE001 - custom 4D masks are not accepted by every transformers release
        pytest.skip(f"this transformers version rejects a custom 4D mask: {err}")
    enc = encode_graph(dec, [], [d2, d3, d1], GraphPlan.from_edges(3, [(0, 2), (1, 2)]))
    assert enc.segment.positions.tolist() == positions
    for l in range(2):
        assert (enc.segment.keys[l] - keys[l]).abs().max() < 1e-5
        assert (enc.segment.values[l] - values[l]).abs().max() < 1e-5


def _query_logits(dec, prefix, chunks, plan, query):
    enc = encode_graph(dec, prefix, chunks, plan)
    with torch.no_grad():
        logits, _, _ = dec.forward(torch.as_tensor([query]), enc.query_start + torch.arange(len(query)).unsqueeze(0), enc.segment)
    return logits


def test_free_from_positional_bias_in_source_order():
    """Shared PEs and no attention between sources: any order of the sources gives the same answer."""
    dec = RopeDecoder(tiny_model("llama"))
    prefix, query = _tokens(4, 10), _tokens(5, 11)
    sources = [_tokens(n, 20 + n) for n in (5, 8, 3)]
    target = _tokens(6, 30)
    ref = None
    for perm in itertools.permutations(range(3)):
        chunks = [sources[i] for i in perm] + [target]
        logits = _query_logits(dec, prefix, chunks, GraphPlan.star(3), query)
        ref = logits if ref is None else ref
        assert (logits - ref).abs().max() < 1e-5


def test_sequential_encoding_is_order_sensitive():
    """Control for the test above: the same permutation changes a sequentially encoded prompt."""
    dec = RopeDecoder(tiny_model("llama"))
    a, b, q = _tokens(5, 1), _tokens(8, 2), _tokens(4, 3)
    out = []
    for seq in (a + b + q, b + a + q):
        with torch.no_grad():
            logits, _, _ = dec.forward(torch.as_tensor([seq]), torch.arange(len(seq)).unsqueeze(0))
        out.append(logits)
    assert (out[0] - out[1]).abs().max() > 1e-3


def test_shared_position_ranges_and_context_window():
    """Sources share [P, P+L), targets [P+L, P+2L); the span does not grow with the number of chunks."""
    dec = RopeDecoder(tiny_model("llama"))
    prefix = _tokens(4, 0)
    for k in (2, 5, 9):
        refs = [_tokens(3 + i % 4, i) for i in range(k)]
        center = _tokens(7, 99)
        enc = encode_graph(dec, prefix, refs + [center], GraphPlan.star(k))
        L = max(len(r) for r in refs)
        pos = enc.segment.positions.tolist()
        assert pos[:4] == [0, 1, 2, 3]
        cursor = 4
        for r in refs:
            assert pos[cursor : cursor + len(r)] == list(range(4, 4 + len(r)))
            cursor += len(r)
        assert pos[cursor:] == list(range(4 + L, 4 + L + 7))
        assert enc.query_start == 4 + L + 7  # independent of k
        assert enc.num_context_tokens == 4 + sum(map(len, refs)) + 7


def test_parallel_plan_is_independent_shared_range_encoding():
    dec = RopeDecoder(tiny_model("mistral"))
    chunks = [_tokens(n, n) for n in (4, 7, 2)]
    enc = encode_graph(dec, [], chunks, GraphPlan.parallel(3))
    cursor = 0
    for c in chunks:
        with torch.no_grad():
            _, alone = dec.encode(torch.as_tensor([c]), torch.arange(len(c)).unsqueeze(0))
        for l in range(2):
            assert (enc.segment.keys[l][:, :, cursor : cursor + len(c)] - alone.keys[l]).abs().max() < 1e-5
        cursor += len(c)
    assert enc.query_start == 7


def test_rag_plans_follow_the_paper():
    top3 = GraphPlan.rag_top_m(10, 3)
    assert set(top3.rounds[0]) == set(range(7))  # the remaining chunks are targets
    assert all(src == [(7, 0), (8, 0), (9, 0)] for src in top3.rounds[0].values())
    assert sorted(top3.query_view) == sorted([(c, 0) for c in (7, 8, 9)] + [(c, 1) for c in range(7)])
    official = GraphPlan.rag_top_m(10, 3, "official")
    assert set(official.rounds[0]) == set(range(10)) and len(official.query_view) == 20
    full = GraphPlan.rag_full(4)
    assert full.num_edges() == 16 and len(full.query_view) == 8
    assert GraphPlan.rag_full(4, self_loops=False).num_edges() == 12
    reordered = GraphPlan.rag_top_m(4, 1, order=[2, 0, 3, 1])  # chunk 1 is the most relevant
    assert all(src == [(1, 0)] for src in reordered.rounds[0].values())


def test_global_layout_uses_the_longest_chunk_overall():
    """@global: level t starts at P + t * L with L the longest chunk (literal Sec. 3.1); default: after the longest source."""
    dec = RopeDecoder(tiny_model("llama"))
    chunks = [_tokens(9, 1), _tokens(4, 2)]  # chunk 1 is the top-1 source (4 tokens), chunk 0 the longer target
    prompt = Prompt([1, 2], chunks, [3])
    default = prepare(dec, prompt, parse_method("graphkv-top1"))
    literal = prepare(dec, prompt, parse_method("graphkv-top1@global"))
    assert default.segment.positions.tolist() == [0, 1] + list(range(2, 6)) + list(range(6, 15))
    assert literal.segment.positions.tolist() == [0, 1] + list(range(2, 6)) + list(range(11, 20))
    assert (default.query_start, literal.query_start) == (15, 20)


def test_multi_hop_chain_uses_one_position_range_per_hop():
    """Appendix A.1.2: Top1 -> Top2-4 -> rest, with ranges [P, P+L), [P+L, P+2L), [P+2L, P+3L)."""
    dec = RopeDecoder(tiny_model("llama"))
    chunks = [_tokens(3 + i, i) for i in range(6)]  # ascending relevance: chunk 5 is the top chunk
    method = parse_method("graphkv-chain1-3")
    plan = method.plan(Prompt([], chunks, []))
    assert plan.rounds[0] == {4: [(5, 0)], 3: [(5, 0)], 2: [(5, 0)]}
    assert set(plan.rounds[1]) == {0, 1} and plan.rounds[1][0] == [(4, 1), (3, 1), (2, 1)]
    enc = encode_graph(dec, [], chunks, plan)
    assert enc.query_start == len(chunks[5]) + max(len(chunks[c]) for c in (2, 3, 4)) + max(len(chunks[c]) for c in (0, 1))
    assert enc.num_context_tokens == sum(map(len, chunks))


def test_attention_cost_is_v_l2_plus_e_l2():
    """Sec. 3.1: building the context costs O(|V| L^2 + |E| L^2) query-key pairs."""
    dec = RopeDecoder(tiny_model("llama"))
    chunks = [_tokens(5, i) for i in range(6)]
    plan = GraphPlan.from_edges(6, [(0, 3), (1, 3), (2, 4), (0, 5)])
    enc = encode_graph(dec, [], chunks, plan)
    level0 = {c for c, lv in plan.instances() if lv == 0}
    expected = len(level0) * 15 + 3 * 15  # causal pairs of every encoded chunk (3 targets)
    expected += 5 * (2 * 5) + 5 * 5 + 5 * 5  # each target token sees all tokens of its sources
    assert enc.attention_pairs == expected


def test_sequential_layout_reproduces_official_rag_positions():
    """pcw.py (gapemp): a target sees its sources at consecutive positions from 0, the query sees everything in a row."""
    model = tiny_model("llama")
    dec = RopeDecoder(model)
    chunks = [_tokens(4, 1), _tokens(6, 2), _tokens(5, 3)]
    prefix = _tokens(3, 4)
    enc = encode_graph(dec, prefix, chunks, GraphPlan.rag_full(3), layout="sequential")
    assert enc.segment.positions.tolist() == list(range(3 + 2 * 15))
    assert enc.query_start == 33
    # target copy of chunk 1 == chunk 1 read after the three independently encoded chunks laid out in a row
    keys, values = _hf_kv(model, chunks[0] + chunks[1] + chunks[2] + chunks[1], list(range(21)), _block_mask([4, 6, 5], 6))
    target = enc.segment.slice(3 + 15 + 4, 3 + 15 + 10)
    assert target.positions.tolist() == list(range(22, 28))
    moved = dec.move(target, torch.arange(15, 21))
    for l in range(2):
        assert (moved.keys[l] - keys[l][:, :, 15:]).abs().max() < 1e-5
        assert (moved.values[l] - values[l][:, :, 15:]).abs().max() < 1e-5


def _block_mask(source_lengths, target_len):
    n = sum(source_lengths) + target_len
    allow = torch.zeros(n, n, dtype=torch.bool)
    cursor = 0
    for s in source_lengths:
        allow[cursor : cursor + s, cursor : cursor + s] = torch.ones(s, s, dtype=torch.bool).tril()
        cursor += s
    allow[cursor:, :cursor] = True
    allow[cursor:, cursor:] = torch.ones(target_len, target_len, dtype=torch.bool).tril()
    return torch.where(allow, 0.0, torch.finfo(torch.float32).min)[None, None]


def test_one_source_graph_generates_like_sequential_reading():
    """End to end: with no prefix, a single source -> target edge is sequential reading."""
    model = tiny_model("llama")
    dec = RopeDecoder(model)
    src, tgt, q = _tokens(8, 1), _tokens(5, 2), _tokens(4, 3)
    prompt = Prompt([], [src, tgt], q, src + tgt + q)
    star = generate(dec, prompt, parse_method("graphkv-star"), max_new_tokens=10)
    seq = generate(dec, prompt, Method("sequential"), max_new_tokens=10)
    assert star.token_ids == seq.token_ids
    assert star.context_tokens == 13


def test_methods_place_the_question_as_their_papers_do():
    dec = RopeDecoder(tiny_model("llama"))
    prefix, query = [1] + _tokens(4, 0), _tokens(3, 1)
    chunks = [_tokens(n, n) for n in (6, 9, 4)]
    prompt = Prompt(prefix, chunks, query)
    assert prepare(dec, prompt, Method("sequential")).query_start == 0
    assert prepare(dec, prompt, Method("block")).query_start == 5 + 19
    pcw = prepare(dec, prompt, Method("pcw", bos_id=1))
    assert pcw.query_start == 10  # longest window: BOS + 9 tokens
    assert len(pcw.segment) == 5 + 19  # one BOS kept (the prefix already starts with it)
    ape = prepare(dec, prompt, Method("ape"))
    assert ape.query_start == 5 + 9 and int(ape.ape.context_mask.sum()) == 19
    gkv = prepare(dec, prompt, parse_method("graphkv-top1"))
    assert gkv.query_start == 5 + 4 + 9  # source = top-1 (last chunk, 4 tokens); targets start after it
    tokens, _, _ = decode(dec, gkv, max_new_tokens=3)
    assert len(tokens) <= 3
