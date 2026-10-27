"""Graph-structure experiments: chunkers, relevance scorers, query views, answer NLL, runner."""

import json

import pytest
import torch

from graphkv_neurips.attention import cross_boundary_attention, query_attention_mass
from graphkv_neurips.chunking import make_spans, parse_chunker, sentence_spans
from graphkv_neurips.graph import GraphPlan, encode_graph
from graphkv_neurips.methods import Method, Prompt, answer_nll, parse_method, prepare
from graphkv_neurips.model import RopeDecoder, Segment
from graphkv_neurips.selection import ascending_order, bm25, retriever_scores, score_chunks
from graphkv_neurips.testing import tiny_model

TEXT = (
    "- Title: Alpha\nAlpha is a small town. It has a river and a mill.\n"
    "- Title: Beta\nBeta was founded in 1901. The mayor of Beta is Ann Lee! Many people visit Beta.\n"
    "- Title: Gamma\nGamma makes cheese. Its cheese is sold far away? Yes, very far.\n"
)
WORDS = sorted(set(TEXT.replace("\n", " ").replace(".", " . ").replace("!", " ! ").replace("?", " ? ").replace(",", " , ").replace(":", " : ").replace("-", " - ").split()))


def word_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    extra = "Who is the mayor ? Ann Lee answer question please write a high-quality for given using only provided search documents some of which might be irrelevant Question <|user|> <|assistant|> You are an intelligent AI assistant".split()
    vocab = {w: i for i, w in enumerate(["[UNK]", "[EOS]"] + sorted(set(WORDS + extra)))}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", eos_token="[EOS]")


@pytest.fixture(scope="module")
def tok():
    return word_tokenizer()


@pytest.fixture(scope="module")
def dec():
    return RopeDecoder(tiny_model("llama", vocab_size=128))


# ------------------------------------------------------------------ chunkers


@pytest.mark.parametrize("spec", ["fixed:8", "sentences", "sentences:12", "paragraphs", "paragraphs:20", "attention:10", "attention-sent:12"])
def test_chunkers_partition_the_text(spec, tok, dec):
    spans = make_spans(TEXT, parse_chunker(spec), tok, decoder=dec)
    assert "".join(TEXT[a:b] for a, b in spans) == TEXT
    assert len(spans) >= 2


def test_fixed_windows_have_the_requested_size(tok):
    spans = make_spans(TEXT, parse_chunker("fixed:8"), tok)
    sizes = [len(tok(TEXT[a:b], add_special_tokens=False).input_ids) for a, b in spans]
    assert all(s == 8 for s in sizes[:-1]) and 0 < sizes[-1] <= 8


def test_sentences_and_paragraphs(tok):
    assert len(sentence_spans(TEXT)) == 11  # 3 title lines + 8 sentences
    paras = make_spans(TEXT, parse_chunker("paragraphs"), tok)
    assert len(paras) == 3  # each title line merges with the paragraph after it
    assert TEXT[paras[1][0] : paras[1][1]].startswith("- Title: Beta")


def test_documents_chunker_uses_given_spans(tok):
    docs = [(0, 60), (60, len(TEXT))]
    assert make_spans(TEXT, parse_chunker("documents"), tok, doc_spans=docs) == docs
    with pytest.raises(ValueError):
        parse_chunker("fixed")


def test_cross_boundary_attention_matches_huggingface_attentions(dec):
    model = dec.model
    ids = torch.randint(3, 128, (40,), generator=torch.Generator().manual_seed(0)).tolist()
    cand, window, sink, layers = [10, 20, 31], 5, 2, [0, 1]
    ours = cross_boundary_attention(dec, ids, cand, layers=layers, window=window, sink=sink)
    with torch.no_grad():
        att = model(torch.as_tensor([ids]), output_attentions=True).attentions
    for j, b in enumerate(cand):
        rows = range(b, min(b + window, len(ids)))
        ref = sum(float(att[l][0, :, t, sink:b].sum(-1).mean()) for l in layers for t in rows) / (len(rows) * len(layers))
        assert abs(float(ours[j]) - ref) < 1e-4


def test_query_attention_mass_is_a_partial_distribution(dec):
    _, ctx = dec.encode(torch.randint(3, 128, (1, 30)), torch.arange(30).unsqueeze(0))
    mass = query_attention_mass(dec, ctx, [5, 6, 7], 30, layers=[0, 1])
    assert mass.shape == (30,) and float(mass.min()) >= 0 and 0 < float(mass.sum()) < 1


# ------------------------------------------------------------------ scorers


def test_scorers():
    chunks = ["the river and the mill", "mayor of beta is ann lee", "cheese far away"]
    s = bm25("who is the mayor of beta", chunks)
    assert max(range(3), key=lambda i: s[i]) == 1
    assert retriever_scores([(0, 5), (5, 12)], [(0, 6), (6, 12)], [0.1, 0.9]) == [0.1, 0.9]
    assert ascending_order([0.3, 0.9, 0.1]) == [2, 0, 1]
    oracle = score_chunks("oracle", chunks, "who is the mayor", answers=["Ann Lee"])
    assert ascending_order(oracle)[-1] == 1
    assert score_chunks("random", chunks, "q", seed=1) == score_chunks("random", chunks, "q", seed=1)
    assert ascending_order(score_chunks("first", chunks, "q"))[-1] == 0


# ------------------------------------------------------------------ views (drop-source ablation)


def _chunks():
    return [torch.randint(3, 128, (n,), generator=torch.Generator().manual_seed(n)).tolist() for n in (5, 9, 4, 7)]


def test_dropping_sources_keeps_target_positions(dec):
    chunks, prefix = _chunks(), [3, 4, 5]
    full = encode_graph(dec, prefix, chunks, GraphPlan.top_m(4, 2))
    drop = encode_graph(dec, prefix, chunks, GraphPlan.top_m(4, 2, view="targets"))
    ctrl = encode_graph(dec, prefix, chunks, GraphPlan.top_m(4, 2, edges=False, view="targets"))
    src = encode_graph(dec, prefix, chunks, GraphPlan.top_m(4, 2, view="sources"))
    # sources = chunks 2, 3 (top-2), targets = chunks 0, 1
    assert len(full.segment) == 3 + 25 and len(drop.segment) == 3 + 14 and len(src.segment) == 3 + 11
    assert drop.segment.positions.tolist() == full.segment.positions.tolist()[:3] + full.segment.positions.tolist()[3 + 11 :]
    assert ctrl.segment.positions.tolist() == drop.segment.positions.tolist()
    assert drop.query_start == ctrl.query_start == full.query_start
    # same visible tokens and positions; only the edges differ
    assert (drop.segment.keys[1] - ctrl.segment.keys[1]).abs().max() > 1e-3
    # without edges a target is just its independent encoding at the target positions
    _, alone = dec.encode(torch.as_tensor([chunks[0]]), (3 + 7 + torch.arange(5)).unsqueeze(0))
    assert (ctrl.segment.keys[1][:, :, 3:8] - alone.keys[1]).abs().max() < 1e-5


def test_view_specs_parse():
    p = Prompt([1, 2], _chunks(), [3])
    assert parse_method("graphkv-top2:targets").plan(p).query_view == [(0, 1), (1, 1)]
    assert parse_method("noedge-top2:targets").plan(p).rounds == [{0: [], 1: []}]
    assert parse_method("graphkv-full:targets").plan(p).query_view == [(c, 1) for c in range(4)]
    assert parse_method("parallel").plan(p).query_view == [(c, 0) for c in range(4)]
    with pytest.raises(ValueError):
        parse_method("block:targets")


def test_level0_cache_is_reused(dec):
    chunks, prefix = _chunks(), [3, 4]
    cache = {i: s for i, s in enumerate(dec.encode_batch(chunks, [2] * 4))}
    a = encode_graph(dec, prefix, chunks, GraphPlan.top_m(4, 1), level0_cache=cache)
    b = encode_graph(dec, prefix, chunks, GraphPlan.top_m(4, 1))
    assert (a.segment.keys[0] - b.segment.keys[0]).abs().max() < 1e-5


# ------------------------------------------------------------------ answer NLL


def test_answer_nll_matches_huggingface_loss(dec):
    model = dec.model
    ids = torch.randint(3, 128, (20,), generator=torch.Generator().manual_seed(2)).tolist()
    answer = [9, 10, 11]
    prepared = prepare(dec, Prompt([], [], [], ids), Method("sequential"))
    ours = answer_nll(dec, prepared, answer)
    ours_chunked = answer_nll(dec, prepared, answer, chunk_size=7)
    full = torch.as_tensor([ids + answer])
    labels = torch.full_like(full, -100)
    labels[0, 20:] = full[0, 20:]
    with torch.no_grad():
        ref = float(model(full, labels=labels).loss)
    assert abs(ours - ref) < 1e-4 and abs(ours_chunked - ref) < 1e-4


# ------------------------------------------------------------------ runner


def test_run_structure_end_to_end_and_resume(tmp_path, tok):
    from graphkv_neurips import run_structure

    model = tiny_model("llama", vocab_size=len(tok), eos_token_id=1, bos_token_id=None, pad_token_id=None)
    path = tmp_path / "tiny"
    tok.save_pretrained(path)
    model.save_pretrained(path)
    data = tmp_path / "data.jsonl"
    with open(data, "w") as f:
        for i in range(3):
            f.write(json.dumps({"question": "Who is the mayor of Beta ?", "answers": ["Ann Lee"], "context": TEXT}) + "\n")
    out = tmp_path / "out"
    argv = ["--model", str(path), "--data-file", str(data), "--chunker", "sentences:12", "--scorer", "bm25",
            "--conditions", "sequential,graphkv-top2,graphkv-top2:targets,noedge-top2:targets", "--metric", "both",
            "--max-new-tokens", "3", "--dtype", "float32", "--device", "cpu", "--prompt-style", "tulu", "--output-dir", str(out)]
    run_structure.main(argv)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["n_common"] == 3 and set(summary["conditions"]) == {"sequential", "graphkv-top2", "graphkv-top2:targets", "noedge-top2:targets"}
    assert any(c["a"] == "graphkv-top2:targets" and c["b"] == "noedge-top2:targets" for c in summary["comparisons"])
    rec = json.loads((out / "graphkv-top2.jsonl").read_text().splitlines()[0])
    assert rec["answer_location"] in ("source_only", "target_only", "both", "none") and "nll" in rec and "generated" in rec
    run_structure.main(argv + ["--scorer", "attention"])  # everything done: resumes without new work
    assert len((out / "sequential.jsonl").read_text().splitlines()) == 3
    out2 = tmp_path / "out2"
    run_structure.main([a if a != str(out) else str(out2) for a in argv[:-2]] + ["--output-dir", str(out2), "--scorer", "attention", "--metric", "nll"])
    assert json.loads((out2 / "summary.json").read_text())["n_common"] == 3
