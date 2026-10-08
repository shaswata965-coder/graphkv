"""Offline end-to-end test of the experiment CLI and plotting."""

import csv
import json

import pytest
import torch

from graphkv import plot, run
from graphkv.engine import sequence_nll
from graphkv.models import make_adapter
from graphkv.selftest import tiny_model

WORDS = "the cat sat on a mat and dog ran to park while bird sang in tree".split()


def make_local_model(tmp_path):
    """Save a tiny Llama + word-level tokenizer, like a local checkpoint directory."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {w: i for i, w in enumerate(["[UNK]", "[EOS]"] + WORDS)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", eos_token="[EOS]")
    model = tiny_model("llama", eos_token_id=1, bos_token_id=None, pad_token_id=None)
    path = tmp_path / "tiny-llama"
    fast.save_pretrained(path)
    model.save_pretrained(path)
    return path


def test_sequence_nll_matches_huggingface():
    model = tiny_model("gpt2")
    ids = list(range(3, 23))
    nll = sequence_nll(make_adapter(model), ids)
    with torch.inference_mode():
        loss = model(input_ids=torch.tensor([ids]), labels=torch.tensor([ids])).loss
    assert sum(nll) / len(nll) == pytest.approx(float(loss), rel=1e-5)


def test_prefix_match():
    assert run.prefix_match([1, 2, 3, 4], [1, 2, 9, 4]) == 0.5
    assert run.prefix_match([1, 2], None) is None


def test_cli_sweep_resume_and_plot(tmp_path):
    model_dir = make_local_model(tmp_path)
    g = torch.Generator().manual_seed(0)
    prompts = tmp_path / "prompts.txt"
    prompts.write_text(
        "\n".join(" ".join(WORDS[i] for i in torch.randint(0, len(WORDS), (40,), generator=g).tolist()) for _ in range(3))
    )
    out = tmp_path / "out"
    argv = [
        "--model", str(model_dir), "--device", "cpu", "--prompts-file", str(prompts),
        "--num-prompts", "3", "--min-chars", "0", "--min-context", "8",
        "--epsilons", "0.5,100", "--intervals", "4,8", "--max-new-tokens", "12",
        "--ignore-eos", "--warmup", "0", "--output-dir", str(out),
    ]  # fmt: skip
    assert run.main(argv) == 0

    rows = list(csv.DictReader((out / "summary.csv").open()))
    assert [r["config_id"] for r in rows] == [
        "baseline",
        "graphkv-e0.5-M4-s4-Tconv-head",
        "graphkv-e100-M4-s4-Tconv-head",
        "graphkv-e0.5-M8-s4-Tconv-head",
        "graphkv-e100-M8-s4-Tconv-head",
    ]
    base = rows[0]
    assert float(base["prefix_match"]) == 1.0 and float(base["max_compression_pct"]) == 0.0
    for key in ("tf_ppl", "seq_ppl", "oracle_ppl", "gen_ppl", "decode_tok_s", "latency_s"):
        assert float(base[key]) > 0
    assert float(rows[2]["max_compression_pct"]) > 50  # huge epsilon merges almost everything

    per_prompt = [json.loads(line) for line in (out / "per_prompt.jsonl").read_text().splitlines()]
    assert len(per_prompt) == 5 * 3 * 2  # configs x prompts x protocols

    # Resuming skips finished configurations and appends nothing.
    assert run.main(argv) == 0
    assert len(list(csv.DictReader((out / "summary.csv").open()))) == 5
    assert len((out / "per_prompt.jsonl").read_text().splitlines()) == 30

    assert plot.main([str(out), "--table-interval", "4", "--table-epsilons", "0.5,100"]) == 0
    assert (out / "fig2_metrics.png").stat().st_size > 0
    assert "Max Compression" in (out / "table1.md").read_text()
