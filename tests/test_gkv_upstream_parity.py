"""Parity with the official implementation (third_party/GraphKV, pinned commit).

The official functions are run unmodified on a tiny random Llama, except for
two environment shims that do not change what they compute:

* FlashAttention-2 is replaced by an attention function with the same
  semantics for unpadded inputs (causal mask aligned to the bottom right, the
  4D mask that transformers builds from ``cache_position`` is ignored), since
  ``flash-attn`` needs CUDA. The official code relies on exactly this alignment.
* ``apply_rotary_pos_emb`` keeps the key dtype instead of casting to bfloat16,
  so the comparison runs in float32.

A byte-level tokenizer stands in for the Tulu tokenizer. The test needs
transformers 4.50 (the official pin), e.g.::

    python -m venv .venv-upstream && .venv-upstream/bin/pip install torch transformers==4.50.0 pytest
    git submodule update --init third_party/GraphKV
    .venv-upstream/bin/python -m pytest tests/test_gkv_upstream_parity.py
"""

import importlib
import os
import sys
import types

import pytest
import torch
import torch.nn.functional as F

transformers = pytest.importorskip("transformers")
UPSTREAM = os.path.join(os.path.dirname(__file__), "..", "third_party", "GraphKV")
if not os.path.exists(os.path.join(UPSTREAM, "pcw_parallel.py")):
    pytest.skip("third_party/GraphKV is not checked out (git submodule update --init)", allow_module_level=True)
if not transformers.__version__.startswith("4.50"):
    pytest.skip(f"the official code is pinned to transformers 4.50 (found {transformers.__version__})", allow_module_level=True)

from graphkv_neurips.graph import GraphPlan  # noqa: E402
from graphkv_neurips.methods import Method, Prompt, decode, parse_method, prepare  # noqa: E402
from graphkv_neurips.model import RopeDecoder  # noqa: E402
from graphkv_neurips.testing import tiny_model  # noqa: E402

EOS = 1


class ByteTokenizer:
    """Minimal tokenizer with the calls the official scripts make; ``pad = eos`` as they set it."""

    eos_token_id = EOS
    pad_token_id = EOS
    padding_side = "right"

    def __init__(self):
        self.decoded = []

    def encode(self, text, add_special_tokens=False):
        return [b + 3 for b in text.encode("utf-8")]

    def __call__(self, text, return_tensors=None, truncation=False, max_length=None, padding=False, add_special_tokens=False):
        texts = [text] if isinstance(text, str) else list(text)
        rows = [self.encode(t) for t in texts]
        if truncation and max_length is not None:
            rows = [r[:max_length] for r in rows]
        width = max(len(r) for r in rows)
        if len(rows) > 1 and not padding:
            raise ValueError("unequal rows need padding")
        ids = torch.tensor([r + [self.pad_token_id] * (width - len(r)) for r in rows])
        return types.SimpleNamespace(input_ids=ids)

    def decode(self, token_ids, skip_special_tokens=False):
        ids = [int(t) for t in token_ids]
        self.decoded.append(ids)
        return bytes(i - 3 for i in ids if 3 <= i < 259).decode("utf-8", errors="replace")


def _flashlike(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    groups = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(groups, dim=1)
    value = value.repeat_interleave(groups, dim=1)
    q_len, k_len = query.shape[2], key.shape[2]
    mask = torch.ones(q_len, k_len, dtype=torch.bool).tril(diagonal=k_len - q_len)
    out = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=scaling)
    return out.transpose(1, 2).contiguous(), None


def _keep_dtype_rope(k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos, sin = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
    half = k.shape[-1] // 2
    rotated = torch.cat((-k[..., half:], k[..., :half]), dim=-1)
    return k * cos + rotated * sin


def _stub_server_deps():
    class _App:
        def __init__(self, *a, **k):
            pass

        def route(self, *a, **k):
            return lambda f: f

    sys.modules.setdefault("flask", types.SimpleNamespace(Flask=_App, request=None))
    sys.modules.setdefault("flask_cors", types.SimpleNamespace(CORS=lambda *a, **k: None))
    sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))


@pytest.fixture(scope="module")
def upstream():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS["flashlike"] = _flashlike
    _stub_server_deps()
    sys.path.insert(0, os.path.abspath(UPSTREAM))
    mods = {name: importlib.import_module(name) for name in ("pcw", "pcw_parallel", "server.block_generate_server")}
    for mod in mods.values():
        mod.apply_rotary_pos_emb = _keep_dtype_rope
    yield mods
    sys.path.remove(os.path.abspath(UPSTREAM))


@pytest.fixture(scope="module")
def setup():
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    model = tiny_model("llama", vocab_size=300, seed=3)
    model.config._attn_implementation = "flashlike"
    emb = LlamaRotaryEmbedding(config=model.config).eval()
    return model, emb, RopeDecoder(model)


TEXTS = dict(
    prefix="<|user|>\nRead the documents.\n\n",
    query="\n Question: who? \n<|assistant|>\n",
    chunks=["- Title: A\nalpha beta gamma.\n", "- Title: Bee\nbuzz.\n", "- Title: C\nsea and sky and sand.\n", "- Title: D\ndelta delta.\n"],
    center="Central paper text, longer than the others.",
    neighbors=["ref one text", "reference two, a bit longer", "three"],
)


def _ours(dec, tok, prefix, chunks, query, method, max_new):
    prompt = Prompt(tok.encode(prefix), [tok.encode(c) for c in chunks], tok.encode(query))
    out, _, _ = decode(dec, prepare(dec, prompt, method, pad_id=EOS), max_new_tokens=max_new, stop_ids=[EOS])
    return out if len(out) == max_new else out + [EOS]


def _compare(official, ours):
    """Greedy outputs must agree token for token over the whole generation."""
    assert len(official) > 0
    assert official == ours


def test_arxiv_graphkv_star(upstream, setup):
    model, emb, dec = setup
    tok = ByteTokenizer()
    upstream["pcw_parallel"].gapemp_graph(tok, model, emb, TEXTS["prefix"], TEXTS["center"], list(TEXTS["neighbors"]), TEXTS["query"], "Tulu", 1, 1, None)
    official = tok.decoded[-1]
    center = "\nNow you will read the center paper and answer a related question: \n" + TEXTS["center"]
    ours = _ours(dec, tok, TEXTS["prefix"], TEXTS["neighbors"] + [center], TEXTS["query"], parse_method("graphkv-star"), 256)
    _compare(official, ours)


def test_arxiv_graphkv_with_distractor(upstream, setup):
    model, emb, dec = setup
    tok = ByteTokenizer()
    centers = [TEXTS["center"], "Another central paper."]
    neigh = [list(TEXTS["neighbors"]), ["distractor ref", "second distractor reference text here"]]
    upstream["pcw_parallel"].gapemp_graph_batch(tok, model, emb, TEXTS["prefix"], centers, [list(n) for n in neigh], TEXTS["query"], "Tulu", 1, 1, None)
    official = tok.decoded[-1]
    middle = "\nNow you will read the center paper and answer a related question: : \n"
    chunks = neigh[0] + [middle + centers[0]] + neigh[1] + [middle + centers[1]]
    a, b = len(neigh[0]), len(neigh[1])
    edges = [(i, a) for i in range(a)] + [(a + 1 + i, a + 1 + b) for i in range(b)]
    method = Method("graphkv", plan=lambda p: GraphPlan.from_edges(len(p.chunks), edges), layout="paper-local")
    _compare(official, _ours(dec, tok, TEXTS["prefix"], chunks, TEXTS["query"], method, 256))


def test_arxiv_block_rag(upstream, setup):
    model, emb, dec = setup
    tok = ByteTokenizer()
    upstream["pcw_parallel"].block(tok, model, emb, TEXTS["prefix"], TEXTS["center"], list(TEXTS["neighbors"]), TEXTS["query"], "Tulu", 1, 1, None)
    official = tok.decoded[-1]
    middle = "\nNow you will read the center paper and answer a related question: \n"
    ours = _ours(dec, tok, TEXTS["prefix"], TEXTS["neighbors"] + [middle, TEXTS["center"]], TEXTS["query"], Method("block"), 256)
    _compare(official, ours)


@pytest.mark.parametrize("spec", ["graphkv-full@official", "graphkv-top3@official", "graphkv-top1@official"])
def test_rag_graphkv_official_variants(upstream, setup, spec):
    model, emb, dec = setup
    tok = ByteTokenizer()
    pcw = upstream["pcw"]
    if spec == "graphkv-full@official":
        pcw.gapemp(tok, model, emb, TEXTS["prefix"], "", TEXTS["query"], list(TEXTS["chunks"]), "Tulu", 1, 1, None)
    else:
        k = int(spec[len("graphkv-top")])
        pcw.gapemp_appr(tok, model, emb, TEXTS["prefix"], "", TEXTS["query"], list(TEXTS["chunks"]), "Tulu", 1, 1, k)
    official = tok.decoded[-1]
    ours = _ours(dec, tok, TEXTS["prefix"], TEXTS["chunks"], TEXTS["query"], parse_method(spec), 1024)
    _compare(official, ours)


def test_rag_block_rag(upstream, setup):
    from transformers import GenerationConfig

    model, emb, dec = setup
    tok = ByteTokenizer()
    cfg = GenerationConfig(do_sample=False, num_beams=1, eos_token_id=EOS, max_new_tokens=64, pad_token_id=EOS)
    upstream["server.block_generate_server"].block_generate(
        blocks=[TEXTS["prefix"]] + list(TEXTS["chunks"]), instruction=TEXTS["query"], generation_config=cfg,
        model=model, emb=emb, tokenizer=tok, num_local_attention_blocks=10000,
    )
    official = tok.decoded[-1]
    _compare(official, _ours(dec, tok, TEXTS["prefix"], TEXTS["chunks"], TEXTS["query"], Method("block"), 64))


def test_sequential(upstream, setup):
    model, emb, dec = setup
    tok = ByteTokenizer()
    text = TEXTS["prefix"] + "".join(TEXTS["chunks"]) + TEXTS["query"]
    upstream["pcw"].vanilla(tok, model, text)
    official = tok.decoded[-1]
    ids = tok.encode(text)
    out, _, _ = decode(dec, prepare(dec, Prompt([], [], [], ids), Method("sequential")), max_new_tokens=512, stop_ids=[EOS])
    _compare(official, out if len(out) == 512 else out + [EOS])
