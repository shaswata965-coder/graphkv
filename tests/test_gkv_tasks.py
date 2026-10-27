"""ArXiv-QA and paper-classification task builders."""

import json
import random
import types

import torch

from graphkv_neurips.prompts import ARXIV_CENTER_INTRO, ARXIV_CENTER_INTRO_BATCH, ARXIV_SEQ_INTRO, CLASSIFY_CENTER_INTRO, PromptBuilder
from graphkv_neurips.run_arxiv import build, distractor_plan
from graphkv_neurips.run_classify import predict_class
from graphkv_neurips.tag_data import _SAFE_PICKLE, _store

CLASSES = ["Case Based", "Genetic Algorithms", "Neural Networks", "Theory"]


class CharTok:
    name_or_path = "tulu"
    chat_template = None

    def __call__(self, text, add_special_tokens=False):
        return types.SimpleNamespace(input_ids=[ord(c) % 250 + 3 for c in text])


def test_distractors_follow_the_official_random_stream():
    random.seed(42)  # arxiv_inference.py: seed_everything(seed), then one random.sample per question
    expected = [random.sample(list(range(1, 61)), 2) for _ in range(60)]
    assert distractor_plan(3, 42) == expected
    assert distractor_plan(1, 42) == [[]] * 60


def test_arxiv_prompts_follow_the_official_construction():
    b = PromptBuilder(CharTok(), "tulu")
    groups = [("CENTER A", ["ref a1", "ref a2"]), ("CENTER B", ["ref b1"])]
    prompt, m = build(b, "sequential", groups, "Q?")
    prefix, query = b.wrap(b_instruction(), "\n Question: Q? \n")
    text = prefix + "\n\n".join(["ref a1", "ref a2"]) + ARXIV_SEQ_INTRO + "CENTER A" + "ref b1" + ARXIV_SEQ_INTRO + "CENTER B" + query
    assert prompt.joined == b.encode(text) and m.name == "sequential"
    prompt, m = build(b, "graphkv", groups, "Q?")
    assert [bytes_of(c) for c in prompt.chunks] == ["ref a1", "ref a2", ARXIV_CENTER_INTRO_BATCH + "CENTER A", "ref b1", ARXIV_CENTER_INTRO_BATCH + "CENTER B"]
    plan = m.plan(prompt)
    assert plan.rounds == [{2: [(0, 0), (1, 0)], 4: [(3, 0)]}] and m.layout == "paper-local"
    prompt, m = build(b, "graphkv", groups[:1], "Q?")
    assert bytes_of(prompt.chunks[-1]) == ARXIV_CENTER_INTRO + "CENTER A"
    prompt, m = build(b, "block", groups[:1], "Q?")
    assert [bytes_of(c) for c in prompt.chunks] == ["ref a1", "ref a2", ARXIV_CENTER_INTRO, "CENTER A"]
    prompt, m = build(b, "block", groups, "Q?")
    assert [bytes_of(c) for c in prompt.chunks] == ["ref a1", "ref a2", "CENTER A", "ref b1", "CENTER B"]


def b_instruction():
    from graphkv_neurips.prompts import ARXIV_INSTRUCTION

    return ARXIV_INSTRUCTION


def bytes_of(ids):
    return "".join(chr(i - 3) for i in ids)  # CharTok is invertible for ASCII


def test_classification_prompt_and_answer_parsing():
    b = PromptBuilder(CharTok(), "tulu")
    tp = b.classify_text("Title: center", ["Title: r1", "Title: r2"], CLASSES)
    assert tp.chunks == ["Title: r1", "Title: r2", CLASSIFY_CENTER_INTRO + "Title: center"]
    assert "Classify the central paper into one of the following categories: Case Based, Genetic Algorithms" in tp.query
    assert tp.text() == tp.prefix + "Title: r1\n\nTitle: r2" + CLASSIFY_CENTER_INTRO + "Title: center" + tp.query
    assert b.classify_text("Title: center", [], CLASSES).chunks == ["Title: center"]  # Query-Only
    assert predict_class("Hmm, theory or neural nets? Answer: Neural Networks.", CLASSES) == 2
    assert predict_class("It is about Genetic Algorithms and Theory", CLASSES) == 1
    assert predict_class("no idea", CLASSES) is None


def test_allow_listed_unpickler_reads_pyg_layout_and_refuses_code(tmp_path):
    import pickle

    import pytest

    class Fake:  # pickled under a torch_geometric-like module name below
        pass

    store = types.SimpleNamespace()
    payload = {"_store": None}
    # build the same object graph the PyG file has: Data._store (GlobalStorage) ._mapping
    from graphkv_neurips.tag_data import _Inert

    storage = _Inert()
    storage.__setstate__({"_mapping": {"y": torch.tensor([1, 2]), "edge_index": torch.zeros(2, 0, dtype=torch.long)}})
    data = _Inert()
    data.__setstate__({"_store": storage})
    assert set(_store(data)) == {"y", "edge_index"}
    evil = tmp_path / "evil.pkl"
    with open(evil, "wb") as f:
        pickle.dump(eval, f)  # a callable from builtins must be refused
    with open(evil, "rb") as f, pytest.raises(pickle.UnpicklingError):
        _SAFE_PICKLE.Unpickler(f).load()
    del Fake, store, payload
