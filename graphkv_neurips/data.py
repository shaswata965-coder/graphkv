"""Loaders for the official processed data (Hugging Face dataset ``Graph-COM/GraphKV``).

``datahub/rag/*`` holds the RAG benchmarks with 10 retrieved chunks per
question (processing of Block-Attention / MultiHop-RAG / MorehopQA), and
``datahub/arxiv/*`` the ArXiv-QA ego-graphs. Files are downloaded on first use
into the Hugging Face cache, or read from ``data_dir`` if given.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

HF_REPO = "Graph-COM/GraphKV"

RAG_FILES = {
    "2wiki": "datahub/rag/2wiki_dev/dataset",
    "hqa": "datahub/rag/hqa_eval/dataset",
    "tqa": "datahub/rag/tqa_eval/dataset",
    "nqa": "datahub/rag/nq_eval/dataset",
    "multihopqa": "datahub/rag/multihopqa/GIST-large-Embedding-v0_retrieval_test.json",
    "morehopqa": "datahub/rag/morehopqa/with_human_verification_ascend.jsonl",
}

# question types behind the per-column numbers of Tables 2 and 3
TYPE_SOURCES = {
    "2wiki": ("xanhho/2WikiMultihopQA", "dev.parquet", "type"),
    "hqa": ("hotpotqa/hotpot_qa", "distractor/validation-00000-of-00001.parquet", "type"),
}


@dataclass
class RAGExample:
    index: int
    question: str
    answers: object  # str, list of str, or [[str]] (NarrativeQA), scored by metrics.score_example
    documents: list
    group: str = "all"
    meta: Dict = field(default_factory=dict)


def fetch(path: str, data_dir: Optional[str] = None, repo: str = HF_REPO) -> str:
    if data_dir is not None:
        local = os.path.join(data_dir, path)
        if os.path.exists(local):
            return local
        local = os.path.join(data_dir, path.split("datahub/", 1)[-1])
        if os.path.exists(local):
            return local
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, path, repo_type="dataset")


def question_types(task: str) -> Optional[List[str]]:
    if task not in TYPE_SOURCES:
        return None
    import pandas as pd
    from huggingface_hub import hf_hub_download

    repo, filename, column = TYPE_SOURCES[task]
    return pd.read_parquet(hf_hub_download(repo, filename, repo_type="dataset"))[column].tolist()


def load_rag(task: str, data_dir: Optional[str] = None, with_types: bool = True) -> List[RAGExample]:
    if task not in RAG_FILES:
        raise ValueError(f"unknown RAG task '{task}' (choose from {', '.join(RAG_FILES)})")
    path = fetch(RAG_FILES[task], data_dir)
    if task == "multihopqa":
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        return [
            RAGExample(i, r["query"], r["answer"], r["retrieval_list"], r["question_type"])
            for i, r in enumerate(raw)
        ]
    with open(path, encoding="utf-8") as f:
        raw = [json.loads(line) for line in f if line.strip()]
    if task == "morehopqa":
        # inference.py sorts MorehopQA by the number of hops (descending) before running;
        # the index here is the position in the file.
        return [RAGExample(i, r["question"], r["answers"], r["documents"], f"hop{r['no_of_hops']}") for i, r in enumerate(raw)]
    types = question_types(task) if with_types else None
    if types is not None and len(types) != len(raw):
        types = None
    out = []
    for i, r in enumerate(raw):
        out.append(RAGExample(i, r["question"], r["answers"], r["documents"], types[i] if types else "all"))
    return out


@dataclass
class ArxivExample:
    set_id: int
    center: str
    references: List[str]
    question: str
    answer: str


def load_arxiv(data_dir: Optional[str] = None, set_ids=range(1, 61)) -> List[ArxivExample]:
    """The 60 ArXiv-QA questions (``arxiv_inference.py: load_data``)."""
    out = []
    for set_id in set_ids:
        with open(fetch(f"datahub/arxiv/text_data/{set_id}.json", data_dir), encoding="utf-8") as f:
            data = json.load(f)
        value = list(data.values())[0]
        refs = [v for v in value["1"].values() if v is not None]
        with open(fetch(f"datahub/arxiv/questions/{set_id}.txt", data_dir), encoding="utf-8") as f:
            question = f.read()
        with open(fetch(f"datahub/arxiv/answers/{set_id}.txt", data_dir), encoding="utf-8") as f:
            answer = f.read()
        out.append(ArxivExample(set_id, value["0"], refs, question, answer))
    return out
