"""Relevance scores that decide which chunks become the top-k *sources*.

==============  ====================================================================
``retriever``   the retriever score of the document a chunk comes from (max over
                overlapping documents); only for data that ships with scores
``bm25``        lexical BM25 between the question and each chunk (no extra model)
``attention``   the LLM itself: attention mass the question puts on each chunk when
                it reads all chunks encoded independently (needs the model; the
                chunk encodings are re-used by Graph-KV, so this costs one
                question pass)
``first`` /     position controls: earliest / latest chunks are "most relevant"
``last``
``random``      seeded random scores (control: does relevance matter at all?)
``oracle``      chunks containing a gold answer first (diagnostic upper bound)
==============  ====================================================================

Higher score = more relevant. :func:`ascending_order` turns scores into the
``Prompt.order`` used by the top-m plans.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from typing import List, Optional, Sequence, Tuple

from .metrics import normalize_answer

SCORERS = ("retriever", "bm25", "attention", "first", "last", "random", "oracle")

_WORD = re.compile(r"[a-z0-9]+")


def _words(text: str) -> List[str]:
    return _WORD.findall(text.lower())


def bm25(question: str, chunks: Sequence[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
    docs = [_words(c) for c in chunks]
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / max(1, n)
    df = Counter(w for d in docs for w in set(d))
    out = []
    for d in docs:
        tf = Counter(d)
        s = 0.0
        for w in set(_words(question)):
            if w not in tf:
                continue
            idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
            s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / max(1e-9, avgdl)))
        out.append(s)
    return out


def retriever_scores(chunk_spans: Sequence[Tuple[int, int]], doc_spans: Sequence[Tuple[int, int]], doc_scores: Sequence[float]) -> List[float]:
    out = []
    for a, b in chunk_spans:
        overlapping = [s for (da, db), s in zip(doc_spans, doc_scores) if min(b, db) > max(a, da)]
        out.append(max(overlapping) if overlapping else float("-inf"))
    return out


def contains_answer(chunk: str, answers: Sequence[str]) -> bool:
    text = normalize_answer(chunk)
    return any(normalize_answer(a) and normalize_answer(a) in text for a in answers)


def score_chunks(
    kind: str,
    chunks: Sequence[str],
    question: str,
    *,
    chunk_spans: Optional[Sequence[Tuple[int, int]]] = None,
    doc_spans: Optional[Sequence[Tuple[int, int]]] = None,
    doc_scores: Optional[Sequence[float]] = None,
    answers: Sequence[str] = (),
    seed: int = 0,
) -> List[float]:
    """Scores for every scorer except ``attention`` (which needs the model, see run_structure)."""
    n = len(chunks)
    if kind == "retriever":
        if doc_scores is None or doc_spans is None or chunk_spans is None:
            raise ValueError("the retriever scorer needs document spans and scores")
        return retriever_scores(chunk_spans, doc_spans, doc_scores)
    if kind == "bm25":
        return bm25(question, chunks)
    if kind == "first":
        return [float(n - i) for i in range(n)]
    if kind == "last":
        return [float(i) for i in range(n)]
    if kind == "random":
        rng = random.Random(seed)
        return [rng.random() for _ in range(n)]
    if kind == "oracle":
        tie = bm25(question, chunks)
        top = max(tie) if tie else 0.0
        return [float(contains_answer(c, answers)) + 1e-3 * t / (top + 1e-9) for c, t in zip(chunks, tie)]
    raise ValueError(f"unknown scorer '{kind}' (choose from {', '.join(SCORERS)})")


def ascending_order(scores: Sequence[float]) -> List[int]:
    """Chunk indices from least to most relevant (ties: earlier chunk counts as less relevant)."""
    return sorted(range(len(scores)), key=lambda i: (scores[i], i))
