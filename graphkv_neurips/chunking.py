"""Turn one context text into graph nodes (chunks).

Every chunker returns character spans that partition the text exactly, so
all chunkings of an example see the same tokens and only the node structure
differs. ``size`` is a target length in tokens, which lets structures be
compared at matched granularity (``fixed:128`` vs ``sentences:128`` vs
``attention:128``) and granularity be swept for one structure.

=====================  ==============================================================
``documents``          the retrieved documents themselves (the paper's RAG nodes)
``fixed:N``            consecutive windows of N tokens
``sentences``          one node per sentence; ``sentences:N`` groups sentences to ~N tokens
``paragraphs``         one node per line/paragraph (short lines such as titles merge
                       with the next); ``paragraphs:N`` groups them to ~N tokens
``attention:N``        about len/N nodes, cut at word boundaries where the following
                       tokens attend least to the preceding text (attention shifts)
``attention-sent:N``   the same, but only sentence ends are candidate cut points
=====================  ==============================================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

Span = Tuple[int, int]

KINDS = ("documents", "fixed", "sentences", "paragraphs", "attention", "attention-sent")


@dataclass(frozen=True)
class Chunker:
    kind: str
    size: Optional[int] = None

    @property
    def name(self) -> str:
        return self.kind if self.size is None else f"{self.kind}:{self.size}"


def parse_chunker(spec: str) -> Chunker:
    kind, _, size = spec.partition(":")
    if kind not in KINDS:
        raise ValueError(f"unknown chunker '{kind}' (choose from {', '.join(KINDS)})")
    n = int(size) if size else None
    if kind in ("fixed", "attention", "attention-sent") and not n:
        raise ValueError(f"'{kind}' needs a size, e.g. {kind}:128")
    if n is not None and n < 1:
        raise ValueError("size must be positive")
    return Chunker(kind, n)


def _token_offsets(tokenizer, text: str) -> List[Span]:
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return [tuple(o) for o in enc["offset_mapping"]]


def _count(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def _cuts_to_spans(cuts: Sequence[int], n: int) -> List[Span]:
    points = [0] + sorted(c for c in set(cuts) if 0 < c < n) + [n]
    return [(a, b) for a, b in zip(points, points[1:]) if b > a]


def _group(spans: List[Span], text: str, tokenizer, size: int) -> List[Span]:
    """Merge consecutive spans until each group holds at least ``size`` tokens (the last may be shorter)."""
    out: List[Span] = []
    start, tokens = None, 0
    for a, b in spans:
        if start is None:
            start = a
        tokens += _count(tokenizer, text[a:b])
        if tokens >= size:
            out.append((start, b))
            start, tokens = None, 0
    if start is not None:
        if out and tokens < size // 2:
            out[-1] = (out[-1][0], spans[-1][1])  # fold a short remainder into the previous group
        else:
            out.append((start, spans[-1][1]))
    return out


def sentence_spans(text: str) -> List[Span]:
    cuts = [m.end() for m in re.finditer(r"(?<=[.!?])[\"')\]]*\s+|\n+", text)]
    return _cuts_to_spans(cuts, len(text))


def paragraph_spans(text: str, tokenizer, min_tokens: int = 16) -> List[Span]:
    """Lines/paragraphs; a piece shorter than ``min_tokens`` (e.g. a title) is merged with what follows."""
    spans = _cuts_to_spans([m.end() for m in re.finditer(r"\n+", text)], len(text))
    merged: List[Span] = []
    start = None
    for a, b in spans:
        start = a if start is None else start
        if _count(tokenizer, text[start:b]) >= min_tokens:
            merged.append((start, b))
            start = None
    if start is not None:
        if merged:
            merged[-1] = (merged[-1][0], len(text))
        else:
            merged.append((start, len(text)))
    return merged


def fixed_spans(text: str, tokenizer, size: int) -> List[Span]:
    offsets = _token_offsets(tokenizer, text)
    cuts = [offsets[i][0] for i in range(size, len(offsets), size)]
    return _cuts_to_spans(cuts, len(text))


def attention_spans(text: str, tokenizer, decoder, size: int, *, sentence_only: bool = False, layers=None, window: int = 16, sink: int = 4) -> List[Span]:
    from .attention import cross_boundary_attention

    offsets = _token_offsets(tokenizer, text)
    ids = tokenizer(text, add_special_tokens=False).input_ids
    n = len(ids)
    n_chunks = max(1, round(n / size))
    if n_chunks == 1:
        return [(0, len(text))]
    gap = max(1, size // 2)
    if sentence_only:
        ends = {a for a, _ in sentence_spans(text)[1:]}
        cand = [i for i in range(1, n) if offsets[i][0] in ends or offsets[i][0] + 1 in ends]
    else:
        cand = [i for i in range(1, n) if offsets[i][0] > 0 and (text[offsets[i][0] - 1].isspace() or text[offsets[i][0]].isspace())]
    cand = [i for i in cand if gap <= i <= n - gap and i > sink]
    if not cand:
        return fixed_spans(text, tokenizer, size)
    scores = cross_boundary_attention(decoder, ids, cand, layers=layers, window=window, sink=sink)
    chosen: List[int] = []
    for j in scores.argsort().tolist():
        b = cand[j]
        if all(abs(b - c) >= gap for c in chosen):
            chosen.append(b)
            if len(chosen) == n_chunks - 1:
                break
    return _cuts_to_spans([offsets[b][0] for b in chosen], len(text))


def make_spans(
    text: str,
    chunker: Chunker,
    tokenizer,
    *,
    doc_spans: Optional[Sequence[Span]] = None,
    decoder=None,
    probe_layers=None,
) -> List[Span]:
    """Character spans of the chunks; they partition ``text``."""
    kind, size = chunker.kind, chunker.size
    if kind == "documents":
        if doc_spans is None:
            raise ValueError("the 'documents' chunker needs document spans")
        spans = list(doc_spans)
    elif kind == "fixed":
        spans = fixed_spans(text, tokenizer, size)
    elif kind == "sentences":
        spans = sentence_spans(text)
        if size:
            spans = _group(spans, text, tokenizer, size)
    elif kind == "paragraphs":
        spans = paragraph_spans(text, tokenizer)
        if size:
            spans = _group(spans, text, tokenizer, size)
    else:
        if decoder is None:
            raise ValueError("attention chunking needs the model")
        spans = attention_spans(text, tokenizer, decoder, size, sentence_only=(kind == "attention-sent"), layers=probe_layers)
    if not spans or spans[0][0] != 0 or spans[-1][1] != len(text) or any(a[1] != b[0] for a, b in zip(spans, spans[1:])):
        raise AssertionError(f"chunker '{chunker.name}' did not partition the text")
    return spans
