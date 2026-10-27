"""Prompt construction (Appendix A.2): prefix, text chunks and question.

``style="tulu"`` reproduces the strings of the official scripts byte for byte
(``utils.py: build_prefix, build_suffix, docs2blocks``; ``arxiv_inference.py``),
which are written for the Tulu-3 models (``<|user|>`` / ``<|assistant|>``).
``style="chat"`` puts the same instruction, chunks and question into one user
turn of the tokenizer's own chat template, for other instruction-tuned models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from .methods import Prompt

RAG_INSTRUCTION = (
    "You are an intelligent AI assistant. Please answer questions based on the user's instructions. "
    "Below are some reference documents that may help you in answering the user's question.\n\n"
)
ARXIV_INSTRUCTION = (
    "You are an intelligent AI assistant. You will first read the related works of a paper, "
    "then you will read the paper. Then answer the question.\n\n"
)
CLASSIFY_INSTRUCTION = (
    "You are an intelligent AI assistant. You will first read a list of titles or abstracts of papers cited by a "
    "central paper. Then, you will read the title or abstract of the central paper itself. Finally, you will answer "
    "a question related to the central paper:\n\n"
)
ARXIV_CENTER_INTRO = "\nNow you will read the center paper and answer a related question: \n"  # pcw_parallel.gapemp_graph
ARXIV_SEQ_INTRO = "\n\n Now please read the paper:"  # arxiv_inference.py, sequential prompt


def rag_question(question: str, task: str) -> str:
    """Question block of ``build_suffix`` without the Tulu turn markers."""
    if task in ("2wiki", "hqa", "tqa", "nqa"):
        return (
            "Please write a high-quality answer for the given question using only the provided search documents "
            f"(some of which might be irrelevant). \n Question: {question} \n"
        )
    if task == "multihopqa":
        return (
            "Please write a high-quality answer for the given question using only the provided search documents. "
            "The answer to the question is a word or entity. If the provided information is insufficient to answer "
            "the question, respond 'Insufficient Information'. Please finally give your answer started with: "
            f"'The answer is:'. \n Question: {question} \n"
        )
    if task == "morehopqa":
        return (
            "Please write a high-quality answer for the given question using only the provided search documents. "
            "(If the answer is a date, format is as follows: YYYY-MM-DD (ISO standard).) After thinking step by step, "
            f"give your final answer following 'Answer:' \n Question: {question} \n"
        )
    raise ValueError(f"unknown RAG task '{task}'")


def rag_blocks(documents: Sequence, task: str) -> List[str]:
    """``docs2blocks(..., order='ascent')``: chunks sorted by ascending retrieval score."""
    if task in ("hqa", "2wiki", "tqa", "nqa"):
        docs = sorted(documents, key=lambda d: d["score"])
        return ["- Title: " + d["title"] + "\n" + d["text"] + "\n" for d in docs]
    if task == "multihopqa":
        docs = sorted(documents, key=lambda d: d["score"])
        return [d["text"].replace("[Excerpt from document]\nt", "T") + "\n" for d in docs]
    if task == "morehopqa":
        return [d + "\n" for d in documents]  # the file is already in ascending order
    raise ValueError(f"unknown RAG task '{task}'")


@dataclass
class TextPrompt:
    prefix: str
    chunks: List[str]
    query: str
    joined: Optional[str] = None  # text of the sequential baseline if it differs from prefix + chunks + query

    def text(self) -> str:
        return self.joined if self.joined is not None else self.prefix + "".join(self.chunks) + self.query


class PromptBuilder:
    # Llama-3 templates stamp today's date into the system header; pin it so prompts are reproducible.
    CHAT_KWARGS = {"date_string": "26 Jul 2024"}

    def __init__(self, tokenizer, style: str = "auto") -> None:
        if style == "auto":
            name = (getattr(tokenizer, "name_or_path", "") or "").lower()
            style = "tulu" if "tulu" in name else "chat"
        if style not in ("tulu", "chat"):
            raise ValueError("style must be 'tulu', 'chat' or 'auto'")
        if style == "chat" and not getattr(tokenizer, "chat_template", None):
            raise ValueError("this tokenizer has no chat template; use style='tulu'")
        self.tokenizer = tokenizer
        self.style = style

    # -- text --------------------------------------------------------------

    def wrap(self, instruction: str, question: str) -> tuple:
        """Return ``(prefix, query)`` strings around the chunks."""
        if self.style == "tulu":
            return "<|user|>\n" + instruction, question + "<|assistant|>\n"
        sentinel = "⁣GRAPHKV_CHUNKS⁣"
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction + sentinel + question.rstrip()}], tokenize=False, add_generation_prompt=True,
            **self.CHAT_KWARGS,
        )
        prefix, query = rendered.split(sentinel)
        return prefix, query

    def rag_text(self, question: str, documents: Sequence, task: str) -> TextPrompt:
        prefix, query = self.wrap(RAG_INSTRUCTION, rag_question(question, task))
        return TextPrompt(prefix, rag_blocks(documents, task), query)

    def arxiv_text(self, center: str, references: Sequence[str], question: str) -> TextPrompt:
        """ArXiv-QA (``arxiv_inference.py``): references are sources, the central paper the target.

        Graph-KV and the parallel baselines read the central paper as
        ``ARXIV_CENTER_INTRO + center``; the sequential prompt uses the official
        ``'\\n\\n'.join(references) + ARXIV_SEQ_INTRO + center``.
        """
        prefix, query = self.wrap(ARXIV_INSTRUCTION, f"\n Question: {question} \n")
        joined = prefix + "\n\n".join(references) + ARXIV_SEQ_INTRO + center + query
        return TextPrompt(prefix, list(references) + [ARXIV_CENTER_INTRO + center], query, joined)

    def classify_text(self, center: str, references: Sequence[str], classes: Sequence[str]) -> TextPrompt:
        """Paper topic classification (Appendix A.2.3): references are sources, the central paper the target."""
        question = f"Classify the central paper into one of the following categories: {', '.join(classes)}. Provide your answer following 'Answer:'"
        prefix, query = self.wrap(CLASSIFY_INSTRUCTION, "\n" + question + " \n")
        center_block = "\n\n They are all cited by the following central paper: : " + center + "\n"
        refs = [r + "\n" for r in references]
        return TextPrompt(prefix, refs + [center_block], query)

    # -- tokens ------------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        return list(self.tokenizer(text, add_special_tokens=False).input_ids)

    def tokenize(self, tp: TextPrompt, order: Optional[List[int]] = None) -> Prompt:
        chunks = [self.encode(c) for c in tp.chunks]
        if any(len(c) == 0 for c in chunks):
            raise ValueError("a text chunk tokenized to zero tokens")
        return Prompt(self.encode(tp.prefix), chunks, self.encode(tp.query), self.encode(tp.text()), order)
