"""Answer scoring, ported from the official evaluation (``rag_eval.py``, ``arxiv_eval.py``).

* Most tasks use ``best_subspan_em``: after SQuAD normalisation, a prediction is
  correct when it *contains* any gold answer (Appendix A.2.1, following
  Block-Attention / Lost-in-the-Middle).
* MultiHop-RAG uses the benchmark's own lenient word-overlap accuracy.
* MorehopQA first extracts the text after the last ``Answer:``.

``group_scores`` reproduces the per-type columns of Tables 2 and 3.
"""

from __future__ import annotations

import re
import string
from typing import Dict, Iterable, List, Optional, Sequence, Union

Answers = Union[str, Sequence[str]]

STOP_MARKERS = ("<|im_end|>", "<|eot_id|>")


def normalize_answer(s: str) -> str:
    """SQuAD normalisation: lower case, no punctuation, no articles, single spaces."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def best_subspan_em(prediction: str, ground_truths: Iterable[str]) -> float:
    normalized_prediction = normalize_answer(prediction)
    for ground_truth in ground_truths:
        if normalize_answer(ground_truth).lower() in normalized_prediction.lower():
            return 1.0
    return 0.0


def strip_generation(generated: str) -> str:
    """The official scorer keeps the text before the first end-of-turn marker."""
    for marker in STOP_MARKERS:
        generated = generated.split(marker)[0]
    return generated


def as_answer_list(answers: Answers) -> List[str]:
    if isinstance(answers, str):
        return [answers]
    return list(answers)


def subspan_accuracy(generated: str, answers: Answers) -> float:
    return best_subspan_em(strip_generation(generated), as_answer_list(answers))


# --- MultiHop-RAG (Tang & Yang, 2024) -------------------------------------------------
# The paper scores MultiHop-RAG with an unreleased ``qa_evaluate2``; this is the
# benchmark's published ``qa_evaluate.py`` (answer extraction + word overlap).


def multihop_extract_answer(text: str) -> str:
    match = re.search(r'The answer to the question is "(.*?)"', text)
    return match.group(1) if match else text


def multihop_accuracy(generated: str, gold: str) -> float:
    """Correct when prediction and gold share a lower-cased whitespace-separated word."""
    prediction = multihop_extract_answer(generated) if "The answer" in generated else generated
    return float(bool(set(prediction.lower().split()) & set(gold.lower().split())))


# --- MorehopQA ------------------------------------------------------------------------


def morehop_extract_answer(generated: str, strict: bool = False) -> str:
    """Text after the last ``answer:`` (case-insensitive).

    Without a match the released ``rag_eval.py`` keeps the whole output; with
    ``strict`` the prediction is empty instead. The paper's Table 2 baselines
    (and the Graph-KV Top-1/Top-3 rows) follow the strict rule, its Graph-KV-Full
    row the released one (see docs/neurips/README.md).
    """
    matches = list(re.finditer(r"answer:\s*(.*)", generated, re.IGNORECASE))
    if matches:
        return matches[-1].group(1)
    return "" if strict else generated


def morehop_accuracy(generated: str, answers: Answers, strict: bool = False) -> float:
    return best_subspan_em(strip_generation(morehop_extract_answer(generated, strict)), as_answer_list(answers))


# --- dispatch -------------------------------------------------------------------------

TASKS = ("2wiki", "hqa", "tqa", "nqa", "multihopqa", "morehopqa", "morehopqa-strict", "arxiv", "classify")


def score_example(task: str, generated: str, answers) -> float:
    if task == "multihopqa":
        return multihop_accuracy(generated, answers if isinstance(answers, str) else answers[0])
    if task in ("morehopqa", "morehopqa-strict"):
        return morehop_accuracy(generated, answers, strict=task.endswith("strict"))
    if task == "nqa" and answers and isinstance(answers[0], (list, tuple)):
        answers = answers[0]  # NarrativeQA answers are stored as [[...]]
    return subspan_accuracy(generated, answers)


def group_scores(scores: Sequence[float], groups: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """Mean accuracy (in %) per group, plus ``"all"``; groups keep first-seen order."""
    out: Dict[str, List[float]] = {}
    if groups is not None:
        if len(groups) != len(scores):
            raise ValueError("groups and scores must have the same length")
        for g, s in zip(groups, scores):
            out.setdefault(str(g), []).append(s)
    out["all"] = list(scores)
    return {g: 100.0 * sum(v) / len(v) for g, v in out.items() if v}
