"""Evaluation prompts, following the paper's protocol (Sec. II-B).

WikiText-2 test entries longer than ``min_chars`` characters are kept,
truncated to their first ``max_words`` words, and the first ``num_prompts`` of
them are used for every configuration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

WIKITEXT_REPOS = ("Salesforce/wikitext", "wikitext")


def load_wikitext_split(config: str = "wikitext-2-raw-v1", split: str = "test") -> List[str]:
    """Return the raw lines of a WikiText split (needs the ``datasets`` package)."""
    from datasets import load_dataset

    last_error: Optional[Exception] = None
    for repo in WIKITEXT_REPOS:
        try:
            return list(load_dataset(repo, config, split=split)["text"])
        except Exception as err:  # noqa: BLE001 - try the legacy dataset name next
            last_error = err
    raise RuntimeError(f"could not load WikiText ({config}/{split}): {last_error}")


def select_prompts(
    texts: List[str],
    num_prompts: int = 100,
    min_chars: int = 200,
    max_words: int = 200,
) -> List[str]:
    """Apply the paper's filtering: ``> min_chars`` characters, first ``max_words`` words."""
    prompts = []
    for text in texts:
        text = text.strip()
        if len(text) <= min_chars:
            continue
        words = text.split()
        if len(words) > max_words:
            text = " ".join(words[:max_words])
        prompts.append(text)
        if len(prompts) == num_prompts:
            break
    return prompts


def load_prompts_file(path: str) -> List[str]:
    """Read prompts from ``.jsonl`` (one ``{"text": ...}`` per line) or ``.txt`` (one per line)."""
    p = Path(path)
    lines = [line for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if p.suffix == ".jsonl":
        return [json.loads(line)["text"] for line in lines]
    return [line.strip() for line in lines]


def load_prompts(
    num_prompts: int = 100,
    min_chars: int = 200,
    max_words: int = 200,
    prompts_file: Optional[str] = None,
    wikitext_config: str = "wikitext-2-raw-v1",
) -> List[str]:
    if prompts_file:
        texts = load_prompts_file(prompts_file)
    else:
        texts = load_wikitext_split(wikitext_config, "test")
    prompts = select_prompts(texts, num_prompts, min_chars, max_words)
    if not prompts:
        raise ValueError("no prompts passed the length filter")
    return prompts
