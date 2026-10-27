"""ArXiv-QA (Sec. 4.2, Fig. 6) exactly as ``arxiv_inference.py`` builds it.

    python -m graphkv_neurips.run_arxiv --model ldsjmdy/Tulu3-Block-FT --method graphkv --groups 1 --output-dir runs/arxiv

* ``--groups`` 1 / 2 / 3 = 0 / 1 / 2 distractor ego-graphs (the official ``--batch_size``);
  distractors are drawn as the official script does: ``random.seed(seed)`` once, then
  ``random.sample(range(1, 61), groups - 1)`` per question (the question's own paper can be drawn).
* ``--order first|last``: where the relevant ego-graph goes (sequential and Block-RAG).
* Methods and their official construction:

  ==============  =================================================================================
  ``sequential``  prefix + per group: ``'\\n\\n'.join(references) + '\\n\\n Now please read the
                  paper:' + center``, + question; at most 512 new tokens (``pcw.vanilla``)
  ``graphkv``     references are sources at a shared range, each central paper (after
                  ``ARXIV_CENTER_INTRO``) the target of its own references; groups side by side
                  (``gapemp_graph`` / ``gapemp_graph_batch``); at most 256 new tokens
  ``block``       one group: references, ``ARXIV_CENTER_INTRO`` and the center as separate blocks;
                  several groups: references + center per group (``block`` / ``block_batch``); 256
  ==============  =================================================================================

The full-text contexts are ~88k tokens per question (264k with two distractors), so this needs a
large GPU (the paper used A100s); it is provided for completeness and parity, not for Colab T4/L4.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import List, Tuple

import torch

from .data import ArxivExample, load_arxiv
from .graph import GraphPlan
from .methods import Method, Prompt, decode, prepare
from .metrics import score_example
from .model import RopeDecoder
from .prompts import ARXIV_CENTER_INTRO, ARXIV_CENTER_INTRO_BATCH, ARXIV_INSTRUCTION, ARXIV_SEQ_INTRO, PromptBuilder
from .run_rag import load_model, stop_ids

MAX_NEW = {"sequential": 512, "graphkv": 256, "block": 256}


def distractor_plan(groups: int, seed: int, set_ids=range(1, 61)) -> List[List[int]]:
    """The distractor set ids per question, in the order the official loop draws them."""
    rng = random.Random(seed)
    ids = list(set_ids)
    return [rng.sample(ids, groups - 1) if groups > 1 else [] for _ in ids]


def build(builder: PromptBuilder, method: str, groups: List[Tuple[str, List[str]]], question: str) -> Tuple[Prompt, Method]:
    """``groups``: (center, references) in reading order."""
    prefix, query = builder.wrap(ARXIV_INSTRUCTION, f"\n Question: {question} \n")
    enc = builder.encode
    if method == "sequential":
        text = prefix + "".join("\n\n".join(refs) + ARXIV_SEQ_INTRO + center for center, refs in groups) + query
        return Prompt([], [], [], enc(text)), Method("sequential")
    if method == "block":
        blocks: List[str] = []
        if len(groups) == 1:
            center, refs = groups[0]
            blocks = list(refs) + [ARXIV_CENTER_INTRO, center]
        else:
            for center, refs in groups:
                blocks += list(refs) + [center]
        return Prompt(enc(prefix), [enc(b) for b in blocks], enc(query)), Method("block")
    if method == "graphkv":
        intro = ARXIV_CENTER_INTRO if len(groups) == 1 else ARXIV_CENTER_INTRO_BATCH
        chunks: List[str] = []
        edges = []
        for center, refs in groups:
            first = len(chunks)
            chunks += list(refs) + [intro + center]
            target = len(chunks) - 1
            edges += [(first + i, target) for i in range(len(refs))]
        plan = GraphPlan.from_edges(len(chunks), edges)
        return Prompt(enc(prefix), [enc(c) for c in chunks], enc(query)), Method("graphkv", plan=lambda p: plan, layout="paper-local")
    raise ValueError(f"unknown method '{method}'")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="ldsjmdy/Tulu3-Block-FT")
    ap.add_argument("--method", choices=["sequential", "graphkv", "block"], required=True)
    ap.add_argument("--groups", type=int, choices=[1, 2, 3], default=1)
    ap.add_argument("--order", choices=["first", "last"], default="last")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=None, help="default: the official limit of the method")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)

    data = {e.set_id: e for e in load_arxiv(args.data_dir)}
    tok, model = load_model(args.model, args.dtype, args.device)
    dec = RopeDecoder(model)
    builder = PromptBuilder(tok, "tulu")
    stops = [tok.eos_token_id]
    max_new = args.max_new_tokens or MAX_NEW[args.method]
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, f"{args.method}_{args.groups}_{args.order}_{args.seed}.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            done = {json.loads(line)["set_id"] for line in f if line.strip()}
    plans = distractor_plan(args.groups, args.seed)
    for set_id, distractors in zip(range(1, 61), plans):
        if set_id in done:
            continue
        ex: ArxivExample = data[set_id]
        others = [(data[d].center, data[d].references) for d in distractors]
        own = [(ex.center, ex.references)]
        groups = own + others if args.order == "first" else others + own
        prompt, method = build(builder, args.method, groups, ex.question)
        tokens, _, _ = decode(dec, prepare(dec, prompt, method, pad_id=tok.pad_token_id or 0), max_new_tokens=max_new, stop_ids=stops)
        text = tok.decode(tokens, skip_special_tokens=True)
        rec = dict(set_id=set_id, distractors=distractors, generated=text, answers=ex.answer, score=score_example("arxiv", text, ex.answer))
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print(f"{args.method} groups={args.groups} order={args.order}: {int(sum(r['score'] for r in rows))} / {len(rows)} correct")


if __name__ == "__main__":
    main()
