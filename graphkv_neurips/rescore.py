"""Re-score the generations released with the paper and compare with its tables.

The authors published the raw outputs of every RAG and ArXiv-QA run under
``results/`` of the Hugging Face dataset ``Graph-COM/GraphKV``. Scoring them
with the official metrics (``metrics.py``) checks that the numbers in
Tables 2 and 3 and Figure 6 follow from those outputs.

    python -m graphkv_neurips.rescore --results-dir <dir with results/> --output docs/neurips/rescored.md

The paper truncates (does not round) to two decimals; so does the comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple

from .data import question_types
from .metrics import group_scores, score_example
from .paper_numbers import FIGURE6, RELEASED_ARXIV_RUNS, RELEASED_RAG_RUNS, TABLE2, TABLE3

RAG_TASKS = ("nqa", "2wiki", "tqa", "hqa", "multihopqa", "morehopqa")
TABLE_OF = {"nqa": TABLE3, "2wiki": TABLE3, "tqa": TABLE3, "hqa": TABLE3, "multihopqa": TABLE2, "morehopqa": TABLE2}


def trunc2(x: float) -> float:
    return math.floor(x * 100 + 1e-9) / 100


def _load(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _groups(task: str, rows: List[dict], types: Optional[List[str]]) -> Optional[List[str]]:
    if task == "multihopqa":
        return [r["question_type"] for r in rows]
    if task == "morehopqa":
        return [f"hop{r['no_of_hops']}" for r in rows]
    if types is not None and len(types) == len(rows):
        return types
    return None


def score_rag_file(task: str, path: str, types: Optional[List[str]] = None, scorer: Optional[str] = None) -> Tuple[int, Dict[str, float]]:
    """Accuracy per column; ``scorer`` overrides the metric (``morehopqa-strict``)."""
    rows = _load(path)
    scores = [score_example(scorer or task, r.get("generated", ""), r["answers"]) for r in rows]
    out = group_scores(scores, _groups(task, rows, types))
    keyed = {(task if g == "all" and task in ("nqa", "tqa") else f"{task}/{g}"): v for g, v in out.items()}
    return len(rows), keyed


SCORERS = {"morehopqa": ("morehopqa", "morehopqa-strict")}


def rescore_rag(results_dir: str) -> List[dict]:
    rows = []
    for task in RAG_TASKS:
        types = question_types(task)
        for (model_dir, run), (backbone, method) in RELEASED_RAG_RUNS.items():
            path = os.path.join(results_dir, task, model_dir, f"{run}_ascent.jsonl")
            if not os.path.exists(path):
                continue
            for scorer in SCORERS.get(task, (task,)):
                n, scores = score_rag_file(task, path, types, scorer)
                paper = TABLE_OF[task].get((backbone, method), {})
                for column, value in scores.items():
                    rows.append(
                        dict(task=task, scorer=scorer, backbone=backbone, method=method, file=f"{model_dir}/{run}_ascent.jsonl", n=n,
                             column=column, ours=trunc2(value), paper=paper.get(column))
                    )
    return rows


def rescore_arxiv(results_dir: str) -> List[dict]:
    rows = []
    for distractors, runs in RELEASED_ARXIV_RUNS.items():
        for name, rel in runs.items():
            path = os.path.join(results_dir, "arxiv", rel)
            if not os.path.exists(path):
                continue
            data = _load(path)
            correct = int(sum(score_example("arxiv", r["generated"], r["answers"]) for r in data))
            rows.append(dict(distractors=distractors, run=name, file=rel, n=len(data), ours=correct, paper=FIGURE6[distractors].get(name)))
    return rows


def to_markdown(rag: List[dict], arxiv: List[dict]) -> str:
    lines = ["## RAG (Tables 2 and 3): released outputs re-scored", ""]
    lines.append("| task | scorer | backbone | method | column | n | re-scored | paper | diff |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    tally: Dict[str, List[int]] = {}
    for r in rag:
        diff = "" if r["paper"] is None else f"{r['ours'] - r['paper']:+.2f}"
        if r["paper"] is not None:
            t = tally.setdefault(r["scorer"], [0, 0])
            t[1] += 1
            t[0] += abs(r["ours"] - r["paper"]) <= 0.011  # the paper truncates; allow one unit in the last digit
        paper = "" if r["paper"] is None else f"{r['paper']:.2f}"
        lines.append(f"| {r['task']} | {r['scorer']} | {r['backbone']} | {r['method']} | {r['column']} | {r['n']} | {r['ours']:.2f} | {paper} | {diff} |")
    lines.append("")
    for scorer, (ok, total) in tally.items():
        lines.append(f"* scorer `{scorer}`: {ok} of {total} cells with a paper value agree within 0.01.")
    lines.append("")
    lines += ["## ArXiv-QA (Figure 6): correct answers out of 60", "", "| distractors | run | file | re-scored | paper |", "|---|---|---|---|---|"]
    for r in arxiv:
        lines.append(f"| {r['distractors']} | {r['run']} | {r['file']} | {r['ours']} | {r['paper']} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True, help="local copy of results/ from the Graph-COM/GraphKV dataset")
    ap.add_argument("--output", default=None, help="write the comparison as Markdown")
    ap.add_argument("--json", default=None, help="write all rows as JSON")
    args = ap.parse_args(argv)
    rag = rescore_rag(args.results_dir)
    arxiv = rescore_arxiv(args.results_dir)
    md = to_markdown(rag, arxiv)
    print(md)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(md)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"rag": rag, "arxiv": arxiv}, f, indent=1)


if __name__ == "__main__":
    main()
