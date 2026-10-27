"""Replicate the paper's RAG tables on a sample of questions.

Table 3 covers NarrativeQA, 2Wiki, TriviaQA and HotpotQA, Table 2 MultiHop-RAG and MorehopQA: ~37k
questions per method, several hundred GPU hours. This answers ``--num-examples`` questions per
dataset (stratified by question type) with the paper's model and methods, greedy with the official
limit of 1024 new tokens, and prints every cell of the table three times:

* ``ours``: this implementation on the sampled questions;
* ``released``: the authors' released generations for the same questions, scored the same way;
* ``paper``: the number printed in the paper, over all questions.

``ours`` vs ``released`` is the replication test: per question they should agree up to bfloat16
near-ties. ``released`` vs ``paper`` is the sampling error of a small sample only; over all questions
the released outputs give the paper's numbers (docs/neurips/rescored_official_outputs.md).

    python -m graphkv_neurips.replicate --table 3 --model ldsjmdy/Tulu3-Block-FT --num-examples 50 \\
        --output-dir runs/table3 [--time-budget-min 100 --cu-per-hour 4.8]

Each row is scored with the metric that reproduces its printed number: MorehopQA with the strict
answer extraction except Graph-KV Full, MultiHop-RAG with the benchmark's own metric (the paper's
scorer was not released, so that paper column is not comparable). Answers are appended per
question, so a rerun resumes; ``--report-only`` prints the tables from the saved answers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, List, Optional

import torch

from .compare_official import OFFICIAL_RUNS
from .data import fetch, load_rag
from .methods import generate, parse_method
from .metrics import score_example
from .model import RopeDecoder
from .paper_numbers import TABLE2, TABLE3, TABLE5_6
from .prompts import PromptBuilder
from .run_rag import load_model, reorder, select, stop_ids
from .run_structure import safe_name
from .verify_released import _clean

TABLES = {3: ("nqa", "2wiki", "tqa", "hqa"), 2: ("multihopqa", "morehopqa")}
TASK_NAMES = {"nqa": "NarrativeQA", "2wiki": "2Wiki", "tqa": "TriviaQA", "hqa": "HotpotQA", "multihopqa": "MultiHop-RAG", "morehopqa": "MorehopQA"}
# per-type columns, in the order of the paper's tables
COLUMNS = {
    "2wiki": ["comparison", "inference", "bridge_comparison", "compositional"],
    "hqa": ["comparison", "bridge"],
    "multihopqa": ["inference_query", "comparison_query", "temporal_query", "null_query"],
    "morehopqa": [f"hop{h}" for h in range(1, 6)],
}
BACKBONES = {"ldsjmdy/Tulu3-Block-FT": "8B-Block", "ldsjmdy/Tulu3-SFT": "8B-SFT", "ldsjmdy/Tulu3-RAG": "8B-RAG"}
# the rows each backbone has in Tables 2 and 3
PAPER_METHODS = {
    "8B-Block": "sequential,block,graphkv-top1@official,graphkv-top3@official,graphkv-full@official",
    "8B-SFT": "sequential,pcw,ape",
    "8B-RAG": "sequential,pcw,ape",
}
METHOD_NAMES = {
    "sequential": "Sequential", "pcw": "PCW", "ape": "APE", "block": "Block-RAG",
    "graphkv-top1@official": "Graph-KV Top-1", "graphkv-top3@official": "Graph-KV Top-3", "graphkv-full@official": "Graph-KV Full",
}


def scorer(task: str, method: str) -> str:
    """The metric that reproduces the paper's number for this row (docs/neurips/VERIFICATION.md)."""
    if task == "morehopqa" and method != "Graph-KV Full":
        return "morehopqa-strict"
    return task


def paper_value(backbone: str, method: str, task: str, group: str) -> Optional[float]:
    key = task if task in ("nqa", "tqa") and group == "all" else f"{task}/{group}"
    for table in (TABLE3, TABLE2):
        if key in table.get((backbone, method), {}):
            return table[(backbone, method)][key]
    if backbone == "8B-SFT":  # Tables 5/6, the untuned backbone
        return TABLE5_6.get(method, {}).get(key)
    return None


def released_outputs(task: str, model: str, run: Optional[str], questions: Dict[int, str], data_dir=None) -> Dict[int, Optional[str]]:
    """The authors' generation for each wanted example index (None: released row without output).

    Rows are matched by question: MorehopQA was run in another order than its data file."""
    if run is None:
        return {}
    try:
        path = fetch(f"results/{task}/{model.replace('/', '')}/{run}_ascent.jsonl", data_dir)
    except Exception:  # noqa: BLE001 - not every run was released
        return {}
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    out: Dict[int, Optional[str]] = {}
    for i, q in questions.items():
        if i < len(lines):
            rec = json.loads(lines[i])
            if rec.get("question") == q:
                out[i] = rec.get("generated")
    missing = {q: i for i, q in questions.items() if i not in out}
    if missing:
        for line in lines:
            rec = json.loads(line)
            i = missing.get(rec.get("question"))
            if i is not None:
                out[i] = rec.get("generated")
    return out


def same_text(ours: str, released: str) -> bool:
    """Equal up to white space (the released text was decoded with ``clean_up_tokenization_spaces``)."""
    return "".join(_clean(ours).split()) == "".join(_clean(released).split())


def _records(path: str) -> Dict[int, dict]:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["index"]: r for r in (json.loads(line) for line in f if line.strip())}


def _accuracy(scores: Dict[int, float], idx: List[int]) -> Optional[float]:
    vals = [scores[i] for i in idx if i in scores]
    return 100 * sum(vals) / len(vals) if vals else None


def _fmt(x: Optional[float], digits: int = 1) -> str:
    return "" if x is None else f"{x:.{digits}f}"


# ----------------------------------------------------------------------------- run


def run(args) -> None:
    t_start = time.time()
    released_model = args.released_model or args.model
    backbone = BACKBONES.get(released_model)
    if args.methods is None and backbone is None:
        raise SystemExit(f"no paper rows for {released_model}; give --methods or --released-model")
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()] if args.tasks else list(TABLES[args.table])
    labels = [m.strip() for m in (args.methods or PAPER_METHODS[backbone]).split(",") if m.strip()]
    examples, n_total = {}, {}
    for t in tasks:
        data = load_rag(t, args.data_dir)
        n_total[t] = len(data)
        examples[t] = select(data, args.num_examples, "stratified", args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    config = dict(vars(args), tasks=tasks, methods=labels, released_model=released_model, n_total=n_total, torch=torch.__version__)
    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=1)

    tok, model = load_model(args.model, args.dtype, args.device)
    dec = RopeDecoder(model)
    builder = PromptBuilder(tok, args.prompt_style)
    stops = stop_ids(tok, model)
    pad = tok.pad_token_id if tok.pad_token_id is not None else (stops[0] if stops else 0)
    methods = {label: parse_method(label, bos_id=tok.bos_token_id) for label in labels}
    paths = {(t, label): os.path.join(args.output_dir, t, safe_name(label) + ".jsonl") for t in tasks for label in labels}
    done = {key: set(_records(path)) for key, path in paths.items()}
    # datasets take turns, so a run stopped by the time budget leaves every table partly filled
    queue = []
    for k in range(max(len(v) for v in examples.values())):
        queue += [(t, examples[t][k]) for t in tasks if k < len(examples[t])]
    todo = [(t, ex) for t, ex in queue if any(ex.index not in done[t, label] for label in labels)]
    print(f"loaded model and {len(queue)} questions in {time.time() - t_start:.0f}s; {len(todo)} to answer with {len(labels)} methods", flush=True)

    t_loop = time.time()
    for k, (task, ex) in enumerate(todo):
        if args.time_budget_min is not None and (time.time() - t_start) / 60 > args.time_budget_min:
            print(f"time budget of {args.time_budget_min} min reached after {k} questions; rerun the same command to resume", flush=True)
            break
        tp, order = reorder(builder.rag_text(ex.question, ex.documents, task), "ascent", 0)
        prompt = builder.tokenize(tp, order)
        for label in labels:
            if ex.index in done[task, label]:
                continue
            t0 = time.perf_counter()
            gen = generate(dec, prompt, methods[label], max_new_tokens=args.max_new_tokens, stop_ids=stops, pad_id=pad)
            rec = dict(index=ex.index, group=ex.group, question=ex.question, answers=ex.answers,
                       generated=tok.decode(gen.token_ids, skip_special_tokens=True), new_tokens=len(gen.token_ids),
                       seconds=round(time.perf_counter() - t0, 2))
            os.makedirs(os.path.dirname(paths[task, label]), exist_ok=True)
            with open(paths[task, label], "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n = k + 1
        if n == args.warmup or n % 25 == 0:
            per, left = (time.time() - t_loop) / n, len(todo) - n
            msg = f"[{n}/{len(todo)} questions] {per:.1f}s per question ({len(labels)} methods), about {per * left / 60:.0f} min left"
            if args.cu_per_hour:
                msg += f" (~{per * left / 3600 * args.cu_per_hour:.1f} compute units)"
            print(msg, flush=True)


# ----------------------------------------------------------------------------- report


def report(output_dir: str, data_dir=None) -> str:
    with open(os.path.join(output_dir, "run_config.json")) as f:
        cfg = json.load(f)
    tasks, labels, released_model = cfg["tasks"], cfg["methods"], cfg["released_model"]
    backbone = BACKBONES.get(released_model, released_model)
    lines = [f"Model `{cfg['model']}`, compared with the released outputs and paper rows of `{released_model}` ({backbone}); "
             f"greedy, at most {cfg['max_new_tokens']} new tokens.", ""]
    summary: Dict[str, Dict[str, tuple]] = {label: {} for label in labels}
    same = {label: [0, 0, 0, 0] for label in labels}  # same text, released texts, same correctness, released scores
    sizes = []
    for task in tasks:
        recs = {label: _records(os.path.join(output_dir, task, safe_name(label) + ".jsonl")) for label in labels}
        common = sorted(set.intersection(*(set(r) for r in recs.values())))
        lines += [f"### {TASK_NAMES.get(task, task)}", ""]
        if not common:
            lines += ["no question answered by every method yet", ""]
            continue
        sizes.append(len(common))
        first = recs[labels[0]]
        groups = [g for g in COLUMNS.get(task, []) if any(first[i]["group"] == g for i in common)] + ["all"]
        idx = {g: [i for i in common if g == "all" or first[i]["group"] == g] for g in groups}
        questions = {i: first[i]["question"] for i in common}
        lines += [f"{len(common)} of {cfg['n_total'][task]} questions, answered by every method.", "",
                  "| method | numbers | " + " | ".join(f"{g} (n={len(idx[g])})" for g in groups) + " | same text as released | same correctness |",
                  "|---" * (len(groups) + 4) + "|"]
        for label in labels:
            name = METHOD_NAMES.get(label, label)
            metric = scorer(task, name)
            ours = {i: score_example(metric, recs[label][i]["generated"], first[i]["answers"]) for i in common}
            texts = released_outputs(task, released_model, OFFICIAL_RUNS.get(label), questions, data_dir)
            released = {i: score_example(metric, t or "", first[i]["answers"]) for i, t in texts.items()}
            n_text = sum(t is not None for t in texts.values())
            n_same = sum(t is not None and same_text(recs[label][i]["generated"], t) for i, t in texts.items())
            n_agree = sum(ours[i] == released[i] for i in released)
            for j, v in enumerate((n_same, n_text, n_agree, len(released))):
                same[label][j] += v
            row_ours = [_accuracy(ours, idx[g]) for g in groups]
            row_rel = [_accuracy(released, idx[g]) for g in groups]
            row_paper = [paper_value(backbone, name, task, g) for g in groups]
            agree = f"{n_same}/{n_text} | {n_agree}/{len(released)}" if released else "not released | "
            cover = "" if len(released) in (0, len(common)) else f" ({len(released)} released)"
            lines.append(f"| {name} | ours | " + " | ".join(map(_fmt, row_ours)) + f" | {agree} |")
            lines.append(f"| | released, same questions{cover} | " + " | ".join(map(_fmt, row_rel)) + " | | |")
            lines.append("| | paper, all questions | " + " | ".join(_fmt(v, 2) for v in row_paper) + " | | |")
            summary[label][task] = (row_ours[-1], row_rel[-1], row_paper[-1])
        lines.append("")

    lines += ["### Summary: the All column (ours / released on the same questions / paper)", "",
              "| method | " + " | ".join(TASK_NAMES.get(t, t) for t in tasks) + " | same text as released | same correctness |",
              "|---" * (len(tasks) + 3) + "|"]
    for label in labels:
        cells = []
        for t in tasks:
            o, r, p = summary[label].get(t, (None, None, None))
            cells.append(" / ".join(_fmt(v, d) or "-" for v, d in ((o, 1), (r, 1), (p, 2))))
        s, t, a, n = same[label]
        tail = f"{s}/{t} | {a}/{n}" if n else "not released | "
        lines.append(f"| {METHOD_NAMES.get(label, label)} | " + " | ".join(cells) + f" | {tail} |")
    s, t, a, n = (sum(v[j] for v in same.values()) for j in range(4))
    lines.append("")
    if n:
        lines.append(f"Our answer has the same correctness as the released one on {a} of {n} questions ({100 * a / n:.1f}%), "
                     f"the same text on {s} of {t} ({100 * s / max(1, t):.1f}%). Different text after a bfloat16 near-tie is expected; "
                     "different correctness should be rare.")
    if sizes:
        half = 196 * math.sqrt(0.25 / min(sizes))
        lines.append(f"With {min(sizes)} questions per dataset an accuracy is only known to about +-{half:.0f} points (95%), so "
                     "`released` and `paper` may differ by that much; the replication test is `ours` against `released`.")
    if "multihopqa" in tasks:
        lines.append("MultiHop-RAG: the paper's scorer (`qa_evaluate2`) was not released; scored with the benchmark's own metric, "
                     "so its paper column is not comparable (released outputs re-scored: docs/neurips/rescored_official_outputs.md).")
    if "morehopqa" in tasks:
        lines.append("MorehopQA: strict answer extraction (no `Answer:` counts as wrong) for every row but Graph-KV Full, "
                     "which is how the printed numbers were computed (docs/neurips/VERIFICATION.md).")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", type=int, choices=sorted(TABLES), default=3)
    ap.add_argument("--tasks", default=None, help="comma-separated datasets instead of a whole table")
    ap.add_argument("--model", default="ldsjmdy/Tulu3-Block-FT")
    ap.add_argument("--released-model", default=None,
                    help="whose released outputs and paper rows to compare with (default: --model; set it to debug with a small model)")
    ap.add_argument("--methods", default=None, help="default: the paper's rows for the backbone")
    ap.add_argument("--num-examples", type=int, default=50, help="questions per dataset, stratified by question type")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--prompt-style", choices=["auto", "tulu", "chat"], default="auto")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--time-budget-min", type=float, default=None, help="stop cleanly after this many minutes")
    ap.add_argument("--cu-per-hour", type=float, default=None, help="Colab compute units per hour, for the cost projection")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--report-only", action="store_true", help="print the tables from the answers saved in --output-dir")
    args = ap.parse_args(argv)
    if not args.report_only:
        run(args)
    md = report(args.output_dir, args.data_dir)
    with open(os.path.join(args.output_dir, "replication.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
