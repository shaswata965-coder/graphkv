"""Paper topic classification on Cora / PubMed (Sec. 4.3, Table 4, Appendix A.2.3).

    python -m graphkv_neurips.run_classify --model ldsjmdy/Tulu3-Block-FT --dataset cora \\
        --methods query-only,sequential,block,graphkv --num-nodes 542 --output-dir runs/cora

Each test node is a central paper; its citation neighbours are the references.
``graphkv`` encodes the references as sources (shared positions) and the central
paper as their target (the ego-graph of Sec. 4.3); ``query-only`` reads only the
central paper. As in the paper, the central paper always comes last, and
sequential encoding and Block-RAG are run with the reference order shuffled by
seeds 42, 43 and 44 (mean and standard deviation reported); Graph-KV does not
depend on the order.

Not specified by the paper and chosen here (documented in docs/neurips): the
test split is the one shipped with ``Graph-COM/Text-Attributed-Graphs``;
neighbours are taken in both edge directions (the graphs are undirected); the
predicted class is the first class name found after the last ``Answer:``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
from typing import List, Optional, Sequence

import torch

from .methods import decode, parse_method, prepare
from .metrics import normalize_answer
from .model import RopeDecoder
from .prompts import PromptBuilder
from .run_rag import load_model, stop_ids
from .tag_data import load_tag

SEEDED = ("sequential", "block", "pcw", "ape")


def predict_class(text: str, classes: Sequence[str]) -> Optional[int]:
    """Index of the first class name mentioned after the last 'Answer:' (or anywhere if there is none)."""
    m = list(re.finditer(r"answer:\s*", text, re.IGNORECASE))
    tail = normalize_answer(text[m[-1].end():] if m else text)
    best, best_pos = None, None
    for i, c in enumerate(classes):
        pos = tail.find(normalize_answer(c))
        if pos >= 0 and (best_pos is None or pos < best_pos):
            best, best_pos = i, pos
    return best


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="ldsjmdy/Tulu3-Block-FT")
    ap.add_argument("--dataset", choices=["cora", "pubmed"], default="cora")
    ap.add_argument("--methods", default="query-only,sequential,block,graphkv")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--num-nodes", type=int, default=0, help="0 = the whole test split")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--prompt-style", choices=["auto", "tulu", "chat"], default="auto")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--time-budget-min", type=float, default=None)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)

    import time

    t_start = time.time()
    graph = load_tag(args.dataset, args.data_dir)
    test = graph.masks["test_mask"].nonzero().view(-1).tolist()
    if args.num_nodes:
        test = test[: args.num_nodes]
    tok, model = load_model(args.model, args.dtype, args.device)
    dec = RopeDecoder(model)
    builder = PromptBuilder(tok, args.prompt_style)
    stops = stop_ids(tok, model)
    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)
    runs = []
    for name in [m.strip() for m in args.methods.split(",") if m.strip()]:
        for seed in (seeds if name in SEEDED else [None]):
            runs.append((name, seed))
    for name, seed in runs:
        path = os.path.join(args.output_dir, f"{name}" + (f"_{seed}" if seed is not None else "") + ".jsonl")
        done = set()
        if os.path.exists(path):
            with open(path) as f:
                done = {json.loads(line)["node"] for line in f if line.strip()}
        for node in test:
            if node in done:
                continue
            if args.time_budget_min is not None and (time.time() - t_start) / 60 > args.time_budget_min:
                print("time budget reached; rerun to resume", flush=True)
                break
            refs = [graph.texts[n] for n in graph.neighbors(node)]
            if seed is not None:
                random.Random(seed * 100003 + node).shuffle(refs)
            if name == "query-only":
                refs = []
            tp = builder.classify_text(graph.texts[node], refs, graph.classes)
            prompt = builder.tokenize(tp)
            if name == "query-only":
                spec = "sequential"
            elif name == "graphkv":
                spec = "graphkv-star"
            else:
                spec = name
            method = parse_method(spec, bos_id=tok.bos_token_id)
            tokens, _, _ = decode(dec, prepare(dec, prompt, method, pad_id=tok.pad_token_id or 0), max_new_tokens=args.max_new_tokens, stop_ids=stops)
            text = tok.decode(tokens, skip_special_tokens=True)
            pred = predict_class(text, graph.classes)
            rec = dict(node=node, label=int(graph.labels[node]), pred=pred, correct=float(pred == int(graph.labels[node])), n_refs=len(refs), generated=text)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    summary = {}
    for name in dict.fromkeys(n for n, _ in runs):
        accs = []
        for seed in (seeds if name in SEEDED else [None]):
            path = os.path.join(args.output_dir, f"{name}" + (f"_{seed}" if seed is not None else "") + ".jsonl")
            if os.path.exists(path):
                with open(path) as f:
                    rows = [json.loads(line) for line in f if line.strip()]
                if rows:
                    accs.append(100 * sum(r["correct"] for r in rows) / len(rows))
        if accs:
            summary[name] = dict(mean=statistics.mean(accs), std=statistics.pstdev(accs) if len(accs) > 1 else 0.0, runs=len(accs))
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    for name, s in summary.items():
        print(f"{name:12s} {s['mean']:.2f} ± {s['std']:.2f} ({s['runs']} run(s))")


if __name__ == "__main__":
    main()
