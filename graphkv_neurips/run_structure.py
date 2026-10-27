"""Graph-structure experiments: which nodes, which edges, what the query keeps.

One run = one dataset x one chunker x one relevance scorer x several
conditions (methods), all on the same examples:

    python -m graphkv_neurips.run_structure --model ldsjmdy/Tulu3-Block-FT --task 2wiki \\
        --chunker fixed:128 --scorer bm25 \\
        --conditions sequential,graphkv-top3,graphkv-top3:targets,noedge-top3:targets \\
        --metric both --num-examples 200 --output-dir runs/exp

Pipeline per example: the documents are joined into one context text (the
official ascending-score order), the chunker cuts it into nodes, the scorer
ranks them, and every condition builds its KV context and answers.

* ``--metric generate`` greedy answers scored with the task metric (as the paper);
  ``nll`` teacher-forced NLL of the gold answer (one forward pass, ~5-20x cheaper,
  for sweeps); ``both``.
* ``--time-budget-min`` stops cleanly before the budget is spent (rerun = resume);
  after ``--warmup`` examples the projected time and compute units are printed.
* For ``*-top<m>`` conditions every record notes whether the gold answer occurs
  in a source, a target, both or neither (the drop-source ablation reads this).

``--data-file`` takes any JSONL with ``question``, ``answers`` and either
``context`` (one text) or ``documents`` (strings or {title, text, score}).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from typing import Dict, List, Optional

import torch

from .attention import chunk_totals, query_attention_mass
from .chunking import make_spans, parse_chunker
from .data import RAGExample, load_rag
from .methods import Prompt, answer_nll, decode, parse_method, prepare
from .metrics import group_scores, score_example
from .model import RopeDecoder, Segment
from .prompts import RAG_INSTRUCTION, PromptBuilder, TextPrompt, rag_blocks, rag_question
from .run_rag import load_model, select, stop_ids
from .selection import ascending_order, contains_answer, score_chunks

DEFAULT_CONDITIONS = "sequential,parallel,graphkv-top3,graphkv-top3:targets,noedge-top3:targets"


# ----------------------------------------------------------------------------- data


def load_jsonl_examples(path: str) -> List[RAGExample]:
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            r = json.loads(line)
            docs = r.get("documents")
            if docs is None:
                docs = [r["context"]]
            out.append(RAGExample(i, r["question"], r["answers"], docs, str(r.get("group", "all"))))
    return out


def answer_list(answers) -> List[str]:
    if isinstance(answers, str):
        return [answers]
    flat = []
    for a in answers:
        flat.extend(a if isinstance(a, (list, tuple)) else [a])
    return [str(a) for a in flat]


def build_context(ex: RAGExample, task: Optional[str]):
    """Context text, document spans and document scores (ascending relevance, as the official prompts)."""
    if task is not None:
        blocks = rag_blocks(ex.documents, task)
        if task in ("hqa", "2wiki", "tqa", "nqa", "multihopqa"):
            scores = sorted(d["score"] for d in ex.documents)
        else:
            scores = list(range(len(blocks)))
    else:
        blocks, scores = [], []
        for i, d in enumerate(ex.documents):
            if isinstance(d, dict):
                text = (f"- Title: {d['title']}\n" if d.get("title") else "") + d["text"]
                scores.append(d.get("score", i))
            else:
                text = d
                scores.append(i)
            blocks.append(text if text.endswith("\n") else text + "\n")
    spans, cursor = [], 0
    for b in blocks:
        spans.append((cursor, cursor + len(b)))
        cursor += len(b)
    return "".join(blocks), spans, scores


def question_block(question: str, task: Optional[str]) -> str:
    return rag_question(question, task if task is not None else "nqa")


# ----------------------------------------------------------------------------- helpers


def safe_name(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", label.replace(":", "__").replace("@", "_at_"))


def top_m_of(label: str) -> Optional[int]:
    m = re.search(r"-top(\d+)", label)
    return int(m.group(1)) if m else None


def answer_location(chunks: List[str], order: List[int], m: int, answers: List[str]) -> str:
    top = set(order[-m:])
    in_src = any(contains_answer(chunks[i], answers) for i in top)
    in_tgt = any(contains_answer(chunks[i], answers) for i in range(len(chunks)) if i not in top)
    return {(True, True): "both", (True, False): "source_only", (False, True): "target_only", (False, False): "none"}[(in_src, in_tgt)]


@torch.no_grad()
def attention_scores(dec: RopeDecoder, prompt: Prompt, pad_id: int):
    """LLM relevance: question attention over independently encoded chunks (shared positions [P, P+len))."""
    p = len(prompt.prefix)
    segs = dec.encode_batch(prompt.chunks, [p] * len(prompt.chunks), pad_id=pad_id)
    parts = list(segs)
    if p:
        _, prefix = dec.encode(torch.as_tensor([prompt.prefix]), torch.arange(p).unsqueeze(0))
        parts = [prefix] + parts
    ctx = Segment.cat(parts)
    mass = query_attention_mass(dec, ctx, prompt.query, p + max(len(c) for c in prompt.chunks))
    scores = chunk_totals(mass, [len(c) for c in prompt.chunks], offset=p)
    return scores, {i: s for i, s in enumerate(segs)}


def paired_bootstrap(a: List[float], b: List[float], reps: int = 2000, seed: int = 0):
    """Mean of a - b with a 95% bootstrap interval (paired over examples)."""
    diffs = [x - y for x, y in zip(a, b)]
    if not diffs:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(reps))
    return sum(diffs) / len(diffs), means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


# ----------------------------------------------------------------------------- main


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", choices=["2wiki", "hqa", "tqa", "nqa", "multihopqa", "morehopqa"], default=None)
    ap.add_argument("--data-file", default=None, help="JSONL with question / answers / context or documents")
    ap.add_argument("--chunker", default="documents")
    ap.add_argument("--scorer", default="retriever")
    ap.add_argument("--conditions", default=DEFAULT_CONDITIONS)
    ap.add_argument("--metric", choices=["generate", "nll", "both"], default="both")
    ap.add_argument("--num-examples", type=int, default=100)
    ap.add_argument("--sample", choices=["first", "random", "stratified"], default="stratified")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--prompt-style", choices=["auto", "tulu", "chat"], default="auto")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--probe-layers", default=None, help="comma-separated layers for attention chunking/scoring")
    ap.add_argument("--time-budget-min", type=float, default=None, help="stop cleanly after this many minutes")
    ap.add_argument("--cu-per-hour", type=float, default=None, help="Colab compute units per hour, for the cost projection")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)
    if (args.task is None) == (args.data_file is None):
        ap.error("give exactly one of --task and --data-file")

    t_start = time.time()
    examples = load_rag(args.task, args.data_dir) if args.task else load_jsonl_examples(args.data_file)
    examples = select(examples, args.num_examples, args.sample, args.seed)
    tok, model = load_model(args.model, args.dtype, args.device, args.threads)
    dec = RopeDecoder(model)
    builder = PromptBuilder(tok, args.prompt_style)
    stops = stop_ids(tok, model)
    pad = tok.pad_token_id if tok.pad_token_id is not None else (stops[0] if stops else 0)
    chunker = parse_chunker(args.chunker)
    labels = [c.strip() for c in args.conditions.split(",") if c.strip()]
    methods = {c: parse_method(c, bos_id=tok.bos_token_id) for c in labels}
    layers = [int(x) for x in args.probe_layers.split(",")] if args.probe_layers else None
    os.makedirs(args.output_dir, exist_ok=True)
    config = dict(vars(args), prompt_style_resolved=builder.style, n_selected=len(examples), torch=torch.__version__)
    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=1)

    paths = {c: os.path.join(args.output_dir, safe_name(c) + ".jsonl") for c in labels}
    done: Dict[str, set] = {}
    for c, path in paths.items():
        done[c] = set()
        if os.path.exists(path):
            with open(path) as f:
                done[c] = {json.loads(line)["index"] for line in f if line.strip()}

    print(f"loaded model and {len(examples)} examples in {time.time() - t_start:.0f}s", flush=True)
    t_loop, n_new = time.time(), 0
    for k, ex in enumerate(examples):
        todo = [c for c in labels if ex.index not in done[c]]
        if not todo:
            continue
        if args.time_budget_min is not None and (time.time() - t_start) / 60 > args.time_budget_min:
            print(f"time budget of {args.time_budget_min} min reached after {k} examples; rerun the same command to resume", flush=True)
            break
        answers = answer_list(ex.answers)
        text, doc_spans, doc_scores = build_context(ex, args.task)
        spans = make_spans(text, chunker, tok, doc_spans=doc_spans, decoder=dec, probe_layers=layers)
        chunks = [text[a:b] for a, b in spans]
        prefix, query = builder.wrap(RAG_INSTRUCTION, question_block(ex.question, args.task))
        prompt = builder.tokenize(TextPrompt(prefix, chunks, query))
        cache = None
        if args.scorer == "attention":
            scores, cache = attention_scores(dec, prompt, pad)
        else:
            scores = score_chunks(args.scorer, chunks, ex.question, chunk_spans=spans, doc_spans=doc_spans, doc_scores=doc_scores, answers=answers, seed=args.seed + ex.index)
        order = ascending_order(scores)
        prompt.order = order
        answer_ids = builder.encode(answers[0]) if args.metric != "generate" else None
        for c in todo:
            method = methods[c]
            t0 = time.perf_counter()
            prepared = prepare(dec, prompt, method, pad_id=pad, level0_cache=cache)
            prep_s = time.perf_counter() - t0
            rec = dict(index=ex.index, group=ex.group, condition=c, chunker=chunker.name, scorer=args.scorer,
                       n_chunks=len(chunks), mean_chunk_tokens=sum(map(len, prompt.chunks)) / len(chunks),
                       context_tokens=len(prepared.segment) if prepared.segment is not None else 0,
                       position_span=prepared.position_span, prepare_s=round(prep_s, 4))
            m = top_m_of(c)
            if m is not None:
                rec["answer_location"] = answer_location(chunks, order, min(m, len(chunks)), answers)
            if args.metric in ("nll", "both"):
                rec["nll"] = answer_nll(dec, prepared, answer_ids)
            if args.metric in ("generate", "both"):
                tokens, first_s, decode_s = decode(dec, prepared, max_new_tokens=args.max_new_tokens, stop_ids=stops)
                rec["generated"] = tok.decode(tokens, skip_special_tokens=True)
                rec["score"] = score_example(args.task or "nqa", rec["generated"], ex.answers)
                rec.update(new_tokens=len(tokens), first_token_s=round(first_s, 4), decode_s=round(decode_s, 4))
            with open(paths[c], "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_new += 1
        if n_new == args.warmup or (n_new % 25 == 0):
            per = (time.time() - t_loop) / n_new
            left = sum(1 for e in examples[k + 1 :] if any(e.index not in done[c] for c in labels))
            msg = f"[{n_new} new examples] {per:.1f}s/example, about {per * left / 60:.0f} min left"
            if args.cu_per_hour:
                msg += f" (~{per * left / 3600 * args.cu_per_hour:.1f} compute units)"
            print(msg, flush=True)
    write_summary(args.output_dir, labels)


def _rows(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_summary(output_dir: str, labels: List[str]) -> dict:
    rows = {c: _rows(os.path.join(output_dir, safe_name(c) + ".jsonl")) for c in labels}
    common = set.intersection(*[{r["index"] for r in rs} for rs in rows.values()]) if rows and all(rows.values()) else set()
    summary: Dict[str, dict] = {}
    for c, rs in rows.items():
        rs = [r for r in rs if r["index"] in common]
        if not rs:
            continue
        s = dict(n=len(rs), n_chunks=sum(r["n_chunks"] for r in rs) / len(rs), mean_chunk_tokens=sum(r["mean_chunk_tokens"] for r in rs) / len(rs),
                 context_tokens=sum(r["context_tokens"] for r in rs) / len(rs), position_span=sum(r["position_span"] for r in rs) / len(rs))
        if "score" in rs[0]:
            s["accuracy"] = group_scores([r["score"] for r in rs], [r["group"] for r in rs])
        if "nll" in rs[0]:
            s["nll"] = sum(r["nll"] for r in rs) / len(rs)
        if "answer_location" in rs[0]:
            by = {}
            for loc in ("source_only", "target_only", "both", "none"):
                sub = [r for r in rs if r["answer_location"] == loc]
                if sub:
                    by[loc] = dict(n=len(sub), accuracy=100 * sum(r.get("score", float("nan")) for r in sub) / len(sub) if "score" in sub[0] else None,
                                   nll=sum(r["nll"] for r in sub) / len(sub) if "nll" in sub[0] else None)
            s["by_answer_location"] = by
        summary[c] = s
    # paired differences against the first condition, and the source->target transfer test
    comparisons = []
    by_index = {c: {r["index"]: r for r in rs if r["index"] in common} for c, rs in rows.items()}
    idx = sorted(common)
    pairs = [(c, labels[0]) for c in labels[1:]]
    for c in labels:
        if c.startswith("graphkv-top") and c.endswith(":targets"):
            control = "noedge-" + c[len("graphkv-"):]
            if control in by_index:
                pairs.append((c, control))
    for a, b in pairs:
        for key in ("score", "nll"):
            if idx and key in by_index[a][idx[0]]:
                scale = 100.0 if key == "score" else 1.0
                mean, lo, hi = paired_bootstrap([by_index[a][i][key] * scale for i in idx], [by_index[b][i][key] * scale for i in idx])
                comparisons.append(dict(a=a, b=b, metric=key, diff=mean, ci_low=lo, ci_high=hi, n=len(idx)))
    out = dict(conditions=summary, comparisons=comparisons, n_common=len(common))
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    lines = [f"examples answered by every condition: {len(common)}", "",
             "| condition | n | accuracy | NLL | chunks | tokens/chunk | context KV | position span |", "|---|---|---|---|---|---|---|---|"]
    for c, s in summary.items():
        acc = f"{s['accuracy']['all']:.1f}" if "accuracy" in s else ""
        nll = f"{s['nll']:.3f}" if "nll" in s else ""
        lines.append(f"| {c} | {s['n']} | {acc} | {nll} | {s['n_chunks']:.1f} | {s['mean_chunk_tokens']:.0f} | {s['context_tokens']:.0f} | {s['position_span']:.0f} |")
    loc_rows = [(c, s["by_answer_location"]) for c, s in summary.items() if "by_answer_location" in s]
    if loc_rows:
        lines += ["", "accuracy / NLL by where the gold answer occurs (top-m conditions)", "",
                  "| condition | source_only | target_only | both | none |", "|---|---|---|---|---|"]
        for c, by in loc_rows:
            cells = []
            for loc in ("source_only", "target_only", "both", "none"):
                v = by.get(loc)
                if not v:
                    cells.append("")
                    continue
                acc = f"{v['accuracy']:.1f}" if v["accuracy"] is not None else "-"
                nll = f"{v['nll']:.2f}" if v["nll"] is not None else "-"
                cells.append(f"{acc} / {nll} (n={v['n']})")
            lines.append(f"| {c} | " + " | ".join(cells) + " |")
    if comparisons:
        lines += ["", "paired differences (a - b) with 95% bootstrap intervals; accuracy in points, NLL in nats/token (lower is better)", "",
                  "| a | b | metric | diff | 95% CI |", "|---|---|---|---|---|"]
        for d in comparisons:
            lines.append(f"| {d['a']} | {d['b']} | {d['metric']} | {d['diff']:+.3f} | [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}] |")
    with open(os.path.join(output_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return out


if __name__ == "__main__":
    main()
