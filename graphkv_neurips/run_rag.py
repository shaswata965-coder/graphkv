"""RAG experiments of Sec. 4.1 (Tables 2, 3, 5, 6) with any supported model.

    python -m graphkv_neurips.run_rag --model Qwen/Qwen2.5-0.5B-Instruct --task 2wiki \\
        --methods sequential,pcw,ape,block,graphkv-top1,graphkv-top3,graphkv-full \\
        --num-examples 100 --output-dir results/neurips/2wiki-qwen0.5b

Every method answers the same examples with greedy decoding (at most 256 new
tokens, Appendix A.2.1). Results are appended to ``<output-dir>/<method>.jsonl``
and a rerun skips finished examples. ``summary.md`` / ``summary.json`` hold
accuracy per question type. ``--order`` changes where the chunks are placed
(not which chunks are the top-m), to measure positional bias.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Dict, List, Optional

import torch

from .data import load_rag
from .methods import generate, parse_method
from .metrics import group_scores, score_example
from .model import RopeDecoder
from .prompts import PromptBuilder, TextPrompt

DEFAULT_METHODS = "sequential,pcw,ape,block,graphkv-top1,graphkv-top3,graphkv-full"


def resolve_dtype(dtype: str, device: str) -> torch.dtype:
    """``auto``: bfloat16 where it is fast (CPU with AVX512-BF16/AMX, Ampere+ GPUs), float16 on older GPUs (T4)."""
    if dtype == "auto":
        if device.startswith("cuda"):
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.bfloat16
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]


def load_model(name: str, dtype: str = "auto", device: str = "cpu", threads: Optional[int] = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if threads:
        torch.set_num_threads(threads)
    torch_dtype = resolve_dtype(dtype, device)
    tok = AutoTokenizer.from_pretrained(name)
    kwargs = {}
    if device.startswith("cuda"):
        try:
            import accelerate  # noqa: F401  - lets from_pretrained place the weights directly on the GPU

            kwargs["device_map"] = device
        except ImportError:
            pass
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch_dtype, **kwargs)
    except TypeError:  # transformers < 4.56
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch_dtype, **kwargs)
    model = model.to(device).eval()
    return tok, model


def stop_ids(tokenizer, model) -> List[int]:
    ids = set()
    for src in (tokenizer.eos_token_id, getattr(model.generation_config, "eos_token_id", None)):
        if isinstance(src, int):
            ids.add(src)
        elif src:
            ids.update(int(s) for s in src)
    return sorted(ids)


def reorder(tp: TextPrompt, mode: str, seed: int) -> tuple:
    """Place the chunks (ascending score in the data) in another order; returns the prompt and the relevance order."""
    n = len(tp.chunks)
    if mode == "ascent":
        perm = list(range(n))
    elif mode == "descent":
        perm = list(range(n - 1, -1, -1))
    elif mode == "random":
        perm = list(range(n))
        random.Random(seed).shuffle(perm)
    else:
        raise ValueError("order must be ascent, descent or random")
    chunks = [tp.chunks[i] for i in perm]
    order = sorted(range(n), key=lambda pos: perm[pos])  # positions sorted by ascending score
    return TextPrompt(tp.prefix, chunks, tp.query), order


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=["2wiki", "hqa", "tqa", "nqa", "multihopqa", "morehopqa"])
    ap.add_argument("--methods", default=DEFAULT_METHODS)
    ap.add_argument("--num-examples", type=int, default=100)
    ap.add_argument("--sample", choices=["first", "random", "stratified"], default="stratified",
                    help="which examples: the first N, N random ones, or N spread evenly over the question types")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--order", choices=["ascent", "descent", "random"], default="ascent")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--prompt-style", choices=["auto", "tulu", "chat"], default="auto")
    ap.add_argument("--data-dir", default=None, help="local copy of the Graph-COM/GraphKV dataset (default: download)")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)

    examples = load_rag(args.task, args.data_dir)
    examples = select(examples, args.num_examples, args.sample, args.seed)
    tok, model = load_model(args.model, args.dtype, args.device, args.threads)
    dec = RopeDecoder(model)
    builder = PromptBuilder(tok, args.prompt_style)
    stops = stop_ids(tok, model)
    pad = tok.pad_token_id if tok.pad_token_id is not None else (stops[0] if stops else 0)
    methods = [parse_method(m.strip(), bos_id=tok.bos_token_id) for m in args.methods.split(",") if m.strip()]
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump(dict(vars(args), prompt_style_resolved=builder.style, n_selected=len(examples), torch=torch.__version__), f, indent=1)

    for method in methods:
        path = os.path.join(args.output_dir, f"{method.label}.jsonl")
        done = set()
        if os.path.exists(path):
            with open(path) as f:
                done = {json.loads(line)["index"] for line in f if line.strip()}
        t_method = time.time()
        for k, ex in enumerate(examples):
            if ex.index in done:
                continue
            tp = builder.rag_text(ex.question, ex.documents, args.task)
            tp, order = reorder(tp, args.order, args.seed + ex.index)
            prompt = builder.tokenize(tp, order)
            gen = generate(dec, prompt, method, max_new_tokens=args.max_new_tokens, stop_ids=stops, pad_id=pad)
            text = tok.decode(gen.token_ids, skip_special_tokens=True)
            score = score_example(args.task, text, ex.answers)
            rec = dict(
                index=ex.index, group=ex.group, question=ex.question, answers=ex.answers, generated=text, score=score,
                new_tokens=len(gen.token_ids), context_tokens=gen.context_tokens, position_span=gen.position_span,
                attention_pairs=gen.attention_pairs, prepare_s=round(gen.prepare_s, 4), first_token_s=round(gen.first_token_s, 4),
                decode_s=round(gen.decode_s, 4),
            )
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if k % 10 == 0:
                print(f"[{method.label}] {k + 1}/{len(examples)} acc so far: {summarize_file(path)['all']:.1f} ({time.time() - t_method:.0f}s)", flush=True)
    write_summary(args.output_dir, [m.label for m in methods])


def select(examples, n: int, mode: str, seed: int):
    if n <= 0 or n >= len(examples):
        return examples
    if mode == "first":
        return examples[:n]
    rng = random.Random(seed)
    if mode == "random":
        return sorted(rng.sample(examples, n), key=lambda e: e.index)
    groups: Dict[str, list] = {}
    for e in examples:
        groups.setdefault(e.group, []).append(e)
    per = {g: n * len(v) // len(examples) for g, v in groups.items()}
    rest = n - sum(per.values())
    for g in sorted(groups, key=lambda g: -len(groups[g]))[:rest]:
        per[g] += 1
    picked = []
    for g, v in groups.items():
        picked += rng.sample(v, min(per[g], len(v)))
    return sorted(picked, key=lambda e: e.index)


def summarize_file(path: str) -> Dict[str, float]:
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return group_scores([r["score"] for r in rows], [r["group"] for r in rows])


def write_summary(output_dir: str, labels: List[str]) -> dict:
    summary = {}
    for label in labels:
        path = os.path.join(output_dir, f"{label}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        acc = group_scores([r["score"] for r in rows], [r["group"] for r in rows])
        mean = lambda key: sum(r[key] for r in rows) / len(rows)  # noqa: E731
        summary[label] = dict(
            n=len(rows), accuracy=acc, context_tokens=mean("context_tokens"), position_span=mean("position_span"),
            attention_pairs=mean("attention_pairs"), ttft_s=mean("prepare_s") + mean("first_token_s"),
            prepare_s=mean("prepare_s"), new_tokens=mean("new_tokens"),
        )
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    groups = sorted({g for s in summary.values() for g in s["accuracy"] if g != "all"})
    lines = ["| method | n | " + " | ".join(groups + ["all"]) + " | position span | context KV | TTFT (s) |", "|---" * (len(groups) + 6) + "|"]
    for label, s in summary.items():
        cells = [f"{s['accuracy'].get(g, float('nan')):.1f}" for g in groups + ["all"]]
        lines.append(f"| {label} | {s['n']} | " + " | ".join(cells) + f" | {s['position_span']:.0f} | {s['context_tokens']:.0f} | {s['ttft_s']:.2f} |")
    with open(os.path.join(output_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return summary


if __name__ == "__main__":
    main()
