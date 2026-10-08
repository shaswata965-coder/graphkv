"""Run the paper's experiments: baseline vs. graph-based compression over (epsilon, M).

Example (quick smoke run on CPU)::

    python -m graphkv.run --model gpt2 --num-prompts 5 --epsilons 5,10,20 --intervals 16

Full GPT-2 sweep as in the paper (Sec. II-B)::

    python -m graphkv.run --model gpt2 --num-prompts 100 \\
        --epsilons 0.5,1,2,5,10,15,20,25,30,35,40 --intervals 8,16,32

Outputs in ``--output-dir``: ``summary.csv`` (one row per configuration),
``per_prompt.jsonl`` (one row per prompt, configuration and protocol),
``prompts.jsonl`` and ``run_config.json``. Re-running with the same
``--output-dir`` skips configurations that are already in ``summary.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .cache import GRANULARITIES, GraphKVConfig
from .data import load_prompts
from .engine import DecodeResult, GraphKVEngine, sequence_nll
from .models import make_adapter
from .utils import environment_info, load_model_and_tokenizer, resolve_device, resolve_dtype, set_seed

PAPER_EPSILONS = "0.5,1,2,5,10,15,20,25,30,35,40"
PAPER_INTERVALS = "8,16,32"

SUMMARY_FIELDS = [
    "config_id",
    "method",
    "epsilon",
    "interval",
    "num_sink",
    "max_iters",
    "granularity",
    "weighted_merge",
    "proportional_attention",
    "num_prompts",
    "tf_ppl",
    "tf_tokens",
    "seq_ppl",
    "oracle_ppl",
    "gen_ppl",
    "prefix_match",
    "max_compression_pct",
    "final_compression_pct",
    "decode_tok_s",
    "e2e_tok_s",
    "latency_s",
    "ttft_s",
    "per_token_ms",
    "compress_ms_per_call",
    "compressions_per_prompt",
    "kv_mb_peak",
    "kv_mb_final",
    "tf_max_compression_pct",
    "wall_s",
]


@dataclass
class PromptItem:
    text: str
    ids: List[int]
    tf_prompt: List[int]
    tf_target: List[int]


def parse_floats(text: str) -> List[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def parse_ints(text: str) -> List[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def parse_iters(text: str) -> Optional[int]:
    return None if text.lower() in ("none", "inf", "converge") else int(text)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("model")
    g.add_argument("--model", default="gpt2", help="HF Hub id or local directory (e.g. a fine-tuned checkpoint)")
    g.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N | mps")
    g.add_argument("--dtype", default="float32", help="float32 | float16 | bfloat16")
    g.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: torch's choice)")

    g = p.add_argument_group("graph-based compression")
    g.add_argument("--epsilons", default=PAPER_EPSILONS, help="comma-separated distance thresholds")
    g.add_argument("--intervals", default=PAPER_INTERVALS, help="comma-separated compression intervals M")
    g.add_argument("--num-sink", type=int, default=4, help="attention-sink tokens kept uncompressed")
    g.add_argument("--max-iters", type=parse_iters, default=None, help="label-propagation iterations T (default: until convergence)")
    g.add_argument("--granularity", choices=GRANULARITIES, default="head", help="cluster per KV head (paper) or per token")
    g.add_argument("--weighted-merge", action="store_true", help="extension: size-weighted centroids")
    g.add_argument("--proportional-attention", action="store_true", help="extension: +log(cluster size) attention bias")
    g.add_argument("--no-baseline", action="store_true", help="skip the standard-cache baseline")

    g = p.add_argument_group("data")
    g.add_argument("--num-prompts", type=int, default=100)
    g.add_argument("--min-chars", type=int, default=200, help="keep entries longer than this")
    g.add_argument("--max-words", type=int, default=200, help="truncate entries to this many words")
    g.add_argument("--prompts-file", default=None, help=".txt (one prompt per line) or .jsonl ({'text': ...}) instead of WikiText-2")
    g.add_argument("--min-context", type=int, default=32, help="minimum prompt tokens kept before the teacher-forced reference")

    g = p.add_argument_group("decoding")
    g.add_argument("--max-new-tokens", type=int, default=50)
    g.add_argument("--do-sample", action="store_true", help="sample instead of greedy decoding")
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--top-k", type=int, default=0)
    g.add_argument("--top-p", type=float, default=1.0)
    g.add_argument("--ignore-eos", action="store_true", help="always generate --max-new-tokens tokens")
    g.add_argument("--skip-generation", action="store_true", help="only run the teacher-forced perplexity protocol")
    g.add_argument("--skip-teacher-forced", action="store_true", help="only run the generation protocol")
    g.add_argument("--skip-oracle", action="store_true", help="do not score generations with the uncompressed model")
    g.add_argument("--warmup", type=int, default=1, help="untimed warm-up generations per configuration")

    g = p.add_argument_group("output")
    g.add_argument("--output-dir", default=None, help="default: results/<model>-<timestamp>")
    g.add_argument("--seed", type=int, default=42)
    return p


def prepare_prompts(tokenizer, texts: List[str], args, max_positions: Optional[int]) -> List[PromptItem]:
    items = []
    limit = None if max_positions is None else max_positions - args.max_new_tokens
    truncated = 0
    for text in texts:
        ids = tokenizer(text)["input_ids"]
        if limit is not None and len(ids) > limit:
            ids, truncated = ids[:limit], truncated + 1
        ref_len = min(args.max_new_tokens, len(ids) - args.min_context)
        tf_prompt, tf_target = (ids[:-ref_len], ids[-ref_len:]) if ref_len >= 1 else ([], [])
        items.append(PromptItem(text, ids, tf_prompt, tf_target))
    if truncated:
        print(f"note: {truncated} prompts truncated to {limit} tokens to fit the model's position limit")
    skipped = sum(1 for it in items if not it.tf_target)
    if skipped and not args.skip_teacher_forced:
        print(f"note: {skipped} prompts are too short for the teacher-forced protocol (< {args.min_context + 1} tokens)")
    return items


def config_list(args) -> List[Optional[GraphKVConfig]]:
    configs: List[Optional[GraphKVConfig]] = [] if args.no_baseline else [None]
    for interval in parse_ints(args.intervals):
        for eps in parse_floats(args.epsilons):
            configs.append(
                GraphKVConfig(
                    epsilon=eps,
                    interval=interval,
                    num_sink=args.num_sink,
                    max_iters=args.max_iters,
                    granularity=args.granularity,
                    weighted_merge=args.weighted_merge,
                    proportional_attention=args.proportional_attention,
                )
            )
    return configs


def config_id(cfg: Optional[GraphKVConfig]) -> str:
    if cfg is None:
        return "baseline"
    t = "conv" if cfg.max_iters is None else str(cfg.max_iters)
    ext = ("-w" if cfg.weighted_merge else "") + ("-pa" if cfg.proportional_attention else "")
    return f"graphkv-e{cfg.epsilon:g}-M{cfg.interval}-s{cfg.num_sink}-T{t}-{cfg.granularity}{ext}"


def result_row(res: DecodeResult) -> dict:
    return {
        "prompt_len": res.prompt_len,
        "num_tokens": res.num_tokens,
        "nll_sum": res.nll_sum,
        "ppl": res.perplexity,
        "prefill_s": res.prefill_s,
        "decode_s": res.decode_s,
        "total_s": res.total_s,
        "compress_s": res.compress_s,
        "num_compressions": len(res.events),
        "max_compression": res.max_compression,
        "final_compression": res.final_compression,
        "peak_kv_bytes": res.peak_kv_bytes,
        "final_kv_bytes": res.final_kv_bytes,
        "final_kv_bytes_allocated": res.final_kv_bytes_allocated,
        "final_layer_lengths": [round(x, 2) for x in res.final_layer_lengths],
    }


def prefix_match(tokens: List[int], reference: Optional[List[int]]) -> Optional[float]:
    if not reference:
        return None
    n = 0
    for a, b in zip(tokens, reference):
        if a != b:
            break
        n += 1
    return n / len(reference)


def mean(xs) -> float:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(cfg: Optional[GraphKVConfig], gen_rows: List[dict], tf_rows: List[dict], wall_s: float) -> dict:
    row = {
        "config_id": config_id(cfg),
        "method": "baseline" if cfg is None else "graphkv",
        "epsilon": "" if cfg is None else cfg.epsilon,
        "interval": "" if cfg is None else cfg.interval,
        "num_sink": "" if cfg is None else cfg.num_sink,
        "max_iters": "" if cfg is None else ("conv" if cfg.max_iters is None else cfg.max_iters),
        "granularity": "" if cfg is None else cfg.granularity,
        "weighted_merge": "" if cfg is None else cfg.weighted_merge,
        "proportional_attention": "" if cfg is None else cfg.proportional_attention,
        "num_prompts": max(len(gen_rows), len(tf_rows)),
        "wall_s": round(wall_s, 2),
    }
    if tf_rows:
        tokens = sum(r["num_tokens"] for r in tf_rows)
        row["tf_ppl"] = math.exp(sum(r["nll_sum"] for r in tf_rows) / tokens)
        row["tf_tokens"] = tokens
        row["tf_max_compression_pct"] = 100 * mean(r["max_compression"] for r in tf_rows)
    if gen_rows:
        tokens = sum(r["num_tokens"] for r in gen_rows)
        calls = sum(r["num_compressions"] for r in gen_rows)
        row.update(
            gen_ppl=math.exp(sum(r["nll_sum"] for r in gen_rows) / tokens),
            prefix_match=mean(r.get("prefix_match") for r in gen_rows),
            max_compression_pct=100 * mean(r["max_compression"] for r in gen_rows),
            final_compression_pct=100 * mean(r["final_compression"] for r in gen_rows),
            decode_tok_s=tokens / sum(r["decode_s"] for r in gen_rows),
            e2e_tok_s=tokens / sum(r["total_s"] for r in gen_rows),
            latency_s=mean(r["total_s"] for r in gen_rows),
            ttft_s=mean(r["prefill_s"] for r in gen_rows),
            per_token_ms=1000 * sum(r["decode_s"] for r in gen_rows) / tokens,
            compress_ms_per_call=1000 * sum(r["compress_s"] for r in gen_rows) / calls if calls else 0.0,
            compressions_per_prompt=calls / len(gen_rows),
            kv_mb_peak=mean(r["peak_kv_bytes"] for r in gen_rows) / 2**20,
            kv_mb_final=mean(r["final_kv_bytes"] for r in gen_rows) / 2**20,
        )
        if all("oracle_nll_sum" in r for r in gen_rows):
            row["oracle_ppl"] = math.exp(sum(r["oracle_nll_sum"] for r in gen_rows) / tokens)
            seq_tokens = sum(r["seq_tokens"] for r in gen_rows)
            row["seq_ppl"] = math.exp(sum(r["seq_nll_sum"] for r in gen_rows) / seq_tokens)
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}


def load_done(out: Path) -> Dict[str, dict]:
    path = out / "summary.csv"
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {row["config_id"]: row for row in csv.DictReader(f)}


def load_baseline_tokens(out: Path) -> Dict[int, List[int]]:
    path = out / "per_prompt.jsonl"
    tokens: Dict[int, List[int]] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["config_id"] == "baseline" and row["protocol"] == "generate":
                tokens[row["prompt_idx"]] = row["tokens"]
    return tokens


def append_summary(out: Path, row: dict) -> None:
    path = out / "summary.csv"
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)


def fmt(row: dict, key: str, spec: str) -> str:
    v = row.get(key)
    return "-" if v is None or v == "" or (isinstance(v, float) and math.isnan(v)) else format(v, spec)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.skip_generation and args.skip_teacher_forced:
        print("nothing to do: both protocols skipped", file=sys.stderr)
        return 2
    if args.threads:
        torch.set_num_threads(args.threads)
    set_seed(args.seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    out = Path(args.output_dir or f"results/{Path(args.model).name}-{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.model} on {device} ({dtype}) ...")
    model, tokenizer = load_model_and_tokenizer(args.model, device, dtype)
    adapter = make_adapter(model)
    print(
        f"  {model.config.model_type}: {adapter.num_layers} layers, {adapter.num_kv_heads} KV heads x {adapter.head_dim} dims"
    )

    texts = load_prompts(args.num_prompts, args.min_chars, args.max_words, args.prompts_file)
    items = prepare_prompts(tokenizer, texts, args, adapter.max_positions)
    print(f"  {len(items)} prompts, mean length {mean(len(it.ids) for it in items):.0f} tokens")

    run_config = {"args": vars(args), "environment": environment_info(device), "model_type": model.config.model_type}
    (out / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str))
    with (out / "prompts.jsonl").open("w") as f:
        for i, it in enumerate(items):
            f.write(json.dumps({"prompt_idx": i, "text": it.text, "num_tokens": len(it.ids)}) + "\n")

    stop_ids = [] if args.ignore_eos or tokenizer.eos_token_id is None else [tokenizer.eos_token_id]
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        stop_token_ids=stop_ids,
    )

    done = load_done(out)
    baseline_tokens = load_baseline_tokens(out)
    configs = config_list(args)
    print(f"{len(configs)} configurations ({sum(config_id(c) in done for c in configs)} already done) -> {out}\n")
    header = f"{'config':42s} {'tf_ppl':>8s} {'seq_ppl':>8s} {'comp%':>6s} {'tok/s':>8s} {'lat(s)':>7s} {'match':>6s}"
    print(header + "\n" + "-" * len(header))

    for cfg in configs:
        cid = config_id(cfg)
        if cid in done:
            continue
        engine = GraphKVEngine(adapter, cfg)
        for _ in range(args.warmup):
            engine.generate(items[0].ids, **gen_kwargs)
        gen_rows, tf_rows = [], []
        t_start = time.perf_counter()
        for idx, it in enumerate(items):
            common = {"config_id": cid, "prompt_idx": idx, "epsilon": None if cfg is None else cfg.epsilon,
                      "interval": None if cfg is None else cfg.interval}
            if not args.skip_generation:
                # Same seed per prompt for every configuration -> identical sampling noise.
                gen = torch.Generator().manual_seed(args.seed + idx)
                res = engine.generate(it.ids, generator=gen, **gen_kwargs)
                row = {**common, "protocol": "generate", **result_row(res), "tokens": res.tokens}
                if cfg is None:
                    baseline_tokens[idx] = res.tokens
                row["prefix_match"] = prefix_match(res.tokens, baseline_tokens.get(idx))
                if not args.skip_oracle:
                    # Untimed: the uncompressed model scores prompt + generated text.
                    nll = sequence_nll(adapter, it.ids + res.tokens)
                    row.update(oracle_nll_sum=sum(nll[len(it.ids) - 1 :]), seq_nll_sum=sum(nll), seq_tokens=len(nll))
                gen_rows.append(row)
            if not args.skip_teacher_forced and it.tf_target:
                res = engine.score(it.tf_prompt, it.tf_target)
                tf_rows.append({**common, "protocol": "teacher_forced", **result_row(res)})
        summary = summarize(cfg, gen_rows, tf_rows, time.perf_counter() - t_start)
        # Written only once a configuration is complete, so an interrupted run resumes cleanly.
        with (out / "per_prompt.jsonl").open("a") as log:
            for row in gen_rows + tf_rows:
                log.write(json.dumps(row) + "\n")
        append_summary(out, summary)
        print(
            f"{cid:42s} {fmt(summary, 'tf_ppl', '8.2f')} {fmt(summary, 'seq_ppl', '8.2f')} {fmt(summary, 'max_compression_pct', '6.1f')} "
            f"{fmt(summary, 'decode_tok_s', '8.1f')} {fmt(summary, 'latency_s', '7.3f')} {fmt(summary, 'prefix_match', '6.2f')}",
            flush=True,
        )

    print(f"\nwrote {out / 'summary.csv'}\nplot with: python -m graphkv.plot {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
