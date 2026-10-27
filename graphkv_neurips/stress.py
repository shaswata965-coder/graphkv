"""Stress test of Sec. 4.4 / Appendix A.2.4: memory and time-to-first-token on synthetic star graphs.

A star graph has ``k`` neighbour nodes (sources) pointing at one central node
(target); every node holds the same text repeated or truncated to ``words``
words. Sequential encoding reads all nodes and the question as one sequence;
Graph-KV prefills the neighbours independently, then the centre on top of
their KV caches.

* ``memory``: peak resident memory of one prefill + first token, in a fresh
  subprocess per configuration (``/proc/self/clear_refs`` resets the peak
  after the model is loaded), for a growing number of neighbours.
* ``ttft``: wall-clock time to the first answer token with 10 neighbours and a
  growing number of words per node. For Graph-KV both the full time and the
  time with the node KV caches already pre-filled (re-used across queries) are
  reported.

    python -m graphkv_neurips.stress memory --model Qwen/Qwen2.5-0.5B-Instruct --words 500 --neighbors 2,4,8,16
    python -m graphkv_neurips.stress ttft --model Qwen/Qwen2.5-0.5B-Instruct --words 100,200,400,800
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import List

import torch

from .graph import GraphPlan, encode_graph
from .methods import Method, Prepared, Prompt, decode, prepare

DEFAULT_TEXT = (
    "Modern large language models are inherently auto-regressive, requiring input to be serialized into flat "
    "sequences regardless of their structural dependencies. This serialization hinders the model's ability to "
    "leverage structural inductive biases, especially in tasks such as retrieval-augmented generation and reasoning "
    "on data with native graph structures, where inter-segment dependencies are crucial."
)


def node_text(words: int, text: str = DEFAULT_TEXT) -> str:
    base = text.split()
    out = (base * (words // len(base) + 1))[:words]
    return " ".join(out) + "\n"


def star_prompt(tok, neighbors: int, words: int) -> Prompt:
    node = tok(node_text(words), add_special_tokens=False).input_ids
    prefix = tok("You will read several papers, then answer.\n", add_special_tokens=False).input_ids
    query = tok("\nQuestion: what is the central paper about?\nAnswer:", add_special_tokens=False).input_ids
    chunks = [list(node) for _ in range(neighbors + 1)]
    return Prompt(list(prefix), chunks, list(query))


def _rss_peak_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    return float("nan")


def _rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return float("nan")


def _reset_peak() -> None:
    with open("/proc/self/clear_refs", "w") as f:
        f.write("5")


def _child(args) -> None:
    """One configuration in this process; prints a JSON line."""
    from .run_rag import load_model

    tok, model = load_model(args.model, args.dtype, "cpu", args.threads)
    from .model import RopeDecoder

    dec = RopeDecoder(model)
    prompt = star_prompt(tok, args.k, args.w)
    if args.method == "sequential":
        method = Method("sequential")
    else:
        method = Method("graphkv", plan=lambda p: GraphPlan.star(len(p.chunks) - 1))
    import gc

    gc.collect()
    base = _rss_mb()
    _reset_peak()
    t0 = time.perf_counter()
    if args.method == "sequential":
        prepared = prepare(dec, prompt, method)
    else:
        enc = encode_graph(dec, prompt.prefix, prompt.chunks, method.plan(prompt), max_batch_tokens=args.batch_tokens)
        prepared = Prepared(enc.segment, prompt.query, enc.query_start)
    _, first_s, _ = decode(dec, prepared, max_new_tokens=1)
    elapsed = time.perf_counter() - t0
    peak = _rss_peak_mb()
    n_tokens = len(prompt.prefix) + sum(map(len, prompt.chunks)) + len(prompt.query)
    print(json.dumps(dict(method=args.method, neighbors=args.k, words=args.w, tokens=n_tokens, base_mb=round(base, 1), peak_mb=round(peak, 1), extra_mb=round(peak - base, 1), seconds=round(elapsed, 3))))


def run_memory(args) -> List[dict]:
    rows = []
    for words in [int(w) for w in args.words.split(",")]:
        for method in ("sequential", "graphkv"):
            for k in [int(n) for n in args.neighbors.split(",")]:
                cmd = [sys.executable, "-m", "graphkv_neurips.stress", "_child", "--model", args.model, "--dtype", args.dtype,
                       "--method", method, "--k", str(k), "--w", str(words), "--batch-tokens", str(args.batch_tokens)]
                if args.threads:
                    cmd += ["--threads", str(args.threads)]
                out = subprocess.run(cmd, capture_output=True, text=True)
                if out.returncode != 0:
                    rows.append(dict(method=method, neighbors=k, words=words, error=out.stderr.strip().splitlines()[-1:]))
                    print(rows[-1], flush=True)
                    continue
                rows.append(json.loads(out.stdout.strip().splitlines()[-1]))
                print(rows[-1], flush=True)
    return rows


@torch.no_grad()
def run_ttft(args) -> List[dict]:
    from .model import RopeDecoder
    from .run_rag import load_model

    tok, model = load_model(args.model, args.dtype, "cpu", args.threads)
    dec = RopeDecoder(model)
    rows = []
    for words in [int(w) for w in args.words.split(",")]:
        prompt = star_prompt(tok, args.ttft_neighbors, words)
        plan = GraphPlan.star(args.ttft_neighbors)
        for rep in range(args.repeats + 1):  # first repetition warms up
            t0 = time.perf_counter()
            seq = prepare(dec, prompt, Method("sequential"))
            _, first, _ = decode(dec, seq, max_new_tokens=1)
            seq_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            enc = encode_graph(dec, prompt.prefix, prompt.chunks, plan)
            prefill_s = time.perf_counter() - t0
            cached = Prepared(enc.segment, prompt.query, enc.query_start)
            t1 = time.perf_counter()
            _, first_g, _ = decode(dec, cached, max_new_tokens=1)
            query_s = time.perf_counter() - t1
            if rep:
                rows.append(dict(words=words, tokens=len(seq.query_ids), seq_ttft_s=seq_s, graphkv_ttft_s=prefill_s + query_s,
                                 graphkv_prefilled_ttft_s=query_s, graphkv_prefill_s=prefill_s, seq_position_span=len(seq.query_ids),
                                 graphkv_position_span=enc.query_start + len(prompt.query)))
                print(rows[-1], flush=True)
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["memory", "ttft", "_child"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--words", default="500")
    ap.add_argument("--neighbors", default="2,4,8,16")
    ap.add_argument("--ttft-neighbors", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--batch-tokens", type=int, default=16384, help="Graph-KV: padded tokens per prefill batch")
    ap.add_argument("--output", default=None)
    # child-only
    ap.add_argument("--method", default="sequential")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--w", type=int, default=500)
    args = ap.parse_args(argv)
    if args.mode == "_child":
        _child(args)
        return
    rows = run_memory(args) if args.mode == "memory" else run_ttft(args)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
