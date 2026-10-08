"""Side-by-side generation with a standard and a graph-compressed KV cache.

Example::

    python -m graphkv.demo --model gpt2 --epsilon 10 --interval 16
    python -m graphkv.demo --model gpt2 --prompt "The history of the Roman Empire" --max-new-tokens 80
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .cache import GRANULARITIES, GraphKVConfig
from .data import load_prompts
from .engine import GraphKVEngine
from .models import make_adapter
from .run import parse_iters
from .utils import load_model_and_tokenizer, resolve_device, resolve_dtype, set_seed


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="gpt2")
    p.add_argument("--prompt", default=None, help="default: first WikiText-2 evaluation prompt")
    p.add_argument("--epsilon", type=float, default=10.0)
    p.add_argument("--interval", type=int, default=16)
    p.add_argument("--num-sink", type=int, default=4)
    p.add_argument("--max-iters", type=parse_iters, default=None)
    p.add_argument("--granularity", choices=GRANULARITIES, default="head")
    p.add_argument("--weighted-merge", action="store_true", help="extension: size-weighted centroids")
    p.add_argument("--proportional-attention", action="store_true", help="extension: +log(cluster size) attention bias")
    p.add_argument("--max-new-tokens", type=int, default=50)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    set_seed(args.seed)
    device = resolve_device(args.device)
    model, tokenizer = load_model_and_tokenizer(args.model, device, resolve_dtype(args.dtype, device))
    adapter = make_adapter(model)
    prompt = args.prompt or load_prompts(num_prompts=1)[0]
    ids = tokenizer(prompt)["input_ids"]
    cfg = GraphKVConfig(
        epsilon=args.epsilon,
        interval=args.interval,
        num_sink=args.num_sink,
        max_iters=args.max_iters,
        granularity=args.granularity,
        weighted_merge=args.weighted_merge,
        proportional_attention=args.proportional_attention,
    )

    print(f"model: {args.model} on {device} | prompt: {len(ids)} tokens\n")
    print("PROMPT:", prompt[:400] + (" ..." if len(prompt) > 400 else ""), "\n")
    for name, engine in (("standard cache", GraphKVEngine(adapter)), (f"graph-compressed ({cfg})", GraphKVEngine(adapter, cfg))):
        res = engine.generate(ids, args.max_new_tokens)
        print(f"=== {name}")
        print(tokenizer.decode(res.tokens).strip())
        print(
            f"--- {res.num_tokens} tokens in {res.total_s:.2f}s ({res.num_tokens / res.decode_s:.1f} tok/s decode), "
            f"KV cache {res.final_kv_bytes / 2**20:.2f} MB, compression {100 * res.final_compression:.1f}% "
            f"(max {100 * res.max_compression:.1f}%)"
        )
        for e in res.events:
            lens = e.layer_lengths_after
            print(
                f"    step {e.step:3d}: {e.entries_before} -> {e.entries_after} entries "
                f"(per-head length per layer {min(lens):.1f}..{max(lens):.1f}), {1000 * e.seconds:.1f} ms"
            )
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
