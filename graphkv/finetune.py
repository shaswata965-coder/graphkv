"""Fine-tune a causal LM on WikiText-2, following Sec. II-B of the paper.

The paper fine-tunes GPT-2 and TinyLlama for one epoch on WikiText-2 with
weight decay 0.01, 500 warm-up steps, batch size 4 (GPT-2) / 1 (TinyLlama)
and seed 42. Learning rate, sequence length and schedule are not stated; the
defaults below are those of the Hugging Face ``Trainer`` (lr 5e-5, linear
decay, AdamW, grad-norm clipping at 1.0), with 512-token blocks.

Examples::

    python -m graphkv.finetune --model gpt2 --batch-size 4 --output-dir checkpoints/gpt2-wikitext2
    python -m graphkv.finetune --model TinyLlama/TinyLlama_v1.1 --batch-size 1 \\
        --bf16 --gradient-checkpointing --output-dir checkpoints/tinyllama-wikitext2

Full fine-tuning of a 1.1B model needs roughly 18 GB of accelerator memory
for weights, gradients and AdamW state (fp32), plus activations.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional

import torch

from .data import load_wikitext_split
from .utils import environment_info, resolve_device, set_seed


def tokenize_split(tokenizer, lines: List[str], chunk_lines: int = 2000) -> List[int]:
    """Tokenize the raw split as one continuous text (no per-line special tokens)."""
    ids: List[int] = []
    for i in range(0, len(lines), chunk_lines):
        ids.extend(tokenizer("".join(lines[i : i + chunk_lines]), add_special_tokens=False, verbose=False)["input_ids"])
    return ids


def make_blocks(ids: List[int], block_size: int) -> torch.Tensor:
    n = len(ids) // block_size
    return torch.tensor(ids[: n * block_size], dtype=torch.long).view(n, block_size)


@torch.no_grad()
def evaluate(model, blocks: torch.Tensor, device, batch_size: int, autocast) -> float:
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(blocks), batch_size):
        batch = blocks[i : i + batch_size].to(device)
        with autocast():
            loss = model(input_ids=batch, labels=batch).loss
        tokens = batch.numel() - batch.shape[0]
        total += float(loss) * tokens
        count += tokens
    model.train()
    return math.exp(total / count)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="gpt2")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=None, help="stop early (for quick tests)")
    p.add_argument("--bf16", action="store_true", help="bfloat16 autocast (CUDA / recent CPUs)")
    p.add_argument("--fp16", action="store_true", help="float16 autocast with loss scaling (CUDA only)")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--eval-batches", type=int, default=None, help="limit validation blocks (default: all)")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    set_seed(args.seed)
    device = resolve_device(args.device)
    if args.fp16 and device.type != "cuda":
        raise SystemExit("--fp16 needs CUDA; use --bf16 or full precision")
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else None

    def autocast():
        return torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype else nullcontext()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)  # fp32 master weights
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.model}: {n_params / 1e6:.0f}M parameters on {device}; ~{n_params * 16 / 2**30:.1f} GB for fp32 weights+grads+AdamW")

    train = make_blocks(tokenize_split(tokenizer, load_wikitext_split(split="train")), args.block_size)
    valid = make_blocks(tokenize_split(tokenizer, load_wikitext_split(split="validation")), args.block_size)
    if args.eval_batches:
        valid = valid[: args.eval_batches * args.batch_size]
    print(f"train: {len(train)} blocks of {args.block_size} tokens, validation: {len(valid)} blocks")

    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]  # biases, norms
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}], lr=args.lr
    )
    steps_per_epoch = math.ceil(len(train) / (args.batch_size * args.grad_accum))
    total_steps = math.ceil(steps_per_epoch * args.epochs)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda") if args.fp16 else None

    ppl_before = evaluate(model, valid, device, args.batch_size, autocast)
    print(f"validation perplexity before: {ppl_before:.2f}")

    generator = torch.Generator().manual_seed(args.seed)
    model.train()
    step, micro, t0, running, since_log = 0, 0, time.time(), 0.0, 0
    while step < total_steps:
        order = torch.randperm(len(train), generator=generator)
        for i in range(0, len(order), args.batch_size):
            batch = train[order[i : i + args.batch_size]].to(device)
            with autocast():
                loss = model(input_ids=batch, labels=batch).loss / args.grad_accum
            (scaler.scale(loss) if scaler else loss).backward()
            running += float(loss.detach())
            micro += 1
            if micro % args.grad_accum:
                continue
            if scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            since_log += 1
            if step % args.log_every == 0 or step == total_steps:
                print(f"step {step}/{total_steps}  loss {running / since_log:.4f}  lr {scheduler.get_last_lr()[0]:.2e}  {time.time() - t0:.0f}s", flush=True)
                running, since_log = 0.0, 0
            if step >= total_steps:
                break

    ppl_after = evaluate(model, valid, device, args.batch_size, autocast)
    print(f"validation perplexity after: {ppl_after:.2f}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    info = {"args": vars(args), "steps": step, "val_ppl_before": ppl_before, "val_ppl_after": ppl_after,
            "environment": environment_info(device)}
    (out / "finetune_config.json").write_text(json.dumps(info, indent=2, default=str))
    print(f"saved to {out}; evaluate with: python -m graphkv.run --model {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
