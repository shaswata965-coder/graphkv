"""End-to-end check with the paper's model: do our contexts reproduce the authors' released generations?

The released RAG outputs (``Graph-COM/GraphKV/results``) were decoded
greedily, so under the computation that produced them every released token is
the model's arg-max given the tokens before it. For each example this script
builds the context of every method with this implementation, feeds the
released answer of every run teacher-forced (one forward pass per method) and
measures how often our arg-max equals the released token.

* diagonal (run evaluated under its own method): should be ~100%; the
  remainder is bf16 noise (FlashAttention on GPU vs SDPA on CPU) and
  re-tokenisation of the decoded text;
* off-diagonal (run evaluated under another method): the negative control,
  showing that the test can tell the methods apart.

    python -m graphkv_neurips.verify_released --model ldsjmdy/Tulu3-Block-FT --task 2wiki --num-examples 10 \\
        --output docs/neurips/verify_released_2wiki.json [--streamed-layers 8]
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from typing import Dict, List

import torch

from .data import fetch, load_rag
from .methods import Prepared, parse_method, prepare
from .model import RopeDecoder, Segment
from .prompts import PromptBuilder
from .run_rag import select

RUNS = {
    "vanilla": "sequential",
    "block": "block",
    "gapemp_appr_1": "graphkv-top1@official",
    "gapemp_appr_3": "graphkv-top3@official",
    "gapemp_appr_5": "graphkv-top5@official",
    "gapemp": "graphkv-full@official",
}
EXTRA_METHODS = ("graphkv-top3",)  # paper layout: shows how far the released code is from Sec. 3.1
SPECIAL = ("<|end_of_text|>", "<|eot_id|>", "<|im_end|>")


def _release_memory() -> None:
    """Free Python garbage and hand freed heap pages back to the OS (matters on RAM-limited CPU runs)."""
    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _clean(text: str) -> str:
    for s in SPECIAL:
        text = text.split(s)[0]
    return text


def _released(task: str, run: str, indices: set, data_dir=None, model_dir="ldsjmdyTulu3-Block-FT") -> Dict[int, str]:
    out = {}
    try:
        path = fetch(f"results/{task}/{model_dir}/{run}_ascent.jsonl", data_dir)
    except Exception:  # noqa: BLE001 - some runs were not released for every task
        return out
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in indices:
                rec = json.loads(line)
                if "generated" in rec:
                    out[i] = _clean(rec["generated"])
    return out


_PUNCT = set(".,!?;:'\")]}")


def cleanup_only(predicted: str, released: str) -> bool:
    """True when two tokens differ only by a space before punctuation.

    transformers 4.50 decoded the released text with ``clean_up_tokenization_spaces``
    (Tulu's tokenizer config), which deletes such spaces, so re-tokenising the stored
    text cannot recover a generated " ," or " 's"."""
    return predicted != released and predicted.strip() == released.strip() and bool(released.strip()) and released.strip()[0] in _PUNCT


def near_tie(margin: float, top_logit: float, ulps: int = 2) -> bool:
    """True when our arg-max beats the released token by at most ``ulps`` bfloat16 steps at that logit size.

    bfloat16 keeps 8 significant bits, so logits around 20 are only resolved to 0.125;
    the official run (bf16 FlashAttention on GPU) and this one can break such ties differently.
    """
    import math

    if top_logit == 0:
        return margin <= 0
    step = 2.0 ** (math.floor(math.log2(abs(top_logit))) - 7)
    return margin <= ulps * step + 1e-6


@torch.no_grad()
def teacher_forced(dec: RopeDecoder, prepared: Prepared, answers: List[List[int]], tok=None) -> List[dict]:
    """Arg-max agreement and NLL of each answer after the prepared context (answers batched)."""
    q = list(prepared.query_ids)
    seg, start = prepared.segment, prepared.query_start
    if len(q) > 1:  # put all but the last question token into the context, so the batch stays small
        pos = start + torch.arange(len(q) - 1).unsqueeze(0)
        _, head = dec.encode(torch.as_tensor([q[:-1]]), pos, seg)
        seg = Segment.cat(([seg] if seg is not None else []) + [head])
        start, q = start + len(q) - 1, q[-1:]
    width = 1 + max(len(a) for a in answers)
    ids = torch.zeros(len(answers), width, dtype=torch.long)
    for b, a in enumerate(answers):
        ids[b, : 1 + len(a)] = torch.as_tensor(q + a)
    pos = start + torch.arange(width).unsqueeze(0).expand(len(answers), -1)
    logits, _, _ = dec.forward(ids, pos, seg, logits="all")
    out = []
    for b, a in enumerate(answers):
        lg = logits[b, : len(a)].float()
        target = torch.as_tensor(a, device=lg.device)
        pred = lg.argmax(-1)
        hit = pred == target
        miss = (~hit).nonzero().view(-1).tolist()
        rec = dict(tokens=len(a), agree=float(hit.float().mean()), first_mismatch=miss[0] if miss else None,
                   nll=float(torch.nn.functional.cross_entropy(lg, target)))
        if tok is not None and miss:
            pairs = [(tok.decode([int(pred[j])]), tok.decode([a[j]])) for j in miss]
            # margin: how much our arg-max beats the released token (in logits); small = a near-tie
            rec["mismatches"] = [
                dict(pos=j, ours=p, released=r, margin=round(float(lg[j, pred[j]] - lg[j, a[j]]), 4), top_logit=round(float(lg[j, pred[j]]), 3))
                for j, (p, r) in zip(miss, pairs)
            ]
            rec["cleanup_only"] = all(cleanup_only(p, r) for p, r in pairs)
            rec["near_tie_only"] = all(near_tie(m["margin"], m["top_logit"]) for m in rec["mismatches"])
        out.append(rec)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="ldsjmdy/Tulu3-Block-FT")
    ap.add_argument("--task", default="2wiki", choices=["2wiki", "hqa", "tqa", "nqa", "multihopqa", "morehopqa"])
    ap.add_argument("--runs", default="vanilla,block,gapemp_appr_1,gapemp_appr_3,gapemp")
    ap.add_argument("--num-examples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--streamed-layers", type=int, default=None, help="stream this many layers from disk (low-RAM machines)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report-only", action="store_true", help="print the Markdown report of an existing --output file")
    args = ap.parse_args(argv)
    if args.report_only:
        with open(args.output) as f:
            saved = json.load(f)
        print(markdown(saved))
        return
    if args.threads:
        torch.set_num_threads(args.threads)

    examples = select(load_rag(args.task, args.data_dir), args.num_examples, "stratified", args.seed)
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    released = {r: _released(args.task, r, {e.index for e in examples}, args.data_dir) for r in runs}
    runs = [r for r in runs if released[r]]
    if args.streamed_layers:
        from .lowmem import load_streaming

        tok, model, hook = load_streaming(args.model, streamed=args.streamed_layers)
        dec = RopeDecoder(model)
        dec.layer_hook = hook
    else:
        from .run_rag import load_model

        tok, model = load_model(args.model, "auto", args.device)
        dec = RopeDecoder(model)
    builder = PromptBuilder(tok, "tulu")
    methods = [RUNS[r] for r in runs] + [m for m in EXTRA_METHODS if m not in RUNS.values()]
    results = []
    if os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)["examples"]
    done = {r["index"] for r in results}
    for ex in examples:
        if ex.index in done:
            continue
        t0 = time.time()
        prompt = builder.tokenize(builder.rag_text(ex.question, ex.documents, args.task))
        answers = {r: list(tok(released[r][ex.index], add_special_tokens=False).input_ids) for r in runs if ex.index in released[r]}
        answers = {r: a for r, a in answers.items() if a}
        rec = dict(index=ex.index, group=ex.group, cells={})
        for spec in methods:
            prepared = prepare(dec, prompt, parse_method(spec, bos_id=tok.bos_token_id), pad_id=tok.pad_token_id or 0)
            names = list(answers)
            for r, res in zip(names, teacher_forced(dec, prepared, [answers[r] for r in names], tok)):
                rec["cells"][f"{r}|{spec}"] = res
            del prepared
            _release_memory()
        rec["seconds"] = round(time.time() - t0, 1)
        results.append(rec)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(dict(model=args.model, task=args.task, runs=runs, methods=methods, examples=results), f, indent=1)
        diag = [rec["cells"][f"{r}|{RUNS[r]}"]["agree"] for r in answers]
        print(f"example {ex.index} ({rec['seconds']}s): own-method agreement " + ", ".join(f"{a:.3f}" for a in diag), flush=True)
    print(report(results, runs, methods))


def markdown(saved: dict) -> str:
    """Agreement matrix plus every disagreement on the diagonal, with its logit margin."""
    results, runs, methods = saved["examples"], saved["runs"], saved["methods"]
    lines = [f"{len(results)} examples of {saved['task']}, model `{saved['model']}`.", "", report(results, runs, methods), ""]
    rows = []
    for ex in results:
        for r in runs:
            c = ex["cells"].get(f"{r}|{RUNS[r]}")
            for m in (c or {}).get("mismatches", []):
                step = None
                if "top_logit" in m and m["top_logit"]:
                    import math

                    step = 2.0 ** (math.floor(math.log2(abs(m["top_logit"]))) - 7)
                rows.append((ex["index"], r, c["tokens"], m, step))
    if rows:
        lines += ["Disagreements between a released run and its own method (all positions):", "",
                  "| example | run | answer tokens | position | ours | released | logit margin | bf16 step at that logit |", "|---|---|---|---|---|---|---|---|"]
        for idx, r, n, m, step in rows:
            lines.append(f"| {idx} | {r} | {n} | {m['pos']} | `{m['ours']!r}` | `{m['released']!r}` | {m['margin']:.3f} | {'' if step is None else f'{step:.3f}'} |")
    return "\n".join(lines)


def report(results: List[dict], runs: List[str], methods: List[str]) -> str:
    """Token-weighted agreement matrix (rows: released run, columns: our method).

    "exact" counts outputs whose every token is our arg-max; mismatches that are only
    the decode clean-up of the released text (a space before punctuation) or bfloat16
    near-ties (see :func:`near_tie`) count as exact.
    """
    lines = ["| released run \\ our method | " + " | ".join(methods) + " |", "|---" * (len(methods) + 1) + "|"]
    for r in runs:
        cells = []
        for m in methods:
            hits = toks = exact = n = 0
            for ex in results:
                c = ex["cells"].get(f"{r}|{m}")
                if c:
                    hits += c["agree"] * c["tokens"]
                    toks += c["tokens"]
                    exact += c["first_mismatch"] is None or bool(c.get("cleanup_only")) or bool(c.get("near_tie_only"))
                    n += 1
            cell = f"{100 * hits / toks:.1f}% ({exact}/{n} exact)" if toks else ""
            cells.append(f"**{cell}**" if RUNS.get(r) == m else cell)
        lines.append(f"| {r} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
