"""Compare a ``run_rag`` output with the generations released by the authors (same examples).

With the same model (``ldsjmdy/Tulu3-Block-FT``), prompt style ``tulu`` and the
``@official`` method variants, greedy outputs should coincide with the released
ones up to numerical noise (bf16, FlashAttention vs SDPA):

    python -m graphkv_neurips.compare_official --ours runs/e0/graphkv-top3@official.jsonl --task 2wiki --run gapemp_appr_3

The released runs have up to 1024 new tokens; ours are compared on their own
length (a match means the official text starts with ours).
"""

from __future__ import annotations

import argparse
import json

from .data import fetch
from .metrics import score_example

OFFICIAL_RUNS = {
    "sequential": "vanilla",
    "block": "block",
    "graphkv-top1@official": "gapemp_appr_1",
    "graphkv-top3@official": "gapemp_appr_3",
    "graphkv-top5@official": "gapemp_appr_5",
    "graphkv-full@official": "gapemp",
}
SPECIAL = ("<|end_of_text|>", "<|eot_id|>", "<|im_end|>")


def _clean(text: str) -> str:
    for s in SPECIAL:
        text = text.replace(s, "")
    return text.strip()


def compare(ours_path: str, task: str, run: str, model_dir: str = "ldsjmdyTulu3-Block-FT", data_dir=None) -> dict:
    with open(ours_path) as f:
        ours = [json.loads(line) for line in f if line.strip()]
    wanted = {r["index"] for r in ours}
    official = {}
    with open(fetch(f"results/{task}/{model_dir}/{run}_ascent.jsonl", data_dir)) as f:
        for i, line in enumerate(f):
            if i in wanted:
                official[i] = json.loads(line)
    same = prefix = 0
    acc_ours = acc_off = 0.0
    n = 0
    for r in ours:
        o = official.get(r["index"])
        if o is None:
            continue
        n += 1
        a, b = _clean(r["generated"]), _clean(o.get("generated", ""))
        same += a == b
        prefix += bool(a) and b.startswith(a)
        acc_ours += r["score"]
        acc_off += score_example(task, o.get("generated", ""), o["answers"])
    return dict(n=n, identical=same / max(1, n), official_starts_with_ours=prefix / max(1, n), accuracy_ours=100 * acc_ours / max(1, n), accuracy_official=100 * acc_off / max(1, n))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", required=True, help="a <method>.jsonl written by run_rag")
    ap.add_argument("--task", required=True)
    ap.add_argument("--run", default=None, help="official run name (default: from the file name)")
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args(argv)
    run = args.run
    if run is None:
        label = args.ours.rsplit("/", 1)[-1][: -len(".jsonl")]
        run = OFFICIAL_RUNS[label]
    print(json.dumps(compare(args.ours, args.task, run, data_dir=args.data_dir), indent=1))


if __name__ == "__main__":
    main()
