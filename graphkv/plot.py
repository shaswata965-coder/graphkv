"""Plot a sweep like Fig. 2 of the paper and print a Table I style summary.

Usage::

    python -m graphkv.plot results/gpt2-20261008-120000
    python -m graphkv.plot results/run_a results/run_b --title "GPT-2 vs TinyLlama"

Writes ``fig2_metrics.png`` and ``table1.md`` into each results directory.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

PPL_METRICS = {
    "tf_ppl": "teacher-forced reference text, compressed model",
    "seq_ppl": "prompt + generated text, scored by uncompressed model",
    "oracle_ppl": "generated text, scored by uncompressed model",
    "gen_ppl": "generated text, scored by compressed model itself",
}

PPL_SHORT = {
    "tf_ppl": "teacher-forced, compressed model",
    "seq_ppl": "prompt + generation, full model",
    "oracle_ppl": "generation, full model",
    "gen_ppl": "generation, compressed model",
}

# (summary column, panel title, y label); the perplexity column is chosen at runtime.
PANELS = [
    ("{ppl}", "Model quality (perplexity)", "Perplexity  (lower is better)"),
    ("decode_tok_s", "Throughput", "Tokens / s  (higher is better)"),
    ("latency_s", "Average latency", "Seconds per prompt  (lower is better)"),
    ("max_compression_pct", "Max cache compression", "Compression saving %  (higher is better)"),
]

# Categorical slots 1-3 of the validated reference palette, with a distinct
# marker per series so identity never relies on color alone.
SERIES_STYLE = [("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "^")]
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def read_summary(results_dir: Path) -> List[dict]:
    with (results_dir / "summary.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def num(row: dict, key: str) -> Optional[float]:
    try:
        v = float(row.get(key, ""))
    except ValueError:
        return None
    return None if math.isnan(v) else v


def plot(results_dir: Path, title: Optional[str] = None, ppl_metric: str = "tf_ppl") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_summary(results_dir)
    baseline = next((r for r in rows if r["method"] == "baseline"), None)
    by_interval: Dict[int, List[dict]] = {}
    for r in rows:
        if r["method"] == "graphkv":
            by_interval.setdefault(int(r["interval"]), []).append(r)

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK_MUTED,
                         "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "text.color": INK})
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), facecolor="#fcfcfb")
    for ax, (key, panel_title, ylabel) in zip(axes.flat, PANELS):
        key = key.format(ppl=ppl_metric)
        if key == ppl_metric:
            panel_title = f"Perplexity: {PPL_SHORT[ppl_metric]}"
        ax.set_facecolor("#fcfcfb")
        for i, (interval, series) in enumerate(sorted(by_interval.items())):
            pts = sorted((num(r, "epsilon"), num(r, key)) for r in series if num(r, key) is not None)
            if not pts:
                continue
            color, marker = SERIES_STYLE[i % len(SERIES_STYLE)]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, marker=marker, linewidth=2,
                    markersize=6, markeredgecolor="#fcfcfb", markeredgewidth=1, label=f"Compressed, M = {interval}")
        if baseline is not None and num(baseline, key) is not None:
            ax.axhline(num(baseline, key), color=INK_MUTED, linestyle="--", linewidth=1.5, label="Standard cache")
        ax.set_title(panel_title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("Epsilon threshold")
        ax.set_ylabel(ylabel)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes.flat[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.suptitle(title or f"Graph-based KV-cache compression: {results_dir.name}", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    out = results_dir / "fig2_metrics.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def table1(results_dir: Path, interval: int = 16, epsilons=(10.0, 20.0, 40.0)) -> str:
    """Markdown version of Table I (baseline values in parentheses for perplexity)."""
    rows = read_summary(results_dir)
    baseline = next((r for r in rows if r["method"] == "baseline"), None)
    picked = []
    for eps in epsilons:
        match = [r for r in rows if r["method"] == "graphkv" and int(r["interval"]) == interval and num(r, "epsilon") == eps]
        if match:
            picked.append((eps, match[0]))
    if not picked:
        return f"(no rows with M = {interval} and epsilon in {list(epsilons)})"

    def cell(r: dict, key: str, spec: str, with_base: bool = False) -> str:
        v = num(r, key)
        if v is None:
            return "-"
        s = format(v, spec)
        if with_base and baseline is not None and num(baseline, key) is not None:
            s += f" ({format(num(baseline, key), spec)})"
        return s

    lines = [
        f"Performance metrics for M = {interval}  ({results_dir.name})",
        "",
        "| Metric | " + " | ".join(f"eps = {e:g}" for e, _ in picked) + " |",
        "|---|" + "---|" * len(picked),
        "| Max Compression (%) | " + " | ".join(cell(r, "max_compression_pct", ".2f") for _, r in picked) + " |",
        *(
            f"| Perplexity [{m}] (baseline) | " + " | ".join(cell(r, m, ".2f", True) for _, r in picked) + " |"
            for m in PPL_METRICS
            if any(num(r, m) is not None for _, r in picked)
        ),
        "| Throughput (tokens/s) | " + " | ".join(cell(r, "decode_tok_s", ".2f") for _, r in picked) + " |",
        "| Latency (s) | " + " | ".join(cell(r, "latency_s", ".3f") for _, r in picked) + " |",
    ]
    if baseline is not None:
        lines += [
            "",
            f"Standard cache: throughput {cell(baseline, 'decode_tok_s', '.2f')} tokens/s, "
            f"latency {cell(baseline, 'latency_s', '.3f')} s",
            "",
            "Perplexity variants: " + "; ".join(f"{k} = {v}" for k, v in PPL_METRICS.items()),
        ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", nargs="+", help="results directories written by graphkv.run")
    p.add_argument("--title", default=None)
    p.add_argument("--table-interval", type=int, default=16)
    p.add_argument("--table-epsilons", default="10,20,40")
    p.add_argument("--ppl-metric", choices=list(PPL_METRICS), default="tf_ppl", help="perplexity shown in the figure")
    args = p.parse_args(argv)
    epsilons = tuple(float(x) for x in args.table_epsilons.split(","))
    for d in args.results:
        path = Path(d)
        print(f"wrote {plot(path, args.title, args.ppl_metric)}")
        table = table1(path, args.table_interval, epsilons)
        (path / "table1.md").write_text(table + "\n")
        print(table + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
