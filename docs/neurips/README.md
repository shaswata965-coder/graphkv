# Graph-KV (NeurIPS 2025): implementation and verification

> H. Wang, P. Wang, M. Li, S. Liu, S. Miao, Z. Wang, P. Li,
> **"Graph-KV: Breaking Sequence via Injecting Structural Biases into Large Language Models"**,
> NeurIPS 2025, [arXiv:2506.07334](https://arxiv.org/abs/2506.07334). Official code: [Graph-COM/GraphKV](https://github.com/Graph-COM/GraphKV).

This directory documents `graphkv_neurips/`. It is a separate package from `graphkv/`, which implements a different
paper with a similar name (the ISCAS 2026 KV-cache compression paper; see the top-level README).

* `third_party/GraphKV`: the official code, pinned as a git submodule (commit `ba01bbc`). It has no licence file,
  so it is referenced, not copied.
* `graphkv_neurips/`: an implementation that follows the paper. It runs on CPU or GPU with transformers ≥ 4.45,
  and is checked token for token against the official functions.
* [`COLAB.md`](COLAB.md): how to run the experiments on Colab within a fixed compute budget.

## The method, mapped to code

| paper | code |
|---|---|
| offline parallel encoding `h^(0)` of every chunk | `model.RopeDecoder.encode_batch` |
| structure-aware attention: a target re-encoded on the KV of its sources `N(j)` (Sec. 3.1) | `graph.encode_graph`, `graph.GraphPlan.rounds` |
| graph-structured block mask (Fig. 1) | the same computation; `tests/test_gkv_graph.py` checks it against one Hugging Face forward pass with that mask |
| shared PEs: sources `[0, L)`, targets `[L, 2L)`, query after (Sec. 3.1, Fig. 2) | `graph.PaperLayout` (L = longest source, as the official ArXiv-QA code); `@global` = `graph.GlobalPaperLayout` (L = longest chunk overall, literal Sec. 3.1) |
| multi-round propagation, `[2L, 3L)`, ... (Sec. 3.1, Appendix A.1.2) | `GraphPlan.chain`, `graphkv-chain1-3` |
| RAG bipartite graph: top-m sources → remaining targets (Fig. 3), Graph-KV-Full | `GraphPlan.top_m`, `GraphPlan.rag_full` |
| Task 1, RAG (Sec. 4.1, Tables 2, 3, 5, 6) | `run_rag.py`, data `data.load_rag` (the official processed data) |
| Task 2, ArXiv-QA (Sec. 4.2, Fig. 6) | `run_arxiv.py` (prompts, distractor sampling, per-method blocks and token limits of `arxiv_inference.py`) |
| Task 3, paper classification (Sec. 4.3, Table 4) | `run_classify.py`, data `tag_data.py` (Cora / PubMed of `Graph-COM/Text-Attributed-Graphs`) |
| Task 4, stress test (Sec. 4.4) | `stress.py` |
| baselines: sequential, PCW, APE (T = S = 0.9, BOS before the question as in APE's code), Block-RAG | `methods.py` |
| prompts of Appendix A.2 | `prompts.py` (`style="tulu"` reproduces the official strings) |
| metrics: subspan EM, MultiHop-RAG, MorehopQA | `metrics.py` |

The decoder (`model.py`) reuses a Hugging Face model's modules and implements only the attention over an explicit
context. It does this because the official code gets its masks from FlashAttention-2's bottom-right causal alignment
(it passes no mask). It therefore needs CUDA, `flash-attn` and transformers 4.50. Supported architectures: Llama
(including Tulu-3), Mistral, Qwen2 and Qwen3.

### Where the official code differs from the paper

1. **Positions in the RAG experiments.**
   * `pcw.py: gapemp / gapemp_appr` (the code behind Tables 2 and 3) lay the source chunks out one after another
     when targets attend to them, and re-rotate everything to consecutive positions for the query.
   * The shared ranges of Sec. 3.1 are used only by the ArXiv-QA code (`pcw_parallel.py`).
   * Both are implemented: `layout="paper"` (default) and `layout="sequential"`, written as `graphkv-top3@official`.
2. **Top-m targets.**
   * Fig. 3 makes the *remaining* chunks the targets.
   * `gapemp_appr` makes *every* chunk a target, including the top-m themselves, and lets the query see both
     copies of every chunk.
   * Both are implemented: `graphkv-top3` and `graphkv-top3@official`.
3. **Graph-KV-Full** keeps a target's own source copy among its sources (self-loops); the same holds here.

## Verification

### 1. Parity with the official code (`tests/test_gkv_upstream_parity.py`, transformers 4.50)

The official functions run unmodified on a tiny random Llama, under two shims:
* FlashAttention-2 is replaced by an attention function with the same bottom-right-causal semantics;
* the bfloat16 cast in their RoPE helper is removed so the comparison can run in float32.

Greedy outputs are compared over the whole generation, up to 1024 tokens.

| official function | this implementation | result |
|---|---|---|
| `pcw_parallel.gapemp_graph` (ArXiv-QA Graph-KV) | `graphkv-star`; `run_arxiv.build` | identical |
| `pcw_parallel.gapemp_graph_batch` (with a distractor ego-graph) | `run_arxiv.build` (star plans, `layout="paper-local"`) | identical |
| `pcw_parallel.block`, `block_batch` (ArXiv-QA Block-RAG) | `block`; `run_arxiv.build` | identical |
| `utils.build_prefix`, `docs2blocks`, `build_suffix` (all 6 RAG tasks) | `prompts.PromptBuilder(style="tulu")` | byte-identical |
| `pcw.gapemp` (RAG Graph-KV-Full) | `graphkv-full@official` | identical |
| `pcw.gapemp_appr`, k = 1 and 3 | `graphkv-top1@official`, `graphkv-top3@official` | identical |
| `server/block_generate_server.block_generate` | `block` | identical |
| `pcw.vanilla` | `sequential` | identical |

### 2. Properties stated in the paper (`tests/test_gkv_graph.py`, any transformers version)

* **Fig. 2:** a target attending to one source equals reading the source, then the target, sequentially.
* **Fig. 1:** Graph-KV equals one Hugging Face forward pass with the graph-structured block mask and shared positions.
* **Positional bias:** any permutation of the sources gives the same logits; sequential encoding does not.
* **Context window:** the position span is P + 2L + |q|, whatever the number of chunks.
* **Cost:** context building takes exactly Σ L_i(L_i+1)/2 + Σ_(targets) L_t · |N(t)| query-key pairs, i.e.
  O(|V|L² + |E|L²).
* The decoder reproduces Hugging Face logits for Llama, Mistral, Qwen2 and Qwen3 (max |Δ| ≈ 2e-7).

### 3. The paper's numbers from the authors' released outputs

`python -m graphkv_neurips.rescore` scores the generations published in `Graph-COM/GraphKV/results` with the official
metrics. Full table: [`rescored_official_outputs.md`](rescored_official_outputs.md).

| table | result |
|---|---|
| **Figure 6 (ArXiv-QA)** | All 23 released runs reproduce the paper's counts exactly, e.g. Graph-KV 27 / 17 / 9 correct out of 60 with 0 / 1 / 2 distractors vs. Block-FT sequential 30 / 7-14 / 0-9. PCW and APE outputs were not released. |
| **Table 3 (NarrativeQA, 2Wiki, TriviaQA, HotpotQA)** | 64 of 67 cells reproduce to 0.01 (the paper truncates). Two cells look like typos: 2Wiki comparison for 8B-RAG is 82.89 (paper 82.29), and 2Wiki inference for 8B-Block sequential is 63.00 (paper 63.08). HotpotQA 8B-Block sequential cannot be checked: only 13 of its 7405 outputs were released. |
| **Table 2, MorehopQA** | The table mixes two scorers. The baseline rows reproduce exactly when a missing `Answer:` counts as an empty answer, but the Graph-KV-Full row reproduces only with the released `rag_eval.py`, which falls back to the whole output. With one scorer for all rows, Full beats the best sequential baseline by **+1.0** (strict: 39.35 vs 38.37) or loses by **−3.0** (released scorer: 44.90 vs 47.85), not the reported **+6.5** (44.90 vs 38.37). Two smaller inconsistencies: Top-1's Avg (37.96) does not match its own hop columns (37.56), and Top-3's Hop-2 (25.72) is copied from the Block-RAG row (the outputs give 22.83). |
| **Table 2, MultiHop-RAG** | Cannot be reproduced: the paper's scorer (`qa_evaluate2`) is not released, and no variant tried here (answer extraction × matching rule) gives its numbers. With the benchmark's published metric the ordering still favours Graph-KV: Full 42.25, Top-3 40.80, Top-1 38.41 vs. the best sequential 37.05 and Block-RAG 34.58. |

The dataset labelled *NarrativeQA* in the paper is the `nq_eval` file of Block-Attention. Its questions (e.g.
"who got the first nobel prize in physics") are Natural Questions, not NarrativeQA.

### 4. End to end with the paper's 8B model

The released generations of `ldsjmdy/Tulu3-Block-FT` were fed teacher-forced through this implementation, on CPU
with a streaming loader. Setup: 12 2Wiki examples; 60 outputs of sequential, Block-RAG and Graph-KV Top-1/Top-3/Full.

Under each run's own method:
* 99.79% of the 5,353 released tokens are this implementation's arg-max;
* 50 of the 60 outputs agree at every token;
* all 11 disagreements are bfloat16 ties (logit margin 0-0.25).

Under other methods the agreement drops to 97.1%. The released top-3 outputs agree with the `@official` layout
(12/12 outputs) but not with the paper's shared ranges (5/12). The full matrix and the component-by-component
audit are in [`VERIFICATION.md`](VERIFICATION.md).

The Colab notebook repeats the check on a GPU with more examples (cell 6a). Its free-generation comparison is
cell 6b.

## Experiments beyond the paper

`python -m graphkv_neurips.run_structure` (see [`COLAB.md`](COLAB.md)):

| option | values |
|---|---|
| nodes (`--chunker`) | `documents`, `fixed:N`, `sentences[:N]`, `paragraphs[:N]`, `attention:N`, `attention-sent:N`. Attention-shift cuts are placed where the following tokens attend least to the preceding text. |
| top-k sources (`--scorer`) | `retriever`, `bm25`, `attention` (the LLM's own question attention), `random`, `first`, `last`, `oracle` |
| edges and views (`--conditions`) | `graphkv-top<m>[:targets|:sources]`, `noedge-top<m>[:targets]` (control: same tokens and positions, no edges), `graphkv-full[:targets]`, `parallel`, the baselines, and `@official` variants |
| metrics (`--metric`) | greedy generation with the task metric, and teacher-forced answer NLL (one forward pass) for sweeps |

Every summary reports paired bootstrap intervals against the first condition, plus the source→target transfer
(`graphkv-top<m>:targets` − `noedge-top<m>:targets`), split by where the gold answer occurs.

## Commands

```bash
pip install -e ".[neurips,test]"
python -m pytest tests/test_gkv_*.py          # parity test runs only under transformers 4.50, skipped otherwise

# the paper's RAG protocol (Tables 2/3/5/6)
python -m graphkv_neurips.run_rag --model ldsjmdy/Tulu3-Block-FT --task 2wiki \
    --methods sequential,block,graphkv-top1,graphkv-top3,graphkv-full --num-examples 200 --output-dir runs/rag

# structure experiments
python -m graphkv_neurips.run_structure --model ldsjmdy/Tulu3-Block-FT --task 2wiki --chunker attention:128 \
    --scorer attention --conditions parallel,graphkv-top3,graphkv-top3:targets,noedge-top3:targets \
    --metric nll --num-examples 300 --output-dir runs/e2

# re-score the released outputs (download results/ from the Graph-COM/GraphKV dataset first)
python -m graphkv_neurips.rescore --results-dir <path>/results --output docs/neurips/rescored_official_outputs.md

# the paper's other tasks
python -m graphkv_neurips.run_arxiv --method graphkv --groups 1 --output-dir runs/arxiv        # Fig. 6 (80 GB GPU)
python -m graphkv_neurips.run_classify --dataset cora --output-dir runs/cora                    # Table 4
python -m graphkv_neurips.stress ttft --words 100,200,400,800                                    # Fig. 8 (Sec. 4.4)

# end-to-end check with the 8B model against the released generations
python -m graphkv_neurips.verify_released --task 2wiki --num-examples 12 --output docs/neurips/verify_released_2wiki.json
```

Parity environment for the official code:

```bash
python -m venv .venv-upstream && .venv-upstream/bin/pip install torch transformers==4.50.0 pytest
git submodule update --init third_party/GraphKV
.venv-upstream/bin/python -m pytest tests/test_gkv_upstream_parity.py
```
