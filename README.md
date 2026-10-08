# GraphKV: graph-based dynamic KV-cache compression

A from-scratch, tested implementation of

> N. Bhattacharya, S. Sathvik, A. Yadav, A. Dixit, B. Kumar,
> **"A Graph-Based Methodology for Dynamic KV-Cache Compression in Transformer Inference"**,
> IEEE ISCAS 2026, DOI 10.1109/ISCAS66217.2026.11562780.

It is designed to run on your own machine: CPU, Apple Silicon (MPS) or an NVIDIA GPU. It needs no
custom kernels or compilation. It reuses the weights and layers of any Hugging Face GPT-2 /
Llama-family checkpoint.

## The method in one paragraph

During autoregressive decoding, every `M` generated tokens the KV cache of every layer is compressed.
The first `N_sink = 4` tokens (attention sinks) are kept as they are. The remaining cached tokens become nodes of a graph, with an
edge between two tokens when the Euclidean distance of their key vectors is below `ε` (Eq. 1).
Min-label propagation finds the connected components (Algorithm 1, lines 6-8). Each component is replaced by
the mean of its keys and the mean of its values (Eq. 2). The cache becomes `[sinks, centroids]` and
decoding continues.

| Paper | Code |
|---|---|
| ε-graph, Eq. 1 | `graphkv/clustering.py: epsilon_adjacency` |
| Label propagation, Alg. 1 lines 6-8 (`T` iterations or to convergence) | `graphkv/clustering.py: label_propagation` |
| Clusters from unique labels, centroids, Eq. 2 | `cluster_assignment`, `merge_clusters` |
| Sink / prunable split, per-layer merge, Alg. 1 lines 1-15 | `graphkv/cache.py: compress_layer, compress_cache` |
| Compression every `M` decoded tokens | `graphkv/engine.py: GraphKVEngine` |
| Evaluation protocol (WikiText-2, 100 prompts, 50 new tokens, ε and M sweep) | `graphkv/run.py`, `graphkv/data.py` |
| Fig. 2 and Table I | `graphkv/plot.py` |
| Fine-tuning on WikiText-2 (1 epoch, wd 0.01, 500 warm-up steps) | `graphkv/finetune.py` |
| FPGA synthesis (Table II) | not included; it needs Vivado and an RTL/HLS design of the cache |

## Setup (local)

Requirements: Python 3.9 or newer and about 2 GB of free disk (GPT-2 is 0.5 GB; TinyLlama is 2.2 GB plus the WikiText dataset).

```bash
git clone <this repo> graphkv && cd graphkv
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 1) PyTorch for YOUR hardware (pick one):
pip install torch --index-url https://download.pytorch.org/whl/cpu      # CPU only
pip install torch                                                       # macOS (Apple Silicon -> MPS) or default CUDA build on Linux
pip install torch --index-url https://download.pytorch.org/whl/cu124    # a specific CUDA version; match your driver

# 2) Everything else
pip install -r requirements.txt       # or: pip install -e ".[experiments,test]"

# 3) Check the installation (offline parity tests against Hugging Face on your device)
python -m graphkv.envcheck --online
python -m pytest
```

`envcheck` prints your library versions and device. It then checks that this code reproduces the Hugging Face
logits **on your device** for GPT-2, Llama, Mistral and Qwen2 (tiny random models, nothing downloaded).
`--online` also checks that the model hub and WikiText-2 can be reached. Models and data are cached by
Hugging Face under `~/.cache/huggingface`. After the first download you can work fully offline with
`export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`.

Tested here with Python 3.13 + PyTorch 2.14 (CPU) against transformers **4.46.3 and 5.19.0**.
Any transformers version ≥ 4.45 should work.

## Quick start

```bash
# Generate side by side with a standard and a compressed cache
python -m graphkv.demo --model gpt2 --epsilon 5 --interval 16

# Small sweep (5 prompts, about 6 minutes on a 4-core CPU)
python -m graphkv.run --model gpt2 --num-prompts 5 --epsilons 2,5,10,20 --intervals 16 --output-dir results/smoke
python -m graphkv.plot results/smoke
```

`--device` defaults to `auto` (CUDA, then MPS, then CPU). `--dtype` defaults to `float32`; `float16` and `bfloat16` are
available on GPU/MPS. `--model` also accepts a local directory, such as a fine-tuned checkpoint.

## Reproducing the paper's experiments

The paper evaluates GPT-2 (124M) and TinyLlama-1.1B, both fine-tuned for one epoch on WikiText-2.
It uses the first 100 WikiText-2 test entries longer than 200 characters, truncated to 200 words, and 50 new
tokens per prompt. It sweeps `ε` from 0.5 to 40 and `M ∈ {8, 16, 32}`.

```bash
# (optional) fine-tune as in the paper; without this the pretrained checkpoints are used
python -m graphkv.finetune --model gpt2 --batch-size 4 --output-dir checkpoints/gpt2-wikitext2

# full sweep: baseline + 11 epsilons x 3 intervals = 34 configurations x 100 prompts
python -m graphkv.run --model checkpoints/gpt2-wikitext2 --output-dir results/gpt2-ft
python -m graphkv.plot results/gpt2-ft            # -> fig2_metrics.png, table1.md

# TinyLlama (paper: batch size 1 for fine-tuning)
python -m graphkv.finetune --model TinyLlama/TinyLlama_v1.1 --batch-size 1 --bf16 \
    --gradient-checkpointing --output-dir checkpoints/tinyllama-wikitext2
python -m graphkv.run --model checkpoints/tinyllama-wikitext2 --output-dir results/tinyllama-ft
```

The defaults of `graphkv.run` are the paper's settings: ε grid `0.5,1,2,5,10,15,20,25,30,35,40`, M `8,16,32`,
100 prompts, 50 new tokens, `N_sink = 4`, seed 42.

**Runtime and memory planning.** Each (prompt, configuration) pair runs a timed generation, an untimed
oracle scoring pass and a teacher-forced pass.
* GPT-2 on a 4-thread CPU: about 3.5 s per pair, so the full 100-prompt sweep takes about 3.5 hours. Reduce it with
  `--num-prompts 20` or fewer ε values.
* TinyLlama on CPU (fp32): about 5 GB RAM and roughly 10× slower than GPT-2. Prefer a GPU, or run a subset.
* Fine-tuning TinyLlama in full needs about 18 GB of accelerator memory for weights, gradients and AdamW state.
  GPT-2 fine-tunes on a CPU in under an hour.
* Interrupted sweeps **resume**: rerun the same command with the same `--output-dir` and finished configurations are skipped.
* Timings are only comparable within one machine and one run. Don't run other heavy jobs at the same time, and use
  `--threads N` on CPU for stable numbers.

Other useful flags: `--prompts-file my.txt` (your own prompts, one per line, or a `.jsonl` file with a `text` key),
`--do-sample --temperature 0.8 --top-p 0.95` (sampling; each prompt uses the same random stream in every
configuration), `--ignore-eos`, `--skip-teacher-forced`, `--skip-oracle`, `--max-iters 5` (a fixed number `T` of
label-propagation iterations), and `--granularity token`.

## Outputs

`results/<run>/summary.csv` has one row per configuration:

| column | meaning |
|---|---|
| `max_compression_pct` | Paper's "Max Compression". Saving vs. a standard cache right after a compression event, maximised over the run, then averaged over prompts. Counted in cache entries (key+value vectors of one KV head) over all layers. |
| `final_compression_pct` | The same saving at the end of generation. |
| `tf_ppl` | **Teacher-forced perplexity.** The last ≤50 tokens of each prompt are held out and fed one by one as if generated, with the cache compressed every `M` tokens. This is the strictest fidelity measure: how well the compressed model predicts real text. |
| `seq_ppl` | Prompt + generated text, scored by the **uncompressed** model. |
| `oracle_ppl` | Only the generated tokens, scored by the uncompressed model. |
| `gen_ppl` | Self-perplexity of the generated tokens under the compressed model itself. |
| `prefix_match` | Fraction of the baseline's generation reproduced token for token before the first divergence (greedy decoding). |
| `decode_tok_s` | Paper's "Throughput". Generated tokens divided by decode time; includes compression time, excludes prefill. |
| `e2e_tok_s` | Generated tokens divided by (prefill + decode) time. |
| `latency_s` | Paper's "Latency". Mean end-to-end time per prompt (prefill + 50 decode steps + compression). |
| `ttft_s`, `per_token_ms`, `compress_ms_per_call` | Prefill time, decode time per token, time per compression call. |
| `kv_mb_peak`, `kv_mb_final` | KV-cache size in MiB counting real entries. Per-head clustering uses a padded tensor internally; the allocated size is in `per_prompt.jsonl`. |

`per_prompt.jsonl` holds the per-prompt data, including generated token ids and the final per-layer cache lengths.
`run_config.json` records all arguments and the software/hardware environment.

## Interpretation choices (read this before comparing to the paper)

The paper leaves several details open. Each choice below is a flag where reasonable.

1. **Clustering is per KV head (default, `--granularity head`).** Algorithm 1 indexes the cache as
   `K[:, :, :N_sink, :]` (batch, heads, sequence, head_dim) and calls `cdist(K_p, K_p)`. That produces one distance
   matrix per head, so each (layer, KV head) is clustered on its 64-dimensional keys. This choice also
   reproduces the paper's ε scale. Measured compression of the prefilled prompt cache over 5 WikiText prompts
   (pretrained checkpoints):

   | ε | 2 | 5 | 10 | 15 | 20 |
   |---|---|---|---|---|---|
   | GPT-2, per head | 4.9% | 27% | 96% | 99% | 99% |
   | GPT-2, whole-token key (768-d) | 0.1% | 0.9% | 3.3% | 8.3% | 24% |
   | TinyLlama, per head | 0.1% | 10.5% | 72% | 97% | 99% |
   | TinyLlama, whole-token key (256-d) | 0% | 0% | 4.8% | 34% | 64% |

   The paper (Fig. 2, Table I) shows a sigmoid that rises between ε≈2 and 15 and saturates at about 97% by ε≈20, which matches
   per-head distances. Heads then hold different numbers of entries. They are stored padded with a validity mask,
   and memory is counted as a ragged layout would store it. `--granularity token` instead builds one graph per
   layer over whole-token keys (all KV heads concatenated). This needs ε about √(kv_heads) times larger.
2. **Positions are the true token positions.** After merging, the next token still gets its real absolute position
   (GPT-2 position embedding / RoPE angle), not the shortened cache length. Cached keys are post-RoPE, as in Hugging Face,
   so centroids average rotated keys.
3. **Compression schedule.** The prompt is prefilled into an uncompressed cache. Compression runs after decode steps
   M, 2M, …, and each time covers the whole prunable zone, including centroids from earlier rounds. With 50 new tokens:
   M=8 compresses 6 times, M=16 3 times, M=32 once.
4. **Repeated merges use the paper's plain mean** of the current entries, so an old centroid counts as one entry.
   `--weighted-merge` (an extension) weights entries by the number of original tokens they represent.
5. **Label propagation runs to convergence** by default, i.e. exact connected components as described in Sec. II-A.2.
   `--max-iters T` runs exactly `T` synchronous rounds of Algorithm 1, line 7.
6. **Perplexity is not defined in the paper.** Its numbers (GPT-2: 23.87 compressed vs 23.42 baseline at 97% compression)
   behave like `seq_ppl`. The unchanged ~260-token prompt dominates that score, so it barely moves even when the
   generated text degrades. For fidelity, look at `tf_ppl` and `prefix_match`. `plot.py --ppl-metric` chooses which
   perplexity the figure shows (default `tf_ppl`).
7. **Throughput and latency** are not defined precisely in the paper. The definitions above are used. Absolute numbers
   depend on hardware (the paper used an NVIDIA L40S) and on this pure-PyTorch decoding loop, so compare against the
   baseline row of the same run.
8. **Decoding** is greedy by default and stops at EOS (the paper says "a maximum of 50 new tokens"). It uses batch size 1,
   because every sequence gets its own cluster structure.
9. **Fine-tuning** hyperparameters not given in the paper use Hugging Face `Trainer` defaults: lr 5e-5, linear decay,
   AdamW, grad clipping 1.0, plus 512-token blocks.
10. **TinyLlama checkpoint.** The paper does not say which one. The examples use `TinyLlama/TinyLlama_v1.1` (base model).

### Optional extensions (off by default)

* `--weighted-merge`: centroids are exact means over the original tokens, even across repeated compressions.
* `--proportional-attention`: adds `log(n)` to the attention logit of an entry that represents `n` tokens, as in Token
  Merging's proportional attention. A merged centroid then keeps the attention mass of the tokens it replaced.
  Together with `--weighted-merge`, merging identical entries leaves attention output exactly unchanged (tested).
  On pretrained GPT-2 (5 prompts, M=16) this cut `tf_ppl` from 62.9 to 34.3 at ε=5 (baseline 29.2) and from 256 to 65
  at ε=10.

## What to expect

These findings come from runs of this code on pretrained (not fine-tuned) GPT-2, on CPU.

* The **compression curve** reproduces the paper's shape and scale: about 35% at ε=5, about 95% at ε=10 and about 97.8% at ε≥20
  (M=16), which saturates at `N_sink` plus a few centroids per head.
* **Throughput** shows the paper's non-monotonic pattern. Small ε costs throughput (clustering work with little reduction).
  Large ε recovers to roughly baseline or slightly above, since the attention context becomes tiny. On short (~300-token)
  contexts attention is a small share of compute, so large gains are not expected.
* **Fidelity.** `seq_ppl` stays nearly flat across ε, consistent with the paper's "stable perplexity". The stricter
  `tf_ppl` and the generated text degrade a lot once compression passes roughly 30-40%. At ε≥10 the 50-token
  continuation typically collapses into repetition after the first compression. Judge "near-lossless" with `tf_ppl`,
  `prefix_match` and the generated text in `per_prompt.jsonl`, not with `seq_ppl` alone.

See `docs/example_results.md` for a 20-prompt GPT-2 sweep produced with this code.

## Supported models and limits

* `gpt2` (all sizes) and Llama-style decoders: `llama` (TinyLlama, Llama 2/3, SmolLM), `mistral` (contexts within the
  sliding window) and `qwen2`. Other architectures raise a clear error.
* Batch size 1 per sequence (as in the paper's evaluation).
* GPT-2 has a hard limit of 1024 positions. Longer prompts are truncated, with a notice.

## How correctness is checked (`python -m pytest`, 65 tests)

* With no compression, the decoding loop reproduces Hugging Face logits for GPT-2, Llama, Mistral and Qwen2, both for full
  forward passes and token by token (max |Δlogit| ≈ 1e-7). Teacher-forced NLL equals Hugging Face's loss.
* Label propagation matches a BFS connected-components reference on random graphs. With fixed `T` it matches a literal
  transcription of Algorithm 1.
* Padding entries in the per-head cache have exactly zero effect on the model output. Sinks are never merged.
  Compression events happen at the right steps with correct bookkeeping. ε=0 reproduces the baseline token for token.
* The CLI runs end to end offline (a tiny saved model and tokenizer), including resume and plotting.

## Project layout

```
graphkv/
  clustering.py   ε-graph, label propagation, centroid merging (pure tensor ops)
  cache.py        GraphKVConfig, per-layer/per-head KV cache, compress_layer / compress_cache
  models.py       GPT-2 and Llama-family forward passes over the variable-length cache
  engine.py       generation / teacher-forced scoring with compression every M steps, timing
  data.py         WikiText-2 prompt selection (paper protocol)
  run.py          experiment sweeps        (python -m graphkv.run)
  plot.py         Fig. 2 / Table I         (python -m graphkv.plot)
  finetune.py     WikiText-2 fine-tuning   (python -m graphkv.finetune)
  demo.py         side-by-side generation  (python -m graphkv.demo)
  envcheck.py     installation self-test   (python -m graphkv.envcheck)
tests/            pytest suite (offline)
```
